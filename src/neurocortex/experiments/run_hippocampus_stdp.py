"""ステップ1の追試（12.6.6節）: 分離層を学習させると12.6.5節の敗因は消えるか。

12.6.5節の結論は「固定ランダム射影 + k-WTA は純粋に損」であり、敗因は k-WTA の
**勝者反転**（手がかりがわずかに動くと上位k個の顔ぶれが変わる）と特定された。
学習は第k位と第k+1位のマージンを広げうるので、反転しにくくなる余地がある。
これが本実験の仮説であり、マージンと勝者保存率を直接測って機構ごと検証する。

条件:
    cortex      … 皮質表現をそのままキーに（12.6.5節の勝者。基準線）
    dict        … 同上の厳密最近傍
    sdr         … 固定ランダム射影 + k-WTA（12.6.5節の敗者）
    sdr-hebb    … 競合ヘブ学習で射影を学習（STDPが縮退する先。対照群）
    spike       … バックボーンのスパイクを入力にした固定ランダム射影
    spike-stdp  … 同じ入力をSTDPで学習（8.1節の指数窓、時間軸＝トークン位置）
    dense-hebb  … k-WTAなしの連続値射影を同じ予算で学習（疎化の寄与を分離）
    none        … 海馬なし（チャンスレベルの確認）

`spike` と `spike-stdp` が対になっているのが要点である。STDPは二値スパイク列を
入力に取るため入力表現が連続値から変わる。未学習の `spike` を置かないと、
STDPの効果と入力表現の変化が混ざって区別できない。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_hippocampus_stdp
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..hippocampus import (
    HippocampalMemory,
    fit_hebbian,
    fit_stdp,
    separation_diagnostics,
)
from ..tasks import FactSpec, make_fact_batch
from .run_hippocampus import build_backbone, encode_keys

# 条件名 → (分離層モード, キー入力, 学習則, 厳密最近傍か)
CONDITIONS: dict[str, tuple[str, str, str, bool]] = {
    "cortex": ("identity", "cortex", "none", False),
    "dict": ("identity", "cortex", "none", True),
    "sdr": ("sdr", "cortex", "none", False),
    "sdr-hebb": ("sdr", "cortex", "hebb", False),
    "spike": ("sdr", "spike", "none", False),
    "spike-stdp": ("sdr", "spike", "stdp", False),
    "dense-hebb": ("dense", "cortex", "hebb", False),
    "none": ("identity", "cortex", "none", False),
}


@torch.no_grad()
def encode_spikes(model, prompts: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """最終ブロックのスパイク列 [N, T, d_model] を取り出す（STDPの前シナプス入力）。

    `lm.py` には手を入れず、前方フックで取得する。
    """
    captured: list[torch.Tensor] = []
    handle = model.blocks[-1].act_attn.register_forward_hook(
        lambda _m, _i, out: captured.append(out.detach())
    )
    try:
        for i in range(0, prompts.shape[0], chunk):
            model.encode(prompts[i : i + chunk])
    finally:
        handle.remove()
    return torch.cat(captured)


def make_separator(cond: str, d_model: int, args, seed: int, train_x: torch.Tensor,
                   train_s: torch.Tensor, k: int) -> HippocampalMemory:
    """条件に応じた海馬モジュールを作り、必要なら分離層を学習させる。"""
    mode, source, rule, exact = CONDITIONS[cond]
    mem = HippocampalMemory(
        d_model, value_dim=d_model, n_units=args.n_units, k=k,
        mode=mode, beta=args.betas[0], exact=exact, gain=args.gain, seed=seed,
    )
    g = torch.Generator().manual_seed(seed + 7_000)
    if rule == "hebb":
        fit_hebbian(mem.separator, train_x, epochs=args.epochs, eta=args.eta, generator=g)
    elif rule == "stdp":
        fit_stdp(mem.separator, train_s, epochs=args.epochs, a_plus=args.a_plus,
                 a_minus=args.a_minus, tau=args.tau, generator=g)
    return mem


@torch.no_grad()
def evaluate(model, spec: FactSpec, cond: str, mem: HippocampalMemory,
             hw: torch.Tensor, hq: torch.Tensor, sw: torch.Tensor, sq: torch.Tensor,
             objects: torch.Tensor) -> dict:
    """1条件ぶんの想起精度と、敗因に対応する診断値を返す。"""
    _mode, source, _rule, _exact = CONDITIONS[cond]
    kw, kq = (sw, sq) if source == "spike" else (hw, hq)

    if cond == "none":
        inject = torch.zeros_like(hq)
        diag: dict = {}
    else:
        mem.clear()
        mem.write(kw, model.head.weight[objects])
        inject, _gate = mem.read(kq)
        diag = separation_diagnostics(mem.separator, kw, kq)

    logits = model.head(hq + inject)
    obj = logits[:, spec.object_offset : spec.object_offset + spec.n_object]
    pred = obj.argmax(dim=-1) + spec.object_offset
    return {"n_correct": int((pred == objects).sum()),
            "n_queries": int(objects.shape[0]), **diag}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[100, 1000, 10000])
    ap.add_argument("--cue-noises", type=float, nargs="+", default=[0.02, 0.1])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--ks", type=int, nargs="+", default=[40, 160, 640])
    # βは条件ごとに掃引して最良値で比較する。キー表現が変わるとコサイン類似度の
    # 分布も変わるため、固定すると「βがどちらの分布に合っていたか」を測ってしまう
    # （12.6.5節で一度この誤りを犯している）。上限は十分に高く取る。
    ap.add_argument("--betas", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6])
    # ゲインは全条件に同一に効くのでグリッドを共有すれば対称性は保たれる。
    # 12.6.5節の掃引ではほぼ全セルで16が選ばれたため固定する。
    ap.add_argument("--gain", type=float, default=16.0)
    ap.add_argument("--n-units", type=int, default=2048)
    ap.add_argument("--n-train", type=int, default=20000, help="分離層の学習サンプル数")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--a-plus", type=float, default=0.01)
    ap.add_argument("--a-minus", type=float, default=0.008)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--n-prefix", type=int, default=8)
    ap.add_argument("--n-suffix", type=int, default=4096)
    ap.add_argument("--n-object", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("results/hippocampus/stdp_curve.json"))
    args = ap.parse_args()

    spec = FactSpec(n_prefix=args.n_prefix, n_suffix=args.n_suffix, n_object=args.n_object)
    t0 = time.time()
    rows: list[dict] = []

    for seed in range(args.seeds):
        model = build_backbone(spec, args.d_model, 3, 4, seed, args.train_steps, 3e-3, 64)
        g = torch.Generator().manual_seed(seed + 1_000)
        # 分離層の学習データ。評価に使う事実とは別に引く（漏れの防止）。
        tr_prompts, _ = make_fact_batch(spec, args.n_train, g)
        train_x = encode_keys(model, tr_prompts)
        train_s = encode_spikes(model, tr_prompts)

        # 学習は件数・雑音水準に依存しないので、条件×k ごとに1回だけ行う。
        mems = {(c, k): make_separator(c, args.d_model, args, seed, train_x, train_s, k)
                for c in args.conditions
                for k in (args.ks if CONDITIONS[c][0] == "sdr" else [args.ks[0]])}

        for noise in args.cue_noises:
            for n in args.counts:
                prompts, objects = make_fact_batch(spec, n, g)
                hw = encode_keys(model, prompts)
                sw = encode_spikes(model, prompts)[:, -1]
                scale = noise / args.d_model**0.5
                hq = hw + scale * hw.norm(dim=-1, keepdim=True) * torch.randn(
                    hw.shape, generator=g)
                sq = sw + scale * sw.norm(dim=-1, keepdim=True).clamp_min(1e-6) * torch.randn(
                    sw.shape, generator=g)
                for cond in args.conditions:
                    ks = args.ks if CONDITIONS[cond][0] == "sdr" else [args.ks[0]]
                    # exact（dict）と none は β に依存しないので掃引しない。
                    betas = [0.0] if cond in ("dict", "none") else args.betas
                    best = None
                    for k in ks:
                        mem = mems[(cond, k)]
                        for b in betas:
                            mem.store.beta = b
                            r = evaluate(model, spec, cond, mem,
                                         hw, hq, sw, sq, objects)
                            r = dict(r, accuracy=r["n_correct"] / r["n_queries"],
                                     k=k, beta=b)
                            if best is None or r["accuracy"] > best["accuracy"]:
                                best = r
                    best |= {"condition": cond, "n_facts": n,
                             "cue_noise": noise, "seed": seed,
                             "beta_grid": betas, "k_grid": ks}
                    rows.append(best)
                    print(f"seed={seed} noise={noise} N={n:>6} {cond:>10}: "
                          f"{best['accuracy']:.4f} (β={best['beta']:g}, k={best['k']}"
                          + (f", margin={best['margin']:.5f}, 保存率={best['winner_retention']:.3f}"
                             if "margin" in best else "") + ")", flush=True)

    summary: dict = {}
    for noise in args.cue_noises:
        for cond in args.conditions:
            for n in args.counts:
                sel = [r["accuracy"] for r in rows if r["condition"] == cond
                       and r["n_facts"] == n and r["cue_noise"] == noise]
                t = torch.tensor(sel)
                summary[f"{noise}|{cond}|{n}"] = {
                    "mean": float(t.mean()), "std": float(t.std(unbiased=False))}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "分離層をSTDP／競合ヘブ学習で学習させた場合の想起精度（12.6.6節）",
        "条件": vars(args) | {"out": str(args.out)},
        "chance": spec.chance,
        "summary": summary, "rows": rows,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nチャンスレベル {spec.chance:.4f} / 書き出し: {args.out}")
    for noise in args.cue_noises:
        print(f"--- 手がかり劣化 {noise}")
        for cond in args.conditions:
            line = "  ".join(f"N={n}:{summary[f'{noise}|{cond}|{n}']['mean']:.3f}"
                             for n in args.counts)
            print(f"{cond:>10}  {line}")


if __name__ == "__main__":
    main()
