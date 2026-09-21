"""ステップ1（海馬モジュール, 12.6.4節）の主実験。

主指標: 蓄積件数 N を増やしたとき、パターン分離層（k-WTA疎SDR）を通したキーが、
皮質表現をそのままキーにした場合より想起精度の劣化が緩やかであること。

条件（12.6.4節のアブレーション3水準 + 統制3）:
    sdr      … 固定ランダム射影 + k-WTA（本設計）
    dense    … 同次元の密なランダム射影（疎化だけを取り除く）
    cortex   … 皮質表現をそのままキーにする（分離層なし）
    dict     … 皮質表現の厳密最近傍（β→∞。ハッシュ表と等価な対照群。統制3）
    sdr-dict … 疎SDRキーの厳密最近傍（分離の効果が読み出しの滑らかさ由来か否かの分離）
    none     … 海馬なし（統制2の下限。チャンスレベルのはず）

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_hippocampus \
        --counts 1 10 100 1000 10000 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..hippocampus import HippocampalMemory
from ..lm import SpikingLM
from ..neurons import NeuronConfig
from ..tasks import FactSpec, make_fact_batch

# 条件名 → (分離層モード, 厳密最近傍か)
CONDITIONS: dict[str, tuple[str, bool]] = {
    "sdr": ("sdr", False),
    "dense": ("dense", False),
    "cortex": ("identity", False),
    "dict": ("identity", True),
    "sdr-dict": ("sdr", True),
}


def build_backbone(spec: FactSpec, d_model: int, n_layers: int, n_heads: int,
                   seed: int, train_steps: int, lr: float, batch_size: int) -> SpikingLM:
    """皮質バックボーンを用意して凍結する。

    三つ組の「書式」だけを学習させる。目的語は主語と独立にサンプリングされるので、
    バックボーンが主語→目的語の対応を憶えることは原理的にありえない（統制2）。
    """
    torch.manual_seed(seed)
    model = SpikingLM(
        spec.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        max_len=spec.seq_len, cfg=NeuronConfig(),
    )
    if train_steps > 0:
        g = torch.Generator().manual_seed(seed + 90_000)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(train_steps):
            prompts, objects = make_fact_batch(spec, batch_size, g)
            seq = torch.cat([prompts, objects[:, None]], dim=1)
            logits = model(seq[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, spec.vocab_size), seq[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    # 統制5: 以降、皮質側に勾配は一切流れない。
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_keys(model: SpikingLM, prompts: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """`[接頭][接尾][関係]` の最終位置の表現 ln_f(x) を取り出す。"""
    out = []
    for i in range(0, prompts.shape[0], chunk):
        out.append(model.encode(prompts[i : i + chunk])[:, -1])
    return torch.cat(out)


@torch.no_grad()
def run_condition(model: SpikingLM, spec: FactSpec, prompts: torch.Tensor,
                  objects: torch.Tensor, condition: str, args, beta: float,
                  gain: float, k: int, sep_seed: int, h: torch.Tensor | None = None,
                  h_query: torch.Tensor | None = None) -> dict:
    """1条件・1件数ぶんの想起精度を測る。

    `h` は書き込み時の皮質表現、`h_query` は想起時の（劣化した）手がかり。
    両者を分けているのは、手がかりが無傷ならパターン補完が不要になるためである
    （6.1節のCA3は「断片的な手がかり」からの復元を役割としている）。
    """
    if h is None:
        h = encode_keys(model, prompts)  # [N, d_model] 皮質表現
    if h_query is None:
        h_query = h
    d_model = h.shape[-1]

    if condition == "none":
        inject = torch.zeros_like(h)
        gate = torch.zeros(h.shape[0])
        n_stored = 0
    else:
        mode, exact = CONDITIONS[condition]
        mem = HippocampalMemory(
            d_model, value_dim=d_model, n_units=args.n_units,
            mode=mode, beta=beta, exact=exact, gain=gain, seed=sep_seed, k=k,
        )
        # 値は目的語トークンの出力埋め込み方向。これを ln_f(x) に足すと、
        # head が線形なのでその目的語のロジットが持ち上がる。学習は不要。
        values = model.head.weight[objects]
        mem.write(h, values)
        inject, gate = mem.read(h_query)
        n_stored = len(mem.store)

    logits = model.head(h_query + inject)
    # 目的語の語彙に絞って判定する（書式の予測誤りと想起の誤りを混ぜないため）。
    obj_logits = logits[:, spec.object_offset : spec.object_offset + spec.n_object]
    pred = obj_logits.argmax(dim=-1) + spec.object_offset
    return {
        "condition": condition,
        "cue_noise": args.cue_noise,
        "n_facts": int(prompts.shape[0]),
        "accuracy": float((pred == objects).float().mean()),
        "n_stored": n_stored,
        "mean_gate": float(gate.mean()),
        "beta": beta,
        "gain": gain,
        "k": k,
        "n_correct": int((pred == objects).sum()),
        "n_queries": int(objects.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[1, 10, 100, 1000, 10000])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", nargs="+",
                    default=["sdr", "dense", "cortex", "dict", "sdr-dict", "none"])
    ap.add_argument("--n-prefix", type=int, default=8, help="小さいほど主語どうしが似る")
    ap.add_argument("--n-suffix", type=int, default=4096)
    ap.add_argument("--n-object", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-units", type=int, default=2048)
    # 疎度kも条件ごとに掃引する。固定すると疎SDR条件だけを過小調整することになり、
    # 統制3（学習予算の対称性）が崩れる。
    ap.add_argument("--ks", type=int, nargs="+", default=[40, 160, 640])
    # βは条件ごとに掃引して最良値で比較する。キー表現が変わるとスコア分布も変わる
    # ため、単一のβで比べると「βがどちらに有利だったか」を測ってしまう
    # （12.6.1節 統制3「学習予算の対称性」と同じ趣旨）。
    # 【重要】上限を十分に高く取ること。皮質表現どうしのコサイン類似度は互いに
    # 接近しているため、βが小さいと softmax が候補を分離できず、キー表現の差ではなく
    # 「βがどちらのスコア分布に合っていたか」を測ってしまう。上限300で測ったときは
    # 皮質キーが 0.021、同じキーの厳密最近傍が 1.000 という矛盾が生じていた。
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[1e2, 1e3, 1e4, 1e5, 1e6])
    # ゲインもβと同じくハイパーパラメータなので、条件ごとに掃引して最良値で比較する。
    ap.add_argument("--gains", type=float, nargs="+", default=[4.0, 16.0, 64.0])
    # 手がかりの劣化。0.0 では書き込み時と想起時の手がかりが完全に一致するため、
    # 厳密最近傍（dict）が自明に完璧になりパターン補完の出番がない。
    ap.add_argument("--cue-noise", type=float, default=0.0,
                    help="想起時の手がかりに加えるガウス雑音の相対強度")
    ap.add_argument("--min-queries", type=int, default=500,
                    help="1点あたりの最低クエリ数。Nが小さいとき試行を繰り返す")
    ap.add_argument("--sep-seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/hippocampus/recall_curve.json"))
    args = ap.parse_args()

    spec = FactSpec(n_prefix=args.n_prefix, n_suffix=args.n_suffix, n_object=args.n_object)
    t0 = time.time()
    rows: list[dict] = []
    for seed in range(args.seeds):
        model = build_backbone(spec, args.d_model, args.n_layers, args.n_heads,
                               seed, args.train_steps, args.lr, args.batch_size)
        g = torch.Generator().manual_seed(seed + 1_000)
        for n in args.counts:
            # Nが小さいと1試行あたりのクエリ数がNしかなく、精度の推定が粗くなる
            # （N=1 では1問正解/不正解で0か1になる）。最低 min_queries 問になるまで
            # 別の事実集合で試行を繰り返す。
            n_rep = max(1, -(-args.min_queries // n))
            batches = []
            for _ in range(n_rep):
                prompts, objects = make_fact_batch(spec, n, g)
                hw = encode_keys(model, prompts)
                if args.cue_noise > 0.0:
                    # 成分あたりの典型的な大きさに比例させた雑音（表現のスケールに依存しない）
                    scale = args.cue_noise * hw.norm(dim=-1, keepdim=True) / hw.shape[-1] ** 0.5
                    hq = hw + scale * torch.randn(hw.shape, generator=g)
                else:
                    hq = hw
                batches.append((prompts, objects, hw, hq))
            for cond in args.conditions:
                # exact（dict系）とnoneはβに依存しないので掃引しない。
                betas = [0.0] if cond in ("none", "dict", "sdr-dict") else args.betas
                gains = [0.0] if cond == "none" else args.gains
                # 疎度kは疎SDRを使う条件にしか効かない。
                ks = args.ks if cond in ("sdr", "sdr-dict") else [args.ks[0]]
                best = None
                for b in betas:
                  for gain in gains:
                    for k in ks:
                        agg = [run_condition(model, spec, p, o, cond, args, b, gain, k,
                                             args.sep_seed + seed, hw, hq)
                               for p, o, hw, hq in batches]
                        correct = sum(r["n_correct"] for r in agg)
                        total = sum(r["n_queries"] for r in agg)
                        row = dict(agg[0], accuracy=correct / total,
                                   n_correct=correct, n_queries=total, n_reps=n_rep,
                                   mean_gate=sum(r["mean_gate"] for r in agg) / len(agg))
                        if best is None or row["accuracy"] > best["accuracy"]:
                            best = row
                best["seed"] = seed
                best["beta_grid"] = betas
                best["gain_grid"] = gains
                best["k_grid"] = ks
                rows.append(best)
                print(f"seed={seed} N={n:>6} {cond:>8}: {best['accuracy']:.4f}"
                      f" (β={best['beta']:g}, gain={best['gain']:g}, k={best['k']},"
                      f" {best['n_queries']}問)", flush=True)

    summary: dict[str, dict[str, dict]] = {}
    for cond in args.conditions:
        summary[cond] = {}
        for n in args.counts:
            accs = torch.tensor([r["accuracy"] for r in rows
                                 if r["condition"] == cond and r["n_facts"] == n])
            summary[cond][str(n)] = {
                "mean": float(accs.mean()),
                "std": float(accs.std(unbiased=False)),
                "min": float(accs.min()),
                "max": float(accs.max()),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ1 海馬モジュールの想起精度曲線（12.6.4節）",
        "条件": vars(args) | {"out": str(args.out)},
        "chance": spec.chance,
        "summary": summary,
        "rows": rows,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nチャンスレベル {spec.chance:.4f} / 書き出し: {args.out}")
    for cond in args.conditions:
        line = "  ".join(f"N={n}:{summary[cond][str(n)]['mean']:.3f}" for n in args.counts)
        print(f"{cond:>8}  {line}")


if __name__ == "__main__":
    main()
