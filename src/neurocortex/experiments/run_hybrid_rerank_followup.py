"""ハイブリッド再ランクの残課題（12.6.91節）の検証（12.6.116〜117節）。

1. 内積・密度・テンプレートの各スコア単体での Top-1 精度を正解ラベルで測る
   （λ=(1,0,0)/(0,1,0)/(0,0,1) で `read_with_hybrid_rerank` を呼び、`top1_index` を比べる）。
2. 線形和に代わるゲート式合成（`combine="gate"`）を線形和と比べる。

過剰適合を防ぐため、λ・ゲート幅は開発用シード（DEV_SEEDS）だけで選び、
評価用シード（EVAL_SEEDS）は選んだ設定で最後に一度だけ測る。評価結果のファイルが
既にあるときは `--force` なしでは評価を走らせない。

データ生成は12.6.90〜91節と同じ（`torch.randn` の768次元キー、書き込み済みキーに
`0.125·‖key‖` のガウスノイズを加えたクエリ、元の番号が正解）。大容量のストアは
`STEP41_STORE_ROOT`（既定はシステムの一時ディレクトリ）に置き、測定後に削除する。

    PYTHONPATH=src python -m neurocortex.experiments.run_hybrid_rerank_followup --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_hybrid_rerank_followup --phase eval
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..hippocampus import DiskBackedAssociativeStore
from ..run_env import env_info

DEV_SEEDS = [0, 1, 2]
EVAL_SEEDS = [100, 101, 102]
KEY_DIM, VALUE_DIM = 768, 512
NOISE_STD_RATIO = 0.125
K_REFINE = 150

SINGLE = {"内積単体": (1.0, 0.0, 0.0), "密度単体": (0.0, 1.0, 0.0), "テンプレート単体": (0.0, 0.0, 1.0)}
LINEAR_GRID = {f"線形和 λ=({a},{b},{c})": (a, b, c) for a, b, c in
               [(0.6, 0.3, 0.1), (0.4, 0.45, 0.15), (0.2, 0.6, 0.2)]}
GATE_MARGINS = [0.1, 0.25, 0.5, 1.0, 2.0]
GATE_SECONDARY = {"密度": (0.0, 1.0, 0.0), "テンプレート": (0.0, 0.0, 1.0), "密度+テンプレート": (0.0, 0.3, 0.1)}


def build_store(root: Path, n: int, seed: int, batch: int = 10000) -> DiskBackedAssociativeStore:
    store = DiskBackedAssociativeStore(KEY_DIM, VALUE_DIM, persist_dir=root, exact=True)
    g = torch.Generator().manual_seed(seed)
    for start in range(0, n, batch):
        cur = min(batch, n - start)
        store.write(torch.randn(cur, KEY_DIM, generator=g), torch.randn(cur, VALUE_DIM, generator=g))
    return store


def labeled_queries(store: DiskBackedAssociativeStore, n_queries: int, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed + 10_000)
    true_idx = np.sort(rng.choice(store.write_count, size=n_queries, replace=False))
    keys = torch.from_numpy(np.array(store._keys_memmap()[true_idx])).float()
    g = torch.Generator().manual_seed(seed + 20_000)
    noise = torch.randn(keys.shape, generator=g) * keys.norm(dim=-1, keepdim=True) * NOISE_STD_RATIO
    return keys + noise, true_idx


def candidate_recall(store: DiskBackedAssociativeStore, q: torch.Tensor, true_idx: np.ndarray,
                     k_cand: int) -> dict:
    """正解が段階1の候補集合・内積上位 K_REFINE 件に入っている割合（再ランクで届く上限）。"""
    _, idx = store._pq_index.search(q.numpy(), k_cand)
    keys = torch.from_numpy(np.array(store._keys_memmap()[np.where(idx >= 0, idx, 0).reshape(-1)])).float()
    inner = torch.einsum("bd,bkd->bk", q, keys.view(q.shape[0], k_cand, -1))
    top = idx[np.arange(q.shape[0])[:, None], inner.topk(min(K_REFINE, k_cand), dim=-1).indices.numpy()]
    return {"候補集合に正解": float((idx == true_idx[:, None]).any(1).mean()),
            "内積上位k_refineに正解": float((top == true_idx[:, None]).any(1).mean())}


def run_seed(n: int, seed: int, n_queries: int, store_root: Path, conditions: dict) -> dict:
    root = Path(tempfile.mkdtemp(prefix=f"rerank_n{n}_s{seed}_", dir=store_root))
    try:
        store = build_store(root, n, seed)
        q, true_idx = labeled_queries(store, n_queries, seed)
        k_cand = int(np.sqrt(n))
        correct, gated = {}, {}
        t0 = time.time()
        for name, cfg in conditions.items():
            lam, combine, margin = cfg
            _, stats, bd = store.read_with_hybrid_rerank(
                q, k_candidates=k_cand, k_refine=K_REFINE, lambda_inner=lam[0], lambda_density=lam[1],
                lambda_template=lam[2], use_gpu=False, return_score_breakdown=True, combine=combine,
                gate_margin=margin)
            correct[name] = stats.top1_index.numpy() == true_idx
            if combine == "gate":
                gated[name] = bd["gated_fraction"]
        out = {"N": n, "seed": seed, "n_queries": n_queries, "k_candidates": k_cand, "k_refine": K_REFINE,
               **candidate_recall(store, q, true_idx, k_cand),
               "top1": {k: float(v.mean()) for k, v in correct.items()},
               "ゲートが開いた割合": gated, "計測秒": round(time.time() - t0, 1)}
        inner_ok = correct["内積単体"]
        out["内積と比べた差分"] = {k: {"内積誤り→正解": int((~inner_ok & v).sum()), "内積正解→誤り": int((inner_ok & ~v).sum())}
                            for k, v in correct.items() if k != "内積単体"}
        return out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def dev_conditions() -> dict:
    conds = {k: (v, "linear", 0.0) for k, v in {**SINGLE, **LINEAR_GRID}.items()}
    for sname, lam in GATE_SECONDARY.items():
        for m in GATE_MARGINS:
            conds[f"ゲート({sname}, 幅{m})"] = (lam, "gate", m)
    return conds


def mean_top1(runs: list[dict]) -> dict:
    names = runs[0]["top1"].keys()
    return {k: float(np.mean([r["top1"][k] for r in runs])) for k in names}


def select(dev_runs: list[dict], conds: dict) -> dict:
    """開発用シードの平均 Top-1 で線形和・ゲートの最良設定を1つずつ選ぶ（同点は先に並ぶ方）。"""
    m = mean_top1(dev_runs)
    best = {}
    for kind, prefix in [("線形和", "線形和"), ("ゲート", "ゲート")]:
        cands = [k for k in conds if k.startswith(prefix)]
        best[kind] = max(cands, key=lambda k: m[k])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "eval"], required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[20000, 100000])
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("results/hybrid_rerank_followup"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    store_root = Path(os.environ.get("STEP41_STORE_ROOT", tempfile.gettempdir()))
    store_root.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    dev_path, eval_path = args.out / "dev.json", args.out / "eval.json"

    conds = dev_conditions()
    if args.phase == "dev":
        seeds = DEV_SEEDS
    else:
        if eval_path.exists() and not args.force:
            raise SystemExit(f"{eval_path} が既にある。評価用シードは一度だけ測る（再測定は --force）")
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        chosen = dev["選んだ設定"]
        keep = list(SINGLE) + [chosen["線形和"], chosen["ゲート"]]
        conds = {k: conds[k] for k in keep}
        seeds = EVAL_SEEDS

    result: dict = {"環境": env_info(), "phase": args.phase, "seeds": seeds, "noise_std_ratio": NOISE_STD_RATIO,
                    "説明": "ハイブリッド再ランクの単体スコア精度とゲート式合成（12.6.116〜117節）", "N別": {}}
    for n in args.n:
        runs = []
        for s in seeds:
            r = run_seed(n, s, args.n_queries, store_root, conds)
            print(json.dumps({k: r[k] for k in ("N", "seed", "候補集合に正解", "内積上位k_refineに正解", "top1")},
                             ensure_ascii=False), flush=True)
            runs.append(r)
        result["N別"][str(n)] = {"平均top1": mean_top1(runs), "シード別": runs}
    if args.phase == "dev":
        # 選択は全 N の平均で行う（N ごとに選ぶと評価対象の自由度が増える）
        all_runs = [r for v in result["N別"].values() for r in v["シード別"]]
        result["選んだ設定"] = select(all_runs, conds)
        result["条件の定義"] = {k: {"lambda": v[0], "combine": v[1], "gate_margin": v[2]} for k, v in conds.items()}
    else:
        result["選んだ設定"] = chosen
    path = dev_path if args.phase == "dev" else eval_path
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {path}")


if __name__ == "__main__":
    main()
