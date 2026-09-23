"""ステップ34（小脳モジュールとの予測的プリフェッチ結合, 12.6.76〜77節）の主実験。

`DiskBackedAssociativeStore.prefetch`（ステップ33の`_get_chunk`・`cache_size`・
`_chunk_cache`を経路として使う、読み出し前の先読み）を、3条件で比較する。

  (i)   プリフェッチなし: ステップ33のcache_size>0構成そのまま（比較対象）
  (ii)  ランダムプリフェッチ: 統制条件。直近アクセスチャンク集合からランダムに1件先読み
  (iii) 予測的プリフェッチ: 頻度＋1次マルコフ連鎖ベースの「持続する予測器」
        （`ForwardModel`の「エピソードをまたいだ持続」という設計思想のみ参考にする。
        cerebellum.py本体・数式は一切流用しない）

クエリ列は2種類:
  (A) ステップ33のZipf偏り分布クエリ（系列的局所性なし、比較対照）
  (B) 系列的局所性を持つ合成クエリ列（全体をK区間に分け、各区間で参照チャンク範囲が
      スライドしていく）

N=100〜30万でスイープ、キャッシュ容量64チャンク固定。read壁時計時間（中央値）、
無駄プリフェッチ率、予測器の的中率を測定し、正確性（read出力の完全一致）を検証する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_cerebellum_prefetch
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..hippocampus import DiskBackedAssociativeStore

DEFAULT_N_LIST = (100, 1_000, 10_000, 100_000, 300_000)
CACHE_SIZE = 64
QUERY_BATCH = 32  # 1回のread呼び出しで渡すクエリ件数（ステップ33と同一）
N_QUERY_CALLS_DEFAULT = None  # main()でNごとに決める（Nそのものがクエリ「件数」の意味）
N_WARMUP = 1
N_MEASURE = 5
DISK_KEY_CHUNK = 4096
MARKOV_WINDOW = 50  # 頻度・マルコフ推定に使う直近履歴長


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set_bytes() -> int:
    """参考値（合格条件には含めない）。ステップ32・33と同じ実装。"""
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters),
                                            wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError(f"GetProcessMemoryInfoに失敗した (GetLastError={ctypes.get_last_error()})")
    return int(counters.PeakWorkingSetSize)


# ---------------------------------------------------------------------------
# 予測器: 頻度＋1次マルコフ連鎖ベース、「持続する予測器」（エピソードをまたいで
# 状態を保持し逐次更新される）という設計思想のみ`ForwardModel`から踏襲する。
# 数式（デルタ則によるチャンクID回帰）は一切流用しない。離散的なチャンクIDに
# 対しては頻度表・遷移頻度表という素朴な統計モデルを用いる。
# ---------------------------------------------------------------------------

class MarkovChunkPredictor:
    """直近アクセスされたchunk_idの列から次chunk_idを予測する、持続する予測器。

    - 頻度表: 単純な出現頻度（遷移データが薄い初期段階のフォールバック）。
    - 1次マルコフ遷移表: `transition[prev][next] += 1`。直近`prev`からの遷移で
      最頻出の`next`を予測する。
    どちらも逐次更新（`update`）され、エピソード（クエリ列）をまたいで状態を
    リセットしない「持続する予測器」である。
    """

    def __init__(self) -> None:
        self.freq: dict[int, int] = defaultdict(int)
        self.transition: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._prev: int | None = None

    def predict(self) -> int | None:
        """現在の状態から次にアクセスされそうなchunk_idを1件予測する。"""
        if self._prev is not None and self._prev in self.transition:
            trans_row = self.transition[self._prev]
            if trans_row:
                return max(trans_row.items(), key=lambda kv: kv[1])[0]
        if self.freq:
            return max(self.freq.items(), key=lambda kv: kv[1])[0]
        return None

    def update(self, chunk_id: int) -> None:
        """実際にアクセスされたchunk_idを観測し、頻度表・遷移表を更新する。"""
        self.freq[chunk_id] += 1
        if self._prev is not None:
            self.transition[self._prev][chunk_id] += 1
        self._prev = chunk_id


class RandomChunkPredictor:
    """統制条件（ii）: 直近アクセスチャンク集合からランダムに1件選ぶ。"""

    def __init__(self, n_chunks: int, seed: int) -> None:
        self.n_chunks = n_chunks
        self.rng = np.random.default_rng(seed)
        self._seen: set[int] = set()

    def predict(self) -> int | None:
        if not self._seen:
            return None
        return int(self.rng.choice(list(self._seen)))

    def update(self, chunk_id: int) -> None:
        self._seen.add(chunk_id)


# ---------------------------------------------------------------------------
# クエリ列生成
# ---------------------------------------------------------------------------

def make_skewed_queries(n_queries: int, key_dim: int, n_chunks: int, key_chunk: int,
                         keys: torch.Tensor, seed: int, zipf_s: float = 1.5) -> torch.Tensor:
    """(A) ステップ33と同じZipf偏り分布クエリ（系列的局所性なし、比較対照）。"""
    ranks = np.arange(1, n_chunks + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** zipf_s)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    chosen_chunks = rng.choice(n_chunks, size=n_queries, p=weights)
    queries = torch.empty(n_queries, key_dim)
    for i, c in enumerate(chosen_chunks):
        start = c * key_chunk
        end = min(start + key_chunk, keys.shape[0])
        row = start + int(rng.integers(0, max(end - start, 1)))
        row = min(row, keys.shape[0] - 1)
        noise = torch.from_numpy(rng.normal(scale=0.01, size=key_dim)).to(torch.float32)
        queries[i] = keys[row] + noise
    return torch.nn.functional.normalize(queries, dim=-1)


def make_sequential_locality_queries(n_queries: int, key_dim: int, n_chunks: int, key_chunk: int,
                                      keys: torch.Tensor, seed: int, n_intervals: int = 20,
                                      window_chunks: int = 3) -> torch.Tensor:
    """(B) 系列的局所性を持つ合成クエリ列（新設）。

    クエリ列全体をK=`n_intervals`区間に分け、各区間では参照chunk_idの範囲
    （幅`window_chunks`チャンク）に限定してサンプリングする。区間ごとに
    参照範囲の中心をチャンク空間内で緩やかにスライドさせることで、
    系列的な局所性（同一・近傍チャンクへの参照が連続する）を作る。
    """
    rng = np.random.default_rng(seed)
    queries = torch.empty(n_queries, key_dim)
    per_interval = max(1, n_queries // n_intervals)
    idx = 0
    for interval in range(n_intervals):
        n_this = per_interval if interval < n_intervals - 1 else n_queries - idx
        if n_this <= 0:
            break
        # 中心をチャンク空間全体にわたって緩やかにスライド
        frac = interval / max(1, n_intervals - 1)
        center = frac * (n_chunks - 1)
        lo = max(0, int(center - window_chunks / 2))
        hi = min(n_chunks - 1, lo + window_chunks - 1)
        lo = max(0, hi - window_chunks + 1)
        for _ in range(n_this):
            c = int(rng.integers(lo, hi + 1))
            start = c * key_chunk
            end = min(start + key_chunk, keys.shape[0])
            row = start + int(rng.integers(0, max(end - start, 1)))
            row = min(row, keys.shape[0] - 1)
            noise = torch.from_numpy(rng.normal(scale=0.01, size=key_dim)).to(torch.float32)
            queries[idx] = keys[row] + noise
            idx += 1
    return torch.nn.functional.normalize(queries[:idx], dim=-1)


# ---------------------------------------------------------------------------
# 1回の測定: 与えられたクエリ列を1件ずつ`read`し、条件に応じてプリフェッチを行う。
# ---------------------------------------------------------------------------

def _chunk_of_arg(arg: int, key_chunk: int) -> int:
    return arg // key_chunk if arg >= 0 else -1


def run_condition(store: DiskBackedAssociativeStore, queries: torch.Tensor,
                   condition: str, n_chunks: int, seed: int) -> dict:
    """クエリ列を1件ずつread。条件(i)(ii)(iii)に応じてread前にprefetchする。

    「無駄プリフェッチ」: プリフェッチしたチャンクが、直後のreadの最良一致
    チャンク（`arg // key_chunk`）と一致しなかった場合。
    「的中率」: 予測器（(iii)のみ意味を持つ）が実際の次アクセスチャンクを当てた割合。
    """
    predictor = None
    if condition == "ii_random":
        predictor = RandomChunkPredictor(n_chunks, seed)
    elif condition == "iii_predictive":
        predictor = MarkovChunkPredictor()

    n_prefetch = 0
    n_wasted = 0
    n_hits = 0
    outs = []
    bests = []
    args = []

    for i in range(queries.shape[0]):
        q = queries[i : i + 1]
        predicted_chunk = None
        if predictor is not None:
            predicted_chunk = predictor.predict()
            if predicted_chunk is not None:
                store.prefetch(predicted_chunk)
                n_prefetch += 1

        out, stats = store.read(q)
        actual_chunk = _chunk_of_arg(int(stats.top1_index[0]), store.key_chunk)

        if predicted_chunk is not None:
            if predicted_chunk == actual_chunk:
                n_hits += 1
            else:
                n_wasted += 1

        if predictor is not None:
            predictor.update(actual_chunk)

        outs.append(out)
        bests.append(stats.max_score)
        args.append(stats.top1_index)

    return {
        "out": torch.cat(outs, dim=0),
        "best": torch.cat(bests, dim=0),
        "arg": torch.cat(args, dim=0),
        "n_prefetch": n_prefetch,
        "n_wasted": n_wasted,
        "n_hits": n_hits,
        "wasted_rate": (n_wasted / n_prefetch) if n_prefetch > 0 else None,
        "hit_rate": (n_hits / n_prefetch) if n_prefetch > 0 else None,
    }


def measure_condition_timing(store: DiskBackedAssociativeStore, queries: torch.Tensor,
                              condition: str, n_chunks: int, seed: int,
                              n_measure: int) -> tuple[list[float], dict]:
    """read壁時計時間を複数回計測し中央値を取る。最後の1回の出力・統計を返す。

    キャッシュ状態は測定間でクリアしない（ステップ33と同じ、ホットキャッシュ想定）。
    ただし予測器の頻度・遷移表は測定ごとにリセットする（各回を独立試行として
    的中率・無駄プリフェッチ率を計測するため）。
    """
    times = []
    result = None
    for m in range(N_WARMUP + n_measure):
        t0 = time.perf_counter()
        result = run_condition(store, queries, condition, n_chunks, seed + m * 7919)
        dt = time.perf_counter() - t0
        if m >= N_WARMUP:
            times.append(dt)
    return times, result


def measure(n: int, d_model: int, value_dim: int, seed: int, condition: str,
            query_kind: str, persist_root: Path, n_measure: int = N_MEASURE) -> dict:
    persist_dir = persist_root / f"n{n}_seed{seed}_{condition}_{query_kind}"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    cache_size = 0 if condition == "i_none" else CACHE_SIZE
    peak_baseline = peak_working_set_bytes()
    g = torch.Generator().manual_seed(seed)
    store = DiskBackedAssociativeStore(d_model, value_dim, persist_dir=persist_dir,
                                        exact=True, key_chunk=DISK_KEY_CHUNK,
                                        cache_size=cache_size)
    keys = torch.nn.functional.normalize(torch.randn(n, d_model, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim, generator=g), dim=-1)
    store.write(keys, values)

    n_chunks = max(1, (n + DISK_KEY_CHUNK - 1) // DISK_KEY_CHUNK)
    # クエリ件数はNに比例させすぎると数十万回のPythonループ×read呼び出しで
    # 極めて遅くなるため、一定件数（min(n, 400)）に固定してread壁時計時間の
    # 中央値を比較する（Nはストア規模、クエリ件数はワークロード長として分離）。
    n_queries = min(max(n // 100, 100), 400)
    query_seed = seed + 90_000
    if query_kind == "zipf":
        queries = make_skewed_queries(n_queries, d_model, n_chunks, DISK_KEY_CHUNK,
                                       keys, query_seed)
    else:
        queries = make_sequential_locality_queries(n_queries, d_model, n_chunks, DISK_KEY_CHUNK,
                                                     keys, query_seed)

    times, result = measure_condition_timing(store, queries, condition, n_chunks, seed, n_measure)
    peak_after = peak_working_set_bytes()

    out = {
        "read_time_median_sec": statistics.median(times),
        "read_time_all_sec": times,
        "peak_working_set_delta_bytes": max(0, peak_after - peak_baseline),
        "n_chunks": n_chunks,
        "n_queries": n_queries,
        **result,
    }
    shutil.rmtree(persist_dir, ignore_errors=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--value-dim", type=int, default=64)
    ap.add_argument("--n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    ap.add_argument("--n-measure", type=int, default=N_MEASURE)
    ap.add_argument("--out", type=Path,
                    default=Path("results/cerebellum_prefetch/prefetch_comparison.json"))
    args = ap.parse_args()

    t0 = time.time()
    persist_root = Path(tempfile.mkdtemp(prefix="cerebellum_prefetch_"))

    conditions = ["i_none", "ii_random", "iii_predictive"]
    query_kinds = ["zipf", "sequential"]

    # per[query_kind][condition][n] = {read_time: [], wasted_rate: [], hit_rate: []}
    per: dict[str, dict[str, dict[int, dict]]] = {
        qk: {c: {n: {"read_time": [], "wasted_rate": [], "hit_rate": [], "peak_ws": []}
                 for n in args.n_list}
             for c in conditions}
        for qk in query_kinds
    }
    accuracy_mismatches: list[str] = []

    try:
        for query_kind in query_kinds:
            for n in args.n_list:
                for seed in range(args.seeds):
                    baseline = measure(n, args.d_model, args.value_dim, seed, "i_none",
                                        query_kind, persist_root, args.n_measure)
                    per[query_kind]["i_none"][n]["read_time"].append(baseline["read_time_median_sec"])
                    per[query_kind]["i_none"][n]["peak_ws"].append(baseline["peak_working_set_delta_bytes"])

                    for condition in ("ii_random", "iii_predictive"):
                        cur = measure(n, args.d_model, args.value_dim, seed, condition,
                                      query_kind, persist_root, args.n_measure)
                        per[query_kind][condition][n]["read_time"].append(cur["read_time_median_sec"])
                        per[query_kind][condition][n]["peak_ws"].append(cur["peak_working_set_delta_bytes"])
                        if cur["wasted_rate"] is not None:
                            per[query_kind][condition][n]["wasted_rate"].append(cur["wasted_rate"])
                        if cur["hit_rate"] is not None:
                            per[query_kind][condition][n]["hit_rate"].append(cur["hit_rate"])

                        # 主張(b): 全条件・全N・全クエリ列・全シードでread出力が完全一致。
                        if not torch.equal(baseline["out"], cur["out"]):
                            accuracy_mismatches.append(
                                f"{query_kind} n={n} seed={seed} {condition}: value(out)不一致")
                        if not torch.equal(baseline["best"], cur["best"]):
                            accuracy_mismatches.append(
                                f"{query_kind} n={n} seed={seed} {condition}: best(max_score)不一致")
                        if not torch.equal(baseline["arg"], cur["arg"]):
                            accuracy_mismatches.append(
                                f"{query_kind} n={n} seed={seed} {condition}: arg(top1_index)不一致")
                    print(f"  完了: {query_kind} n={n} seed={seed}  経過={time.time()-t0:.1f}s",
                          flush=True)
    finally:
        shutil.rmtree(persist_root, ignore_errors=True)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "std": None, "median": None, "per_seed": []}
        t = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)),
                "median": float(statistics.median(values)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for query_kind in query_kinds:
        summary[query_kind] = {}
        for n in args.n_list:
            summary[query_kind][str(n)] = {}
            baseline_stats = _stats(per[query_kind]["i_none"][n]["read_time"])
            summary[query_kind][str(n)]["i_none_read_time_sec"] = baseline_stats
            for condition in ("ii_random", "iii_predictive"):
                rt_stats = _stats(per[query_kind][condition][n]["read_time"])
                wr_stats = _stats(per[query_kind][condition][n]["wasted_rate"])
                hr_stats = _stats(per[query_kind][condition][n]["hit_rate"])
                summary[query_kind][str(n)][f"{condition}_read_time_sec"] = rt_stats
                summary[query_kind][str(n)][f"{condition}_wasted_rate"] = wr_stats
                summary[query_kind][str(n)][f"{condition}_hit_rate"] = hr_stats
                if baseline_stats["median"]:
                    reduction_vs_none = 1.0 - (rt_stats["median"] / baseline_stats["median"])
                else:
                    reduction_vs_none = None
                summary[query_kind][str(n)][f"{condition}_read_time_reduction_vs_i_none"] = reduction_vs_none
            rand_med = summary[query_kind][str(n)]["ii_random_read_time_sec"]["median"]
            pred_med = summary[query_kind][str(n)]["iii_predictive_read_time_sec"]["median"]
            if rand_med:
                summary[query_kind][str(n)]["iii_predictive_read_time_reduction_vs_ii_random"] = (
                    1.0 - (pred_med / rand_med))
            else:
                summary[query_kind][str(n)]["iii_predictive_read_time_reduction_vs_ii_random"] = None

    # 合格条件(a): 系列的局所性ありクエリ・N最大・キャッシュ64で、
    # (iii)のread壁時計時間中央値が(i)比20%以上短縮 かつ (ii)比10%以上短縮。
    max_n = max(args.n_list)
    seq_summary = summary["sequential"].get(str(max_n), {})
    claim_a_reduction_vs_none = seq_summary.get("iii_predictive_read_time_reduction_vs_i_none")
    claim_a_reduction_vs_random = seq_summary.get("iii_predictive_read_time_reduction_vs_ii_random")
    claim_a_ok = bool(
        claim_a_reduction_vs_none is not None and claim_a_reduction_vs_none >= 0.20
        and claim_a_reduction_vs_random is not None and claim_a_reduction_vs_random >= 0.10
    )
    # zipf（系列的局所性なし）側の同じ指標も参考値として記録する（実施前メモが
    # 想定したとおり逆効果になりうる。その場合、主張(a)は限定付き成立とみなす）。
    zipf_summary = summary["zipf"].get(str(max_n), {})
    claim_a_zipf_reduction_vs_none = zipf_summary.get("iii_predictive_read_time_reduction_vs_i_none")
    claim_a_zipf_reduction_vs_random = zipf_summary.get("iii_predictive_read_time_reduction_vs_ii_random")

    # 合格条件(b): 全条件・全N・全クエリ列・全シードでread出力が完全一致。
    claim_b_ok = bool(len(accuracy_mismatches) == 0)

    # 合格条件(c): 系列的局所性ありクエリ・N最大で、(iii)の無駄プリフェッチ率が
    # (ii)比30ポイント以上低い。
    claim_c_pred_wasted = seq_summary.get("iii_predictive_wasted_rate", {}).get("mean")
    claim_c_random_wasted = seq_summary.get("ii_random_wasted_rate", {}).get("mean")
    claim_c_diff_points = None
    claim_c_ok = False
    if claim_c_pred_wasted is not None and claim_c_random_wasted is not None:
        claim_c_diff_points = (claim_c_random_wasted - claim_c_pred_wasted) * 100.0
        claim_c_ok = bool(claim_c_diff_points >= 30.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ34（小脳モジュールとの予測的プリフェッチ結合, 12.6.76〜77節）: "
                "プリフェッチなし(i)・ランダムプリフェッチ(ii)・予測的プリフェッチ(iii、"
                "頻度＋1次マルコフ連鎖ベース)を、Zipf偏りクエリ(A)・系列的局所性クエリ(B)"
                "の2種で比較。キャッシュ容量64チャンク固定",
        "条件": vars(args) | {"out": str(args.out), "cache_size": CACHE_SIZE,
                              "n_warmup": N_WARMUP, "n_measure": args.n_measure,
                              "disk_key_chunk": DISK_KEY_CHUNK,
                              "markov_window_note": "頻度・遷移表は測定ごとにリセット"
                                                     "（各回を独立試行として的中率・"
                                                     "無駄プリフェッチ率を計測するため）",
                              "platform": sys.platform,
                              "peak_memory_metric": ("Windows: GetProcessMemoryInfo."
                                                      "PeakWorkingSetSize（参考値、合格条件に"
                                                      "含めない）" if sys.platform == "win32"
                                                      else "resource.getrusage().ru_maxrss（参考値）")},
        "n_list": list(args.n_list),
        "conditions": conditions,
        "query_kinds": query_kinds,
        "summary": summary,
        "claim_a_query_kind": "sequential",
        "claim_a_n": max_n,
        "claim_a_cache_size": CACHE_SIZE,
        "claim_a_reduction_vs_i_none（20%以上が条件の一部）": claim_a_reduction_vs_none,
        "claim_a_reduction_vs_ii_random（10%以上が条件の一部）": claim_a_reduction_vs_random,
        "claim_a_ok": claim_a_ok,
        "claim_a_zipf_reduction_vs_i_none（参考値、系列的局所性なし）": claim_a_zipf_reduction_vs_none,
        "claim_a_zipf_reduction_vs_ii_random（参考値、系列的局所性なし）": claim_a_zipf_reduction_vs_random,
        "claim_b_accuracy_mismatches": accuracy_mismatches,
        "claim_b_ok（全条件・全N・全クエリ列・全シードでvalue・best・argがtorch.equal完全一致）": claim_b_ok,
        "claim_c_query_kind": "sequential",
        "claim_c_n": max_n,
        "claim_c_predictive_wasted_rate_mean": claim_c_pred_wasted,
        "claim_c_random_wasted_rate_mean": claim_c_random_wasted,
        "claim_c_diff_points（30ポイント以上低いことが条件）": claim_c_diff_points,
        "claim_c_ok": claim_c_ok,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    print(f"主張(a) sequential N={max_n} cache={CACHE_SIZE}: "
          f"vs_none={claim_a_reduction_vs_none} vs_random={claim_a_reduction_vs_random}  ok={claim_a_ok}")
    print(f"主張(b) 不一致件数: {len(accuracy_mismatches)}  ok={claim_b_ok}")
    print(f"主張(c) 無駄プリフェッチ率差={claim_c_diff_points}ポイント  ok={claim_c_ok}")
    print(f"所要時間: {time.time() - t0:.1f}秒")


if __name__ == "__main__":
    main()
