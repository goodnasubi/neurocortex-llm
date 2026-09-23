"""海馬モジュール（12.6.4節・ステップ1）。

皮質バックボーン `SpikingLM` の上に後付けする、勾配を用いない連想メモリ。

  x_t（残差ストリーム）→ DG: パターン分離（k-WTA疎SDR）→ CA3: 連想ストア → m_t
  logits = head(ln_f(x_t) + g * m_t)

12.6.4節の主指標は「想起できること」ではなく「**パターン分離が効くこと**」である。
`softmax(βKᵀq)V` は β→∞ で厳密な最近傍探索（＝dict）に収束するため、想起できた
だけでは dict との区別がつかない。したがって比較可能な形で分離層を差し替えられる
ことが、このモジュールの設計上もっとも重要な性質である（`PatternSeparator.mode`）。

書き込みは推論時オンラインで行い、勾配を一切通さない（12.3.2節）。この性質は
`tests/test_path_audit.py` で機械的に保証する。
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

try:
    import faiss.gpu
    _gpu_resources = faiss.StandardGpuResources()
    _gpu_available = True
except (ImportError, RuntimeError):
    _gpu_resources = None  # type: ignore
    _gpu_available = False


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _row_zscore(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """[..., k] を最終次元方向・行ごとに、`mask`で有効な要素だけを使ってz-score正規化する。

    12.6.89節（ステップ41再検証）: `read_with_hybrid_rerank`の複合スコアで、
    スケールの異なる内積・密度・テンプレート各成分を合成前に比較可能な
    スケールへ揃えるために使う。無効要素（マスク外）は0を返す。
    """
    mask_f = mask.to(x.dtype)
    cnt = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * mask_f).sum(dim=-1, keepdim=True) / cnt
    var = ((x - mean) ** 2 * mask_f).sum(dim=-1, keepdim=True) / cnt
    std = var.clamp_min(eps ** 2).sqrt()
    z = (x - mean) / std
    return z.masked_fill(~mask, 0.0)


class PatternSeparator(nn.Module):
    """歯状回（DG）に対応するパターン分離層（6.1節）。

    3つのモードを持つ。これは利便性ではなく**アブレーションそのもの**であり、
    「効いているのは高次元化なのか疎化なのか」を分離するために必要である
    （12.6.4節のアブレーション3水準）。

      - ``"sdr"``      : 固定ランダム射影 + k-WTA による二値疎表現（本設計）
      - ``"dense"``    : 同次元の密なランダム射影（疎化だけを取り除いた対照）
      - ``"identity"`` : 皮質表現をそのままキーにする（分離層なしの対照）

    射影は固定であり学習しない（ステップ1）。STDPによる学習はステップ5。
    出力は常にL2正規化されているので、内積がそのままコサイン類似度になる。
    k-WTAの出力は非ゼロ成分が常にk個なので、内積は「重なり数 / k」に一致する。
    """

    def __init__(
        self,
        d_model: int,
        n_units: int = 2048,
        k: int = 40,
        mode: str = "sdr",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if mode not in ("sdr", "dense", "identity"):
            raise ValueError(f"未知のモード: {mode}")
        if mode != "identity" and not 1 <= k <= n_units:
            raise ValueError(f"k は 1..{n_units} の範囲（受領: {k}）")
        self.mode = mode
        self.k = k
        self.n_units = n_units
        self.out_dim = d_model if mode == "identity" else n_units
        if mode == "identity":
            self.register_buffer("weight", torch.empty(0), persistent=False)
        else:
            g = torch.Generator().manual_seed(seed)
            w = torch.randn(n_units, d_model, generator=g) / d_model**0.5
            # 学習しないのでパラメータではなくバッファとして持つ。
            # これにより requires_grad を持つテンソルがこの層に一切存在しなくなる。
            self.register_buffer("weight", w, persistent=True)

    @property
    def sparsity(self) -> float:
        """非ゼロ成分の割合。identity では定義できないので 1.0 を返す。"""
        return 1.0 if self.mode == "identity" else self.k / self.n_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., d_model] → [..., out_dim]（L2正規化済み）。"""
        x = _l2_normalize(x)
        if self.mode == "identity":
            return x
        h = x @ self.weight.T
        if self.mode == "dense":
            return _l2_normalize(h)
        # k-WTA: 上位k個だけを1にする。固定閾値ではなく順位で切ることで、
        # 入力ノルムによらず疎度がkに固定される（12.6.4節）。
        idx = h.topk(self.k, dim=-1).indices
        sdr = torch.zeros_like(h).scatter_(-1, idx, 1.0)
        return sdr / self.k**0.5


@dataclass
class StoreStats:
    """読み出し1回分の診断値。"""

    max_score: torch.Tensor  # [B] 最良一致のコサイン類似度
    top1_index: torch.Tensor  # [B] 最良一致のスロット番号（-1 は空ストア）


class AssociativeStore:
    """CA3に対応する自己連想メモリ（追記のみのキー・バリュー表）。

    キーがL2正規化されているので ``scores = K @ q`` はコサイン類似度になる。
    ``sdr`` モードではキーが二値であるため、この積和は**乗算を伴わない**
    （12.11.3節のイベント駆動実装と同じ論法が適用できる。ただし本実装は
    PyTorchの密な行列積で計算しており、その利点は実測していない）。

    ``exact=True`` にすると softmax を使わず最良一致の値をそのまま返す。これは
    β→∞ の極限であり、12.6.4節の統制3「dict 対照群」に対応する。
    """

    def __init__(self, key_dim: int, value_dim: int, beta: float = 50.0,
                 exact: bool = False, device: torch.device | None = None) -> None:
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.beta = beta
        self.exact = exact
        self._device = device or torch.device("cpu")
        self._keys = torch.zeros(0, key_dim, device=self._device)
        self._values = torch.zeros(0, value_dim, device=self._device)
        self.write_count = 0

    def __len__(self) -> int:
        return self._keys.shape[0]

    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """追記する。勾配は通さない（12.3.2節「勾配を通さない局所更新」）。"""
        if keys.shape[0] != values.shape[0]:
            raise ValueError("キーと値の件数が一致しない")
        if keys.shape[-1] != self.key_dim or values.shape[-1] != self.value_dim:
            raise ValueError("次元が一致しない")
        self._keys = torch.cat([self._keys, keys.detach().to(self._device)])
        self._values = torch.cat([self._values, _l2_normalize(values.detach()).to(self._device)])
        self.write_count += keys.shape[0]

    @torch.no_grad()
    def read(self, keys: torch.Tensor, chunk: int = 512) -> tuple[torch.Tensor, StoreStats]:
        """[B, key_dim] → ([B, value_dim], 診断値)。空ストアではゼロを返す。"""
        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        if len(self) == 0:
            return out, StoreStats(best, arg)
        for i in range(0, b, chunk):
            q = keys[i : i + chunk].to(self._device)
            scores = q @ self._keys.T  # コサイン類似度 [b', N]
            top = scores.max(dim=-1)
            best[i : i + chunk] = top.values.to(keys.device)
            arg[i : i + chunk] = top.indices.to(keys.device)
            if self.exact:
                v = self._values[top.indices]
            else:
                v = torch.softmax(self.beta * scores, dim=-1) @ self._values
            out[i : i + chunk] = v.to(keys.device)
        return out, StoreStats(best, arg)

    def clear(self) -> None:
        """統制2「因果的除去」用。ストアを空にする。"""
        self._keys = torch.zeros(0, self.key_dim, device=self._device)
        self._values = torch.zeros(0, self.value_dim, device=self._device)

    # ------------------------------------------------------------------
    # ステップ23（12.6.54節の設計）: ANN候補絞り込みモード。
    # `write`・`read`のコード・挙動は一切変更せず、読み出し専用の付加機能として
    # 追加する。k-meansによる粗量子化＋2段階探索（`clusters`個のクラスタに
    # 割り当てた上で、クエリに近いクラスタ`n_probe`個だけを全件探索する）を
    # 自前実装する。クラスタ内で厳密な argmax を取る（＝`exact=True`相当の返し方）。
    # ------------------------------------------------------------------

    def _ensure_ann_index(self, n_clusters: int, n_iters: int, seed: int) -> None:
        """粗量子化用のk-means重心・クラスタ所属索引を（必要なら）再構築する。

        `write`でキー件数が変わった場合や、ハイパーパラメータ（クラスタ数・
        反復数・シード）が変わった場合のみ再構築する。索引はCPU上に保持する。
        コサイン類似度を使うため、重心も毎反復L2正規化する（球面k-meansの簡易版）。
        """
        n_total = len(self)
        nc = max(1, min(n_clusters, n_total))
        cache_key = (nc, n_iters, seed, n_total)
        if getattr(self, "_ann_cache_key", None) == cache_key:
            return
        keys_cpu = self._keys.detach().to("cpu")
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=g)[:nc]
        centroids = keys_cpu[perm].clone()
        assign = torch.zeros(n_total, dtype=torch.long)
        for _ in range(n_iters):
            sims = keys_cpu @ centroids.T  # [N, nc]
            assign = sims.argmax(dim=1)
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(nc)
            new_centroids.index_add_(0, assign, keys_cpu)
            counts.index_add_(0, assign, torch.ones(n_total))
            empty = counts == 0
            new_centroids = new_centroids / counts.clamp_min(1).unsqueeze(1)
            norm = new_centroids.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            new_centroids = new_centroids / norm
            new_centroids[empty] = centroids[empty]
            centroids = new_centroids
        buckets: dict[int, list[int]] = {}
        for idx, c in enumerate(assign.tolist()):
            buckets.setdefault(c, []).append(idx)
        self._ann_centroids = centroids
        self._ann_buckets = buckets
        self._ann_cache_key = cache_key

    @torch.no_grad()
    def read_approx(
        self,
        keys: torch.Tensor,
        n_clusters: int | None = None,
        n_probe: int = 8,
        n_iters: int = 5,
        seed: int = 0,
        min_candidates: int = 64,
    ) -> tuple[torch.Tensor, StoreStats]:
        """`read`のANN版（k-means粗量子化による候補絞り込み）。

        キーを`n_clusters`個のクラスタに粗量子化し（デフォルトは概ね
        sqrt(N)個）、クエリに最も近いクラスタ`n_probe`個の中だけで厳密な
        内積argmaxを取る。候補が`min_candidates`に満たない場合は全件からの
        一様サンプルで補う（小規模N・空クラスタ時の劣化防止）。
        `exact=True`の`read`と同じ形（softmaxを使わない最良一致）で値を返す。
        空ストアではゼロを返す。
        """
        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        if len(self) == 0:
            return out, StoreStats(best, arg)

        n_total = len(self)
        if n_clusters is None:
            n_clusters = max(1, int(round(n_total ** 0.5)))
        self._ensure_ann_index(n_clusters, n_iters, seed)
        centroids = self._ann_centroids
        buckets = self._ann_buckets
        nc = centroids.shape[0]
        n_probe_eff = min(n_probe, nc)

        keys_cpu = keys.detach().to("cpu")
        csims = keys_cpu @ centroids.T  # [B, nc]
        top_clusters = csims.topk(n_probe_eff, dim=-1).indices  # [B, n_probe_eff]

        rng = np.random.default_rng(seed)
        keys_dev = self._keys  # [N, key_dim] on store device
        values_dev = self._values

        for i in range(b):
            cand: list[int] = []
            for c in top_clusters[i].tolist():
                cand.extend(buckets.get(c, []))
            if cand:
                cand = list(dict.fromkeys(cand))
            if len(cand) < min_candidates:
                extra_n = min(min_candidates - len(cand), n_total)
                if extra_n > 0:
                    extra = rng.choice(n_total, size=extra_n, replace=False).tolist()
                    cand = list(dict.fromkeys(cand + extra))
            cand_idx = torch.tensor(cand, dtype=torch.long, device=self._device)
            q = keys[i : i + 1].to(self._device)
            scores = (q @ keys_dev[cand_idx].T).squeeze(0)  # [len(cand)]
            top = scores.max(dim=-1)
            best[i] = top.values.to(keys.device)
            arg[i] = cand_idx[top.indices].to(keys.device)
            out[i] = values_dev[cand_idx[top.indices]].to(keys.device)
        return out, StoreStats(best, arg)

    # ------------------------------------------------------------------
    # ステップ24（12.6.56節の設計）: `read_approx`のベクトル化再実装。
    # クラスタ索引構築（`_ensure_ann_index`）・候補絞り込みのアルゴリズムは
    # 一切変更しない。変更対象は候補探索部分のみ: クエリごとのPythonループを
    # 「同一の候補クラスタ集合を共有するクエリ群」単位のバッチ行列積に置き換える。
    # ------------------------------------------------------------------

    @torch.no_grad()
    def read_approx_batched(
        self,
        keys: torch.Tensor,
        n_clusters: int | None = None,
        n_probe: int = 8,
        n_iters: int = 5,
        seed: int = 0,
        min_candidates: int = 64,
    ) -> tuple[torch.Tensor, StoreStats]:
        """`read_approx`のベクトル化再実装（12.6.56節）。

        `read_approx`と同一のアルゴリズム（k-means粗量子化による2段階探索、
        `min_candidates`による一様サンプル補完）を使うが、クエリごとの
        Pythonループ（候補集合の構築・部分行列積を1件ずつ実行）を、
        「上位`n_probe`クラスタの集合が同一のクエリ」をまとめてバッチ行列積で
        処理する形に置き換える。候補集合はクラスタ集合が同一なら同一件数に
        揃うため、グループ内でのパディングは不要（グループ間の候補件数の
        不揃いは、グループごとに独立した行列積で吸収される）。
        `read_approx`と数学的に同じ候補選択・厳密argmaxを行うため、
        結果はほぼ一致するはず（乱数補完の消費順序が異なるため`min_candidates`
        による補完候補がわずかに変わりうる点のみ相違しうる）。
        """
        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        if len(self) == 0:
            return out, StoreStats(best, arg)

        n_total = len(self)
        if n_clusters is None:
            n_clusters = max(1, int(round(n_total ** 0.5)))
        self._ensure_ann_index(n_clusters, n_iters, seed)
        centroids = self._ann_centroids
        buckets = self._ann_buckets
        nc = centroids.shape[0]
        n_probe_eff = min(n_probe, nc)

        keys_cpu = keys.detach().to("cpu")
        csims = keys_cpu @ centroids.T  # [B, nc]
        top_clusters = csims.topk(n_probe_eff, dim=-1).indices  # [B, n_probe_eff]
        # クラスタ集合が同一なら候補集合も同一になるよう、ソートしてグループ化する。
        top_clusters_sorted, _ = torch.sort(top_clusters, dim=-1)

        rng = np.random.default_rng(seed)
        keys_dev = self._keys
        values_dev = self._values

        # 全クエリの所属クラスタ集合をグルーピングする（`torch.bincount`相当の
        # 効果を持つ辞書ベースのグルーピング。B件のPythonループを、Bよりも
        # 少ない「重複しないクラスタ集合」の件数分のループに削減する）。
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, row in enumerate(top_clusters_sorted.tolist()):
            groups.setdefault(tuple(row), []).append(i)

        for cluster_key, q_indices in groups.items():
            cand: list[int] = []
            for c in cluster_key:
                cand.extend(buckets.get(c, []))
            if cand:
                cand = list(dict.fromkeys(cand))
            if len(cand) < min_candidates:
                extra_n = min(min_candidates - len(cand), n_total)
                if extra_n > 0:
                    extra = rng.choice(n_total, size=extra_n, replace=False).tolist()
                    cand = list(dict.fromkeys(cand + extra))
            cand_idx = torch.tensor(cand, dtype=torch.long, device=self._device)
            q_idx_t = torch.tensor(q_indices, dtype=torch.long)
            q = keys[q_idx_t].to(self._device)  # [g, key_dim]
            cand_keys = keys_dev[cand_idx]  # [C, key_dim]
            scores = q @ cand_keys.T  # [g, C] グループ内クエリをまとめたバッチ行列積
            top = scores.max(dim=-1)
            best[q_idx_t] = top.values.to(keys.device)
            sel = cand_idx[top.indices]
            arg[q_idx_t] = sel.to(keys.device)
            out[q_idx_t] = values_dev[sel].to(keys.device)
        return out, StoreStats(best, arg)


class DiskBackedAssociativeStore:
    """ステップ32（12.6.72節の設計）: `AssociativeStore`のディスク常駐版。

    `AssociativeStore`の`write`・`read`のコード・挙動は一切変更しない
    （既存クラスに触れず、独立クラスとして並置する）。土台は同じだが、
    `_keys`・`_values`をRAM/GPU上のテンソルとして常時保持する代わりに、
    `numpy.memmap`でディスク上のバイナリファイルに追記し、`read`時にのみ
    必要な範囲（`key_chunk`件ずつ）だけをRAM上にストリームして読み込む。

    正確性は`exact`モード（softmaxを使わない最良一致）でのみ厳密に
    `AssociativeStore(exact=True)`と一致することを保証する。非exactモード
    （softmax加重和）はチャンク単位の2パス方式で数学的に同一の値を計算するが、
    浮動小数点の加算順序がPyTorchの単一`softmax`呼び出しと異なるため、
    `torch.equal`によるビット完全一致までは保証しない（`torch.allclose`相当）。
    """

    def __init__(self, key_dim: int, value_dim: int, persist_dir: str | Path,
                 beta: float = 50.0, exact: bool = False,
                 key_chunk: int = 4096, cache_size: int = 0) -> None:
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.beta = beta
        self.exact = exact
        self.key_chunk = key_chunk
        self.cache_size = cache_size
        # ステップ33（12.6.75節）: チャンク単位のLRUキャッシュ（キー: チャンク開始位置j、
        # 値: (store_keys, store_values)）。cache_size=0では一切参照・更新されず、
        # 既存（ステップ32）の経路と完全に同一のコードパスを通る。
        self._chunk_cache: "OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self._dir = Path(persist_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keys_path = self._dir / "keys.f32.bin"
        self._values_path = self._dir / "values.f32.bin"
        self._meta_path = self._dir / "meta.json"
        if self._meta_path.exists():
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if meta["key_dim"] != key_dim or meta["value_dim"] != value_dim:
                raise ValueError("既存の永続化ディレクトリと次元が一致しない")
            self.write_count = meta["write_count"]
        else:
            self._keys_path.touch()
            self._values_path.touch()
            self.write_count = 0
            self._save_meta()
        # ステップ36（12.6.80節）: faiss IVF インデックス（初期化時は未構築）。
        self._faiss_index = None
        self._faiss_nlist = 100  # IVF クラスタ数
        self._faiss_nprobe = 5   # 探索時のクラスタ調査数

    def _save_meta(self) -> None:
        self._meta_path.write_text(json.dumps({
            "key_dim": self.key_dim, "value_dim": self.value_dim,
            "write_count": self.write_count,
        }), encoding="utf-8")

    def __len__(self) -> int:
        return self.write_count

    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """追記する。`AssociativeStore.write`と同じ検証・正規化規則。ディスクに追記するのみ。"""
        if keys.shape[0] != values.shape[0]:
            raise ValueError("キーと値の件数が一致しない")
        if keys.shape[-1] != self.key_dim or values.shape[-1] != self.value_dim:
            raise ValueError("次元が一致しない")
        keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
        values_np = _l2_normalize(values.detach()).to("cpu", dtype=torch.float32).numpy()
        with open(self._keys_path, "ab") as f:
            keys_np.tofile(f)
        with open(self._values_path, "ab") as f:
            values_np.tofile(f)
        self.write_count += keys.shape[0]
        self._save_meta()
        # 追記により末尾チャンクの内容が変わりうるため、キャッシュを無効化する。
        self._chunk_cache.clear()

    def _get_chunk(self, keys_mm: np.memmap, values_mm: np.memmap,
                    j: int, kc: int) -> tuple[torch.Tensor, torch.Tensor]:
        """チャンク`(j, j+kc)`のキー・バリューを返す（ステップ33のLRUキャッシュ経由）。

        `cache_size <= 0`の場合はキャッシュに一切触れず、ステップ32と同じ
        `np.array(...)` の都度読み出しのみを行う。
        """
        if self.cache_size <= 0:
            store_keys = torch.from_numpy(np.array(keys_mm[j: j + kc]))
            store_values = torch.from_numpy(np.array(values_mm[j: j + kc]))
            return store_keys, store_values
        cached = self._chunk_cache.get(j)
        if cached is not None:
            self._chunk_cache.move_to_end(j)
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        store_keys = torch.from_numpy(np.array(keys_mm[j: j + kc]))
        store_values = torch.from_numpy(np.array(values_mm[j: j + kc]))
        self._chunk_cache[j] = (store_keys, store_values)
        self._chunk_cache.move_to_end(j)
        if len(self._chunk_cache) > self.cache_size:
            self._chunk_cache.popitem(last=False)
        return store_keys, store_values

    def _keys_memmap(self) -> np.memmap:
        return np.memmap(self._keys_path, dtype=np.float32, mode="r",
                          shape=(self.write_count, self.key_dim))

    def _values_memmap(self) -> np.memmap:
        return np.memmap(self._values_path, dtype=np.float32, mode="r",
                          shape=(self.write_count, self.value_dim))

    @torch.no_grad()
    def read(self, keys: torch.Tensor, chunk: int = 512) -> tuple[torch.Tensor, StoreStats]:
        """[B, key_dim] → ([B, value_dim], 診断値)。`AssociativeStore.read`と同じ意味論。

        ストア側（N件のキー・バリュー）を`self.key_chunk`件ずつディスクから
        ストリームして処理するため、ピークメモリはNではなく`key_chunk`と
        クエリのチャンクサイズに依存する。
        """
        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        n_total = self.write_count
        if n_total == 0:
            return out, StoreStats(best, arg)

        keys_mm = self._keys_memmap()
        values_mm = self._values_memmap()
        kc = self.key_chunk

        for i in range(0, b, chunk):
            q = keys[i : i + chunk].to(dtype=torch.float32)  # [b', key_dim]
            bp = q.shape[0]
            chunk_best = torch.full((bp,), float("-inf"))
            chunk_arg = torch.full((bp,), -1, dtype=torch.long)
            # パス1: 全ストアチャンクを走査し、最良一致（value/best/arg）を求める。
            for j in range(0, n_total, kc):
                store_keys, _ = self._get_chunk(keys_mm, values_mm, j, kc)
                scores = q @ store_keys.T  # [b', c]
                top = scores.max(dim=-1)
                better = top.values > chunk_best
                chunk_arg[better] = top.indices[better] + j
                chunk_best[better] = top.values[better]

            if self.exact:
                # 最良一致1件の値だけをディスクから取り出す（追加のストリームは不要）。
                for k in range(bp):
                    a = int(chunk_arg[k])
                    if a >= 0:
                        out[i + k] = torch.from_numpy(np.array(values_mm[a])).to(out.dtype)
            else:
                # パス2: softmax(beta*scores)@values を数値的に安定な形（既知の最大値を
                # 引いてから exp）でチャンクごとに累積する。
                exp_sum = torch.zeros(bp, dtype=torch.float64)
                weighted_val = torch.zeros(bp, self.value_dim, dtype=torch.float64)
                for j in range(0, n_total, kc):
                    store_keys, store_values_f32 = self._get_chunk(keys_mm, values_mm, j, kc)
                    store_values = store_values_f32.to(torch.float64)
                    scores = (q @ store_keys.T).to(torch.float64)  # [b', c]
                    w = torch.exp(self.beta * (scores - chunk_best.to(torch.float64)[:, None]))
                    exp_sum += w.sum(dim=-1)
                    weighted_val += w @ store_values
                v = (weighted_val / exp_sum.clamp_min(1e-300)[:, None]).to(out.dtype)
                out[i : i + chunk] = v.to(keys.device)

            best[i : i + chunk] = chunk_best.to(keys.device)
            arg[i : i + chunk] = chunk_arg.to(keys.device)

        return out, StoreStats(best, arg)

    def clear(self) -> None:
        """統制2「因果的除去」用。ディスク上のファイルを空にする。"""
        self._keys_path.write_bytes(b"")
        self._values_path.write_bytes(b"")
        self.write_count = 0
        self._chunk_cache.clear()
        self._save_meta()

    # ------------------------------------------------------------------
    # ステップ34（12.6.76節の設計）: 小脳モジュールとの予測的プリフェッチ結合。
    # `read`・`_get_chunk`・`cache_size`・`_chunk_cache`のコード・挙動は
    # 一切変更しない（追記のみ）。呼び出し側（`run_cerebellum_prefetch.py`）が
    # 予測したチャンクIDを、実際の`read`呼び出しに先立って本メソッドで
    # キャッシュへ先読みする。
    # ------------------------------------------------------------------

    def prefetch(self, chunk_id: int) -> None:
        """チャンク`chunk_id`（0始まり、`key_chunk`件単位）をディスクから読み、
        `read`時と同じキャッシュ機構（`_get_chunk`）経由でキャッシュへ載せる。

        `cache_size<=0`では`_get_chunk`がキャッシュに触れないため、本メソッドは
        ディスクI/Oを行うだけで実質的な効果を持たない（`read`側と挙動が揃う）。
        `chunk_id`がストア範囲外の場合は何もしない。
        """
        n_total = self.write_count
        if n_total == 0:
            return
        j = chunk_id * self.key_chunk
        if j < 0 or j >= n_total:
            return
        keys_mm = self._keys_memmap()
        values_mm = self._values_memmap()
        self._get_chunk(keys_mm, values_mm, j, self.key_chunk)

    # ------------------------------------------------------------------
    # ステップ35（12.6.78節の設計）: キー単位走査への見直し（方式B: 固定k件）。
    # `read`・`_get_chunk`・`cache_size`・`_chunk_cache`のコード・挙動は
    # 一切変更しない（追記のみ）。
    # ------------------------------------------------------------------

    def _get_key_candidates(self, attention_weights: torch.Tensor, k: int) -> list[int]:
        """アテンション重み分布から上位k個の候補キーを抽出。

        Args:
            attention_weights: [n_keys] の確率分布またはスコア（通常は softmax 計算結果）
            k: 候補数（通常は sqrt(n_keys)）

        Returns:
            上位k個のキーインデックスリスト
        """
        if k >= len(attention_weights):
            return list(range(len(attention_weights)))
        _, top_indices = torch.topk(attention_weights, k=k, dim=-1)
        return top_indices.cpu().tolist()

    @torch.no_grad()
    def read_with_key_candidates(
        self, keys: torch.Tensor, attention_weights: torch.Tensor, chunk: int = 512,
        k_candidate: int | None = None
    ) -> tuple[torch.Tensor, StoreStats]:
        """キー候補絞り込みを使用した読み込み（方式B: 固定k件）。

        attention_weights から上位k件の候補キーを抽出し、
        その候補キーのみを対象に従来と同じ read() を実行する。

        Args:
            keys: [B, key_dim] クエリキー
            attention_weights: [B, n_stored_keys] アテンション重み分布
            chunk: クエリのチャンクサイズ
            k_candidate: 候補サイズ（None の場合は sqrt(n_total)）

        Returns:
            ([B, value_dim], StoreStats) - 従来の read() と同じ
        """
        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        n_total = self.write_count
        if n_total == 0:
            return out, StoreStats(best, arg)

        keys_mm = self._keys_memmap()
        values_mm = self._values_memmap()
        kc = self.key_chunk

        # 候補サイズのデフォルト設定
        if k_candidate is None:
            k_candidate = max(1, int(n_total**0.5))

        for i in range(0, b, chunk):
            q = keys[i : i + chunk].to(dtype=torch.float32)  # [b', key_dim]
            attn = attention_weights[i : i + chunk]  # [b', n_stored_keys]
            bp = q.shape[0]
            chunk_best = torch.full((bp,), float("-inf"))
            chunk_arg = torch.full((bp,), -1, dtype=torch.long)

            # 各バッチごとに候補キーを抽出し、セット化
            candidate_sets = [
                set(self._get_key_candidates(attn[k], k_candidate)) for k in range(bp)
            ]

            # パス1: 全ストアを走査し、候補内に限定して最良一致を求める。
            for j in range(0, n_total, kc):
                store_keys, _ = self._get_chunk(keys_mm, values_mm, j, kc)
                scores = q @ store_keys.T  # [b', kc]

                # バッチごとに処理
                for k in range(bp):
                    candidate_set = candidate_sets[k]
                    # チャンク内のインデックスをチェック
                    actual_chunk_size = min(kc, n_total - j)
                    for local_idx in range(actual_chunk_size):
                        global_idx = j + local_idx
                        if global_idx in candidate_set:
                            score = scores[k, local_idx].item()
                            if score > chunk_best[k]:
                                chunk_best[k] = score
                                chunk_arg[k] = global_idx

            if self.exact:
                # 最良一致1件の値だけをディスクから取り出す。
                for k in range(bp):
                    a = int(chunk_arg[k])
                    if a >= 0:
                        out[i + k] = torch.from_numpy(np.array(values_mm[a])).to(out.dtype)
            else:
                # パス2: softmax(beta*scores)@values を累積。候補キーのみを対象。
                exp_sum = torch.zeros(bp, dtype=torch.float64)
                weighted_val = torch.zeros(bp, self.value_dim, dtype=torch.float64)
                for j in range(0, n_total, kc):
                    store_keys, store_values_f32 = self._get_chunk(keys_mm, values_mm, j, kc)
                    store_values = store_values_f32.to(torch.float64)
                    scores = (q @ store_keys.T).to(torch.float64)  # [b', kc]
                    actual_chunk_size = min(kc, n_total - j)

                    for k in range(bp):
                        candidate_set = candidate_sets[k]
                        for local_idx in range(actual_chunk_size):
                            global_idx = j + local_idx
                            if global_idx in candidate_set:
                                score = scores[k, local_idx]
                                w = torch.exp(self.beta * (score - chunk_best[k]))
                                exp_sum[k] += w.item()
                                weighted_val[k] += w.item() * store_values[local_idx]

                v = (weighted_val / exp_sum.clamp_min(1e-300)[:, None]).to(out.dtype)
                out[i : i + chunk] = v.to(keys.device)

            best[i : i + chunk] = chunk_best.to(keys.device)
            arg[i : i + chunk] = chunk_arg.to(keys.device)

        return out, StoreStats(best, arg)

    # ------------------------------------------------------------------
    # ステップ36（12.6.80節の設計）: faiss IVF インデックスによる高速キー検索。
    # `read`・`_get_chunk`・`cache_size`・`_chunk_cache`のコード・挙動は
    # 一切変更しない（追記のみ）。
    # ------------------------------------------------------------------

    def _train_faiss_index(self, keys_np: np.ndarray) -> None:
        """faiss IVF インデックスを学習し、すべてのキーを追加する。

        Args:
            keys_np: [N, key_dim] のfloat32 numpy配列
        """
        if faiss is None:
            raise ImportError("faiss がインストールされていません (pip install faiss-cpu)")

        n, d = keys_np.shape
        if d != self.key_dim:
            raise ValueError(f"キー次元が一致しない: {d} != {self.key_dim}")

        # IVF インデックス（InnerProduct メトリクス）を作成。
        # quantizer は簡潔性のため Flat（非量子化）を使用。
        nlist = min(self._faiss_nlist, max(1, n // 100))  # 小規模時は調整
        quantizer = faiss.IndexFlatIP(d)
        self._faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)

        # 訓練用サンプルで IVF を訓練（全キーを投入するため n > nlist が前提）。
        if n > nlist:
            train_sample = keys_np[::max(1, n // (nlist * 4))]
            self._faiss_index.train(train_sample.astype(np.float32))

        # すべてのキーを追加。
        self._faiss_index.add(keys_np.astype(np.float32))
        self._faiss_index.nprobe = self._faiss_nprobe

    def add_to_faiss_index(self, keys_np: np.ndarray) -> None:
        """既存の faiss インデックスに新しいキーを追加。

        インデックスが未初期化の場合は _train_faiss_index で初期化。

        Args:
            keys_np: [n, key_dim] のfloat32 numpy配列
        """
        if faiss is None:
            raise ImportError("faiss がインストールされていません (pip install faiss-cpu)")

        n, d = keys_np.shape
        if d != self.key_dim:
            raise ValueError(f"キー次元が一致しない: {d} != {self.key_dim}")

        if self._faiss_index is None:
            self._train_faiss_index(keys_np)
        else:
            self._faiss_index.add(keys_np.astype(np.float32))

    @torch.no_grad()
    def read_with_faiss_search(
        self, keys: torch.Tensor, chunk: int = 512, k_candidates: int = 10
    ) -> tuple[torch.Tensor, StoreStats]:
        """faiss による候補絞り込みを用いた読み込み（バッチ最適化版）。

        全ストアキーに対する最近傍探索を faiss で実行し、上位 k_candidates を
        取得した後、候補インデックスをソートしてバッチ読み込みすることで
        memmap のシーケンシャルアクセスを実現。その後、softmax 加重和を計算する。

        Args:
            keys: [B, key_dim] クエリキー
            chunk: クエリのチャンクサイズ
            k_candidates: faiss 検索で取得する候補数

        Returns:
            ([B, value_dim], StoreStats) - 従来の read() と同じ
        """
        if faiss is None:
            raise ImportError("faiss がインストールされていません (pip install faiss-cpu)")

        if self._faiss_index is None:
            raise RuntimeError("faiss インデックスが未初期化です")

        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        n_total = self.write_count

        if n_total == 0:
            return out, StoreStats(best, arg)

        keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()

        k_search = min(k_candidates, n_total)
        distances, indices = self._faiss_index.search(keys_np, k_search)

        keys_mm = self._keys_memmap()
        values_mm = self._values_memmap()

        for i in range(0, b, chunk):
            q = keys[i : i + chunk].to(dtype=torch.float32)
            bp = q.shape[0]
            cand_indices = indices[i : i + chunk]

            chunk_best = torch.full((bp,), float("-inf"))
            chunk_arg = torch.full((bp,), -1, dtype=torch.long)

            all_candidates = set()
            for b_idx in range(bp):
                for c in cand_indices[b_idx]:
                    if c >= 0 and c < n_total:
                        all_candidates.add(int(c))

            if not all_candidates:
                best[i : i + chunk] = chunk_best.to(keys.device)
                arg[i : i + chunk] = chunk_arg.to(keys.device)
                continue

            sorted_cand_list = np.sort(np.array(list(all_candidates)))
            idx_to_pos = {int(c): pos for pos, c in enumerate(sorted_cand_list)}

            cand_keys_batch = torch.from_numpy(
                np.array([keys_mm[c] for c in sorted_cand_list])
            ).to(torch.float32)
            cand_vals_batch = torch.from_numpy(
                np.array([values_mm[c] for c in sorted_cand_list])
            ).to(torch.float64)

            for b_idx in range(bp):
                candidates = set(cand_indices[b_idx].tolist())
                for cand_idx in candidates:
                    if cand_idx >= 0 and cand_idx < n_total:
                        pos = idx_to_pos[int(cand_idx)]
                        key_val = cand_keys_batch[pos]
                        score = q[b_idx] @ key_val
                        if score > chunk_best[b_idx]:
                            chunk_best[b_idx] = score
                            chunk_arg[b_idx] = cand_idx

            if self.exact:
                for k in range(bp):
                    a = int(chunk_arg[k])
                    if a >= 0:
                        out[i + k] = torch.from_numpy(np.array(values_mm[a])).to(
                            out.dtype
                        )
            else:
                exp_sum = torch.zeros(bp, dtype=torch.float64)
                weighted_val = torch.zeros(bp, self.value_dim, dtype=torch.float64)

                for b_idx in range(bp):
                    candidates = set(cand_indices[b_idx].tolist())
                    for cand_idx in candidates:
                        if cand_idx >= 0 and cand_idx < n_total:
                            pos = idx_to_pos[int(cand_idx)]
                            key_val = cand_keys_batch[pos].to(torch.float64)
                            val = cand_vals_batch[pos]
                            score = q[b_idx].to(torch.float64) @ key_val
                            w = torch.exp(self.beta * (score - chunk_best[b_idx]))
                            exp_sum[b_idx] += w.item()
                            weighted_val[b_idx] += w.item() * val

                for k in range(bp):
                    if exp_sum[k] > 0:
                        v = (weighted_val[k] / exp_sum[k]).to(out.dtype)
                        out[i + k] = v.to(keys.device)

            best[i : i + chunk] = chunk_best.to(keys.device)
            arg[i : i + chunk] = chunk_arg.to(keys.device)

        return out, StoreStats(best, arg)

    def _build_pq_index(
        self, keys_np: np.ndarray, M: int = 16, nbits: int = 8
    ) -> None:
        """Product Quantization（PQ）インデックスを構築。

        Args:
            keys_np: [N, key_dim] キーベクトル（float32）
            M: サブクォンタイザ数（次元分割粒度）
            nbits: 各サブクォンタイザのビット幅（8bit = 256コードワード）
        """
        if faiss is None:
            raise ImportError("faiss がインストールされていません")

        d = keys_np.shape[1]
        if d % M != 0:
            raise ValueError(
                f"key_dim ({d}) は M ({M}) で割り切れる必要があります"
            )

        # PQ インデックスを作成
        self._pq_index = faiss.IndexPQ(d, M, nbits)
        self._pq_index.train(keys_np.astype(np.float32))
        self._pq_index.add(keys_np.astype(np.float32))
        self._pq_M = M

    @torch.no_grad()
    def read_with_pq_search(
        self, keys: torch.Tensor, chunk: int = 512, k_candidates: int = 10, M: int = 16
    ) -> tuple[torch.Tensor, StoreStats]:
        """Product Quantization（PQ）による候補絞り込みを用いた読み込み。

        PQ インデックスで候補検索を高速化しつつ、インデックスメモリを削減。

        Args:
            keys: [B, key_dim] クエリキー
            chunk: クエリのチャンクサイズ
            k_candidates: PQ 検索で取得する候補数
            M: PQ のサブクォンタイザ数

        Returns:
            ([B, value_dim], StoreStats) - 従来の read() と同じ
        """
        if faiss is None:
            raise ImportError("faiss がインストールされていません")

        # PQ インデックスが未構築の場合は構築
        if not hasattr(self, "_pq_index") or self._pq_index is None:
            keys_mm = self._keys_memmap()
            self._build_pq_index(keys_mm, M=M)

        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        n_total = self.write_count

        if n_total == 0:
            return out, StoreStats(best, arg)

        keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()

        k_search = min(k_candidates, n_total)
        # PQ インデックスで候補検索（距離ベース）
        distances, indices = self._pq_index.search(keys_np, k_search)

        keys_mm = self._keys_memmap()
        values_mm = self._values_memmap()

        for i in range(0, b, chunk):
            q = keys[i : i + chunk].to(dtype=torch.float32)
            bp = q.shape[0]
            cand_indices = indices[i : i + chunk]

            chunk_best = torch.full((bp,), float("-inf"))
            chunk_arg = torch.full((bp,), -1, dtype=torch.long)

            # 候補インデックスを集約・ソート
            all_candidates = set()
            for b_idx in range(bp):
                for c in cand_indices[b_idx]:
                    if c >= 0 and c < n_total:
                        all_candidates.add(int(c))

            if not all_candidates:
                best[i : i + chunk] = chunk_best.to(keys.device)
                arg[i : i + chunk] = chunk_arg.to(keys.device)
                continue

            sorted_cand_list = np.sort(np.array(list(all_candidates)))
            idx_to_pos = {int(c): pos for pos, c in enumerate(sorted_cand_list)}

            # バッチ読み込み
            cand_keys_batch = torch.from_numpy(
                np.array([keys_mm[c] for c in sorted_cand_list])
            ).to(torch.float32)
            cand_vals_batch = torch.from_numpy(
                np.array([values_mm[c] for c in sorted_cand_list])
            ).to(torch.float64)

            # スコア計算
            for b_idx in range(bp):
                candidates = set(cand_indices[b_idx].tolist())
                for cand_idx in candidates:
                    if cand_idx >= 0 and cand_idx < n_total:
                        pos = idx_to_pos[int(cand_idx)]
                        key_val = cand_keys_batch[pos]
                        score = q[b_idx] @ key_val
                        if score > chunk_best[b_idx]:
                            chunk_best[b_idx] = score
                            chunk_arg[b_idx] = cand_idx

            # 出力値の計算
            if self.exact:
                for k in range(bp):
                    a = int(chunk_arg[k])
                    if a >= 0:
                        out[i + k] = torch.from_numpy(np.array(values_mm[a])).to(
                            out.dtype
                        )
            else:
                exp_sum = torch.zeros(bp, dtype=torch.float64)
                weighted_val = torch.zeros(bp, self.value_dim, dtype=torch.float64)

                for b_idx in range(bp):
                    candidates = set(cand_indices[b_idx].tolist())
                    for cand_idx in candidates:
                        if cand_idx >= 0 and cand_idx < n_total:
                            pos = idx_to_pos[int(cand_idx)]
                            key_val = cand_keys_batch[pos].to(torch.float64)
                            val = cand_vals_batch[pos]
                            score = q[b_idx].to(torch.float64) @ key_val
                            w = torch.exp(self.beta * (score - chunk_best[b_idx]))
                            exp_sum[b_idx] += w.item()
                            weighted_val[b_idx] += w.item() * val

                for k in range(bp):
                    if exp_sum[k] > 0:
                        v = (weighted_val[k] / exp_sum[k]).to(out.dtype)
                        out[i + k] = v.to(keys.device)

            best[i : i + chunk] = chunk_best.to(keys.device)
            arg[i : i + chunk] = chunk_arg.to(keys.device)

        return out, StoreStats(best, arg)

    def read_with_pq_search_gpu(
        self, keys: torch.Tensor, chunk: int = 512, k_candidates: int = 10, M: int = 16
    ) -> tuple[torch.Tensor, StoreStats]:
        """GPU 最適化版 PQ 検索（GPU 利用可能時）。

        ステップ39: GPU faiss IndexPQ による高速化。
        GPU が利用不可の場合は自動的に read_with_pq_search() にフォールバック。

        Args:
            keys: [B, key_dim] クエリキー
            chunk: クエリのチャンクサイズ
            k_candidates: PQ 検索で取得する候補数
            M: PQ のサブクォンタイザ数

        Returns:
            ([B, value_dim], StoreStats)
        """
        if not _gpu_available:
            # GPU 不可の場合は CPU 版にフォールバック
            return self.read_with_pq_search(keys, chunk=chunk, k_candidates=k_candidates, M=M)

        if faiss is None:
            raise ImportError("faiss がインストールされていません")

        # GPU PQ インデックスが未構築の場合は構築
        if not hasattr(self, "_pq_index_gpu") or self._pq_index_gpu is None:
            keys_mm = self._keys_memmap()
            self._build_pq_index_gpu(keys_mm, M=M)

        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        n_total = self.write_count

        if n_total == 0:
            return out, StoreStats(best, arg)

        keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()

        k_search = min(k_candidates, n_total)
        # GPU IndexPQ で候補検索
        distances, indices = self._pq_index_gpu.search(keys_np, k_search)

        keys_mm = self._keys_memmap()
        values_mm = self._values_memmap()

        for i in range(0, b, chunk):
            q = keys[i : i + chunk].to(dtype=torch.float32)
            bp = q.shape[0]
            cand_indices = indices[i : i + chunk]

            chunk_best = torch.full((bp,), float("-inf"))
            chunk_arg = torch.full((bp,), -1, dtype=torch.long)

            # 候補インデックスを集約・ソート
            all_candidates = set()
            for b_idx in range(bp):
                for c in cand_indices[b_idx]:
                    if c >= 0 and c < n_total:
                        all_candidates.add(int(c))

            if not all_candidates:
                best[i : i + chunk] = chunk_best.to(keys.device)
                arg[i : i + chunk] = chunk_arg.to(keys.device)
                continue

            sorted_cand_list = np.sort(np.array(list(all_candidates)))
            idx_to_pos = {int(c): pos for pos, c in enumerate(sorted_cand_list)}

            # バッチ読み込み（CPU 版と同じ）
            cand_keys_batch = torch.from_numpy(
                np.array([keys_mm[c] for c in sorted_cand_list])
            ).to(torch.float32)
            cand_vals_batch = torch.from_numpy(
                np.array([values_mm[c] for c in sorted_cand_list])
            ).to(torch.float64)

            # スコア計算（CPU 版と同じ）
            for b_idx in range(bp):
                candidates = set(cand_indices[b_idx].tolist())
                for cand_idx in candidates:
                    if cand_idx >= 0 and cand_idx < n_total:
                        pos = idx_to_pos[int(cand_idx)]
                        key_val = cand_keys_batch[pos]
                        score = q[b_idx] @ key_val
                        if score > chunk_best[b_idx]:
                            chunk_best[b_idx] = score
                            chunk_arg[b_idx] = cand_idx

            # 出力値の計算（CPU 版と同じ）
            if self.exact:
                for k in range(bp):
                    a = int(chunk_arg[k])
                    if a >= 0:
                        out[i + k] = torch.from_numpy(np.array(values_mm[a])).to(
                            out.dtype
                        )
            else:
                exp_sum = torch.zeros(bp, dtype=torch.float64)
                weighted_val = torch.zeros(bp, self.value_dim, dtype=torch.float64)

                for b_idx in range(bp):
                    candidates = set(cand_indices[b_idx].tolist())
                    for cand_idx in candidates:
                        if cand_idx >= 0 and cand_idx < n_total:
                            pos = idx_to_pos[int(cand_idx)]
                            key_val = cand_keys_batch[pos].to(torch.float64)
                            val = cand_vals_batch[pos]
                            score = q[b_idx].to(torch.float64) @ key_val
                            w = torch.exp(self.beta * (score - chunk_best[b_idx]))
                            exp_sum[b_idx] += w.item()
                            weighted_val[b_idx] += w.item() * val

                for k in range(bp):
                    if exp_sum[k] > 0:
                        v = (weighted_val[k] / exp_sum[k]).to(out.dtype)
                        out[i + k] = v.to(keys.device)

            best[i : i + chunk] = chunk_best.to(keys.device)
            arg[i : i + chunk] = chunk_arg.to(keys.device)

        return out, StoreStats(best, arg)

    def _build_pq_index_gpu(self, keys_np: np.ndarray, M: int = 16, nbits: int = 8) -> None:
        """GPU 最適化版 PQ インデックス構築。

        GPU 利用不可の場合は CPU 版にフォールバック。
        """
        if not _gpu_available:
            # GPU 不可の場合は CPU 版にフォールバック
            return self._build_pq_index(keys_np, M=M, nbits=nbits)

        d = keys_np.shape[1]
        self._pq_index_gpu = faiss.index_factory(d, f"PQ{M}x{nbits}", faiss.METRIC_L2)
        self._pq_index_gpu.train(keys_np)
        self._pq_index_gpu.add(keys_np)

    @torch.no_grad()
    def read_with_hybrid_rerank(
        self,
        keys: torch.Tensor,
        chunk: int = 512,
        k_candidates: int = 707,
        k_refine: int = 150,
        lambda_inner: float = 0.6,
        lambda_density: float = 0.3,
        lambda_template: float = 0.1,
        M: int = 16,
        use_gpu: bool = True,
        return_score_breakdown: bool = False,
    ):
        """ハイブリッド再ランク（段階1+段階2+段階3）による精度向上。

        ステップ40設計：3段階の複合再ランク戦略
        - 段階1：GPU PQ で高速候補検索 k=√N ≈ 707
        - 段階2：内積・密度・テンプレート複合スコア計算
        - 段階3：複合スコアで上位k_refineまで絞り込み

        Args:
            keys: [B, key_dim] クエリキー
            chunk: チャンクサイズ
            k_candidates: 段階1の候補数（√N推奨）
            k_refine: 段階3で評価する上位候補数（100-200推奨）
            lambda_inner: 内積スコアの重み（既定0.6）
            lambda_density: 密度スコアの重み（既定0.3）
            lambda_template: テンプレートスコアの重み（既定0.1）
            M: PQ のサブクォンタイザ数
            use_gpu: GPU 使用フラグ
            return_score_breakdown: True の場合、選択された最良候補における
                内積・密度・テンプレート各成分の寄与度（重み適用前の生スコア
                平均・標準偏差、および重み適用後の複合スコアに対する寄与率）
                を3つ目の戻り値として返す（主張(c)検証用）。既存の
                `_compute_inner_scores` 等はPythonループで候補ごとに
                memmapへ逐次アクセスするため大規模Nでは非現実的に遅く、
                このモードは `read_with_hybrid_rerank` 内で既に計算済みの
                ベクトル化スコアを再利用することで大規模Nでも実用的な
                速度で成分別寄与度を計測できる。

        Returns:
            return_score_breakdown=False: ([B, value_dim], StoreStats)
            return_score_breakdown=True: ([B, value_dim], StoreStats, dict)
        """
        if faiss is None:
            raise ImportError("faiss がインストールされていません")

        # 段階1：GPU/CPU PQ検索で高速候補取得
        if use_gpu and _gpu_available:
            out_stage1, stats_stage1 = self.read_with_pq_search_gpu(
                keys, chunk=chunk, k_candidates=k_candidates, M=M
            )
        else:
            out_stage1, stats_stage1 = self.read_with_pq_search(
                keys, chunk=chunk, k_candidates=k_candidates, M=M
            )

        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        n_total = self.write_count

        if n_total == 0:
            if return_score_breakdown:
                return out, StoreStats(best, arg), {
                    "n_samples": 0,
                    "inner_mean": 0.0, "inner_std": 0.0,
                    "density_mean": 0.0, "density_std": 0.0,
                    "template_mean": 0.0, "template_std": 0.0,
                    "inner_contrib_pct": 0.0,
                    "density_contrib_pct": 0.0,
                    "template_contrib_pct": 0.0,
                }
            return out, StoreStats(best, arg)

        # 段階2+段階3：複合スコアベースの再ランク
        keys_mm = self._keys_memmap()
        values_mm = self._values_memmap()

        # PQ インデックスから候補を取得（段階1の再実行）
        keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
        if use_gpu and _gpu_available:
            distances, indices = self._pq_index_gpu.search(keys_np, k_candidates)
        else:
            if not hasattr(self, "_pq_index") or self._pq_index is None:
                self._build_pq_index(keys_mm, M=M)
            distances, indices = self._pq_index.search(keys_np, k_candidates)

        # 密度スコア用の共有ランダムサンプル（クエリ・候補間で使い回すことで
        # Pythonループでの逐次memmapアクセスを避け、E2E応答時間を短縮する。
        # 元実装は候補ごと・クエリごとに500点を再サンプルしてPythonループで
        # 距離計算しており、k_candidates=32, k_refine=16, N=1000 の規模でも
        # 数秒かかり性能要件（<400ms/500K件相当）を満たさなかった。
        density_sample_size = min(n_total, 500)
        density_sample_idx = np.random.choice(
            n_total, size=density_sample_size, replace=False
        )
        density_sample_keys = torch.from_numpy(
            np.array(keys_mm[density_sample_idx])
        ).to(torch.float32)

        sel_inner_scores = []
        sel_density_scores = []
        sel_template_scores = []

        for i in range(0, b, chunk):
            bp = min(chunk, b - i)
            q = keys[i : i + bp].to(dtype=torch.float32)
            cand_indices = indices[i : i + bp]  # [bp, k_candidates] (numpy)

            valid_mask = (cand_indices >= 0) & (cand_indices < n_total)
            safe_indices = np.where(valid_mask, cand_indices, 0)

            # 候補キーをまとめて1回のmemmapアクセスで取得
            cand_keys = torch.from_numpy(
                np.array(keys_mm[safe_indices.reshape(-1)])
            ).to(torch.float32).view(bp, cand_indices.shape[1], -1)

            # 段階2：複合スコアをベクトル化して計算
            # 内積スコア [bp, k]
            score_inner = torch.einsum("bd,bkd->bk", q, cand_keys)

            # 密度スコア [bp, k]：共有サンプルとの平均距離の逆数
            cand_flat = cand_keys.reshape(-1, cand_keys.shape[-1])
            dist = torch.cdist(cand_flat, density_sample_keys)
            score_density = (1.0 / (1.0 + dist.mean(dim=-1))).view(
                bp, cand_indices.shape[1]
            )

            # テンプレート一致スコア [bp, k]：クエリと候補のコサイン類似度
            score_template = torch.nn.functional.cosine_similarity(
                q.unsqueeze(1).expand(-1, cand_keys.shape[1], -1),
                cand_keys,
                dim=-1,
            )
            score_template = (score_template + 1.0) / 2.0
            score_template = score_template.clamp(min=0.0)

            valid_mask_t = torch.from_numpy(valid_mask)

            # 12.6.89節で判明した欠陥1の修正：内積スコアはキーが未正規化のため
            # 数百オーダーになりうる一方、密度・テンプレートスコアは[0, 1]程度に
            # 収まっており、正規化なしで重み付け合成すると内積単独に退化する
            # （λによる重み付けが事実上機能しない）。そこで3成分を合成前に
            # クエリ（行）ごとの有効候補内でz-score正規化する。min-maxではなく
            # z-scoreを選ぶ理由は、候補集合内の分布の「広がり」に対して相対的な
            # 押し出し度合いを揃えられるため（min-maxは外れ値1点に全体スケールが
            # 引きずられやすい）。行ごと（クエリごと）に正規化するのは、
            # 候補集合の絶対スケール自体がクエリやNに依存して変動するためで、
            # 全体1回の正規化では依然としてクエリ間のスケール差が残ってしまう。
            z_inner = _row_zscore(score_inner, valid_mask_t)
            z_density = _row_zscore(score_density, valid_mask_t)
            z_template = _row_zscore(score_template, valid_mask_t)

            # 複合スコア（正規化後の3成分をλで重み付け合成）
            score_hybrid = (
                lambda_inner * z_inner
                + lambda_density * z_density
                + lambda_template * z_template
            )
            score_hybrid = score_hybrid.masked_fill(~valid_mask_t, float("-inf"))

            # 上位k_refineの中で最高スコアのものを選択
            # （argmaxは全候補中の最良と一致するため、k_refineでの絞り込みは
            #  出力インデックスの選定には影響しない。段階3の意図は将来の
            #  複数候補利用に備えた絞り込みであり、現状のAPIは最良1件のみ返す）
            chunk_best_full, chunk_arg_full = score_hybrid.max(dim=-1)

            if return_score_breakdown:
                sel_idx = chunk_arg_full  # [bp] 各クエリで選択された候補の列インデックス
                row_idx = torch.arange(bp)
                sel_valid = valid_mask_t[row_idx, sel_idx]
                if sel_valid.any():
                    # 寄与度は実際に合成に使われた正規化後（z-score）のスコアで
                    # 集計する。生スコアのままでは内積成分のスケールが桁違いに
                    # 大きく、寄与度計算自体が旧来の欠陥を再現してしまうため。
                    sel_inner_scores.append(
                        z_inner[row_idx, sel_idx][sel_valid].numpy()
                    )
                    sel_density_scores.append(
                        z_density[row_idx, sel_idx][sel_valid].numpy()
                    )
                    sel_template_scores.append(
                        z_template[row_idx, sel_idx][sel_valid].numpy()
                    )

            chunk_arg = safe_indices[np.arange(bp), chunk_arg_full.numpy()]
            chunk_arg = torch.from_numpy(chunk_arg).to(torch.long)
            has_valid = valid_mask.any(axis=-1)
            chunk_arg = torch.where(
                torch.from_numpy(has_valid), chunk_arg, torch.full_like(chunk_arg, -1)
            )
            chunk_best = torch.where(
                torch.from_numpy(has_valid),
                chunk_best_full,
                torch.full_like(chunk_best_full, float("-inf")),
            )

            # 出力値の計算
            if self.exact:
                for k in range(bp):
                    a = int(chunk_arg[k])
                    if a >= 0:
                        out[i + k] = torch.from_numpy(np.array(values_mm[a])).to(
                            out.dtype
                        )
            else:
                for k in range(bp):
                    a = int(chunk_arg[k])
                    if a >= 0:
                        out[i + k] = torch.from_numpy(np.array(values_mm[a])).to(
                            out.dtype
                        )

            best[i : i + bp] = chunk_best.to(keys.device)
            arg[i : i + bp] = chunk_arg.to(keys.device)

        if return_score_breakdown:
            if sel_inner_scores:
                inner_arr = np.concatenate(sel_inner_scores)
                density_arr = np.concatenate(sel_density_scores)
                template_arr = np.concatenate(sel_template_scores)

                inner_mean, inner_std = float(inner_arr.mean()), float(inner_arr.std())
                density_mean, density_std = float(density_arr.mean()), float(density_arr.std())
                template_mean, template_std = float(template_arr.mean()), float(template_arr.std())

                # 重み適用後の各成分が複合スコアに占める寄与率。z-score正規化後は
                # 各サンプルの符号が±に散らばり得るため、mean()を先に取ると
                # 打ち消し合って過小評価される。よってサンプルごとの絶対値の
                # 平均（mean(|lambda*z|)）を寄与度の代表値として使う。
                w_inner = float(np.abs(lambda_inner * inner_arr).mean())
                w_density = float(np.abs(lambda_density * density_arr).mean())
                w_template = float(np.abs(lambda_template * template_arr).mean())
                w_total = w_inner + w_density + w_template
                if w_total > 0:
                    inner_pct = 100.0 * w_inner / w_total
                    density_pct = 100.0 * w_density / w_total
                    template_pct = 100.0 * w_template / w_total
                else:
                    inner_pct = density_pct = template_pct = 0.0

                breakdown = {
                    "n_samples": int(inner_arr.shape[0]),
                    "inner_mean": inner_mean, "inner_std": inner_std,
                    "density_mean": density_mean, "density_std": density_std,
                    "template_mean": template_mean, "template_std": template_std,
                    "inner_contrib_pct": inner_pct,
                    "density_contrib_pct": density_pct,
                    "template_contrib_pct": template_pct,
                }
            else:
                breakdown = {
                    "n_samples": 0,
                    "inner_mean": 0.0, "inner_std": 0.0,
                    "density_mean": 0.0, "density_std": 0.0,
                    "template_mean": 0.0, "template_std": 0.0,
                    "inner_contrib_pct": 0.0,
                    "density_contrib_pct": 0.0,
                    "template_contrib_pct": 0.0,
                }
            return out, StoreStats(best, arg), breakdown

        return out, StoreStats(best, arg)

    def _local_density_at(self, candidate_idx: int, k_neighbor: int = 20) -> float:
        """候補インデックス周辺の局所密度スコアを計算（段階2補助）。

        周辺k個の近傍との距離平均の逆数を密度スコアとする。
        """
        keys_mm = self._keys_memmap()
        cand_key = torch.from_numpy(np.array(keys_mm[candidate_idx])).to(
            torch.float32
        )

        # 全キーとの距離計算（簡易版：ランダムサンプル）
        # ベクトル化：Pythonループで500回距離計算すると極めて遅いため
        # （ステップ40テスト時にE2E応答が5秒超となり性能要件を満たさなかった）、
        # サンプルをまとめてmemmapから読み出しtorchでバッチ計算する。
        n_total = min(self.write_count, 500)
        sample_indices = np.random.choice(self.write_count, size=n_total, replace=False)
        sample_indices = sample_indices[sample_indices != candidate_idx]

        if sample_indices.size == 0:
            return 1.0

        sample_keys = torch.from_numpy(np.array(keys_mm[sample_indices])).to(
            torch.float32
        )
        distances = torch.norm(sample_keys - cand_key.unsqueeze(0), dim=1)

        # 平均距離の逆数を密度スコアとする（近いほど高スコア）
        mean_distance = distances.mean().item()
        density_score = 1.0 / (1.0 + mean_distance)  # sigmoid形
        return density_score

    def _template_match_score(
        self, candidate_idx: int, query_key: torch.Tensor
    ) -> float:
        """テンプレート一致度スコア（段階2補助）。

        候補キーと平均キーのコサイン類似度を用いる（簡易版）。
        """
        keys_mm = self._keys_memmap()
        cand_key = torch.from_numpy(np.array(keys_mm[candidate_idx])).to(
            torch.float32
        )

        # コサイン類似度
        cos_sim = torch.nn.functional.cosine_similarity(
            query_key.unsqueeze(0), cand_key.unsqueeze(0)
        ).item()

        # [0, 1] の範囲にスケール
        return max(0.0, (cos_sim + 1.0) / 2.0)

    def _compute_inner_scores(
        self,
        keys: torch.Tensor,
        k_candidates: int,
        k_refine: int,
        use_gpu: bool = True,
    ) -> np.ndarray:
        """複合スコア内の内積成分を抽出（主張(c)検証用）"""
        scores = []
        keys_mm = self._keys_memmap()

        for i in range(keys.shape[0]):
            q = keys[i].to(torch.float32)
            candidate_scores = []
            for j in range(min(k_candidates, self.write_count)):
                key_val = torch.from_numpy(np.array(keys_mm[j])).to(torch.float32)
                score = (q @ key_val).item()
                candidate_scores.append(score)

            scores.extend(candidate_scores[:k_refine])

        return np.array(scores)

    def _compute_density_scores(
        self, keys: torch.Tensor, k_candidates: int, k_refine: int
    ) -> np.ndarray:
        """複合スコア内の密度成分を抽出（主張(c)検証用）"""
        scores = []
        for i in range(min(k_candidates, self.write_count)):
            score = self._local_density_at(i, k_neighbor=20)
            scores.append(score)

        return np.array(scores[:k_refine] * keys.shape[0])

    def _compute_template_scores(
        self, keys: torch.Tensor, k_candidates: int, k_refine: int
    ) -> np.ndarray:
        """複合スコア内のテンプレート成分を抽出（主張(c)検証用）"""
        scores = []
        for i in range(min(k_candidates, self.write_count)):
            score = self._template_match_score(i, keys[0].to(torch.float32))
            scores.append(score)

        return np.array(scores[:k_refine] * keys.shape[0])


class HippocampalMemory(nn.Module):
    """分離層と連想ストアを束ね、皮質の残差ストリームに読み出しを注入する。

    ゲート ``g = σ(gate_slope * (max_score - gate_bias))`` は、最良一致の類似度が
    低い（＝知らない事実である）ときに注入を遮断する。6.1節がCA1に帰している
    新規性検出に対応する。ステップ1では学習せず定数で置く（12.6.4節）。
    """

    def __init__(
        self,
        d_model: int,
        value_dim: int,
        n_units: int = 2048,
        k: int = 40,
        mode: str = "sdr",
        beta: float = 50.0,
        exact: bool = False,
        gain: float = 4.0,
        gate_slope: float = 20.0,
        gate_bias: float = 0.3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.separator = PatternSeparator(d_model, n_units, k, mode, seed)
        self.store = AssociativeStore(self.separator.out_dim, value_dim, beta, exact)
        self.gain = gain
        self.gate_slope = gate_slope
        self.gate_bias = gate_bias

    @torch.no_grad()
    def write(self, x: torch.Tensor, values: torch.Tensor) -> None:
        """皮質表現 x [N, d_model] をキーに変換して値を書き込む。"""
        self.store.write(self.separator(x), values)

    @torch.no_grad()
    def read(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """皮質表現 x [B, d_model] から注入ベクトル g*m と ゲート値 g を返す。"""
        m, stats = self.store.read(self.separator(x))
        g = torch.sigmoid(self.gate_slope * (stats.max_score - self.gate_bias))
        return self.gain * g[:, None] * m, g

    def clear(self) -> None:
        self.store.clear()


# --- 分離層の学習（12.6.6節・ロードマップ段階2） -----------------------------
#
# 【重要な但し書き】側方抑制（k-WTA）付きのSTDPは、レート符号化された入力に対して
# 「勝者の重みを入力ベクトルの方へ動かす」更新に帰着し、オンライン競合学習とほぼ
# 等価になることが知られている（Diehl & Cook 2015 系の定式化）。したがって
# `fit_stdp` が `fit_hebbian` に勝てなければ「STDPである必然性」は何も示せない。
# 12.9節で整数スパイク方式が階段関数に縮退したのと同じ構図なので、両方を実装して
# 明示的に比較する。


@torch.no_grad()
def fit_hebbian(sep: PatternSeparator, x: torch.Tensor, epochs: int = 5,
                eta: float = 0.05, batch_size: int = 256,
                generator: torch.Generator | None = None) -> dict:
    """競合ヘブ学習。勝者の重みを入力へ寄せ、行ごとにL2正規化する（Oja型の安定化）。

    STDP + 側方抑制がレート符号化入力に対して縮退する先そのものであり、
    「STDPだから効いた」と「クラスタリングしたから効いた」を分離する対照群。
    """
    if sep.mode == "identity":
        raise ValueError("identity には学習する重みがない")
    n = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, generator=generator)
        lr = eta * (1.0 - ep / max(1, epochs))  # 線形に冷ます
        for i in range(0, n, batch_size):
            xb = _l2_normalize(x[perm[i : i + batch_size]])
            h = xb @ sep.weight.T
            idx = h.topk(sep.k, dim=-1).indices  # [b, k] 勝者
            # 勝者 i について W[i] += lr * (x - W[i])。同一勝者が複数回選ばれる
            # 場合は加算される（発火頻度の高いユニットほど強く引かれる）。
            flat = idx.reshape(-1)
            src = xb[:, None, :].expand(-1, sep.k, -1).reshape(-1, xb.shape[-1])
            delta = torch.zeros_like(sep.weight)
            delta.index_add_(0, flat, src)
            cnt = torch.zeros(sep.weight.shape[0], device=xb.device)
            cnt.index_add_(0, flat, torch.ones_like(flat, dtype=xb.dtype))
            hit = cnt > 0
            delta[hit] = delta[hit] / cnt[hit, None] - sep.weight[hit]
            sep.weight[hit] += lr * delta[hit]
            sep.weight.copy_(_l2_normalize(sep.weight))
    return {"rule": "hebbian", "epochs": epochs, "eta": eta, "n_samples": n}


@torch.no_grad()
def fit_stdp(sep: PatternSeparator, spikes: torch.Tensor, epochs: int = 5,
             a_plus: float = 0.01, a_minus: float = 0.008, tau: float = 0.9,
             batch_size: int = 256, generator: torch.Generator | None = None,
             callback: "Callable[[int, PatternSeparator], None] | None" = None,
             raw_update_callback: "Callable[[int, float], None] | None" = None,
             weight_decay: float = 0.0) -> dict:
    """ペア型STDP（8.1節の指数窓）。時間軸はトークン位置そのもの（A案）。

    前シナプス入力は皮質バックボーンの二値スパイク列 `spikes` [N, T, d]、
    後シナプスは本層の k-WTA 出力である。指数トレースを用いた標準形::

        trace_pre  <- tau * trace_pre  + s_pre
        trace_post <- tau * trace_post + s_post
        dW += a_plus * s_post^T @ trace_pre - a_minus * trace_post^T @ s_pre

    第1項が「前が先に発火 → 増強」、第2項が「後が先 → 抑圧」に対応する。

    `raw_update_callback`（epoch, raw_norm）は`callback`とは独立の軽量な追加引数で、
    485行目付近の正規化前の生の更新量`scale * dw / b`のフロベニウスノルムを
    epoch内バッチ合計で累積してepoch末に通知する（STDPアルゴリズム自体は変更しない）。

    `weight_decay`（既定0.0、ステップ28で追加）は`callback`・`raw_update_callback`と
    同じ非侵襲的な拡張点で、正規化前の生の更新量に対して明示的なL2減衰
    `raw_update -= weight_decay * sep.weight`を適用する（正規化より前）。
    STDPのdw計算式自体（490行目）は変更しない。既定値0.0では従来と完全に同一の挙動。
    """
    if sep.mode == "identity":
        raise ValueError("identity には学習する重みがない")
    n, t_len, d = spikes.shape
    for ep in range(epochs):
        perm = torch.randperm(n, generator=generator)
        scale = 1.0 - ep / max(1, epochs)
        raw_update_norm_sum = 0.0
        for i in range(0, n, batch_size):
            sb = spikes[perm[i : i + batch_size]]  # [b, T, d]
            b = sb.shape[0]
            tr_pre = torch.zeros(b, d)
            tr_post = torch.zeros(b, sep.weight.shape[0])
            dw = torch.zeros_like(sep.weight)
            for t in range(t_len):
                s_pre = sb[:, t]                      # [b, d]
                h = s_pre @ sep.weight.T              # [b, M]
                idx = h.topk(sep.k, dim=-1).indices
                s_post = torch.zeros_like(h).scatter_(-1, idx, 1.0)
                tr_pre = tau * tr_pre + s_pre
                tr_post = tau * tr_post + s_post
                dw += a_plus * (s_post.T @ tr_pre) - a_minus * (tr_post.T @ s_pre)
            raw_update = scale * dw / b
            if weight_decay != 0.0:
                raw_update = raw_update - weight_decay * sep.weight
            if raw_update_callback is not None:
                raw_update_norm_sum += float(raw_update.norm())
            sep.weight += raw_update
            sep.weight.copy_(_l2_normalize(sep.weight))
        if callback is not None:
            callback(ep, sep)
        if raw_update_callback is not None:
            raw_update_callback(ep, raw_update_norm_sum)
    return {"rule": "stdp", "epochs": epochs, "a_plus": a_plus,
            "a_minus": a_minus, "tau": tau, "n_samples": n}


@torch.no_grad()
def separation_diagnostics(sep: PatternSeparator, x: torch.Tensor,
                           x_noisy: torch.Tensor) -> dict:
    """12.6.5節が特定した敗因（k-WTAの勝者反転）を直接測る。

    - `margin`  : 第k位と第k+1位の活性の差（大きいほど反転しにくい）
    - `winner_retention`: 手がかりを劣化させたとき、勝者集合が保たれる割合
    """
    if sep.mode != "sdr":
        # k-WTA を持たない条件では「勝者」が定義できない（identity には重みがなく、
        # dense は全ユニットが非ゼロ）。無意味な値を返さず空にする。
        return {}
    h = _l2_normalize(x) @ sep.weight.T
    top = h.topk(sep.k + 1, dim=-1).values
    margin = (top[:, sep.k - 1] - top[:, sep.k]).mean()
    a = sep(x) != 0
    b = sep(x_noisy) != 0
    retention = (a & b).sum(dim=-1).float() / sep.k
    return {"margin": float(margin), "winner_retention": float(retention.mean())}


@torch.no_grad()
def max_pairwise_cosine(weight: torch.Tensor) -> float:
    """重み行列の行ベクトル間の最大ペアワイズコサイン類似度（対角除く）。

    ステップ26〜28（12.6.60〜65節）の`run_stdp_stability.degeneracy_stats`が
    STDP縮退（少数ユニットへの重みベクトルの集中）の指標として使う
    `max_pairwise_cos`の計算本体。ステップ29（12.6.66節）で`tests/spiking_audit.py`の
    `warn_if_degenerate`からも同じロジックを再利用するため、計算の重複を避ける
    共通ヘルパーとしてここに切り出す。
    """
    w = weight.detach()
    m = w.shape[0]
    wn = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sims = wn @ wn.T
    off_diag = sims - torch.eye(m) * 2.0  # 対角を確実に除外
    return float(off_diag.max())
