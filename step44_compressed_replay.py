#!/usr/bin/env python3
"""ステップ44.12: 隠れ状態・圧縮表現を保存する海馬のメモリと、生の断片の保存（44.10 の R）の比較。

44.9/44.10 と同じ課題（Europarl 4 言語を順に学ぶ、step44 のモデル）で、次の条件を同じコードで走らせる。
- V   … vanilla・naive（44.9 の V と同じ計算）
- R   … フェーズ A ＋生の断片（単語 ID 列）の保存・再生（44.10 の R と同じ計算。M・r は可変で下限の対照も作る）
- S   … R と同じ保存断片を、再生のたびに断片内で語順を並べ替えて再生（頻度だけを残し文脈を壊す対照）
- C   … 計算量の対照。再生の代わりに現在のタスクの学習断片を r 本足す（1 更新の断片数・更新回数は R と同じ）
- H   … 隠れ状態のメモリ。タスクの終わりに保存断片の第 ℓ 層の出力を積直量子化（PQ）で圧縮して保存し、
         再生では復元した隠れ状態を第 ℓ+1 層以降（ℓ = L なら lm_head と海馬ヘッドだけ）に通して学ぶ。
         損失の目標（次の語）のため単語 ID も保存する。下位の層は再生では学ばない（古くなる）。
- HR  … H ＋再固定化（44.13）。フェーズ境界（新しいタスクの学習を始める前）ごとに、保存済みの隠れ状態を
         現在のネットワークで作り直して保存し直す（目標 ID 列を入力に現在の下位層を通し、PQ 符号帳も再学習）。
         保存層は設計で第 2 層に固定。作り直しは重みを変えない順伝播のみで、学習の計算量は H と同一。
- G   … 生成的リプレイ（44.13）。保存は断片の先頭 PREFIX_LEN トークンの prefix のみ（索引）。
         フェーズ開始時に現在のモデルが各 prefix の続きを温度 1.0 のサンプリングで自己回帰生成して
         そのフェーズの再生集合を作り、フェーズの間は R と同じ形で全層に再生する。
評価は 44.9 と同じ backbone lm_head の CE・正確度に加え、出力の周辺分布（評価位置の予測分布の平均）の CE を測り、
CE = CE_marg − G（G: 文脈の利得）に分けて、忘却を「出力分布（語の頻度）」と「文脈表現」に切り分ける。

使い方: python step44_compressed_replay.py --mode H --layer 2 --seeds 100 --out-dir results/step44_compressed_replay/dev
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import step44_continual_forgetting as cf  # noqa: E402
from colab_step44_full_scale_experiment import BrainInspiredLLMFullScale, FullScaleTrainer, WordChunkDataset  # noqa: E402
from neurocortex.run_env import env_info  # noqa: E402
from step44_continual_replay import EpisodicMemory, concat_replay  # noqa: E402

MODES = ("V", "R", "S", "C", "H", "HR", "G")
ID_BYTES = 2  # 語彙 9,372 < 65,536 なので単語 ID は uint16
PREFIX_LEN = 16  # G の保存する索引（断片の先頭トークン数。44.13 設計で固定）


# ---------------------------------------------------------------- 層の分割（model.forward と同じ計算）

def _masks(seq_len: int, batch: int, device):
    causal = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
    pad = ~torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    return causal, pad


def forward_lower(model, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """埋め込み＋第 1..layer 層の出力（model.forward の途中までと同じ計算）。"""
    B, T = input_ids.shape
    h = model.embedding(input_ids) + model.pos_embedding(
        torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1))
    causal, pad = _masks(T, B, input_ids.device)
    for lyr in model.transformer_layers[:layer]:
        h = lyr(h, src_mask=causal, is_causal=True, src_key_padding_mask=pad)
    return h


def forward_upper(model, hidden: torch.Tensor, layer: int, labels: torch.Tensor) -> dict:
    """第 layer+1 層以降と lm_head・補助ヘッドの損失（model.forward の残りと同じ計算。フェーズ A の損失）。"""
    B, T, _ = hidden.shape
    causal, pad = _masks(T, B, hidden.device)
    h = hidden
    for lyr in model.transformer_layers[layer:]:
        h = lyr(h, src_mask=causal, is_causal=True, src_key_padding_mask=pad)
    logits = model.lm_head(h)
    loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
    if model.hippocampus is not None:
        loss = loss + model.config.hippocampus_loss_weight * model.hippocampus(h, labels)["loss"]
    return {"logits": logits, "loss": loss}


# ---------------------------------------------------------------- 隠れ状態のメモリ（PQ 圧縮）

class HiddenMemory:
    """海馬の圧縮表現のメモリ。タスクごとに M 断片の第 ℓ 層の隠れ状態を PQ で符号化して保存する。

    符号帳はタスクごとにそのタスクの保存ベクトルで学ぶ（faiss.ProductQuantizer、m 個の部分空間 × 2^nbits 語）。
    保存の大きさ = 符号（m×nbits/8 バイト/位置）＋目標の単語 ID（2 バイト/位置）＋符号帳（fp16）。
    再生用には復元した隠れ状態を持つ（計算の都合。大きさの計算には含めない）。
    """

    def __init__(self, per_task: int, layer: int, pq_m: int = 16, pq_nbits: int = 8):
        self.per_task, self.layer, self.pq_m, self.pq_nbits = per_task, layer, pq_m, pq_nbits
        self.hidden: list[torch.Tensor] = []  # 復元した (M, T, d)
        self.labels: list[torch.Tensor] = []  # (M, T) の単語 ID
        self.recon_rel_err: list[float] = []
        self.nbytes: list[int] = []

    def __len__(self):
        return sum(len(h) for h in self.hidden)

    @torch.no_grad()
    def store(self, model, dataset, generator: torch.Generator) -> torch.Tensor:
        import faiss
        n = len(dataset)
        idx = torch.randperm(n, generator=generator)[:min(self.per_task, n)]  # R と同じ断片を選ぶ
        ids = dataset.chunks[idx].clone()
        was = model.training
        model.eval()
        h = torch.cat([forward_lower(model, ids[k:k + 32], self.layer) for k in range(0, len(ids), 32)])
        model.train(was)
        M, T, d = h.shape
        x = np.ascontiguousarray(h.reshape(-1, d).numpy(), dtype=np.float32)
        pq = faiss.ProductQuantizer(d, self.pq_m, self.pq_nbits)
        pq.train(x)
        codes = pq.compute_codes(x)
        xr = pq.decode(codes)
        self.recon_rel_err.append(float(((xr - x) ** 2).sum() / (x ** 2).sum()))
        self.hidden.append(torch.from_numpy(xr.reshape(M, T, d).copy()))
        self.labels.append(ids)
        codebook = d * (2 ** self.pq_nbits) * 2  # fp16 の符号帳
        self.nbytes.append(int(codes.nbytes + ids.numel() * ID_BYTES + codebook))
        return idx

    def sample(self, r: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        allh, alll = torch.cat(self.hidden), torch.cat(self.labels)
        k = torch.randint(0, len(allh), (r,), generator=generator)
        return allh[k], alll[k]

    @torch.no_grad()
    def drift(self, model) -> list[float]:
        """保存した（復元）隠れ状態と、現在のネットワークで同じ断片から作る隠れ状態の相対二乗差（タスクごと）。"""
        was = model.training
        model.eval()
        out = []
        for h, ids in zip(self.hidden, self.labels):
            cur = torch.cat([forward_lower(model, ids[k:k + 32], self.layer) for k in range(0, len(ids), 32)])
            out.append(float(((cur - h) ** 2).sum() / (cur ** 2).sum()))
        model.train(was)
        return out

    @torch.no_grad()
    def refresh(self, model) -> None:
        """保存済みの隠れ状態を現在のネットワークで作り直して保存し直す（再固定化、44.13）。

        入力は保存済みの目標 ID 列（追加の情報は保存しない）。PQ 符号帳はタスクごとに再学習し、
        復元した隠れ状態と復元誤差を置き換える。符号帳の大きさは変わらないので保存の大きさは不変。
        重みは変えない（no_grad・eval の決定的な順伝播のみ）。
        """
        import faiss
        was = model.training
        model.eval()
        for k, ids in enumerate(self.labels):
            h = torch.cat([forward_lower(model, ids[j:j + 32], self.layer) for j in range(0, len(ids), 32)])
            M, T, d = h.shape
            x = np.ascontiguousarray(h.reshape(-1, d).numpy(), dtype=np.float32)
            pq = faiss.ProductQuantizer(d, self.pq_m, self.pq_nbits)
            pq.train(x)
            codes = pq.compute_codes(x)
            xr = pq.decode(codes)
            self.recon_rel_err[k] = float(((xr - x) ** 2).sum() / (x ** 2).sum())
            self.hidden[k] = torch.from_numpy(xr.reshape(M, T, d).copy())
        model.train(was)


# ---------------------------------------------------------------- 生成的リプレイのメモリ（44.13）

@torch.no_grad()
def generate_continuations(model, prefixes: torch.Tensor, seq_len: int,
                           generator: torch.Generator) -> torch.Tensor:
    """各 prefix の続きを現在のモデルの出力分布から温度 1.0 のサンプリングで自己回帰生成し、(M, seq_len) を返す。

    生成器は呼び出し側が用意した独立の torch.Generator。モデルの重み・モードは変えない。
    """
    seq = prefixes.clone()
    while seq.size(1) < seq_len:
        logits = model(input_ids=seq, labels=None, attention_mask=torch.ones_like(seq))["logits"][:, -1]
        probs = torch.softmax(logits.double() / 1.0, dim=-1).float()
        nxt = torch.multinomial(probs, 1, generator=generator)
        seq = torch.cat([seq, nxt], dim=1)
    return seq


class GenerativeMemory:
    """生成的リプレイのメモリ（44.13）。断片の先頭 prefix_len トークンだけを保存する（索引）。

    フェーズ開始時に generate で現在のモデルに各 prefix の続きを生成させ、そのフェーズの再生集合を作る。
    保存の大きさ = 保存本数 × prefix_len × ID_BYTES（目標の ID 列は生成した断片自体が持つので別に数えない）。
    """

    def __init__(self, per_task: int, prefix_len: int = PREFIX_LEN):
        self.per_task, self.prefix_len = per_task, prefix_len
        self.prefixes: list[torch.Tensor] = []  # 各タスクの (M, prefix_len) の単語 ID
        self.current: torch.Tensor | None = None  # 現在のフェーズの再生集合（過去タスク分を連結した (総数, T)）
        self.nbytes: list[int] = []

    def __len__(self):
        return sum(len(p) for p in self.prefixes)

    def store(self, dataset, generator: torch.Generator) -> torch.Tensor:
        """R と同じ断片を選び（保存の生成器）、先頭 prefix_len トークンだけを保存する。保存した断片の番号を返す。"""
        n = len(dataset)
        idx = torch.randperm(n, generator=generator)[:min(self.per_task, n)]
        ids = dataset.chunks[idx].clone()
        self.prefixes.append(ids[:, :self.prefix_len].contiguous())
        self.nbytes.append(int(len(ids) * self.prefix_len * ID_BYTES))
        return idx

    @torch.no_grad()
    def generate(self, model, seq_len: int, generator: torch.Generator) -> torch.Tensor:
        """保存済みの全タスクの prefix から、現在のモデルで再生集合を作り直す（フェーズ開始時）。"""
        was = model.training
        model.eval()
        outs = [generate_continuations(model, pre, seq_len, generator) for pre in self.prefixes]
        model.train(was)
        self.current = torch.cat(outs) if outs else None
        return self.current

    def sample(self, r: int, generator: torch.Generator) -> torch.Tensor:
        """現在の再生集合から r 本を一様に復元抽出する（R の sample と同じ流儀）。"""
        k = torch.randint(0, len(self.current), (r,), generator=generator)
        return self.current[k]


def generated_quality(model, generated: torch.Tensor, train_chunks: torch.Tensor, vocab_size: int) -> dict:
    """生成した断片の質（記述的な記録、44.13）。

    - unigram_ce: そのタスクの学習語の 1-gram（加算 0.5）で測った生成文の CE（実文の水準は data の unigram_ce）。
    - freq_l1_vs_train: 学習断片の語頻度分布と生成文の語頻度分布の L1 距離（頻度の再現のずれ）。
    - gain: 生成文を評価断片として読んだときのモデルの文脈の利得（eval_task_decomp と同じ分解）。
    """
    ce1 = unigram_ce(train_chunks, generated, vocab_size)
    a = torch.bincount(train_chunks.reshape(-1), minlength=vocab_size).double()
    b = torch.bincount(generated.reshape(-1), minlength=vocab_size).double()
    l1 = float((a / a.sum() - b / b.sum()).abs().sum())
    was = model.training
    gain = eval_task_decomp(model, WordChunkDataset(generated.reshape(-1).clone(), generated.size(1)))["gain"]
    model.train(was)
    return {"unigram_ce": ce1, "freq_l1_vs_train": l1, "gain": gain}


def shuffle_within(chunks: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """断片ごとに語順を並べ替える（語の頻度は保ち、系列の文脈を壊す）。"""
    return torch.stack([c[torch.randperm(len(c), generator=generator)] for c in chunks])


def raw_nbytes(chunks: torch.Tensor) -> int:
    return int(chunks.numel() * ID_BYTES)


def bag_nbytes(chunks: torch.Tensor) -> int:
    """語順を捨てた保存（語の種類ごとの (ID, 個数)）の大きさ。"""
    return int(len(torch.unique(chunks)) * (ID_BYTES + 2))


# ---------------------------------------------------------------- 評価（CE の分解）

@torch.no_grad()
def eval_task_decomp(model, dataset, batch_size: int = 64) -> dict:
    """cf.eval_task と同じ CE・正確度（同じ計算順）に加え、出力の周辺分布の CE（CE_marg）と文脈の利得 G を返す。

    p̄ = 評価の全位置での予測分布の平均、CE_marg = −平均 log p̄(y)、G = CE_marg − CE。
    CE_marg は出力分布（語の頻度）の合い具合、G は文脈を使って頻度の予測より良くした分。
    """
    model.eval()
    ce_sum, correct, n = 0.0, 0, 0
    pbar, tgts = None, []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x = batch["input_ids"]
        logits = model(input_ids=x, labels=None, attention_mask=batch["attention_mask"])["logits"][:, :-1]
        tgt = x[:, 1:]
        ce_sum += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                    reduction="sum").item()
        correct += int((logits.argmax(-1) == tgt).sum().item())
        n += tgt.numel()
        p = torch.softmax(logits.double(), -1).sum((0, 1))
        pbar = p if pbar is None else pbar + p
        tgts.append(tgt.reshape(-1))
    pbar = pbar / n
    ce_marg = float(-torch.log(pbar[torch.cat(tgts)]).mean())
    ce = ce_sum / n
    return {"ce": ce, "acc": correct / n, "ce_marg": ce_marg, "gain": ce_marg - ce}


def unigram_ce(train_ids: torch.Tensor, eval_ids: torch.Tensor, vocab_size: int, alpha: float = 0.5) -> float:
    """学習語の 1-gram（加算平滑化）の評価 CE（文脈を使わない頻度だけのモデルの水準）。"""
    cnt = torch.bincount(train_ids.reshape(-1), minlength=vocab_size).double() + alpha
    p = cnt / cnt.sum()
    return float(-torch.log(p[eval_ids[:, 1:].reshape(-1)]).mean())


def decomp_forgetting(rows: list[list[dict]]) -> dict:
    """最終行と学習直後（対角）の差を、出力分布の項と文脈の項に分ける（過去タスクの平均）。

    dCE = dCE_marg − dG。dCE_marg > 0 は頻度の合い具合の悪化、dG < 0 は文脈の利得の喪失。
    """
    T = len(rows)
    past = range(T - 1)
    d_marg = [rows[-1][j]["ce_marg"] - rows[j][j]["ce_marg"] for j in past]
    d_gain = [rows[-1][j]["gain"] - rows[j][j]["gain"] for j in past]
    return {"dCE_marg": float(np.mean(d_marg)), "dG": float(np.mean(d_gain)),
            "G_last_past": float(np.mean([rows[-1][j]["gain"] for j in past])),
            "G_diag": float(np.mean([rows[j][j]["gain"] for j in range(T)])),
            "CE_marg_last_past": float(np.mean([rows[-1][j]["ce_marg"] for j in past]))}


# ---------------------------------------------------------------- 学習

def latent_step(trainer, model, batch: dict, h_rep: torch.Tensor, y_rep: torch.Tensor, layer: int) -> float:
    """新しいタスクのバッチ（全層）と復元した隠れ状態の再生（第 layer+1 層以降）で 1 回更新する。

    損失は断片数で重みづけた平均（R の連結バッチの全位置平均と同じ重み）。勾配クリップ・最適化は train_step と同じ。
    """
    model.train()
    x = batch["input_ids"]
    l_new = model(input_ids=x, labels=batch["labels"], attention_mask=batch["attention_mask"])["loss"]
    l_rep = forward_upper(model, h_rep, layer, y_rep)["loss"]
    n_new, n_rep = len(x), len(h_rep)
    loss = (n_new * l_new + n_rep * l_rep) / (n_new + n_rep)
    loss.backward()
    trainer._clip_grads()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    trainer.step_count += 1
    return loss.item()


def run(mode: str, seed: int, data: dict, args, log=print, ckpt: str | None = None) -> dict:
    """1 つの (条件, seed) で K タスクを順に学ぶ。V 以外はフェーズ A のモデル。初期化・データ順は 44.9/44.10 と同じ。

    ckpt を渡すと各タスクの評価の後に状態（モデル・オプティマイザ・メモリ・乱数・途中の結果）を保存し、
    次回はそこから再開する（中断しない場合とビット一致。コンテナの再起動対策）。
    """
    assert mode in MODES
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = "V" if mode == "V" else "A"
    cfg = cf.make_config(base, args)
    model = BrainInspiredLLMFullScale(cfg, vocab_size=data["vocab_size"], hidden_size=cfg.hidden_size,
                                      num_layers=cfg.num_layers, nhead=cfg.nhead)
    trainer = FullScaleTrainer(model, cfg)
    layer = layer_of(args, mode)
    memory = (HiddenMemory(args.memory_per_task, layer, args.pq_m, args.pq_nbits) if mode in ("H", "HR")
              else GenerativeMemory(args.memory_per_task, getattr(args, "prefix_len", PREFIX_LEN)) if mode == "G"
              else EpisodicMemory(args.memory_per_task) if mode in "RS" else None)
    order = cf.task_order(data["langs"], seed)
    T = len(order)
    rows = [[None] * T for _ in range(T)]
    losses, n_replayed, nbytes, drift, t0 = [], [], [], [], time.time()
    refresh_log, gen_log = [], []  # HR の作り直し前後の古さ、G のフェーズごとの生成文の質（44.13）
    start = 0
    if ckpt and Path(ckpt).exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        trainer.optimizer.load_state_dict(st["optimizer"])
        trainer.step_count = st["step_count"]
        if memory is not None:
            memory.__dict__.update(st["memory"])
        rows, losses, n_replayed, nbytes, drift = st["rows"], st["losses"], st["n_replayed"], st["nbytes"], st["drift"]
        refresh_log, gen_log = st.get("refresh_log", []), st.get("gen_log", [])
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["np_rng"])
        start, t0 = st["next_task"], time.time() - st["elapsed"]
        log(f"[{mode} seed{seed}] resumed from task{start}")
    for i, lang in enumerate(order):
        if i < start:
            continue
        if mode == "HR" and i > 0:
            # フェーズ境界の再固定化: 保存済みの過去タスクすべての隠れ状態を作り直す（重みは変えない）。
            d_before = memory.drift(model)
            memory.refresh(model)
            refresh_log.append({"task": i, "n_chunks": int(len(memory)), "before": d_before,
                                "after": memory.drift(model)})
        if mode == "G" and i > 0:
            # フェーズ開始時に、現在のモデルで過去タスクの再生集合を生成し直す（生成専用の生成器）。
            g_gen = torch.Generator().manual_seed(seed * 1000 + 900 + i)
            cur = memory.generate(model, args.max_seq_length, g_gen)
            per, off = {}, 0
            for j, lj in enumerate(order[:i]):
                seg = cur[off:off + len(memory.prefixes[j])]
                off += len(memory.prefixes[j])
                per[lj] = generated_quality(model, seg, data["tasks"][lj]["train"].chunks, data["vocab_size"])
            gen_log.append({"task": i, "per_task": per})
        trainer.optimizer.zero_grad()
        task_losses, nrep = [], 0
        g_rep = torch.Generator().manual_seed(seed * 1000 + 700 + i)
        replay_on = args.replay_size > 0 and i > 0 and mode != "V"
        for epoch in range(args.epochs):
            g = torch.Generator().manual_seed(seed * 1000 + i * 10 + epoch)  # 44.9 と同じ並び
            for batch in DataLoader(data["tasks"][lang]["train"], batch_size=cfg.batch_size, shuffle=True,
                                    generator=g):
                if replay_on and mode in ("H", "HR"):
                    h_rep, y_rep = memory.sample(args.replay_size, g_rep)
                    task_losses.append(latent_step(trainer, model, batch, h_rep, y_rep, layer))
                else:
                    if replay_on and mode in "RS":
                        rep = memory.sample(args.replay_size, g_rep)
                        batch = concat_replay(batch, shuffle_within(rep, g_rep) if mode == "S" else rep)
                    elif replay_on and mode == "G":
                        batch = concat_replay(batch, memory.sample(args.replay_size, g_rep))
                    elif replay_on and mode == "C":
                        cur = data["tasks"][lang]["train"].chunks
                        batch = concat_replay(batch, cur[torch.randint(0, len(cur), (args.replay_size,),
                                                                       generator=g_rep)])
                    task_losses.append(trainer.train_step(batch))
                nrep += args.replay_size if replay_on else 0
        losses.append(float(np.mean(task_losses[-20:])))
        n_replayed.append(nrep)
        g_store = torch.Generator().manual_seed(seed * 1000 + 500 + i)
        if mode in ("H", "HR"):
            drift.append(memory.drift(model))  # 再生してきた過去タスクの隠れ状態の古さ（このタスクの保存前）
            memory.store(model, data["tasks"][lang]["train"], g_store)
            nbytes.append(memory.nbytes[-1])
        elif mode == "G":
            memory.store(data["tasks"][lang]["train"], g_store)
            nbytes.append(memory.nbytes[-1])
        elif mode in "RS":
            memory.store(data["tasks"][lang]["train"], g_store)
            nbytes.append(raw_nbytes(memory.slots[-1]) if mode == "R" else bag_nbytes(memory.slots[-1]))
        for j, lj in enumerate(order):
            rows[i][j] = eval_task_decomp(model, data["tasks"][lj]["eval"])
        log(f"[{mode}{'' if mode != 'H' else layer} seed{seed}] task{i} {lang} done {time.time() - t0:.0f}s "
            f"replayed {nrep} diag ce {rows[i][i]['ce']:.3f} | row ce "
            + " ".join(f"{r['ce']:.2f}" for r in rows[i]) + " | marg " + " ".join(f"{r['ce_marg']:.2f}" for r in rows[i]))
        if ckpt and i < T - 1:
            tmp = f"{ckpt}.tmp"
            torch.save({"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                        "step_count": trainer.step_count, "memory": dict(memory.__dict__) if memory is not None else None,
                        "rows": rows, "losses": losses, "n_replayed": n_replayed, "nbytes": nbytes, "drift": drift,
                        "refresh_log": refresh_log, "gen_log": gen_log,
                        "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(), "next_task": i + 1,
                        "elapsed": time.time() - t0}, tmp)
            Path(tmp).replace(ckpt)
    acc = [[r["acc"] for r in row] for row in rows]
    ce = [[r["ce"] for r in row] for row in rows]
    metrics = cf.summarize(acc, ce)
    metrics.update(decomp_forgetting(rows))
    res = {"mode": mode, "base_phase": base, "seed": seed, "order": order, "acc": acc, "ce": ce,
           "ce_marg": [[r["ce_marg"] for r in row] for row in rows], "metrics": metrics,
           "unigram_ce": {l: data["tasks"][l]["unigram_ce"] for l in order},
           "final_train_loss_per_task": losses, "n_params": sum(p.numel() for p in model.parameters()),
           "memory_per_task": args.memory_per_task if memory is not None else 0, "replay_size": args.replay_size,
           "layer": layer if mode in ("H", "HR") else None, "n_replayed_per_task": n_replayed,
           "memory_bytes_per_task": nbytes, "memory_bytes_total": int(sum(nbytes[:-1])),  # 最後のタスクの保存は使われない
           "train_seconds": time.time() - t0, "env": env_info()}
    if mode in ("H", "HR"):
        res.update({"pq_m": args.pq_m, "pq_nbits": args.pq_nbits, "pq_recon_rel_err": memory.recon_rel_err,
                    "drift_before_store": drift, "drift_final": memory.drift(model)})
    if mode == "HR":
        res.update({"refresh_drift": refresh_log,
                    "refresh_forward_chunks_total": int(sum(e["n_chunks"] for e in refresh_log))})
    if mode == "G":
        res.update({"prefix_len": memory.prefix_len, "gen_quality": gen_log})
    return res


def add_unigram(data: dict) -> dict:
    for l in data["langs"]:
        t = data["tasks"][l]
        t["unigram_ce"] = unigram_ce(t["train"].chunks, t["eval"].chunks, data["vocab_size"])
    return data


def layer_of(args, mode: str | None = None) -> int:
    """保存層。H は既定で最終層（読み出しだけの再生）、HR は 44.13 設計で第 2 層に固定。どちらも --layer で明示できる。"""
    if args.layer is not None:
        return args.layer
    return 2 if (mode or getattr(args, "mode", None)) == "HR" else args.num_layers


def tag_of(args) -> str:
    """結果ファイルの条件名（例: R_M64_r16, H2_M64_r16_pq16x8, HR2_M64_r16_pq16x8, G_M64_r16）。"""
    t = args.mode + (str(layer_of(args, args.mode)) if args.mode in ("H", "HR") else "")
    if args.mode != "V":
        t += f"_M{args.memory_per_task}_r{args.replay_size}"
    if args.mode in ("H", "HR"):
        t += f"_pq{args.pq_m}x{args.pq_nbits}"
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--memory-per-task", type=int, default=64)
    ap.add_argument("--replay-size", type=int, default=16)
    ap.add_argument("--layer", type=int, default=None, help="H の保存層（既定は最終層 = 読み出しだけの再生）")
    ap.add_argument("--pq-m", type=int, default=16)
    ap.add_argument("--pq-nbits", type=int, default=8)
    ap.add_argument("--prefix-len", type=int, default=PREFIX_LEN, help="G の保存する索引の長さ（44.13 設計で 16 に固定）")
    ap.add_argument("--europarl-dir", default=str(ROOT / "data" / "europarl_raw"))
    ap.add_argument("--langs", nargs="+", default=list(cf.LANGS))
    ap.add_argument("--per-lang-vocab", type=int, default=2500)
    ap.add_argument("--n-train", type=int, default=250_000)
    ap.add_argument("--n-eval", type=int, default=25_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ckpt-dir", default=None, help="タスクごとのチェックポイントの置き場（再起動からの再開用）")
    args = ap.parse_args(argv)
    assert args.grad_accum == 1, "再生の更新は勾配累積 1 を前提とする"
    if args.threads:
        torch.set_num_threads(args.threads)
        try:
            import faiss
            faiss.omp_set_num_threads(args.threads)
        except ImportError:
            pass
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = add_unigram(cf.build_tasks(Path(args.europarl_dir), tuple(args.langs), args.per_lang_vocab, args.n_train,
                                      args.n_eval, args.max_seq_length))
    print(f"vocab {data['vocab_size']} unigram_ce "
          + str({l: round(data['tasks'][l]['unigram_ce'], 3) for l in data['langs']}), flush=True)
    for seed in args.seeds:
        path = out / f"{tag_of(args)}_seed{seed}.json"
        if path.exists():
            continue
        ck = None
        if args.ckpt_dir:
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            ck = str(Path(args.ckpt_dir) / f"{tag_of(args)}_seed{seed}.pt")
        res = run(args.mode, seed, data, args, log=lambda s: print(s, flush=True), ckpt=ck)
        res["tag"] = tag_of(args)
        res["config"] = {k: v for k, v in vars(args).items()}
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"saved {path} metrics {res['metrics']}", flush=True)
        if ck and Path(ck).exists():
            Path(ck).unlink()


if __name__ == "__main__":
    main()
