#!/usr/bin/env python3
"""ステップ44.16: フル規模での再検証（文脈への過適合と CX4 の概念的再検証）。

44.12〜44.15 の小規模プロトコル（step44_compressed_replay）を、768d・12層・系列長256・
WikiText-103（GPT-2 BPE）のフル規模へ対応づけた実行系。設計の固定版は
docs/brain-structure-research.md のステップ44.16 設計の節（判断点10件は 2026-11-08 に
ユーザー確定、構成 C と実行基盤 pilot_first は typesafe-ai の正式判定とも一致）。

小規模との主な差分（設計で固定したもの）:
- タスク: WikiText-103 train を文書単位で4つの連続部分コーパス T1〜T4 に分け固定順に学ぶ。
  評価用保持データは各部分コーパスから文書単位で分離する（同一文書が学習と評価にまたがらない）。
- 露出の正規化: 再生規則は「更新あたりの本数」ではなく、1断片・1タスク段階あたりの
  期待提示回数 E（R 7.75／R_M4・C4 124.0／X4・CX4 15.5、小規模の実測値）で引き継ぐ。
  段階の総提示回数 Q＝E×プール本数を、段階の全更新へ端数繰り越しの累算で決定的に配る。
- 保存 M は断片の本数を維持（R 64／M4 系 4）。切り出し窓は 128 トークン（断片の 1/2）。
- 学習は 1 パス既定（--passes 2 が事前登録済みエスカレーション）。lr 5e-5、実効バッチ64
  （マイクロ8×勾配累積8）、fp16 は GPU 実行時。
- 真正性の担保: 起動時に git_commit の null／+dirty を拒否する（--allow-unclean-tree で
  明示した場合のみ許可）。チェックポイントはタスク段階ごと＋一定更新ごとに保存し、
  SHA-256 のサイドカーを添える。再開は中断しない場合と一致する（CPU ではビット一致を
  テストで確認する）。

使い方（パイロット）:
  python step44_fullscale_replay.py --mode CX4 --seeds 990 --out-dir <Drive>/pilot_ops \
      --corpus-frac 0.03125 --device cuda --fp16 --ckpt-dir <Drive>/ckpt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import step44_compressed_replay as cr  # noqa: E402  eval_task_decomp / decomp_forgetting / crop_windows / unigram_ce
import step44_continual_forgetting as cf  # noqa: E402  summarize
from colab_step44_full_scale_experiment import (  # noqa: E402
    BrainInspiredLLMFullScale,
    FullScaleTrainer,
    Step444Config,
    WordChunkDataset,
)
from neurocortex.run_env import env_info  # noqa: E402
from step44_continual_replay import EpisodicMemory, concat_replay  # noqa: E402

MODES = ("V", "R", "R_M4", "X4", "C4", "CX4")
EPISODIC_MODES = ("R", "R_M4", "X4", "C4", "CX4")
MEMORY_PER_TASK = {"R": 64, "R_M4": 4, "X4": 4, "C4": 4, "CX4": 4}  # 断片の本数を維持（設計 §2.4）
EXPOSURE_E = {"R": 7.75, "R_M4": 124.0, "X4": 15.5, "C4": 124.0, "CX4": 15.5}  # 小規模の実測値（設計 §2.5）
WINDOW_MODES = ("C4", "CX4")
WINDOW_LEN = 128  # 断片 256 トークンの 1/2（比例案、設計 §2.4）
EVAL_TOKENS = 262_144  # タスクごとの評価トークン量の暫定値（パイロット後に固定する、設計 §2.7）
ID_BYTES = 4  # BPE の語彙 50,257 は uint16 に収まらないため保存の会計は 4 バイト/トークンとする


# ---------------------------------------------------------------- 露出正規化スケジューラ

class ExposureScheduler:
    """段階内の総提示回数 total を n_steps 回の呼び出しに決定的に配る（整数の累算、端数繰り越し）。

    i 回目の呼び出しまでに渡す累計は floor(i*total/n_steps) で、合計はちょうど total になる。
    状態は (done, given) の2整数だけで、チェックポイントにそのまま保存できる。
    """

    def __init__(self, total: int, n_steps: int):
        assert total >= 0 and n_steps >= 1
        self.total, self.n_steps = int(total), int(n_steps)
        self.done, self.given = 0, 0

    def next(self) -> int:
        self.done += 1
        target = (self.done * self.total) // self.n_steps
        k = target - self.given
        self.given = target
        return k

    def state(self) -> dict:
        return {"total": self.total, "n_steps": self.n_steps, "done": self.done, "given": self.given}

    @classmethod
    def from_state(cls, st: dict) -> "ExposureScheduler":
        s = cls(st["total"], st["n_steps"])
        s.done, s.given = st["done"], st["given"]
        return s


def literal_rule_presentations(mode: str, n_optimizer_steps: int, pool: int) -> float:
    """小規模の規則を字面どおり移植した場合の、1断片あたり提示回数の見通し（棄却理由の記録用）。

    R・R_M4・C4 は r=16 本/更新、X4・CX4 は pool//4 本/更新。いずれも提示回数は更新数に比例する。
    """
    if pool <= 0 or n_optimizer_steps <= 0:
        return 0.0
    r = 16 if mode in ("R", "R_M4", "C4") else max(1, pool // 4)
    return r * n_optimizer_steps / pool


# ---------------------------------------------------------------- 真正性: 起動時チェックとハッシュ

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha256(config: dict) -> str:
    return sha256_bytes(json.dumps(config, sort_keys=True, ensure_ascii=False).encode())


def check_clean_tree(allow: bool = False) -> str | None:
    """git_commit が null／+dirty なら実行を開始しない（44.4 撤回の再発防止、設計 §4.1）。

    許可は --allow-unclean-tree の明示がある場合だけ。戻り値は記録用のコミット文字列。
    """
    commit = env_info()["git_commit"]
    if allow:
        return commit
    if commit is None:
        raise RuntimeError("git_commit が取得できません（git clone したツリーから実行してください）。"
                           "明示的に許可する場合だけ --allow-unclean-tree を使ってください")
    if commit.endswith("+dirty"):
        raise RuntimeError(f"作業ツリーが dirty です（{commit}）。修正は必ずコミットしてから実行してください。"
                           "明示的に許可する場合だけ --allow-unclean-tree を使ってください")
    return commit


def write_checkpoint(path: str, state: dict) -> str:
    """一時ファイルへ書いてから置き換え、SHA-256 のサイドカー（<path>.sha256）を添える。"""
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    Path(tmp).replace(path)
    digest = sha256_file(Path(path))
    Path(f"{path}.sha256").write_text(digest + "\n")
    return digest


def verify_checkpoint(path: str) -> bool:
    side = Path(f"{path}.sha256")
    return side.exists() and side.read_text().strip() == sha256_file(Path(path))


# ---------------------------------------------------------------- WikiText-103 のタスク構築

_DOC_HEAD = re.compile(r"^ = [^=]+ = *$")


def wikitext_documents(lines: list[str]) -> list[str]:
    """WikiText の生テキストを行ごとに受け取り、記事（文書）単位の文字列の列を返す。

    記事の区切りは見出し行「 = Title = 」。見出しより前の行は捨てる（通常は空）。
    文書の同定は、この列での位置（0 始まりの番号）で行う。
    """
    docs, cur = [], None
    for line in lines:
        if _DOC_HEAD.match(line.rstrip("\n")):
            if cur:
                docs.append("".join(cur))
            cur = [line if line.endswith("\n") else line + "\n"]
        elif cur is not None:
            cur.append(line if line.endswith("\n") else line + "\n")
    if cur:
        docs.append("".join(cur))
    return [d for d in docs if d.strip()]


def tokenize_documents(texts: list[str], cache: Path | None = None) -> list[torch.Tensor]:
    """文書ごとに GPT-2 BPE でトークン化する。cache があれば読み書きする（文書の列そのものを保存）。"""
    if cache is not None and cache.exists():
        return torch.load(cache, weights_only=False)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    docs = [torch.tensor(tok(t)["input_ids"], dtype=torch.long) for t in texts]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        torch.save(docs, tmp)
        tmp.replace(cache)
    return docs


def documents_sha256(docs: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.numpy().astype("<i8").tobytes())
    return h.hexdigest()


def split_documents(docs: list[torch.Tensor], n_tasks: int = 4, seq_len: int = 256,
                    eval_tokens: int = EVAL_TOKENS, corpus_frac: float = 1.0) -> dict:
    """トークン化済み文書の列を、連続した n_tasks 個の部分コーパスへ分ける（設計 §2.2 案A1）。

    - 部分コーパスの境界は文書単位で、トークン量がほぼ均等になる位置に置く（公式 train の並びを保つ）。
    - 各部分コーパスの末尾の文書から、評価専用の保持データを文書単位で分離する
      （同一文書が学習と評価にまたがらない）。評価集合は先頭から eval_tokens トークンに切る。
    - corpus_frac < 1 のとき（パイロット）は、各部分コーパスの学習文書を先頭からその割合だけ使う。
      評価集合は変えない（評価時間の実測もパイロットの目的のため）。
    戻り値は data（tasks/order/vocab_size）と split（文書番号の内訳）の組。
    """
    assert 1.0 >= corpus_frac > 0
    total = sum(len(d) for d in docs)
    parts: list[list[int]] = [[] for _ in range(n_tasks)]
    cum = 0
    for idx, d in enumerate(docs):
        k = min(n_tasks - 1, (cum * n_tasks) // max(total, 1))
        parts[k].append(idx)
        cum += len(d)
    tasks, split = {}, {}
    for k, doc_ids in enumerate(parts):
        assert len(doc_ids) >= 2, f"部分コーパス {k} の文書が少なすぎます（{len(doc_ids)}）"
        eval_ids, eval_tok = [], 0
        j = len(doc_ids)
        while j > 1 and eval_tok < eval_tokens:  # 末尾から文書単位で保持側へ（学習側に最低1文書を残す）
            j -= 1
            eval_ids.append(doc_ids[j])
            eval_tok += len(docs[doc_ids[j]])
        eval_ids.reverse()
        train_ids_all = doc_ids[:j]
        if corpus_frac < 1.0:
            budget = math.ceil(sum(len(docs[i]) for i in train_ids_all) * corpus_frac)
            train_ids, used = [], 0
            for i in train_ids_all:
                if used >= budget:
                    break
                train_ids.append(i)
                used += len(docs[i])
        else:
            train_ids = train_ids_all
        train_stream = torch.cat([docs[i] for i in train_ids])
        eval_stream = torch.cat([docs[i] for i in eval_ids])[:eval_tokens]
        name = f"T{k + 1}"
        tasks[name] = {"train": WordChunkDataset(train_stream, seq_len),
                       "eval": WordChunkDataset(eval_stream, seq_len)}
        split[name] = {"train_doc_ids": train_ids, "eval_doc_ids": eval_ids,
                       "train_tokens": int(len(train_stream)), "eval_tokens": int(len(eval_stream))}
    return {"tasks": tasks, "order": [f"T{k + 1}" for k in range(n_tasks)], "split": split}


def build_data(docs: list[torch.Tensor], vocab_size: int, seq_len: int = 256,
               eval_tokens: int = EVAL_TOKENS, corpus_frac: float = 1.0) -> dict:
    """split_documents の結果に語彙・出所情報・1-gram 水準を加えた、run() が受け取る data を作る。"""
    sp = split_documents(docs, seq_len=seq_len, eval_tokens=eval_tokens, corpus_frac=corpus_frac)
    data = {"vocab_size": vocab_size, "tasks": sp["tasks"], "order": sp["order"],
            "provenance": {"documents_sha256": documents_sha256(docs), "n_documents": len(docs),
                           "total_tokens": int(sum(len(d) for d in docs)), "split": sp["split"],
                           "corpus_frac": corpus_frac, "eval_tokens": eval_tokens}}
    for name in data["order"]:
        t = data["tasks"][name]
        t["unigram_ce"] = cr.unigram_ce(t["train"].chunks, t["eval"].chunks, vocab_size)
    data["unigram_ce"] = {name: data["tasks"][name]["unigram_ce"] for name in data["order"]}
    return data


def load_wikitext_texts(data_dir: str | None) -> list[str]:
    """学習テキスト（行の列）を得る。data_dir があれば生の train.txt、無ければ HF datasets の train。"""
    if data_dir:
        return (Path(data_dir) / "train.txt").read_text(encoding="utf-8").splitlines()
    from datasets import load_dataset
    return list(load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")["text"])


# ---------------------------------------------------------------- 学習

def window_step(trainer: FullScaleTrainer, model, batch: dict, rep: torch.Tensor) -> float:
    """新しいタスクのバッチ（全長）と切り出した窓の再生（短い系列）で 1 回のマイクロ更新をする。

    損失は断片数で重みづけた平均（小規模の raw_window_step と同じ規則）。勾配累積・fp16・
    勾配クリップは FullScaleTrainer.train_step と同じ扱いにし、最適化は累積の区切りでのみ行う。
    """
    model.train()
    dev = trainer.device
    cfg = trainer.config
    x = batch["input_ids"].to(dev)
    rep = rep.to(dev)

    def combined_loss():
        l_new = model(input_ids=x, labels=batch["labels"].to(dev),
                      attention_mask=batch["attention_mask"].to(dev))["loss"]
        l_rep = model(input_ids=rep, labels=rep, attention_mask=torch.ones_like(rep))["loss"]
        return (len(x) * l_new + len(rep) * l_rep) / (len(x) + len(rep))

    step_now = (trainer.step_count + 1) % cfg.gradient_accumulation_steps == 0
    if cfg.fp16 and trainer.scaler:
        from torch.cuda.amp import autocast
        with autocast():
            loss = combined_loss() / cfg.gradient_accumulation_steps
        trainer.scaler.scale(loss).backward()
        if step_now:
            trainer.scaler.unscale_(trainer.optimizer)
            trainer._clip_grads()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            trainer.optimizer.zero_grad()
    else:
        loss = combined_loss() / cfg.gradient_accumulation_steps
        loss.backward()
        if step_now:
            trainer._clip_grads()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
    trainer.step_count += 1
    return loss.item() * cfg.gradient_accumulation_steps


def _run_config(mode: str, seed: int, args, data: dict) -> dict:
    """結果に記録する設定全体（正規化済み）。config_sha256 はこの dict から作る。"""
    return {"step": "44.16", "mode": mode, "seed": seed,
            "hidden_size": args.hidden_size, "num_layers": args.num_layers, "nhead": args.nhead,
            "max_seq_length": args.max_seq_length, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "lr": args.lr, "fp16": bool(args.fp16),
            "passes": args.passes, "memory_per_task": MEMORY_PER_TASK.get(mode, 0),
            "exposure_E": EXPOSURE_E.get(mode), "window_len": WINDOW_LEN if mode in WINDOW_MODES else None,
            "eval_tokens": data["provenance"]["eval_tokens"], "corpus_frac": data["provenance"]["corpus_frac"],
            "documents_sha256": data["provenance"]["documents_sha256"]}


def run(mode: str, seed: int, data: dict, args, log=print, ckpt: str | None = None) -> dict:
    """1 つの (条件, seed) で T1〜T4 を固定順に学ぶ。V 以外はフェーズ A のモデル。

    ckpt を渡すと、タスク段階ごと＋ --ckpt-every-steps（最適化ステップ単位）ごとに状態を保存し、
    次回はそこから再開する（中断しない場合と一致する。再開に使ったチェックポイントは検証する）。
    """
    assert mode in MODES
    if not getattr(args, "allow_unclean_tree", False):
        check_clean_tree()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    cfg = Step444Config(phase=("V" if mode == "V" else "A"), device=args.device,
                        fp16=bool(args.fp16) and str(args.device).startswith("cuda"),
                        max_seq_length=args.max_seq_length, hidden_size=args.hidden_size,
                        num_layers=args.num_layers, nhead=args.nhead, batch_size=args.batch_size,
                        learning_rate=args.lr, gradient_accumulation_steps=args.grad_accum)
    model = BrainInspiredLLMFullScale(cfg, vocab_size=data["vocab_size"], hidden_size=cfg.hidden_size,
                                      num_layers=cfg.num_layers, nhead=cfg.nhead)
    trainer = FullScaleTrainer(model, cfg)
    memory = EpisodicMemory(MEMORY_PER_TASK[mode]) if mode in EPISODIC_MODES else None
    order = list(data["order"])
    T = len(order)
    rows = [[None] * T for _ in range(T)]
    losses, nbytes = [], []
    replay_counts, replay_present, replay_tokens = [], [], []  # 段階ごとの断片別提示回数・総提示数・再生トークン数
    stage_seconds, eval_seconds = [], 0.0
    total_micro = 0
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # 再開用の状態（段階の途中も含む）
    st0 = {"stage": 0, "epoch": 0, "batch_idx": 0}
    sched_state, counts_state = None, None
    g_rep_state = None
    resumed = False
    task_losses_state: list[float] = []
    nrep_state, ntok_state, stage_train_state = 0, 0, 0.0
    if ckpt and Path(ckpt).exists():
        if not verify_checkpoint(ckpt):
            raise RuntimeError(f"チェックポイントの SHA-256 が一致しません: {ckpt}")
        st = torch.load(ckpt, weights_only=False)
        log(f"[{mode} seed{seed}] resumed from checkpoint sha256 {sha256_file(Path(ckpt))[:16]}… "
            f"(stage {st['stage']}, epoch {st['epoch']}, batch {st['batch_idx']})")
        model.load_state_dict(st["model"])
        trainer.optimizer.load_state_dict(st["optimizer"])
        if trainer.scaler and st.get("scaler"):
            trainer.scaler.load_state_dict(st["scaler"])
        trainer.step_count = st["step_count"]
        if memory is not None and st.get("memory") is not None:
            memory.__dict__.update(st["memory"])
        rows, losses, nbytes = st["rows"], st["losses"], st["nbytes"]
        replay_counts, replay_present, replay_tokens = st["replay_counts"], st["replay_present"], st["replay_tokens"]
        stage_seconds, eval_seconds, total_micro = st["stage_seconds"], st["eval_seconds"], st["total_micro"]
        started_utc = st["started_utc"]
        st0 = {"stage": st["stage"], "epoch": st["epoch"], "batch_idx": st["batch_idx"]}
        resumed = True
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda_rng"]])
        np.random.set_state(st["np_rng"])
        g_rep_state = st.get("g_rep_state")
        sched_state, counts_state = st.get("sched"), st.get("counts")
        task_losses_state = st.get("task_losses", [])
        nrep_state, ntok_state, stage_train_state = st.get("nrep", 0), st.get("ntok", 0), st.get("stage_train", 0.0)

    def save_ckpt(stage, epoch, batch_idx, sched, counts, task_losses, nrep, ntok, stage_train, g_rep=None):
        if not ckpt:
            return
        write_checkpoint(ckpt, {
            "model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
            "scaler": trainer.scaler.state_dict() if trainer.scaler else None,
            "step_count": trainer.step_count,
            "memory": dict(memory.__dict__) if memory is not None else None,
            "rows": rows, "losses": losses, "nbytes": nbytes,
            "replay_counts": replay_counts, "replay_present": replay_present, "replay_tokens": replay_tokens,
            "stage_seconds": stage_seconds, "eval_seconds": eval_seconds, "total_micro": total_micro,
            "started_utc": started_utc, "stage": stage, "epoch": epoch, "batch_idx": batch_idx,
            "sched": sched.state() if sched is not None else None,
            "g_rep_state": g_rep.get_state() if g_rep is not None else None,
            "counts": counts, "task_losses": task_losses, "nrep": nrep, "ntok": ntok,
            "stage_train": stage_train,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
            "np_rng": np.random.get_state()})

    accum = cfg.gradient_accumulation_steps
    for i, name in enumerate(order):
        if i < st0["stage"]:
            continue
        resume_here = resumed and i == st0["stage"]
        train_ds = data["tasks"][name]["train"]
        n_micro_epoch = -(-len(train_ds) // cfg.batch_size)
        n_steps_stage = n_micro_epoch * args.passes
        pool = torch.cat(memory.slots) if (memory is not None and len(memory)) else None
        replay_on = pool is not None and i > 0
        E = EXPOSURE_E.get(mode)
        sched = None
        if replay_on:
            sched = (ExposureScheduler.from_state(sched_state)
                     if (resume_here and sched_state is not None)
                     else ExposureScheduler(int(round(E * len(pool))), n_steps_stage))
        counts = (counts_state if (resume_here and counts_state is not None)
                  else (torch.zeros(len(pool), dtype=torch.long) if pool is not None else None))
        g_rep = torch.Generator().manual_seed(seed * 1000 + 700 + i)
        if resume_here and g_rep_state is not None:
            g_rep.set_state(g_rep_state)  # 再生の抽選列を、保存時点の続きから再開する
        task_losses = list(task_losses_state) if resume_here else []
        nrep = nrep_state if resume_here else 0
        ntok = ntok_state if resume_here else 0
        stage_train = stage_train_state if resume_here else 0.0
        start_epoch = st0["epoch"] if resume_here else 0
        t_stage = time.time()
        trainer.optimizer.zero_grad()
        for epoch in range(start_epoch, args.passes):
            g = torch.Generator().manual_seed(seed * 1000 + i * 10 + epoch)
            loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
            start_batch = st0["batch_idx"] if (resume_here and epoch == start_epoch) else 0
            for b, batch in enumerate(loader):
                if b < start_batch:
                    continue
                k = sched.next() if sched is not None else 0
                if k > 0:
                    idx = torch.randint(0, len(pool), (k,), generator=g_rep)
                    counts += torch.bincount(idx, minlength=len(pool))
                    rep = pool[idx]
                    ntok += int(rep.numel()) if mode not in WINDOW_MODES else 0
                    if mode in WINDOW_MODES:
                        rep = cr.crop_windows(rep, WINDOW_LEN, g_rep)
                        ntok += int(rep.numel())
                        loss_val = window_step(trainer, model, batch, rep)
                    else:
                        loss_val = trainer.train_step(concat_replay(batch, rep))
                    nrep += k
                else:
                    loss_val = trainer.train_step(batch)
                task_losses.append(loss_val)
                total_micro += 1
                if (ckpt and args.ckpt_every_steps > 0 and trainer.step_count % accum == 0
                        and (trainer.step_count // accum) % args.ckpt_every_steps == 0):
                    save_ckpt(i, epoch, b + 1, sched, counts, task_losses, nrep, ntok,
                              stage_train + (time.time() - t_stage), g_rep=g_rep)
        stage_train += time.time() - t_stage
        stage_seconds.append(stage_train)
        losses.append(float(np.mean(task_losses[-20:])))
        replay_present.append(nrep)
        replay_tokens.append(ntok)
        if counts is not None:
            per_task, off = [], 0
            for s in memory.slots:
                per_task.append([int(v) for v in counts[off:off + len(s)]])
                off += len(s)
            replay_counts.append(per_task)
        else:
            replay_counts.append([])
        if memory is not None:
            memory.store(train_ds, torch.Generator().manual_seed(seed * 1000 + 500 + i))
            nbytes.append(int(memory.slots[-1].numel() * ID_BYTES))
        t_eval = time.time()
        for j, nj in enumerate(order):
            rows[i][j] = cr.eval_task_decomp(model, data["tasks"][nj]["eval"],
                                             batch_size=args.eval_batch_size)
        eval_seconds += time.time() - t_eval
        log(f"[{mode} seed{seed}] stage{i} {name} done train {stage_train:.0f}s presented {nrep} "
            f"diag ce {rows[i][i]['ce']:.3f} | row ce " + " ".join(f"{r['ce']:.2f}" for r in rows[i]))
        # 次の段階へ進む前に、途中の状態を引き継がない形で保存する
        sched_state, counts_state = None, None
        g_rep_state = None
        resumed = False
        task_losses_state, nrep_state, ntok_state, stage_train_state = [], 0, 0, 0.0
        st0 = {"stage": i + 1, "epoch": 0, "batch_idx": 0}
        if i < T - 1:
            save_ckpt(i + 1, 0, 0, None, None, [], 0, 0, 0.0)
    # 保存断片そのものの評価（過適合の直接指標、記述的。44.14 と同じ）
    stored_eval, stored_gap = None, None
    if memory is not None and memory.slots:
        seq_len = memory.slots[0].size(1)
        stored_eval = [cr.eval_task_decomp(model, WordChunkDataset(memory.slots[j].reshape(-1).clone(), seq_len),
                                            batch_size=args.eval_batch_size) for j in range(T)]
        past = range(T - 1)
        stored_gap = {"gap_ce_past": float(np.mean([rows[-1][j]["ce"] - stored_eval[j]["ce"] for j in past])),
                      "gap_gain_past": float(np.mean([stored_eval[j]["gain"] - rows[-1][j]["gain"] for j in past]))}
    acc = [[r["acc"] for r in row] for row in rows]
    ce = [[r["ce"] for r in row] for row in rows]
    metrics = cf.summarize(acc, ce)
    metrics.update(cr.decomp_forgetting(rows))
    train_seconds = float(sum(stage_seconds))
    audit = exposure_audit(mode, replay_counts, replay_present, order)
    lit = []
    for i in range(T):
        pool_i = MEMORY_PER_TASK.get(mode, 0) * i
        n_opt = (total_micro // max(T, 1)) // accum  # 段階あたりの最適化ステップの近似（記録用）
        lit.append(literal_rule_presentations(mode, n_opt, pool_i) if pool_i else 0.0)
    config = _run_config(mode, seed, args, data)
    res = {"mode": mode, "base_phase": cfg.phase, "seed": seed, "order": order, "acc": acc, "ce": ce,
           "ce_marg": [[r["ce_marg"] for r in row] for row in rows], "metrics": metrics,
           "unigram_ce": data["unigram_ce"],
           "final_train_loss_per_task": losses, "n_params": sum(p.numel() for p in model.parameters()),
           "memory_per_task": MEMORY_PER_TASK.get(mode, 0), "exposure_E": EXPOSURE_E.get(mode),
           "window_len": WINDOW_LEN if mode in WINDOW_MODES else None,
           "n_presented_per_task": replay_present, "replay_counts_per_task": replay_counts,
           "replay_tokens_per_task": replay_tokens, "stored_eval": stored_eval, "stored_gap": stored_gap,
           "memory_bytes_per_task": nbytes, "memory_bytes_total": int(sum(nbytes[:-1])),
           "train_seconds": train_seconds, "eval_seconds": float(eval_seconds),
           "stage_seconds": stage_seconds,
           "steps_per_sec": (total_micro / train_seconds) if train_seconds > 0 else None,
           "tokens_per_sec": (total_tokens_trained(data, args) / train_seconds) if train_seconds > 0 else None,
           "exposure_audit": audit, "literal_rule_presentations": lit,
           "started_utc": started_utc,
           "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "config": config, "config_sha256": config_sha256(config),
           "data_provenance": data["provenance"], "env": env_info()}
    return res


def total_tokens_trained(data: dict, args) -> int:
    return int(sum(len(data["tasks"][n]["train"]) for n in data["order"])
               * args.max_seq_length * args.passes)


def exposure_audit(mode: str, replay_counts: list, replay_present: list, order: list) -> dict:
    """段階ごとに、実提示回数（総数と1断片平均）を目標 E と並べた監査記録（操作の検証用）。"""
    E = EXPOSURE_E.get(mode)
    per_stage = []
    for i, per_task in enumerate(replay_counts):
        flat = [v for t in per_task for v in t]
        per_stage.append({"stage": i, "pool": len(flat), "total_presentations": int(sum(flat)),
                          "mean_per_fragment": float(np.mean(flat)) if flat else 0.0,
                          "max_per_fragment": int(max(flat)) if flat else 0})
    return {"target_E": E, "per_stage": per_stage, "recorded_presentations": replay_present}


def tag_of(mode: str) -> str:
    """結果ファイルの条件名（例: R_M64_E7.75, CX4_M4_E15.5_w128）。"""
    if mode == "V":
        return "V"
    t = mode if mode == "R_M4" else f"{mode}_M{MEMORY_PER_TASK[mode]}"  # R_M4 は名前自体に M=4 を含む
    t += f"_E{EXPOSURE_E[mode]:g}"
    if mode in WINDOW_MODES:
        t += f"_w{WINDOW_LEN}"
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default=None, help="生の WikiText-103（train.txt）の置き場。無ければ HF datasets")
    ap.add_argument("--doc-cache", default=None, help="トークン化済み文書のキャッシュ（.pt）")
    ap.add_argument("--eval-tokens", type=int, default=EVAL_TOKENS)
    ap.add_argument("--corpus-frac", type=float, default=1.0, help="パイロット用: 学習コーパスの割合（評価は変えない）")
    ap.add_argument("--passes", type=int, default=1, help="1 が既定。2 は事前登録済みエスカレーション時のみ")
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument("--nhead", type=int, default=12)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8, help="マイクロバッチ")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-every-steps", type=int, default=500, help="段階途中の保存間隔（最適化ステップ単位。0 で段階境界のみ）")
    ap.add_argument("--allow-unclean-tree", action="store_true",
                    help="git_commit の null／+dirty でも実行する（開発・スモーク用。本実行では使わない）")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    commit = check_clean_tree(allow=args.allow_unclean_tree)
    print(f"git_commit: {commit}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = load_wikitext_texts(args.data_dir)
    docs_text = wikitext_documents(texts)
    print(f"documents: {len(docs_text)}", flush=True)
    docs = tokenize_documents(docs_text, Path(args.doc_cache) if args.doc_cache else None)
    data = build_data(docs, vocab_size=50257, seq_len=args.max_seq_length,
                      eval_tokens=args.eval_tokens, corpus_frac=args.corpus_frac)
    print(f"total tokens {data['provenance']['total_tokens']:,} "
          f"sha256 {data['provenance']['documents_sha256'][:16]}…", flush=True)
    for seed in args.seeds:
        path = out / f"{tag_of(args.mode)}_seed{seed}.json"
        if path.exists():
            print(f"skip {path} (exists)", flush=True)
            continue
        ck = None
        if args.ckpt_dir:
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            ck = str(Path(args.ckpt_dir) / f"{tag_of(args.mode)}_seed{seed}.pt")
        res = run(args.mode, seed, data, args, log=lambda s: print(s, flush=True), ckpt=ck)
        res["tag"] = tag_of(args.mode)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        tmp.replace(path)
        print(f"saved {path} metrics {res['metrics']}", flush=True)
        if ck and Path(ck).exists():
            Path(ck).unlink()
            Path(f"{ck}.sha256").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
