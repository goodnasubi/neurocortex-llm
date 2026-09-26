#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップ44.4: フル規模実験（WikiText-103 実データ × Phase A/B/C）

目的:
- WikiText-103 全データでの長期学習
- Phase A（海馬） vs Phase B（基底核追加） vs Phase C（小脳追加）の比較
- 実規模データでの安定性・性能を測定

実行環境:
- Google Colab（T4/A100 GPU推奨）
- ローカルGPU（任意）

実行方法:
  python colab_step44_full_scale_experiment.py
"""

import sys
from pathlib import Path
# Colab環境では __file__ が未定義なので条件付き実行
if '__file__' in globals():
    sys.path.insert(0, str(Path(__file__).parent))

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logger = logging.getLogger(__name__)


@dataclass
class Step444Config:
    """ステップ44.4 フル規模実験設定"""

    # フェーズ制御
    phase: str = "A"  # "A", "B", or "C"
    hippocampus_enabled: bool = True
    basal_ganglia_enabled: bool = False  # Phase B/C で有効
    cerebellum_enabled: bool = False  # Phase C で有効

    # データセット（WikiText-103 実データ）
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    max_seq_length: int = 256  # Colab T4 メモリ対応
    vocab_size: int = 50257

    # 学習設定（フル規模）
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 10  # 実規模データなので少数エポック
    num_seeds: int = 3   # 統計的妥当性確保
    gradient_accumulation_steps: int = 2  # 実効 batch size = 64

    # モジュール Loss 重み（最適化済み）
    hippocampus_loss_weight: float = 0.05  # ステップ44.3.1 で最適化済み
    basal_ganglia_loss_weight: float = 0.05
    cerebellum_loss_weight: float = 0.02

    # リソース
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True if torch.cuda.is_available() else False
    max_grad_norm: float = 1.0
    eval_interval: int = 500  # 500 step ごとに評価

    # 出力
    results_dir: str = "results/step44_full_scale_experiment"
    # チェックポイント（Colab のセッション切れ対策。Google Drive 上のパスを推奨）
    checkpoint_dir: str = "results/step44_full_scale_experiment/checkpoints"
    checkpoint_every_steps: int = 2000
    # 実行時間の上限（time.time() の値）。超えたらチェックポイントを保存して TimeBudgetExceeded を送出する
    deadline: Optional[float] = None

    def __post_init__(self):
        # Phase に応じた有効化制御
        if self.phase == "B":
            self.basal_ganglia_enabled = True
        elif self.phase == "C":
            self.basal_ganglia_enabled = True
            self.cerebellum_enabled = True


class WikiText103Dataset(Dataset):
    """WikiText-103 実データセット"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.max_length = max_length
        self.tokenizer = tokenizer

        # トークン化
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'labels': self.encodings['input_ids'][idx].masked_fill(self.encodings['attention_mask'][idx] == 0, -100),
            'attention_mask': self.encodings['attention_mask'][idx]
        }


class HippocampusHead(nn.Module):
    """海馬モジュール（Pattern Completion）"""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.pattern_separator = nn.Linear(hidden_size, hidden_size)
        self.pattern_completer = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Pattern separation
        separated = self.pattern_separator(hidden_states)
        separated = torch.relu(separated) * 0.1

        # Pattern completion
        completed = self.pattern_completer(separated)
        completed = torch.tanh(completed)

        # Output logits
        logits = self.output_proj(completed)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': completed
        }


class BasalGangliaHead(nn.Module):
    """基底核モジュール（Actor-Critic）"""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, vocab_size)
        )

    def forward(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        batch_size, seq_len, hidden_size = hidden_states.shape

        values = self.critic(hidden_states)
        policy_logits = self.actor(hidden_states)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = policy_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, policy_logits.size(-1)), shift_labels.view(-1))

        return {
            'logits': policy_logits,
            'loss': loss,
            'values': values,
            'hidden_states': hidden_states
        }


class CerebellumHead(nn.Module):
    """小脳モジュール（Error Correction）"""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.error_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, vocab_size)
        )

    def forward(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        batch_size, seq_len, hidden_size = hidden_states.shape

        error_correction = self.error_predictor(hidden_states)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_error = error_correction[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_error.view(-1, error_correction.size(-1)), shift_labels.view(-1))

        return {
            'error_correction': error_correction,
            'loss': loss,
            'hidden_states': hidden_states
        }


class BrainInspiredLLMFullScale(nn.Module):
    """フル規模実験用 Brain-Inspired LLM"""

    def __init__(self, config: Step444Config, vocab_size: int = 50257, hidden_size: int = 768, num_layers: int = 12):
        super().__init__()
        self.config = config

        # Backbone（GPT-2 相当）
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(config.max_seq_length, hidden_size)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.1
            ) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Brain modules
        self.hippocampus = HippocampusHead(hidden_size, vocab_size) if config.hippocampus_enabled else None
        self.basal_ganglia = BasalGangliaHead(hidden_size, vocab_size) if config.basal_ganglia_enabled else None
        self.cerebellum = CerebellumHead(hidden_size, vocab_size) if config.cerebellum_enabled else None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        batch_size, seq_len = input_ids.shape

        # Embedding + positional encoding
        hidden = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = hidden + self.pos_embedding(pos_ids)

        # Transformer layers
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        for layer in self.transformer_layers:
            hidden = layer(hidden, src_mask=causal_mask, is_causal=True,
                           src_key_padding_mask=~attention_mask.bool())

        backbone_logits = self.lm_head(hidden)

        # Initialize losses
        losses = {'backbone_loss': 0.0}
        total_loss = 0.0
        backbone_ce = None

        # Backbone loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = backbone_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            backbone_loss = loss_fn(shift_logits.view(-1, backbone_logits.size(-1)), shift_labels.view(-1))
            losses['backbone_loss'] = backbone_loss.item()
            total_loss = backbone_loss
            backbone_ce = backbone_loss

        # Phase A: Hippocampus
        if self.config.hippocampus_enabled and self.hippocampus is not None:
            hippo_out = self.hippocampus(hidden, labels)
            hippo_loss = hippo_out['loss']
            if hippo_loss is not None:
                losses['hippocampus_loss'] = hippo_loss.item()
                total_loss = total_loss + self.config.hippocampus_loss_weight * hippo_loss

        # Phase B: Basal Ganglia
        if self.config.basal_ganglia_enabled and self.basal_ganglia is not None:
            bg_out = self.basal_ganglia(hidden, labels)
            bg_loss = bg_out['loss']
            if bg_loss is not None:
                losses['basal_ganglia_loss'] = bg_loss.item()
                total_loss = total_loss + self.config.basal_ganglia_loss_weight * bg_loss

        # Phase C: Cerebellum
        if self.config.cerebellum_enabled and self.cerebellum is not None:
            cereb_out = self.cerebellum(hidden, labels)
            cereb_loss = cereb_out['loss']
            if cereb_loss is not None:
                losses['cerebellum_loss'] = cereb_loss.item()
                total_loss = total_loss + self.config.cerebellum_loss_weight * cereb_loss

        return {
            'logits': backbone_logits,
            'loss': total_loss,
            'backbone_ce': backbone_ce,
            'losses': losses,
            'hidden_states': hidden
        }


class FullScaleTrainer:
    """フル規模実験用トレーナー"""

    def __init__(self, model: nn.Module, config: Step444Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.scaler = GradScaler() if config.fp16 else None

        self.train_losses = []
        self.val_losses = []
        self.step_count = 0

    def train_step(self, batch: Dict) -> float:
        """Single training step"""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
        attention_mask = attention_mask.to(self.device)

        if self.config.fp16 and self.scaler:
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )
                loss = outputs['loss'] / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (self.step_count + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            loss = outputs['loss'] / self.config.gradient_accumulation_steps
            loss.backward()

            if (self.step_count + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.step_count += 1
        return loss.item() * self.config.gradient_accumulation_steps

    def evaluate(self, eval_loader: DataLoader) -> Dict:
        """全フェーズ共通の backbone lm_head の非重み付き次トークン CE と PPL を返す。

        学習損失 total_loss は有効ヘッド数で項が増えるため、フェーズ間比較に使わない。
        各ヘッドの非重み付き CE は参考値として記録する。
        """
        self.model.eval()
        total_ce = 0.0
        total_tokens = 0
        head_sums: Dict[str, float] = {}
        head_counts: Dict[str, int] = {}

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
                attention_mask = attention_mask.to(self.device)

                if self.config.fp16:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask
                    )

                n_tokens = int((labels[..., 1:] != -100).sum().item())
                if n_tokens == 0:
                    continue
                total_ce += outputs['backbone_ce'].item() * n_tokens
                total_tokens += n_tokens
                for name, value in outputs['losses'].items():
                    if name != 'backbone_loss':
                        head_sums[name] = head_sums.get(name, 0.0) + value * n_tokens
                        head_counts[name] = head_counts.get(name, 0) + n_tokens

        val_ce = total_ce / max(total_tokens, 1)
        return {
            'val_ce': val_ce,
            'val_ppl': float(np.exp(val_ce)),
            'head_ce': {k: head_sums[k] / max(head_counts[k], 1) for k in head_sums},
        }


class TimeBudgetExceeded(Exception):
    """実行時間の上限に達し、チェックポイントを保存して中断したことを表す。"""


def run_full_scale_experiment(config: Step444Config, seed: int) -> Dict:
    """Run full-scale experiment for single phase and seed"""

    logger.info(f"{'='*80}")
    logger.info(f"Phase {config.phase} | Seed {seed} | Full-Scale Experiment")
    logger.info(f"{'='*80}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load WikiText-103 dataset（ダミーデータへの切り替えは結果を無意味にするため行わない）
    if not HAS_DATASETS:
        raise RuntimeError("datasets ライブラリが必要です: pip install datasets")
    logger.info("Loading WikiText-103...")
    train_texts = [t for t in load_dataset(config.dataset_name, config.dataset_config, split='train')['text'] if t.strip()]
    val_texts = [t for t in load_dataset(config.dataset_name, config.dataset_config, split='validation')['text'] if t.strip()]

    # Create datasets
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = WikiText103Dataset(train_texts, tokenizer, max_length=config.max_seq_length)
    val_dataset = WikiText103Dataset(val_texts, tokenizer, max_length=config.max_seq_length)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Model & Trainer
    model = BrainInspiredLLMFullScale(config)
    trainer = FullScaleTrainer(model, config)

    results = {
        'phase': config.phase,
        'seed': seed,
        'metric': 'backbone lm_head の非重み付き次トークン CE（全フェーズ共通）',
        'train_losses': [],
        'val_losses': [],
        'val_ppls': [],
        'val_head_ce': [],
    }

    # 途中再開
    ckpt_path = Path(config.checkpoint_dir) / f"phase{config.phase}_seed{seed}.pt"
    state = {'epoch': 0, 'batch_idx': 0, 'epoch_losses': []}
    if ckpt_path.exists():
        state = load_checkpoint(ckpt_path, trainer, results)
        logger.info(f"Resumed from {ckpt_path}: epoch {state['epoch']+1}, batch {state['batch_idx']}")

    # Training
    for epoch in range(state['epoch'], config.num_epochs):
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}")

        # 再開時に同じ順序でバッチを飛ばせるよう、エポックごとのシャッフルをシードで固定する
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=g)
        resuming = epoch == state['epoch']
        start_batch = state['batch_idx'] if resuming else 0
        epoch_losses = list(state['epoch_losses']) if resuming else []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Phase {config.phase}")

        for batch_idx, batch in pbar:
            if batch_idx < start_batch:
                continue
            loss = trainer.train_step(batch)
            epoch_losses.append(loss)
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            # 勾配累積の途中で保存すると再開時に累積分が失われるため、optimizer step 直後のみ保存する
            if trainer.step_count % config.gradient_accumulation_steps == 0:
                out_of_time = config.deadline is not None and time.time() >= config.deadline
                if out_of_time or (batch_idx + 1) % config.checkpoint_every_steps == 0:
                    save_checkpoint(ckpt_path, trainer, results,
                                    {'epoch': epoch, 'batch_idx': batch_idx + 1, 'epoch_losses': epoch_losses})
                if out_of_time:
                    raise TimeBudgetExceeded(f"Phase {config.phase} seed {seed}: epoch {epoch+1}, batch {batch_idx+1}")

        avg_train_loss = float(np.mean(epoch_losses))
        trainer.train_losses.append(avg_train_loss)
        results['train_losses'].append(avg_train_loss)

        # Evaluate
        ev = trainer.evaluate(val_loader)
        trainer.val_losses.append(ev['val_ce'])
        results['val_losses'].append(ev['val_ce'])
        results['val_ppls'].append(ev['val_ppl'])
        results['val_head_ce'].append(ev['head_ce'])

        logger.info(f"Epoch {epoch+1} | Train(total): {avg_train_loss:.4f} | Val CE: {ev['val_ce']:.4f} | Val PPL: {ev['val_ppl']:.2f}")
        save_checkpoint(ckpt_path, trainer, results, {'epoch': epoch + 1, 'batch_idx': 0, 'epoch_losses': []})

    return results


def save_checkpoint(path: Path, trainer: 'FullScaleTrainer', results: Dict, state: Dict) -> None:
    """一時ファイルに書いてから置き換える（書き込み中のセッション切れで壊れないように）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save({
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'scaler': trainer.scaler.state_dict() if trainer.scaler else None,
        'step_count': trainer.step_count,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'results': results,
        'state': state,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, trainer: 'FullScaleTrainer', results: Dict) -> Dict:
    ckpt = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(ckpt['model'])
    trainer.optimizer.load_state_dict(ckpt['optimizer'])
    if trainer.scaler and ckpt['scaler']:
        trainer.scaler.load_state_dict(ckpt['scaler'])
    trainer.step_count = ckpt['step_count']
    trainer.train_losses = ckpt['train_losses']
    trainer.val_losses = ckpt['val_losses']
    results.update(ckpt['results'])
    torch.set_rng_state(ckpt['torch_rng'])
    if ckpt['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    return ckpt['state']


def main(argv: Optional[List[str]] = None):
    """ステップ44.4 フル規模実験実行。

    Colab のセッション切れに備え、(phase, seed) ごとに結果を保存し、完了済みは飛ばす。
    途中で切れた実行はチェックポイントから再開する。複数セッションに分けて実行できる:
        python colab_step44_full_scale_experiment.py --phases A --seeds 0 \
            --results-dir /content/drive/MyDrive/step44 --checkpoint-dir /content/drive/MyDrive/step44/ckpt
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phases', nargs='+', default=['A', 'B', 'C'], choices=['A', 'B', 'C'])
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--results-dir', default=None)
    parser.add_argument('--checkpoint-dir', default=None)
    parser.add_argument('--max-hours', type=float, default=None,
                        help='この時間を超えたらチェックポイントを保存して正常終了する（Kaggle の実行時間上限対策）')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info(f"{'='*80}")
    logger.info("STEP 44.4: Full-Scale Experiment (WikiText-103 × Phase A/B/C)")
    logger.info(f"{'='*80}")

    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    try:
        _run_phases(args, deadline)
    except TimeBudgetExceeded as e:
        logger.info(f"時間の上限に達したため中断しました（{e}）。同じコマンドを再実行すると続きから再開します。")

    # 完了済みの実行をまとめる（未完了の組み合わせは含まれない）
    results_dir = Path(args.results_dir or Step444Config().results_dir)
    summary: Dict[str, Dict] = {}
    for f in sorted(results_dir.glob('phase*_seed*.json')):
        r = json.loads(f.read_text())
        ppls = summary.setdefault(r['phase'], {'final_val_ppl': {}})['final_val_ppl']
        ppls[str(r['seed'])] = r['val_ppls'][-1]
    for ph in summary.values():
        v = list(ph['final_val_ppl'].values())
        ph['mean'], ph['std'], ph['n'] = float(np.mean(v)), float(np.std(v)), len(v)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary: {json.dumps(summary)}")


def _run_phases(args, deadline: Optional[float]) -> None:
    for phase in args.phases:
        config = Step444Config(phase=phase, deadline=deadline)
        if args.results_dir:
            config.results_dir = args.results_dir
        config.checkpoint_dir = args.checkpoint_dir or str(Path(config.results_dir) / 'checkpoints')
        output_dir = Path(config.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for seed in (args.seeds if args.seeds is not None else range(config.num_seeds)):
            run_file = output_dir / f'phase{phase}_seed{seed}.json'
            if run_file.exists():
                logger.info(f"Skip Phase {phase} seed {seed}: {run_file} exists")
                continue
            result = run_full_scale_experiment(config, seed)
            result['config'] = asdict(config)
            with open(run_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved: {run_file}")
            # 完了した実行のチェックポイント（数GB）は不要なので消す（Kaggle の出力容量対策）
            (Path(config.checkpoint_dir) / f"phase{phase}_seed{seed}.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
