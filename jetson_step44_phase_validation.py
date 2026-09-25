#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップ44.3: 段階的統合検証（Jetson Nano / ローカル GPU 版）

目的:
- 海馬 → 基底核 → 小脳 の段階的統合
- Phase A/B/C の各段階での学習安定性確認
- 小規模データセットでの動作確認（Jetson GPU で実行可能）

実行環境:
- Jetson Nano (2GB/4GB VRAM)
- またはローカル GPU (任意)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

logger = logging.getLogger(__name__)


@dataclass
class Step443Config:
    """ステップ44.3 段階的統合検証設定"""

    # 実験制御
    phase: str = "A"  # "A", "B", or "C"
    hippocampus_enabled: bool = True
    basal_ganglia_enabled: bool = False  # Phase B/C で有効
    cerebellum_enabled: bool = False  # Phase C で有効

    # データセット（小規模）
    num_train_samples: int = 1000  # Jetson 対応
    num_val_samples: int = 100
    seq_length: int = 128  # Jetson メモリ対応
    vocab_size: int = 50257

    # 学習設定
    batch_size: int = 8  # Jetson 対応
    learning_rate: float = 5e-5
    num_epochs: int = 3  # クイック検証
    num_seeds: int = 1   # Jetson での速度優先

    # モジュール Loss 重み（最適化版：重み付けを大幅削減）
    hippocampus_loss_weight: float = 0.05  # 0.2 → 0.05
    basal_ganglia_loss_weight: float = 0.05  # 0.2 → 0.05
    cerebellum_loss_weight: float = 0.02  # 0.1 → 0.02

    # リソース
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True if torch.cuda.is_available() else False
    max_grad_norm: float = 1.0

    # 出力
    results_dir: str = "results/step44_phase_validation"

    def __post_init__(self):
        # Phase に応じた有効化制御
        if self.phase == "B":
            self.basal_ganglia_enabled = True
        elif self.phase == "C":
            self.basal_ganglia_enabled = True
            self.cerebellum_enabled = True


class DummyLLMDataset(Dataset):
    """Jetson 検証用ダミーデータセット"""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': torch.ones(self.seq_length)
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
        """
        hidden_states: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Pattern separation (sparse representation)
        separated = self.pattern_separator(hidden_states)
        separated = torch.relu(separated) * 0.1  # Sparsity-inducing activation

        # Pattern completion (associative retrieval)
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
        """
        Actor-Critic policy logits
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Critic: state value
        values = self.critic(hidden_states)  # (batch, seq_len, 1)

        # Actor: policy logits
        policy_logits = self.actor(hidden_states)  # (batch, seq_len, vocab_size)

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
        """
        Error correction signal
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Predict error correction
        error_correction = self.error_predictor(hidden_states)

        loss = None
        if labels is not None:
            # Error correction loss
            loss_fn = nn.CrossEntropyLoss()
            shift_error = error_correction[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_error.view(-1, error_correction.size(-1)), shift_labels.view(-1))

        return {
            'error_correction': error_correction,
            'loss': loss,
            'hidden_states': hidden_states
        }


class BrainInspiredLLMPhase(nn.Module):
    """段階的統合 LLM"""

    def __init__(self, config: Step443Config, vocab_size: int = 50257, hidden_size: int = 256):
        super().__init__()
        self.config = config

        # Backbone (simplified)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=512,
            batch_first=True
        )
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
        """
        Forward pass with staged module integration
        """
        batch_size, seq_len = input_ids.shape

        # Backbone
        hidden = self.embedding(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.transformer(hidden, src_key_padding_mask=~attention_mask.bool())
        backbone_logits = self.lm_head(hidden)

        # Initialize losses
        losses = {'backbone_loss': 0.0}
        total_loss = 0.0

        # Backbone loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = backbone_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            backbone_loss = loss_fn(shift_logits.view(-1, backbone_logits.size(-1)), shift_labels.view(-1))
            losses['backbone_loss'] = backbone_loss.item()
            total_loss = backbone_loss

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
            'losses': losses,
            'hidden_states': hidden
        }


class PhaseTrainer:
    """各フェーズの学習"""

    def __init__(self, model: nn.Module, config: Step443Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None) -> Dict:
        """1 epoch の学習"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Phase {self.config.phase}")

        for batch_idx, batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
            attention_mask = attention_mask.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            loss = outputs['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)

        return {'train_loss': avg_loss}

    def evaluate(self, eval_loader: DataLoader) -> float:
        """評価"""
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )
                loss = outputs['loss']
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)
        self.val_losses.append(avg_loss)
        self.model.train()
        return avg_loss


def run_phase_validation(config: Step443Config, seed: int) -> Dict:
    """単一 Phase の検証実行"""

    logger.info(f"{'='*80}")
    logger.info(f"Phase {config.phase} | Seed {seed}")
    logger.info(f"{'='*80}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Dataset & DataLoader
    train_dataset = DummyLLMDataset(config.num_train_samples, config.seq_length, config.vocab_size)
    val_dataset = DummyLLMDataset(config.num_val_samples, config.seq_length, config.vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
    model = BrainInspiredLLMPhase(config)
    trainer = PhaseTrainer(model, config)

    results = {
        'phase': config.phase,
        'seed': seed,
        'train_losses': [],
        'val_losses': []
    }

    # Training
    for epoch in range(config.num_epochs):
        epoch_result = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)

        results['train_losses'].append(epoch_result['train_loss'])
        results['val_losses'].append(val_loss)

        logger.info(f"Epoch {epoch+1}/{config.num_epochs} | Train: {epoch_result['train_loss']:.4f} | Val: {val_loss:.4f}")

    return results


def main():
    """ステップ44.3 実行"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*80}")
    logger.info("STEP 44.3: Phase-wise Module Validation (Jetson / Local GPU)")
    logger.info(f"{'='*80}")

    # 各 Phase を順序実行
    phases = ['A', 'B', 'C']
    all_results = {}

    for phase in phases:
        config = Step443Config(phase=phase)
        logger.info(f"\n>>> Starting Phase {phase}")

        output_dir = Path(config.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        phase_results = []
        for seed in range(config.num_seeds):
            result = run_phase_validation(config, seed)
            phase_results.append(result)

        all_results[phase] = phase_results

        # 中間結果保存
        phase_file = output_dir / f'step44_phase_{phase}_results.json'
        with open(phase_file, 'w') as f:
            json.dump(phase_results, f, indent=2)
        logger.info(f"Phase {phase} results saved: {phase_file}")

    # 最終結果保存
    final_file = Path(config.results_dir) / 'step44_phase_validation_results.json'
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"{'='*80}")
    logger.info("STEP 44.3: Complete OK")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
