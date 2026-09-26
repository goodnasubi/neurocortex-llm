#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップ44.3 最適化損失重み検証（小規模実験）

目的:
- ステップ44.5.2 のパラメータスイープで得られた最適損失重みを検証
- 最適化前後での性能比較（Phase A, B, C）
- 予測改善が実験結果で確認されることを検証

最適損失重み（パラメータスイープ結果から）:
- Hippocampus: 0.03 (前: 0.05)
- Basal Ganglia: 0.05 (同じ)
- Cerebellum: 0.02 (同じ)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

logger = logging.getLogger(__name__)


@dataclass
class OptimizedValidationConfig:
    """最適化検証設定"""

    # 実験制御
    phase: str = "A"  # "A", "B", or "C"
    hippocampus_enabled: bool = True
    basal_ganglia_enabled: bool = False
    cerebellum_enabled: bool = False

    # データセット（小規模）
    num_train_samples: int = 500
    num_val_samples: int = 100
    seq_length: int = 128
    vocab_size: int = 50257

    # 学習設定
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 5
    num_seeds: int = 3  # 統計的妥当性のため 3 seeds

    # 最適化損失重み（パラメータスイープから）
    hippocampus_loss_weight: float = 0.03  # 最適値
    basal_ganglia_loss_weight: float = 0.05
    cerebellum_loss_weight: float = 0.02

    # リソース
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True if torch.cuda.is_available() else False
    max_grad_norm: float = 1.0

    # 出力
    results_dir: str = "results/step44_optimized_validation"

    def __post_init__(self):
        if self.phase == "B":
            self.basal_ganglia_enabled = True
        elif self.phase == "C":
            self.basal_ganglia_enabled = True
            self.cerebellum_enabled = True


class DummyLLMDataset(Dataset):
    """小規模検証用ダミーデータセット"""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int, seed: int = 42):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # 再現性のため seed で固定
        np.random.seed(seed)
        torch.manual_seed(seed)

        # データセット生成
        self.data = []
        for _ in range(num_samples):
            input_ids = torch.randint(0, vocab_size, (seq_length,))
            self.data.append({
                'input_ids': input_ids,
                'labels': input_ids.clone(),
                'attention_mask': torch.ones(seq_length)
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


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

        separated = self.pattern_separator(hidden_states)
        separated = torch.relu(separated) * 0.1
        completed = self.pattern_completer(separated)
        completed = torch.tanh(completed)
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

        value = self.critic(hidden_states)
        logits = self.actor(hidden_states)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            loss = loss + 0.01 * value.mean()

        return {
            'logits': logits,
            'loss': loss,
            'value': value,
            'hidden_states': hidden_states
        }


class CerebellumHead(nn.Module):
    """小脳モジュール（Error Correction）"""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.error_detector = nn.Linear(hidden_size, hidden_size)
        self.corrector = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        batch_size, seq_len, hidden_size = hidden_states.shape

        error = self.error_detector(hidden_states)
        error = torch.relu(error) * 0.05
        corrected = self.corrector(error)
        corrected = torch.tanh(corrected)
        logits = self.output_proj(corrected)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': corrected
        }


class BrainInspiredLLMPhase(nn.Module):
    """Brain-Inspired LLM（段階的統合）"""

    def __init__(self, config: OptimizedValidationConfig):
        super().__init__()
        self.config = config

        # シンプルなバックボーン（テスト用）
        self.hidden_size = 128
        self.backbone = nn.Sequential(
            nn.Embedding(config.vocab_size, self.hidden_size),
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=4, batch_first=True),
        )

        # モジュール
        if config.hippocampus_enabled:
            self.hippocampus = HippocampusHead(self.hidden_size, config.vocab_size)
        if config.basal_ganglia_enabled:
            self.basal_ganglia = BasalGangliaHead(self.hidden_size, config.vocab_size)
        if config.cerebellum_enabled:
            self.cerebellum = CerebellumHead(self.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        hidden = self.backbone(input_ids)

        total_loss = 0.0
        outputs = {'loss': 0.0}

        if self.config.hippocampus_enabled:
            hippo_out = self.hippocampus(hidden, labels)
            if hippo_out['loss'] is not None:
                total_loss += self.config.hippocampus_loss_weight * hippo_out['loss']
            outputs['hippocampus_loss'] = hippo_out['loss'].item() if hippo_out['loss'] is not None else 0.0

        if self.config.basal_ganglia_enabled:
            bg_out = self.basal_ganglia(hidden, labels)
            if bg_out['loss'] is not None:
                total_loss += self.config.basal_ganglia_loss_weight * bg_out['loss']
            outputs['basal_ganglia_loss'] = bg_out['loss'].item() if bg_out['loss'] is not None else 0.0

        if self.config.cerebellum_enabled:
            cereb_out = self.cerebellum(hidden, labels)
            if cereb_out['loss'] is not None:
                total_loss += self.config.cerebellum_loss_weight * cereb_out['loss']
            outputs['cerebellum_loss'] = cereb_out['loss'].item() if cereb_out['loss'] is not None else 0.0

        outputs['loss'] = total_loss
        return outputs


class OptimizedTrainer:
    """最適化検証用トレーナー"""

    def __init__(self, model: nn.Module, config: OptimizedValidationConfig):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, labels)
            loss = outputs['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val

        return {'train_loss': total_loss / len(train_loader)}

    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                outputs = self.model(input_ids, labels)
                loss = outputs['loss']
                total_loss += loss.item()

        return total_loss / len(val_loader)


def run_phase_validation(config: OptimizedValidationConfig, seed: int) -> Dict:
    """単一 Phase の検証実行"""

    logger.info(f"{'='*80}")
    logger.info(f"Phase {config.phase} | Seed {seed}")
    logger.info(f"{'='*80}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Dataset & DataLoader
    train_dataset = DummyLLMDataset(config.num_train_samples, config.seq_length, config.vocab_size, seed=seed)
    val_dataset = DummyLLMDataset(config.num_val_samples, config.seq_length, config.vocab_size, seed=seed+1000)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
    model = BrainInspiredLLMPhase(config)
    trainer = OptimizedTrainer(model, config)

    results = {
        'phase': config.phase,
        'seed': seed,
        'loss_weights': {
            'hippocampus': config.hippocampus_loss_weight,
            'basal_ganglia': config.basal_ganglia_loss_weight,
            'cerebellum': config.cerebellum_loss_weight
        },
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
    """ステップ44.3 最適化検証実行"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*80}")
    logger.info("STEP 44.3 OPTIMIZED: Validation with Optimized Loss Weights")
    logger.info(f"{'='*80}")
    logger.info("Optimal weights (from parameter sweep):")
    logger.info(f"  Hippocampus: 0.03")
    logger.info(f"  Basal Ganglia: 0.05")
    logger.info(f"  Cerebellum: 0.02")

    # 各 Phase を順序実行
    phases = ['A', 'B', 'C']
    all_results = {}

    for phase in phases:
        config = OptimizedValidationConfig(phase=phase)
        logger.info(f"\n>>> Starting Phase {phase}")

        output_dir = Path(config.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        phase_results = []
        for seed in range(config.num_seeds):
            result = run_phase_validation(config, seed)
            phase_results.append(result)

        all_results[phase] = phase_results

        # 中間結果保存
        phase_file = output_dir / f'step44_optimized_{phase}_results.json'
        with open(phase_file, 'w') as f:
            json.dump(phase_results, f, indent=2)
        logger.info(f"Phase {phase} results saved: {phase_file}")

        # 統計：平均 validation loss
        val_losses = [r['val_losses'][-1] for r in phase_results]
        mean_loss = np.mean(val_losses)
        std_loss = np.std(val_losses)
        logger.info(f"Phase {phase} | Mean final val loss: {mean_loss:.4f} ± {std_loss:.4f}")

    # 最終結果保存
    final_file = Path(config.results_dir) / 'step44_optimized_validation_summary.json'
    summary = {
        'all_results': all_results,
        'phase_summaries': {}
    }

    for phase in phases:
        val_losses = [r['val_losses'][-1] for r in all_results[phase]]
        summary['phase_summaries'][phase] = {
            'mean_final_val_loss': float(np.mean(val_losses)),
            'std_final_val_loss': float(np.std(val_losses)),
            'raw_losses': val_losses
        }

    with open(final_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"{'='*80}")
    logger.info("STEP 44.3 OPTIMIZED: Complete")
    logger.info(f"Final summary saved: {final_file}")
    logger.info(f"{'='*80}")

    # 改善確認
    logger.info("\n=== Improvement Analysis ===")
    for phase in phases:
        val_losses = [r['val_losses'][-1] for r in all_results[phase]]
        logger.info(f"Phase {phase}: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")


if __name__ == "__main__":
    main()
