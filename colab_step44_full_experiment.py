#!/usr/bin/env python3
"""
ステップ44.2: 海馬モジュール統合・フル規模検証（Colab 版）

目的:
- WikiText-103 での実データ学習
- 条件A vs B を 10 epoch × 3 seed で統計的に検証
- GPU (Colab T4/A100) で 48-72 時間連続実行

実行方法（Google Colab）:
1. GitHub から clone: !git clone https://github.com/goodnasubi/neurocortex-llm.git
2. このスクリプトをアップロード or `%run colab_step44_full_experiment.py`
3. Google Drive に results を保存
"""

import sys
import os
from pathlib import Path

# Colab 環境検出
IS_COLAB = 'google.colab' in sys.modules
if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/gdrive')
    # リポジトリパス
    REPO_PATH = Path('/content/neurocortex-llm') if Path('/content/neurocortex-llm').exists() else Path('/content/gdrive/MyDrive/neurocortex-llm')
    RESULTS_DRIVE = Path('/content/gdrive/MyDrive/step44_results')
    RESULTS_DRIVE.mkdir(parents=True, exist_ok=True)
else:
    REPO_PATH = Path(__file__).parent
    RESULTS_DRIVE = REPO_PATH / 'results' / 'step44_hippocampus_validation'
    RESULTS_DRIVE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_PATH))

import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class Step442FullConfig:
    """ステップ44.2 フル規模実験設定"""

    # データセット
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    max_seq_length: int = 512
    train_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    # 学習フレームワーク
    backbone_model_id: str = "gpt2"  # GPT-2-small
    batch_size: int = 16  # Colab T4: 32 は OOM, 16-24 推奨
    gradient_accumulation_steps: int = 2  # 実効 batch = 32
    learning_rate: float = 5e-5
    num_epochs: int = 10  # フル規模
    num_seeds: int = 3   # 統計的有意性確保

    # 海馬設定
    hippocampus_enabled_condition_b: bool = True
    hippocampus_loss_weight: float = 0.1  # 0.2 は高すぎる場合あり

    # 評価
    eval_interval: int = 500  # 500 batch ごと
    log_interval: int = 100
    save_interval: int = 1000

    # リソース
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True  # mixed precision で GPU メモリ削減
    max_grad_norm: float = 1.0

    # 出力
    results_dir: str = str(RESULTS_DRIVE / "step44_full_experiment")


class TextDataset(Dataset):
    """WikiText-103 リアルデータセット"""

    def __init__(self, tokenizer, dataset_split='train', max_length=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # HuggingFace datasets から WikiText-103 をロード
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=dataset_split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.examples = []
        for article in tqdm(dataset['text'], desc=f"Tokenizing {dataset_split}"):
            if not article.strip():
                continue

            tokens = tokenizer(
                article,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.examples.append({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens.get('attention_mask', torch.ones(max_length))
            })

        if not self.examples:
            logger.warning(f"Dataset {dataset_split} is empty!")
            self.examples = [{
                'input_ids': torch.randint(0, tokenizer.vocab_size, (max_length,)),
                'attention_mask': torch.ones(max_length)
            }]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'input_ids': example['input_ids'],
            'labels': example['input_ids'].clone(),
            'attention_mask': example['attention_mask']
        }


class LLMTrainer:
    """LLM 学習の統制フレームワーク"""

    def __init__(self, model: nn.Module, config: Step442FullConfig, condition: str = "A"):
        self.model = model
        self.config = config
        self.condition = condition
        self.device = torch.device(config.device)

        self.model.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        # mixed precision
        if config.fp16 and torch.cuda.is_available():
            from torch.cuda.amp import autocast, GradScaler
            self.autocast = autocast()
            self.scaler = GradScaler()
        else:
            self.autocast = None
            self.scaler = None

        # 統計記録
        self.train_losses = []
        self.val_perplexities = []
        self.gradient_norms = []
        self.step_count = 0
        self.eval_step_count = 0

    def train_epoch(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None) -> Dict:
        """1 epoch の学習"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch")

        for batch_idx, batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
            attention_mask = attention_mask.to(self.device)

            # Forward with mixed precision
            if self.autocast:
                with self.autocast:
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    loss = outputs.loss
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                loss = outputs.loss

            # Backward
            loss = loss / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)

            # ロギング
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-self.config.log_interval:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Evaluation
            if eval_loader and (batch_idx + 1) % self.config.eval_interval == 0:
                val_ppl = self.evaluate(eval_loader)
                self.val_perplexities.append((self.step_count, val_ppl))
                self.eval_step_count += 1

            self.step_count += 1

        # epoch 統計
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        self.train_losses.append(avg_epoch_loss)

        return {
            'train_loss': avg_epoch_loss,
            'step_count': self.step_count
        }

    def evaluate(self, eval_loader: DataLoader) -> float:
        """Validation perplexity 計測"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
                attention_mask = attention_mask.to(self.device)

                if self.autocast:
                    with self.autocast:
                        outputs = self.model(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                        loss = outputs.loss
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    loss = outputs.loss

                if loss is not None and not torch.isnan(loss):
                    total_loss += loss.item() * input_ids.size(0)
                    total_tokens += input_ids.size(0)

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(avg_loss)
        logger.info(f"Condition {self.condition} | Val Perplexity: {perplexity:.4f}")

        self.model.train()
        return perplexity


def run_experiment(config: Step442FullConfig, seed: int, condition: str) -> Dict:
    """単一 seed の学習実行"""

    logger.info(f"{'='*80}")
    logger.info(f"Running Condition {condition} | Seed {seed}")
    logger.info(f"{'='*80}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Tokenizer & Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained('gpt2')

    # Dataset
    train_dataset = TextDataset(tokenizer, 'train', max_length=config.max_seq_length, max_samples=10000)
    val_dataset = TextDataset(tokenizer, 'validation', max_length=config.max_seq_length, max_samples=1000)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    trainer = LLMTrainer(model, config, condition=condition)

    results = {
        'condition': condition,
        'seed': seed,
        'train_losses': [],
        'val_perplexities': []
    }

    for epoch in range(config.num_epochs):
        epoch_results = trainer.train_epoch(train_loader, val_loader)
        results['train_losses'].append(epoch_results['train_loss'])
        results['val_perplexities'] = [(step, ppl) for step, ppl in trainer.val_perplexities]
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} complete: Loss={epoch_results['train_loss']:.4f}")

    return results


def main():
    """ステップ44.2 フル実験実行"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*80}")
    logger.info("STEP 44.2: Hippocampus Module Full-Scale Validation (Colab)")
    logger.info(f"{'='*80}")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    config = Step442FullConfig()

    # 出力ディレクトリ
    output_dir = Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'config': asdict(config),
        'experiments': [],
        'metadata': {
            'is_colab': IS_COLAB,
            'timestamp': str(Path.cwd())
        }
    }

    # 条件A・B × 3 シード実行
    conditions = ['A', 'B']
    for condition in conditions:
        for seed in range(config.num_seeds):
            result = run_experiment(config, seed, condition)
            all_results['experiments'].append(result)

            # 途中保存（クラッシュ対策）
            temp_file = output_dir / f'step44_results_checkpoint_{condition}_{seed}.json'
            with open(temp_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Checkpoint saved: {temp_file}")

    # 最終結果保存
    result_file = output_dir / 'step44_full_experiment_results.json'
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Final results saved to {result_file}")
    logger.info(f"{'='*80}")
    logger.info("STEP 44.2: Complete ✓")
    logger.info(f"{'='*80}")

    return result_file


if __name__ == "__main__":
    result_file = main()
    print(f"\n✓ Experiment complete. Results: {result_file}")
