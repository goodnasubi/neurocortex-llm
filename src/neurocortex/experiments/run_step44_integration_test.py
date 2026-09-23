#!/usr/bin/env python3
"""
ステップ44.1: 脳型LLM統合フレームワークの初期テスト

目的:
- LLMバックボーン（Llama 3B相当）のロード確認
- 海馬・基底核・小脳モジュールの統合確認
- forward pass の正常動作確認
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import logging
from neurocortex.integration_llm import create_brain_llm, BrainModuleConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_step44_integration():
    """ステップ44.1 統合テスト"""

    logger.info("=" * 80)
    logger.info("STEP 44.1: Brain-Inspired LLM Integration Test")
    logger.info("=" * 80)

    # 設定（初期段階は軽量モデルで テスト）
    config = BrainModuleConfig(
        # 注: meta-llama/Llama-2-3b-hf は認証が必要なため、テストでは小規模モデルを使用
        # 本実装では gpt2-medium などで代替可能
        backbone_model_id="gpt2-medium",  # 355M params, テスト用
        hippocampus_enabled=True,
        basal_ganglia_enabled=True,
        cerebellum_enabled=True,
        stdp_enabled=True
    )

    logger.info(f"Config:")
    logger.info(f"  Backbone: {config.backbone_model_id}")
    logger.info(f"  Modules: Hippo={config.hippocampus_enabled}, BG={config.basal_ganglia_enabled}, Cereb={config.cerebellum_enabled}")

    # モデル作成
    try:
        logger.info("Creating Brain-Inspired LLM...")
        model = create_brain_llm(config)
        logger.info("✓ Model created successfully")
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False

    # Device設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"✓ Model loaded on device: {device}")

    # 簡単な forward pass テスト
    try:
        logger.info("Running forward pass test...")

        # ダミー入力
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.backbone_model_id == "gpt2-medium" and 50257 or 32000,
                                  (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = input_ids.clone()

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Output shape: {outputs['logits'].shape}")
        logger.info(f"  Losses: {list(outputs['losses'].keys())}")

        # 損失チェック
        for loss_name, loss_val in outputs['losses'].items():
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                logger.warning(f"  ✗ {loss_name} is NaN/Inf: {loss_val}")
                return False
            else:
                logger.info(f"  ✓ {loss_name}: {loss_val:.6f}")

    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("=" * 80)
    logger.info("STEP 44.1: All tests passed ✓")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    success = test_step44_integration()
    sys.exit(0 if success else 1)
