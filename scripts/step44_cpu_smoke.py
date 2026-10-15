"""ステップ44.4 修正済みスクリプトの CPU 縮小予備実行（配線確認用。本実験の代替ではない）。

WikiText-2 の一部・小型モデル（192次元・2層）で A/B/C を同条件で学習し、共通指標（backbone val PPL）と
基底核 critic 損失の推移を記録する。
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_step44_full_scale_experiment import (  # noqa: E402
    BrainInspiredLLMFullScale, FullScaleTrainer, Step444Config, WikiText103Dataset)
from datasets import load_dataset  # noqa: E402
from transformers import GPT2Tokenizer  # noqa: E402

STEPS, N_TRAIN, N_VAL, T = int(sys.argv[1]) if len(sys.argv) > 1 else 150, 3000, 200, 64

tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
tr = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")["text"] if t.strip()][:N_TRAIN]
va = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="validation")["text"] if t.strip()][:N_VAL]
train_ds, val_ds = WikiText103Dataset(tr, tok, T), WikiText103Dataset(va, tok, T)

out = {}
for phase in "ABC":
    torch.manual_seed(0)
    cfg = Step444Config(phase=phase, device="cpu", fp16=False, max_seq_length=T, batch_size=8,
                        learning_rate=1e-3, gradient_accumulation_steps=1)
    model = BrainInspiredLLMFullScale(cfg, vocab_size=len(tok), hidden_size=192, num_layers=2)
    model.transformer_layers = torch.nn.ModuleList([
        torch.nn.TransformerEncoderLayer(192, 4, 768, batch_first=True, dropout=0.1) for _ in range(2)])
    trainer = FullScaleTrainer(model, cfg)
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, generator=g)
    step, critic = 0, []
    while step < STEPS:
        for b in loader:
            trainer.train_step(b)
            if phase != "A" and step % 50 == 0:
                with torch.no_grad():
                    critic.append(model(**b)["losses"].get("bg_critic_loss"))
            step += 1
            if step >= STEPS:
                break
    ev = trainer.evaluate(DataLoader(val_ds, batch_size=8))
    out[phase] = {"eval": {k: v for k, v in ev.items() if isinstance(v, (int, float, dict))}, "bg_critic_loss": critic}
    print(phase, json.dumps(out[phase]), flush=True)

dst = Path("results/step44_cpu_smoke")
dst.mkdir(parents=True, exist_ok=True)
(dst / "results.json").write_text(json.dumps({"steps": STEPS, "note": "配線確認用の縮小予備実行", **out}, indent=2, ensure_ascii=False))
