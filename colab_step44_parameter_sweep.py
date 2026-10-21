"""ステップ44.4 モジュール損失重みのパラメータスイープ（再構築版）

旧スイープ（results/step44_parameter_sweep/loss_weight_sweep_results.json）はコードが未コミットで、
Phase A の値が 11.53×(1−w_h) に厳密に一致するなど学習出力ではないため破棄した。本スクリプトはその置き換え。

- 格子: w_h ∈ {0.01,0.02,0.03,0.05} × w_bg ∈ {0.01,0.02,0.03,0.05} × w_cb ∈ {0.005,0.01,0.015,0.02}（旧スイープと同じ）
- Phase A は w_h のみ、B は (w_h, w_bg)、C は3つすべてに依存するので、必要な組み合わせだけ学習する
  （A 4 + B 16 + C 64 = 84 実行 / シード）。
- 指標: 全フェーズ共通の backbone lm_head の非重み付き val CE / PPL（colab_step44_full_scale_experiment の evaluate）。
  学習損失（重み付き和）は比較に使わない。
- 比較は同じ w_h の Phase A とのペア差 ΔPPL = PPL(B or C) − PPL(A)（同一シード・同一データ順序）。
- 基底核 critic は td_sg（フル規模スクリプトの既定）。
- (phase, 重み, seed) ごとに JSON を保存し、完了済みは飛ばす（Colab の分割実行用）。

使い方（Colab, 予算は --steps で指定。既定はフル規模の1/10程度の短縮学習）:
    python colab_step44_parameter_sweep.py --results-dir /content/drive/MyDrive/step44_sweep --seeds 0 1 2
CPU での配線確認:
    python colab_step44_parameter_sweep.py --preset smoke
"""
import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from colab_step44_full_scale_experiment import (
    BrainInspiredLLMFullScale, FullScaleTrainer, Step444Config, WikiText103Dataset)
from neurocortex.run_env import env_info  # noqa: E402  （上の import が src をパスに追加する）

logger = logging.getLogger(__name__)

W_H = [0.01, 0.02, 0.03, 0.05]
W_BG = [0.01, 0.02, 0.03, 0.05]
W_CB = [0.005, 0.01, 0.015, 0.02]

PRESETS = {
    # Colab T4 想定。WikiText-103、GPT-2 small 相当。
    'colab': dict(dataset_config='wikitext-103-v1', hidden_size=768, num_layers=12, seq_len=256,
                  batch_size=32, steps=5000, n_train=None, n_val=None, lr=5e-5),
    # CPU での配線確認用（効果の判定には使わない）
    'smoke': dict(dataset_config='wikitext-2-v1', hidden_size=48, num_layers=1, seq_len=32,
                  batch_size=8, steps=20, n_train=400, n_val=40, lr=1e-3),
}


def run_key(phase: str, w: Tuple[float, ...], seed: int) -> str:
    return f"phase{phase}_" + "_".join(f"{x:g}" for x in w) + f"_seed{seed}"


def required_runs() -> List[Tuple[str, Tuple[float, ...]]]:
    """学習が必要な (phase, 重み) の組。重複する設定は1回だけ学習する。"""
    runs = [('A', (h,)) for h in W_H]
    runs += [('B', (h, b)) for h, b in itertools.product(W_H, W_BG)]
    runs += [('C', (h, b, c)) for h, b, c in itertools.product(W_H, W_BG, W_CB)]
    return runs


def make_config(phase: str, w: Tuple[float, ...], p: Dict) -> Step444Config:
    cfg = Step444Config(phase=phase, max_seq_length=p['seq_len'], batch_size=p['batch_size'],
                        learning_rate=p['lr'], gradient_accumulation_steps=1)
    cfg.hippocampus_loss_weight = w[0]
    if len(w) > 1:
        cfg.basal_ganglia_loss_weight = w[1]
    if len(w) > 2:
        cfg.cerebellum_loss_weight = w[2]
    return cfg


def train_eval(phase: str, w: Tuple[float, ...], seed: int, p: Dict, train_ds, val_ds, vocab_size: int,
               device: Optional[str] = None) -> Dict:
    cfg = make_config(phase, w, p)
    if device is not None:
        cfg.device = device
        cfg.fp16 = device == 'cuda'
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BrainInspiredLLMFullScale(cfg, vocab_size=vocab_size, hidden_size=p['hidden_size'],
                                      num_layers=p['num_layers'])
    trainer = FullScaleTrainer(model, cfg)
    # データ順序はシードのみで決める（同一シードのフェーズ間でペア比較するため）
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=p['batch_size'], shuffle=True, generator=g, drop_last=True)
    step, losses = 0, []
    while step < p['steps']:
        for batch in loader:
            losses.append(trainer.train_step(batch))
            step += 1
            if step >= p['steps']:
                break
    ev = trainer.evaluate(DataLoader(val_ds, batch_size=p['batch_size']))
    return {'phase': phase, 'weights': list(w), 'seed': seed, 'steps': step,
            'final_train_loss': float(np.mean(losses[-50:])), **ev}


def summarize(results_dir: Path) -> Dict:
    """同じ w_h・シードの Phase A とのペア差 ΔPPL を集計する。"""
    runs = [json.loads(f.read_text()) for f in sorted(results_dir.glob('phase*_seed*.json'))]
    a = {(r['weights'][0], r['seed']): r['val_ppl'] for r in runs if r['phase'] == 'A'}
    rows: Dict[str, Dict] = {}
    for r in runs:
        if r['phase'] == 'A' or (r['weights'][0], r['seed']) not in a:
            continue
        key = f"{r['phase']}:" + ",".join(f"{x:g}" for x in r['weights'])
        rows.setdefault(key, {'phase': r['phase'], 'weights': r['weights'], 'delta_ppl': {}, 'val_ppl': {}})
        rows[key]['delta_ppl'][str(r['seed'])] = r['val_ppl'] - a[(r['weights'][0], r['seed'])]
        rows[key]['val_ppl'][str(r['seed'])] = r['val_ppl']
    for row in rows.values():
        d = list(row['delta_ppl'].values())
        row['delta_mean'], row['delta_std'], row['n'] = float(np.mean(d)), float(np.std(d)), len(d)
    phase_a = {}
    for (h, s), v in a.items():
        phase_a.setdefault(f"{h:g}", {})[str(s)] = v
    ranked = sorted(rows.values(), key=lambda r: r['delta_mean'])
    return {'metric': 'backbone の非重み付き val PPL。delta は同一 w_h・同一シードの Phase A との差（負が改善）',
            'phase_a_val_ppl': phase_a, 'rows': ranked}


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='colab', choices=list(PRESETS))
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--phases', nargs='+', default=['A', 'B', 'C'], choices=['A', 'B', 'C'])
    ap.add_argument('--results-dir', default='results/step44_parameter_sweep_v2')
    ap.add_argument('--device', default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    p = dict(PRESETS[args.preset])
    if args.steps is not None:
        p['steps'] = args.steps
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token

    def texts(split, n):
        t = [x for x in load_dataset('Salesforce/wikitext', p['dataset_config'], split=split)['text'] if x.strip()]
        return t[:n] if n else t
    train_ds = WikiText103Dataset(texts('train', p['n_train']), tok, p['seq_len'])
    val_ds = WikiText103Dataset(texts('validation', p['n_val']), tok, p['seq_len'])

    for seed in args.seeds:
        for phase, w in required_runs():
            if phase not in args.phases:
                continue
            f = out / f"{run_key(phase, w, seed)}.json"
            if f.exists():
                continue
            r = train_eval(phase, w, seed, p, train_ds, val_ds, len(tok), args.device)
            r['preset'] = args.preset
            r['budget'] = p
            r['environment'] = env_info()
            f.write_text(json.dumps(r, indent=2))
            logger.info(f"{f.name}: val_ppl={r['val_ppl']:.3f}")

    summary = summarize(out)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"Summary written: {out / 'summary.json'}")


if __name__ == '__main__':
    main()
