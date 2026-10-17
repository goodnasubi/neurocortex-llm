"""基底核 critic の収束対策の小規模比較（CPU）。12.7節の残課題「critic の収束速度」。

条件: baseline（td_sg）/ value_init（価値出力の初期バイアス）/ avg_reward（報酬の EMA 基準化）。
Phase B の小型モデル（48次元・1層、WikiText-2 の一部、系列長32）を同じシード・同じデータ順序で学習し、
各条件の検証バッチで critic の説明分散 EV = 1 − critic_loss / Var(TD ターゲット) と critic 損失を記録する。
報酬の基準化でターゲットの大きさが変わるため、損失の絶対値ではなく EV で比較する。
backbone の val PPL も記録する（critic は勾配停止のため、条件間で一致するはず）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_step44_full_scale_experiment import (  # noqa: E402
    BrainInspiredLLMFullScale, FullScaleTrainer, Step444Config, WikiText103Dataset)

CONDITIONS = {
    'baseline': dict(bg_value_init=False, bg_reward_baseline='none'),
    'value_init': dict(bg_value_init=True, bg_reward_baseline='none'),
    'avg_reward': dict(bg_value_init=False, bg_reward_baseline='ema'),
}


def critic_eval(model, loader):
    model.eval()
    cl, tv = [], []
    with torch.no_grad():
        for b in loader:
            m = model.module if hasattr(model, 'module') else model
            out = m.basal_ganglia(m_hidden(m, b), b['labels'])
            cl.append(out['critic_loss'].item())
            tv.append(out['td_target_var'].item())
    model.train()
    cl, tv = float(np.mean(cl)), float(np.mean(tv))
    return {'critic_loss': cl, 'target_var': tv, 'ev': 1 - cl / tv if tv > 0 else float('nan')}


def m_hidden(m, b):
    return m(input_ids=b['input_ids'], labels=b['labels'], attention_mask=b['attention_mask'])['hidden_states']


def run(cond, seed, train_ds, val_ds, vocab, steps, eval_every):
    torch.manual_seed(seed)
    cfg = Step444Config(phase='B', device='cpu', fp16=False, max_seq_length=32, batch_size=8,
                        learning_rate=1e-3, gradient_accumulation_steps=1, **CONDITIONS[cond])
    model = BrainInspiredLLMFullScale(cfg, vocab_size=vocab, hidden_size=48, num_layers=1)
    trainer = FullScaleTrainer(model, cfg)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, generator=g, drop_last=True)
    val = DataLoader(val_ds, batch_size=8)
    curve, step = [], 0
    curve.append({'step': 0, **critic_eval(model, val)})
    while step < steps:
        for b in loader:
            trainer.train_step(b)
            step += 1
            if step % eval_every == 0:
                curve.append({'step': step, **critic_eval(model, val)})
            if step >= steps:
                break
    ppl = trainer.evaluate(val)['val_ppl']
    return {'curve': curve, 'val_ppl': ppl}


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    seeds = [0, 1, 2]
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    tr = [t for t in load_dataset('Salesforce/wikitext', 'wikitext-2-v1', split='train')['text'] if t.strip()][:3000]
    va = [t for t in load_dataset('Salesforce/wikitext', 'wikitext-2-v1', split='validation')['text'] if t.strip()][:80]
    train_ds, val_ds = WikiText103Dataset(tr, tok, 32), WikiText103Dataset(va, tok, 32)
    res = {}
    for cond in CONDITIONS:
        for s in seeds:
            r = run(cond, s, train_ds, val_ds, len(tok), steps, max(steps // 8, 1))
            res.setdefault(cond, {})[str(s)] = r
            print(cond, s, json.dumps({k: round(v, 3) for k, v in r['curve'][-1].items()}), round(r['val_ppl'], 2), flush=True)
    summary = {}
    for cond, rs in res.items():
        evs = [r['curve'][-1]['ev'] for r in rs.values()]
        mid = [r['curve'][len(r['curve']) // 2]['ev'] for r in rs.values()]
        summary[cond] = {'final_ev_mean': float(np.mean(evs)), 'final_ev_std': float(np.std(evs)),
                         'mid_ev_mean': float(np.mean(mid)),
                         'final_critic_loss_mean': float(np.mean([r['curve'][-1]['critic_loss'] for r in rs.values()])),
                         'val_ppl_mean': float(np.mean([r['val_ppl'] for r in rs.values()]))}
    out = Path('results/step44_bg_critic_convergence')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps({'steps': steps, 'seeds': seeds, 'summary': summary, 'runs': res},
                                                 indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
