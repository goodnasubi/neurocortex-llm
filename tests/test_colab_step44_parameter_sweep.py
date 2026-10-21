import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colab_step44_parameter_sweep import make_config, required_runs, run_key, summarize, train_eval  # noqa: E402

V = 40
P = dict(hidden_size=24, num_layers=1, seq_len=12, batch_size=4, steps=3, lr=1e-3)


class _DS(torch.utils.data.Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randint(0, V, (n, P['seq_len']), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {'input_ids': self.x[i], 'labels': self.x[i], 'attention_mask': torch.ones_like(self.x[i])}


def test_grid_covers_old_sweep_without_duplicates():
    runs = required_runs()
    assert len(runs) == 4 + 16 + 64
    assert len(set(runs)) == len(runs)


def test_weights_applied_per_phase():
    c = make_config('C', (0.03, 0.05, 0.015), P)
    assert (c.hippocampus_loss_weight, c.basal_ganglia_loss_weight, c.cerebellum_loss_weight) == (0.03, 0.05, 0.015)
    assert c.basal_ganglia_enabled and c.cerebellum_enabled
    a = make_config('A', (0.02,), P)
    assert a.hippocampus_loss_weight == 0.02 and not a.basal_ganglia_enabled


def test_train_eval_is_deterministic_and_uses_common_metric():
    tr, va = _DS(16, 0), _DS(8, 1)
    r1 = train_eval('B', (0.01, 0.02), 0, P, tr, va, V, device='cpu')
    r2 = train_eval('B', (0.01, 0.02), 0, P, tr, va, V, device='cpu')
    assert r1['val_ppl'] == r2['val_ppl']
    assert abs(r1['val_ppl'] - torch.tensor(r1['val_ce']).exp().item()) < 1e-3


def test_summary_pairs_with_phase_a_same_wh_and_seed(tmp_path):
    def put(phase, w, seed, ppl):
        (tmp_path / f"{run_key(phase, w, seed)}.json").write_text(
            json.dumps({'phase': phase, 'weights': list(w), 'seed': seed, 'val_ppl': ppl}))
    put('A', (0.01,), 0, 100.0)
    put('A', (0.02,), 0, 90.0)
    put('B', (0.02, 0.05), 0, 85.0)
    put('C', (0.01, 0.01, 0.005), 0, 103.0)
    s = summarize(tmp_path)
    by = {r['phase']: r for r in s['rows']}
    assert by['B']['delta_mean'] == -5.0
    assert by['C']['delta_mean'] == 3.0
    assert s['rows'][0]['phase'] == 'B'
