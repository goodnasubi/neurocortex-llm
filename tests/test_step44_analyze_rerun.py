import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step44_analyze_rerun import analyze, load_runs, main, paired_t  # noqa: E402


def _write(d, phase, seed, ppls, critic=None):
    head = [{'bg_critic_loss': c} for c in critic] if critic else [{} for _ in ppls]
    (d / f'phase{phase}_seed{seed}.json').write_text(json.dumps(
        {'phase': phase, 'seed': seed, 'val_ppls': ppls, 'val_head_ce': head}))


def test_paired_t_known_values():
    r = paired_t([1.0, 2.0, 3.0])
    assert r['mean'] == 2.0 and abs(r['std'] - 1.0) < 1e-12
    assert abs(r['t'] - 2.0 * 3 ** 0.5) < 1e-9


def test_analysis_pairs_by_seed_and_tracks_critic(tmp_path):
    for s, (a, b) in enumerate([(100, 98), (110, 107), (105, 104)]):
        _write(tmp_path, 'A', s, [200, a])
        _write(tmp_path, 'B', s, [210, b], critic=[90.0, 9.0])
    _write(tmp_path, 'B', 9, [1, 1])  # A に対応シードがない実行はペア比較から除外
    res = analyze(load_runs(tmp_path))
    d = res['paired_diff']['B-A']
    assert d['seeds'] == [0, 1, 2] and d['diffs'] == [-2, -3, -1]
    assert res['bg_critic_loss']['B']['0']['ratio_last_first'] == 0.1
    assert res['final_val_ppl']['A']['mean'] == 105


def test_main_writes_analysis(tmp_path):
    _write(tmp_path, 'A', 0, [10, 9])
    _write(tmp_path, 'C', 0, [11, 8])
    main(['--results-dir', str(tmp_path)])
    res = json.loads((tmp_path / 'analysis.json').read_text())
    assert res['paired_diff']['C-A']['diffs'] == [-1]
