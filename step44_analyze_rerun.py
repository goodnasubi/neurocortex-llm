"""ステップ44.4 再実験（colab_step44_full_scale_experiment.py の出力）の分析

入力: {results_dir}/phase{A,B,C}_seed{N}.json
出力: {out}/analysis.json（と matplotlib があれば val PPL 推移の図）

- 指標は backbone の非重み付き val PPL（最終エポック）。
- フェーズ間比較は同一シードのペア差（B−A, C−A, C−B）。平均・標準偏差・対応のある t 検定（n≥2）。
- 基底核 critic: Phase B/C の val_head_ce['bg_critic_loss'] の初回→最終エポックの推移。

使い方: python step44_analyze_rerun.py --results-dir results/step44_full_scale_rerun
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def load_runs(results_dir: Path) -> Dict[str, Dict[int, Dict]]:
    runs: Dict[str, Dict[int, Dict]] = {}
    for f in sorted(results_dir.glob('phase*_seed*.json')):
        r = json.loads(f.read_text())
        if not r.get('val_ppls'):
            continue
        runs.setdefault(r['phase'], {})[int(r['seed'])] = r
    return runs


def paired_t(diffs: List[float]) -> Dict:
    n = len(diffs)
    mean = sum(diffs) / n if n else float('nan')
    if n < 2:
        return {'n': n, 'mean': mean, 'std': float('nan'), 't': float('nan'), 'p': float('nan')}
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float('inf') * (1 if mean > 0 else -1)
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 1)) if math.isfinite(t) else 0.0
    except ImportError:
        p = float('nan')
    return {'n': n, 'mean': mean, 'std': sd, 't': t, 'p': p}


def analyze(runs: Dict[str, Dict[int, Dict]]) -> Dict:
    final = {ph: {s: r['val_ppls'][-1] for s, r in rs.items()} for ph, rs in runs.items()}
    out: Dict = {
        'metric': 'backbone の非重み付き val PPL（最終エポック）。差は同一シードのペア差（負が改善）',
        'final_val_ppl': {},
        'paired_diff': {},
        'bg_critic_loss': {},
        'epochs': {ph: {str(s): len(r['val_ppls']) for s, r in rs.items()} for ph, rs in runs.items()},
    }
    for ph, v in sorted(final.items()):
        xs = list(v.values())
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float('nan')
        out['final_val_ppl'][ph] = {'per_seed': {str(s): x for s, x in sorted(v.items())}, 'mean': m, 'std': sd}
    for a, b in [('B', 'A'), ('C', 'A'), ('C', 'B')]:
        if a in final and b in final:
            seeds = sorted(set(final[a]) & set(final[b]))
            diffs = [final[a][s] - final[b][s] for s in seeds]
            if diffs:
                out['paired_diff'][f'{a}-{b}'] = {'seeds': seeds, 'diffs': diffs, **paired_t(diffs)}
    for ph in ('B', 'C'):
        for s, r in sorted(runs.get(ph, {}).items()):
            c = [h.get('bg_critic_loss') for h in r.get('val_head_ce', []) if h.get('bg_critic_loss') is not None]
            if c:
                out['bg_critic_loss'].setdefault(ph, {})[str(s)] = {
                    'first': c[0], 'last': c[-1], 'ratio_last_first': c[-1] / c[0] if c[0] else float('nan'),
                    'per_epoch': c}
    return out


def plot(runs: Dict[str, Dict[int, Dict]], path: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    colors = {'A': '#3b6ea5', 'B': '#c0703a', 'C': '#4a9a5b'}
    fig, ax = plt.subplots(figsize=(6, 4))
    for ph, rs in sorted(runs.items()):
        for i, (s, r) in enumerate(sorted(rs.items())):
            ax.plot(range(1, len(r['val_ppls']) + 1), r['val_ppls'], color=colors.get(ph), alpha=0.8,
                    label=f'Phase {ph}' if i == 0 else None)
    ax.set_xlabel('epoch')
    ax.set_ylabel('val PPL (backbone, unweighted)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results/step44_full_scale_rerun')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    rd = Path(args.results_dir)
    out = Path(args.out or rd)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(rd)
    if not runs:
        raise SystemExit(f'{rd} に phase*_seed*.json がない')
    res = analyze(runs)
    (out / 'analysis.json').write_text(json.dumps(res, indent=2, ensure_ascii=False))
    plot(runs, out / 'val_ppl_curves.png')
    print(json.dumps({k: res[k] for k in ('final_val_ppl', 'paired_diff')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
