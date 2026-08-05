"""Paired component ablation for MWI-H."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_tsplib_suite import BEST_KNOWN, PARAMETERS
from src.metrics import gap_percent
from src.mwi_h import run_mwih
from src.tsplib_parser import load_directory

VARIANTS = [
    'full',
    'uniform_weights',
    'no_directional_penalty',
    'no_persistence',
    'no_local_search',
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(ROOT / 'data'))
    ap.add_argument('--out-dir', default=str(ROOT / 'results' / 'ablation'))
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--seed-base', type=int, default=20260805)
    ap.add_argument('--instances', nargs='*', default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    instances = load_directory(args.data_dir)
    if args.instances:
        wanted = set(args.instances)
        instances = [i for i in instances if i.name in wanted]

    params = dict(PARAMETERS['MWI-H'])
    rows = []
    for inst_idx, inst in enumerate(instances):
        optimum = BEST_KNOWN[inst.name]
        for run in range(1, args.runs + 1):
            seed = args.seed_base + run - 1 + 10000 * inst_idx
            for variant in VARIANTS:
                started = time.perf_counter()
                result = run_mwih(inst.dist, seed=seed, variant=variant, **params)
                rows.append({
                    'instance': inst.name,
                    'dimension': inst.dimension,
                    'run': run,
                    'seed': seed,
                    'variant': variant,
                    'best_known': optimum,
                    'best_length': result['best_length'],
                    'gap_percent': gap_percent(result['best_length'], optimum),
                    'final_edge_diversity': result['edge_diversity_curve'][-1],
                    'mean_edge_diversity': float(np.mean(result['edge_diversity_curve'])),
                    'final_effective_count': result['neff_curve'][-1],
                    'runtime_seconds': time.perf_counter() - started,
                })
            print(f'{inst.name}: completed paired run {run}/{args.runs}', flush=True)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / 'ablation_per_run.csv', index=False)
    summary = (
        per_run.groupby(['instance', 'dimension', 'variant'], as_index=False)
        .agg(
            runs=('run', 'count'),
            mean_gap_percent=('gap_percent', 'mean'),
            std_gap_percent=('gap_percent', 'std'),
            median_gap_percent=('gap_percent', 'median'),
            mean_edge_diversity=('mean_edge_diversity', 'mean'),
            mean_runtime_seconds=('runtime_seconds', 'mean'),
        )
    )
    summary['rank_within_instance'] = summary.groupby('instance')['mean_gap_percent'].rank(method='average')
    summary.to_csv(out / 'ablation_summary.csv', index=False)

    pivot = summary.pivot(index='instance', columns='variant', values='mean_gap_percent').dropna()
    stat, p_value = friedmanchisquare(*[pivot[v].values for v in VARIANTS])
    tests = [{'test': 'Friedman', 'comparison': '', 'statistic': stat, 'p_raw': p_value, 'p_holm': p_value, 'n_instances': len(pivot)}]
    pair_rows = []
    for variant in VARIANTS[1:]:
        w, p = wilcoxon(pivot['full'], pivot[variant], alternative='two-sided')
        diff = pivot['full'] - pivot[variant]
        pair_rows.append({
            'test': 'Wilcoxon',
            'comparison': f'full vs {variant}',
            'statistic': w,
            'p_raw': p,
            'median_gap_difference': float(np.median(diff)),
            'wins': int((diff < 0).sum()),
            'ties': int(np.isclose(diff, 0.0).sum()),
            'losses': int((diff > 0).sum()),
            'n_instances': len(pivot),
        })
    running = 0.0
    for rank, row in enumerate(sorted(pair_rows, key=lambda x: x['p_raw']), start=1):
        running = max(running, min(1.0, row['p_raw'] * (len(pair_rows) - rank + 1)))
        row['p_holm'] = running
    tests.extend(pair_rows)
    pd.DataFrame(tests).to_csv(out / 'ablation_tests.csv', index=False)
    print('Wrote ablation outputs to', out)


if __name__ == '__main__':
    main()
