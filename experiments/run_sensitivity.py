"""Paired beta/gamma sensitivity experiment for MWI-H."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_tsplib_suite import BEST_KNOWN, PARAMETERS
from src.metrics import gap_percent
from src.mwi_h import run_mwih
from src.tsplib_parser import load_directory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(ROOT / 'data'))
    ap.add_argument('--out-dir', default=str(ROOT / 'results' / 'sensitivity'))
    ap.add_argument('--instances', nargs='*', default=['berlin52', 'ch150', 'lin318'])
    ap.add_argument('--betas', nargs='*', type=float, default=[0.0, 2.0, 6.0, 12.0])
    ap.add_argument('--gammas', nargs='*', type=float, default=[0.0, 1.0, 4.0, 8.0])
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--seed-base', type=int, default=20260805)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.instances)
    instances = [inst for inst in load_directory(args.data_dir) if inst.name in wanted]
    params = dict(PARAMETERS['MWI-H'])
    rows = []
    for inst_idx, inst in enumerate(instances):
        for run in range(1, args.runs + 1):
            seed = args.seed_base + run - 1 + 10000 * inst_idx
            for beta in args.betas:
                for gamma in args.gammas:
                    setting = dict(params)
                    setting.update(beta=beta, gamma=gamma)
                    started = time.perf_counter()
                    result = run_mwih(inst.dist, seed=seed, variant='full', **setting)
                    rows.append({
                        'instance': inst.name,
                        'dimension': inst.dimension,
                        'run': run,
                        'seed': seed,
                        'beta': beta,
                        'gamma': gamma,
                        'best_known': BEST_KNOWN[inst.name],
                        'best_length': result['best_length'],
                        'gap_percent': gap_percent(result['best_length'], BEST_KNOWN[inst.name]),
                        'final_effective_count': result['neff_curve'][-1],
                        'final_edge_diversity': result['edge_diversity_curve'][-1],
                        'runtime_seconds': time.perf_counter() - started,
                    })
            print(f'{inst.name}: completed paired run {run}/{args.runs}', flush=True)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / 'sensitivity_per_run.csv', index=False)
    summary = (
        per_run.groupby(['instance', 'dimension', 'beta', 'gamma'], as_index=False)
        .agg(
            runs=('run', 'count'),
            mean_gap_percent=('gap_percent', 'mean'),
            std_gap_percent=('gap_percent', 'std'),
            median_gap_percent=('gap_percent', 'median'),
            mean_final_effective_count=('final_effective_count', 'mean'),
            mean_final_edge_diversity=('final_edge_diversity', 'mean'),
            mean_runtime_seconds=('runtime_seconds', 'mean'),
        )
    )
    summary.to_csv(out / 'sensitivity_summary.csv', index=False)
    print('Wrote sensitivity outputs to', out)


if __name__ == '__main__':
    main()
