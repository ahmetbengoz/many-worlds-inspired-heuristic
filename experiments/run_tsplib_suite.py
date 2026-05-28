from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tsplib_parser import load_directory
from src.metrics import gap_percent
from src.mwi_h import run_mwih
from src.simulated_annealing import run_sa
from src.genetic_algorithm import run_ga
from src.ant_colony import run_aco
from src.artificial_bee_colony import run_abc
from src.stats_tests import friedman_and_wilcoxon

BEST_KNOWN = {
    'berlin52': 7542,
    'eil76': 538,
    'pr76': 108159,
    'rat99': 1211,
    'kroA100': 21282,
    'ch150': 6528,
    'lin318': 42029,
    'tsp225': 3919,
    'pcb442': 50778,
    # Added because uploaded zip includes these standard TSPLIB instances.
    'pr439': 107217,
    'd493': 35002,
}

PARAMETERS = {
    'MWI-H': {'population_size': 16, 'iterations': 50, 'beta': 6.0, 'gamma': 4.0, 'local_samples': 4, 'candidate_k': 12, 'ls_passes': 1, 'ls_moves': 10, 'init_passes': 1, 'init_moves': 25, 'elite_count': 2, 'restart_patience': 20},
    'SA': {'iterations': 7500, 't0': 1000.0, 'alpha': 0.9993},
    'GA': {'population_size': 40, 'generations': 250, 'mutation_rate': 0.25, 'local_samples': 8},
    'ACO': {'ants': 30, 'iterations': 180, 'alpha': 1.0, 'beta': 3.0, 'rho': 0.25, 'q': 100.0, 'candidate_k': 30, 'local_samples': 6},
    'ABC/BCO': {'food_sources': 30, 'cycles': 250, 'limit': 40, 'local_samples': 12},
}

FAST_PARAMETERS = {
    # Chat-validation budget: intentionally small so every uploaded instance can be exercised in-session.
    # Use PARAMETERS above with --runs 30 to reproduce the standard manuscript protocol.
    'MWI-H': {'population_size': 12, 'iterations': 40, 'beta': 6.0, 'gamma': 4.0, 'local_samples': 6, 'candidate_k': 12, 'ls_passes': 1, 'ls_moves': 20, 'init_passes': 1, 'init_moves': 30, 'elite_count': 1, 'restart_patience': 20},
    'SA': {'iterations': 500, 't0': 1000.0, 'alpha': 0.9975},
    'GA': {'population_size': 12, 'generations': 30, 'mutation_rate': 0.25, 'local_samples': 1},
    'ACO': {'ants': 6, 'iterations': 12, 'alpha': 1.0, 'beta': 3.0, 'rho': 0.25, 'q': 100.0, 'candidate_k': 10, 'local_samples': 1},
    'ABC/BCO': {'food_sources': 10, 'cycles': 30, 'limit': 15, 'local_samples': 2},
}


def run_one(algo: str, dist, seed: int, params: dict):
    if algo == 'MWI-H':
        return run_mwih(dist, seed=seed, **params)
    if algo == 'SA':
        return run_sa(dist, seed=seed, **params)
    if algo == 'GA':
        return run_ga(dist, seed=seed, **params)
    if algo == 'ACO':
        return run_aco(dist, seed=seed, **params)
    if algo == 'ABC/BCO':
        return run_abc(dist, seed=seed, **params)
    raise ValueError(algo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(ROOT / 'data' / 'tsplib'))
    ap.add_argument('--out-dir', default=str(ROOT / 'results'))
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--seed-base', type=int, default=20260520)
    ap.add_argument('--fast', action='store_true', help='Use reduced budget for quick reproducibility checks.')
    ap.add_argument('--instances', nargs='*', default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    instances = load_directory(args.data_dir)
    if args.instances:
        wanted = set(args.instances)
        instances = [i for i in instances if i.name in wanted or Path(i.name).stem in wanted]
    params_all = FAST_PARAMETERS if args.fast else PARAMETERS
    algorithms = ['MWI-H', 'SA', 'GA', 'ACO', 'ABC/BCO']

    summary_rows = []
    per_run_rows = []
    conv_rows = []
    ent_rows = []
    instance_rows = []

    for inst in instances:
        best_known = BEST_KNOWN.get(inst.name, BEST_KNOWN.get(Path(inst.name).stem))
        instance_rows.append({'instance': inst.name, 'dimension': inst.dimension, 'edge_weight_type': inst.edge_weight_type, 'best_known': best_known})
        print(f'Instance {inst.name} n={inst.dimension}')
        for algo in algorithms:
            params = params_all[algo]
            lengths = []
            gaps = []
            for r in range(args.runs):
                seed = args.seed_base + r + 1000 * algorithms.index(algo) + 10000 * instances.index(inst)
                print(f'  {algo} run {r+1}/{args.runs} seed={seed}', flush=True)
                res = run_one(algo, inst.dist, seed, params)
                best_len = res['best_length']
                gap = gap_percent(best_len, best_known)
                lengths.append(best_len)
                gaps.append(gap if gap is not None else np.nan)
                per_run_rows.append({
                    'instance': inst.name, 'dimension': inst.dimension, 'algorithm': algo, 'run': r + 1, 'seed': seed,
                    'best_known': best_known, 'best_length': best_len,
                    'gap_percent': gap,
                    'final_mean_length': res.get('final_mean_length'),
                    'budget_mode': 'fast' if args.fast else 'standard',
                })
                conv = res.get('convergence', [])
                for it, val in enumerate(conv):
                    conv_rows.append({'instance': inst.name, 'algorithm': algo, 'run': r + 1, 'iteration_index': it, 'best_length': val, 'gap_percent': gap_percent(val, best_known)})
                if algo == 'MWI-H':
                    H = res.get('entropy_curve', [])
                    Ne = res.get('neff_curve', [])
                    active = res.get('active_curve', [])
                    mean_curve = res.get('mean_curve', [])
                    for it in range(len(H)):
                        ent_rows.append({'instance': inst.name, 'algorithm': algo, 'run': r + 1, 'iteration': it,
                                         'entropy_H': H[it], 'N_eff': Ne[it], 'active_trajectories': active[it],
                                         'mean_length': mean_curve[it] if it < len(mean_curve) else np.nan})
            arr = np.asarray(lengths, dtype=float)
            gap_arr = np.asarray(gaps, dtype=float)
            summary_rows.append({
                'instance': inst.name, 'dimension': inst.dimension, 'algorithm': algo, 'runs': args.runs,
                'best_known': best_known,
                'best': int(np.min(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                'median': float(np.median(arr)),
                'mean_gap_percent': float(np.nanmean(gap_arr)) if not np.all(np.isnan(gap_arr)) else np.nan,
                'best_gap_percent': float(np.nanmin(gap_arr)) if not np.all(np.isnan(gap_arr)) else np.nan,
                'budget_mode': 'fast' if args.fast else 'standard',
            })

    instance_df = pd.DataFrame(instance_rows)
    per_run_df = pd.DataFrame(per_run_rows)
    conv_df = pd.DataFrame(conv_rows)
    ent_df = pd.DataFrame(ent_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df['rank_within_instance'] = summary_df.groupby('instance')['mean_gap_percent'].rank(method='average')
    ranks_df, stats_df = friedman_and_wilcoxon(per_run_df)

    instance_df.to_csv(out / 'dataset_summary.csv', index=False)
    per_run_df.to_csv(out / 'per_run_results.csv', index=False)
    conv_df.to_csv(out / 'convergence_curves.csv', index=False)
    ent_df.to_csv(out / 'entropy_curves.csv', index=False)
    summary_df.to_csv(out / 'performance_summary.csv', index=False)
    ranks_df.to_csv(out / 'statistical_ranks.csv', index=False)
    stats_df.to_csv(out / 'statistical_tests.csv', index=False)
    with open(out / 'parameter_settings.json', 'w', encoding='utf-8') as f:
        json.dump(params_all, f, indent=2)
    print('Wrote outputs to', out)

if __name__ == '__main__':
    main()
