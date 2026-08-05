from __future__ import annotations
import numpy as np
import pandas as pd


def friedman_and_wilcoxon(per_run: pd.DataFrame, target_algorithm: str = 'MWI-H') -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except Exception:
        ranks = _average_ranks(per_run)
        return ranks, pd.DataFrame([{'test': 'scipy_unavailable', 'note': 'Install scipy for Friedman/Wilcoxon tests.'}])
    # The benchmark instance is the independent experimental block. Repeated
    # stochastic runs estimate each algorithm-instance mean; they are not 30
    # independent problem instances and must not be used as Friedman blocks.
    instance_means = (
        per_run.groupby(['instance', 'algorithm'], as_index=False)['gap_percent']
        .mean()
    )
    pivot = instance_means.pivot(index='instance', columns='algorithm', values='gap_percent')
    pivot = pivot.dropna(axis=0)
    algos = list(pivot.columns)
    if len(algos) < 3 or len(pivot) < 2:
        return _average_ranks(per_run), pd.DataFrame([{'test': 'insufficient_data'}])
    stat, p = friedmanchisquare(*[pivot[a].values for a in algos])
    ranks = pivot.rank(axis=1, method='average').mean().sort_values().reset_index()
    ranks.columns = ['algorithm', 'average_rank']
    stats_rows = [{'test': 'Friedman', 'statistic': float(stat), 'p_value': float(p), 'n_blocks': int(len(pivot))}]
    if target_algorithm in algos:
        m = max(1, len(algos)-1)
        pair_rows = []
        for a in algos:
            if a == target_algorithm:
                continue
            try:
                wstat, wp = wilcoxon(pivot[target_algorithm].values, pivot[a].values, zero_method='wilcox', alternative='two-sided')
                diff = pivot[target_algorithm].values - pivot[a].values
                pair_rows.append({
                    'test': f'Wilcoxon_vs_{target_algorithm}',
                    'comparison': f'{target_algorithm} vs {a}',
                    'statistic': float(wstat),
                    'p_raw': float(wp),
                    'median_gap_difference': float(np.median(diff)),
                    'wins': int(np.sum(diff < 0)),
                    'ties': int(np.sum(np.isclose(diff, 0.0))),
                    'losses': int(np.sum(diff > 0)),
                })
            except Exception as e:
                pair_rows.append({'test': f'Wilcoxon_vs_{target_algorithm}', 'comparison': f'{target_algorithm} vs {a}', 'error': str(e)})
        # Holm correction
        valid = [r for r in pair_rows if 'p_raw' in r]
        valid_sorted = sorted(valid, key=lambda x: x['p_raw'])
        running = 0.0
        for rank, r in enumerate(valid_sorted, start=1):
            running = max(running, min(1.0, r['p_raw'] * (m - rank + 1)))
            r['p_holm'] = running
        stats_rows.extend(pair_rows)
    return ranks, pd.DataFrame(stats_rows)


def _average_ranks(per_run: pd.DataFrame) -> pd.DataFrame:
    instance_means = (
        per_run.groupby(['instance', 'algorithm'], as_index=False)['gap_percent']
        .mean()
    )
    pivot = instance_means.pivot(index='instance', columns='algorithm', values='gap_percent').dropna(axis=0)
    if pivot.empty:
        return pd.DataFrame()
    ranks = pivot.rank(axis=1, method='average').mean().sort_values().reset_index()
    ranks.columns = ['algorithm', 'average_rank']
    return ranks
