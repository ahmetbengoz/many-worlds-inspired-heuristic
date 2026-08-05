"""Operator-controlled iterated local-search baseline for symmetric TSP."""
from __future__ import annotations

import numpy as np

from .operators import (
    candidate_lists,
    double_bridge_kick,
    insertion_mutation,
    nearest_neighbor_tour,
    swap_mutation,
    tour_length,
    two_opt_candidate_descent,
)


def run_ils(
    dist: np.ndarray,
    seed: int = 0,
    iterations: int = 800,
    candidate_k: int = 12,
    ls_passes: int = 1,
    ls_moves: int = 10,
    init_passes: int = 1,
    init_moves: int = 25,
    restart_patience: int = 80,
) -> dict:
    """Run ILS with the same 2-opt and perturbation operators used by MWI-H.

    Matching the number of high-level candidate generations (800) isolates the
    contribution of population weighting and interaction from the contribution
    of the shared TSP-specific neighborhood operators.
    """
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    cand = candidate_lists(dist, k=min(candidate_k, max(4, n - 1)))
    current = nearest_neighbor_tour(dist, rng, random_k=4)
    current, _ = two_opt_candidate_descent(
        current, dist, cand, max_passes=init_passes, max_moves=init_moves
    )
    current_len = tour_length(current, dist)
    best = current.copy()
    best_len = current_len
    last_improve = 0
    convergence = []

    for it in range(iterations):
        candidate = double_bridge_kick(current, rng)
        if rng.random() < 0.30:
            candidate = swap_mutation(candidate, rng)
        if rng.random() < 0.20:
            candidate = insertion_mutation(candidate, rng)
        candidate, _ = two_opt_candidate_descent(
            candidate, dist, cand, max_passes=ls_passes, max_moves=ls_moves
        )
        cand_len = tour_length(candidate, dist)
        if cand_len <= current_len:
            current, current_len = candidate, cand_len
        if cand_len < best_len:
            best, best_len = candidate.copy(), cand_len
            last_improve = it
        if it - last_improve >= restart_patience:
            current = double_bridge_kick(best, rng)
            current, _ = two_opt_candidate_descent(
                current, dist, cand, max_passes=ls_passes, max_moves=ls_moves
            )
            current_len = tour_length(current, dist)
            last_improve = it
        convergence.append(int(best_len))

    return {
        'algorithm': 'ILS',
        'best_length': int(best_len),
        'best_tour': best.tolist(),
        'final_mean_length': float(current_len),
        'convergence': convergence,
    }
