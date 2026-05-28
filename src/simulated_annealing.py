from __future__ import annotations
import numpy as np
from .operators import nearest_neighbor_tour, random_two_opt_move, tour_length


def run_sa(dist: np.ndarray, seed: int = 0, iterations: int = 7500, t0: float = 1000.0, alpha: float = 0.9993) -> dict:
    rng = np.random.default_rng(seed)
    cur = nearest_neighbor_tour(dist, rng, random_k=4)
    cur_len = tour_length(cur, dist)
    best = cur.copy(); best_len = cur_len
    convergence = []
    temp = t0
    record_every = max(1, iterations // 250)
    for it in range(iterations):
        cand, delta = random_two_opt_move(cur, dist, rng)
        cand_len = cur_len + delta
        if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            cur, cur_len = cand, int(cand_len)
            if cur_len < best_len:
                best, best_len = cur.copy(), cur_len
        temp *= alpha
        if it % record_every == 0:
            convergence.append(best_len)
    return {'algorithm': 'SA', 'best_length': int(best_len), 'best_tour': best.tolist(), 'final_mean_length': float(cur_len), 'convergence': convergence}
