from __future__ import annotations
import numpy as np
from .operators import tour_length, nearest_neighbor_tour, best_of_random_two_opt


def _candidate_lists(dist: np.ndarray, k: int = 30):
    n = dist.shape[0]
    order = np.argsort(dist, axis=1)
    return order[:, 1:min(k+1, n)]


def _construct_tour(dist, eta, pher, cand_lists, rng, alpha, beta):
    n = dist.shape[0]
    tour = np.empty(n, dtype=np.int32)
    unvisited = np.ones(n, dtype=bool)
    cur = int(rng.integers(n))
    tour[0] = cur; unvisited[cur] = False
    for pos in range(1, n):
        cand = cand_lists[cur]
        cand = cand[unvisited[cand]]
        if len(cand) == 0:
            cand = np.flatnonzero(unvisited)
        weights = (pher[cur, cand] ** alpha) * (eta[cur, cand] ** beta)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            nxt = int(rng.choice(cand))
        else:
            weights = weights / weights.sum()
            nxt = int(rng.choice(cand, p=weights))
        tour[pos] = nxt; unvisited[nxt] = False; cur = nxt
    return tour


def run_aco(dist: np.ndarray, seed: int = 0, ants: int = 30, iterations: int = 180,
            alpha: float = 1.0, beta: float = 3.0, rho: float = 0.25, q: float = 100.0,
            candidate_k: int = 30, local_samples: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    nn = nearest_neighbor_tour(dist, rng, random_k=1)
    nn_len = max(1, tour_length(nn, dist))
    pher = np.full((n, n), 1.0 / (n * nn_len), dtype=float)
    cand_lists = _candidate_lists(dist, candidate_k)
    eta = 1.0 / np.maximum(dist, 1)
    best = nn.copy(); best_len = nn_len
    convergence = []
    for it in range(iterations):
        tours = []
        lengths = []
        for _ in range(ants):
            t = _construct_tour(dist, eta, pher, cand_lists, rng, alpha, beta)
            if rng.random() < 0.40:
                t, _ = best_of_random_two_opt(t, dist, rng, samples=local_samples)
            l = tour_length(t, dist)
            tours.append(t); lengths.append(l)
            if l < best_len:
                best = t.copy(); best_len = int(l)
        pher *= (1.0 - rho)
        # deposit from iteration best and global best
        for t, l, factor in [(tours[int(np.argmin(lengths))], min(lengths), 1.0), (best, best_len, 2.0)]:
            dep = factor * q / max(float(l), 1.0)
            for a, b in zip(t, np.roll(t, -1)):
                pher[a, b] += dep; pher[b, a] += dep
        convergence.append(best_len)
    return {'algorithm': 'ACO', 'best_length': int(best_len), 'best_tour': best.tolist(), 'final_mean_length': float(np.mean(lengths)), 'convergence': convergence}
