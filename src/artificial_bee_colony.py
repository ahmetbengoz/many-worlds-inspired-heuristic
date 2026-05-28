from __future__ import annotations
import numpy as np
from .operators import tour_length, nearest_neighbor_tour, random_tour, best_of_random_two_opt, swap_mutation, insertion_mutation


def _neighbor(tour, dist, rng, local_samples):
    if rng.random() < 0.65:
        cand, _ = best_of_random_two_opt(tour, dist, rng, samples=local_samples)
        return cand
    return swap_mutation(tour, rng) if rng.random() < 0.5 else insertion_mutation(tour, rng)


def run_abc(dist: np.ndarray, seed: int = 0, food_sources: int = 30, cycles: int = 250,
            limit: int = 40, local_samples: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    foods = []
    for i in range(food_sources):
        foods.append(nearest_neighbor_tour(dist, rng, random_k=4) if i < food_sources//3 else random_tour(n, rng))
    foods = np.asarray(foods, dtype=np.int32)
    lengths = np.asarray([tour_length(t, dist) for t in foods], dtype=float)
    trials = np.zeros(food_sources, dtype=int)
    best_idx = int(np.argmin(lengths)); best = foods[best_idx].copy(); best_len = int(lengths[best_idx])
    convergence = []
    for cy in range(cycles):
        # Employed bees
        for i in range(food_sources):
            cand = _neighbor(foods[i], dist, rng, local_samples)
            cand_len = tour_length(cand, dist)
            if cand_len <= lengths[i]:
                foods[i] = cand; lengths[i] = cand_len; trials[i] = 0
            else:
                trials[i] += 1
        # Onlooker bees
        inv = 1.0 / (lengths - lengths.min() + 1.0)
        probs = inv / inv.sum()
        for _ in range(food_sources):
            i = int(rng.choice(food_sources, p=probs))
            cand = _neighbor(foods[i], dist, rng, local_samples)
            cand_len = tour_length(cand, dist)
            if cand_len <= lengths[i]:
                foods[i] = cand; lengths[i] = cand_len; trials[i] = 0
            else:
                trials[i] += 1
        # Scout bees
        for i in np.where(trials >= limit)[0]:
            foods[i] = nearest_neighbor_tour(dist, rng, random_k=6) if rng.random() < 0.5 else random_tour(n, rng)
            lengths[i] = tour_length(foods[i], dist); trials[i] = 0
        cur = int(np.argmin(lengths))
        if int(lengths[cur]) < best_len:
            best = foods[cur].copy(); best_len = int(lengths[cur])
        convergence.append(best_len)
    return {'algorithm': 'ABC/BCO', 'best_length': best_len, 'best_tour': best.tolist(), 'final_mean_length': float(lengths.mean()), 'convergence': convergence}
