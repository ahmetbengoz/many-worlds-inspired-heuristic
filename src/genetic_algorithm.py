from __future__ import annotations
import numpy as np
from .operators import tour_length, nearest_neighbor_tour, random_tour, order_crossover, swap_mutation, insertion_mutation, best_of_random_two_opt


def _tournament(lengths: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    idx = rng.choice(len(lengths), size=k, replace=False)
    return int(idx[np.argmin(lengths[idx])])


def run_ga(dist: np.ndarray, seed: int = 0, population_size: int = 40, generations: int = 250,
           mutation_rate: float = 0.25, local_samples: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    pop = []
    for i in range(population_size):
        pop.append(nearest_neighbor_tour(dist, rng, random_k=5) if i < population_size//4 else random_tour(n, rng))
    pop = np.asarray(pop, dtype=np.int32)
    lengths = np.asarray([tour_length(t, dist) for t in pop], dtype=float)
    best_idx = int(np.argmin(lengths)); best = pop[best_idx].copy(); best_len = int(lengths[best_idx])
    convergence = []
    for gen in range(generations):
        elite_count = max(1, population_size // 10)
        elite_idx = np.argsort(lengths)[:elite_count]
        new_pop = [pop[i].copy() for i in elite_idx]
        while len(new_pop) < population_size:
            p1 = pop[_tournament(lengths, rng)]
            p2 = pop[_tournament(lengths, rng)]
            child = order_crossover(p1, p2, rng)
            if rng.random() < mutation_rate:
                child = swap_mutation(child, rng) if rng.random() < 0.5 else insertion_mutation(child, rng)
            if rng.random() < 0.35:
                child, _ = best_of_random_two_opt(child, dist, rng, samples=local_samples)
            new_pop.append(child)
        pop = np.asarray(new_pop, dtype=np.int32)
        lengths = np.asarray([tour_length(t, dist) for t in pop], dtype=float)
        cur = int(np.argmin(lengths))
        if int(lengths[cur]) < best_len:
            best = pop[cur].copy(); best_len = int(lengths[cur])
        convergence.append(best_len)
    return {'algorithm': 'GA', 'best_length': best_len, 'best_tour': best.tolist(), 'final_mean_length': float(lengths.mean()), 'convergence': convergence}
