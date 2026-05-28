"""MWI-H: measurement-inspired trajectory-persistence metaheuristic.

The implementation combines route-improvement mechanisms while preserving the
mathematical identity of the method: normalized influence weights, entropy-based
trajectory persistence, and a directed transition kernel. The main engineering
change is bounded candidate-list 2-opt descent plus controlled double-bridge
perturbation, so the method can make credible local-optimum escape claims on
TSPLIB rather than relying on weak random mutation.
"""
from __future__ import annotations
import numpy as np

from .operators import (
    tour_length,
    random_tour,
    nearest_neighbor_tour,
    best_of_random_two_opt,
    swap_mutation,
    insertion_mutation,
    guided_position_move,
    order_crossover,
    candidate_lists,
    two_opt_candidate_descent,
    double_bridge_kick,
)
from .metrics import influence_weights, entropy, effective_count


def directed_transition_kernel(p: np.ndarray, fhat: np.ndarray, gamma: float = 3.0, eps: float = 1e-12) -> np.ndarray:
    """Directed, row-normalized inter-trajectory transition kernel.

    tau_ij is not generally equal to tau_ji because the row anchor fhat_i changes
    the uphill penalty max(0, fhat_j - fhat_i).
    """
    n = len(p)
    tau = np.empty((n, n), dtype=float)
    for i in range(n):
        penalty = np.maximum(0.0, fhat - fhat[i])
        row = p * np.exp(-gamma * penalty)
        row[i] += eps
        tau[i] = row / (row.sum() + eps)
    return tau


def _initial_population(dist: np.ndarray, rng: np.random.Generator, population_size: int, cand: np.ndarray,
                        init_passes: int, init_moves: int) -> np.ndarray:
    n = dist.shape[0]
    pop = []
    # Deterministic-ish high-quality anchors from different starts.
    starts = list(rng.choice(n, size=min(population_size // 2, n), replace=False))
    for s in starts:
        t = nearest_neighbor_tour(dist, rng, start=int(s), random_k=1)
        t, _ = two_opt_candidate_descent(t, dist, cand, max_passes=init_passes, max_moves=init_moves)
        pop.append(t)
    # Stochastic nearest-neighbour variants preserve multiple basins.
    while len(pop) < max(2, int(0.75 * population_size)):
        t = nearest_neighbor_tour(dist, rng, random_k=4)
        t, _ = two_opt_candidate_descent(t, dist, cand, max_passes=max(1, init_passes // 2), max_moves=max(10, init_moves // 2))
        pop.append(t)
    # Random trajectories, lightly improved, maintain non-collapse diversity.
    while len(pop) < population_size:
        t = random_tour(n, rng)
        t, _ = best_of_random_two_opt(t, dist, rng, samples=32)
        pop.append(t.astype(np.int32))
    return np.asarray(pop, dtype=np.int32)


def run_mwih(
    dist: np.ndarray,
    seed: int = 0,
    population_size: int = 36,
    iterations: int = 320,
    beta: float = 6.0,
    gamma: float = 4.0,
    local_samples: int = 32,
    influence_eps: float = 1e-6,
    candidate_k: int = 24,
    ls_passes: int = 2,
    ls_moves: int = 60,
    init_passes: int = 3,
    init_moves: int = 120,
    elite_count: int = 2,
    restart_patience: int = 55,
) -> dict:
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    cand = candidate_lists(dist, k=min(candidate_k, max(4, n - 1)))

    population = _initial_population(dist, rng, population_size, cand, init_passes, init_moves)
    lengths = np.asarray([tour_length(t, dist) for t in population], dtype=float)
    personal_best = population.copy()
    personal_best_lengths = lengths.copy()

    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_len = int(lengths[best_idx])
    last_improve = 0

    convergence = []
    entropy_curve = []
    neff_curve = []
    active_curve = []
    mean_curve = []

    for it in range(iterations):
        p, fhat = influence_weights(lengths, beta=beta)
        tau = directed_transition_kernel(p, fhat, gamma=gamma)
        H = entropy(p)
        Neff = effective_count(p)
        convergence.append(best_len)
        mean_curve.append(float(lengths.mean()))
        entropy_curve.append(H)
        neff_curve.append(Neff)
        active_curve.append(int((p > influence_eps).sum()))

        order = np.argsort(lengths)
        elites = order[:max(1, min(elite_count, population_size))]
        new_population = population.copy()
        new_lengths = lengths.copy()

        for i in range(population_size):
            # Elites are still allowed to search, but their original trajectories persist.
            j = int(rng.choice(population_size, p=tau[i]))
            r = rng.random()
            if r < 0.42:
                candidate = order_crossover(population[i], population[j], rng)
            elif r < 0.70:
                candidate = guided_position_move(population[i], population[j], rng)
            elif r < 0.86:
                candidate = personal_best[i].copy()
                candidate = double_bridge_kick(candidate, rng)
            else:
                # Low-influence trajectories periodically explore from the global best basin.
                candidate = best_tour.copy() if p[i] < (1.0 / population_size) else population[i].copy()
                candidate = double_bridge_kick(candidate, rng)

            # Small stochastic variation before deterministic bounded intensification.
            if rng.random() < 0.30:
                candidate = swap_mutation(candidate, rng)
            if rng.random() < 0.20:
                candidate = insertion_mutation(candidate, rng)
            if rng.random() < 0.20:
                candidate, _ = best_of_random_two_opt(candidate, dist, rng, samples=local_samples)

            candidate, _ = two_opt_candidate_descent(candidate, dist, cand, max_passes=ls_passes, max_moves=ls_moves)
            cand_len = tour_length(candidate, dist)

            if cand_len <= lengths[i]:
                accept = True
            else:
                # Controlled uphill acceptance preserves non-collapsing trajectories.
                scale = max(1.0, float(np.std(lengths)))
                uphill = np.exp(-(cand_len - lengths[i]) / scale)
                accept = rng.random() < min(0.18, 0.5 * float(p[j]) + 0.04) * uphill
            if accept:
                new_population[i] = candidate
                new_lengths[i] = cand_len

            # Personal trajectory memory.
            if cand_len < personal_best_lengths[i]:
                personal_best[i] = candidate.copy()
                personal_best_lengths[i] = cand_len

        # Explicit elitist persistence: preserves best measured trajectories without collapsing all others.
        worst = np.argsort(new_lengths)[-len(elites):]
        for src, dst in zip(elites, worst):
            if lengths[src] < new_lengths[dst]:
                new_population[dst] = population[src].copy()
                new_lengths[dst] = lengths[src]

        # Stagnation-aware basin escape, applied only to a small low-quality tail.
        if it - last_improve >= restart_patience:
            tail = np.argsort(new_lengths)[-max(2, population_size // 6):]
            for idx in tail:
                kicked = double_bridge_kick(best_tour, rng)
                kicked, _ = two_opt_candidate_descent(kicked, dist, cand, max_passes=ls_passes, max_moves=ls_moves)
                k_len = tour_length(kicked, dist)
                new_population[idx] = kicked
                new_lengths[idx] = k_len
            last_improve = it

        population, lengths = new_population, new_lengths
        cur_best_idx = int(np.argmin(lengths))
        if int(lengths[cur_best_idx]) < best_len:
            best_len = int(lengths[cur_best_idx])
            best_tour = population[cur_best_idx].copy()
            last_improve = it

    return {
        'algorithm': 'MWI-H',
        'best_length': best_len,
        'best_tour': best_tour.tolist(),
        'final_mean_length': float(lengths.mean()),
        'convergence': convergence,
        'mean_curve': mean_curve,
        'entropy_curve': entropy_curve,
        'neff_curve': neff_curve,
        'active_curve': active_curve,
    }
