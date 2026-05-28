"""Permutation and TSP neighborhood operators."""
from __future__ import annotations

import numpy as np


def tour_length(tour: np.ndarray, dist: np.ndarray) -> int:
    return int(dist[tour, np.roll(tour, -1)].sum())


def random_tour(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n).astype(np.int32)


def nearest_neighbor_tour(dist: np.ndarray, rng: np.random.Generator, start: int | None = None, random_k: int = 1) -> np.ndarray:
    n = dist.shape[0]
    if start is None:
        start = int(rng.integers(n))
    unvisited = np.ones(n, dtype=bool)
    tour = np.empty(n, dtype=np.int32)
    cur = start
    tour[0] = cur
    unvisited[cur] = False
    for pos in range(1, n):
        candidates = np.flatnonzero(unvisited)
        d = dist[cur, candidates]
        k = min(random_k, len(candidates))
        if k <= 1:
            nxt = candidates[int(np.argmin(d))]
        else:
            idx = np.argpartition(d, k - 1)[:k]
            nxt = candidates[int(rng.choice(idx))]
        tour[pos] = nxt
        unvisited[nxt] = False
        cur = int(nxt)
    return tour


def two_opt_delta(tour: np.ndarray, dist: np.ndarray, i: int, k: int) -> int:
    n = len(tour)
    a = int(tour[i - 1]); b = int(tour[i])
    c = int(tour[k]); d = int(tour[(k + 1) % n])
    return int(dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d])


def apply_two_opt(tour: np.ndarray, i: int, k: int) -> np.ndarray:
    new = tour.copy()
    new[i:k+1] = new[i:k+1][::-1]
    return new


def random_two_opt_move(tour: np.ndarray, dist: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    n = len(tour)
    i, k = sorted(rng.choice(np.arange(1, n), size=2, replace=False))
    if i == k:
        return tour.copy(), 0
    delta = two_opt_delta(tour, dist, int(i), int(k))
    return apply_two_opt(tour, int(i), int(k)), int(delta)


def best_of_random_two_opt(tour: np.ndarray, dist: np.ndarray, rng: np.random.Generator, samples: int = 24) -> tuple[np.ndarray, int]:
    n = len(tour)
    best_i, best_k, best_delta = None, None, 0
    for _ in range(samples):
        i, k = sorted(rng.choice(np.arange(1, n), size=2, replace=False))
        if i == k:
            continue
        dlt = two_opt_delta(tour, dist, int(i), int(k))
        if dlt < best_delta:
            best_i, best_k, best_delta = int(i), int(k), int(dlt)
    if best_i is None:
        return tour.copy(), 0
    return apply_two_opt(tour, best_i, best_k), best_delta


def swap_mutation(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    new = tour.copy()
    a, b = rng.choice(len(tour), size=2, replace=False)
    new[a], new[b] = new[b], new[a]
    return new


def insertion_mutation(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(tour)
    new = tour.copy()
    i, j = rng.choice(n, size=2, replace=False)
    city = new[i]
    new = np.delete(new, i)
    new = np.insert(new, j, city)
    return new.astype(np.int32)


def order_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(parent1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=np.int32)
    segment = parent1[a:b+1]
    child[a:b+1] = segment
    used = set(int(x) for x in segment)
    fill = [int(x) for x in parent2 if int(x) not in used]
    idx = 0
    for pos in list(range(0, a)) + list(range(b+1, n)):
        child[pos] = fill[idx]
        idx += 1
    return child


def guided_position_move(tour: np.ndarray, guide: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small path-relinking style move: place one city closer to its guide position."""
    n = len(tour)
    new = tour.copy()
    city = int(guide[int(rng.integers(n))])
    cur_pos = int(np.where(new == city)[0][0])
    guide_pos = int(np.where(guide == city)[0][0])
    if cur_pos == guide_pos:
        return new
    city_val = new[cur_pos]
    new = np.delete(new, cur_pos)
    guide_pos = min(guide_pos, n - 1)
    new = np.insert(new, guide_pos, city_val)
    return new.astype(np.int32)


def candidate_lists(dist: np.ndarray, k: int = 20) -> np.ndarray:
    """Nearest-neighbour candidate lists for local 2-opt search."""
    n = dist.shape[0]
    order = np.argsort(dist, axis=1)
    return order[:, 1:min(k + 1, n)].astype(np.int32)


def two_opt_candidate_descent(
    tour: np.ndarray,
    dist: np.ndarray,
    cand_lists: np.ndarray,
    max_passes: int = 2,
    max_moves: int = 50,
) -> tuple[np.ndarray, int]:
    """First-improvement 2-opt descent restricted to nearest-neighbour candidates.

    This is deliberately bounded. It gives MWI-H a real route-improvement operator
    without turning the method into a full deterministic TSP solver.
    """
    n = len(tour)
    work = tour.copy().astype(np.int32)
    total_delta = 0
    moves = 0
    if n < 5:
        return work, 0
    for _pass in range(max_passes):
        improved_this_pass = False
        pos = np.empty(n, dtype=np.int32)
        pos[work] = np.arange(n, dtype=np.int32)
        for i in range(1, n - 2):
            a = int(work[i - 1])
            b = int(work[i])
            # Try replacing edge (a,b) by (a,c), where c is spatially close to a.
            for c in cand_lists[a]:
                k = int(pos[int(c)])
                if k <= i or k >= n - 1:
                    continue
                # Avoid adjacent-edge degeneracy.
                if k == i:
                    continue
                c_int = int(work[k])
                d = int(work[(k + 1) % n])
                delta = int(dist[a, c_int] + dist[b, d] - dist[a, b] - dist[c_int, d])
                if delta < 0:
                    work[i:k + 1] = work[i:k + 1][::-1]
                    total_delta += delta
                    moves += 1
                    improved_this_pass = True
                    # Update only the reversed segment positions.
                    pos[work[i:k + 1]] = np.arange(i, k + 1, dtype=np.int32)
                    break
            if moves >= max_moves:
                return work, total_delta
        if not improved_this_pass:
            break
    return work, total_delta


def double_bridge_kick(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Double-bridge perturbation for escaping 2-opt local basins."""
    n = len(tour)
    if n < 12:
        return swap_mutation(tour, rng)
    cuts = sorted(rng.choice(np.arange(1, n), size=4, replace=False))
    a, b, c, d = cuts
    p1, p2, p3, p4, p5 = tour[:a], tour[a:b], tour[b:c], tour[c:d], tour[d:]
    return np.concatenate([p1, p3, p2, p4, p5]).astype(np.int32)


def edge_agreement_ratio(tour_a: np.ndarray, tour_b: np.ndarray) -> float:
    """Undirected edge-overlap diversity diagnostic between two tours."""
    def edges(t):
        return {tuple(sorted((int(x), int(y)))) for x, y in zip(t, np.roll(t, -1))}
    ea = edges(tour_a); eb = edges(tour_b)
    return len(ea & eb) / max(1, len(ea))
