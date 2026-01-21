import numpy as np
from .utils import tour_length, random_tour

def simulated_annealing(D, iterations=800, T0=1000.0, cooling=0.995):
    """
    Simple Simulated Annealing baseline for TSP using 2-opt-like segment reversal.
    Returns best-so-far tour length history.
    """
    n = D.shape[0]
    T = T0

    current = random_tour(n)
    best = current.copy()
    best_len = tour_length(current, D)

    history = []
    for _ in range(iterations):
        i, j = np.random.choice(n, 2, replace=False)
        if i > j:
            i, j = j, i

        candidate = current.copy()
        candidate[i:j] = candidate[i:j][::-1]

        delta = tour_length(candidate, D) - tour_length(current, D)
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-12)):
            current = candidate

        cur_len = tour_length(current, D)
        if cur_len < best_len:
            best = current.copy()
            best_len = cur_len

        history.append(best_len)
        T *= cooling

    return history
