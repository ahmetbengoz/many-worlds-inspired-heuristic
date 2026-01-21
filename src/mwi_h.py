import numpy as np
from .utils import tour_length, random_tour

def mwi_h(D, iterations=800, worlds=20, beta=0.001):
    """
    Many-Worlds–Inspired Heuristic (MWI-H) for TSP.
    - Maintains multiple trajectories (worlds)
    - Updates influence via soft weighting (exp(-beta * length))
    - Propagates via neighborhood move (segment reversal)

    Returns best-so-far tour length history.
    """
    n = D.shape[0]
    trajectories = [random_tour(n) for _ in range(worlds)]
    history = []

    for _ in range(iterations):
        lengths = np.array([tour_length(t, D) for t in trajectories], dtype=float)
        weights = np.exp(-beta * lengths)
        weights_sum = weights.sum()
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            # Fallback: uniform weights if numerical underflow occurs
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights_sum

        new_trajectories = []
        for _w in range(worlds):
            idx = np.random.choice(worlds, p=weights)
            base = trajectories[idx].copy()

            i, j = np.random.choice(n, 2, replace=False)
            if i > j:
                i, j = j, i
            base[i:j] = base[i:j][::-1]

            new_trajectories.append(base)

        trajectories = new_trajectories
        history.append(min(tour_length(t, D) for t in trajectories))

    return history
