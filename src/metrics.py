"""Metrics for TSP metaheuristic experiments."""
from __future__ import annotations
import numpy as np


def normalize_objectives(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span <= eps:
        return np.zeros_like(values, dtype=float)
    return (values - values.min()) / span


def influence_weights(values: np.ndarray, beta: float = 5.0, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    fhat = normalize_objectives(values, eps)
    logits = -beta * fhat
    logits -= logits.max()
    w = np.exp(logits)
    p = w / w.sum()
    return p, fhat


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    return float(-(p * np.log(p + eps)).sum())


def effective_count(p: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.exp(entropy(p, eps)))


def gap_percent(value: float, best_known: float | None) -> float | None:
    if best_known is None or best_known <= 0:
        return None
    return 100.0 * (float(value) - float(best_known)) / float(best_known)


def population_edge_diversity(population: np.ndarray) -> float:
    """Mean pairwise edge disagreement for symmetric TSP tours.

    The value is zero when every trajectory has the same undirected edge set and
    approaches one as edge overlap vanishes. Unlike entropy of the influence
    vector, this metric measures diversity in solution space.
    """
    pop = np.asarray(population)
    m, n = pop.shape
    if m < 2 or n == 0:
        return 0.0
    edge_sets = []
    for tour in pop:
        edge_sets.append({
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(tour, np.roll(tour, -1))
        })
    disagreements = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            overlap = len(edge_sets[i] & edge_sets[j]) / n
            disagreements.append(1.0 - overlap)
    return float(np.mean(disagreements))
