"""Metrics for TSP metaheuristic experiments."""
from __future__ import annotations
import numpy as np


def normalize_objectives(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / (values.max() - values.min() + eps)


def influence_weights(values: np.ndarray, beta: float = 5.0, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    fhat = normalize_objectives(values, eps)
    logits = -beta * fhat
    logits -= logits.max()
    w = np.exp(logits)
    p = w / (w.sum() + eps)
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
