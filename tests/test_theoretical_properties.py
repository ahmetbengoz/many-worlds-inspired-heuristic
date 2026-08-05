import numpy as np

from src.metrics import influence_weights, population_edge_diversity
from src.mwi_h import directed_transition_kernel


def test_influence_weights_are_simplex_and_affine_invariant():
    values = np.array([-4.0, 1.0, 7.0, 13.0])
    p1, _ = influence_weights(values, beta=4.0)
    p2, _ = influence_weights(3.5 * values + 91.0, beta=4.0)
    assert np.all(p1 > 0.0)
    assert np.isclose(p1.sum(), 1.0)
    assert np.allclose(p1, p2)


def test_transition_kernel_is_row_stochastic_and_excludes_self():
    p = np.array([0.55, 0.30, 0.15])
    fhat = np.array([0.0, 0.4, 1.0])
    tau = directed_transition_kernel(p, fhat, gamma=3.0)
    assert np.all(tau >= 0.0)
    assert np.allclose(tau.sum(axis=1), 1.0)
    assert np.allclose(np.diag(tau), 0.0)
    assert not np.allclose(tau, tau.T)


def test_edge_diversity_distinguishes_identical_and_different_tours():
    identical = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    different = np.array([[0, 1, 2, 3], [0, 2, 1, 3]], dtype=np.int32)
    assert np.isclose(population_edge_diversity(identical), 0.0)
    assert population_edge_diversity(different) > 0.0
