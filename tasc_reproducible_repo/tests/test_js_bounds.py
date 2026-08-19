import numpy as np
from tasc_paper.distances import js_divergence, js_distance


def test_js_bounds_base2():
    p = np.array([1, 0, 0], dtype=float)
    q = np.array([0, 1, 0], dtype=float)
    assert 0 <= js_divergence(p, q, base=2.0) <= 1
    assert 0 <= js_distance(p, q, base=2.0) <= 1
    assert abs(js_divergence(p, q, base=2.0) - 1.0) < 1e-12


def test_js_zero_for_identical_distributions():
    p = np.array([2, 3, 5], dtype=float)
    assert js_divergence(p, p, base=2.0) == 0.0
