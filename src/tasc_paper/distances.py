"""Shared distance utilities for TaSC paper experiments.

The most important guardrail here is that Jensen-Shannon divergence is computed
from normalized distributions and returned in base-2 units, so it is bounded by
0 <= JSD <= 1. scipy.spatial.distance.jensenshannon returns sqrt(JSD), so the
value is squared for divergence and left as-is for distance.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def normalize_distribution(x: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if eps:
        arr = arr + float(eps)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Cannot normalize an empty or non-finite distribution")
    return arr / s


def js_divergence(p: np.ndarray, q: np.ndarray, *, base: float = 2.0, eps: float = 0.0) -> float:
    """Return Jensen-Shannon divergence.

    With base=2, this is bounded by [0, 1].
    """
    pp = normalize_distribution(p, eps=eps)
    qq = normalize_distribution(q, eps=eps)
    return float(jensenshannon(pp, qq, base=base) ** 2)


def js_distance(p: np.ndarray, q: np.ndarray, *, base: float = 2.0, eps: float = 0.0) -> float:
    """Return Jensen-Shannon distance, sqrt(JS divergence)."""
    pp = normalize_distribution(p, eps=eps)
    qq = normalize_distribution(q, eps=eps)
    return float(jensenshannon(pp, qq, base=base))
