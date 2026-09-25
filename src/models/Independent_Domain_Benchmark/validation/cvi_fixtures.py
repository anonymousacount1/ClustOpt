"""Deterministic toy fixtures for modern-CVI reference validation.

**None of these is a scientific benchmark dataset.** They exist solely to compare
our Python implementations against external reference implementations, and no
value computed on them may influence the frozen 50-dataset corpus, the protocol,
or any hypothesis.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

SEED = 20260911
#: predeclared agreement tolerances, frozen BEFORE any comparison was run
TOLERANCE = {"cvnn": {"atol": 1e-9, "rtol": 1e-9},
             "cvdd": {"atol": 1e-8, "rtol": 1e-8},
             "dcsi": {"atol": 1e-8, "rtol": 1e-8},
             "cdbw": {"atol": 1e-12, "rtol": 1e-12}}


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def build() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """fixture_id -> (X, labels). Labels are 1-based to match R conventions."""
    r = _rng()
    F: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # 1. clearly separated spherical clusters
    X = np.vstack([r.normal([0, 0], 0.25, (40, 2)),
                   r.normal([6, 0], 0.25, (40, 2)),
                   r.normal([3, 6], 0.25, (40, 2))])
    F["separated_3"] = (X, np.repeat([1, 2, 3], 40))

    # 2. overlapping clusters
    X = np.vstack([r.normal([0, 0], 1.5, (40, 2)),
                   r.normal([1.2, 0.8], 1.5, (40, 2)),
                   r.normal([0.6, 1.6], 1.5, (40, 2))])
    F["overlapping_3"] = (X, np.repeat([1, 2, 3], 40))

    # 3. anisotropic (sheared) clusters
    B = np.vstack([r.normal([0, 0], 0.6, (40, 2)),
                   r.normal([4, 0], 0.6, (40, 2))])
    F["anisotropic_2"] = (B @ np.array([[1.0, 0.75], [0.3, 1.0]]),
                          np.repeat([1, 2], 40))

    # 4. unequal density
    X = np.vstack([r.normal([0, 0], 0.2, (60, 2)),
                   r.normal([4, 4], 1.4, (30, 2))])
    F["unequal_density_2"] = (X, np.repeat([1, 2], [60, 30]))

    # 5. non-convex ring plus core
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ring = np.c_[4 * np.cos(t), 4 * np.sin(t)] + r.normal(0, 0.12, (60, 2))
    core = r.normal([0, 0], 0.35, (40, 2))
    F["ring_and_core"] = (np.vstack([ring, core]),
                          np.repeat([1, 2], [60, 40]))

    # 6. duplicate points
    base = r.normal([0, 0], 0.3, (20, 2))
    X = np.vstack([base, base, r.normal([5, 5], 0.3, (40, 2))])
    F["duplicates_2"] = (X, np.repeat([1, 2], [40, 40]))

    # 7. singleton cluster
    X = np.vstack([r.normal([0, 0], 0.3, (40, 2)),
                   r.normal([5, 5], 0.3, (39, 2)),
                   np.array([[12.0, 12.0]])])
    F["singleton_3"] = (X, np.repeat([1, 2, 3], [40, 39, 1]))

    # 8. degenerate single cluster (invalid for every index here)
    F["one_cluster"] = (r.normal([0, 0], 1.0, (60, 2)), np.ones(60, int))

    # 9. four clusters, different K
    X = np.vstack([r.normal(c, 0.3, (25, 2))
                   for c in ([0, 0], [5, 0], [0, 5], [5, 5])])
    F["separated_4"] = (X, np.repeat([1, 2, 3, 4], 25))

    # 10. touching clusters
    X = np.vstack([r.normal([0, 0], 0.8, (40, 2)),
                   r.normal([1.9, 0], 0.8, (40, 2))])
    F["touching_2"] = (X, np.repeat([1, 2], 40))

    # 11. three elongated parallel bands (K=3, non-spherical but convex)
    X = np.vstack([np.c_[r.uniform(-4, 4, 40), r.normal(c, 0.18, 40)]
                   for c in (-2.0, 0.0, 2.0)])
    F["elongated_3"] = (X, np.repeat([1, 2, 3], 40))

    # 12. three concentric rings (K=3, strongly non-convex; connectedness
    #     must not be confused with compactness)
    rings = []
    for rad, m in ((1.0, 40), (3.0, 60), (5.0, 80)):
        a = np.linspace(0, 2 * np.pi, m, endpoint=False)
        rings.append(np.c_[rad * np.cos(a), rad * np.sin(a)]
                     + r.normal(0, 0.10, (m, 2)))
    F["nested_rings_3"] = (np.vstack(rings), np.repeat([1, 2, 3], [40, 60, 80]))

    # 13. four clusters with very different densities and sizes (K=4)
    X = np.vstack([r.normal([0, 0], 0.15, (50, 2)),
                   r.normal([5, 0], 0.90, (30, 2)),
                   r.normal([0, 5], 0.35, (40, 2)),
                   r.normal([5, 5], 0.55, (25, 2))])
    F["mixed_density_4"] = (X, np.repeat([1, 2, 3, 4], [50, 30, 40, 25]))

    # 14. two interleaved moons (K=2, the canonical density-separable case)
    a = np.linspace(0, np.pi, 45)
    m1 = np.c_[np.cos(a), np.sin(a)] + r.normal(0, 0.08, (45, 2))
    m2 = np.c_[1 - np.cos(a), 0.5 - np.sin(a)] + r.normal(0, 0.08, (45, 2))
    F["two_moons_2"] = (np.vstack([m1, m2]), np.repeat([1, 2], 45))
    return F


def comparison_sets() -> Dict[str, List[np.ndarray]]:
    """Alternative labelings per fixture, so CVNN has a set to normalise over.

    CVNN is only defined relative to a compared set of candidate clusterings, so
    every fixture is paired with deterministic alternatives.
    """
    r = np.random.default_rng(SEED + 1)
    out: Dict[str, List[np.ndarray]] = {}
    for fid, (X, y) in build().items():
        n = len(y)
        alts = [y.copy()]
        # a deterministic 2-way split by the first coordinate
        alts.append((X[:, 0] > np.median(X[:, 0])).astype(int) + 1)
        # a deterministic 3-way split by angle
        ang = np.arctan2(X[:, 1] - X[:, 1].mean(), X[:, 0] - X[:, 0].mean())
        alts.append(np.digitize(ang, np.quantile(ang, [1 / 3, 2 / 3])) + 1)
        # a fixed permutation-based labelling (stable, seeded)
        alts.append(r.permutation(np.repeat([1, 2], [n // 2, n - n // 2])))
        out[fid] = alts
    return out
