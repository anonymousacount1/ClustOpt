"""
neighbor_search.py
==================

Thin wrapper over scikit-learn ``NearestNeighbors`` that queries once up to
``MAX_NEIGHBORS`` and returns the (distance, index) arrays, which the predictor
reuses for every ``n_neighbors`` and every neighbour weighting (plan §13).

Euclidean uses the default fast path; cosine forces brute-force (the only sklearn
algorithm that supports the cosine metric).  Neighbour ordering is by ascending
distance and is deterministic for fixed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import config as C


@dataclass
class NeighborResult:
    distances: np.ndarray   # [Nq, K] ascending
    indices: np.ndarray     # [Nq, K] into the training matrix
    k: int


class ViewNeighborIndex:
    """A fitted neighbour index over one view's (scaled) training features."""

    def __init__(self, distance: str) -> None:
        if distance not in ("euclidean", "cosine"):
            raise ValueError(f"Unknown distance '{distance}'.")
        self.distance = distance
        algorithm = "brute" if distance == "cosine" else "auto"
        self._nn = NearestNeighbors(metric=distance, algorithm=algorithm)
        self._n_train = 0

    def fit(self, x_train_scaled: np.ndarray) -> "ViewNeighborIndex":
        self._n_train = int(x_train_scaled.shape[0])
        self._nn.fit(np.ascontiguousarray(x_train_scaled, dtype=np.float64))
        return self

    def query(
        self, x_query_scaled: np.ndarray, max_neighbors: int = C.MAX_NEIGHBORS
    ) -> NeighborResult:
        k = int(min(max_neighbors, self._n_train))
        dist, idx = self._nn.kneighbors(
            np.ascontiguousarray(x_query_scaled, dtype=np.float64),
            n_neighbors=k,
            return_distance=True,
        )
        return NeighborResult(distances=dist, indices=idx, k=k)
