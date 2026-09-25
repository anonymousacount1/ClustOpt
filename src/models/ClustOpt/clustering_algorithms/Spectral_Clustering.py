"""Spectral-clustering wrapper for the ClustOpt search space."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.cluster import SpectralClustering

from models.ClustOpt.clustering_algorithms.abstract_clustering import ClusteringAlgorithm


class SpectralClusteringWrapper(ClusteringAlgorithm):
    def instantiate(self, params: Dict[str, Any]):
        self.model = SpectralClustering(**dict(params))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)
