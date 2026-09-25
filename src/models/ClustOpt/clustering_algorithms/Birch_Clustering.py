"""Birch wrapper for the ClustOpt search space."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.cluster import Birch

from models.ClustOpt.clustering_algorithms.abstract_clustering import ClusteringAlgorithm


class BirchClusteringWrapper(ClusteringAlgorithm):
    def instantiate(self, params: Dict[str, Any]):
        sk_params = dict(params)
        # `copy` was removed from sklearn's Birch signature; ignore if present
        # so legacy configs keep working.
        sk_params.pop("copy", None)
        self.model = Birch(**sk_params)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit(X).labels_
