"""Agglomerative-clustering wrapper for the ClustOpt search space."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from models.ClustOpt.clustering_algorithms.abstract_clustering import ClusteringAlgorithm


class AgglomerativeClusteringWrapper(ClusteringAlgorithm):
    def instantiate(self, params: Dict[str, Any]):
        sk_params = dict(params)

        linkage = sk_params.get("linkage", "ward")
        if linkage == "ward":
            sk_params["metric"] = "euclidean"
        elif "affinity" in sk_params:
            sk_params["metric"] = sk_params.pop("affinity")

        self.model = AgglomerativeClustering(**sk_params)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)
