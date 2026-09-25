"""OPTICS-clustering wrapper for the ClustOpt search space."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.cluster import OPTICS

from models.ClustOpt.clustering_algorithms.abstract_clustering import ClusteringAlgorithm


class OPTICSClusteringWrapper(ClusteringAlgorithm):
    def instantiate(self, params: Dict[str, Any]):
        sk_params = dict(params)
        # OPTICS only honours `p` when the metric is `minkowski`; drop it otherwise
        # to avoid sklearn raising `ValueError: 'p' is only valid for ...`.
        if sk_params.get("metric", "minkowski") != "minkowski":
            sk_params.pop("p", None)

        self.model = OPTICS(**sk_params)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)
