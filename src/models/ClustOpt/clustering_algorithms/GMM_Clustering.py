"""Gaussian Mixture Model wrapper for the ClustOpt search space."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.mixture import GaussianMixture

from models.ClustOpt.clustering_algorithms.abstract_clustering import ClusteringAlgorithm


class GMMClustering(ClusteringAlgorithm):
    def instantiate(self, params: Dict[str, Any]):
        return GaussianMixture(**dict(params))
