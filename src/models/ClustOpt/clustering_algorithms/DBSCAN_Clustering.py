"""DBSCAN wrapper for the ClustOpt search space."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.cluster import DBSCAN

from models.ClustOpt.clustering_algorithms.abstract_clustering import ClusteringAlgorithm


class DBSCANClustering(ClusteringAlgorithm):
    def instantiate(self, params: Dict[str, Any]):
        return DBSCAN(**dict(params))
