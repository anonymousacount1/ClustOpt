"""Search-space layer.

Wraps a config dict describing one or more clustering algorithms and their
hyper-parameter ranges into a single :class:`ClusteringSearchSpace` object
that can:

* expose a flat skopt-style search space (with ``{algo}__{param}`` keys),
* sample random configurations for a given algorithm,
* instantiate the corresponding fitted/unfitted estimator from a config row.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Dict, List, Type

import numpy as np
from skopt.space import Categorical, Integer, Real

from models.ClustOpt.clustering_algorithms.Agglomerative_Clustering import (
    AgglomerativeClusteringWrapper,
)
from models.ClustOpt.clustering_algorithms.Birch_Clustering import BirchClusteringWrapper
from models.ClustOpt.clustering_algorithms.DBSCAN_Clustering import DBSCANClustering
from models.ClustOpt.clustering_algorithms.GMM_Clustering import GMMClustering
from models.ClustOpt.clustering_algorithms.HDBSCANConstrained_Clustering import (
    HDBSCANConstrainedClustering,
)
from models.ClustOpt.clustering_algorithms.HDBSCAN_Clustering import HDBSCANClustering
from models.ClustOpt.clustering_algorithms.KMeans_Clustering import KMeansClustering
from models.ClustOpt.clustering_algorithms.MeanShift_Clustering import (
    MeanShiftClusteringWrapper,
)
from models.ClustOpt.clustering_algorithms.MiniBatchKMeans_Clustering import (
    MiniBatchKMeansClusteringWrapper,
)
from models.ClustOpt.clustering_algorithms.OPTICS_Clustering import (
    OPTICSClusteringWrapper,
)
from models.ClustOpt.clustering_algorithms.Spectral_Clustering import (
    SpectralClusteringWrapper,
)


CLUSTERING_ALGORITHM_REGISTRY: Dict[str, Type] = {
    "kmeans": KMeansClustering,
    "minibatch_kmeans": MiniBatchKMeansClusteringWrapper,
    "gmm": GMMClustering,
    "dbscan": DBSCANClustering,
    "hdbscan": HDBSCANClustering,
    "spectral": SpectralClusteringWrapper,
    "agglomerative": AgglomerativeClusteringWrapper,
    "optics": OPTICSClusteringWrapper,
    "meanshift": MeanShiftClusteringWrapper,
    "birch": BirchClusteringWrapper,
}


# Algorithm families that REQUIRE a ``-N`` suffix in config keys, e.g.
# ``hdbscan_constrained-2``, ``hdbscan_constrained-3``. The bare base name
# is intentionally NOT registered: callers must pick an explicit N so the
# number of persistence bins is unambiguous from the algorithm key alone.
_SUFFIXED_FAMILIES: Dict[str, Type] = {
    "hdbscan_constrained": HDBSCANConstrainedClustering,
}


def resolve_algorithm_class(name: str) -> Type | None:
    """Resolve an algorithm config-key to its wrapper class.

    Supports literal registry keys (``"hdbscan"``, ``"kmeans"``, ...) and
    suffixed variants of families listed in :data:`_SUFFIXED_FAMILIES`
    (``"hdbscan_constrained-2"``, ``"hdbscan_constrained-3"``,
    ``"hdbscan_constrained-N"`` with ``N >= 2``). The bare base name of a
    suffixed family (``"hdbscan_constrained"`` on its own) does not resolve.
    """
    if name in CLUSTERING_ALGORITHM_REGISTRY:
        return CLUSTERING_ALGORITHM_REGISTRY[name]
    for base, cls in _SUFFIXED_FAMILIES.items():
        prefix = f"{base}-"
        if not name.startswith(prefix):
            continue
        try:
            n = int(name[len(prefix):])
        except ValueError:
            continue
        if n >= 2:
            return cls
    return None


def unwrap_single_value(val: Any) -> Any:
    """Collapse a single-element sequence (numpy / list / generic Sequence) to its scalar."""
    if isinstance(val, (list, np.ndarray, Sequence)) and not isinstance(val, str):
        if len(val) == 1:
            return val[0]
    return val


class ClusteringSearchSpace:
    """Container for the algorithm wrappers participating in the search."""

    def __init__(self, config_data: Dict[str, Any]):
        if not config_data:
            raise ValueError("Search space configuration must contain at least one algorithm.")

        resolved: Dict[str, Type] = {}
        unknown: List[str] = []
        for name in config_data:
            cls = resolve_algorithm_class(name)
            if cls is None:
                unknown.append(name)
            else:
                resolved[name] = cls

        if unknown:
            family_hint = ", ".join(
                f"'{base}-N' with N >= 2 (e.g. '{base}-2', '{base}-3')"
                for base in sorted(_SUFFIXED_FAMILIES.keys())
            )
            raise ValueError(
                f"Unsupported clustering algorithm(s): {unknown}. "
                f"Known algorithms: {sorted(CLUSTERING_ALGORITHM_REGISTRY.keys())}. "
                f"Suffixed families: {family_hint}."
            )

        self.algorithms = {
            name: resolved[name](name, param_cfg)
            for name, param_cfg in config_data.items()
        }

    # ------------------------------------------------------------- factory API

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "ClusteringSearchSpace":
        return cls(config_data)

    # ------------------------------------------------------------------ views

    def get_all_algorithms(self) -> List[str]:
        return list(self.algorithms.keys())

    def get_full_search_space(self) -> Dict[str, Any]:
        space: Dict[str, Any] = {"algorithm": Categorical(list(self.algorithms.keys()))}
        for name, algo in self.algorithms.items():
            for param_name, param_space in algo.get_search_space().items():
                space[f"{name}__{param_name}"] = param_space
        return space

    # --------------------------------------------------------- random sampling

    def sample_random_config(self, algo_name: str) -> Dict[str, Any]:
        if algo_name not in self.algorithms:
            raise ValueError(
                f"Algorithm '{algo_name}' not found in search space. "
                f"Known: {sorted(self.algorithms.keys())}."
            )

        algo = self.algorithms[algo_name]
        config: Dict[str, Any] = {"algorithm": algo_name}

        for param_name, param_space in algo.get_search_space().items():
            key = f"{algo_name}__{param_name}"
            config[key] = self._sample_single(param_space)
        return config

    @staticmethod
    def _sample_single(param_space: Any) -> Any:
        if isinstance(param_space, (Integer, Real)):
            if param_space.low == param_space.high:
                return param_space.low
            return unwrap_single_value(param_space.rvs())
        if isinstance(param_space, Categorical):
            return unwrap_single_value(param_space.rvs())
        if isinstance(param_space, list):
            return random.choice(param_space)
        return param_space

    # ---------------------------------------------------------- instantiation

    def instantiate_from_config(self, config: Dict[str, Any]):
        algo_name = config.get("algorithm")
        if algo_name not in self.algorithms:
            raise ValueError(
                f"Config refers to algorithm '{algo_name}' which is not in the search space."
            )
        algo = self.algorithms[algo_name]
        prefix = f"{algo_name}__"
        params = {k[len(prefix):]: v for k, v in config.items() if k.startswith(prefix)}
        return algo.instantiate(params)
