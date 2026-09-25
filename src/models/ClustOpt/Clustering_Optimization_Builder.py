"""Top-level builder that assembles a ClustOpt pipeline from a JSON config.

The builder is responsible for:

* parsing and validating the configuration file,
* constructing the :class:`ClusteringSearchSpace`,
* constructing the requested :class:`BaseCVI` evaluator,
* wiring everything into the requested :class:`BaseSearchAlgorithm`.

Two configuration shapes are accepted to remain compatible with older
runners that ship inside this repository:

(1) Modern (recommended) shape::

        {
          "search_space":   {...},
          "search_algorithm": {"type": "optuna", "params": {...}},
          "cvi":            {"type": "generic", "metrics": {...}, "params": {...}}
        }

(2) Legacy shape (still supported)::

        {
          "search_space":            {...},
          "search_algorithm":        "brute_force",
          "search_algorithm_params": {...},
          "cvi":                     {"silhouette": 1.0, "calinski_harabasz": 1.0}
        }

In the legacy CVI shape the type defaults to ``generic`` and the dict is
treated as ``{metric_name: weight}``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from models.ClustOpt.cluster_validity_indices.cvi_base import BaseCVI
from models.ClustOpt.cluster_validity_indices.cvi_registry import CVI_REGISTRY
from models.ClustOpt.search_algorithm.Base_Search_Algorithm import BaseSearchAlgorithm
from models.ClustOpt.search_algorithm.search_registry import SEARCH_ALGORITHM_REGISTRY
from models.ClustOpt.search_space.Clustering_Search_Space import ClusteringSearchSpace


REQUIRED_TOP_LEVEL_KEYS = ("search_space", "search_algorithm", "cvi")


class ClusteringOptimizationBuilder:
    """Build a clustering-optimisation pipeline from a JSON configuration file."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._raw_config = self._load_config(self.config_path)

        for key in REQUIRED_TOP_LEVEL_KEYS:
            if key not in self._raw_config:
                raise ValueError(
                    f"Missing required key '{key}' in configuration file "
                    f"'{self.config_path}'."
                )

        self.search_space_config: Mapping[str, Any] = self._raw_config["search_space"]
        self.cvi_config: Mapping[str, Any] = self._raw_config["cvi"]

        sa_name, sa_params = self._parse_search_algorithm_config(self._raw_config)
        self.search_algorithm_name = sa_name
        self.search_algorithm_params = sa_params

        # Optional ``runtime`` block. When absent every field keeps its exact-mode
        # default and behaviour is bitwise identical to legacy.
        runtime_block = self._raw_config.get("runtime", {})
        if not isinstance(runtime_block, Mapping):
            raise ValueError("'runtime' configuration must be a mapping when present.")
        self.runtime_config: Dict[str, Any] = dict(runtime_block)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _parse_search_algorithm_config(
        config: Mapping[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Resolve both the modern dict form and the legacy string form."""
        sa_cfg = config["search_algorithm"]

        if isinstance(sa_cfg, str):
            name = sa_cfg
            params = dict(config.get("search_algorithm_params", {}))
        elif isinstance(sa_cfg, Mapping):
            if "type" not in sa_cfg:
                raise ValueError(
                    "Invalid 'search_algorithm' configuration: expected a string "
                    "or a mapping containing a 'type' field."
                )
            name = str(sa_cfg["type"])
            params = dict(sa_cfg.get("params", {}))
        else:
            raise ValueError(
                "Invalid 'search_algorithm' configuration type: "
                f"{type(sa_cfg).__name__}. Expected str or mapping."
            )

        return name.lower(), params

    # ----------------------------------------------------------------- builders

    def build_search_space(self) -> ClusteringSearchSpace:
        return ClusteringSearchSpace.from_dict(self.search_space_config)

    def build_evaluator(self) -> BaseCVI:
        cvi_type, metrics_weights, params = self._parse_cvi_config(self.cvi_config)

        if cvi_type not in CVI_REGISTRY:
            raise NotImplementedError(
                f"CVI type '{cvi_type}' is not supported. "
                f"Known types: {sorted(CVI_REGISTRY.keys())}."
            )

        cvi_cls = CVI_REGISTRY[cvi_type]

        # Dynamic CVI types (regressor_dynamic / oracle_dynamic) may carry a
        # separate ``params_generic`` block holding the standard generic-CVI
        # construction options (remove_noise / noise_label / min_clusters).
        # The generic baselines never set this key, so they are unaffected.
        if isinstance(self.cvi_config, Mapping) and "params_generic" in self.cvi_config:
            params = {**params, "params_generic": self.cvi_config["params_generic"]}

        return cvi_cls(
            metrics=list(metrics_weights.keys()),
            weights=dict(metrics_weights),
            **params,
        )

    @staticmethod
    def _parse_cvi_config(
        cvi_cfg: Any,
    ) -> tuple[str, Dict[str, float], Dict[str, Any]]:
        if not isinstance(cvi_cfg, Mapping):
            raise ValueError("Invalid CVI configuration: expected a mapping.")

        # Legacy shape: a plain {metric: weight} dict without a "type" key.
        if "type" not in cvi_cfg:
            return "generic", {str(k): float(v) for k, v in cvi_cfg.items()}, {}

        cvi_type = str(cvi_cfg.get("type", "generic")).lower()
        metrics_cfg = cvi_cfg.get("metrics", {})

        if isinstance(metrics_cfg, Mapping):
            metrics_weights = {str(k): float(v) for k, v in metrics_cfg.items()}
        elif isinstance(metrics_cfg, list):
            metrics_weights = {str(m): 1.0 for m in metrics_cfg}
        else:
            raise ValueError(
                "Invalid CVI 'metrics' format. Expected a mapping "
                "{metric_name: weight} or a list of metric names."
            )

        return cvi_type, metrics_weights, dict(cvi_cfg.get("params", {}))

    def build_search_algorithm(
        self,
        search_space: ClusteringSearchSpace,
        evaluator: BaseCVI,
    ) -> BaseSearchAlgorithm:
        key = self.search_algorithm_name.lower()
        if key not in SEARCH_ALGORITHM_REGISTRY:
            raise NotImplementedError(
                f"Search algorithm '{key}' is not supported. "
                f"Known algorithms: {sorted(SEARCH_ALGORITHM_REGISTRY.keys())}."
            )

        algorithm_cls = SEARCH_ALGORITHM_REGISTRY[key]
        return algorithm_cls(search_space, evaluator, **self.search_algorithm_params)

    def build_pipeline(self) -> BaseSearchAlgorithm:
        search_space = self.build_search_space()
        evaluator = self.build_evaluator()
        sa = self.build_search_algorithm(search_space, evaluator)
        # Propagate the runtime config (e.g. metric_mode='fast') to the search
        # algorithm so it ends up on the GlobalMetricContext at search time.
        # Backward-compatible: when runtime is absent, exact-mode defaults
        # apply throughout.
        sa.runtime_config = dict(self.runtime_config)
        return sa
