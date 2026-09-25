"""Dynamic metric-selection support for ClustOpt CVI evaluators.

This package contains the machinery used by the ``regressor_dynamic`` and
``oracle_dynamic`` CVI types to turn a *predicted* (MLP regressor) or a *true*
(oracle) per-metric utility vector into a concrete ``{metric_name: weight}``
mapping that the existing generic CVI evaluator can consume unchanged.

Nothing in this package touches the generic CVI behaviour, the metric
implementations, or the utility-generation pipeline; it only *reads* their
artifacts and produces a standard weight dictionary.

Public entry points:

* :func:`resolve_regressor_weights` - MLP regressor -> weight dict.
* :func:`resolve_oracle_weights`    - oracle utility vector -> weight dict.
* :class:`RegressorPredictor`       - cached MLP predictor.
"""
from __future__ import annotations

from .naming import resolve_metric_key, strip_prefix, build_name_to_key
from .weighting import select_top_k, compute_weights
from .regressor_predictor import RegressorPredictor, get_regressor_predictor
from .knn_predictor_provider import KNNUtilityProvider, get_knn_provider
from .loaders import load_feature_vector, load_oracle_utilities
from .resolvers import (
    resolve_regressor_weights,
    resolve_oracle_weights,
    resolve_knn_weights,
)

__all__ = [
    "resolve_metric_key",
    "strip_prefix",
    "build_name_to_key",
    "select_top_k",
    "compute_weights",
    "RegressorPredictor",
    "get_regressor_predictor",
    "KNNUtilityProvider",
    "get_knn_provider",
    "load_feature_vector",
    "load_oracle_utilities",
    "resolve_regressor_weights",
    "resolve_oracle_weights",
    "resolve_knn_weights",
]
