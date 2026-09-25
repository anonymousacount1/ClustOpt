"""ClustOpt: configurable clustering-optimisation pipeline.

Exposes the public builder entry point so callers can do::

    from models.ClustOpt import ClusteringOptimizationBuilder
"""
from __future__ import annotations

import warnings as _warnings

# Mute a single upstream deprecation: the `hdbscan` PyPI package (<=0.8.40)
# still calls `sklearn.utils.check_array(..., force_all_finite=...)`, which
# scikit-learn 1.6 renamed to `ensure_all_finite`. The filter matches only
# that specific message so other FutureWarnings keep surfacing.
_warnings.filterwarnings(
    "ignore",
    message=r".*force_all_finite.*was renamed to.*ensure_all_finite.*",
    category=FutureWarning,
)

from models.ClustOpt.Clustering_Optimization_Builder import ClusteringOptimizationBuilder

__all__ = ["ClusteringOptimizationBuilder"]
