"""The validity-index vocabulary used by CLUSTOPT.

CLUSTOPT's inventory is 60 indices: 14 established/classical indices and the 46
structural indices introduced with the method (22 pattern-based, 24
image-based). The comparator indices CDbw, CVDD, CVNN and DCSI used in the
auxiliary external index analysis are NOT part of this inventory.
"""
from __future__ import annotations

from ..common import ensure_src_on_path

ensure_src_on_path()


def registry():
    """Return the name -> metric-object registry of the 60 indices."""
    from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import METRIC_REGISTRY
    return METRIC_REGISTRY


def list_indices():
    return sorted(registry().keys())


def evaluate(name, X, labels):
    """Raw value of one index for one partition (X: n x d array)."""
    return registry()[name].evaluate(X, labels)
