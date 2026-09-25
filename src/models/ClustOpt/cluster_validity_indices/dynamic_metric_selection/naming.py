"""Metric-name <-> registry-key resolution.

The MLP regressor targets and the oracle utility vectors are keyed by each
metric's ``.name`` attribute (e.g. ``convexity_solidity_image``). The ClustOpt
metric *registry*, however, is keyed by config-facing keys (e.g.
``convexity_ratio_image_based``). For most metrics the two coincide, but a
handful differ. The generic CVI evaluator selects metrics by *registry key*
yet looks up per-metric weights by the metric's ``.name``.

These helpers bridge the two namespaces so a dynamically resolved utility
vector (named by ``.name``) can be handed to :class:`GenericCVIEvaluator`
without ``KeyError``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
    METRIC_REGISTRY,
)


def strip_prefix(name: str, prefix: str = "utility__") -> str:
    """Remove ``prefix`` from ``name`` if present (e.g. ``utility__silhouette``)."""
    if prefix and name.startswith(prefix):
        return name[len(prefix):]
    return name


@lru_cache(maxsize=1)
def build_name_to_key() -> Dict[str, str]:
    """Map each metric's ``.name`` attribute to its registry key.

    The mapping is a clean bijection over the current registry (verified at
    import time). Cached because the registry is static for the process.
    """
    name_to_key: Dict[str, str] = {}
    for key, metric in METRIC_REGISTRY.items():
        name = getattr(metric, "name", key)
        name_to_key[name] = key
    return name_to_key


def resolve_metric_key(name: str) -> Optional[str]:
    """Resolve a (possibly ``.name``-style) metric name to a registry key.

    Returns ``None`` if the metric is unknown so callers can skip it gracefully
    rather than crashing the whole evaluation.
    """
    if name in METRIC_REGISTRY:
        return name
    return build_name_to_key().get(name)
