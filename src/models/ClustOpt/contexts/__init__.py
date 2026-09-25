"""Shared computation contexts for ClustOpt metric evaluation.

Two context objects expose lazy, cached artefacts that the heaviest metrics
can reuse instead of recomputing per call:

* :class:`GlobalMetricContext` — built **once per (X_decision, X_full)**, so
  per-search constants like global rasterisation bounds live here.
* :class:`PartitionMetricContext` — built **once per candidate partition
  (labels)**, so per-cluster artefacts (2D projections, alpha shapes,
  rasters, skeletons, distance transforms) live here.

Both contexts are *opt-in*: metric implementations that don't accept context
keyword args continue to work unchanged. Optimised metrics pull what they
need from the context via the public properties / helper methods.
"""
from __future__ import annotations

from models.ClustOpt.contexts.global_metric_context import GlobalMetricContext
from models.ClustOpt.contexts.partition_metric_context import PartitionMetricContext

__all__ = ["GlobalMetricContext", "PartitionMetricContext"]
