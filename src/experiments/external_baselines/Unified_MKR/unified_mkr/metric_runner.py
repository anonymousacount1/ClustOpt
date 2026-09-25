"""Compute the full de-duplicated internal-metric vector for a partition.

For one view (``X_decision`` / ``X_full``) a :class:`ViewMetricEngine` is built
once. It owns:

* a ClustOpt :class:`GenericCVIEvaluator` over the 60 final utility-map metrics,
* a ClustOpt :class:`GlobalMetricContext` (shared across the 32 partitions of
  the view — this is the reuse of ClustOpt's in-memory metric cache), and
* the ML2DAC-only CVI adapter (Dunn / Coggins-Jain / COP).

Per partition it returns one value per unified metric. ClustOpt metrics keep
their raw ``evaluate()`` value (``-inf`` sentinels -> NaN); ML2DAC-only CVIs are
computed on the noise-filtered partition, mirroring ``remove_noise=True``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from . import ml2dac_cvi_adapter
from .configuration import get_cvi_metric_names, get_cvi_params


class ViewMetricEngine:
    def __init__(
        self,
        config_root: str,
        X_decision: np.ndarray,
        X_full: np.ndarray,
        *,
        repo_root: Optional[str] = None,
    ):
        from models.ClustOpt.cluster_validity_indices.cvi_implementations.cvi_generic import (
            GenericCVIEvaluator,
        )
        from models.ClustOpt.contexts import GlobalMetricContext

        self.clustopt_names: List[str] = get_cvi_metric_names(config_root)
        params = get_cvi_params(config_root)
        self.remove_noise = bool(params.get("remove_noise", True))
        self.noise_label = int(params.get("noise_label", -1))
        self.min_clusters = int(params.get("min_clusters", 2))

        self.evaluator = GenericCVIEvaluator(
            metrics=self.clustopt_names,
            remove_noise=self.remove_noise,
            noise_label=self.noise_label,
            min_clusters=self.min_clusters,
        )
        self.X_decision = X_decision
        self.X_full = X_full
        self.global_ctx = GlobalMetricContext(X_decision, X_full, runtime_config=None)
        self.ml2dac_only_names = ml2dac_cvi_adapter.required_metric_names()
        self.repo_root = repo_root
        self._subset_evaluators: Dict[tuple, object] = {}

    @property
    def metric_names(self) -> List[str]:
        return self.clustopt_names + self.ml2dac_only_names

    # ----------------------------------------------------------------- helpers
    def _clustopt_raw(self, labels: np.ndarray, names: List[str]) -> Dict[str, float]:
        """Compute a subset of ClustOpt metrics (raw), reusing the global cache."""
        from models.ClustOpt.cluster_validity_indices.cvi_implementations.cvi_generic import (
            GenericCVIEvaluator,
        )
        from models.ClustOpt.contexts import PartitionMetricContext

        if not names:
            return {}
        key = tuple(names)
        evaluator = self._subset_evaluators.get(key)
        if evaluator is None:
            if list(names) == list(self.clustopt_names):
                evaluator = self.evaluator
            else:
                evaluator = GenericCVIEvaluator(
                    metrics=list(names),
                    remove_noise=self.remove_noise,
                    noise_label=self.noise_label,
                    min_clusters=self.min_clusters,
                )
            self._subset_evaluators[key] = evaluator
        partition_ctx = PartitionMetricContext(
            labels, self.global_ctx, noise_label=self.noise_label
        )
        raw = evaluator.evaluate(
            labels, self.X_decision, self.X_full,
            context=self.global_ctx, partition_context=partition_ctx,
        )
        out: Dict[str, float] = {}
        for name in names:
            val = raw.get(name, float("nan"))
            out[name] = float(val) if val is not None and math.isfinite(val) else float("nan")
        return out

    def _ml2dac(self, labels: np.ndarray) -> Dict[str, float]:
        if self.remove_noise:
            mask = labels != self.noise_label
        else:
            mask = np.ones(labels.shape[0], dtype=bool)
        vals = ml2dac_cvi_adapter.compute_ml2dac_only(
            self.X_decision[mask], labels[mask], repo_root=self.repo_root
        )
        return {n: vals.get(n, float("nan")) for n in self.ml2dac_only_names}

    # --------------------------------------------------------------- public API
    def compute(self, labels: np.ndarray) -> Dict[str, float]:
        """Full compute: all 60 ClustOpt metrics + 3 ML2DAC CVIs (raw)."""
        labels = np.asarray(labels)
        out = self._clustopt_raw(labels, self.clustopt_names)
        out.update(self._ml2dac(labels))
        return out

    def compute_reused(
        self, labels: np.ndarray, match
    ) -> Dict[str, float]:
        """Reuse the precomputed ClustOpt metrics in ``match``; compute the rest.

        Reuses every metric stored in ``match.reused`` (exact-name CSV columns)
        and freshly computes only the metrics absent from the CSV plus the 3
        ML2DAC-only CVIs. The caller is responsible for having checked
        ``PrecomputedView.guard_ok`` first.
        """
        labels = np.asarray(labels)
        out: Dict[str, float] = dict(match.reused)
        missing = [m for m in self.clustopt_names if m not in out]
        out.update(self._clustopt_raw(labels, missing))
        out.update(self._ml2dac(labels))
        for m in self.clustopt_names:
            out.setdefault(m, float("nan"))
        return out
