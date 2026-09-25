"""Generic CVI implementation that delegates to a configurable set of metrics.

The evaluator follows the same three-stage contract as :class:`BaseCVI`:

* :meth:`evaluate` computes raw per-metric values, choosing the appropriate
  data view (decision vs. full) and respecting whether the metric needs the
  full original labels (noise included) or the filtered labels.
* :meth:`normalize_scores` deterministically squashes each value to ``[0, 1]``
  using the metric's own :meth:`safe_normalize`.
* :meth:`aggregate_score` returns the weighted mean of the normalised values.

Metric *implementations* are intentionally untouched: the orchestrator only
prepares inputs and aggregates outputs.

The :meth:`evaluate` method also accepts optional ``context`` and
``partition_context`` keyword arguments. When supplied (typically by the
search algorithm), they are forwarded to each metric's
:meth:`BaseMetric.safe_evaluate` so optimised metrics can reuse cached
artefacts. Callers that don't pass contexts keep the original behaviour.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.ClustOpt.cluster_validity_indices.cvi_base import BaseCVI
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import (
    BaseMetric,
    MetricContext,
)
from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
    METRIC_REGISTRY,
)


class GenericCVIEvaluator(BaseCVI):
    """Weighted-mean CVI evaluator built on top of the metric registry."""

    def __init__(
        self,
        metrics: List[str],
        weights: Optional[Dict[str, float]] = None,
        remove_noise: bool = True,
        noise_label: int = -1,
        min_clusters: int = 2,
    ):
        if not metrics:
            raise ValueError("'metrics' must contain at least one metric name.")

        unknown = [m for m in metrics if m not in METRIC_REGISTRY]
        if unknown:
            raise KeyError(
                f"Unknown metric(s): {unknown}. "
                f"Known metrics: {sorted(METRIC_REGISTRY.keys())}."
            )

        self.metric_names: List[str] = list(metrics)
        self.metrics: List[BaseMetric] = [METRIC_REGISTRY[m] for m in metrics]
        self.weights: Dict[str, float] = dict(weights or {})
        self.remove_noise = bool(remove_noise)
        self.noise_label = int(noise_label)
        self.min_clusters = int(min_clusters)

    # ----------------------------------------------------------------- helpers

    def _noise_mask(self, labels: np.ndarray) -> np.ndarray:
        """Return a boolean mask keeping only non-noise points."""
        if not self.remove_noise:
            return np.ones(labels.shape[0], dtype=bool)
        return labels != self.noise_label

    @staticmethod
    def _apply_mask(
        X: np.ndarray, labels: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return X[mask], labels[mask]

    # -------------------------------------------------------------- evaluation

    def evaluate(
        self,
        labels: np.ndarray,
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
        *,
        context: Optional[Any] = None,
        partition_context: Optional[Any] = None,
    ) -> Dict[str, float]:
        labels = np.asarray(labels)

        # Unfiltered views (metrics that need original noise-aware labels).
        Xd_orig = X_decision
        Xf_orig = X_decision if X_full is None else X_full

        # Filtered views: noise removed. If the caller passes the *same*
        # array for ``X_decision`` and ``X_full`` (common when there's only
        # one feature space, e.g. the AMP-partition runner), we reuse the
        # single filtered slice. That share lets per-partition caches
        # downstream collide on ``id(X)`` instead of producing two
        # equal-but-distinct arrays.
        mask = self._noise_mask(labels)
        Xd_filt, ld_filt = self._apply_mask(Xd_orig, labels, mask)
        if X_full is None or X_full is X_decision:
            Xf_filt, lf_filt = Xd_filt, ld_filt
        else:
            Xf_filt, lf_filt = self._apply_mask(Xf_orig, labels, mask)

        # Validity check based on what regular metrics see (post-filter).
        if np.unique(ld_filt).size < self.min_clusters:
            return {m.name: float("-inf") for m in self.metrics}

        raw: Dict[str, float] = {}
        for m in self.metrics:
            use_full = (m.space == "full") and (X_full is not None)
            if getattr(m, "needs_original_labels", False):
                X = Xf_orig if use_full else Xd_orig
                y = labels
            else:
                X = Xf_filt if use_full else Xd_filt
                y = lf_filt if use_full else ld_filt
            raw[m.name] = m.safe_evaluate(
                X, y, context=context, partition_context=partition_context,
            )
        return raw

    def normalize_scores(
        self,
        raw_scores: Dict[str, float],
        labels: np.ndarray,
    ) -> Dict[str, float]:
        labels = np.asarray(labels)
        mask = self._noise_mask(labels)
        filtered = labels[mask]
        ctx = MetricContext(
            n_samples=int(filtered.shape[0]),
            n_clusters=int(np.unique(filtered).size),
        )

        return {
            m.name: m.safe_normalize(raw_scores.get(m.name, float("-inf")), ctx)
            for m in self.metrics
        }

    def aggregate_score(self, normalized_scores: Dict[str, float]) -> float:
        total_w = 0.0
        acc = 0.0
        for m in self.metrics:
            w = float(self.weights.get(m.name, 1.0))
            total_w += w
            acc += w * float(normalized_scores.get(m.name, 0.0))
        return acc / total_w if total_w > 0 else 0.0
