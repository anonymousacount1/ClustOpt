# metrics/silhouette.py
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from sklearn.metrics import silhouette_score
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext, BaseMetric


class SilhouetteMetric(BaseMetric):
    name = "silhouette"
    space = "full"
    higher_is_better = True

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        # Default ``n_jobs=1``: on Windows, joblib's loky backend spawns one
        # subprocess per chunk for ``n_jobs > 1`` and the spawn overhead
        # (~100 ms per worker) is larger than the silhouette work itself for
        # the per-config call cadence here. Single-thread is consistently
        # faster end-to-end.
        return float(silhouette_score(X, labels))

    def evaluate_ctx(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        context: Optional[Any] = None,
        partition_context: Optional[Any] = None,
    ) -> float:
        # --- Fast mode (opt-in via runtime.metric_mode='fast') -----------
        # Sample only when n exceeds the cap; below the cap we still
        # produce the exact score, so small partitions never pay an
        # accuracy tax. Uses sklearn's native ``sample_size`` so the
        # subsampling is deterministic and well-tested.
        if context is not None and getattr(context, "metric_mode", "exact") == "fast":
            n = int(np.asarray(labels).shape[0])
            sample_size = int(context.fast_metric_sample_size)
            if n > sample_size:
                return float(silhouette_score(
                    X, labels,
                    sample_size=sample_size,
                    random_state=int(context.fast_metric_random_state),
                ))
            # n <= cap: fall through to exact precomputed path below.

        # --- Exact mode (default) ----------------------------------------
        # ``space='full'`` => slice the global X_full distance matrix by the
        # current noise mask. The full matrix is built once per search.
        if partition_context is not None:
            D = partition_context.filtered_full_distance()
            if D is not None:
                return float(silhouette_score(D, labels, metric="precomputed"))
        return self.evaluate(X, labels)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # raw in [-1,1] -> [0,1]
        return (raw + 1.0) / 2.0
