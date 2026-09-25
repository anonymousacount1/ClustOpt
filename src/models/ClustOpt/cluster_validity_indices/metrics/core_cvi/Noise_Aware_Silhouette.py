# metrics/noise_aware_silhouette.py
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from sklearn.metrics import silhouette_score
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class NoiseAwareSilhouetteMetric(BaseMetric):
    name = "noise_aware_silhouette"
    space = "decision"  # recommendation: keep consistent with silhouette
    higher_is_better = True
    needs_original_labels = True  # <-- key for approach B

    def __init__(self, noise_label: int = -1, alpha: float = 1.0):
        self.noise_label = int(noise_label)
        self.alpha = float(alpha)

    def _core(self, X: np.ndarray, labels: np.ndarray, D: Optional[np.ndarray] = None) -> float:
        labels_arr = np.asarray(labels)
        if labels_arr.size == 0:
            return float("-inf")

        mask = labels_arr != self.noise_label
        if mask.sum() < 2:
            return float("-inf")

        uniq = np.unique(labels_arr[mask])
        if uniq.size < 2:
            return float("-inf")

        if D is not None:
            sil = float(silhouette_score(D, labels_arr[mask], metric="precomputed"))
        else:
            sil = float(silhouette_score(X[mask], labels_arr[mask]))

        noise_ratio = float(np.mean(~mask))
        penalty = (1.0 - noise_ratio) ** self.alpha
        # sil in [-1,1]; penalty in [0,1] -> raw stays in [-1,1]
        return sil * penalty

    def _core_fast_sampled(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        sample_size: int,
        random_state: int,
    ) -> float:
        """Fast-mode core: noise filter exactly, then subsample the survivors."""
        labels_arr = np.asarray(labels)
        if labels_arr.size == 0:
            return float("-inf")

        # Noise handling is identical to the exact path.
        mask = labels_arr != self.noise_label
        if mask.sum() < 2:
            return float("-inf")
        uniq = np.unique(labels_arr[mask])
        if uniq.size < 2:
            return float("-inf")

        Xf = X[mask]
        lf = labels_arr[mask]
        n = lf.shape[0]
        if n > sample_size:
            # sklearn's silhouette_score sampling: deterministic, supported on
            # every version we ship; far cheaper than computing per-cluster
            # strata ourselves and matches the documented fast-mode contract.
            sil = float(silhouette_score(
                Xf, lf,
                sample_size=int(sample_size),
                random_state=int(random_state),
            ))
        else:
            sil = float(silhouette_score(Xf, lf))

        noise_ratio = float(np.mean(~mask))
        penalty = (1.0 - noise_ratio) ** self.alpha
        return sil * penalty

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        return self._core(X, labels, D=None)

    def evaluate_ctx(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        context: Optional[Any] = None,
        partition_context: Optional[Any] = None,
    ) -> float:
        # --- Fast mode -----------------------------------------------------
        if context is not None and getattr(context, "metric_mode", "exact") == "fast":
            return self._core_fast_sampled(
                X, labels,
                sample_size=int(context.fast_metric_sample_size),
                random_state=int(context.fast_metric_random_state),
            )

        # --- Exact mode (default) -----------------------------------------
        D = (
            partition_context.filtered_decision_distance()
            if partition_context is not None
            else None
        )
        return self._core(X, labels, D=D)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        return (raw + 1.0) / 2.0
