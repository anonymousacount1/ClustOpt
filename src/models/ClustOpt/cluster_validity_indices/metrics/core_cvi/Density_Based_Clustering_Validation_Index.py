# metrics/dbcv.py
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext, BaseMetric


_MIN_CLUSTER_KEEP = 2  # never let a real cluster drop below this in the sample


def _stratified_subsample(
    labels: np.ndarray,
    *,
    sample_size: int,
    noise_label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices for a stratified subsample of size ~``sample_size``.

    Strategy: drop noise points first, then take the same proportional slice
    from every real cluster (with a floor of ``_MIN_CLUSTER_KEEP`` per cluster
    so small clusters aren't lost). The result keeps the cluster cardinality
    correct so DBCV's per-cluster scoring stays well-defined.
    """
    labels = np.asarray(labels)
    valid_mask = labels != noise_label
    valid_idx = np.where(valid_mask)[0]
    valid_labels = labels[valid_idx]
    n_valid = valid_idx.size
    if n_valid <= sample_size:
        return valid_idx

    parts = []
    for lab in np.unique(valid_labels):
        cluster_idx = valid_idx[valid_labels == lab]
        m = cluster_idx.size
        k = int(round(sample_size * m / n_valid))
        k = max(_MIN_CLUSTER_KEEP, min(k, m))
        parts.append(rng.choice(cluster_idx, size=k, replace=False))
    return np.concatenate(parts)


class DBCVMetric(BaseMetric):
    name = "dbcv"
    space = "decision"
    higher_is_better = True

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        import hdbscan  # type: ignore
        # hdbscan.validity.validity_index returns score typically in [-1,1]
        return float(hdbscan.validity.validity_index(X, labels))

    def evaluate_ctx(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        context: Optional[Any] = None,
        partition_context: Optional[Any] = None,
    ) -> float:
        # --- Fast mode ---------------------------------------------------
        # Run DBCV on a deterministic stratified subsample of the valid
        # points. Noise filtering and invalid-label handling stay identical
        # to the exact path; the only difference is the row count fed to
        # hdbscan.validity.validity_index.
        if context is not None and getattr(context, "metric_mode", "exact") == "fast":
            X_arr = np.asarray(X)
            labels_arr = np.asarray(labels)
            sample_size = int(context.fast_metric_sample_size)
            rng = np.random.default_rng(int(context.fast_metric_random_state))
            idx = _stratified_subsample(
                labels_arr,
                sample_size=sample_size,
                noise_label=-1,
                rng=rng,
            )
            if idx.size < 2:
                return float("-inf")
            X_sub = X_arr[idx]
            y_sub = labels_arr[idx]
            if np.unique(y_sub).size < 2:
                return float("-inf")
            import hdbscan  # type: ignore
            return float(hdbscan.validity.validity_index(X_sub, y_sub))

        # --- Exact mode (default) ----------------------------------------
        # Precomputed path uses the global X_decision distance matrix sliced
        # by the current noise mask — the same matrix
        # ``noise_aware_silhouette`` consumes, so they share one
        # ``pairwise_distances`` call per search. hdbscan's MST core requires
        # float64 (the global cache already stores that) and ``d`` must be
        # provided when ``metric='precomputed'`` so the validity normalisation
        # factor stays correct.
        if partition_context is not None and X.ndim == 2:
            D = partition_context.filtered_decision_distance()
            if D is not None and D.dtype == np.float64:
                import hdbscan  # type: ignore
                return float(
                    hdbscan.validity.validity_index(
                        D, labels, metric="precomputed", d=int(X.shape[1]),
                    )
                )
        return self.evaluate(X, labels)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # [-1,1] -> [0,1]
        return (raw + 1.0) / 2.0
