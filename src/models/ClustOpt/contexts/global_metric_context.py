"""Per-search lazy cache for shared metric inputs.

Built once per ``(X_decision, X_full)`` pair by the search algorithm and
shared across every configuration evaluated in that search. All fields are
computed on first access; nothing is eagerly materialised.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


# Memory guard: don't materialise an n x n distance matrix above this.
# 5 000 points => ~190 MB at float64. We keep the cap there because the
# bigger win — silhouette / noise_aware_silhouette / DBCV sharing one
# matrix — is dominated on very large data by the per-configuration slice
# cost ( ``D[np.ix_(idx, idx)]`` is an O(n²) memcpy that scales worse than
# the savings on a ~10 000-point view). For larger views, sklearn /
# hdbscan's chunked internal paths are net-faster.
_MAX_PAIRWISE_N = 5_000


def _identity_key(X) -> int:
    return id(X)


class GlobalMetricContext:
    """Caches dataset-level artefacts that don't depend on the partition."""

    __slots__ = (
        "X_decision",
        "X_full",
        "_pairwise_by_id",
        "_bounds_full",
        "_bounds_decision",
        "_pca_full",
        "runtime_config",
    )

    def __init__(
        self,
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
        *,
        runtime_config: Optional[Dict[str, Any]] = None,
    ):
        self.X_decision = np.asarray(X_decision)
        self.X_full = np.asarray(X_decision if X_full is None else X_full)
        # Cache keyed by ``id(X)``: silhouette / noise_aware_silhouette / DBCV
        # all want the pairwise distance matrix on the same underlying array,
        # which never changes across configurations. The matrix is therefore
        # built once per *search*, not per *configuration*.
        self._pairwise_by_id: Dict[int, Optional[np.ndarray]] = {}
        self._bounds_full: Optional[Tuple[float, float, float, float]] = None
        self._bounds_decision: Optional[Tuple[float, float, float, float]] = None
        self._pca_full: Optional[np.ndarray] = None
        # Optional runtime knobs (fast-mode subsampling, seeds, ...). Read by
        # individual metrics that opt-in via ``evaluate_ctx``; metrics that
        # don't reference them get exact behaviour.
        self.runtime_config: Dict[str, Any] = dict(runtime_config or {})

    # -------------------------------------------------------- runtime helpers
    @property
    def metric_mode(self) -> str:
        """``'exact'`` (default) or ``'fast'``."""
        return str(self.runtime_config.get("metric_mode", "exact")).lower()

    @property
    def fast_metric_sample_size(self) -> int:
        return int(self.runtime_config.get("fast_metric_sample_size", 3000))

    @property
    def fast_metric_random_state(self) -> int:
        return int(self.runtime_config.get("fast_metric_random_state", 42))

    # ---------------------------------------------------- pairwise distances
    def pairwise_distance_matrix(self, *, space: str = "full") -> Optional[np.ndarray]:
        """Return the cached pairwise distance matrix for ``X_full`` /
        ``X_decision``. Convenience wrapper around :meth:`distance_for`."""
        if space == "full":
            return self.distance_for(self.X_full)
        if space == "decision":
            return self.distance_for(self.X_decision)
        raise ValueError(f"Unknown space: {space!r}")

    def distance_for(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Cache the pairwise distance matrix on the given array.

        Returns ``None`` when the array is above the memory guard. Always
        returns ``float64`` so that DBCV's Cython MST core (which requires
        double precision) can consume the matrix unchanged.
        """
        key = id(X)
        if key in self._pairwise_by_id:
            return self._pairwise_by_id[key]
        if X.shape[0] > _MAX_PAIRWISE_N:
            self._pairwise_by_id[key] = None
            return None
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X)
        if D.dtype != np.float64:
            D = D.astype(np.float64, copy=False)
        self._pairwise_by_id[key] = D
        return D

    # ----------------------------------------------------------- raster bounds
    def bounds(self, *, space: str = "full") -> Tuple[float, float, float, float]:
        """Cached version of :func:`compute_global_bounds`."""
        if space == "full":
            if self._bounds_full is None:
                self._bounds_full = self._compute_bounds(self.X_full)
            return self._bounds_full
        if space == "decision":
            if self._bounds_decision is None:
                self._bounds_decision = self._compute_bounds(self.X_decision)
            return self._bounds_decision
        raise ValueError(f"Unknown space: {space!r}")

    @staticmethod
    def _compute_bounds(X: np.ndarray, eps: float = 1e-9) -> Tuple[float, float, float, float]:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] < 2:
            # 1D input: return degenerate-ish bounds; image metrics check for this.
            x = X.reshape(-1)
            xmin, xmax = float(np.min(x)), float(np.max(x))
            if abs(xmax - xmin) < eps:
                xmax = xmin + eps
            return (xmin, xmax, xmin, xmax)
        xmin = float(np.min(X[:, 0])); xmax = float(np.max(X[:, 0]))
        ymin = float(np.min(X[:, 1])); ymax = float(np.max(X[:, 1]))
        if abs(xmax - xmin) < eps:
            xmax = xmin + eps
        if abs(ymax - ymin) < eps:
            ymax = ymin + eps
        return (xmin, xmax, ymin, ymax)

    # ---------------------------------------------------- 2D PCA of X_full
    def pca_full_2d(self) -> Optional[np.ndarray]:
        """Return a 2D PCA projection of X_full, computed lazily once."""
        if self._pca_full is not None:
            return self._pca_full
        X = self.X_full
        if X.ndim != 2 or X.shape[1] < 2:
            return None
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if Xc.shape[1] == 2:
            self._pca_full = Xc.astype(float)
        else:
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            self._pca_full = (U[:, :2] * S[:2]).astype(float)
        return self._pca_full
