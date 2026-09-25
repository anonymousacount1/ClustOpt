# metrics/pattern/ladderness.py
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class LaddernessMetric(BaseMetric):
    """
    Ladderness (Step/Bands) metric.
    Scores how much clusters exhibit multiple discrete levels (bands) across the axis
    orthogonal to their main direction.

    Returns [0,1], higher is better.
    """
    name = "ladderness"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        bins: int = 64,
        min_rel_height: float = 0.05,
        peak_distance_bins: int = 2,
        tau_peaks: float = 3.0,
        elong_r0: float = 2.0,
        fallback_weight: float = 0.25,
        min_cluster_points: int = 8,
        eps: float = 1e-12,
        adaptive_bins: bool = True,
    ):
        self.bins = int(bins)
        self.min_rel_height = float(min_rel_height)
        self.peak_distance_bins = int(peak_distance_bins)
        self.tau_peaks = float(tau_peaks)
        self.elong_r0 = float(elong_r0)
        self.fallback_weight = float(fallback_weight)
        self.min_cluster_points = int(min_cluster_points)
        self.eps = float(eps)
        self.adaptive_bins = bool(adaptive_bins)

    # ---------------- helpers ----------------
    def _pca_rotate_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for ladderness.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]  # stable 2D projection
        return Z

    def _smooth_hist(self, hist: np.ndarray) -> np.ndarray:
        if hist.size < 5:
            return hist.astype(float)
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        return np.convolve(hist, kernel, mode="same")

    def _choose_bins(self, n: int) -> int:
        if not self.adaptive_bins:
            return self.bins
        # simple robust heuristic: sqrt(n), clipped
        b = int(np.clip(np.sqrt(max(n, 1)), 16, 96))
        return b

    def _band_strength(self, y: np.ndarray) -> tuple[float, int]:
        if y.size < 5:
            return 0.0, 0

        bins = self._choose_bins(int(y.size))
        hist, _ = np.histogram(y, bins=bins)
        hist = self._smooth_hist(hist)

        hmax = float(hist.max()) if hist.size else 0.0
        if hmax <= 0.0:
            return 0.0, 0

        height = self.min_rel_height * hmax
        peaks, props = find_peaks(hist, height=height, distance=self.peak_distance_bins)
        K = int(peaks.size)
        if K == 0:
            return 0.0, 0

        heights = props.get("peak_heights", np.zeros(K, dtype=float))
        heights_norm = np.clip(heights / (hmax + self.eps), 0.0, 1.0)
        mean_height = float(np.mean(heights_norm))

        K_strength = 1.0 - np.exp(-K / max(self.tau_peaks, self.eps))
        strength = float(np.clip(K_strength * mean_height, 0.0, 1.0))
        return strength, K

    def _elongation_score(self, Z: np.ndarray) -> float:
        if Z.shape[0] < 3:
            return 0.0
        x = Z[:, 0]
        y = Z[:, 1]
        s1 = float(np.std(x))
        s2 = float(np.std(y)) + self.eps
        ratio = s1 / s2
        score = 1.0 - np.exp(-ratio / max(self.elong_r0, self.eps))
        return float(np.clip(score, 0.0, 1.0))

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._pca_rotate_2d(Xi)  # (n,2)
        band_strength, K = self._band_strength(Z[:, 1])

        e = self._elongation_score(Z)
        if K <= 1:
            # no discrete levels -> small fallback based on elongation
            return float(np.clip(self.fallback_weight * e, 0.0, 1.0))

        score = band_strength * e
        return float(np.clip(score, 0.0, 1.0))

    # ---------------- contract ----------------
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0]:
            return float("-inf")

        uniq = np.unique(labels)
        if uniq.size < 1:
            return float("-inf")

        total = 0.0
        total_w = 0.0
        for lab in uniq:
            Xi = X[labels == lab]
            s = self._cluster_score(Xi)
            if not np.isfinite(s) or s == float("-inf"):
                continue
            w = float(Xi.shape[0])
            total += w * s
            total_w += w

        if total_w <= 0.0:
            return float("-inf")

        return float(total / total_w)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
