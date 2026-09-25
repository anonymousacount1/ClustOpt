# metrics/pattern/periodicity_1d.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

try:
    from scipy.fft import rfft
except Exception:
    from numpy.fft import rfft  # type: ignore


class Periodicity1DMetric(BaseMetric):
    """
    Periodicity-1D (spectral) metric in [0,1], higher is better.

    For each cluster:
      - PCA rotate to 2D via SVD (no sklearn)
      - build histogram of PC2 (across axis)
      - remove DC, ignore very low-freq bins
      - score = max(non-DC peak) / sum(non-DC)
      - blend with elongation gating

    Aggregate: size-weighted mean over clusters.
    """
    name = "periodicity_1d"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        bins: int = 128,
        adaptive_bins: bool = True,
        min_bins_occupied: int = 8,
        ignore_low_freq_frac: float = 0.02,
        w_elon: float = 0.3,
        elong_r0: float = 2.0,
        min_cluster_points: int = 8,
        eps: float = 1e-12,
    ):
        self.bins = int(bins)
        self.adaptive_bins = bool(adaptive_bins)
        self.min_bins_occupied = int(min_bins_occupied)
        self.ignore_low_freq_frac = float(ignore_low_freq_frac)
        self.w_elon = float(w_elon)
        self.elong_r0 = float(elong_r0)
        self.min_cluster_points = int(min_cluster_points)
        self.eps = float(eps)

    # ---------- helpers ----------
    def _choose_bins(self, n: int) -> int:
        if not self.adaptive_bins:
            return self.bins
        b = int(np.clip(np.sqrt(max(n, 1)), 16, 128))
        return b

    def _pca_rotate_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for periodicity_1d.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _elongation_score(self, Z: np.ndarray) -> float:
        if Z.shape[0] < 3:
            return 0.0
        s1 = float(np.std(Z[:, 0]))
        s2 = float(np.std(Z[:, 1])) + self.eps
        ratio = s1 / s2
        score = 1.0 - np.exp(-(ratio) / max(self.elong_r0, self.eps))
        return float(np.clip(score, 0.0, 1.0))

    def _periodicity_hist_score(self, y: np.ndarray, bins: int) -> float:
        if y.size < self.min_cluster_points:
            return 0.0

        hist, _ = np.histogram(y, bins=bins)
        if np.count_nonzero(hist) < self.min_bins_occupied:
            return 0.0

        h = hist.astype(float)
        if h.sum() <= 0:
            return 0.0

        # Detrend: subtract a smoothed envelope so that gaussian-shaped
        # histograms don't leak energy into the low-freq peak. A box-filter
        # estimate of the local envelope works fine here.
        n = h.size
        win = max(3, n // 8)
        # rolling mean via convolution
        kernel = np.ones(win, dtype=float) / win
        envelope = np.convolve(h, kernel, mode="same")
        d = h - envelope
        if np.allclose(d, 0.0):
            return 0.0

        # Now look at FFT of the detrended signal. A genuine periodic comb
        # produces one strong line in the spectrum; smooth/blob envelopes
        # were absorbed by the detrending step.
        H = np.abs(rfft(d))
        if H.size <= 2:
            return 0.0
        H[0] = 0.0  # DC
        total = float(np.sum(H))
        if total <= self.eps:
            return 0.0
        peak = float(np.max(H))
        return float(np.clip(peak / total, 0.0, 1.0))

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        Z = self._pca_rotate_2d(Xi)
        y = Z[:, 1]
        bins = self._choose_bins(int(y.size))
        s_period = self._periodicity_hist_score(y, bins=bins)
        s_elon = self._elongation_score(Z)

        w = float(np.clip(self.w_elon, 0.0, 1.0))
        score = (1.0 - w) * s_period + w * s_elon
        return float(np.clip(score, 0.0, 1.0))

    # ---------- contract ----------
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0]:
            return float("-inf")
        if X.shape[1] < 2:
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
