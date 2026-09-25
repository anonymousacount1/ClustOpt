# metrics/pattern/bimodality_anti_thickness.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

try:
    import cv2
except Exception as e:
    cv2 = None
    _cv2_import_error = e


class BimodalityAntiThicknessMetric(BaseMetric):
    """
    Bimodality (Anti-thickness) score in [0,1], higher is better.

    Idea:
      - Fit a line to each cluster (robust).
      - Compute perpendicular distances d to the line.
      - If the cluster is actually two thin parallel bands merged together,
        distribution of d is bimodal (two peaks).
      - Score combines: peak separation + valley depth + elongation gate.

    Noise is expected to be removed by CVI (remove_noise=True).
    """
    name = "bimodality_anti_thickness"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        min_cluster_points: int = 20,
        bins: int = 80,
        adaptive_bins: bool = True,
        min_bins_occupied: int = 10,
        smooth_kernel: tuple[int, ...] = (1, 2, 3, 2, 1),
        min_peak_rel_height: float = 0.15,   # relative to max hist
        min_peak_distance_bins: int = 6,
        sep_r0: float = 1.0,                 # separation saturation in std units
        w_valley: float = 0.55,
        w_sep: float = 0.45,
        w_elon: float = 0.30,                # blend with elongation
        elong_r0: float = 2.0,
        eps: float = 1e-12,
    ):
        if cv2 is None:
            raise ImportError(f"OpenCV (cv2) is required for BimodalityAntiThicknessMetric: {_cv2_import_error}")

        self.min_cluster_points = int(min_cluster_points)
        self.bins = int(bins)
        self.adaptive_bins = bool(adaptive_bins)
        self.min_bins_occupied = int(min_bins_occupied)

        k = np.array(smooth_kernel, dtype=float)
        self.kernel = k / (k.sum() if k.sum() != 0 else 1.0)

        self.min_peak_rel_height = float(min_peak_rel_height)
        self.min_peak_distance_bins = int(min_peak_distance_bins)
        self.sep_r0 = float(sep_r0)

        self.w_valley = float(w_valley)
        self.w_sep = float(w_sep)
        self.w_elon = float(w_elon)

        self.elong_r0 = float(elong_r0)
        self.eps = float(eps)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for bimodality metric.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _choose_bins(self, n: int) -> int:
        if not self.adaptive_bins:
            return self.bins
        # sqrt(n) clipped is robust for hist-based peak finding
        return int(np.clip(np.sqrt(max(n, 1)), 32, 128))

    def _smooth(self, h: np.ndarray) -> np.ndarray:
        if h.size < self.kernel.size:
            return h.astype(float)
        return np.convolve(h.astype(float), self.kernel, mode="same")

    def _fit_perp_distances(self, Z: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns:
          d: perpendicular distances to fitted line (signed)
          elongation score in [0,1]
        """
        pts = Z.astype(np.float32)
        line = cv2.fitLine(pts, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        vx, vy, x0, y0 = [float(v) for v in line.reshape(-1)]

        u = np.array([vx, vy], dtype=float)
        nu = float(np.linalg.norm(u))
        if nu <= self.eps:
            u = np.array([1.0, 0.0], dtype=float)
        else:
            u /= nu
        nvec = np.array([-u[1], u[0]], dtype=float)

        p0 = np.array([x0, y0], dtype=float)
        delta = Z - p0[None, :]
        t = delta @ u
        d = delta @ nvec  # signed

        # elongation gate similar to other metrics
        s1 = float(np.std(t))
        s2 = float(np.std(d)) + self.eps
        ratio = s1 / s2
        s_elon = float(np.clip(1.0 - np.exp(-ratio / max(self.elong_r0, self.eps)), 0.0, 1.0))

        return d.astype(float), s_elon

    def _find_two_peaks(self, h: np.ndarray) -> tuple[int, int] | None:
        """
        Simple deterministic peak picker without scipy:
        - peak: h[i] >= h[i-1] and h[i] >= h[i+1] and h[i] >= rel_height*max
        - enforce min distance between the two selected peaks
        """
        if h.size < 5:
            return None
        hmax = float(np.max(h))
        if hmax <= self.eps:
            return None
        thr = self.min_peak_rel_height * hmax

        peaks = []
        for i in range(1, h.size - 1):
            if h[i] >= thr and h[i] >= h[i - 1] and h[i] >= h[i + 1]:
                peaks.append(i)

        if len(peaks) < 2:
            return None

        # sort candidates by height descending
        peaks = sorted(peaks, key=lambda i: h[i], reverse=True)

        p1 = peaks[0]
        p2 = None
        for j in peaks[1:]:
            if abs(j - p1) >= self.min_peak_distance_bins:
                p2 = j
                break
        if p2 is None:
            return None

        # order them left->right
        return (p1, p2) if p1 < p2 else (p2, p1)

    def _bimodality_score_from_d(self, d: np.ndarray) -> float:
        """
        Convert signed distances d to bimodality score in [0,1].
        """
        if d.size < self.min_cluster_points:
            return float("-inf")

        sd = float(np.std(d))
        if sd <= self.eps:
            return 0.0

        # standardize (scale invariance)
        z = (d - float(np.mean(d))) / (sd + self.eps)

        bins = self._choose_bins(int(z.size))
        h, edges = np.histogram(z, bins=bins)
        if np.count_nonzero(h) < self.min_bins_occupied:
            return 0.0

        hs = self._smooth(h)

        pair = self._find_two_peaks(hs)
        if pair is None:
            return 0.0

        p1, p2 = pair
        h1 = float(hs[p1])
        h2 = float(hs[p2])
        if max(h1, h2) <= self.eps:
            return 0.0

        # valley between peaks
        valley = float(np.min(hs[p1:p2 + 1])) if p2 > p1 else min(h1, h2)

        # valley depth: 1 when deep valley, 0 when no valley
        valley_score = float(np.clip((min(h1, h2) - valley) / (max(h1, h2) + self.eps), 0.0, 1.0))

        # separation in std units using bin centers
        centers = 0.5 * (edges[:-1] + edges[1:])
        sep = float(abs(centers[p2] - centers[p1]))  # already in std units
        sep_score = float(np.clip(1.0 - np.exp(-sep / max(self.sep_r0, self.eps)), 0.0, 1.0))

        # combine
        wsum = self.w_valley + self.w_sep
        if wsum <= self.eps:
            wsum = 1.0
        score = (self.w_valley * valley_score + self.w_sep * sep_score) / wsum
        return float(np.clip(score, 0.0, 1.0))

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(Xi)
        d, s_elon = self._fit_perp_distances(Z)

        s_bi = self._bimodality_score_from_d(d)

        # blend with elongation to avoid blobs scoring high by random bimodality
        w = float(np.clip(self.w_elon, 0.0, 1.0))
        score = (1.0 - w) * s_bi + w * s_elon
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
        return float(v ** 0.6)
