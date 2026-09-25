# metrics/pattern/gridness_2d.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class Gridness2DMetric(BaseMetric):
    """
    Gridness-2D score in [0,1], higher means points align on a 2D grid structure.

    Per cluster:
      - project to 2D (PCA via SVD if d>2)
      - min-max normalize to [0,1]^2
      - build occupancy grid (GxG) with binary presence
      - compute:
          * row_strength: concentration of occupied rows (peaky row-sums)
          * col_strength: concentration of occupied cols (peaky col-sums)
          * row_periodicity: regular spacing of occupied rows (FFT peak ratio)
          * col_periodicity: regular spacing of occupied cols (FFT peak ratio)
      - score = mean(row_strength*row_periodicity, col_strength*col_periodicity)
      - (optional) blend with elongation gate to avoid blobs

    Aggregate: size-weighted mean across clusters.

    Expected noise removed by CVI.
    """
    name = "gridness_2d"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        grid_size: int = 32,
        min_cluster_points: int = 40,
        min_occupied_rows: int = 3,
        min_occupied_cols: int = 3,
        rowcol_presence_thresh: int = 2,   # a row is "occupied" if >= thresh occupied cells
        eps: float = 1e-12,
        w_elon: float = 0.0,               # keep 0 by default (gridness is 2D by nature)
        elong_r0: float = 2.0,
    ):
        self.grid_size = int(grid_size)
        self.min_cluster_points = int(min_cluster_points)
        self.min_occupied_rows = int(min_occupied_rows)
        self.min_occupied_cols = int(min_occupied_cols)
        self.rowcol_presence_thresh = int(rowcol_presence_thresh)
        self.eps = float(eps)
        self.w_elon = float(w_elon)
        self.elong_r0 = float(elong_r0)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for gridness_2d.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _minmax01(self, Z: np.ndarray) -> np.ndarray:
        zmin = np.min(Z, axis=0)
        zmax = np.max(Z, axis=0)
        rng = np.maximum(zmax - zmin, self.eps)
        return (Z - zmin[None, :]) / rng[None, :]

    def _occupancy_grid(self, U01: np.ndarray) -> np.ndarray:
        G = self.grid_size
        # map to [0, G-1]
        ij = np.floor(U01 * G).astype(int)
        ij = np.clip(ij, 0, G - 1)
        occ = np.zeros((G, G), dtype=np.uint8)
        occ[ij[:, 1], ij[:, 0]] = 1  # y as rows, x as cols
        return occ

    def _peaky_strength(self, sums: np.ndarray) -> float:
        """
        sums: nonnegative counts per row/col (length G)
        Goal: high when only few rows/cols are heavily occupied (grid lines),
              low when occupancy spread.
        Use normalized Gini-like concentration: (max - mean)/(max + eps) in [0,1].
        """
        mx = float(np.max(sums))
        if mx <= self.eps:
            return 0.0
        mu = float(np.mean(sums))
        return float(np.clip((mx - mu) / (mx + self.eps), 0.0, 1.0))

    def _fft_periodicity(self, binary_vec: np.ndarray) -> float:
        """
        binary_vec: 0/1 vector of length G, indicates which rows/cols are 'active'
        Return [0,1] measure of dominant periodic component (excluding DC).
        """
        x = binary_vec.astype(float)
        if np.sum(x) <= 1.0:
            return 0.0
        x = x - np.mean(x)
        F = np.abs(np.fft.rfft(x))
        if F.size <= 2:
            return 0.0
        # exclude DC (index 0)
        spec = F[1:]
        ssum = float(np.sum(spec))
        if ssum <= self.eps:
            return 0.0
        peak = float(np.max(spec))
        # peak ratio
        return float(np.clip(peak / ssum, 0.0, 1.0))

    def _elongation_score(self, Z: np.ndarray) -> float:
        s1 = float(np.std(Z[:, 0]))
        s2 = float(np.std(Z[:, 1])) + self.eps
        ratio = max(s1, s2) / min(s1, s2)
        return float(np.clip(1.0 - np.exp(-ratio / max(self.elong_r0, self.eps)), 0.0, 1.0))

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(Xi)
        U01 = self._minmax01(Z)
        occ = self._occupancy_grid(U01)

        row_sums = occ.sum(axis=1).astype(int)
        col_sums = occ.sum(axis=0).astype(int)

        # Adaptive threshold: rows are "active" if they stand out clearly above
        # the typical row mass. A fixed integer threshold marks every row of a
        # dense lattice (whose vertical lines also drop pixels into all rows)
        # as active, which then collapses the FFT periodicity signal.
        row_med = float(np.median(row_sums)) if row_sums.size else 0.0
        col_med = float(np.median(col_sums)) if col_sums.size else 0.0
        row_thr = max(float(self.rowcol_presence_thresh), 1.6 * row_med + 1.0)
        col_thr = max(float(self.rowcol_presence_thresh), 1.6 * col_med + 1.0)
        row_active = (row_sums >= row_thr).astype(np.uint8)
        col_active = (col_sums >= col_thr).astype(np.uint8)

        n_row_active = int(np.sum(row_active))
        n_col_active = int(np.sum(col_active))
        if n_row_active < self.min_occupied_rows or n_col_active < self.min_occupied_cols:
            return 0.0

        row_strength = self._peaky_strength(row_sums)
        col_strength = self._peaky_strength(col_sums)

        row_period = self._fft_periodicity(row_active)
        col_period = self._fft_periodicity(col_active)

        score_rows = row_strength * row_period
        score_cols = col_strength * col_period
        score = 0.5 * (score_rows + score_cols)

        # optional elongation blend (usually keep 0)
        w = float(np.clip(self.w_elon, 0.0, 1.0))
        if w > 0:
            s_elon = self._elongation_score(Z)
            score = (1.0 - w) * score + w * s_elon

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
