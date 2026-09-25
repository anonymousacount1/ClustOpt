# metrics/pattern/piecewise_linearity.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class PiecewiseLinearityMetric(BaseMetric):
    """
    Piecewise Linearity (polyline-likeness) score in [0,1], higher is better.

    Per cluster:
      - project to 2D (PCA via SVD if d>2, centered if d==2)
      - use PC1 as x, PC2 as y; sort by x
      - compute SSE of best single line vs best 2-seg (one split) vs greedy 3-seg
      - improvement = (SSE1 - SSEk)/SSE1
      - score = squash(improvement) * elongation_gate * segment_quality_gate

    Aggregate: size-weighted mean over clusters.
    """
    name = "piecewise_linearity"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        max_segments: int = 3,           # 2 or 3 are practical (3 uses greedy split)
        min_cluster_points: int = 12,
        min_seg_points: int = 6,         # per segment
        max_candidates: int = 60,         # candidate split positions (subsampled)
        imp_scale: float = 0.20,          # squash scale: smaller => more sensitive
        min_seg_r2: float = 0.20,         # segment quality threshold
        w_elon: float = 0.35,             # blend factor with elongation (0..1)
        elong_r0: float = 2.0,            # elongation saturation
        eps: float = 1e-12,
    ):
        self.max_segments = int(max_segments)
        self.min_cluster_points = int(min_cluster_points)
        self.min_seg_points = int(min_seg_points)
        self.max_candidates = int(max_candidates)
        self.imp_scale = float(imp_scale)
        self.min_seg_r2 = float(min_seg_r2)
        self.w_elon = float(w_elon)
        self.elong_r0 = float(elong_r0)
        self.eps = float(eps)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for piecewise_linearity.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _elongation_score(self, Z: np.ndarray) -> float:
        # same philosophy as ladderness/periodicity
        s1 = float(np.std(Z[:, 0]))
        s2 = float(np.std(Z[:, 1])) + self.eps
        ratio = s1 / s2
        return float(np.clip(1.0 - np.exp(-ratio / max(self.elong_r0, self.eps)), 0.0, 1.0))

    @staticmethod
    def _prefix_sums(x: np.ndarray, y: np.ndarray):
        # prefix sums for O(1) segment regression stats
        Sx = np.concatenate([[0.0], np.cumsum(x)])
        Sy = np.concatenate([[0.0], np.cumsum(y)])
        Sxx = np.concatenate([[0.0], np.cumsum(x * x)])
        Sxy = np.concatenate([[0.0], np.cumsum(x * y)])
        Syy = np.concatenate([[0.0], np.cumsum(y * y)])
        return Sx, Sy, Sxx, Sxy, Syy

    def _seg_stats(self, i: int, j: int, Sx, Sy, Sxx, Sxy, Syy):
        # segment is [i, j)  (i inclusive, j exclusive)
        n = j - i
        if n <= 1:
            return (0.0, 0.0, float("inf"), 0.0)  # a,b,SSE,R2

        sx = Sx[j] - Sx[i]
        sy = Sy[j] - Sy[i]
        sxx = Sxx[j] - Sxx[i]
        sxy = Sxy[j] - Sxy[i]
        syy = Syy[j] - Syy[i]

        denom = n * sxx - sx * sx
        if abs(denom) <= self.eps:
            # x almost constant -> best is horizontal line y=mean
            a = 0.0
            b = sy / n
        else:
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n

        # SSE for y - (a x + b)
        # SSE = Σy^2 - 2aΣxy -2bΣy + a^2 Σx^2 + 2ab Σx + b^2 n
        sse = syy - 2.0 * a * sxy - 2.0 * b * sy + (a * a) * sxx + 2.0 * a * b * sx + (b * b) * n
        sse = float(max(0.0, sse))

        # R2 in segment
        sst = syy - (sy * sy) / n
        if sst <= self.eps:
            r2 = 0.0
        else:
            r2 = float(1.0 - (sse / sst))
            r2 = float(np.clip(r2, 0.0, 1.0))

        return float(a), float(b), float(sse), float(r2)

    def _candidate_splits(self, n: int) -> np.ndarray:
        # valid split indices respecting min_seg_points
        lo = self.min_seg_points
        hi = n - self.min_seg_points
        if hi <= lo:
            return np.array([], dtype=int)

        candidates = np.arange(lo, hi, dtype=int)
        if candidates.size <= self.max_candidates:
            return candidates

        # subsample evenly
        idx = np.linspace(0, candidates.size - 1, self.max_candidates).astype(int)
        return candidates[idx]

    def _best_two_segments(self, Sx, Sy, Sxx, Sxy, Syy, n: int):
        # returns best_sse, best_k, seg_r2_min
        candidates = self._candidate_splits(n)
        if candidates.size == 0:
            return float("inf"), -1, 0.0

        best_sse = float("inf")
        best_k = -1
        best_r2_min = 0.0

        for k in candidates:
            _, _, sse1, r2_1 = self._seg_stats(0, k, Sx, Sy, Sxx, Sxy, Syy)
            _, _, sse2, r2_2 = self._seg_stats(k, n, Sx, Sy, Sxx, Sxy, Syy)
            sse = sse1 + sse2
            if sse < best_sse:
                best_sse = sse
                best_k = int(k)
                best_r2_min = float(min(r2_1, r2_2))

        return float(best_sse), int(best_k), float(best_r2_min)

    def _best_three_segments_greedy(self, Sx, Sy, Sxx, Sxy, Syy, n: int, k_best: int):
        # greedy: split into [0,k] and [k,n], then try splitting one side further
        if k_best <= 0:
            return float("inf"), 0.0

        # baseline 2-seg
        _, _, sseA, r2A = self._seg_stats(0, k_best, Sx, Sy, Sxx, Sxy, Syy)
        _, _, sseB, r2B = self._seg_stats(k_best, n, Sx, Sy, Sxx, Sxy, Syy)

        best_sse = sseA + sseB
        best_r2_min = min(r2A, r2B)

        # try split left segment further
        nL = k_best
        if nL >= 2 * self.min_seg_points:
            # candidates within left: [min_seg, nL-min_seg)
            lo = self.min_seg_points
            hi = nL - self.min_seg_points
            cand = np.arange(lo, hi, dtype=int)
            if cand.size > 0:
                if cand.size > self.max_candidates:
                    idx = np.linspace(0, cand.size - 1, self.max_candidates).astype(int)
                    cand = cand[idx]
                for k2 in cand:
                    _, _, sse1, r2_1 = self._seg_stats(0, k2, Sx, Sy, Sxx, Sxy, Syy)
                    _, _, sse2, r2_2 = self._seg_stats(k2, nL, Sx, Sy, Sxx, Sxy, Syy)
                    sse = sse1 + sse2 + sseB
                    if sse < best_sse:
                        best_sse = sse
                        best_r2_min = min(r2_1, r2_2, r2B)

        # try split right segment further
        nR = n - k_best
        if nR >= 2 * self.min_seg_points:
            lo = k_best + self.min_seg_points
            hi = n - self.min_seg_points
            cand = np.arange(lo, hi, dtype=int)
            if cand.size > 0:
                if cand.size > self.max_candidates:
                    idx = np.linspace(0, cand.size - 1, self.max_candidates).astype(int)
                    cand = cand[idx]
                for k2 in cand:
                    _, _, sse1, r2_1 = self._seg_stats(k_best, k2, Sx, Sy, Sxx, Sxy, Syy)
                    _, _, sse2, r2_2 = self._seg_stats(k2, n, Sx, Sy, Sxx, Sxy, Syy)
                    sse = sseA + sse1 + sse2
                    if sse < best_sse:
                        best_sse = sse
                        best_r2_min = min(r2A, r2_1, r2_2)

        return float(best_sse), float(best_r2_min)

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(Xi)
        # x along main direction, y orthogonal
        x = Z[:, 0]
        y = Z[:, 1]

        # if not enough spread -> no meaningful piecewise structure
        if float(np.std(x)) <= self.eps:
            return 0.0

        # sort by x
        order = np.argsort(x)
        x = x[order].astype(float)
        y = y[order].astype(float)
        n = int(x.size)

        # prefix sums for O(1) segment regression
        Sx, Sy, Sxx, Sxy, Syy = self._prefix_sums(x, y)

        # SSE of single line
        _, _, sse1, r2_1 = self._seg_stats(0, n, Sx, Sy, Sxx, Sxy, Syy)
        if not np.isfinite(sse1) or sse1 <= self.eps:
            return 0.0

        # best 2 segments
        sse2, k_best, r2min2 = self._best_two_segments(Sx, Sy, Sxx, Sxy, Syy, n)
        best_sse = sse2
        seg_r2_min = r2min2

        # optionally best 3 segments (greedy)
        if self.max_segments >= 3 and k_best > 0:
            sse3, r2min3 = self._best_three_segments_greedy(Sx, Sy, Sxx, Sxy, Syy, n, k_best)
            if sse3 < best_sse:
                best_sse = sse3
                seg_r2_min = r2min3

        # relative improvement
        improvement = float(max(0.0, (sse1 - best_sse) / (sse1 + self.eps)))

        # segment quality gate (avoid “cheating” with wiggly segments)
        # if segments are not linear enough -> reduce score
        quality = 1.0
        if seg_r2_min < self.min_seg_r2:
            # soft penalty
            quality = float(np.clip(seg_r2_min / max(self.min_seg_r2, self.eps), 0.0, 1.0))

        # squash improvement to [0,1]
        imp_scale = max(self.imp_scale, self.eps)
        score_pw = float(1.0 - np.exp(-improvement / imp_scale))

        # elongation gate (avoid blobs)
        s_elon = self._elongation_score(Z)

        w = float(np.clip(self.w_elon, 0.0, 1.0))
        score = (1.0 - w) * score_pw + w * s_elon
        score *= quality
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
        # already [0,1]
        return float(raw)
