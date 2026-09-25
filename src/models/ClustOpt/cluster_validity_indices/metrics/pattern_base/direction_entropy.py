# metrics/pattern/direction_entropy.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class DirectionEntropyMetric(BaseMetric):
    """
    Direction Entropy score in [0,1], higher means more diverse local directions (curvy/zigzag/complex).

    Per cluster:
      - project to 2D (PCA via SVD if d>2; centered if d==2)
      - sort by PC1 to induce an ordering
      - compute step directions theta_i from successive differences
      - histogram theta into B bins on [-pi, pi)
      - compute normalized entropy H/log(B) in [0,1]
      - blend with elongation gate to avoid blobs dominating

    Aggregate: size-weighted mean across clusters.
    """
    name = "direction_entropy"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        bins: int = 24,
        min_cluster_points: int = 18,
        min_steps: int = 12,
        step_eps: float = 1e-8,     # ignore tiny steps
        eps: float = 1e-12,
        w_elon: float = 0.25,
        elong_r0: float = 2.0,
    ):
        self.bins = int(bins)
        self.min_cluster_points = int(min_cluster_points)
        self.min_steps = int(min_steps)
        self.step_eps = float(step_eps)
        self.eps = float(eps)
        self.w_elon = float(w_elon)
        self.elong_r0 = float(elong_r0)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for direction_entropy.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _elongation_score(self, Z: np.ndarray) -> float:
        x = Z[:, 0]
        y = Z[:, 1]
        s1 = float(np.std(x))
        s2 = float(np.std(y)) + self.eps
        ratio = s1 / s2
        return float(np.clip(1.0 - np.exp(-ratio / max(self.elong_r0, self.eps)), 0.0, 1.0))

    def _entropy_norm(self, theta: np.ndarray) -> float:
        # histogram on [-pi, pi)
        B = max(4, self.bins)
        h, _ = np.histogram(theta, bins=B, range=(-np.pi, np.pi))
        total = float(np.sum(h))
        if total <= 0.0:
            return 0.0
        p = h.astype(float) / total
        p = p[p > 0]
        H = -float(np.sum(p * np.log(p + self.eps)))
        Hmax = float(np.log(B))
        return float(np.clip(H / max(Hmax, self.eps), 0.0, 1.0))

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(Xi)

        # Use kNN edge angles, but weighted by edge length so that the few
        # truly distant edges (along the shape's principal axes) dominate
        # over the very short noisy edges. Combine with the angular vector
        # strength (a more numerically stable measure of "directional
        # concentration") and convert to a spread (entropy-like) measure.
        # A straight line ends up with one dominant angle and low spread;
        # a ring/disk spreads across all angles.
        k = int(min(max(4, self.min_steps // 2), Z.shape[0] - 1))
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
            _, idx = nn.kneighbors(Z)
        except Exception:
            return 0.0

        rows = np.repeat(np.arange(Z.shape[0]), k)
        cols = idx[:, 1 : k + 1].reshape(-1)
        V = Z[cols] - Z[rows]
        step_len = np.linalg.norm(V, axis=1)
        mask = step_len > self.step_eps
        V = V[mask]
        step_len = step_len[mask]
        if V.shape[0] < self.min_steps:
            return 0.0

        # Keep only the longest edges so short noise edges (where dx ~ dy ~
        # noise) don't fill the angular histogram with random directions.
        keep_top = max(self.min_steps, int(0.25 * step_len.size))
        order = np.argsort(step_len)[::-1][:keep_top]
        V = V[order]
        step_len = step_len[order]

        theta = np.arctan2(V[:, 1], V[:, 0])  # [-pi, pi]
        two_theta = 2.0 * theta              # orientation-invariant (theta = theta + pi)
        w = step_len.astype(float)
        w = w / max(float(np.sum(w)), self.eps)
        c = float(np.sum(w * np.cos(two_theta)))
        s = float(np.sum(w * np.sin(two_theta)))
        R = float(np.clip(np.sqrt(c * c + s * s), 0.0, 1.0))
        # R = 1 means one dominant direction (straight line); R = 0 means
        # uniform across angles (blob/ring). Convert to a spread score.
        spread = float(1.0 - R)

        s_elon = self._elongation_score(Z)
        we = float(np.clip(self.w_elon, 0.0, 1.0))
        score = (1.0 - we) * spread + we * s_elon
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
