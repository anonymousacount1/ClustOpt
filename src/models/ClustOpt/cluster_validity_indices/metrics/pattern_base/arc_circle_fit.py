# metrics/pattern/arc_circle_fit.py
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

try:
    import cv2
except Exception as e:
    cv2 = None
    _cv2_import_error = e


class ArcCircleFitMetric(BaseMetric):
    """
    Arc/Circle/ellipse-likeness score in [0,1] (higher is better).

    Per cluster:
      - project to 2D (if needed) using PCA/SVD on centered points
      - compute minEnclosingCircle center -> radial distances
      - radial_score = exp(-clip(std(r)/mean(r), 0, radial_clip))
      - (optional) fitEllipse -> axis_ratio = b/a -> shape_score in [0,1]
      - cluster_score = radial_score * (0.6 + 0.4 * shape_score)

    Aggregate:
      - size-weighted average over clusters

    Notes:
      - expects noise to be removed by CVI policy (remove_noise=True)
      - uses full space by default (pattern geometry)
    """
    name = "arc_circle_fit"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        radial_clip: float = 2.0,
        min_cluster_points: int = 5,
        use_ellipse: bool = True,
        ellipse_fallback: float = 1.0,
        eps: float = 1e-12,
    ):
        if cv2 is None:
            raise ImportError(f"OpenCV (cv2) is required for ArcCircleFitMetric: {_cv2_import_error}")

        self.radial_clip = float(radial_clip)
        self.min_cluster_points = int(min_cluster_points)
        self.use_ellipse = bool(use_ellipse)
        self.ellipse_fallback = float(ellipse_fallback)
        self.eps = float(eps)

    # ---------------------------
    # helpers
    # ---------------------------
    def _to_2d(self, pts: np.ndarray) -> np.ndarray:
        """
        Return centered 2D representation of pts.
        - if d==2: return centered pts
        - if d>2: PCA via SVD to 2D (on centered data)
        - if d<2: raise ValueError
        """
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2:
            raise ValueError("pts must be 2D array")
        n, d = pts.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for arc/circle fit.")
        centered = pts - pts.mean(axis=0, keepdims=True)
        if d == 2:
            return centered

        # PCA via SVD: centered = U S Vt -> project = U[:, :2] * S[:2]
        # Equivalent to centered @ Vt.T[:, :2]
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = U[:, :2] * S[:2]
        return proj

    def _cluster_score(self, pts: np.ndarray) -> float:
        """
        Compute score for a single cluster, returns [0,1].
        """
        if pts.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(pts)  # (n,2) centered
        if Z.shape[0] < self.min_cluster_points:
            return float("-inf")

        # OpenCV expects float32
        Z32 = np.asarray(Z, dtype=np.float32)

        # min enclosing circle
        (cx, cy), r = cv2.minEnclosingCircle(Z32)
        center = np.array([cx, cy], dtype=np.float32)

        # radial distances
        dr = Z32 - center[None, :]
        radii = np.sqrt(np.sum(dr * dr, axis=1))
        mean_r = float(np.mean(radii))
        if mean_r <= self.eps:
            return float("-inf")

        rel_std = float(np.std(radii) / (mean_r + self.eps))
        rel_std = float(np.clip(rel_std, 0.0, self.radial_clip))
        radial_score = float(np.exp(-rel_std))  # in (0,1]

        shape_score = 1.0
        if self.use_ellipse:
            # fitEllipse requires >=5 points
            if Z32.shape[0] >= 5:
                try:
                    ellipse = cv2.fitEllipse(Z32.reshape(-1, 1, 2))
                    (ex, ey), (MA, ma), angle = ellipse  # MA, ma are axes lengths
                    a = float(max(MA, ma))
                    b = float(min(MA, ma))
                    if a <= self.eps:
                        shape_score = self.ellipse_fallback
                    else:
                        shape_score = float(np.clip(b / a, 0.0, 1.0))
                except Exception:
                    shape_score = self.ellipse_fallback
            else:
                shape_score = self.ellipse_fallback

        # combine: emphasize radial consistency, lightly include ellipse circularity
        score = radial_score * (0.6 + 0.4 * shape_score)
        return float(np.clip(score, 0.0, 1.0))

    # ---------------------------
    # contract
    # ---------------------------
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        if X.ndim != 2 or X.shape[0] != labels.shape[0]:
            return float("-inf")

        uniq = np.unique(labels)
        if uniq.size < 1:
            return float("-inf")

        total = 0.0
        total_w = 0.0

        for l in uniq:
            pts = X[labels == l]
            if pts.shape[0] < self.min_cluster_points:
                continue

            s = self._cluster_score(pts)
            if not np.isfinite(s) or s == float("-inf"):
                continue

            w = float(pts.shape[0])
            total += w * s
            total_w += w

        if total_w <= 0.0:
            return float("-inf")

        return float(total / total_w)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # raw already in [0,1]
        return float(raw)
