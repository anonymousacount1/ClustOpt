# models/ClustOpt/cluster_validity_indices/metrics/image_base/gridness_fft_acf.py
from __future__ import annotations
import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import RasterParams, rasterize_points, Bounds2D
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


def _autocorr2(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    mx = float(np.max(x))
    if mx > 0:
        x = x / mx
    x = x - float(np.mean(x))
    F = np.fft.fft2(x)
    P = F * np.conj(F)
    ac = np.fft.ifft2(P).real
    ac = np.fft.fftshift(ac)
    ac -= float(np.min(ac))
    if float(np.max(ac)) > 0:
        ac = ac / float(np.max(ac))
    return ac.astype(np.float32)


def _top_peaks(ac: np.ndarray, k: int = 8, suppress: int = 9) -> list[tuple[int, int, float]]:
    H, W = ac.shape
    A = ac.copy()
    cy, cx = H // 2, W // 2
    s = int(max(3, suppress))
    A[cy - s:cy + s + 1, cx - s:cx + s + 1] = 0.0

    peaks: list[tuple[int, int, float]] = []
    for _ in range(k):
        idx = np.argmax(A)
        v = float(A.flat[idx])
        if v <= 0:
            break
        y, x = np.unravel_index(idx, A.shape)
        peaks.append((y, x, v))
        A[max(0, y - s):min(H, y + s + 1), max(0, x - s):min(W, x + s + 1)] = 0.0
    return peaks


class GridnessFFTACFMetric(ImagePerClusterMetric):
    """
    Gridness (FFT/ACF) per-cluster

    Use 2D autocorrelation. Grid-like patterns produce strong off-center peaks
    in symmetric pairs along two axes.
    Score uses:
    - strength of top peaks
    - symmetry (existence of opposite peak)
    - orthogonality approx (two main directions)

    Returns [0,1].
    """

    name = "gridness_fft_acf"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 150,
        grid_size: int = 256,
        raster_mode: str = "density",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        peak_k: int = 10,
        suppress: int = 10,
        min_peak_val: float = 0.12,
        orth_tol_deg: float = 25.0,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.peak_k = int(peak_k)
        self.suppress = int(suppress)
        self.min_peak_val = float(min_peak_val)
        self.orth_tol_deg = float(orth_tol_deg)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or float(np.max(img)) <= 0.0:
            return 0.0

        ac = _autocorr2(img)
        peaks = [p for p in _top_peaks(ac, k=self.peak_k, suppress=self.suppress) if p[2] >= self.min_peak_val]
        if len(peaks) < 2:
            return 0.0

        H, W = ac.shape
        cy, cx = H // 2, W // 2

        # build vector list
        vecs = []
        for y, x, v in peaks:
            dy = float(y - cy)
            dx = float(x - cx)
            if abs(dx) + abs(dy) < 1e-6:
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            vecs.append((dx, dy, ang, v))

        if len(vecs) < 2:
            return 0.0

        # opposite-pair support
        def has_opposite(dx, dy, thr=0.35):
            ox, oy = -dx, -dy
            best = 0.0
            for dx2, dy2, _a2, v2 in vecs:
                # normalized dot similarity
                n1 = np.hypot(ox, oy) + 1e-9
                n2 = np.hypot(dx2, dy2) + 1e-9
                cos = (ox * dx2 + oy * dy2) / (n1 * n2)
                if cos > (1.0 - thr):
                    best = max(best, float(v2))
            return best

        pair_strengths = []
        angles = []
        for dx, dy, ang, v in vecs[:6]:
            opp = has_opposite(dx, dy)
            if opp > 0:
                pair_strengths.append(min(v, opp))
                angles.append(ang)

        if len(pair_strengths) < 2:
            # some periodicity but not grid
            return float(np.clip(np.mean([v for *_ , v in vecs]) * 0.6, 0.0, 1.0))

        # orthogonality: pick two dominant angles and see if ~90 deg
        angles = np.array(angles, dtype=np.float32)
        # reduce to [0,180)
        ang = (angles % 180.0)
        # take two strongest pairs
        idx = np.argsort(-np.array(pair_strengths))[:2]
        a1 = float(ang[idx[0]])
        a2 = float(ang[idx[1]])
        diff = abs(a1 - a2)
        diff = min(diff, 180.0 - diff)
        orth = float(np.exp(-((diff - 90.0) ** 2) / (2.0 * (self.orth_tol_deg ** 2))))

        strength = float(np.clip(np.mean(pair_strengths), 0.0, 1.0))
        score = strength * (0.6 + 0.4 * orth)
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
