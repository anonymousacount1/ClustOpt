# models/ClustOpt/cluster_validity_indices/metrics/image_base/hough_circle_arc_strength.py
from __future__ import annotations

import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import (
    RasterParams,
    rasterize_points,
    canny_edges,
    Bounds2D,
)
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


def _auto_canny_thresholds(img: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    v = float(np.median(img))
    low = int(max(0, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    if high <= 0:
        return 50, 150
    if low >= high:
        low = max(0, high // 2)
    return low, high


def _circle_fit_kasa(points_xy: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float] | None:
    """
    Kasa circle fit (אלגברי).
    מחזיר (cx, cy, r) בפיקסלים.
    """
    if points_xy.shape[0] < 6:
        return None
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)

    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        D, E, F = sol
        cx = D / 2.0
        cy = E / 2.0
        r2 = F + cx * cx + cy * cy
        if r2 <= eps:
            return None
        r = float(np.sqrt(r2))
        return float(cx), float(cy), float(r)
    except Exception:
        return None


def _perimeter_precision(edges: np.ndarray, cx: float, cy: float, r: float, tol: float, n_samples: int) -> float:
    """
    כמה נקודות דגימה על ההיקף פוגעות ב-edge (עם tol ע"י dilation).
    """
    H, W = edges.shape[:2]
    if r <= 2.0:
        return 0.0

    k = int(max(0, round(tol)))
    if k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        E = cv2.dilate(edges, kernel, iterations=1)
    else:
        E = edges

    n = int(max(90, n_samples))
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = np.round(cx + r * np.cos(thetas)).astype(np.int32)
    ys = np.round(cy + r * np.sin(thetas)).astype(np.int32)

    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size == 0:
        return 0.0

    hits = int(np.count_nonzero(E[ys, xs]))
    return float(hits) / float(xs.size)


def _edge_band_stats(edges: np.ndarray, cx: float, cy: float, r: float, band: float) -> tuple[int, int, np.ndarray]:
    """
    מחזיר:
    - n_total_edges
    - n_edges_in_band (abs(dist-r) <= band)
    - angles של ה-edge points בתוך הבנד (ברדיאנים, [-pi, pi])
    """
    ys, xs = np.nonzero(edges)
    n_total = int(xs.size)
    if n_total == 0:
        return 0, 0, np.empty((0,), dtype=np.float64)

    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    dist = np.sqrt(dx * dx + dy * dy)
    mask = np.abs(dist - r) <= float(max(1.0, band))
    xs_b = xs[mask].astype(np.float64)
    ys_b = ys[mask].astype(np.float64)
    n_band = int(xs_b.size)
    if n_band == 0:
        return n_total, 0, np.empty((0,), dtype=np.float64)

    ang = np.arctan2(ys_b - cy, xs_b - cx)
    return n_total, n_band, ang


def _angular_coverage(angles: np.ndarray, n_bins: int = 36, min_count_per_bin: int = 1) -> float:
    """
    כמה מה-bins הזוויתיים מכוסים ע"י edge points.
    קשת/מעגל אמיתיים -> כיסוי זוויתי משמעותי.
    קווים/פסים -> כיסוי זוויתי נמוך (angles מרוכזים).
    """
    if angles.size == 0:
        return 0.0
    n_bins = int(max(12, n_bins))
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    covered = int(np.count_nonzero(hist >= int(max(1, min_count_per_bin))))
    return float(covered) / float(n_bins)


def _point_inlier_ratio(pts_px: np.ndarray, cx: float, cy: float, r: float, tol: float) -> float:
    """
    כמה מנקודות הקלאסטר עצמן קרובות להיקף (abs(||p-c||-r) <= tol).
    זה “שובר” false positives שנובעים רק מ-edges של פסים.
    """
    if pts_px.shape[0] == 0:
        return 0.0
    dx = pts_px[:, 0].astype(np.float64) - cx
    dy = pts_px[:, 1].astype(np.float64) - cy
    dist = np.sqrt(dx * dx + dy * dy)
    inl = np.abs(dist - r) <= float(max(1.0, tol))
    return float(np.mean(inl))


class HoughCircleArcStrengthMetric(ImagePerClusterMetric):
    """
    Hough Circle/Arc Strength — per-cluster.

    מטרה:
    - ציון גבוה לקלאסטרים שהם קשת/מעגל (AMP-like)
    - ציון נמוך לקלאסטרים שהם פסים/קווים/עננים (PRI-GT וכו')

    איך:
    - מייצרים תמונה לקלאסטר
    - מוצאים edges
    - מוצאים קנדידטים עם HoughCircles (+ fallback Kasa)
    - נותנים score שמחייב:
        (א) precision על ההיקף
        (ב) inlier ratio על נקודות הקלאסטר
        (ג) angular coverage על edges בתוך band
        (ד) edge-band recall (כמה edges מוסברים)
    """

    name = "hough_circle_arc_strength"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 60,
        grid_size: int = 256,
        raster_mode: str = "density",
        blur_ksize: int = 5,
        dilate_iters: int = 2,
        canny_low: int | None = None,
        canny_high: int | None = None,

        # Hough params
        dp: float = 1.2,
        minDist_frac: float = 0.30,
        param1: float = 140.0,
        param2: float = 22.0,
        minRadius_frac: float = 0.08,
        maxRadius_frac: float = 0.55,

        # support params
        support_tol: float = 2.0,
        support_samples: int = 240,

        # gating thresholds (חשוב!)
        min_point_inlier: float = 0.12,
        min_angle_coverage: float = 0.08,
        min_edge_recall: float = 0.01,

        angle_bins: int = 36,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )

        self.canny_low = canny_low
        self.canny_high = canny_high

        self.dp = float(dp)
        self.minDist_frac = float(minDist_frac)
        self.param1 = float(param1)
        self.param2 = float(param2)
        self.minRadius_frac = float(minRadius_frac)
        self.maxRadius_frac = float(maxRadius_frac)

        self.support_tol = float(support_tol)
        self.support_samples = int(max(90, support_samples))

        self.min_point_inlier = float(min_point_inlier)
        self.min_angle_coverage = float(min_angle_coverage)
        self.min_edge_recall = float(min_edge_recall)
        self.angle_bins = int(angle_bins)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or int(np.max(img)) == 0:
            return 0.0

        if self.canny_low is None or self.canny_high is None:
            low, high = _auto_canny_thresholds(img)
        else:
            low, high = int(self.canny_low), int(self.canny_high)

        edges = canny_edges(img, low=low, high=high)
        H, W = edges.shape[:2]
        n_edges = int(np.count_nonzero(edges))
        if n_edges < 25:
            return 0.0

        minDist = float(self.minDist_frac * min(H, W))
        minR = int(max(3, round(self.minRadius_frac * min(H, W))))
        maxR = int(max(minR + 2, round(self.maxRadius_frac * min(H, W))))

        blur = cv2.GaussianBlur(img, (5, 5), 1.2)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=minDist,
            param1=self.param1,
            param2=self.param2,
            minRadius=minR,
            maxRadius=maxR,
        )

        candidates: list[tuple[float, float, float]] = []
        if circles is not None and circles.size > 0:
            circles = circles.reshape(-1, 3)
            for (cx, cy, r) in circles:
                candidates.append((float(cx), float(cy), float(r)))

        # ---- build pts in pixel coords (same mapping as rasterize bounds) ----
        xmin, xmax, ymin, ymax = bounds
        dx = float(xmax - xmin)
        dy = float(ymax - ymin)
        if dx <= 0.0 or dy <= 0.0:
            return 0.0

        g = int(self.raster_params.grid_size)
        xs = (Xi[:, 0] - xmin) / dx
        ys = (Xi[:, 1] - ymin) / dy
        px = np.clip(xs * (g - 1), 0, g - 1)
        py = np.clip((1.0 - ys) * (g - 1), 0, g - 1)  # flip y to image coords
        pts_px = np.stack([px, py], axis=1).astype(np.float32)

        # fallback: Kasa fit (אבל הוא “מסוכן” בלי gating, לכן רק כקנדידט נוסף)
        fit = _circle_fit_kasa(pts_px)
        if fit is not None:
            candidates.append(fit)

        if not candidates:
            return 0.0

        best = 0.0

        for (cx, cy, r) in candidates:
            if r <= 2.0:
                continue

            # 1) precision on perimeter samples
            prec = _perimeter_precision(edges, cx=cx, cy=cy, r=r, tol=self.support_tol, n_samples=self.support_samples)

            # 2) edge-band recall + angles
            n_total, n_band, ang = _edge_band_stats(edges, cx=cx, cy=cy, r=r, band=self.support_tol)
            if n_total <= 0:
                continue
            recall = float(n_band) / float(n_total)

            # 3) angular coverage
            cov = _angular_coverage(ang, n_bins=self.angle_bins, min_count_per_bin=1)

            # 4) point inlier ratio (cluster points near circle)
            pinl = _point_inlier_ratio(pts_px, cx=cx, cy=cy, r=r, tol=self.support_tol)

            # ---- gating to kill false positives ----
            if pinl < self.min_point_inlier:
                continue
            if cov < self.min_angle_coverage:
                continue
            if recall < self.min_edge_recall:
                continue

            # ---- final score ----
            # precision: “על ההיקף יש edges”
            # pinl: “הנקודות באמת יושבות על מעגל”
            # cov: “יש קשת משמעותית ולא כמה נקודות מקריות”
            # recall: “חלק מה-edges מוסברים”
            score = float(np.sqrt(max(0.0, prec) * max(0.0, pinl)))
            score *= float(np.sqrt(max(0.0, cov)))
            score *= float(min(1.0, 0.5 + 0.5 * recall))  # מתגמל recall אבל לא שובר קשתות חלקיות

            if score > best:
                best = score

        return float(np.clip(best, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # The raw score is the product of four sub-scores (each in [0,1]), so a
        # plausible-but-not-perfect arc easily drops to ~0.1. Apply a sqrt
        # squash so realistic arc/circle structures land closer to 0.5-0.9
        # while still penalising clusters that fail any of the four gates.
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.4)
