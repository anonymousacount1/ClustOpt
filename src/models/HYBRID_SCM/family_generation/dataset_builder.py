"""Build an atomic ``DatasetConfig`` for a planned ``DatasetJob`` in memory.

The builder does **not** write a JSON file: it constructs a ``DatasetConfig``
dataclass that can be passed straight into the existing
``models.HYBRID_SCM.src.generator.generate_dataset`` pipeline.

The builder honours, in this order:
  1. The family's bounds, feature names, rendering defaults, postprocess.
  2. The subfamily's ``component_strategy`` (which HYBRID_SCM component type
     to use and how many components per cluster).
  3. The subfamily's ``parameter_ranges`` (optional ranges to sample from).
  4. The difficulty translation: noise, perturbations, global difficulty.

The implementation aims for *validity*, not perfect realism: every produced
DatasetConfig must pass ``io.config_io._validate_dataset`` and must run end
to end through ``generate_dataset``. Parameter ranges per subfamily can be
refined later without touching the runner.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from models.HYBRID_SCM.io.config_io import ComponentConfig, DatasetConfig

from .difficulty import (
    difficulty_component_noise_sigma,
    difficulty_dataset_perturbations,
    difficulty_global_difficulty,
    sample_total_points,
)
from .planning import DatasetJob, FamilyPlan


# Component-types that genuinely act as field generators rather than per-cluster
# shapes. They are emitted as a single component per cluster regardless of any
# requested ``components_per_cluster``.
_FIELD_TYPES = {
    "blob_field",
    "ridge_field",
    "fractal_dust",
    "oriented_texture_field",
    "micro_blob_texture",
    "striped_texture",
    "checker_texture",
    "speckle_texture",
    "anisotropic_grain_field",
    "grid_lattice",
}


def _uniform(
    rng: np.random.Generator,
    ranges: Dict[str, Any],
    key: str,
    default_low: float,
    default_high: float,
) -> float:
    bounds = ranges.get(key)
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        low, high = float(bounds[0]), float(bounds[1])
    else:
        low, high = float(default_low), float(default_high)
    if high < low:
        low, high = high, low
    if high == low:
        return float(low)
    return float(rng.uniform(low, high))


def _choice(
    rng: np.random.Generator,
    ranges: Dict[str, Any],
    key: str,
    default: List[Any],
) -> Any:
    vals = ranges.get(key)
    if isinstance(vals, (list, tuple)) and len(vals) > 0:
        idx = int(rng.integers(0, len(vals)))
        return vals[idx]
    return default[int(rng.integers(0, len(default)))]


def _sample_cluster_centers(
    rng: np.random.Generator,
    cluster_count: int,
    bounds: Dict[str, List[float]],
    margin_frac: float = 0.18,
    min_sep_frac: float = 0.18,
) -> np.ndarray:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = high - low
    inner_low = low + margin_frac * span
    inner_high = high - margin_frac * span
    inner_span = inner_high - inner_low
    min_sep = float(min_sep_frac * float(np.linalg.norm(inner_span))) / max(1.0, math.sqrt(cluster_count))

    centers: List[np.ndarray] = []
    for _ in range(cluster_count):
        chosen = None
        for _attempt in range(200):
            c = rng.uniform(inner_low, inner_high)
            if not centers:
                chosen = c
                break
            d = min(float(np.linalg.norm(c - other)) for other in centers)
            if d >= min_sep:
                chosen = c
                break
        if chosen is None:
            chosen = rng.uniform(inner_low, inner_high)
        centers.append(chosen)
    return np.asarray(centers)


def _sample_nested_cluster_centers(
    rng: np.random.Generator,
    cluster_count: int,
    bounds: Dict[str, List[float]],
    jitter_frac: float = 0.0,
    margin_frac: float = 0.22,
) -> np.ndarray:
    """Return cluster centers that share a single base center.

    All ``cluster_count`` centers are placed at the same base point inside the
    inner region of ``bounds``. If ``jitter_frac`` is positive, each center is
    perturbed independently by a small random offset proportional to the
    bounding-box span. This produces "nested" (jitter==0) or "shifted-center"
    (jitter>0) patterns, while ``_sample_cluster_centers`` retains its old
    well-separated behaviour for every other subfamily.
    """
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = high - low
    inner_low = low + margin_frac * span
    inner_high = high - margin_frac * span
    base = rng.uniform(inner_low, inner_high)

    centers: List[np.ndarray] = []
    if jitter_frac <= 0.0:
        for _ in range(cluster_count):
            centers.append(np.array(base, dtype=float))
    else:
        offset_scale = float(jitter_frac) * span
        for _ in range(cluster_count):
            offset = rng.uniform(-offset_scale, offset_scale)
            centers.append(np.array(base + offset, dtype=float))
    return np.asarray(centers)


def _allocate_points(total: int, n_buckets: int, rng: np.random.Generator) -> List[int]:
    """Spread ``total`` points across ``n_buckets`` so each bucket has > 0."""
    if n_buckets <= 0:
        return []
    if n_buckets == 1:
        return [int(total)]
    base = total // n_buckets
    rem = total - base * n_buckets
    counts = [base] * n_buckets
    order = list(range(n_buckets))
    rng.shuffle(order)
    for i in range(rem):
        counts[order[i]] += 1
    # Avoid zero-point components which would fail validation.
    for i in range(n_buckets):
        if counts[i] < 1:
            j = int(np.argmax(counts))
            if counts[j] > 1:
                counts[j] -= 1
                counts[i] += 1
    return [int(c) for c in counts]


# =========================================================
# Per-type component param builders
# =========================================================

ComponentParams = Tuple[Dict[str, Any], Dict[str, Any]]
"""(params, operators) for a single component."""


def _params_blob(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    sigma_lo = float(ranges.get("sigma_min", 0.03)) * span
    sigma_hi = float(ranges.get("sigma_max", 0.09)) * span
    anisotropy = float(ranges.get("anisotropy", 0.5))
    sx = float(rng.uniform(sigma_lo, sigma_hi))
    sy = float(rng.uniform(sigma_lo, sigma_hi))
    if anisotropy > 0:
        if rng.random() < 0.5:
            sy = sy * float(rng.uniform(1.0 - anisotropy, 1.0))
        else:
            sx = sx * float(rng.uniform(1.0 - anisotropy, 1.0))
    return (
        {"mu": [float(center[0]), float(center[1])], "sigma": [sx, sy]},
        {},
    )


def _params_ellipse(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    size_scale = float(ranges.get("_size_scale", 1.0))
    rx = _uniform(rng, ranges, "rx", 0.05, 0.12) * span * size_scale
    ry_ratio = _uniform(rng, ranges, "ry_ratio", 0.45, 0.95)
    rotation = _uniform(rng, ranges, "rotation", -math.pi, math.pi)
    thickness = _uniform(rng, ranges, "thickness", 0.003, 0.015) * span
    # Optional half-ellipse / arc range. Defaults to full ellipse to keep
    # existing subfamilies (e.g. ``hybrid_mixed_geometry.full_mixed_scene``)
    # producing complete ellipses unchanged.
    angle_low = _uniform(rng, ranges, "angle_low", 0.0, 0.0)
    angle_high = _uniform(rng, ranges, "angle_high", 2.0 * math.pi, 2.0 * math.pi)
    params = {
        "cx": float(center[0]),
        "cy": float(center[1]),
        "rx": float(rx),
        "ry": float(rx * ry_ratio),
        "rotation": float(rotation),
        "angle_low": float(angle_low),
        "angle_high": float(angle_high),
        "thickness": float(thickness),
        "fill": bool(ranges.get("fill", False)),
    }
    return params, {}


def _params_parallel_bands(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    slope = _uniform(rng, ranges, "slope", -1.0, 1.0)
    spacing = _uniform(rng, ranges, "spacing", 0.08, 0.22) * span
    thickness = _uniform(rng, ranges, "thickness", 0.004, 0.020) * span
    n_bands = int(_uniform(rng, ranges, "n_bands", 2, 4))
    n_bands = max(2, min(6, n_bands))
    x_low = float(low[0] + 0.05 * (high[0] - low[0]))
    x_high = float(high[0] - 0.05 * (high[0] - low[0]))
    intercept0 = float(center[1]) - 0.5 * spacing * (n_bands - 1)
    params = {
        "slope": float(slope),
        "intercept0": float(intercept0),
        "n_bands": int(n_bands),
        "spacing": float(spacing),
        "x_low": x_low,
        "x_high": x_high,
        "thickness": float(thickness),
        "band_allocation": "balanced",
    }
    return params, {}


def _params_band(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    slope = _uniform(rng, ranges, "slope", -1.2, 1.2)
    thickness = _uniform(rng, ranges, "thickness", 0.005, 0.025) * span
    if bool(ranges.get("x_span_full_bounds", False)):
        x_lo = float(low[0])
        x_hi = float(high[0])
    else:
        half_w = 0.35 * (high[0] - low[0])
        x_lo = float(center[0] - half_w)
        x_hi = float(center[0] + half_w)
    intercept = float(center[1] - slope * center[0])
    params: Dict[str, Any] = {
        "slope": float(slope),
        "intercept": intercept,
        "x_low": x_lo,
        "x_high": x_hi,
        "thickness": float(thickness),
    }
    # Pass-through ``interval_generation`` for periodic gaps / fragmentation.
    # The runtime ``Band`` component already consumes this key natively.
    ig = ranges.get("interval_generation")
    if isinstance(ig, dict) and ig:
        params["interval_generation"] = dict(ig)
    return params, {}


def _params_ladder(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    slope = _uniform(rng, ranges, "slope", -0.5, 0.5)
    n_rungs = int(_uniform(rng, ranges, "n_rungs", 4, 8))
    n_rungs = max(3, min(12, n_rungs))
    rung_spacing = _uniform(rng, ranges, "rung_spacing", 0.05, 0.12) * span
    seg_len = _uniform(rng, ranges, "segment_x_len", 0.05, 0.12) * span
    thickness = _uniform(rng, ranges, "thickness", 0.003, 0.015) * span
    half_w = 0.45 * (high[0] - low[0])
    return (
        {
            "slope": float(slope),
            "base_intercept": float(center[1]),
            "n_rungs": int(n_rungs),
            "rung_spacing": float(rung_spacing),
            "segment_x_len": float(seg_len),
            "x_center_low": float(center[0] - 0.3 * half_w),
            "x_center_high": float(center[0] + 0.3 * half_w),
            "thickness": float(thickness),
        },
        {},
    )


def _params_grid_lattice(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span_x = float(high[0] - low[0])
    span_y = float(high[1] - low[1])
    n_vert = int(_uniform(rng, ranges, "n_vertical", 3, 6))
    n_horz = int(_uniform(rng, ranges, "n_horizontal", 3, 6))
    spacing_x = _uniform(rng, ranges, "spacing_x", 0.08, 0.16) * span_x
    spacing_y = _uniform(rng, ranges, "spacing_y", 0.08, 0.16) * span_y
    line_thickness = _uniform(rng, ranges, "line_thickness", 0.003, 0.012) * float(np.mean([span_x, span_y]))
    missing = _uniform(rng, ranges, "missing_lines_prob", 0.0, 0.1)
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "n_vertical": int(max(2, n_vert)),
            "n_horizontal": int(max(2, n_horz)),
            "spacing_x": float(spacing_x),
            "spacing_y": float(spacing_y),
            "line_thickness": float(line_thickness),
            "jitter": float(_uniform(rng, ranges, "jitter", 0.0, 0.02)) * float(np.mean([span_x, span_y])),
            "missing_lines_prob": float(missing),
        },
        {},
    )


def _params_circle(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    size_scale = float(ranges.get("_size_scale", 1.0))
    r = _uniform(rng, ranges, "radius", 0.05, 0.12) * span * size_scale
    thickness = _uniform(rng, ranges, "thickness", 0.003, 0.012) * span
    angle_low = 0.0
    angle_high = float(_uniform(rng, ranges, "angle_high", math.pi, 2.0 * math.pi))
    if angle_high - angle_low < 0.5:
        angle_high = 2.0 * math.pi
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "r": float(r),
            "angle_low": angle_low,
            "angle_high": angle_high,
            "thickness": float(thickness),
        },
        {},
    )


def _params_ring_annulus(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    size_scale = float(ranges.get("_size_scale", 1.0))
    inner = _uniform(rng, ranges, "inner_radius", 0.04, 0.09) * span * size_scale
    outer = inner + _uniform(rng, ranges, "thickness", 0.012, 0.04) * span * size_scale
    ellipse_ratio = _uniform(rng, ranges, "ellipse_ratio", 0.7, 1.0)
    rot = _uniform(rng, ranges, "rotation", -math.pi, math.pi)
    angle_low = 0.0
    angle_high = float(_uniform(rng, ranges, "angle_high", math.pi, 2.0 * math.pi))
    if angle_high - angle_low < 0.5:
        angle_high = 2.0 * math.pi
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "inner_radius": float(inner),
            "outer_radius": float(outer),
            "ellipse_ratio": float(ellipse_ratio),
            "rotation": float(rot),
            "angle_low": float(angle_low),
            "angle_high": float(angle_high),
        },
        {},
    )


def _params_parabola(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    half_w = 0.30 * (high[0] - low[0])
    a_scale = 1.0 / max(half_w * half_w, 1e-6)
    a = float(_uniform(rng, ranges, "a", 0.4, 1.6)) * a_scale * float(_choice(rng, ranges, "sign", [-1.0, 1.0]))
    thickness = _uniform(rng, ranges, "thickness", 0.003, 0.015) * span
    return (
        {
            "a": float(a),
            "x0": float(center[0]),
            "b": float(center[1]),
            "x_low": float(center[0] - half_w),
            "x_high": float(center[0] + half_w),
            "thickness": float(thickness),
        },
        {},
    )


def _params_arch_chain(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span_y = float(high[1] - low[1])
    span_x = float(high[0] - low[0])
    half_w = 0.30 * span_x
    x_lo = float(center[0] - half_w)
    x_hi = float(center[0] + half_w)
    amp_min = _uniform(rng, ranges, "amp_min", 0.08, 0.18) * span_y
    amp_max = amp_min + _uniform(rng, ranges, "amp_max_extra", 0.05, 0.18) * span_y
    sharp_lo = _uniform(rng, ranges, "sharpness_min", 1.6, 2.4)
    sharp_hi = sharp_lo + _uniform(rng, ranges, "sharpness_extra", 0.4, 1.4)
    thickness = _uniform(rng, ranges, "thickness", 0.003, 0.015) * span_y
    return (
        {
            "x_low": x_lo,
            "x_high": x_hi,
            "y_base": float(center[1] - 0.2 * span_y),
            "mode": "up_down",
            "amp_min": float(amp_min),
            "amp_max": float(amp_max),
            "amp_schedule": "hump",
            "sharpness_min": float(sharp_lo),
            "sharpness_max": float(sharp_hi),
            "thickness": float(thickness),
            "interval_generation": {
                "mode": "regular",
                "n_intervals": int(_uniform(rng, ranges, "n_segments", 2, 5)),
            },
        },
        {},
    )


def _params_piecewise_polyline(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    n_segments = int(_uniform(rng, ranges, "n_segments", 2, 5))
    n_vertices = max(3, n_segments + 1)
    seg_len = _uniform(rng, ranges, "segment_length", 0.07, 0.16) * span
    vertices = [[float(center[0]), float(center[1])]]
    prev = vertices[-1]
    angle = float(rng.uniform(-math.pi, math.pi))
    for _ in range(n_vertices - 1):
        angle = angle + float(rng.uniform(-math.pi / 2.5, math.pi / 2.5))
        nx = prev[0] + math.cos(angle) * seg_len
        ny = prev[1] + math.sin(angle) * seg_len
        vertices.append([float(nx), float(ny)])
        prev = vertices[-1]
    thickness = _uniform(rng, ranges, "thickness", 0.003, 0.012) * span
    corner_rounding = _uniform(rng, ranges, "corner_rounding", 0.0, 0.6)
    return (
        {
            "vertices": vertices,
            "corner_rounding": float(corner_rounding),
            "base_thickness": float(thickness),
        },
        {},
    )


def _params_polygon(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    n_sides = int(_uniform(rng, ranges, "n_sides", 3, 6))
    n_sides = max(3, n_sides)
    radius = _uniform(rng, ranges, "radius", 0.05, 0.12) * span
    rotation = _uniform(rng, ranges, "rotation", -math.pi, math.pi)
    corner_rounding = _uniform(rng, ranges, "corner_rounding", 0.0, 0.5)
    boundary_thickness = _uniform(rng, ranges, "boundary_thickness", 0.004, 0.015) * span
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "n_sides": int(n_sides),
            "radius": float(radius),
            "rotation": float(rotation),
            "corner_rounding": float(corner_rounding),
            "boundary_thickness": float(boundary_thickness),
            "fill": bool(ranges.get("fill", False)),
        },
        {},
    )


def _params_rectangle_frame(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    width = _uniform(rng, ranges, "width", 0.10, 0.22) * span
    height_ratio = _uniform(rng, ranges, "height_ratio", 0.45, 1.1)
    rot = _uniform(rng, ranges, "rotation", -math.pi, math.pi)
    thickness = _uniform(rng, ranges, "frame_thickness", 0.003, 0.012) * span
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(width * height_ratio),
            "rotation": float(rot),
            "frame_thickness": float(thickness),
            "fill": bool(ranges.get("fill", False)),
        },
        {},
    )


def _params_junction(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    n_arms = int(_uniform(rng, ranges, "n_arms", 3, 4))
    arm_lengths = [
        float(_uniform(rng, ranges, "arm_length", 0.06, 0.14)) * span
        for _ in range(max(2, n_arms))
    ]
    angles = sorted([float(rng.uniform(-math.pi, math.pi)) for _ in range(n_arms)])
    thickness = _uniform(rng, ranges, "thickness", 0.003, 0.012) * span
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "arm_lengths": arm_lengths,
            "angles": angles,
            "thickness": float(thickness),
        },
        {},
    )


def _params_branching_tree(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    seg_len = _uniform(rng, ranges, "segment_length", 0.04, 0.10) * span
    return (
        {
            "root": [float(center[0]), float(center[1])],
            "branching_factor": int(_uniform(rng, ranges, "branching_factor", 2, 3)),
            "depth": int(_uniform(rng, ranges, "depth", 2, 4)),
            "angle_spread": float(_uniform(rng, ranges, "angle_spread", 0.8, 1.6)),
            "segment_length": float(seg_len),
            "length_decay": float(_uniform(rng, ranges, "length_decay", 0.55, 0.85)),
            "segment_thickness": float(_uniform(rng, ranges, "segment_thickness", 0.003, 0.012)) * span,
            "pruning_probability": float(_uniform(rng, ranges, "pruning_probability", 0.0, 0.25)),
            "angle_jitter": float(_uniform(rng, ranges, "angle_jitter", 0.04, 0.20)),
        },
        {},
    )


def _params_spiral(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    r_start = _uniform(rng, ranges, "r_start", 0.01, 0.04) * span
    r_end = r_start + _uniform(rng, ranges, "r_end_extra", 0.06, 0.13) * span
    turns = _uniform(rng, ranges, "turns", 1.8, 3.5)
    thickness = _uniform(rng, ranges, "thickness", 0.003, 0.012) * span
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "r_start": float(r_start),
            "r_end": float(r_end),
            "turns": float(turns),
            "angle_offset": float(rng.uniform(-math.pi, math.pi)),
            "thickness": float(thickness),
        },
        {},
    )


def _params_meander_curve(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    half_w = 0.30 * (high[0] - low[0])
    return (
        {
            "x_low": float(center[0] - half_w),
            "x_high": float(center[0] + half_w),
            "y_center": float(center[1]),
            "amplitude": float(_uniform(rng, ranges, "amplitude", 0.04, 0.12)) * span,
            "frequency": float(_uniform(rng, ranges, "frequency", 1.5, 3.5)),
            "phase": float(rng.uniform(-math.pi, math.pi)),
            "amplitude_drift": float(_uniform(rng, ranges, "amplitude_drift", 0.0, 0.06)),
            "frequency_drift": float(_uniform(rng, ranges, "frequency_drift", 0.0, 0.05)),
            "n_anchors": int(_uniform(rng, ranges, "n_anchors", 32, 64)),
            "anchor_jitter": float(_uniform(rng, ranges, "anchor_jitter", 0.0, 0.02)),
            "base_thickness": float(_uniform(rng, ranges, "thickness", 0.003, 0.012)) * span,
        },
        {},
    )


def _params_blob_field(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    width = _uniform(rng, ranges, "width", 0.18, 0.35) * (high[0] - low[0])
    height = _uniform(rng, ranges, "height", 0.18, 0.35) * (high[1] - low[1])
    n_blobs = int(_uniform(rng, ranges, "n_blobs", 4, 8))
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(height),
            "n_blobs": int(max(2, n_blobs)),
            "blob_sigma_min": float(_uniform(rng, ranges, "blob_sigma_min", 0.01, 0.03)) * span,
            "blob_sigma_max": float(_uniform(rng, ranges, "blob_sigma_max", 0.03, 0.07)) * span,
            "strength_jitter": float(_uniform(rng, ranges, "strength_jitter", 0.0, 0.3)),
        },
        {},
    )


def _params_ridge_field(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    width = _uniform(rng, ranges, "width", 0.18, 0.32) * (high[0] - low[0])
    height = _uniform(rng, ranges, "height", 0.18, 0.32) * (high[1] - low[1])
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(height),
            "n_ridges": int(_uniform(rng, ranges, "n_ridges", 3, 6)),
            "orientation": float(_uniform(rng, ranges, "orientation", -math.pi, math.pi)),
            "spacing": float(_uniform(rng, ranges, "spacing", 0.04, 0.10)) * span,
            "ridge_sigma": float(_uniform(rng, ranges, "ridge_sigma", 0.005, 0.020)) * span,
            "jitter": float(_uniform(rng, ranges, "jitter", 0.0, 0.04)) * span,
        },
        {},
    )


def _params_fractal_dust(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    span = float(np.mean(np.asarray(bounds["high"]) - np.asarray(bounds["low"])))
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "base_scale": float(_uniform(rng, ranges, "base_scale", 0.10, 0.20)) * span,
            "depth": int(_uniform(rng, ranges, "depth", 2, 4)),
            "children_per_level": int(_uniform(rng, ranges, "children_per_level", 3, 5)),
            "decay": float(_uniform(rng, ranges, "decay", 0.4, 0.7)),
            "jitter": float(_uniform(rng, ranges, "jitter", 0.0, 0.05)) * span,
        },
        {},
    )


def _params_oriented_texture_field(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    width = _uniform(rng, ranges, "width", 0.18, 0.32) * (high[0] - low[0])
    height = _uniform(rng, ranges, "height", 0.18, 0.32) * (high[1] - low[1])
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(height),
            "orientation": float(_uniform(rng, ranges, "orientation", -math.pi, math.pi)),
            "n_ridges": int(_uniform(rng, ranges, "n_ridges", 4, 8)),
            "ridge_spacing": float(_uniform(rng, ranges, "ridge_spacing", 0.03, 0.08)) * span,
            "ridge_sigma": float(_uniform(rng, ranges, "ridge_sigma", 0.003, 0.012)) * span,
            "cross_sigma": float(_uniform(rng, ranges, "cross_sigma", 0.006, 0.018)) * span,
            "phase_jitter": float(_uniform(rng, ranges, "phase_jitter", 0.0, 0.4)),
            "strength_jitter": float(_uniform(rng, ranges, "strength_jitter", 0.0, 0.3)),
        },
        {},
    )


def _params_micro_blob_texture(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    width = _uniform(rng, ranges, "width", 0.18, 0.32) * (high[0] - low[0])
    height = _uniform(rng, ranges, "height", 0.18, 0.32) * (high[1] - low[1])
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(height),
            "n_micro_blobs": int(_uniform(rng, ranges, "n_micro_blobs", 12, 28)),
            "blob_sigma_min": float(_uniform(rng, ranges, "blob_sigma_min", 0.005, 0.015)) * span,
            "blob_sigma_max": float(_uniform(rng, ranges, "blob_sigma_max", 0.015, 0.04)) * span,
            "strength_jitter": float(_uniform(rng, ranges, "strength_jitter", 0.0, 0.3)),
            "clustered_centers": bool(ranges.get("clustered_centers", False)),
        },
        {},
    )


def _params_striped_texture(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    width = _uniform(rng, ranges, "width", 0.18, 0.32) * (high[0] - low[0])
    height = _uniform(rng, ranges, "height", 0.18, 0.32) * (high[1] - low[1])
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(height),
            "orientation": float(_uniform(rng, ranges, "orientation", -math.pi, math.pi)),
            "n_stripes": int(_uniform(rng, ranges, "n_stripes", 4, 8)),
            "stripe_spacing": float(_uniform(rng, ranges, "stripe_spacing", 0.03, 0.07)) * span,
            "stripe_sigma": float(_uniform(rng, ranges, "stripe_sigma", 0.004, 0.014)) * span,
            "phase_offset": float(_uniform(rng, ranges, "phase_offset", 0.0, 1.0)),
            "duty_cycle": float(_uniform(rng, ranges, "duty_cycle", 0.3, 0.7)),
        },
        {},
    )


def _params_checker_texture(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    width = _uniform(rng, ranges, "width", 0.18, 0.32) * (high[0] - low[0])
    height = _uniform(rng, ranges, "height", 0.18, 0.32) * (high[1] - low[1])
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(height),
            "cell_size_x": float(_uniform(rng, ranges, "cell_size_x", 0.03, 0.07)) * span,
            "cell_size_y": float(_uniform(rng, ranges, "cell_size_y", 0.03, 0.07)) * span,
            "orientation": float(_uniform(rng, ranges, "orientation", -math.pi, math.pi)),
            "on_probability": float(_uniform(rng, ranges, "on_probability", 0.45, 0.65)),
            "jitter": float(_uniform(rng, ranges, "jitter", 0.0, 0.03)) * span,
        },
        {},
    )


def _params_speckle_texture(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    width = _uniform(rng, ranges, "width", 0.18, 0.32) * (high[0] - low[0])
    height = _uniform(rng, ranges, "height", 0.18, 0.32) * (high[1] - low[1])
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(height),
            "speckle_rate": float(_uniform(rng, ranges, "speckle_rate", 0.3, 0.8)),
            "clustered_fraction": float(_uniform(rng, ranges, "clustered_fraction", 0.0, 0.4)),
            "cluster_sigma": float(_uniform(rng, ranges, "cluster_sigma", 0.01, 0.04)) * span,
        },
        {},
    )


def _params_anisotropic_grain_field(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: Dict[str, List[float]],
    ranges: Dict[str, Any],
) -> ComponentParams:
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    span = float(np.mean(high - low))
    width = _uniform(rng, ranges, "width", 0.18, 0.32) * (high[0] - low[0])
    height = _uniform(rng, ranges, "height", 0.18, 0.32) * (high[1] - low[1])
    return (
        {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": float(width),
            "height": float(height),
            "n_grains": int(_uniform(rng, ranges, "n_grains", 12, 28)),
            "grain_sigma_long": float(_uniform(rng, ranges, "grain_sigma_long", 0.012, 0.035)) * span,
            "grain_sigma_short": float(_uniform(rng, ranges, "grain_sigma_short", 0.003, 0.012)) * span,
            "orientation": float(_uniform(rng, ranges, "orientation", -math.pi, math.pi)),
            "orientation_jitter": float(_uniform(rng, ranges, "orientation_jitter", 0.0, 0.4)),
            "strength_jitter": float(_uniform(rng, ranges, "strength_jitter", 0.0, 0.3)),
        },
        {},
    )


_PARAM_BUILDERS: Dict[str, Callable[..., ComponentParams]] = {
    "blob": _params_blob,
    "ellipse": _params_ellipse,
    "parallel_bands": _params_parallel_bands,
    "band": _params_band,
    "ladder": _params_ladder,
    "grid_lattice": _params_grid_lattice,
    "circle": _params_circle,
    "ring_annulus": _params_ring_annulus,
    "parabola": _params_parabola,
    "arch_chain": _params_arch_chain,
    "piecewise_polyline": _params_piecewise_polyline,
    "polygon": _params_polygon,
    "rectangle_frame": _params_rectangle_frame,
    "junction": _params_junction,
    "branching_tree": _params_branching_tree,
    "spiral": _params_spiral,
    "meander_curve": _params_meander_curve,
    "blob_field": _params_blob_field,
    "ridge_field": _params_ridge_field,
    "fractal_dust": _params_fractal_dust,
    "oriented_texture_field": _params_oriented_texture_field,
    "micro_blob_texture": _params_micro_blob_texture,
    "striped_texture": _params_striped_texture,
    "checker_texture": _params_checker_texture,
    "speckle_texture": _params_speckle_texture,
    "anisotropic_grain_field": _params_anisotropic_grain_field,
}


def _resolve_component_choice(
    rng: np.random.Generator,
    strategy: Dict[str, Any],
    parameter_ranges: Dict[str, Any],
    cluster_idx: int,
) -> Tuple[str, Dict[str, Any]]:
    """Pick a component type for the next cluster and return its parameter
    ranges dict. Supports two strategy shapes:

      * ``component_type``: single fixed type for every cluster.
      * ``component_types``: list of dicts ``{type, weight, parameter_ranges}``
        from which the type is drawn. Selection mode is controlled by
        ``strategy.assignment.mode`` (default ``"random"``):
          - ``"random"``      : weighted random per cluster (existing behaviour)
          - ``"primary_first"``: cluster 0 uses ``choices[0]``; all other
            clusters use ``choices[1]``. With a single entry, every cluster
            uses it. Extra choices (index >= 2) are ignored.
    """
    choices = strategy.get("component_types")
    if isinstance(choices, list) and len(choices) > 0:
        assignment = dict(strategy.get("assignment") or {})
        mode = str(assignment.get("mode", "random")).lower()
        if mode == "primary_first":
            idx = 0 if cluster_idx == 0 else min(1, len(choices) - 1)
        else:
            weights = [float(c.get("weight", 1.0)) for c in choices]
            total = sum(weights) or 1.0
            probs = [w / total for w in weights]
            idx = int(rng.choice(len(choices), p=probs))
        entry = choices[idx]
        comp_type = str(entry["type"])
        ranges = dict(entry.get("parameter_ranges") or parameter_ranges)
        return comp_type, ranges

    comp_type = str(strategy.get("component_type", "blob"))
    return comp_type, dict(parameter_ranges)


# =========================================================
# Public entry point
# =========================================================

def build_dataset_config(
    plan: FamilyPlan,
    job: "DatasetJob",
    dataset_id: str,
) -> Tuple[DatasetConfig, int]:
    """Build a ``DatasetConfig`` in memory for one planned dataset.

    Returns ``(config, n_points_requested)`` where ``n_points_requested`` is
    the integer total point count sampled for this job (used for manifest
    accounting).
    """
    rng = np.random.default_rng(int(job.dataset_seed))

    total_points = sample_total_points(rng, job.difficulty, plan.n_points_policy)

    strategy = dict(job.subfamily.component_strategy)
    components_per_cluster = int(strategy.get("components_per_cluster", 1))
    components_per_cluster = max(1, components_per_cluster)

    # --- Cluster placement -------------------------------------------------
    # Default "random" preserves the historical behaviour for every existing
    # subfamily. "nested" co-locates all cluster centers (with optional jitter)
    # and emits a per-cluster radius-decay schedule, so cluster 0 is the
    # outermost shape and cluster N-1 the innermost.
    placement = dict(strategy.get("cluster_placement") or {})
    placement_mode = str(placement.get("mode", "random")).lower()
    nested_scales: List[float]
    shared_overrides: Dict[str, List[float]] = {}
    if placement_mode == "nested":
        jitter_range = placement.get("center_jitter_frac", [0.0, 0.0])
        if isinstance(jitter_range, (list, tuple)) and len(jitter_range) == 2:
            jitter_frac = float(rng.uniform(float(jitter_range[0]), float(jitter_range[1])))
        else:
            jitter_frac = float(jitter_range)
        decay_range = placement.get("radius_decay", [0.55, 0.75])
        if isinstance(decay_range, (list, tuple)) and len(decay_range) == 2:
            decay = float(rng.uniform(float(decay_range[0]), float(decay_range[1])))
        else:
            decay = float(decay_range)
        decay = max(0.30, min(0.95, decay))
        centers = _sample_nested_cluster_centers(
            rng, job.cluster_count, plan.bounds, jitter_frac=jitter_frac
        )
        nested_scales = [float(decay ** k) for k in range(job.cluster_count)]
        # Pre-sample dataset-level shared parameter values so every cluster in
        # a nested-mode dataset uses the same rotation/angle/aspect, etc. This
        # is what makes a "nested arches" dataset have all arches facing the
        # same way (per the user's reference). Per-cluster differences are
        # still expressed through the size-scale schedule.
        shared_keys = placement.get("shared_param_keys") or []
        if isinstance(shared_keys, (list, tuple)):
            base_ranges = job.subfamily.parameter_ranges or {}
            for key in shared_keys:
                bounds_val = base_ranges.get(key)
                if isinstance(bounds_val, (list, tuple)) and len(bounds_val) == 2:
                    lo, hi = float(bounds_val[0]), float(bounds_val[1])
                    if hi < lo:
                        lo, hi = hi, lo
                    fixed = float(rng.uniform(lo, hi))
                    shared_overrides[str(key)] = [fixed, fixed]
    else:
        centers = _sample_cluster_centers(rng, job.cluster_count, plan.bounds)
        nested_scales = [1.0] * job.cluster_count

    # Build components: one or more per cluster (depending on strategy and type).
    pending: List[Tuple[int, str, Dict[str, Any], Dict[str, Any]]] = []
    for cluster_idx in range(job.cluster_count):
        comp_type, comp_ranges = _resolve_component_choice(
            rng, strategy, job.subfamily.parameter_ranges, cluster_idx
        )
        if placement_mode == "nested":
            comp_ranges = dict(comp_ranges)
            if "_size_scale" not in comp_ranges:
                comp_ranges["_size_scale"] = float(nested_scales[cluster_idx])
            for shared_key, shared_val in shared_overrides.items():
                comp_ranges[shared_key] = list(shared_val)
        repeat = 1 if comp_type in _FIELD_TYPES else components_per_cluster
        for k in range(repeat):
            builder = _PARAM_BUILDERS.get(comp_type)
            if builder is None:
                raise ValueError(
                    f"Unsupported component_type '{comp_type}' for subfamily "
                    f"'{job.subfamily.subfamily_id}'. Supported types: "
                    f"{sorted(_PARAM_BUILDERS.keys())}"
                )
            params, operators = builder(rng, centers[cluster_idx], plan.bounds, comp_ranges)
            pending.append((cluster_idx, comp_type, params, operators))

    counts = _allocate_points(total_points, len(pending), rng)

    noise_sigma = difficulty_component_noise_sigma(job.difficulty, plan.bounds)

    components: List[ComponentConfig] = []
    type_counters: Dict[str, int] = {}
    for (cluster_idx, comp_type, params, operators), n_points in zip(pending, counts):
        type_counters[comp_type] = type_counters.get(comp_type, 0) + 1
        comp_name = f"c{cluster_idx:02d}_{comp_type}_{type_counters[comp_type]:02d}"
        params_with_dims = dict(params)
        params_with_dims.setdefault("dims", [0, 1])
        noise_params: Dict[str, Any] = {}
        if noise_sigma > 0:
            noise_params["sigma"] = float(noise_sigma)
        components.append(
            ComponentConfig(
                name=comp_name,
                type=comp_type,
                n_points=int(max(1, n_points)),
                params=params_with_dims,
                operators=dict(operators or {}),
                noise=("gaussian" if noise_sigma > 0 else "none"),
                noise_params=noise_params,
                label=int(cluster_idx),
            )
        )

    cfg_tags = {
        "family_id": plan.family_id,
        "family_name": plan.family_name,
        "subfamily_id": job.subfamily.subfamily_id,
        "cluster_count": int(job.cluster_count),
        "difficulty": job.difficulty,
        "local_idx": int(job.local_idx),
        "global_idx": int(job.global_idx),
        "family_seed": int(plan.family_seed),
        "dataset_seed": int(job.dataset_seed),
        "components_per_cluster": int(components_per_cluster),
        "n_points_requested": int(total_points),
    }

    cfg = DatasetConfig(
        name=dataset_id,
        seed=int(job.dataset_seed),
        n_features=int(plan.n_features),
        feature_names=list(plan.feature_names),
        bounds={"low": list(plan.bounds["low"]), "high": list(plan.bounds["high"])},
        components=components,
        noise_label=-1,
        noise_label_mode="preserve",
        replay_mode="fresh",
        # The validator's allowed-keys set is incomplete for several supported
        # component params (e.g. polygon.cx/cy/fill, which the component
        # itself reads at runtime). Disabling the strict unknown-keys check
        # is intentional and only affects this in-memory family pipeline; the
        # other validations (component types, noise types, cycles, bounds,
        # perturbations, rendering, global difficulty, postprocess) still run.
        strict_validation=False,
        postprocess=dict(plan.postprocess or {}),
        global_difficulty=difficulty_global_difficulty(job.difficulty, plan.bounds),
        rendering=dict(plan.default_rendering or {}),
        dataset_perturbations=difficulty_dataset_perturbations(job.difficulty, plan.bounds),
        tags=cfg_tags,
    )
    return cfg, int(total_points)
