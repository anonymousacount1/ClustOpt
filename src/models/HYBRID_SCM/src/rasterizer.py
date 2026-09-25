from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
except Exception:  # pragma: no cover - optional dependency fallback
    gaussian_filter = None
    binary_opening = None
    binary_closing = None


DEFAULT_RENDERING_SPEC: Dict[str, Any] = {
    "enabled": False,
    "image_size": [256, 256],
    "render_mode": "gaussian_splat",
    "sigma_px": 1.2,
    "normalize": "minmax",
    "threshold": 0.15,
    "morph_open": 0,
    "morph_close": 0,
    "per_cluster_masks": False,
    "axis_map": None,
    "binary_blur_sigma": 0.0,
}


def _resolve_axis_map(feature_names: List[str], axis_map: Any | None) -> Tuple[int, int, Dict[str, Any]]:
    if axis_map is None:
        x_idx, y_idx = 0, 1 if len(feature_names) > 1 else 0
        return x_idx, y_idx, {"x": feature_names[x_idx], "y": feature_names[y_idx], "x_idx": x_idx, "y_idx": y_idx}

    def _to_idx(v: Any) -> int:
        if isinstance(v, int):
            idx = v
        elif isinstance(v, str):
            if v not in feature_names:
                raise ValueError(f"Unknown feature name in axis_map: {v}")
            idx = feature_names.index(v)
        else:
            raise TypeError("axis_map values must be feature names or integer indices.")
        if not (0 <= idx < len(feature_names)):
            raise ValueError(f"axis_map index out of range: {idx}")
        return idx

    if isinstance(axis_map, (list, tuple)) and len(axis_map) == 2:
        x_idx = _to_idx(axis_map[0])
        y_idx = _to_idx(axis_map[1])
    elif isinstance(axis_map, dict):
        if "x" not in axis_map or "y" not in axis_map:
            raise ValueError("axis_map dict must contain keys 'x' and 'y'.")
        x_idx = _to_idx(axis_map["x"])
        y_idx = _to_idx(axis_map["y"])
    else:
        raise TypeError("axis_map must be None, [x, y], or {'x': ..., 'y': ...}.")

    return x_idx, y_idx, {"x": feature_names[x_idx], "y": feature_names[y_idx], "x_idx": x_idx, "y_idx": y_idx}


def _resolve_axis_limits(values: np.ndarray, axis_limits: Any | None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x = values[:, 0]
    y = values[:, 1]

    if isinstance(axis_limits, dict):
        xlim = axis_limits.get("x")
        ylim = axis_limits.get("y")
        if xlim is None or ylim is None:
            raise ValueError("axis_limits dict must contain 'x' and 'y'.")
        xlim = (float(xlim[0]), float(xlim[1]))
        ylim = (float(ylim[0]), float(ylim[1]))
        return xlim, ylim

    def _auto(arr: np.ndarray) -> Tuple[float, float]:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if np.isclose(lo, hi):
            pad = 1.0 if np.isclose(lo, 0.0) else max(1e-6, 0.05 * abs(lo))
            return lo - pad, hi + pad
        pad = 0.02 * (hi - lo)
        return lo - pad, hi + pad

    return _auto(x), _auto(y)


def _normalize_image(img: np.ndarray, mode: str) -> np.ndarray:
    img = np.asarray(img, dtype=float)
    if img.size == 0:
        return img
    if mode == "none":
        return np.clip(img, 0.0, None)
    if mode == "minmax":
        lo = float(np.min(img))
        hi = float(np.max(img))
        if hi <= lo + 1e-12:
            return np.zeros_like(img, dtype=float)
        return (img - lo) / (hi - lo)
    if mode == "percentile":
        hi = float(np.percentile(img, 99.5))
        if hi <= 1e-12:
            return np.zeros_like(img, dtype=float)
        return np.clip(img / hi, 0.0, 1.0)
    raise ValueError(f"Unsupported normalize mode: {mode}")


def _apply_morphology(mask: np.ndarray, morph_open: int, morph_close: int) -> np.ndarray:
    out = mask.astype(bool)
    if morph_open > 0 and binary_opening is not None:
        out = binary_opening(out, iterations=int(morph_open))
    if morph_close > 0 and binary_closing is not None:
        out = binary_closing(out, iterations=int(morph_close))
    return out


def _render_single(
    values_xy: np.ndarray,
    image_size: Tuple[int, int],
    render_mode: str,
    sigma_px: float,
    normalize: str,
    threshold: float,
    morph_open: int,
    morph_close: int,
    axis_limits: Any | None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    h, w = int(image_size[0]), int(image_size[1])
    xlim, ylim = _resolve_axis_limits(values_xy, axis_limits)
    if values_xy.size == 0:
        empty = np.zeros((h, w), dtype=float)
        return empty, empty.astype(bool), {"axis_limits": {"x": list(xlim), "y": list(ylim)}}

    hist, _, _ = np.histogram2d(
        values_xy[:, 1],
        values_xy[:, 0],
        bins=[h, w],
        range=[[ylim[0], ylim[1]], [xlim[0], xlim[1]]],
    )

    img = hist.astype(float)
    mode = str(render_mode).lower()
    if mode == "gaussian_splat":
        if gaussian_filter is not None and float(sigma_px) > 0:
            img = gaussian_filter(img, sigma=float(sigma_px), mode="nearest")
    elif mode == "hard_bin":
        pass
    else:
        raise ValueError(f"Unsupported render_mode: {render_mode}")

    img = _normalize_image(img, str(normalize).lower())
    mask = img >= float(threshold)
    mask = _apply_morphology(mask, int(morph_open), int(morph_close))

    info = {
        "axis_limits": {"x": [float(xlim[0]), float(xlim[1])], "y": [float(ylim[0]), float(ylim[1])]},
        "foreground_pixels": int(mask.sum()),
        "foreground_fraction": float(mask.mean()) if mask.size else 0.0,
        "max_intensity": float(np.max(img)) if img.size else 0.0,
    }
    return img, mask, info


def resolve_rendering_spec(rendering_spec: Dict[str, Any] | None, feature_names: List[str], bounds: Dict[str, Any] | None = None) -> Dict[str, Any]:
    spec = dict(DEFAULT_RENDERING_SPEC)
    spec.update(dict(rendering_spec or {}))
    x_idx, y_idx, axis_info = _resolve_axis_map(feature_names, spec.get("axis_map"))
    spec["axis_map"] = axis_info

    axis_limits = spec.get("axis_limits", None)
    if axis_limits is None and bounds is not None and "low" in bounds and "high" in bounds:
        low = bounds["low"]
        high = bounds["high"]
        if len(low) > max(x_idx, y_idx) and len(high) > max(x_idx, y_idx):
            axis_limits = {
                "x": [float(low[x_idx]), float(high[x_idx])],
                "y": [float(low[y_idx]), float(high[y_idx])],
            }
    if axis_limits is not None:
        spec["axis_limits"] = axis_limits

    spec["image_size"] = [int(spec["image_size"][0]), int(spec["image_size"][1])]
    spec["sigma_px"] = float(spec["sigma_px"])
    spec["threshold"] = float(spec["threshold"])
    spec["morph_open"] = int(spec["morph_open"])
    spec["morph_close"] = int(spec["morph_close"])
    spec["per_cluster_masks"] = bool(spec["per_cluster_masks"])
    spec["binary_blur_sigma"] = float(spec.get("binary_blur_sigma", 0.0))
    spec["enabled"] = bool(spec["enabled"])
    return spec


def _postprocess_binary_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    out = np.asarray(mask, dtype=float)
    if sigma > 0 and gaussian_filter is not None:
        out = gaussian_filter(out, sigma=float(sigma), mode="nearest")
        out = out >= 0.5
    else:
        out = out >= 0.5
    return out.astype(bool)


def render_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    rendering_spec: Dict[str, Any] | None,
) -> Dict[str, Any]:
    spec = resolve_rendering_spec(rendering_spec, feature_names)
    x_idx = int(spec["axis_map"]["x_idx"])
    y_idx = int(spec["axis_map"]["y_idx"])
    values_xy = X[:, [x_idx, y_idx]] if len(X) > 0 else np.zeros((0, 2), dtype=float)

    grayscale, binary_mask, info = _render_single(
        values_xy=values_xy,
        image_size=tuple(spec["image_size"]),
        render_mode=spec["render_mode"],
        sigma_px=spec["sigma_px"],
        normalize=spec["normalize"],
        threshold=spec["threshold"],
        morph_open=spec["morph_open"],
        morph_close=spec["morph_close"],
        axis_limits=spec.get("axis_limits"),
    )

    binary_mask = _postprocess_binary_mask(binary_mask, float(spec.get("binary_blur_sigma", 0.0)))

    cluster_masks: Dict[str, np.ndarray] = {}
    cluster_summaries: Dict[str, Dict[str, Any]] = {}
    if spec.get("per_cluster_masks", False) and len(X) > 0:
        for lab in np.unique(y):
            lab_mask = y == lab
            gray_i, bin_i, info_i = _render_single(
                values_xy=values_xy[lab_mask],
                image_size=tuple(spec["image_size"]),
                render_mode=spec["render_mode"],
                sigma_px=spec["sigma_px"],
                normalize=spec["normalize"],
                threshold=spec["threshold"],
                morph_open=spec["morph_open"],
                morph_close=spec["morph_close"],
                axis_limits=spec.get("axis_limits"),
            )
            bin_i = _postprocess_binary_mask(bin_i, float(spec.get("binary_blur_sigma", 0.0)))
            key = str(int(lab))
            cluster_masks[key] = bin_i
            cluster_summaries[key] = {
                "foreground_pixels": int(bin_i.sum()),
                "foreground_fraction": float(bin_i.mean()) if bin_i.size else 0.0,
                "max_intensity": float(np.max(gray_i)) if gray_i.size else 0.0,
                **info_i,
            }

    return {
        "spec": spec,
        "grayscale": grayscale,
        "binary_mask": binary_mask,
        "cluster_masks": cluster_masks,
        "summary": {
            "enabled": bool(spec.get("enabled", False)),
            "image_shape": list(grayscale.shape),
            "axis_map": spec["axis_map"],
            "render_mode": spec["render_mode"],
            "normalize": spec["normalize"],
            "threshold": float(spec["threshold"]),
            "binary_blur_sigma": float(spec.get("binary_blur_sigma", 0.0)),
            **info,
            "cluster_summaries": cluster_summaries,
        },
    }

# =========================
# Stage 1-4 extensions
# =========================
try:
    from scipy.ndimage import distance_transform_edt, sobel, binary_erosion, label
except Exception:  # pragma: no cover
    distance_transform_edt = None
    sobel = None
    binary_erosion = None
    label = None

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover
    skeletonize = None

DEFAULT_RENDERING_SPEC.update({
    "edge_sigma": 0.0,
    "skeletonize": True,
    "distance_transform": True,
    "local_thickness": True,
    "gradient_orientation": True,
    "cluster_id_raster": True,
    "extra_views": {
        "distance_transform": True,
        "skeleton_mask": True,
        "edge_map": True,
        "gradient_orientation_map": True,
        "local_thickness_map": True,
        "cluster_id_raster": True,
    },
})


def resolve_rendering_spec(rendering_spec: Dict[str, Any] | None, feature_names: List[str], bounds: Dict[str, Any] | None = None) -> Dict[str, Any]:  # type: ignore[override]
    spec = dict(DEFAULT_RENDERING_SPEC)
    spec.update(dict(rendering_spec or {}))
    ev = dict(DEFAULT_RENDERING_SPEC.get("extra_views", {}))
    ev.update(dict(spec.get("extra_views", {}) or {}))
    spec["extra_views"] = ev
    x_idx, y_idx, axis_info = _resolve_axis_map(feature_names, spec.get("axis_map"))
    spec["axis_map"] = axis_info

    axis_limits = spec.get("axis_limits", None)
    if axis_limits is None and bounds is not None and "low" in bounds and "high" in bounds:
        low = bounds["low"]
        high = bounds["high"]
        if len(low) > max(x_idx, y_idx) and len(high) > max(x_idx, y_idx):
            axis_limits = {"x": [float(low[x_idx]), float(high[x_idx])], "y": [float(low[y_idx]), float(high[y_idx])]}
    if axis_limits is not None:
        spec["axis_limits"] = axis_limits

    spec["image_size"] = [int(spec["image_size"][0]), int(spec["image_size"][1])]
    for k in ["sigma_px", "threshold", "binary_blur_sigma", "edge_sigma"]:
        spec[k] = float(spec.get(k, 0.0))
    for k in ["morph_open", "morph_close"]:
        spec[k] = int(spec.get(k, 0))
    for k in ["per_cluster_masks", "enabled", "skeletonize", "distance_transform", "local_thickness", "gradient_orientation", "cluster_id_raster"]:
        spec[k] = bool(spec.get(k, False))
    return spec


def _compute_edge_map(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if binary_erosion is not None:
        er = binary_erosion(mask)
        return mask & (~er)
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[1:, :] != mask[:-1, :]
    out[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return out


def _compute_gradient_orientation_map(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray
    if sobel is not None:
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)
    else:
        gy, gx = np.gradient(gray)
    ang = np.arctan2(gy, gx)
    mag = np.sqrt(gx * gx + gy * gy)
    if np.max(mag) > 1e-12:
        ang = (ang + np.pi) / (2.0 * np.pi)
        ang *= (mag / np.max(mag))
    else:
        ang = np.zeros_like(gray)
    return ang


def _compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    if distance_transform_edt is not None:
        return distance_transform_edt(mask)
    return mask.astype(float)


def _compute_skeleton(mask: np.ndarray) -> np.ndarray:
    if skeletonize is not None:
        return skeletonize(mask)
    dt = _compute_distance_transform(mask)
    if np.any(mask):
        thr = max(1.0, np.percentile(dt[mask], 80))
        return mask & (dt >= thr)
    return mask.copy()


def _compute_cluster_id_raster(values_xy: np.ndarray, y: np.ndarray, image_size: Tuple[int, int], axis_limits: Any | None) -> np.ndarray:
    h, w = int(image_size[0]), int(image_size[1])
    out = np.full((h, w), fill_value=-1, dtype=int)
    if len(values_xy) == 0:
        return out
    xlim, ylim = _resolve_axis_limits(values_xy, axis_limits)
    xs = (values_xy[:, 0] - xlim[0]) / max(1e-12, xlim[1] - xlim[0])
    ys = (values_xy[:, 1] - ylim[0]) / max(1e-12, ylim[1] - ylim[0])
    cols = np.clip((xs * (w - 1)).astype(int), 0, w - 1)
    rows = np.clip(((1.0 - ys) * (h - 1)).astype(int), 0, h - 1)
    for r, c, lab in zip(rows, cols, y):
        out[r, c] = int(lab)
    return out


def render_dataset(X: np.ndarray, y: np.ndarray, feature_names: List[str], rendering_spec: Dict[str, Any] | None) -> Dict[str, Any]:  # type: ignore[override]
    spec = resolve_rendering_spec(rendering_spec, feature_names)
    x_idx = int(spec["axis_map"]["x_idx"])
    y_idx = int(spec["axis_map"]["y_idx"])
    values_xy = X[:, [x_idx, y_idx]] if len(X) > 0 else np.zeros((0, 2), dtype=float)

    grayscale, binary_mask, info = _render_single(
        values_xy=values_xy,
        image_size=tuple(spec["image_size"]),
        render_mode=spec["render_mode"],
        sigma_px=spec["sigma_px"],
        normalize=spec["normalize"],
        threshold=spec["threshold"],
        morph_open=spec["morph_open"],
        morph_close=spec["morph_close"],
        axis_limits=spec.get("axis_limits"),
    )
    binary_mask = _postprocess_binary_mask(binary_mask, float(spec.get("binary_blur_sigma", 0.0)))

    extra_flags = dict(spec.get("extra_views", {}) or {})
    extra_flags["distance_transform"] = bool(spec.get("distance_transform", extra_flags.get("distance_transform", False)))
    extra_flags["local_thickness_map"] = bool(spec.get("local_thickness", extra_flags.get("local_thickness_map", False)))
    extra_flags["gradient_orientation_map"] = bool(spec.get("gradient_orientation", extra_flags.get("gradient_orientation_map", False)))
    extra_flags["cluster_id_raster"] = bool(spec.get("cluster_id_raster", extra_flags.get("cluster_id_raster", False)))
    extra_flags["skeleton_mask"] = bool(spec.get("skeletonize", extra_flags.get("skeleton_mask", False)))
    extra_flags.setdefault("edge_map", True)

    distance_transform = _compute_distance_transform(binary_mask) if extra_flags.get("distance_transform", False) else None
    skeleton_mask = _compute_skeleton(binary_mask) if extra_flags.get("skeleton_mask", False) else None
    edge_map = _compute_edge_map(binary_mask) if extra_flags.get("edge_map", False) else None
    gradient_orientation_map = _compute_gradient_orientation_map(grayscale) if extra_flags.get("gradient_orientation_map", False) else None
    local_thickness_map = (2.0 * distance_transform) if (distance_transform is not None and extra_flags.get("local_thickness_map", False)) else None
    cluster_id_raster = _compute_cluster_id_raster(values_xy, y, tuple(spec["image_size"]), spec.get("axis_limits")) if extra_flags.get("cluster_id_raster", False) else None

    cluster_masks: Dict[str, np.ndarray] = {}
    cluster_summaries: Dict[str, Dict[str, Any]] = {}
    if spec.get("per_cluster_masks", False) and len(X) > 0:
        for lab in np.unique(y):
            lab_mask = y == lab
            gray_i, bin_i, info_i = _render_single(
                values_xy=values_xy[lab_mask], image_size=tuple(spec["image_size"]), render_mode=spec["render_mode"], sigma_px=spec["sigma_px"],
                normalize=spec["normalize"], threshold=spec["threshold"], morph_open=spec["morph_open"], morph_close=spec["morph_close"], axis_limits=spec.get("axis_limits"),
            )
            bin_i = _postprocess_binary_mask(bin_i, float(spec.get("binary_blur_sigma", 0.0)))
            key = str(int(lab))
            cluster_masks[key] = bin_i
            cluster_summaries[key] = {
                "foreground_pixels": int(bin_i.sum()),
                "foreground_fraction": float(bin_i.mean()) if bin_i.size else 0.0,
                "max_intensity": float(np.max(gray_i)) if gray_i.size else 0.0,
                **info_i,
            }

    available_views = ["grayscale", "binary_mask"]
    if distance_transform is not None:
        available_views.append("distance_transform")
    if skeleton_mask is not None:
        available_views.append("skeleton_mask")
    if edge_map is not None:
        available_views.append("edge_map")
    if gradient_orientation_map is not None:
        available_views.append("gradient_orientation_map")
    if local_thickness_map is not None:
        available_views.append("local_thickness_map")
    if cluster_id_raster is not None:
        available_views.append("cluster_id_raster")

    summary = {
        "enabled": bool(spec.get("enabled", False)),
        "image_shape": list(grayscale.shape),
        "axis_map": spec["axis_map"],
        "render_mode": spec["render_mode"],
        "normalize": spec["normalize"],
        "threshold": float(spec["threshold"]),
        "binary_blur_sigma": float(spec.get("binary_blur_sigma", 0.0)),
        "available_views": available_views,
        **info,
        "cluster_summaries": cluster_summaries,
    }
    if distance_transform is not None:
        summary["distance_transform_max"] = float(np.max(distance_transform)) if distance_transform.size else 0.0
    if skeleton_mask is not None:
        summary["skeleton_pixels"] = int(np.sum(skeleton_mask))
    if edge_map is not None:
        summary["edge_pixels"] = int(np.sum(edge_map))

    return {
        "spec": spec,
        "grayscale": grayscale,
        "density_image": grayscale,
        "binary_mask": binary_mask,
        "distance_transform": distance_transform,
        "skeleton_mask": skeleton_mask,
        "edge_map": edge_map,
        "gradient_orientation_map": gradient_orientation_map,
        "local_thickness_map": local_thickness_map,
        "cluster_id_raster": cluster_id_raster,
        "cluster_masks": cluster_masks,
        "summary": summary,
    }
