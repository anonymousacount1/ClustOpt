from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _resolve_refs(param_spec: Dict[str, Any], parent_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not param_spec:
        return {}
    resolved: Dict[str, Any] = {}
    for k, v in param_spec.items():
        if isinstance(v, dict) and "ref" in v:
            ref = v["ref"]
            comp = ref["component"]
            key = ref["key"]
            if comp not in parent_params:
                raise KeyError(f"Ref param '{k}' points to unknown parent component '{comp}'.")
            if key not in parent_params[comp]:
                raise KeyError(f"Ref param '{k}' points to missing key '{key}' in parent '{comp}'.")
            resolved[k] = parent_params[comp][key]
        else:
            resolved[k] = v
    return resolved


def _check_dims(dims: Any, n_features: int, where: str) -> Tuple[int, int]:
    if not isinstance(dims, (list, tuple)) or len(dims) != 2:
        raise ValueError(f"{where}.dims must be [x_dim, y_dim].")
    i, j = int(dims[0]), int(dims[1])
    if i == j or not (0 <= i < n_features) or not (0 <= j < n_features):
        raise ValueError(f"{where}.dims must refer to two different feature indices.")
    return i, j


def _sample_from_intervals(rng: np.random.Generator, n: int, intervals: list[list[float]]) -> np.ndarray:
    lens = np.array([max(0.0, float(b) - float(a)) for a, b in intervals], dtype=float)
    if np.all(lens <= 0):
        raise ValueError("All intervals have non-positive length.")
    probs = lens / lens.sum()
    choices = rng.choice(len(intervals), size=n, p=probs)
    x = np.empty((n,), dtype=float)
    for k in range(len(intervals)):
        idx = np.where(choices == k)[0]
        if idx.size == 0:
            continue
        a, b = intervals[k]
        x[idx] = rng.uniform(float(a), float(b), size=idx.size)
    return x


def _equal_bins_in_interval(rng: np.random.Generator, a: float, b: float, n: int, bins: int, jitter_sigma: float) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=float)
    bins = max(1, int(bins))
    idx = np.arange(n) % bins
    rng.shuffle(idx)
    edges = np.linspace(a, b, bins + 1)
    x = np.empty(n, dtype=float)
    for k in range(bins):
        sel = np.where(idx == k)[0]
        if sel.size == 0:
            continue
        lo, hi = edges[k], edges[k + 1]
        center = 0.5 * (lo + hi)
        width = hi - lo
        jitter = rng.normal(0.0, jitter_sigma if jitter_sigma > 0 else width / 10.0, size=sel.size)
        x[sel] = np.clip(center + jitter, lo, hi)
    return x


def _sample_x(
    rng: np.random.Generator,
    n: int,
    x_low: float,
    x_high: float,
    intervals: Optional[list[list[float]]],
    x_sampling: str,
    x_bins_per_interval: int,
    x_jitter_sigma: float,
) -> np.ndarray:
    x_sampling = str(x_sampling).lower()
    if intervals is None:
        intervals = [[x_low, x_high]]

    if x_sampling == "uniform":
        return _sample_from_intervals(rng, n, intervals)

    if x_sampling != "equal_bins":
        raise ValueError(f"Unsupported x_sampling '{x_sampling}'.")

    lens = np.array([max(1e-12, float(b) - float(a)) for a, b in intervals], dtype=float)
    probs = lens / lens.sum()
    counts = rng.multinomial(int(n), probs)
    parts = []
    for (a, b), c in zip(intervals, counts):
        parts.append(_equal_bins_in_interval(rng, float(a), float(b), int(c), x_bins_per_interval, x_jitter_sigma))
    x = np.concatenate(parts) if parts else np.zeros((0,), dtype=float)
    rng.shuffle(x)
    return x


def _generate_intervals_from_spec(
    rng: np.random.Generator,
    spec: Dict[str, Any],
    x_low: float,
    x_high: float,
) -> list[list[float]]:
    n_windows = int(spec.get("n_windows", 5))
    coverage = float(spec.get("coverage", 0.65))
    min_gap = float(spec.get("min_gap", 0.0))
    edge_margin = float(spec.get("edge_margin", 0.0))
    length_jitter = float(spec.get("length_jitter", 0.15))
    start_jitter = float(spec.get("start_jitter", 0.0))

    span = max(1e-12, x_high - x_low)
    usable_low = x_low + edge_margin
    usable_high = x_high - edge_margin
    usable_span = max(1e-12, usable_high - usable_low)
    total_window = coverage * usable_span
    total_gap = max(0.0, usable_span - total_window)
    base_len = total_window / max(1, n_windows)

    raw_lens = []
    for _ in range(n_windows):
        mult = 1.0 + rng.uniform(-length_jitter, length_jitter)
        raw_lens.append(max(1e-12, base_len * mult))
    raw_lens = np.array(raw_lens, dtype=float)
    raw_lens *= total_window / raw_lens.sum()

    if n_windows == 1:
        gaps = np.array([], dtype=float)
    else:
        fixed_gap_total = min_gap * (n_windows - 1)
        remain = max(0.0, total_gap - fixed_gap_total)
        rand = rng.uniform(0.0, 1.0, size=n_windows - 1)
        rand = remain * rand / max(1e-12, rand.sum()) if rand.sum() > 0 else np.zeros(n_windows - 1)
        gaps = min_gap + rand

    starts = []
    cur = usable_low
    for k in range(n_windows):
        starts.append(cur)
        cur += raw_lens[k]
        if k < len(gaps):
            cur += gaps[k]

    intervals = []
    for a, L in zip(starts, raw_lens):
        shift = rng.uniform(-start_jitter, start_jitter)
        aa = np.clip(a + shift, usable_low, usable_high)
        bb = np.clip(aa + L, usable_low, usable_high)
        if bb - aa <= 1e-12:
            continue
        intervals.append([float(aa), float(bb)])

    intervals.sort(key=lambda z: z[0])
    return intervals


def _normalize_irrelevant_spec(spec: Any, n_features: int) -> Dict[str, Any]:
    default = {"mode": "gaussian_mid", "sigma": 0.25}
    if spec is None:
        return default
    if isinstance(spec, dict):
        out = dict(default)
        out.update(spec)
        return out
    if isinstance(spec, str):
        return {"mode": spec.lower(), "sigma": 0.25}
    if isinstance(spec, (int, float, np.integer, np.floating)):
        return {"mode": "gaussian_mid", "sigma": float(spec)}
    if isinstance(spec, (list, tuple, np.ndarray)):
        vals = list(spec)
        if len(vals) == 0:
            return default
        if len(vals) == 1 and isinstance(vals[0], (int, float, np.integer, np.floating)):
            return {"mode": "gaussian_mid", "sigma": float(vals[0])}
        if len(vals) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            lo, hi = float(vals[0]), float(vals[1])
            if lo <= hi:
                return {"mode": "uniform", "low": lo, "high": hi}
            return {"mode": "gaussian_mid", "sigma": max(abs(lo), abs(hi))}
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            vals_f = [float(v) for v in vals]
            if len(vals_f) == n_features:
                return {"mode": "gaussian_mid", "sigma_per_dim": vals_f}
            return {"mode": "gaussian_mid", "sigma": float(np.mean(np.abs(vals_f)))}
    return default


def _fill_irrelevant_dims(
    rng: np.random.Generator,
    X: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    active_dims: Tuple[int, int],
    spec: Optional[Dict[str, Any]],
) -> None:
    n, d = X.shape
    spec_n = _normalize_irrelevant_spec(spec, d)
    mode = str(spec_n.get("mode", "gaussian_mid")).lower()
    sigma = float(spec_n.get("sigma", 0.25))
    sigma_per_dim = spec_n.get("sigma_per_dim", None)
    low_override = spec_n.get("low", None)
    high_override = spec_n.get("high", None)
    active = set(active_dims)
    for k in range(d):
        if k in active:
            continue
        if mode == "uniform":
            lo = bounds_low[k] if low_override is None else float(low_override)
            hi = bounds_high[k] if high_override is None else float(high_override)
            if hi < lo:
                lo, hi = hi, lo
            X[:, k] = rng.uniform(lo, hi, size=n)
        else:
            sigma_k = float(sigma_per_dim[k]) if isinstance(sigma_per_dim, (list, tuple, np.ndarray)) and k < len(sigma_per_dim) else sigma
            sigma_k = max(1e-12, abs(float(sigma_k)))
            mid = 0.5 * (bounds_low[k] + bounds_high[k])
            X[:, k] = rng.normal(mid, sigma_k, size=n)


def _roi_or_global(bounds_low: np.ndarray, bounds_high: np.ndarray, roi: Optional[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if roi is None:
        return bounds_low, bounds_high
    low = np.array(roi.get("low", bounds_low), dtype=float)
    high = np.array(roi.get("high", bounds_high), dtype=float)
    return low, high


def _quantize(v: np.ndarray, step: Optional[float], origin: float = 0.0) -> np.ndarray:
    if step is None:
        return v
    step = float(step)
    if step <= 0:
        return v
    return origin + np.round((v - origin) / step) * step


def _eps_gaussian_or_donut(
    rng: np.random.Generator,
    n: int,
    sigma: float,
    min_abs_offset: Optional[float],
    max_abs_offset: Optional[float],
) -> np.ndarray:
    eps = rng.normal(0.0, sigma, size=n)
    if min_abs_offset is None and max_abs_offset is None:
        return eps
    if min_abs_offset is None:
        min_abs_offset = 0.0
    mag = np.abs(eps)
    mag = np.maximum(mag, float(min_abs_offset))
    if max_abs_offset is not None:
        mag = np.minimum(mag, float(max_abs_offset))
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    return sign * mag


def _coerce_xy_pair(v: Any, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    if v is None:
        return float(default[0]), float(default[1])
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    return float(v), 0.0


def _as_float_list(v: Any, expected_len: Optional[int] = None, default: float = 1.0) -> List[float]:
    if v is None:
        if expected_len is None:
            return []
        return [float(default)] * int(expected_len)
    if isinstance(v, (list, tuple, np.ndarray)):
        vals = [float(x) for x in v]
    else:
        if expected_len is None:
            return [float(v)]
        vals = [float(v)] * int(expected_len)
    if expected_len is not None:
        if len(vals) < int(expected_len):
            vals = vals + [vals[-1] if vals else float(default)] * (int(expected_len) - len(vals))
        elif len(vals) > int(expected_len):
            vals = vals[: int(expected_len)]
    return vals


def _chaikin_smooth(vertices: np.ndarray, weight: float, closed: bool, n_iter: int = 1) -> np.ndarray:
    pts = np.asarray(vertices, dtype=float)
    if len(pts) < 3 or weight <= 0:
        return pts
    w = float(np.clip(weight, 0.0, 0.45))
    for _ in range(max(1, int(n_iter))):
        new_pts = []
        if not closed:
            new_pts.append(pts[0])
        n = len(pts)
        limit = n if closed else n - 1
        for k in range(limit):
            a = pts[k]
            b = pts[(k + 1) % n]
            q = (1.0 - w) * a + w * b
            r = w * a + (1.0 - w) * b
            new_pts.extend([q, r])
        if not closed:
            new_pts.append(pts[-1])
        pts = np.asarray(new_pts, dtype=float)
    return pts


def _polyline_vertices_from_spec(spec: Dict[str, Any], bounds_low: np.ndarray, bounds_high: np.ndarray) -> np.ndarray:
    if spec.get("vertices", None) is not None:
        verts = np.asarray(spec["vertices"], dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 2:
            raise ValueError("piecewise_polyline.vertices must be a list of [x, y] with length >= 2.")
        return verts

    n_segments = max(1, int(spec.get("n_segments", 3)))
    seg_lengths = _as_float_list(spec.get("segment_lengths", None), expected_len=n_segments, default=1.0)
    ang_raw = spec.get("angles", None)
    if ang_raw is None:
        base = np.linspace(-0.75, 0.75, n_segments)
        angles = base.tolist()
    else:
        angles = _as_float_list(ang_raw)
        if len(angles) == n_segments - 1:
            cur = 0.0
            out = []
            for da in angles:
                cur += float(da)
                out.append(cur)
            angles = out
        elif len(angles) != n_segments:
            angles = _as_float_list(angles[-1] if angles else 0.0, expected_len=n_segments, default=0.0)
    x0 = float(spec.get("x_start", 0.5 * (bounds_low[0] + bounds_high[0])))
    y0 = float(spec.get("y_start", 0.5 * (bounds_low[1] + bounds_high[1])))
    verts = [np.array([x0, y0], dtype=float)]
    cur = verts[0].copy()
    for L, ang in zip(seg_lengths, angles):
        cur = cur + float(L) * np.array([np.cos(float(ang)), np.sin(float(ang))], dtype=float)
        verts.append(cur.copy())
    return np.asarray(verts, dtype=float)


def _sample_on_polyline(
    rng: np.random.Generator,
    vertices: np.ndarray,
    n_points: int,
    thickness: float,
    closed: bool = False,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    if closed:
        verts2 = np.vstack([verts, verts[0]])
    else:
        verts2 = verts
    segs = verts2[1:] - verts2[:-1]
    lens = np.linalg.norm(segs, axis=1)
    if np.all(lens <= 1e-12):
        return np.repeat(verts[:1], repeats=n_points, axis=0)
    probs = lens / lens.sum()
    counts = rng.multinomial(int(n_points), probs)
    pts = []
    for k, c in enumerate(counts):
        if c <= 0:
            continue
        a = verts2[k]
        b = verts2[k + 1]
        d = b - a
        L = max(1e-12, float(lens[k]))
        u = rng.uniform(0.0, 1.0, size=c)
        base = a[None, :] + u[:, None] * d[None, :]
        normal = np.array([-d[1], d[0]], dtype=float) / L
        base += rng.normal(0.0, thickness, size=(c, 1)) * normal[None, :]
        pts.append(base)
    out = np.vstack(pts) if pts else np.repeat(verts[:1], repeats=n_points, axis=0)
    rng.shuffle(out)
    return out


def _project_points_to_polyline(points: np.ndarray, vertices: np.ndarray, closed: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    verts = np.asarray(vertices, dtype=float)
    if closed:
        verts2 = np.vstack([verts, verts[0]])
    else:
        verts2 = verts
    seg_starts = verts2[:-1]
    seg_ends = verts2[1:]
    seg_vecs = seg_ends - seg_starts
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    seg_lens_safe = np.maximum(seg_lens, 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = max(1e-12, float(cum[-1]))

    best_d2 = np.full(len(pts), np.inf, dtype=float)
    best_proj = np.zeros_like(pts)
    best_t = np.zeros((len(pts),), dtype=float)

    for k, (a, v, L) in enumerate(zip(seg_starts, seg_vecs, seg_lens_safe)):
        w = pts - a[None, :]
        u = np.sum(w * v[None, :], axis=1) / (L * L)
        u = np.clip(u, 0.0, 1.0)
        proj = a[None, :] + u[:, None] * v[None, :]
        d2 = np.sum((pts - proj) ** 2, axis=1)
        mask = d2 < best_d2
        best_d2[mask] = d2[mask]
        best_proj[mask] = proj[mask]
        best_t[mask] = (cum[k] + u[mask] * seg_lens[k]) / total
    return best_proj, np.clip(best_t, 0.0, 1.0)


def _sample_on_segment_set(
    rng: np.random.Generator,
    vertices: np.ndarray,
    edges: np.ndarray,
    n_points: int,
    thickness: float,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    eds = np.asarray(edges, dtype=int)
    if len(verts) == 0 or len(eds) == 0:
        return np.zeros((n_points, 2), dtype=float)
    segs = verts[eds[:, 1]] - verts[eds[:, 0]]
    lens = np.linalg.norm(segs, axis=1)
    if np.all(lens <= 1e-12):
        return np.repeat(verts[:1], repeats=n_points, axis=0)
    probs = lens / lens.sum()
    counts = rng.multinomial(int(n_points), probs)
    pts = []
    for k, c in enumerate(counts):
        if c <= 0:
            continue
        a = verts[int(eds[k, 0])]
        b = verts[int(eds[k, 1])]
        d = b - a
        L = max(1e-12, float(lens[k]))
        u = rng.uniform(0.0, 1.0, size=c)
        base = a[None, :] + u[:, None] * d[None, :]
        normal = np.array([-d[1], d[0]], dtype=float) / L
        base += rng.normal(0.0, thickness, size=(c, 1)) * normal[None, :]
        pts.append(base)
    out = np.vstack(pts) if pts else np.repeat(verts[:1], repeats=n_points, axis=0)
    rng.shuffle(out)
    return out


def _project_points_to_segment_set(points: np.ndarray, vertices: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    verts = np.asarray(vertices, dtype=float)
    eds = np.asarray(edges, dtype=int)
    if len(pts) == 0 or len(verts) == 0 or len(eds) == 0:
        return np.zeros_like(pts), np.zeros((len(pts),), dtype=float)

    seg_starts = verts[eds[:, 0]]
    seg_ends = verts[eds[:, 1]]
    seg_vecs = seg_ends - seg_starts
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    seg_lens_safe = np.maximum(seg_lens, 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = max(1e-12, float(cum[-1]))

    best_d2 = np.full(len(pts), np.inf, dtype=float)
    best_proj = np.zeros_like(pts)
    best_t = np.zeros((len(pts),), dtype=float)

    for k, (a, v, L) in enumerate(zip(seg_starts, seg_vecs, seg_lens_safe)):
        w = pts - a[None, :]
        u = np.sum(w * v[None, :], axis=1) / (L * L)
        u = np.clip(u, 0.0, 1.0)
        proj = a[None, :] + u[:, None] * v[None, :]
        d2 = np.sum((pts - proj) ** 2, axis=1)
        mask = d2 < best_d2
        best_d2[mask] = d2[mask]
        best_proj[mask] = proj[mask]
        best_t[mask] = (cum[k] + u[mask] * seg_lens[k]) / total
    return best_proj, np.clip(best_t, 0.0, 1.0)


def _build_branching_tree_geometry(
    rng: np.random.Generator,
    root: Tuple[float, float],
    branching_factor: int,
    depth: int,
    angle_spread: float,
    segment_length: float,
    length_decay: float,
    pruning_probability: float,
    angle_jitter: float,
) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[np.ndarray] = [np.array(root, dtype=float)]
    edges: List[Tuple[int, int]] = []

    def recurse(parent_idx: int, angle: float, level: int, length: float) -> None:
        if level >= depth or length <= 1e-6:
            return
        b = max(1, int(branching_factor))
        base_angles = np.linspace(-0.5 * angle_spread, 0.5 * angle_spread, num=b)
        for da in base_angles:
            if level > 0 and rng.random() < pruning_probability:
                continue
            child_angle = angle + float(da) + rng.normal(0.0, angle_jitter)
            parent = vertices[parent_idx]
            child = parent + float(length) * np.array([np.cos(child_angle), np.sin(child_angle)], dtype=float)
            child_idx = len(vertices)
            vertices.append(child)
            edges.append((parent_idx, child_idx))
            recurse(child_idx, child_angle, level + 1, length * length_decay)

    recurse(0, np.pi / 2.0, 0, segment_length)
    return np.asarray(vertices, dtype=float), np.asarray(edges, dtype=int)


def _build_meander_vertices(
    rng: np.random.Generator,
    x_low: float,
    x_high: float,
    y_center: float,
    amplitude: float,
    frequency: float,
    phase: float,
    amplitude_drift: float,
    frequency_drift: float,
    n_anchors: int,
    anchor_jitter: float,
) -> np.ndarray:
    n_anchors = max(8, int(n_anchors))
    x = np.linspace(x_low, x_high, n_anchors)
    amp_scale = 1.0 + amplitude_drift * np.linspace(-0.5, 0.5, n_anchors)
    freq_scale = 1.0 + frequency_drift * np.linspace(-0.5, 0.5, n_anchors)
    local_phase = phase + 2.0 * np.pi * frequency * freq_scale * (x - x_low) / max(1e-12, x_high - x_low)
    y = y_center + amplitude * amp_scale * np.sin(local_phase)
    if anchor_jitter > 0:
        y = y + rng.normal(0.0, anchor_jitter, size=n_anchors)
    return np.column_stack([x, y])


def _rotation_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _sample_rotated_rect(rng: np.random.Generator, n: int, cx: float, cy: float, width: float, height: float, orientation: float) -> np.ndarray:
    uv = np.column_stack([
        rng.uniform(-0.5 * width, 0.5 * width, size=n),
        rng.uniform(-0.5 * height, 0.5 * height, size=n),
    ])
    rot = _rotation_matrix(orientation)
    pts = uv @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def _sample_points_from_weights(rng: np.random.Generator, pts: np.ndarray, weights: np.ndarray, n_points: int) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, None)
    w = w / np.sum(w)
    idx = rng.choice(len(pts), size=n_points, replace=True, p=w)
    out = pts[idx].copy()
    rng.shuffle(out)
    return out
