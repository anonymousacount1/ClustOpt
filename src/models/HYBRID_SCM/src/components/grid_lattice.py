from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class GridLattice(Component):
    name = "grid_lattice"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "grid_lattice")
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "n_vertical": max(1, int(s.get("n_vertical", 4))),
            "n_horizontal": max(1, int(s.get("n_horizontal", 4))),
            "spacing_x": float(s.get("spacing_x", 1.0)),
            "spacing_y": float(s.get("spacing_y", 1.0)),
            "line_thickness": float(s.get("line_thickness", 0.05)),
            "thickness": float(s.get("line_thickness", 0.05)),
            "jitter": float(s.get("jitter", 0.0)),
            "missing_lines_prob": float(s.get("missing_lines_prob", 0.0)),
            "span_x": None if s.get("span_x", None) is None else float(s.get("span_x")),
            "span_y": None if s.get("span_y", None) is None else float(s.get("span_y")),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "grid_lattice")
        cx, cy = float(p.get("cx", 0.0)), float(p.get("cy", 0.0))
        nv, nh = int(p.get("n_vertical", 4)), int(p.get("n_horizontal", 4))
        sx, sy = float(p.get("spacing_x", 1.0)), float(p.get("spacing_y", 1.0))
        jitter = float(p.get("jitter", 0.0))
        thick = float(p.get("line_thickness", p.get("thickness", 0.05)))
        miss = float(p.get("missing_lines_prob", 0.0))
        span_x = float(p.get("span_x") if p.get("span_x", None) is not None else max((nv - 1) * sx, 0.6 * (bounds_high[i] - bounds_low[i])))
        span_y = float(p.get("span_y") if p.get("span_y", None) is not None else max((nh - 1) * sy, 0.6 * (bounds_high[j] - bounds_low[j])))

        x_pos = cx + (np.arange(nv) - 0.5 * (nv - 1)) * sx
        y_pos = cy + (np.arange(nh) - 0.5 * (nh - 1)) * sy
        if jitter > 0:
            x_pos = x_pos + rng.normal(0.0, jitter, size=nv)
            y_pos = y_pos + rng.normal(0.0, jitter, size=nh)

        keep_v = rng.random(nv) >= miss
        keep_h = rng.random(nh) >= miss
        if not np.any(keep_v):
            keep_v[int(rng.integers(0, nv))] = True
        if not np.any(keep_h):
            keep_h[int(rng.integers(0, nh))] = True
        xv = x_pos[keep_v]
        yh = y_pos[keep_h]

        n_v_pts = int(round(n_points * len(xv) / max(1, len(xv) + len(yh))))
        n_h_pts = n_points - n_v_pts
        pts = []
        if len(xv) > 0 and n_v_pts > 0:
            ids = rng.integers(0, len(xv), size=n_v_pts)
            x = xv[ids] + rng.normal(0.0, thick, size=n_v_pts)
            y = rng.uniform(cy - 0.5 * span_y, cy + 0.5 * span_y, size=n_v_pts)
            pts.append(np.column_stack([x, y]))
        if len(yh) > 0 and n_h_pts > 0:
            ids = rng.integers(0, len(yh), size=n_h_pts)
            x = rng.uniform(cx - 0.5 * span_x, cx + 0.5 * span_x, size=n_h_pts)
            y = yh[ids] + rng.normal(0.0, thick, size=n_h_pts)
            pts.append(np.column_stack([x, y]))
        pts2 = np.vstack(pts) if pts else np.zeros((0, 2), dtype=float)
        if len(pts2) < n_points:
            extra = n_points - len(pts2)
            xe = rng.uniform(cx - 0.5 * span_x, cx + 0.5 * span_x, size=extra)
            ye = rng.uniform(cy - 0.5 * span_y, cy + 0.5 * span_y, size=extra)
            pts2 = np.vstack([pts2, np.column_stack([xe, ye])])
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:n_points, 0]
        X[:, j] = pts2[:n_points, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(GridLattice())
