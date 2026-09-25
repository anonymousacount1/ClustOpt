from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _chaikin_smooth,
    _check_dims,
    _fill_irrelevant_dims,
    _resolve_refs,
    _sample_on_polyline,
)


class Polygon(Component):
    name = "polygon"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "polygon")
        n_sides = max(3, int(s.get("n_sides", 4)))
        cx = float(s.get("cx", 0.0))
        cy = float(s.get("cy", 0.0))
        radius = float(s.get("radius", 1.0))
        rot = float(s.get("rotation", 0.0))
        angles = rot + np.linspace(0.0, 2.0 * np.pi, num=n_sides, endpoint=False)
        verts = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
        rounding = float(s.get("corner_rounding", 0.0))
        if rounding > 0:
            verts = _chaikin_smooth(verts, weight=min(0.45, 0.18 + 0.24 * rounding), closed=True, n_iter=1 + int(rounding > 0.45))
        return {
            "dims": [i, j],
            "cx": cx,
            "cy": cy,
            "n_sides": n_sides,
            "radius": radius,
            "r": radius,
            "rotation": rot,
            "corner_rounding": rounding,
            "vertices": [[float(x), float(y)] for x, y in verts.tolist()],
            "boundary_thickness": float(s.get("boundary_thickness", 0.05)),
            "thickness": float(s.get("boundary_thickness", 0.05)),
            "fill": bool(s.get("fill", False)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "polygon")
        verts = np.asarray(p["vertices"], dtype=float)
        thickness = float(p.get("boundary_thickness", p.get("thickness", 0.05)))
        if bool(p.get("fill", False)):
            center = np.array([float(p.get("cx", 0.0)), float(p.get("cy", 0.0))], dtype=float)
            edge_pts = _sample_on_polyline(rng, verts, n_points, thickness=0.0, closed=True)
            alpha = np.sqrt(rng.uniform(0.0, 1.0, size=n_points))[:, None]
            pts2 = center[None, :] + alpha * (edge_pts - center[None, :])
        else:
            pts2 = _sample_on_polyline(rng, verts, n_points, thickness=thickness, closed=True)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(Polygon())
