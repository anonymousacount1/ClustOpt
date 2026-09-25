from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _coerce_xy_pair,
    _fill_irrelevant_dims,
    _resolve_refs,
)


class RingAnnulus(Component):
    name = "ring_annulus"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "ring_annulus")
        dx, dy = _coerce_xy_pair(s.get("hole_offset", [0.0, 0.0]))
        inner = float(s.get("inner_radius", 0.6))
        outer = float(s.get("outer_radius", 1.0))
        if outer <= inner:
            outer = inner + max(1e-3, 0.25 * inner + 1e-3)
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "inner_radius": inner,
            "outer_radius": outer,
            "angle_low": float(s.get("angle_low", 0.0)),
            "angle_high": float(s.get("angle_high", 2.0 * np.pi)),
            "ellipse_ratio": float(s.get("ellipse_ratio", 1.0)),
            "hole_offset": [dx, dy],
            "r": 0.5 * (inner + outer),
            "thickness": 0.5 * (outer - inner),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "ring_annulus")
        cx, cy = float(p["cx"]), float(p["cy"])
        r_in, r_out = float(p["inner_radius"]), float(p["outer_radius"])
        er = max(1e-6, float(p.get("ellipse_ratio", 1.0)))
        dx_h, dy_h = _coerce_xy_pair(p.get("hole_offset", [0.0, 0.0]))
        a0 = float(p.get("angle_low", 0.0))
        a1 = float(p.get("angle_high", 2.0 * np.pi))
        span = max(1e-12, a1 - a0)

        pts = []
        remaining = n_points
        attempts = 0
        while remaining > 0 and attempts < 20:
            m = max(remaining * 3, 256)
            x = rng.uniform(cx - r_out, cx + r_out, size=m)
            y = rng.uniform(cy - r_out * er, cy + r_out * er, size=m)
            xo = (x - cx) / r_out
            yo = (y - cy) / (r_out * er)
            outer_mask = xo * xo + yo * yo <= 1.0
            xi = (x - (cx + dx_h)) / max(r_in, 1e-12)
            yi = (y - (cy + dy_h)) / (max(r_in, 1e-12) * er)
            inner_mask = xi * xi + yi * yi < 1.0
            theta = np.arctan2(y - cy, x - cx)
            theta = np.where(theta < a0, theta + 2.0 * np.pi, theta)
            sector_mask = (theta >= a0) & (theta <= a0 + span + 1e-12)
            keep = outer_mask & (~inner_mask) & sector_mask
            if np.any(keep):
                pts.append(np.column_stack([x[keep], y[keep]]))
                remaining = n_points - sum(len(z) for z in pts)
            attempts += 1
        pts2 = np.vstack(pts)[:n_points] if pts else np.zeros((0, 2), dtype=float)
        if len(pts2) < n_points:
            theta = rng.uniform(a0, a1, size=n_points - len(pts2))
            rr = np.sqrt(rng.uniform(r_in * r_in, r_out * r_out, size=n_points - len(pts2)))
            fill = np.column_stack([cx + rr * np.cos(theta), cy + er * rr * np.sin(theta)])
            pts2 = np.vstack([pts2, fill])
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(RingAnnulus())
