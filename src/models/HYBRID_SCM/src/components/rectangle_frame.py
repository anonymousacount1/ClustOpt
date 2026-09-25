from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class RectangleFrame(Component):
    name = "rectangle_frame"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "rectangle_frame")
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "width": float(s.get("width", 2.0)),
            "height": float(s.get("height", 1.2)),
            "rotation": float(s.get("rotation", 0.0)),
            "frame_thickness": float(s.get("frame_thickness", 0.05)),
            "fill": bool(s.get("fill", False)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "rectangle_frame")
        w = float(p["width"])
        h = float(p["height"])
        if bool(p.get("fill", False)):
            xy = np.column_stack([rng.uniform(-0.5 * w, 0.5 * w, size=n_points), rng.uniform(-0.5 * h, 0.5 * h, size=n_points)])
        else:
            side = rng.integers(0, 4, size=n_points)
            x = np.empty(n_points)
            y = np.empty(n_points)
            ft = float(p.get("frame_thickness", 0.05))
            m = side == 0
            x[m] = rng.uniform(-0.5 * w, 0.5 * w, size=np.sum(m)); y[m] = -0.5 * h + rng.normal(0.0, ft, size=np.sum(m))
            m = side == 1
            x[m] = rng.uniform(-0.5 * w, 0.5 * w, size=np.sum(m)); y[m] = 0.5 * h + rng.normal(0.0, ft, size=np.sum(m))
            m = side == 2
            x[m] = -0.5 * w + rng.normal(0.0, ft, size=np.sum(m)); y[m] = rng.uniform(-0.5 * h, 0.5 * h, size=np.sum(m))
            m = side == 3
            x[m] = 0.5 * w + rng.normal(0.0, ft, size=np.sum(m)); y[m] = rng.uniform(-0.5 * h, 0.5 * h, size=np.sum(m))
            xy = np.column_stack([x, y])
        rot = float(p.get("rotation", 0.0))
        xr = xy[:, 0] * np.cos(rot) - xy[:, 1] * np.sin(rot)
        yr = xy[:, 0] * np.sin(rot) + xy[:, 1] * np.cos(rot)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = float(p["cx"]) + xr
        X[:, j] = float(p["cy"]) + yr
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(RectangleFrame())
