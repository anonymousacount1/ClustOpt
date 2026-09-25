from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class Ellipse(Component):
    name = "ellipse"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "ellipse")
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "rx": float(s.get("rx", 1.4)),
            "ry": float(s.get("ry", 0.8)),
            "rotation": float(s.get("rotation", 0.0)),
            "angle_low": float(s.get("angle_low", 0.0)),
            "angle_high": float(s.get("angle_high", 2.0 * np.pi)),
            "thickness": float(s.get("thickness", 0.05)),
            "fill": bool(s.get("fill", False)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "ellipse")
        theta = rng.uniform(float(p["angle_low"]), float(p["angle_high"]), size=n_points)
        if bool(p.get("fill", False)):
            rad = np.sqrt(rng.uniform(0.0, 1.0, size=n_points))
        else:
            rad = 1.0 + rng.normal(0.0, float(p.get("thickness", 0.05)), size=n_points)
        x = float(p["rx"]) * rad * np.cos(theta)
        y = float(p["ry"]) * rad * np.sin(theta)
        rot = float(p.get("rotation", 0.0))
        xr = x * np.cos(rot) - y * np.sin(rot)
        yr = x * np.sin(rot) + y * np.cos(rot)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = float(p["cx"]) + xr
        X[:, j] = float(p["cy"]) + yr
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(Ellipse())
