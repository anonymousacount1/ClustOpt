from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class Spiral(Component):
    name = "spiral"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "spiral")
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "r_start": float(s.get("r_start", 0.2)),
            "r_end": float(s.get("r_end", 2.0)),
            "turns": float(s.get("turns", 2.5)),
            "angle_offset": float(s.get("angle_offset", 0.0)),
            "thickness": float(s.get("thickness", 0.05)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "spiral")
        t = np.sort(rng.uniform(0.0, 1.0, size=n_points))
        theta = float(p["angle_offset"]) + 2.0 * np.pi * float(p["turns"]) * t
        r = float(p["r_start"]) + (float(p["r_end"]) - float(p["r_start"])) * t
        r = r + rng.normal(0.0, float(p.get("thickness", 0.05)), size=n_points)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = float(p["cx"]) + r * np.cos(theta)
        X[:, j] = float(p["cy"]) + r * np.sin(theta)
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(Spiral())
