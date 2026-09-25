from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class RidgeField(Component):
    name = "ridge_field"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "ridge_field")
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "width": float(s.get("width", 3.0)),
            "height": float(s.get("height", 2.0)),
            "n_ridges": int(s.get("n_ridges", 5)),
            "orientation": float(s.get("orientation", 0.0)),
            "spacing": float(s.get("spacing", 0.5)),
            "ridge_sigma": float(s.get("ridge_sigma", 0.06)),
            "jitter": float(s.get("jitter", 0.02)),
            "thickness": float(s.get("ridge_sigma", 0.06)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "ridge_field")
        nr = max(1, int(p["n_ridges"]))
        centers = (np.arange(nr) - 0.5 * (nr - 1)) * float(p["spacing"])
        rid = rng.integers(0, nr, size=n_points)
        u = rng.uniform(-0.5 * float(p["width"]), 0.5 * float(p["width"]), size=n_points)
        v = centers[rid] + rng.normal(0.0, float(p["ridge_sigma"]), size=n_points)
        ang = float(p.get("orientation", 0.0))
        x = float(p["cx"]) + u * np.cos(ang) - v * np.sin(ang) + rng.normal(0.0, float(p.get("jitter", 0.0)), size=n_points)
        y = float(p["cy"]) + u * np.sin(ang) + v * np.cos(ang) + rng.normal(0.0, float(p.get("jitter", 0.0)), size=n_points)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = x
        X[:, j] = y
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(RidgeField())
