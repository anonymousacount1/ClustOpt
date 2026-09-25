from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class Ladder(Component):
    name = "ladder"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "ladder")
        return {
            "dims": [i, j],
            "slope": float(s.get("slope", 0.0)),
            "base_intercept": float(s.get("base_intercept", 0.0)),
            "n_rungs": int(s.get("n_rungs", 6)),
            "rung_spacing": float(s.get("rung_spacing", 0.8)),
            "segment_x_len": float(s.get("segment_x_len", 1.2)),
            "x_center_low": float(s.get("x_center_low", -2.5)),
            "x_center_high": float(s.get("x_center_high", 2.5)),
            "thickness": float(s.get("thickness", 0.06)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "ladder")
        n_rungs = max(1, int(p["n_rungs"]))
        rung_idx = rng.integers(0, n_rungs, size=n_points)
        x_centers = rng.uniform(float(p["x_center_low"]), float(p["x_center_high"]), size=n_rungs)
        x = x_centers[rung_idx] + rng.uniform(-float(p["segment_x_len"]) / 2.0, float(p["segment_x_len"]) / 2.0, size=n_points)
        b_rung = float(p["base_intercept"]) + rung_idx * float(p["rung_spacing"])
        y = float(p["slope"]) * x + b_rung + rng.normal(0.0, float(p["thickness"]), size=n_points)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = x
        X[:, j] = y
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(Ladder())
