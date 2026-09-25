from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _fill_irrelevant_dims,
    _generate_intervals_from_spec,
    _resolve_refs,
    _sample_x,
)


class Parabola(Component):
    name = "parabola"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "parabola")
        x_low = float(s.get("x_low", -3.0))
        x_high = float(s.get("x_high", 3.0))
        intervals = s.get("intervals", None)
        if intervals is None and s.get("interval_generation", None) is not None:
            intervals = _generate_intervals_from_spec(rng, s["interval_generation"], x_low=x_low, x_high=x_high)
        return {
            "dims": [i, j],
            "a": float(s.get("a", 0.7)),
            "x0": float(s.get("x0", 0.0)),
            "b": float(s.get("b", 0.0)),
            "x_low": x_low,
            "x_high": x_high,
            "intervals": intervals,
            "thickness": float(s.get("thickness", 0.10)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "parabola")
        x = _sample_x(rng, n_points, float(p["x_low"]), float(p["x_high"]), p.get("intervals", None), "uniform", 20, 0.0)
        y = float(p["a"]) * (x - float(p["x0"])) ** 2 + float(p["b"]) + rng.normal(0.0, float(p["thickness"]), size=n_points)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = x
        X[:, j] = y
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(Parabola())
