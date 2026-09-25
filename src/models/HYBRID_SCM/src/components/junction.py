from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _as_float_list,
    _check_dims,
    _fill_irrelevant_dims,
    _resolve_refs,
)


class Junction(Component):
    name = "junction"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "junction")
        angles = _as_float_list(s.get("angles", [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]))
        lengths = _as_float_list(s.get("arm_lengths", 1.2), expected_len=len(angles), default=1.2)
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "angles": angles,
            "arm_lengths": lengths,
            "thickness": float(s.get("thickness", 0.06)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "junction")
        angles = p["angles"]
        lengths = p["arm_lengths"]
        arm_idx = rng.integers(0, len(angles), size=n_points)
        u = rng.uniform(0.0, 1.0, size=n_points)
        x = np.empty(n_points)
        y = np.empty(n_points)
        for k, ang in enumerate(angles):
            m = arm_idx == k
            if not np.any(m):
                continue
            L = float(lengths[k])
            x[m] = float(p["cx"]) + u[m] * L * np.cos(float(ang)) + rng.normal(0.0, float(p["thickness"]), size=np.sum(m))
            y[m] = float(p["cy"]) + u[m] * L * np.sin(float(ang)) + rng.normal(0.0, float(p["thickness"]), size=np.sum(m))
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = x
        X[:, j] = y
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(Junction())
