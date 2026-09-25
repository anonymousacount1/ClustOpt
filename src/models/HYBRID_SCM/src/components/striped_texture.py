from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _fill_irrelevant_dims,
    _resolve_refs,
    _rotation_matrix,
    _sample_points_from_weights,
    _sample_rotated_rect,
)


class StripedTexture(Component):
    name = "striped_texture"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "striped_texture")
        return {
            "dims": [i, j], "cx": float(s.get("cx", 0.0)), "cy": float(s.get("cy", 0.0)),
            "width": float(s.get("width", 3.0)), "height": float(s.get("height", 2.2)),
            "orientation": float(s.get("orientation", 0.0)),
            "n_stripes": int(s.get("n_stripes", 7)),
            "stripe_spacing": float(s.get("stripe_spacing", 0.35)),
            "stripe_sigma": float(s.get("stripe_sigma", 0.05)),
            "phase_offset": float(s.get("phase_offset", 0.0)),
            "duty_cycle": float(s.get("duty_cycle", 0.55)),
            "thickness": float(s.get("stripe_sigma", 0.05)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "striped_texture")
        pool = _sample_rotated_rect(rng, max(5000, n_points * 8), float(p["cx"]), float(p["cy"]), float(p["width"]), float(p["height"]), float(p["orientation"]))
        uv = (pool - np.array([[float(p["cx"]), float(p["cy"])]], dtype=float)) @ _rotation_matrix(-float(p["orientation"])).T
        spacing = max(1e-6, float(p["stripe_spacing"]))
        sigma = max(1e-6, float(p.get("stripe_sigma", 0.05)))
        freq = max(1, int(p.get("n_stripes", 7)))
        phase = float(p.get("phase_offset", 0.0))
        val = np.cos((2.0 * np.pi / spacing) * uv[:, 1] + phase)
        duty = float(np.clip(p.get("duty_cycle", 0.55), 0.05, 0.95))
        target = np.quantile(val, 1.0 - duty)
        w = np.exp(-((val - target) ** 2) / (2.0 * sigma * sigma)) + 0.05
        pts2 = _sample_points_from_weights(rng, pool, w, n_points)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(StripedTexture())
