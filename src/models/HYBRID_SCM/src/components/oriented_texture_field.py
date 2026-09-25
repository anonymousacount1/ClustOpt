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


class OrientedTextureField(Component):
    name = "oriented_texture_field"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "oriented_texture_field")
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "width": float(s.get("width", 3.0)),
            "height": float(s.get("height", 2.0)),
            "orientation": float(s.get("orientation", 0.0)),
            "n_ridges": int(s.get("n_ridges", 6)),
            "ridge_spacing": float(s.get("ridge_spacing", 0.35)),
            "ridge_sigma": float(s.get("ridge_sigma", 0.06)),
            "cross_sigma": float(s.get("cross_sigma", 0.45)),
            "phase_jitter": float(s.get("phase_jitter", 0.08)),
            "strength_jitter": float(s.get("strength_jitter", 0.15)),
            "thickness": float(s.get("ridge_sigma", 0.06)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "oriented_texture_field")
        pool = _sample_rotated_rect(rng, max(6000, n_points * 8), float(p["cx"]), float(p["cy"]), float(p["width"]), float(p["height"]), float(p["orientation"]))
        rot = _rotation_matrix(-float(p["orientation"]))
        uv = (pool - np.array([[float(p["cx"]), float(p["cy"])]], dtype=float)) @ rot.T
        ridges = max(1, int(p["n_ridges"]))
        spacing = max(1e-6, float(p["ridge_spacing"]))
        centers = (np.arange(ridges) - 0.5 * (ridges - 1)) * spacing
        phase = rng.normal(0.0, float(p.get("phase_jitter", 0.0)), size=ridges)
        strengths = np.clip(1.0 + rng.normal(0.0, float(p.get("strength_jitter", 0.15)), size=ridges), 0.15, None)
        sigma = max(1e-6, float(p.get("ridge_sigma", 0.06)))
        cross = max(1e-6, float(p.get("cross_sigma", 0.45)))
        w = np.zeros((len(pool),), dtype=float)
        for c, ph, st in zip(centers, phase, strengths):
            offset = uv[:, 1] - (c + ph)
            w += st * np.exp(-(offset ** 2) / (2.0 * sigma * sigma)) * np.exp(-(uv[:, 0] ** 2) / (2.0 * cross * cross * max(float(p["width"]), 1e-6)))
        pts2 = _sample_points_from_weights(rng, pool, w, n_points)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(OrientedTextureField())
