from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _fill_irrelevant_dims,
    _resolve_refs,
    _rotation_matrix,
)


class AnisotropicGrainField(Component):
    name = "anisotropic_grain_field"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "anisotropic_grain_field")
        n_grains = max(1, int(s.get("n_grains", 20)))
        cx = float(s.get("cx", 0.0)); cy = float(s.get("cy", 0.0))
        w = float(s.get("width", 3.0)); h = float(s.get("height", 2.2))
        centers = np.column_stack([rng.uniform(cx - 0.5 * w, cx + 0.5 * w, size=n_grains), rng.uniform(cy - 0.5 * h, cy + 0.5 * h, size=n_grains)])
        oris = float(s.get("orientation", 0.0)) + rng.normal(0.0, float(s.get("orientation_jitter", 0.25)), size=n_grains)
        return {
            "dims": [i, j], "cx": cx, "cy": cy, "width": w, "height": h,
            "grain_centers": centers.tolist(),
            "grain_orientations": oris.tolist(),
            "grain_sigma_long": float(s.get("grain_sigma_long", 0.12)),
            "grain_sigma_short": float(s.get("grain_sigma_short", 0.035)),
            "orientation": float(s.get("orientation", 0.0)),
            "orientation_jitter": float(s.get("orientation_jitter", 0.25)),
            "strength_jitter": float(s.get("strength_jitter", 0.15)),
            "thickness": float(s.get("grain_sigma_short", 0.035)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "anisotropic_grain_field")
        centers = np.asarray(p["grain_centers"], dtype=float)
        oris = np.asarray(p["grain_orientations"], dtype=float)
        probs = np.clip(1.0 + rng.normal(0.0, float(p.get("strength_jitter", 0.15)), size=len(centers)), 0.1, None)
        probs /= probs.sum()
        ids = rng.choice(len(centers), size=n_points, p=probs)
        long_s = max(1e-6, float(p.get("grain_sigma_long", 0.12)))
        short_s = max(1e-6, float(p.get("grain_sigma_short", 0.035)))
        pts = np.zeros((n_points, 2), dtype=float)
        for idx in range(n_points):
            gid = ids[idx]
            R = _rotation_matrix(float(oris[gid]))
            eps = np.array([rng.normal(0.0, long_s), rng.normal(0.0, short_s)], dtype=float) @ R.T
            pts[idx] = centers[gid] + eps
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts[:, 0]
        X[:, j] = pts[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(AnisotropicGrainField())
