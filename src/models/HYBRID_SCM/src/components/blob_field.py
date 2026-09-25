from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class BlobField(Component):
    name = "blob_field"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "blob_field")
        nb = max(1, int(s.get("n_blobs", 6)))
        cx = float(s.get("cx", 0.0)); cy = float(s.get("cy", 0.0))
        w = float(s.get("width", 3.0)); h = float(s.get("height", 2.0))
        centers = np.column_stack([rng.uniform(cx - 0.5*w, cx + 0.5*w, size=nb), rng.uniform(cy - 0.5*h, cy + 0.5*h, size=nb)])
        sigmas = rng.uniform(float(s.get("blob_sigma_min", 0.08)), float(s.get("blob_sigma_max", 0.35)), size=nb)
        return {
            "dims": [i, j],
            "centers": centers.tolist(),
            "sigmas": sigmas.tolist(),
            "strength_jitter": float(s.get("strength_jitter", 0.15)),
            "thickness": float(np.mean(sigmas)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "blob_field")
        centers = np.asarray(p["centers"], dtype=float)
        sigmas = np.asarray(p["sigmas"], dtype=float)
        probs = np.ones(len(centers), dtype=float)
        sj = float(p.get("strength_jitter", 0.15))
        if sj > 0:
            probs *= np.clip(1.0 + rng.normal(0.0, sj, size=len(centers)), 0.05, None)
        probs /= probs.sum()
        idx = rng.choice(len(centers), size=n_points, p=probs)
        pts = centers[idx] + rng.normal(0.0, sigmas[idx][:, None], size=(n_points, 2))
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts[:, 0]
        X[:, j] = pts[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(BlobField())
