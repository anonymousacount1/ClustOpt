from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class MicroBlobTexture(Component):
    name = "micro_blob_texture"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "micro_blob_texture")
        n_blobs = max(1, int(s.get("n_micro_blobs", 24)))
        cx = float(s.get("cx", 0.0)); cy = float(s.get("cy", 0.0))
        w = float(s.get("width", 2.6)); h = float(s.get("height", 2.0))
        if bool(s.get("clustered_centers", True)):
            anchor = np.column_stack([rng.uniform(cx - 0.3 * w, cx + 0.3 * w, size=max(2, n_blobs // 6)), rng.uniform(cy - 0.3 * h, cy + 0.3 * h, size=max(2, n_blobs // 6))])
            ids = rng.integers(0, len(anchor), size=n_blobs)
            centers = anchor[ids] + rng.normal(0.0, 0.12 * min(w, h), size=(n_blobs, 2))
        else:
            centers = np.column_stack([rng.uniform(cx - 0.5 * w, cx + 0.5 * w, size=n_blobs), rng.uniform(cy - 0.5 * h, cy + 0.5 * h, size=n_blobs)])
        sigmas = rng.uniform(float(s.get("blob_sigma_min", 0.03)), float(s.get("blob_sigma_max", 0.11)), size=n_blobs)
        return {
            "dims": [i, j], "cx": cx, "cy": cy, "width": w, "height": h,
            "n_micro_blobs": n_blobs,
            "blob_centers": centers.tolist(),
            "blob_sigmas": sigmas.tolist(),
            "strength_jitter": float(s.get("strength_jitter", 0.2)),
            "thickness": float(np.mean(sigmas)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "micro_blob_texture")
        centers = np.asarray(p["blob_centers"], dtype=float)
        sigmas = np.asarray(p["blob_sigmas"], dtype=float)
        probs = np.clip(1.0 + rng.normal(0.0, float(p.get("strength_jitter", 0.2)), size=len(centers)), 0.1, None)
        probs /= probs.sum()
        ids = rng.choice(len(centers), size=n_points, p=probs)
        pts2 = centers[ids] + rng.normal(0.0, sigmas[ids][:, None], size=(n_points, 2))
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(MicroBlobTexture())
