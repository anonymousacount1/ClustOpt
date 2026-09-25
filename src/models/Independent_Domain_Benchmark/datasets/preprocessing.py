"""Frozen, label-free preparation of every external dataset into one 2-D view set.

BENCHMARK PREPARATION, not method work. Everything in this module runs *before*
any method is timed and *outside* the ONLINE_RUNTIME boundary.

The single hard rule: **labels never participate**. Imputation, scaling and PCA
are all fitted on X alone. Labels exist only to (a) enforce the 2 <= K <= 5
eligibility rule during curation and (b) score predictions offline, after they
have been persisted.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: frozen preparation contract, referenced by hash from the dataset manifest
PREPARATION_PROTOCOL = {
    "protocol_id": "idb_prep_v1",
    "steps": [
        "1. numeric feature extraction (drop non-numeric / constant columns)",
        "2. deterministic missing-value handling: median imputation fitted on X",
        "3. StandardScaler fitted on X",
        "4. PCA(n_components=2, random_state=1234) fitted on X",
    ],
    "labels_used": False,
    "label_free_steps": ["imputation", "scaling", "pca"],
    "forbidden": ["UMAP", "t-SNE", "any supervised or label-aware embedding",
                  "any nonlinear manifold embedding"],
    "forbidden_reason": ("nonlinear embeddings can manufacture cluster structure "
                         "and add hyperparameters, which would make the external "
                         "claim uninterpretable"),
    "pca_seed": 1234,
    "native_2d_policy": ("a dataset that is already exactly 2-D numeric is used "
                         "as-is; scaling is still applied so all sources share "
                         "one representation contract, and PCA is skipped"),
    "views": {"1d_x": "X_2d[:, 0]", "1d_y": "X_2d[:, 1]", "2d": "X_2d"},
    "claim_wording": ("clustering of fixed unsupervised 2-D representations of "
                      "real-world labelled datasets -- NOT recovery of "
                      "universally true natural clusters"),
}


@dataclass
class PreparedDataset:
    dataset_id: str
    source: str
    X_2d: np.ndarray
    n_rows: int
    original_dim: int
    pca_applied: bool
    explained_variance_ratio: Optional[Tuple[float, float]]
    explained_variance_total: Optional[float]
    dropped_columns: int
    n_imputed_cells: int
    feature_checksum: str
    report: Dict[str, Any] = field(default_factory=dict)


def _checksum(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()
                          ).hexdigest()[:16]


def prepare(X: np.ndarray, dataset_id: str, source: str, *,
            seed: int = 1234) -> PreparedDataset:
    """X -> frozen 2-D representation. Never sees labels."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    n0, d0 = X.shape
    finite_col = np.isfinite(X).any(axis=0)
    non_constant = np.nanstd(X, axis=0) > 0
    keep = finite_col & non_constant
    dropped = int((~keep).sum())
    X = X[:, keep]

    n_imputed = int((~np.isfinite(X)).sum())
    if n_imputed:
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        idx = ~np.isfinite(X)
        X[idx] = np.take(med, np.where(idx)[1])

    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] == 2:
        X2, evr, tot, applied = Xs, None, None, False
    else:
        p = PCA(n_components=2, random_state=seed)
        X2 = p.fit_transform(Xs)
        evr = tuple(float(v) for v in p.explained_variance_ratio_)
        tot = float(sum(evr))
        applied = True

    return PreparedDataset(
        dataset_id=dataset_id, source=source, X_2d=np.ascontiguousarray(X2),
        n_rows=int(n0), original_dim=int(d0), pca_applied=applied,
        explained_variance_ratio=evr, explained_variance_total=tot,
        dropped_columns=dropped, n_imputed_cells=n_imputed,
        feature_checksum=_checksum(X2),
        report={"protocol_id": PREPARATION_PROTOCOL["protocol_id"],
                "n_rows": int(n0), "original_dim": int(d0),
                "dim_after_cleaning": int(Xs.shape[1]),
                "dropped_columns": dropped, "n_imputed_cells": n_imputed,
                "pca_applied": applied, "explained_variance_ratio": evr,
                "explained_variance_total": tot, "pca_seed": seed,
                "labels_used": False})


def views(X2: np.ndarray) -> Dict[str, np.ndarray]:
    """The frozen three-view protocol, identical to the historical convention."""
    return {"1d_x": X2[:, [0]], "1d_y": X2[:, [1]], "2d": X2}
