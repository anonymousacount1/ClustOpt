"""
knn_predictor.py
================

Two things live here:

1. :func:`predict_from_neighbors` — the pure prediction kernel.  Given cached
   neighbour indices/distances (from a single max-neighbour query) plus the
   training utility matrix, it derives the utility prediction for a given
   ``n_neighbors`` and neighbour weighting.  This is what the configuration
   search calls once per (n_neighbors, weighting) without re-querying.

2. :class:`KnnUtilityPredictor` — the frozen, reloadable final model.  It holds
   three view-specific sub-models (preprocessor + neighbour index + training
   utilities) and exposes the stable ``predict_utility`` API used by Phase E2.

Prediction formula (plan §11)::

    u_hat(x) = sum_j w_j u_j / sum_j w_j      over the K_neighbors nearest j

    uniform:           w_j = 1
    inverse_distance:  w_j = 1 / (d_j + eps)

If exact matches exist (d_j <= exact_match_tol) the prediction averages *only*
the exact-match neighbours, avoiding unstable inverse-distance blow-ups.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .neighbor_search import ViewNeighborIndex
from .preprocessing import FeaturePreprocessor


@dataclass
class PredictionDebug:
    neighbor_indices: np.ndarray       # [Nq, n_neighbors]
    neighbor_distances: np.ndarray     # [Nq, n_neighbors]
    normalized_weights: np.ndarray     # [Nq, n_neighbors]
    effective_neighbor_count: np.ndarray  # [Nq]
    exact_match_row: np.ndarray        # [Nq] bool


def predict_from_neighbors(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    train_utilities: np.ndarray,
    n_neighbors: int,
    weighting: str,
    eps: float = C.EPS,
    exact_match_tol: float = C.EXACT_MATCH_TOL,
    want_debug: bool = False,
):
    """Derive predictions for ``n_neighbors`` and ``weighting`` from cached
    neighbour results.  Returns ``pred`` ([Nq, M]) or ``(pred, PredictionDebug)``.
    """
    idx = neighbor_indices[:, :n_neighbors]
    dist = neighbor_distances[:, :n_neighbors].astype(np.float64)
    nq, n = idx.shape
    neigh_u = train_utilities[idx]                       # [Nq, n, M]

    if weighting == "uniform":
        weights = np.ones((nq, n), dtype=np.float64)
    elif weighting == "inverse_distance":
        weights = 1.0 / (dist + eps)
    else:
        raise ValueError(f"Unknown weighting '{weighting}'.")

    # Exact-match handling: for rows with >=1 neighbour at distance <= tol,
    # average only the exact-match neighbours (uniform over them).
    exact_mask = dist <= exact_match_tol                 # [Nq, n]
    exact_row = exact_mask.any(axis=1)
    if exact_row.any():
        w_exact = exact_mask.astype(np.float64)
        weights = np.where(exact_row[:, None], w_exact, weights)

    wsum = weights.sum(axis=1, keepdims=True)
    wsum = np.where(wsum <= 0.0, 1.0, wsum)
    w_norm = weights / wsum                               # [Nq, n]
    pred = np.einsum("nk,nkm->nm", w_norm, neigh_u)

    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    if not want_debug:
        return pred

    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(
            (w_norm ** 2).sum(axis=1) > 0,
            1.0 / (w_norm ** 2).sum(axis=1),
            0.0,
        )
    debug = PredictionDebug(
        neighbor_indices=idx,
        neighbor_distances=dist,
        normalized_weights=w_norm,
        effective_neighbor_count=n_eff,
        exact_match_row=exact_row,
    )
    return pred, debug


# --------------------------------------------------------------------------- #
# Frozen final model
# --------------------------------------------------------------------------- #
class _ViewModel:
    """Preprocessor + neighbour index + training utilities for one view."""

    def __init__(self, preproc: FeaturePreprocessor, index: ViewNeighborIndex,
                 train_utilities: np.ndarray, train_record_ids: np.ndarray,
                 train_scaled: np.ndarray) -> None:
        self.preproc = preproc
        self.index = index
        self.train_utilities = train_utilities
        self.train_record_ids = train_record_ids
        self.train_scaled = train_scaled       # kept for exact reload / rebuild


class KnnUtilityPredictor:
    """Frozen three-view KNN utility predictor with a stable API."""

    def __init__(self, knn_config: C.KnnConfig, feature_columns: List[str],
                 metric_names: List[str]) -> None:
        self.config = knn_config
        self.feature_columns = list(feature_columns)
        self.metric_names = list(metric_names)
        self._views: Dict[str, _ViewModel] = {}

    # ---- construction ---------------------------------------------------- #
    def fit_view(self, view: str, x_train: np.ndarray, u_train: np.ndarray,
                 record_ids: np.ndarray) -> None:
        preproc = FeaturePreprocessor(self.config.scaler).fit(x_train)
        x_scaled = preproc.transform(x_train)
        index = ViewNeighborIndex(self.config.distance).fit(x_scaled)
        self._views[view] = _ViewModel(
            preproc=preproc, index=index,
            train_utilities=np.asarray(u_train, dtype=np.float64),
            train_record_ids=np.asarray(record_ids),
            train_scaled=x_scaled,
        )

    # ---- inference ------------------------------------------------------- #
    def predict_utility(
        self,
        meta_features: np.ndarray,
        view_type: str,
        return_neighbor_debug: bool = False,
    ) -> np.ndarray:
        if view_type not in self._views:
            raise ValueError(
                f"View '{view_type}' not fitted (have {sorted(self._views)})."
            )
        vm = self._views[view_type]
        x = np.atleast_2d(np.asarray(meta_features, dtype=np.float64))
        if x.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {x.shape[1]}."
            )
        x_scaled = vm.preproc.transform(x)
        nn = vm.index.query(x_scaled, max_neighbors=self.config.n_neighbors)
        out = predict_from_neighbors(
            nn.indices, nn.distances, vm.train_utilities,
            self.config.n_neighbors, self.config.neighbor_weighting,
            want_debug=return_neighbor_debug,
        )
        if return_neighbor_debug:
            pred, debug = out
            return pred, debug
        return out

    # ---- persistence ----------------------------------------------------- #
    def save(self, root: str | Path) -> Dict:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "config": self.config.to_dict(),
            "n_features": len(self.feature_columns),
            "n_metrics": len(self.metric_names),
            "views": {},
        }
        for view, vm in self._views.items():
            vdir = root / view
            vdir.mkdir(parents=True, exist_ok=True)
            vm.preproc.save(vdir / "preprocessor.pkl")
            np.save(vdir / "training_features.npy", vm.train_scaled)
            np.save(vdir / "training_utilities.npy", vm.train_utilities)
            np.save(vdir / "training_record_ids.npy", vm.train_record_ids)
            manifest["views"][view] = {
                "n_train": int(vm.train_scaled.shape[0]),
                "distance": self.config.distance,
            }
        return manifest

    @classmethod
    def load(cls, root: str | Path, feature_columns: List[str],
             metric_names: List[str], knn_config: C.KnnConfig
             ) -> "KnnUtilityPredictor":
        root = Path(root)
        obj = cls(knn_config, feature_columns, metric_names)
        for view in C.VIEWS:
            vdir = root / view
            if not vdir.exists():
                continue
            preproc = FeaturePreprocessor.load(vdir / "preprocessor.pkl")
            x_scaled = np.load(vdir / "training_features.npy")
            u_train = np.load(vdir / "training_utilities.npy")
            rec_ids = np.load(vdir / "training_record_ids.npy", allow_pickle=True)
            index = ViewNeighborIndex(knn_config.distance).fit(x_scaled)
            obj._views[view] = _ViewModel(
                preproc=preproc, index=index, train_utilities=u_train,
                train_record_ids=rec_ids, train_scaled=x_scaled,
            )
        return obj
