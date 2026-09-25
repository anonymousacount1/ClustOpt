"""Cached frozen-KNN utility provider (Phase E2).

Loads the frozen Phase-E1 KNN utility predictor (three view-specific models) and
predicts a per-metric utility vector from a single dataset/view feature vector.
This is the KNN analogue of :class:`RegressorPredictor`: the *only* component
that differs from the corresponding regressor (MLP) methods is the utility
source -- everything downstream (ranking, Top-K, Dynamic-K, weighting) is
unchanged and reused.

Frozen artifact directory consumed (see Phase E1)::

    <knn_model_dir>/
        final_knn_metadata.json          # config + hashes + train splits
        feature_schema.json              # 250 ordered feature names
        utility_metric_schema.json       # 60 ordered "utility__<name>" targets
        x_only/ y_only/ xy_2d/           # per-view preprocessor + index arrays

The frozen model is never retrained here; it is loaded read-only. Providers are
cached process-wide keyed by the resolved model directory so each worker pays
the load cost once, and per-(dataset, view) predictions are memoised so all 12
KNN methods reuse a single query.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# models/ClustOpt/cluster_validity_indices/dynamic_metric_selection/<this>.py
# parents[4] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]

_VIEWS = ("x_only", "y_only", "xy_2d")


def _resolve_dir(model_dir: str | Path) -> Path:
    p = Path(model_dir)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.resolve()


class KNNUtilityProvider:
    """Loads the frozen Phase-E1 KNN models and predicts utility vectors."""

    def __init__(self, knn_model_dir: str | Path) -> None:
        self.model_dir = _resolve_dir(knn_model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"KNN model directory not found: {self.model_dir}")

        self._load_metadata()
        self._load_models()
        # Per-(dataset_dir, view_id) memoised predictions so all 12 methods on a
        # dataset/view reuse a single KNN query (plan §12).
        self._prediction_cache: Dict[Tuple[str, str], Tuple[Dict[str, float], dict]] = {}

    # ------------------------------------------------------------- artifacts
    def _load_metadata(self) -> None:
        meta_path = self.model_dir / "final_knn_metadata.json"
        with meta_path.open("r", encoding="utf-8") as fh:
            self.metadata: Dict = json.load(fh)
        cfg = self.metadata["config"]
        self.config_id: str = str(cfg["config_id"])
        self.n_neighbors: int = int(cfg["n_neighbors"])
        self.scaler: str = str(cfg["scaler"])
        self.distance: str = str(cfg["distance"])
        self.neighbor_weighting: str = str(cfg["neighbor_weighting"])
        self.train_splits: List[int] = [int(s) for s in self.metadata.get("train_splits", [])]
        self.heldout_split: int = int(self.metadata.get("locked_holdout_split", 1))
        self.dynamic_topk_heuristic: str = str(
            self.metadata.get("dynamic_topk_heuristic", ""))

        with (self.model_dir / "feature_schema.json").open("r", encoding="utf-8") as fh:
            fs = json.load(fh)
        self.feature_columns: List[str] = list(fs["feature_columns"])
        with (self.model_dir / "utility_metric_schema.json").open("r", encoding="utf-8") as fh:
            us = json.load(fh)
        self.target_columns: List[str] = list(us["target_columns"])
        self.metric_names: List[str] = list(us["metric_names"])
        self.target_prefix = "utility__"

        # A short, stable content hash of the frozen model (from recorded SHA-256s).
        self.model_hash: str = self._compute_model_hash()

    def _compute_model_hash(self) -> str:
        import hashlib
        hashes = self.metadata.get("artifact_sha256", {})
        if not hashes:
            return ""
        joined = "".join(f"{k}:{v}" for k, v in sorted(hashes.items()))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def _load_models(self) -> None:
        # Reuse the frozen Phase-E1 predictor exactly (no retraining, no rebuild).
        from models.metric_utility_knn.config import KnnConfig
        from models.metric_utility_knn.knn_predictor import KnnUtilityPredictor

        knn_config = KnnConfig(
            n_neighbors=self.n_neighbors, scaler=self.scaler,
            distance=self.distance, neighbor_weighting=self.neighbor_weighting,
        )
        self._predictor = KnnUtilityPredictor.load(
            self.model_dir, self.feature_columns, self.metric_names, knn_config)
        loaded = sorted(self._predictor._views.keys())
        missing = [v for v in _VIEWS if v not in loaded]
        if missing:
            raise FileNotFoundError(
                f"KNN model dir {self.model_dir} is missing view model(s): {missing}")
        logger.info("KNNUtilityProvider: loaded %s views %s from %s (config=%s)",
                    len(loaded), loaded, self.model_dir, self.config_id)

    # ---------------------------------------------------------------- predict
    def _to_ordered_array(self, feature_vector) -> np.ndarray:
        if isinstance(feature_vector, dict):
            missing = [c for c in self.feature_columns if c not in feature_vector]
            if missing:
                raise KeyError(
                    f"Feature vector is missing {len(missing)} expected column(s), "
                    f"e.g. {missing[:5]}")
            arr = np.asarray([feature_vector[c] for c in self.feature_columns],
                             dtype=np.float64)
        else:
            arr = np.asarray(feature_vector, dtype=np.float64).ravel()
            if arr.shape[0] != len(self.feature_columns):
                raise ValueError(
                    f"Feature vector has length {arr.shape[0]} but the KNN model "
                    f"expects {len(self.feature_columns)} features.")
        return arr

    def predict(
        self,
        feature_vector,
        view_id: str,
        *,
        cache_key: Optional[Tuple[str, str]] = None,
        want_neighbor_debug: bool = True,
    ) -> Tuple[Dict[str, float], dict]:
        """Return ``({metric_name: utility}, neighbor_debug)`` for one dataset/view.

        ``metric_name`` is the bare metric name (prefix stripped), matching the
        metric ``.name`` used downstream. When ``cache_key`` is supplied the
        prediction is memoised so all methods on the same dataset/view reuse it.
        """
        if view_id not in _VIEWS:
            raise ValueError(f"Unsupported view '{view_id}' (expected {_VIEWS}).")
        if cache_key is not None:
            hit = self._prediction_cache.get(cache_key)
            if hit is not None:
                return hit

        arr = self._to_ordered_array(feature_vector).reshape(1, -1)
        pred, debug = self._predictor.predict_utility(
            arr, view_id, return_neighbor_debug=True)
        vec = np.asarray(pred, dtype=np.float64).ravel()
        if vec.shape[0] != len(self.metric_names):
            raise ValueError(
                f"KNN predicted {vec.shape[0]} utilities but expected "
                f"{len(self.metric_names)}.")
        utilities = {name: float(score) for name, score in zip(self.metric_names, vec)}

        neighbor_debug = self._compact_neighbor_debug(view_id, debug) \
            if want_neighbor_debug else {}
        result = (utilities, neighbor_debug)
        if cache_key is not None:
            self._prediction_cache[cache_key] = result
        return result

    def _compact_neighbor_debug(self, view_id: str, debug) -> dict:
        """First-5 neighbour record ids / distances / normalized weights."""
        try:
            idx = np.asarray(debug.neighbor_indices).ravel()[:5]
            dist = np.asarray(debug.neighbor_distances).ravel()[:5]
            wts = np.asarray(debug.normalized_weights).ravel()[:5]
            rec_ids = self._predictor._views[view_id].train_record_ids
            neigh_ids = [str(rec_ids[i]) for i in idx]
            return {
                "neighbor_record_ids": neigh_ids,
                "neighbor_distances": [float(d) for d in dist],
                "neighbor_weights": [float(w) for w in wts],
                "effective_neighbor_count": float(
                    np.asarray(debug.effective_neighbor_count).ravel()[0]),
                "exact_match": bool(np.asarray(debug.exact_match_row).ravel()[0]),
            }
        except Exception as exc:  # debug must never break a run
            return {"neighbor_debug_error": str(exc)}

    def config_summary(self) -> Dict:
        return {
            "knn_config_id": self.config_id,
            "knn_n_neighbors": self.n_neighbors,
            "knn_scaler": self.scaler,
            "knn_distance_metric": self.distance,
            "knn_neighbor_weighting": self.neighbor_weighting,
            "knn_train_splits": list(self.train_splits),
            "knn_heldout_split": self.heldout_split,
            "knn_feature_count": len(self.feature_columns),
            "knn_utility_count": len(self.metric_names),
            "knn_model_hash": self.model_hash,
            "knn_model_path": str(self.model_dir),
        }


# --------------------------------------------------------------------------- cache
_PROVIDER_CACHE: Dict[str, KNNUtilityProvider] = {}


def get_knn_provider(knn_model_dir: str | Path) -> KNNUtilityProvider:
    """Return a process-cached :class:`KNNUtilityProvider` for ``knn_model_dir``."""
    key = str(_resolve_dir(knn_model_dir))
    provider = _PROVIDER_CACHE.get(key)
    if provider is None:
        provider = KNNUtilityProvider(knn_model_dir)
        _PROVIDER_CACHE[key] = provider
    return provider
