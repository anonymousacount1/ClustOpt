"""Cached MLP utility-regressor predictor.

Loads the artifacts produced by :mod:`models.metric_utility_mlp.cross_validation`
from a trained run directory and predicts a per-metric utility vector from a
single dataset/view feature vector.

Run-directory layout consumed (see ``cross_validation.py``)::

    <model_run_dir>/
        config.json                      # serialised RunConfig (model arch)
        artifacts/
            feature_columns.json         # 250 ordered feature names
            target_columns.json          # 60 ordered "utility__<name>" targets
            head_mapping.json            # {head_order, heads, ...}
        folds/
            fold_0/best_model.pt         # {"model_state": ..., "epoch": ...}
            fold_0/x_scaler.pkl          # pickled StandardScaler
            fold_1/...

Two checkpoint policies are supported:

* ``fold_ensemble``    - average the predictions of every available fold model
                         (each with its own scaler).
* ``best_single_fold`` - use a single fold (``fold_index``, default 0).

Predictors are cached process-wide keyed by
``(resolved_run_dir, checkpoint_policy, fold_index)`` so the future experiment
runner pays the (heavy) load cost once.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .naming import strip_prefix

logger = logging.getLogger(__name__)


# Repo root: this file lives at
# models/ClustOpt/cluster_validity_indices/dynamic_metric_selection/<this>.py
# parents[0]=dynamic_metric_selection, [1]=cluster_validity_indices,
# [2]=ClustOpt, [3]=models, [4]=repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_run_dir(model_run_dir: str | Path) -> Path:
    """Resolve ``model_run_dir`` against the repo root when it is relative."""
    p = Path(model_run_dir)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.resolve()


class RegressorPredictor:
    """Loads MLP fold artifacts and predicts a metric-utility vector."""

    def __init__(
        self,
        model_run_dir: str | Path,
        checkpoint_policy: str = "fold_ensemble",
        fold_index: int = 0,
    ) -> None:
        self.run_dir = _resolve_run_dir(model_run_dir)
        self.checkpoint_policy = str(checkpoint_policy)
        self.fold_index = int(fold_index)

        if not self.run_dir.is_dir():
            raise FileNotFoundError(
                f"Regressor run directory not found: {self.run_dir}"
            )

        self._load_artifacts()
        self._load_folds()

    # ----------------------------------------------------------- artifacts

    def _artifact_path(self, *candidates: str) -> Path:
        """Return the first existing artifact path among ``candidates``.

        Artifacts are written both to ``artifacts/`` (global) and to each
        ``folds/fold_i/`` directory; we prefer the global copy but fall back to
        fold 0 so the predictor is robust to partial runs.
        """
        for rel in candidates:
            p = self.run_dir / rel
            if p.is_file():
                return p
        raise FileNotFoundError(
            f"None of the expected artifact files exist under {self.run_dir}: "
            f"{candidates}"
        )

    def _load_artifacts(self) -> None:
        cfg_path = self._artifact_path("config.json")
        with cfg_path.open("r", encoding="utf-8") as fh:
            self.run_config = json.load(fh)
        self.model_config: Dict = dict(self.run_config.get("model", {}))

        feat_path = self._artifact_path(
            "artifacts/feature_columns.json", "folds/fold_0/feature_columns.json"
        )
        tgt_path = self._artifact_path(
            "artifacts/target_columns.json", "folds/fold_0/target_columns.json"
        )
        head_path = self._artifact_path(
            "artifacts/head_mapping.json", "folds/fold_0/head_mapping.json"
        )

        with feat_path.open("r", encoding="utf-8") as fh:
            self.feature_columns: List[str] = list(json.load(fh))
        with tgt_path.open("r", encoding="utf-8") as fh:
            self.target_columns: List[str] = list(json.load(fh))
        with head_path.open("r", encoding="utf-8") as fh:
            head_info = json.load(fh)
        # head_mapping.json stores {head_name: [target_index, ...]}.
        self.head_mapping: Dict[str, List[int]] = {
            str(h): [int(i) for i in idxs]
            for h, idxs in head_info.get("heads", {}).items()
        }

        target_prefix = str(
            self.run_config.get("data", {}).get("target_prefix", "utility__")
        )
        self.target_prefix = target_prefix
        # Bare metric names (== metric ``.name``) in canonical output order.
        self.metric_names: List[str] = [
            strip_prefix(c, target_prefix) for c in self.target_columns
        ]

    # --------------------------------------------------------------- folds

    def _discover_fold_dirs(self) -> List[Path]:
        folds_root = self.run_dir / "folds"
        if not folds_root.is_dir():
            raise FileNotFoundError(f"No 'folds' directory under {self.run_dir}")
        fold_dirs = []
        for child in sorted(folds_root.iterdir()):
            if child.is_dir() and child.name.startswith("fold_"):
                if (child / "best_model.pt").is_file():
                    fold_dirs.append(child)
        if not fold_dirs:
            raise FileNotFoundError(
                f"No 'fold_*/best_model.pt' checkpoints under {folds_root}"
            )
        return fold_dirs

    def _load_folds(self) -> None:
        all_folds = self._discover_fold_dirs()

        if self.checkpoint_policy == "fold_ensemble":
            selected = all_folds
        elif self.checkpoint_policy == "best_single_fold":
            wanted = f"fold_{self.fold_index}"
            selected = [d for d in all_folds if d.name == wanted]
            if not selected:
                raise FileNotFoundError(
                    f"Requested {wanted} not found under {self.run_dir / 'folds'}. "
                    f"Available: {[d.name for d in all_folds]}"
                )
        else:
            raise ValueError(
                f"Unsupported checkpoint_policy '{self.checkpoint_policy}'. "
                f"Options: 'fold_ensemble', 'best_single_fold'."
            )

        self._models: List[Tuple] = []  # (torch_model, scaler)
        for fold_dir in selected:
            model = self._build_and_load_model(fold_dir / "best_model.pt")
            scaler = self._load_scaler(fold_dir / "x_scaler.pkl")
            self._models.append((model, scaler))
        self._fold_names = [d.name for d in selected]
        logger.info(
            "RegressorPredictor: loaded %d fold(s) %s from %s (policy=%s)",
            len(self._models), self._fold_names, self.run_dir,
            self.checkpoint_policy,
        )

    def _build_and_load_model(self, ckpt_path: Path):
        # Imports are local so importing this module never hard-requires torch
        # for callers that only use the oracle path.
        import torch  # noqa: WPS433 (local import is intentional)

        from models.metric_utility_mlp.config import ModelConfig
        from models.metric_utility_mlp.model import build_model

        model_cfg = ModelConfig(**self.model_config)
        head_mapping = (
            self.head_mapping if model_cfg.type == "multi_head_mlp" else None
        )
        model = build_model(model_cfg, head_mapping)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
        model.eval()
        return model

    @staticmethod
    def _load_scaler(scaler_path: Path):
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Fold scaler not found: {scaler_path}")
        with scaler_path.open("rb") as fh:
            return pickle.load(fh)

    # ----------------------------------------------------------- inference

    def _prepare_features(self, feature_vector) -> np.ndarray:
        """Coerce ``feature_vector`` into a 1-D float array of the right length.

        Accepts a 1-D array/list (assumed already ordered like
        ``feature_columns``) or a mapping ``{feature_name: value}``.
        """
        if isinstance(feature_vector, dict):
            missing = [c for c in self.feature_columns if c not in feature_vector]
            if missing:
                raise KeyError(
                    f"Feature vector is missing {len(missing)} expected "
                    f"column(s), e.g. {missing[:5]}"
                )
            arr = np.asarray(
                [feature_vector[c] for c in self.feature_columns],
                dtype=np.float64,
            )
        else:
            arr = np.asarray(feature_vector, dtype=np.float64).ravel()
            if arr.shape[0] != len(self.feature_columns):
                raise ValueError(
                    f"Feature vector has length {arr.shape[0]} but the model "
                    f"expects {len(self.feature_columns)} features."
                )
        # Match training-time cleaning (NaN/Inf -> 0) before scaling.
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, feature_vector) -> Dict[str, float]:
        """Predict ``{metric_name: utility_score}`` for one dataset/view.

        ``metric_name`` is the bare metric name (prefix stripped), matching the
        metric ``.name`` attribute used downstream.
        """
        import torch  # local import, see above

        x = self._prepare_features(feature_vector).reshape(1, -1)

        preds = []
        with torch.no_grad():
            for model, scaler in self._models:
                x_scaled = scaler.transform(x)
                x_scaled = np.nan_to_num(
                    x_scaled, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
                tensor = torch.from_numpy(x_scaled)
                out = model(tensor).cpu().numpy().ravel()
                preds.append(out)

        mean_pred = np.mean(np.stack(preds, axis=0), axis=0)
        return {name: float(score) for name, score in zip(self.metric_names, mean_pred)}


# --------------------------------------------------------------------------- cache

_PREDICTOR_CACHE: Dict[Tuple[str, str, int], RegressorPredictor] = {}


def get_regressor_predictor(
    model_run_dir: str | Path,
    checkpoint_policy: str = "fold_ensemble",
    fold_index: int = 0,
) -> RegressorPredictor:
    """Return a (process-cached) :class:`RegressorPredictor`.

    Cache key: ``(resolved_run_dir, checkpoint_policy, fold_index)``. The
    ``fold_index`` only affects ``best_single_fold``; for ``fold_ensemble`` it
    is forced to ``-1`` in the key so all ensemble requests share one entry.
    """
    resolved = str(_resolve_run_dir(model_run_dir))
    key_fold = fold_index if checkpoint_policy == "best_single_fold" else -1
    key = (resolved, str(checkpoint_policy), int(key_fold))
    predictor = _PREDICTOR_CACHE.get(key)
    if predictor is None:
        predictor = RegressorPredictor(model_run_dir, checkpoint_policy, fold_index)
        _PREDICTOR_CACHE[key] = predictor
    return predictor
