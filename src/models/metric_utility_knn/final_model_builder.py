"""
final_model_builder.py
=====================

Freeze the selected KNN configuration into a three-view final model trained on
all development records (splits 2-16), save every artefact with hashes and a
training manifest, then reload in a clean object and verify prediction equality
(plan §21).

Also provides :func:`oof_predictions_for_config`, which reproduces the selected
configuration's out-of-fold predictions across the 15 LOSO folds — used for the
global / view / family / subfamily reporting tables and the OOF parquet dumps.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .cross_validation import build_loso_folds
from .data_loader import UnifiedData
from .knn_predictor import KnnUtilityPredictor, predict_from_neighbors
from .neighbor_search import ViewNeighborIndex
from .preprocessing import FeaturePreprocessor


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dev_view_arrays(data: UnifiedData, view: str):
    m = (data.view == view) & data.split_mask(C.DEV_SPLITS)
    return data.X[m], data.Y[m], data.record_id[m], data.split_id[m], data.dataset_id[m]


def build_final_model(data: UnifiedData, knn_config: C.KnnConfig,
                      models_dir: Path) -> Dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    model = KnnUtilityPredictor(knn_config, data.feature_columns, data.metric_names)

    manifest_rows: List[dict] = []
    per_view_counts: Dict[str, int] = {}
    for view in C.VIEWS:
        X, Y, rec, spl, ds = _dev_view_arrays(data, view)
        model.fit_view(view, X, Y, rec)
        per_view_counts[view] = int(X.shape[0])
        for r, s, d in zip(rec, spl, ds):
            manifest_rows.append({"view": view, "record_id": r,
                                  "split_id": int(s), "dataset_id": d})

    manifest = model.save(models_dir)

    # training manifest (record-level provenance; contains only splits 2-16)
    man_df = pd.DataFrame(manifest_rows)
    man_df.to_parquet(models_dir / "training_manifest.parquet", index=False)

    # schemas
    (models_dir / "feature_schema.json").write_text(
        json.dumps({"feature_columns": data.feature_columns,
                    "n_features": len(data.feature_columns)}, indent=2))
    (models_dir / "utility_metric_schema.json").write_text(
        json.dumps({"target_columns": data.target_columns,
                    "metric_names": data.metric_names,
                    "n_metrics": len(data.metric_names)}, indent=2))

    # hashes of every saved artefact
    hashes = {}
    for p in sorted(models_dir.rglob("*")):
        if p.is_file():
            hashes[str(p.relative_to(models_dir)).replace("\\", "/")] = _sha256(p)

    train_splits = sorted(set(int(s) for s in man_df["split_id"]))
    metadata = {
        "config": knn_config.to_dict(),
        "dynamic_topk_heuristic": C.DYNAMIC_TOPK_HEURISTIC,
        "train_splits": train_splits,
        "locked_holdout_split": C.LOCKED_HOLDOUT_SPLIT,
        "split1_in_training": C.LOCKED_HOLDOUT_SPLIT in train_splits,
        "per_view_train_counts": per_view_counts,
        "n_features": len(data.feature_columns),
        "n_metrics": len(data.metric_names),
        "artifact_sha256": hashes,
        "view_manifest": manifest["views"],
    }
    (models_dir / "final_knn_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def reload_and_verify(data: UnifiedData, knn_config: C.KnnConfig,
                      models_dir: Path, n_sample: int = 64,
                      seed: int = 12345) -> Dict:
    """Reload the model in a fresh object and reproduce predictions exactly."""
    model = KnnUtilityPredictor(knn_config, data.feature_columns,
                                data.metric_names)
    for view in C.VIEWS:
        X, Y, rec, spl, ds = _dev_view_arrays(data, view)
        model.fit_view(view, X, Y, rec)

    reloaded = KnnUtilityPredictor.load(models_dir, data.feature_columns,
                                        data.metric_names, knn_config)

    rng = np.random.default_rng(seed)
    checks = {}
    max_abs_diff = 0.0
    for view in C.VIEWS:
        m = (data.view == view) & data.split_mask(C.DEV_SPLITS)
        Xv = data.X[m]
        idx = rng.choice(Xv.shape[0], size=min(n_sample, Xv.shape[0]),
                         replace=False)
        p1 = model.predict_utility(Xv[idx], view)
        p2 = reloaded.predict_utility(Xv[idx], view)
        d = float(np.max(np.abs(p1 - p2)))
        max_abs_diff = max(max_abs_diff, d)
        checks[view] = {"n_sample": int(len(idx)), "max_abs_diff": d,
                        "identical": bool(d <= 1e-12)}
    return {
        "reload_prediction_max_abs_diff": max_abs_diff,
        "reload_identical": bool(max_abs_diff <= 1e-12),
        "per_view": checks,
    }


def oof_predictions_for_config(data: UnifiedData, knn_config: C.KnnConfig
                               ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Concatenated out-of-fold predictions/targets/metadata for one config."""
    folds = build_loso_folds(data)
    preds, targets, meta = [], [], []
    for fold in folds:
        s = fold.validation_split
        for view in C.VIEWS:
            vmask = (data.view == view) & data.split_mask(C.DEV_SPLITS)
            tr = vmask & (data.split_id != s)
            va = vmask & (data.split_id == s)
            pp = FeaturePreprocessor(knn_config.scaler).fit(data.X[tr])
            xtr_s = pp.transform(data.X[tr])
            xva_s = pp.transform(data.X[va])
            index = ViewNeighborIndex(knn_config.distance).fit(xtr_s)
            nn = index.query(xva_s, max_neighbors=knn_config.n_neighbors)
            pred = predict_from_neighbors(
                nn.indices, nn.distances, data.Y[tr],
                knn_config.n_neighbors, knn_config.neighbor_weighting)
            preds.append(pred)
            targets.append(data.Y[va])
            meta.append(pd.DataFrame({
                "fold": s, "view": view,
                "record_id": data.record_id[va],
                "dataset_id": data.dataset_id[va],
                "split_id": data.split_id[va],
                "family": data.family[va],
                "subfamily": data.subfamily[va],
                "difficulty": data.difficulty[va],
            }))
    return (np.concatenate(preds, axis=0),
            np.concatenate(targets, axis=0),
            pd.concat(meta, ignore_index=True))
