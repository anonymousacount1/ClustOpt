"""AutoClust model training: algorithm selector + ARI-predictor MLP.

Mirrors Experiments/RelatedWork/AutoClust.py:
* algorithm selector: AutoClust finds the most-similar dataset via a KDTree over
  MeanShift-landmarking meta-features (euclidean, NaN->0) and takes that
  dataset's best-ARI algorithm. We materialise this as a k=1 KNeighbors
  classifier over the same landmarking features (a KDTree under the hood), which
  reproduces the single-nearest-neighbour lookup and yields a reportable CV
  accuracy. No scaling (AutoClust's KDTree is unscaled); NaN->0 imputation.
* ARI predictor: MLPRegressor(hidden_layer_sizes=(60, 30, 10), activation='relu')
  exactly as AutoClust. Deviation (documented): we add a train-fitted
  StandardScaler before the MLP because the master stores RAW CVI values across
  very different magnitudes (CH ~1e3 vs silhouette in [-1,1]); AutoClust's CVI
  values were already on a normalised loss scale. NaN CVIs (degenerate
  single-cluster partitions) are imputed with the train median.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MLP_HIDDEN = (60, 30, 10)


# --- algorithm selector ----------------------------------------------------

def train_algorithm_selector(
    train_metafeatures: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
) -> Tuple[Pipeline, dict]:
    mf = train_metafeatures.set_index("record_id")
    y_df = labels.set_index("record_id")["best_algorithm"]
    rid = [r for r in y_df.index if r in mf.index]
    X = mf.reindex(rid)[feature_cols].to_numpy(dtype=float)
    y = y_df.reindex(rid).to_numpy()

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("knn", KNeighborsClassifier(n_neighbors=1)),  # AutoClust k=1 NN lookup
    ])
    pipe.fit(X, y)

    # internal CV (train-only); k=1 NN cross-val accuracy
    classes, counts = np.unique(y, return_counts=True)
    cv_acc = cv_bal = None
    cv_note = ""
    min_class = int(counts.min())
    if len(classes) >= 2 and min_class >= 2:
        n_splits = int(min(5, min_class))
        try:
            y_pred = cross_val_predict(
                Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                          ("knn", KNeighborsClassifier(n_neighbors=1))]),
                X, y, cv=n_splits, n_jobs=-1)
            cv_acc = float(accuracy_score(y, y_pred))
            cv_bal = float(balanced_accuracy_score(y, y_pred))
            cv_note = f"StratifiedKFold(n_splits={n_splits}) cross_val_predict, k=1 NN"
        except Exception as ex:  # noqa: BLE001
            cv_note = f"cv skipped: {ex}"
    else:
        cv_note = "cv skipped (insufficient per-class samples)"

    meta = {
        "trained": True,
        "model_type": "KNeighborsClassifier(n_neighbors=1) [AutoClust KDTree NN lookup]",
        "hyperparameters": {"n_neighbors": 1, "metric": "minkowski(p=2)/euclidean"},
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "target_distribution": {str(c): int(n) for c, n in zip(classes, counts)},
        "n_classes": int(len(classes)),
        "train_record_count": int(len(y)),
        "random_seed": seed,
        "preprocessing": {"imputer": "constant(0.0) [matches AutoClust NaN->0]", "scaler": None},
        "cv_accuracy": cv_acc,
        "cv_balanced_accuracy": cv_bal,
        "cv_note": cv_note,
        "train_record_ids": rid,
        "train_split_ids": sorted(int(s) for s in labels["split_id"].unique()),
    }
    return pipe, meta


# --- ARI predictor ---------------------------------------------------------

def train_ari_predictor(
    reg_df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
) -> Tuple[Pipeline, dict]:
    X = reg_df[feature_cols].to_numpy(dtype=float)
    y = reg_df["ari"].to_numpy(dtype=float)
    groups = reg_df["record_id"].to_numpy()

    def _make() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=MLP_HIDDEN, activation="relu",
                                 random_state=seed, max_iter=300, early_stopping=True,
                                 n_iter_no_change=10, validation_fraction=0.1)),
        ])

    # grouped 80/20 validation (no record's configs split across train/val)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(gss.split(X, y, groups))
    val_pipe = _make()
    val_pipe.fit(X[tr_idx], y[tr_idx])
    y_va_pred = val_pipe.predict(X[va_idx])
    y_va = y[va_idx]
    mae = float(mean_absolute_error(y_va, y_va_pred))
    rmse = float(np.sqrt(mean_squared_error(y_va, y_va_pred)))
    r2 = float(r2_score(y_va, y_va_pred))
    sp, _ = spearmanr(y_va, y_va_pred)
    spearman = float(sp) if sp == sp else float("nan")
    final_loss = float(val_pipe.named_steps["mlp"].loss_)

    # final model fit on ALL train rows
    final_pipe = _make()
    final_pipe.fit(X, y)

    meta = {
        "trained": True,
        "model_type": "MLPRegressor",
        "architecture": {"hidden_layer_sizes": list(MLP_HIDDEN), "activation": "relu",
                         "note": "matches AutoClust MLPRegressor((60,30,10), relu)"},
        "hyperparameters": {"max_iter": 300, "early_stopping": True, "n_iter_no_change": 10,
                            "validation_fraction": 0.1, "random_state": seed},
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "target_column": "ari",
        "train_row_count": int(len(y)),
        "random_seed": seed,
        "preprocessing": {
            "imputer": "median (train-fit) for NaN CVIs from degenerate k=1 partitions",
            "scaler": "StandardScaler (train-fit) [documented deviation: master stores RAW CVIs]",
        },
        "training_loss_final_iter": final_loss,
        "validation_scheme": "GroupShuffleSplit by record_id, test_size=0.2",
        "validation_scores": {
            "n_val_rows": int(len(va_idx)),
            "mae": mae, "rmse": rmse, "r2": r2,
            "spearman_pred_vs_true": spearman,
        },
    }
    return final_pipe, meta
