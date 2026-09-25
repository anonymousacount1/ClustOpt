"""
holdout_evaluation.py
====================

Locked split-1 evaluation (plan §22).  The frozen KNN (trained on splits 2-16)
predicts split-1 utilities exactly once; no retuning happens after this.

Also builds the direct KNN-vs-MLP comparison against the existing split-1 MLP
holdout reference (``overall_results.csv``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import UnifiedData
from .knn_predictor import KnnUtilityPredictor
from .metrics import (compute_full_suite, dynamic_k_metrics,
                      per_metric_diagnostics)

# Metrics where a smaller value is better (everything else: larger is better).
LOWER_BETTER = {
    "best_val_loss", "mse", "rmse", "mae", "macro_mae", "macro_rmse",
    "top1_rank_distance_mean", "top1_utility_error_mean",
    "top1_pred_calibration_error_mean",
    "top3_utility_mae", "top3_utility_rmse", "top5_utility_mae",
    "top5_utility_rmse", "top10_utility_mae", "top10_utility_rmse",
    "spearman_std", "kendall_std", "rank_displacement_mean",
    "rank_displacement_median", "rank_displacement_max",
    "median_absolute_error", "p90_absolute_error", "p95_absolute_error",
    "max_absolute_error", "dynamic_k_mae", "dynamic_k_rmse",
    "dynamic_k_underselection_rate", "dynamic_k_overselection_rate",
}
NOT_COMPARED = {"n_samples"}


def _split1_view_arrays(data: UnifiedData, view: str):
    m = (data.view == view) & (data.split_id == C.LOCKED_HOLDOUT_SPLIT)
    return m


def evaluate_split1(data: UnifiedData, model: KnnUtilityPredictor
                    ) -> Dict:
    preds, targets, meta = [], [], []
    per_view: Dict[str, Dict] = {}
    for view in C.VIEWS:
        m = _split1_view_arrays(data, view)
        Xv, Yv = data.X[m], data.Y[m]
        pv = model.predict_utility(Xv, view)
        preds.append(pv)
        targets.append(Yv)
        meta.append(pd.DataFrame({
            "view": view, "record_id": data.record_id[m],
            "dataset_id": data.dataset_id[m], "split_id": data.split_id[m],
            "family": data.family[m], "subfamily": data.subfamily[m],
            "difficulty": data.difficulty[m],
        }))
        per_view[view] = compute_full_suite(Yv, pv, data.metric_names)

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    meta_df = pd.concat(meta, ignore_index=True)

    suite = compute_full_suite(y_true, y_pred, data.metric_names)
    dk, k_true, k_pred = dynamic_k_metrics(y_true, y_pred)
    per_metric = per_metric_diagnostics(y_true, y_pred, data.metric_names)

    return {
        "y_pred": y_pred, "y_true": y_true, "meta": meta_df,
        "suite": suite, "per_view": per_view, "per_metric": per_metric,
        "k_true": k_true, "k_pred": k_pred,
    }


def load_mlp_reference(mlp_dir: Path = C.MLP_REFERENCE_DIR) -> Dict[str, float]:
    df = pd.read_csv(mlp_dir / "overall_results.csv")
    return {str(r["metric"]): float(r["mean"]) for _, r in df.iterrows()}


def mlp_comparison_table(knn_suite: Dict[str, float],
                         mlp_ref: Dict[str, float]) -> pd.DataFrame:
    rows: List[dict] = []
    for metric, mlp_val in mlp_ref.items():
        if metric in NOT_COMPARED or metric not in knn_suite:
            continue
        knn_val = float(knn_suite[metric])
        delta = knn_val - mlp_val
        lower_better = metric in LOWER_BETTER
        if abs(delta) < 1e-12:
            winner = "tie"
        elif (delta < 0) == lower_better:
            winner = "KNN"
        else:
            winner = "MLP"
        rows.append({
            "metric": metric, "knn": knn_val, "mlp": mlp_val,
            "knn_minus_mlp": delta,
            "direction": "lower_better" if lower_better else "higher_better",
            "winner": winner,
        })
    return pd.DataFrame(rows)
