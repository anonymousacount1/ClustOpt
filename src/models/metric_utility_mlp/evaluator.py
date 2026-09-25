"""
evaluator.py
============

Test-set evaluation for one fold:
  * run the (best) model over a test loader to obtain predictions,
  * compute the full metric suite,
  * compute grouped metrics (per family / view / difficulty / ...),
  * build a tidy ``test_predictions`` frame joined to row metadata,
  * compute the per-metric error table.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import EvaluationConfig
from .metrics import compute_all


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Return (y_true, y_pred, row_index) numpy arrays for a loader."""
    model.eval()
    ys, ps, idxs = [], [], []
    for x, y, ridx in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x)
        ps.append(pred.cpu().numpy())
        ys.append(y.numpy())
        idxs.append(ridx.numpy())
    return (
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(idxs),
    )


def build_predictions_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_index: np.ndarray,
    df: pd.DataFrame,
    target_names: List[str],
    id_columns: List[str],
) -> pd.DataFrame:
    """Wide predictions frame: id columns + true_<m> + pred_<m> per metric."""
    meta = df.iloc[row_index][id_columns].reset_index(drop=True)
    cols: Dict[str, np.ndarray] = {}
    for j, name in enumerate(target_names):
        bare = name
        cols[f"true_{bare}"] = y_true[:, j]
        cols[f"pred_{bare}"] = y_pred[:, j]
    pred_frame = pd.DataFrame(cols)
    return pd.concat([meta, pred_frame], axis=1)


def grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_index: np.ndarray,
    df: pd.DataFrame,
    target_names: List[str],
    group_columns: Dict[str, str],
    eval_cfg: EvaluationConfig,
    min_group_size: int = 20,
) -> Dict[str, pd.DataFrame]:
    """For each logical group column, a frame of key metrics per group value."""
    out: Dict[str, pd.DataFrame] = {}
    meta = df.iloc[row_index].reset_index(drop=True)
    key_metrics = [
        "mae", "rmse", "r2",
        "top1_accuracy", "top3_overlap", "top5_overlap", "top10_overlap",
        "spearman_mean", "kendall_mean", "ndcg@10",
    ]
    for logical, col in group_columns.items():
        if col not in meta.columns:
            continue
        rows = []
        for val, sub_idx in meta.groupby(col, dropna=False).groups.items():
            pos = np.asarray(sub_idx, dtype=np.int64)
            if pos.size < min_group_size:
                continue
            summary, _ = compute_all(
                y_true[pos], y_pred[pos], target_names,
                eval_cfg.topk_list, eval_cfg.ndcg_list, eval_cfg.ndcg_all,
            )
            row = {logical: val, "n_samples": int(pos.size)}
            for km in key_metrics:
                if km in summary:
                    row[km] = summary[km]
            rows.append(row)
        if rows:
            out[logical] = pd.DataFrame(rows).sort_values("n_samples", ascending=False, ignore_index=True)
    return out


def evaluate_fold(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    df: pd.DataFrame,
    target_names: List[str],
    id_columns: List[str],
    group_columns: Dict[str, str],
    eval_cfg: EvaluationConfig,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Returns (summary metrics, per_metric_errors, predictions_frame, grouped)."""
    y_true, y_pred, row_index = predict(model, test_loader, device)
    summary, per_metric = compute_all(
        y_true, y_pred, target_names,
        eval_cfg.topk_list, eval_cfg.ndcg_list, eval_cfg.ndcg_all,
    )
    pred_frame = build_predictions_frame(
        y_true, y_pred, row_index, df, target_names, id_columns
    )
    grouped = grouped_metrics(
        y_true, y_pred, row_index, df, target_names, group_columns, eval_cfg
    )
    return summary, per_metric, pred_frame, grouped
