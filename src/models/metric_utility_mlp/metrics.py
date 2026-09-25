"""
metrics.py
==========

Evaluation metrics for the metric-utility regressor, treating the model as a
regression model, a ranking model, and a top-metric selector.

All functions take ``y_true`` / ``y_pred`` arrays of shape ``[N, M]`` (N samples,
M metrics) with values in ``[0, 1]``.

  * Regression : MSE, RMSE, MAE, R^2 (global + per-metric MAE/RMSE/R^2).
  * Top-1      : accuracy, mean true-rank distance of predicted top, utility err.
  * Top-K      : overlap, ordered accuracy, utility MAE/RMSE (K = 3, 5, 10).
  * Ranking    : Spearman, Kendall, pairwise accuracy, rank displacement.
  * NDCG       : @3, @5, @10, @All (true utility = relevance, linear gain).

The heavy ranking pieces are vectorised; the pairwise / Kendall pass is chunked
over rows to bound memory.  :func:`compute_core_val_metrics` is a cheap subset
used for per-epoch validation logging.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _desc_rank_positions(vals: np.ndarray) -> np.ndarray:
    """Rank position per metric (0 = best/highest), shape preserved [N, M].

    Ties are broken by ``argsort`` order (stable), matching a deterministic
    permutation ranking.
    """
    order = np.argsort(-vals, axis=1, kind="stable")
    pos = np.empty_like(order)
    cols = np.broadcast_to(np.arange(vals.shape[1]), order.shape)
    np.put_along_axis(pos, order, cols, axis=1)
    return pos


def _dcg(relevances: np.ndarray) -> np.ndarray:
    """Linear-gain DCG along axis=1. ``relevances`` shape [N, k]."""
    discounts = np.log2(np.arange(2, relevances.shape[1] + 2))
    return np.sum(relevances / discounts, axis=1)


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]
) -> Tuple[Dict[str, float], pd.DataFrame]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    mse = float(sq_err.mean())
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(mse))

    # Global R^2 over the flattened arrays.
    ss_res = float(sq_err.sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0

    # Per-metric
    per_mae = abs_err.mean(axis=0)
    per_rmse = np.sqrt(sq_err.mean(axis=0))
    col_mean = y_true.mean(axis=0, keepdims=True)
    ss_res_c = sq_err.sum(axis=0)
    ss_tot_c = ((y_true - col_mean) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_r2 = np.where(ss_tot_c > EPS, 1.0 - ss_res_c / ss_tot_c, 0.0)

    per_metric = pd.DataFrame(
        {
            "metric": target_names,
            "mae": per_mae,
            "rmse": per_rmse,
            "r2": per_r2,
            "true_mean": y_true.mean(axis=0),
            "pred_mean": y_pred.mean(axis=0),
        }
    ).sort_values("mae", ascending=False, ignore_index=True)

    summary = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "macro_mae": float(per_mae.mean()),
        "macro_rmse": float(per_rmse.mean()),
        "macro_r2": float(np.mean(per_r2)),
    }
    return summary, per_metric


# --------------------------------------------------------------------------- #
# Top-1
# --------------------------------------------------------------------------- #
def top1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    pred_top = np.argmax(y_pred, axis=1)
    true_top = np.argmax(y_true, axis=1)
    acc = float(np.mean(pred_top == true_top))

    true_pos = _desc_rank_positions(y_true)
    rows = np.arange(y_true.shape[0])
    # True-rank position of the predicted-top metric (0 if it *is* the true top).
    rank_distance = true_pos[rows, pred_top]
    # Utility error: |true best utility - true utility of predicted top|.
    true_best_val = y_true[rows, true_top]
    pred_top_true_val = y_true[rows, pred_top]
    util_err = np.abs(true_best_val - pred_top_true_val)
    # Calibration error of predicted top: |pred - true| at predicted top.
    pred_top_pred_val = y_pred[rows, pred_top]
    calib_err = np.abs(pred_top_pred_val - pred_top_true_val)

    return {
        "top1_accuracy": acc,
        "top1_rank_distance_mean": float(rank_distance.mean()),
        "top1_utility_error_mean": float(util_err.mean()),
        "top1_pred_calibration_error_mean": float(calib_err.mean()),
    }


# --------------------------------------------------------------------------- #
# Top-K
# --------------------------------------------------------------------------- #
def topk_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> Dict[str, float]:
    n, m = y_true.shape
    k = min(k, m)
    true_order = np.argsort(-y_true, axis=1, kind="stable")[:, :k]
    pred_order = np.argsort(-y_pred, axis=1, kind="stable")[:, :k]

    overlaps = np.empty(n)
    ordered = np.empty(n)
    util_abs_err: List[float] = []
    util_sq_err: List[float] = []
    rows = np.arange(n)
    for i in range(n):
        ts = set(true_order[i].tolist())
        ps = set(pred_order[i].tolist())
        inter = ts & ps
        overlaps[i] = len(inter) / k
        ordered[i] = np.mean(true_order[i] == pred_order[i])
        union = ts | ps
        idx = np.fromiter(union, dtype=np.int64)
        e = np.abs(y_true[i, idx] - y_pred[i, idx])
        util_abs_err.extend(e.tolist())
        util_sq_err.extend((e ** 2).tolist())

    return {
        f"top{k}_overlap": float(overlaps.mean()),
        f"top{k}_ordered_accuracy": float(ordered.mean()),
        f"top{k}_utility_mae": float(np.mean(util_abs_err)) if util_abs_err else float("nan"),
        f"top{k}_utility_rmse": float(np.sqrt(np.mean(util_sq_err))) if util_sq_err else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Ranking (Spearman, Kendall, pairwise accuracy, rank displacement)
# --------------------------------------------------------------------------- #
def _spearman_rows(true_pos: np.ndarray, pred_pos: np.ndarray) -> np.ndarray:
    """Per-row Spearman = Pearson of descending rank positions (permutation ranks)."""
    m = true_pos.shape[1]
    a = true_pos - (m - 1) / 2.0
    b = pred_pos - (m - 1) / 2.0
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a ** 2, axis=1) * np.sum(b ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > EPS, num / den, 0.0)
    return out


def _pairwise_and_kendall(
    y_true: np.ndarray, y_pred: np.ndarray, chunk: int = 1024, tie_eps: float = EPS
) -> Tuple[float, float]:
    """Mean pairwise ranking accuracy and mean Kendall-like tau (ties ignored).

    Chunked over rows to bound the [chunk, M, M] memory footprint.
    """
    n, m = y_true.shape
    acc_sum = 0.0
    acc_rows = 0
    tau_sum = 0.0
    tau_rows = 0
    iu, ju = np.triu_indices(m, k=1)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        t = y_true[start:end]
        p = y_pred[start:end]
        td = t[:, iu] - t[:, ju]
        pd_ = p[:, iu] - p[:, ju]
        valid = (np.abs(td) > tie_eps) & (np.abs(pd_) > tie_eps)
        concordant = (np.sign(td) == np.sign(pd_)) & valid
        n_valid = valid.sum(axis=1)
        n_conc = concordant.sum(axis=1)
        has = n_valid > 0
        if has.any():
            row_acc = n_conc[has] / n_valid[has]
            acc_sum += float(row_acc.sum())
            acc_rows += int(has.sum())
            row_tau = (2.0 * n_conc[has] - n_valid[has]) / n_valid[has]
            tau_sum += float(row_tau.sum())
            tau_rows += int(has.sum())
    pairwise_acc = acc_sum / acc_rows if acc_rows else 0.0
    kendall = tau_sum / tau_rows if tau_rows else 0.0
    return pairwise_acc, kendall


def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_pos = _desc_rank_positions(y_true)
    pred_pos = _desc_rank_positions(y_pred)

    spearman = _spearman_rows(true_pos, pred_pos)
    pairwise_acc, kendall = _pairwise_and_kendall(y_true, y_pred)

    disp = np.abs(true_pos - pred_pos)
    row_mean = disp.mean(axis=1)
    row_median = np.median(disp, axis=1)
    row_max = disp.max(axis=1)

    return {
        "spearman_mean": float(spearman.mean()),
        "spearman_std": float(spearman.std()),
        "kendall_mean": float(kendall),
        "pairwise_ranking_accuracy": float(pairwise_acc),
        "rank_displacement_mean": float(row_mean.mean()),
        "rank_displacement_median": float(np.mean(row_median)),
        "rank_displacement_max": float(np.mean(row_max)),
    }


# --------------------------------------------------------------------------- #
# NDCG
# --------------------------------------------------------------------------- #
def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    n, m = y_true.shape
    k = min(k, m)
    pred_order = np.argsort(-y_pred, axis=1, kind="stable")[:, :k]
    ideal_order = np.argsort(-y_true, axis=1, kind="stable")[:, :k]
    rows = np.arange(n)[:, None]
    dcg = _dcg(y_true[rows, pred_order])
    idcg = _dcg(y_true[rows, ideal_order])
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(idcg > EPS, dcg / idcg, 0.0)
    return float(vals.mean())


# --------------------------------------------------------------------------- #
# Aggregators
# --------------------------------------------------------------------------- #
def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    topk_list: List[int],
    ndcg_list: List[int],
    ndcg_all: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Full metric suite. Returns (flat scalar dict, per-metric error frame)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    summary: Dict[str, float] = {}
    reg, per_metric = regression_metrics(y_true, y_pred, target_names)
    summary.update(reg)
    summary.update(top1_metrics(y_true, y_pred))
    for k in topk_list:
        summary.update(topk_metrics(y_true, y_pred, k))
    summary.update(ranking_metrics(y_true, y_pred))
    for k in ndcg_list:
        summary[f"ndcg@{min(k, y_true.shape[1])}"] = ndcg_at_k(y_true, y_pred, k)
    if ndcg_all:
        summary["ndcg@all"] = ndcg_at_k(y_true, y_pred, y_true.shape[1])
    summary["n_samples"] = int(y_true.shape[0])
    return summary, per_metric


def compute_core_val_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Cheap metrics for per-epoch validation logging (no O(M^2) passes)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.abs(y_pred - y_true).mean())
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    top1 = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))
    ndcg10 = ndcg_at_k(y_true, y_pred, 10)
    return {"mae": mae, "rmse": rmse, "top1_accuracy": top1, "ndcg@10": ndcg10}
