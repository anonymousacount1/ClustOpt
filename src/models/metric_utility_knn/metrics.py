"""
metrics.py
==========

Utility-reconstruction metrics for the KNN predictor.

The *core* value / ranking / top-K / NDCG metrics are computed by reusing the
existing MLP evaluation code (``models.metric_utility_mlp.metrics.compute_all``)
verbatim, so the KNN and MLP numbers are directly comparable and no metric is
re-defined under an existing name (plan §14).

On top of the core suite this module adds, all with new names:

  * full-vector similarity     (cosine / pearson, mean & std)
  * absolute-error distribution (median / p90 / p95 / max)
  * top-set similarity          (jaccard / precision / recall for K in 1,3,5,10)
  * Dynamic Top-K fidelity      (via the frozen ClustOpt heuristic)
  * per-metric diagnostics      (mae/rmse/r2/spearman/bias per utility dimension)
  * kendall_std                 (per-row std to complement the MLP's mean)

``best_val_loss`` is defined as the out-of-fold MSE (KNN has no epoch loop).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models.metric_utility_mlp.metrics import compute_all, ndcg_at_k
from models.ClustOpt.dynamic_topk_selection.heuristics.selected_dynamic_topk import (
    select_batch,
)

from . import config as C

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Full-vector similarity
# --------------------------------------------------------------------------- #
def _rowwise_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    num = np.sum(y_true * y_pred, axis=1)
    den = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > EPS, num / den, 0.0)


def _rowwise_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    a = y_true - y_true.mean(axis=1, keepdims=True)
    b = y_pred - y_pred.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a ** 2, axis=1) * np.sum(b ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > EPS, num / den, 0.0)


def _rowwise_kendall(y_true: np.ndarray, y_pred: np.ndarray,
                     chunk: int = 1024, tie_eps: float = EPS) -> np.ndarray:
    """Per-row Kendall-like tau (ties ignored) — mean matches the MLP kendall."""
    n, m = y_true.shape
    iu, ju = np.triu_indices(m, k=1)
    out = np.zeros(n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        t = y_true[start:end]
        p = y_pred[start:end]
        td = t[:, iu] - t[:, ju]
        pd_ = p[:, iu] - p[:, ju]
        valid = (np.abs(td) > tie_eps) & (np.abs(pd_) > tie_eps)
        conc = (np.sign(td) == np.sign(pd_)) & valid
        n_valid = valid.sum(axis=1)
        n_conc = conc.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_tau = np.where(n_valid > 0,
                               (2.0 * n_conc - n_valid) / np.maximum(n_valid, 1),
                               0.0)
        out[start:end] = row_tau
    return out


# --------------------------------------------------------------------------- #
# Top-set similarity
# --------------------------------------------------------------------------- #
def topset_similarity(y_true: np.ndarray, y_pred: np.ndarray,
                      ks=(1, 3, 5, 10)) -> Dict[str, float]:
    n, m = y_true.shape
    out: Dict[str, float] = {}
    true_full = np.argsort(-y_true, axis=1, kind="stable")
    pred_full = np.argsort(-y_pred, axis=1, kind="stable")
    for k in ks:
        kk = min(k, m)
        to = true_full[:, :kk]
        po = pred_full[:, :kk]
        jac = np.empty(n)
        prec = np.empty(n)
        rec = np.empty(n)
        for i in range(n):
            ts = set(to[i].tolist())
            ps = set(po[i].tolist())
            inter = len(ts & ps)
            union = len(ts | ps)
            jac[i] = inter / union if union else 0.0
            prec[i] = inter / len(ps) if ps else 0.0
            rec[i] = inter / len(ts) if ts else 0.0
        out[f"top{k}_jaccard"] = float(jac.mean())
        out[f"top{k}_precision"] = float(prec.mean())
        out[f"top{k}_recall"] = float(rec.mean())
    return out


# --------------------------------------------------------------------------- #
# Dynamic Top-K fidelity
# --------------------------------------------------------------------------- #
def dynamic_k_metrics(y_true: np.ndarray, y_pred: np.ndarray
                      ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Apply the frozen ClustOpt heuristic to true and predicted utilities."""
    k_true, _, _ = select_batch(y_true)
    k_pred, _, _ = select_batch(y_pred)
    k_true = k_true.astype(int)
    k_pred = k_pred.astype(int)
    diff = k_pred - k_true
    out = {
        "dynamic_k_exact_accuracy": float(np.mean(k_pred == k_true)),
        "dynamic_k_within_1_accuracy": float(np.mean(np.abs(diff) <= 1)),
        "dynamic_k_mae": float(np.mean(np.abs(diff))),
        "dynamic_k_rmse": float(np.sqrt(np.mean(diff ** 2))),
        "dynamic_k_signed_error_mean": float(np.mean(diff)),
        "dynamic_k_underselection_rate": float(np.mean(k_pred < k_true)),
        "dynamic_k_overselection_rate": float(np.mean(k_pred > k_true)),
        "dynamic_k_true_mean": float(np.mean(k_true)),
        "dynamic_k_pred_mean": float(np.mean(k_pred)),
    }
    return out, k_true, k_pred


def dynamic_k_confusion(k_true: np.ndarray, k_pred: np.ndarray,
                        k_min: int = 1, k_max: int = 10) -> pd.DataFrame:
    labels = list(range(k_min, k_max + 1))
    idx = {k: i for i, k in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(k_true, k_pred):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return pd.DataFrame(mat, index=[f"true_{k}" for k in labels],
                        columns=[f"pred_{k}" for k in labels])


# --------------------------------------------------------------------------- #
# Per-metric diagnostics (plan §17)
# --------------------------------------------------------------------------- #
def per_metric_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                           metric_names: List[str]) -> pd.DataFrame:
    err = y_pred - y_true
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    col_mean = y_true.mean(axis=0, keepdims=True)
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((y_true - col_mean) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > EPS, 1.0 - ss_res / ss_tot, 0.0)

    # per-metric spearman (rank correlation across records)
    tr = np.argsort(np.argsort(y_true, axis=0), axis=0).astype(np.float64)
    pr = np.argsort(np.argsort(y_pred, axis=0), axis=0).astype(np.float64)
    tr -= tr.mean(axis=0, keepdims=True)
    pr -= pr.mean(axis=0, keepdims=True)
    num = (tr * pr).sum(axis=0)
    den = np.sqrt((tr ** 2).sum(axis=0) * (pr ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        spearman = np.where(den > EPS, num / den, 0.0)

    return pd.DataFrame({
        "metric_name": metric_names,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
        "mean_true_utility": y_true.mean(axis=0),
        "mean_predicted_utility": y_pred.mean(axis=0),
        "prediction_bias": y_pred.mean(axis=0) - y_true.mean(axis=0),
        "std_true_utility": y_true.std(axis=0),
        "std_predicted_utility": y_pred.std(axis=0),
    })


# --------------------------------------------------------------------------- #
# Full suite
# --------------------------------------------------------------------------- #
def compute_full_suite(y_true: np.ndarray, y_pred: np.ndarray,
                       metric_names: List[str]) -> Dict[str, float]:
    """All scalar utility-reconstruction metrics for one (y_true, y_pred) set."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    core, _ = compute_all(
        y_true, y_pred, metric_names,
        topk_list=C.TOPK_LIST, ndcg_list=[3, 5, 10], ndcg_all=True,
    )
    out: Dict[str, float] = dict(core)
    out["best_val_loss"] = float(core["mse"])
    out["ndcg@1"] = float(ndcg_at_k(y_true, y_pred, 1))

    # full-vector similarity
    cos = _rowwise_cosine(y_true, y_pred)
    pear = _rowwise_pearson(y_true, y_pred)
    ken = _rowwise_kendall(y_true, y_pred)
    out["cosine_similarity_mean"] = float(cos.mean())
    out["cosine_similarity_std"] = float(cos.std())
    out["pearson_mean"] = float(pear.mean())
    out["pearson_std"] = float(pear.std())
    out["kendall_std"] = float(ken.std())

    # absolute-error distribution
    abs_err = np.abs(y_pred - y_true).ravel()
    out["median_absolute_error"] = float(np.median(abs_err))
    out["p90_absolute_error"] = float(np.percentile(abs_err, 90))
    out["p95_absolute_error"] = float(np.percentile(abs_err, 95))
    out["max_absolute_error"] = float(abs_err.max())

    # top-set similarity
    out.update(topset_similarity(y_true, y_pred))

    # dynamic-K fidelity
    dk, _, _ = dynamic_k_metrics(y_true, y_pred)
    out.update(dk)

    return out
