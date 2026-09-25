"""External (ground-truth based) clustering metrics for one prediction.

Computes ARI, NMI, AMI, V-measure, homogeneity, completeness, Fowlkes-Mallows,
purity, and (when noise is present) noise precision/recall/F1. All metrics use
the *full* point set; ``selected_k`` (computed in :mod:`k_metrics`) excludes the
noise label.

This is a standalone implementation for the experiment pipeline; it does not
touch the existing :class:`ExternalEvaluator` used by ClustOpt.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

NOISE_LABEL = -1


def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of points whose predicted cluster's majority true-label matches.

    Each predicted cluster is assigned its most common true label; purity is
    the total number of agreeing points divided by N.
    """
    if y_true.size == 0:
        return float("nan")
    total = 0
    for cluster in np.unique(y_pred):
        members = y_true[y_pred == cluster]
        if members.size == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        total += int(counts.max())
    return float(total) / float(y_true.size)


def _noise_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, noise_label: int
) -> Dict[str, Optional[float]]:
    pred_is_noise = y_pred == noise_label
    true_is_noise = y_true == noise_label

    predicted_noise_ratio = float(np.mean(pred_is_noise)) if y_pred.size else float("nan")
    true_noise_present = bool(np.any(true_is_noise))
    true_noise_ratio = float(np.mean(true_is_noise)) if y_true.size else float("nan")

    out: Dict[str, Optional[float]] = {
        "predicted_noise_ratio": predicted_noise_ratio,
        "true_noise_ratio": true_noise_ratio if true_noise_present else None,
        "noise_precision": None,
        "noise_recall": None,
        "noise_f1": None,
    }
    # Precision/recall/F1 only meaningful when true noise exists.
    if not true_noise_present:
        return out

    tp = int(np.sum(pred_is_noise & true_is_noise))
    fp = int(np.sum(pred_is_noise & ~true_is_noise))
    fn = int(np.sum(~pred_is_noise & true_is_noise))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    out["noise_precision"] = float(precision)
    out["noise_recall"] = float(recall)
    out["noise_f1"] = float(f1)
    return out


def compute_external_metrics(
    y_true,
    y_pred,
    *,
    noise_label: int = NOISE_LABEL,
) -> Dict[str, Optional[float]]:
    """Return the full external-metric dict for one prediction.

    Returns NaNs (not exceptions) when the prediction is unusable so a single
    degenerate run never crashes the pipeline.
    """
    nan_result: Dict[str, Optional[float]] = {
        "ari": float("nan"), "nmi": float("nan"), "ami": float("nan"),
        "v_measure": float("nan"), "homogeneity": float("nan"),
        "completeness": float("nan"), "fowlkes_mallows": float("nan"),
        "purity": float("nan"),
        "predicted_noise_ratio": float("nan"), "true_noise_ratio": None,
        "noise_precision": None, "noise_recall": None, "noise_f1": None,
    }
    if y_true is None or y_pred is None:
        return nan_result

    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.size == 0 or yp.size == 0 or yt.shape[0] != yp.shape[0]:
        return nan_result

    try:
        result: Dict[str, Optional[float]] = {
            "ari": float(adjusted_rand_score(yt, yp)),
            "nmi": float(normalized_mutual_info_score(yt, yp)),
            "ami": float(adjusted_mutual_info_score(yt, yp)),
            "v_measure": float(v_measure_score(yt, yp)),
            "homogeneity": float(homogeneity_score(yt, yp)),
            "completeness": float(completeness_score(yt, yp)),
            "fowlkes_mallows": float(fowlkes_mallows_score(yt, yp)),
            "purity": float(_purity(yt, yp)),
        }
    except Exception:
        return nan_result

    result.update(_noise_metrics(yt, yp, noise_label))
    return result
