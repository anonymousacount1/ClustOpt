"""External clustering metrics for the AutoML4Clust baseline.

The formulas here are deliberately identical to the ClustOpt experiment
pipeline (``experiment_execution/external_metrics.py`` + ``k_metrics.py``) so
AutoML4Clust results are directly comparable. The implementation is kept
self-contained (a copy, not an import) so the baseline stays isolated and
robust to refactors of the ClustOpt package -- the only shared contract is the
NOISE_LABEL convention and the metric definitions, both pinned below.

Every function returns NaNs/None on degenerate input rather than raising, so a
single unusable prediction never crashes the pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

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

# Matches ClustOpt: noise points carry label -1 and are excluded from selected_k.
NOISE_LABEL = -1


# ---------------------------------------------------------------------------
# External (ground-truth) metrics -- mirrors external_metrics.py
# ---------------------------------------------------------------------------
def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of points whose predicted cluster's majority true-label matches."""
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
    y_true, y_pred, *, noise_label: int = NOISE_LABEL
) -> Dict[str, Optional[float]]:
    """ARI / NMI / AMI (+ V-measure, homogeneity, completeness, FM, purity, noise)."""
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


# ---------------------------------------------------------------------------
# K metrics -- mirrors k_metrics.py, plus the task's extra naming.
# ---------------------------------------------------------------------------
def selected_k_from_labels(y_pred, *, noise_label: int = NOISE_LABEL) -> int:
    """Number of unique predicted clusters, excluding the noise label."""
    if y_pred is None:
        return 0
    yp = np.asarray(y_pred).reshape(-1)
    if yp.size == 0:
        return 0
    unique = set(np.unique(yp).tolist())
    unique.discard(noise_label)
    return int(len(unique))


def n_noise_points(y_pred, *, noise_label: int = NOISE_LABEL) -> int:
    if y_pred is None:
        return 0
    yp = np.asarray(y_pred).reshape(-1)
    return int(np.sum(yp == noise_label))


def compute_k_metrics(
    y_pred, true_k: Optional[int], *, noise_label: int = NOISE_LABEL
) -> Dict[str, Any]:
    """selected_k / true_k / k_error / abs_k_error / exact_k_match (+ ClustOpt aliases)."""
    selected_k = selected_k_from_labels(y_pred, noise_label=noise_label)
    n_clusters_excl_noise = selected_k
    n_noise = n_noise_points(y_pred, noise_label=noise_label)

    if true_k is None:
        return {
            "selected_k": int(selected_k),
            "true_k": None,
            "k_error": None,
            "abs_k_error": None,
            "exact_k_match": None,
            # ClustOpt aliases for cross-pipeline consistency.
            "k_signed_error": None,
            "k_abs_error": None,
            "k_correct": None,
            "n_predicted_clusters_excluding_noise": int(n_clusters_excl_noise),
            "n_noise_points": int(n_noise),
        }

    true_k_int = int(true_k)
    signed = int(selected_k - true_k_int)
    return {
        "selected_k": int(selected_k),
        "true_k": true_k_int,
        "k_error": signed,
        "abs_k_error": int(abs(signed)),
        "exact_k_match": bool(selected_k == true_k_int),
        # ClustOpt aliases.
        "k_signed_error": signed,
        "k_abs_error": int(abs(signed)),
        "k_correct": bool(selected_k == true_k_int),
        "n_predicted_clusters_excluding_noise": int(n_clusters_excl_noise),
        "n_noise_points": int(n_noise),
    }


def compute_all_metrics(
    y_true, y_pred, true_k: Optional[int], *, noise_label: int = NOISE_LABEL
) -> Dict[str, Any]:
    """Convenience: merged external + K metrics for one prediction.

    Robust: returns a fully-populated dict (NaNs/None where undefined) even for
    degenerate predictions (single cluster, all noise, mismatched length).
    """
    out: Dict[str, Any] = {}
    out.update(compute_external_metrics(y_true, y_pred, noise_label=noise_label))
    out.update(compute_k_metrics(y_pred, true_k, noise_label=noise_label))
    return out


__all__ = [
    "NOISE_LABEL",
    "compute_external_metrics",
    "compute_k_metrics",
    "compute_all_metrics",
    "selected_k_from_labels",
    "n_noise_points",
]
