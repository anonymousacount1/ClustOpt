"""Offline scoring. Runs after predictions are persisted, outside the timer."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

PRIMARY = ("ari",)
SECONDARY = ("nmi", "ami", "fmi", "purity", "k_error")


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    from sklearn import metrics as M
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    k_true = int(len(np.unique(yt)))
    k_pred = int(len(np.unique(yp[yp >= 0]))) if (yp >= 0).any() else 0
    cm = M.cluster.contingency_matrix(yt, yp)
    return {"ari": float(M.adjusted_rand_score(yt, yp)),
            "nmi": float(M.normalized_mutual_info_score(yt, yp)),
            "ami": float(M.adjusted_mutual_info_score(yt, yp)),
            "fmi": float(M.fowlkes_mallows_score(yt, yp)),
            "purity": float(cm.max(axis=0).sum() / cm.sum()),
            "predicted_k": k_pred, "true_k": k_true,
            "k_error": k_pred - k_true, "k_correct": bool(k_pred == k_true)}
