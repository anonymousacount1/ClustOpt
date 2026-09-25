"""Cluster-count (K) metrics.

``true_k`` is used only here, *after* clustering, for evaluation -- never during
search. ``selected_k`` counts non-noise predicted clusters.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

NOISE_LABEL = -1


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


def compute_k_metrics(
    y_pred,
    true_k: Optional[int],
    *,
    noise_label: int = NOISE_LABEL,
) -> Dict[str, Optional[float]]:
    """Return selected_k / true_k / k_correct / k_abs_error / k_signed_error."""
    selected_k = selected_k_from_labels(y_pred, noise_label=noise_label)

    if true_k is None:
        return {
            "selected_k": int(selected_k),
            "true_k": None,
            "k_correct": None,
            "k_abs_error": None,
            "k_signed_error": None,
        }

    true_k_int = int(true_k)
    signed = selected_k - true_k_int
    return {
        "selected_k": int(selected_k),
        "true_k": true_k_int,
        "k_correct": bool(selected_k == true_k_int),
        "k_abs_error": int(abs(signed)),
        "k_signed_error": int(signed),
    }
