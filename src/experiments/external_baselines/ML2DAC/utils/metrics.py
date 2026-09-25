"""External clustering metrics for the ML2DAC baseline.

Re-exported verbatim from the AutoML4Clust baseline so ML2DAC and AutoML4Clust
compute *identical* metrics (which in turn match the ClustOpt experiment
pipeline). Noise label convention: -1, excluded from selected_k.
"""
from __future__ import annotations

from experiments.external_baselines.AutoML4Clust.utils.metrics import (  # noqa: F401
    NOISE_LABEL,
    compute_external_metrics,
    compute_k_metrics,
    compute_all_metrics,
    selected_k_from_labels,
    n_noise_points,
)

__all__ = [
    "NOISE_LABEL",
    "compute_external_metrics",
    "compute_k_metrics",
    "compute_all_metrics",
    "selected_k_from_labels",
    "n_noise_points",
]
