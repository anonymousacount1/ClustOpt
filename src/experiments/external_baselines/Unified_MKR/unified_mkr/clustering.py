"""Run one resolved configuration via ClustOpt's clustering wrappers.

Thin adapter over ``ClusteringSearchSpace.instantiate_from_config`` +
``fit_predict``. Never raises: clustering failures are captured and reported on
the result object so a single bad partition never aborts the build.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .configuration import ConfigSpec

NOISE_LABEL = -1


@dataclass
class ClusterResult:
    labels: Optional[np.ndarray]
    valid_result: bool
    failure_reason: str
    runtime_sec: float
    n_clusters_found: int
    predicted_k: int
    noise_fraction: float


def _summarise(labels: np.ndarray) -> tuple[int, int, float]:
    unique = np.unique(labels)
    n_clusters_found = int(unique.size)
    non_noise = unique[unique != NOISE_LABEL]
    predicted_k = int(non_noise.size)
    noise_fraction = float(np.mean(labels == NOISE_LABEL)) if labels.size else 0.0
    return n_clusters_found, predicted_k, noise_fraction


def run_configuration(search_space, spec: ConfigSpec, X_decision: np.ndarray) -> ClusterResult:
    t0 = time.perf_counter()
    try:
        model = search_space.instantiate_from_config(spec.to_search_config())
        labels = np.asarray(model.fit_predict(X_decision))
        runtime = time.perf_counter() - t0
        if labels.shape[0] != X_decision.shape[0]:
            return ClusterResult(
                None, False, "label_count_mismatch", runtime, 0, 0, 0.0
            )
        n_found, pred_k, noise = _summarise(labels)
        # A partition with no non-noise cluster is a degenerate (still recorded)
        # result; metric validity (min_clusters) is enforced downstream.
        valid = pred_k >= 1
        reason = "" if valid else "no_non_noise_cluster"
        return ClusterResult(labels, valid, reason, runtime, n_found, pred_k, noise)
    except Exception as exc:  # noqa: BLE001 - intentional: never abort the build
        runtime = time.perf_counter() - t0
        return ClusterResult(
            None, False, f"{type(exc).__name__}: {exc}"[:300], runtime, 0, 0, 0.0
        )
