"""Build the frozen 347-dim C2 representation for HISTORICAL ONLINE candidates.

Two things differ from the fixed-32 development domain and are handled explicitly
rather than papered over:

1. **Configurations are Optuna-sampled**, so the 96-row per-view catalogue does
   not apply. The configuration block is parsed per candidate with the same
   ``candidate_schema.configuration_row`` used to build that catalogue, and is
   laid out through ``feature_builder.build_catalogue_matrix`` so the column
   order is identical by construction.

2. **The online algorithm space is broader.** Online runs contain ``optics`` and
   ``spectral``, which are absent from the fixed-32 grid. Their algorithm one-hot
   is legitimately all-zero -- an honest "unseen family" encoding -- and every
   such candidate is COUNTED so the distribution shift is reported, not hidden.

Observability comes from the run itself: the trial log physically contains only
the metrics the policy selected, and that set matches ``selected_metrics.json``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (
    unified_feature_builder as UFB,
)

EXPECTED_DIM = UFB.EXPECTED_DIM          # 347
N_METRICS = 60


def config_block(algorithms: Sequence[str], configs: Sequence[str]
                 ) -> Tuple[np.ndarray, int]:
    """(n, 31) configuration matrix in the frozen catalogue column order.

    Returns the matrix and the count of candidates whose algorithm family is
    absent from the fixed-32 training grid.
    """
    rows = [CS.configuration_row(a, c) for a, c in zip(algorithms, configs)]
    frame = pd.DataFrame(rows)
    frame["view_id"] = "x_only"                     # constant -> stable sort
    frame["candidate_index"] = np.arange(len(frame))
    mat, _ = FB.build_catalogue_matrix(frame)
    unseen = int(sum(1 for r in rows
                     if r["algorithm_base"] not in FB.ALGORITHM_BASES))
    return mat.astype(np.float32), unseen


def build_online_X(*, trial_log: pd.DataFrame, view_id: str,
                   metric_names: Sequence[str], utility_vector: np.ndarray,
                   source: str, k_mode: str, actual_k: int
                   ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Frozen C2 matrix for one historical online run.

    ``utility_vector`` is the 120-dim final Split-1 context (60 MLP + 60 KNN).
    ARI is never touched here.
    """
    n = len(trial_log)
    algs = trial_log["algorithm"].astype(str).tolist()
    cfgs = trial_log["config_str"].astype(str).tolist()
    cfg_mat, n_unseen = config_block(algs, cfgs)

    X = np.empty((n, EXPECTED_DIM), dtype=np.float32)
    col = 0
    X[:, col:col + len(FB.VIEW_IDS)] = 0.0
    X[:, col + FB.VIEW_IDS.index(view_id)] = 1.0
    col += len(FB.VIEW_IDS)

    w = cfg_mat.shape[1]
    X[:, col:col + w] = cfg_mat
    col += w

    for name in FB.PARTITION_COLS:
        src = {"part__n_clusters_wo_noise": "n_clusters_wo_noise",
               "part__noise_ratio": "noise_ratio",
               "part__n_labels_unique": "n_labels_unique"}.get(name)
        if src is not None:
            X[:, col] = pd.to_numeric(trial_log[src],
                                      errors="coerce").to_numpy(np.float32)
        elif name == "part__valid":
            X[:, col] = trial_log["valid"].map(CS.is_valid_flag).to_numpy(np.int8)
        else:                                         # part__has_exception
            e = trial_log["exception"].astype(str).str.strip()
            X[:, col] = ((e != "") & (e.str.lower() != "nan")).to_numpy(np.int8)
        col += 1

    # ---- observability comes from what the run actually computed -----------
    present = [m for m in metric_names if f"{m}_norm" in trial_log.columns]
    obs = np.zeros(N_METRICS, dtype=bool)
    vals = np.full((n, N_METRICS), np.nan, dtype=np.float64)
    for j, m in enumerate(metric_names):
        if f"{m}_norm" in trial_log.columns:
            obs[j] = True
            vals[:, j] = pd.to_numeric(trial_log[f"{m}_norm"],
                                       errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(vals) & obs[None, :]
    X[:, col:col + N_METRICS] = np.where(valid, vals, np.nan)
    X[:, col + N_METRICS:col + 2 * N_METRICS] = np.broadcast_to(obs, (n, N_METRICS))
    X[:, col + 2 * N_METRICS:col + 3 * N_METRICS] = valid
    col += 3 * N_METRICS

    X[:, col:col + len(FB.SOURCES)] = 0.0
    X[:, col + FB.SOURCES.index(source)] = 1.0
    col += len(FB.SOURCES)
    X[:, col:col + len(FB.K_MODES)] = 0.0
    X[:, col + FB.K_MODES.index(k_mode)] = 1.0
    col += len(FB.K_MODES)
    X[:, col] = float(actual_k)
    col += 1

    if col != 227:
        raise AssertionError(f"local block is {col} wide, expected 227")
    X[:, col:col + 120] = utility_vector.astype(np.float32)[None, :]
    col += 120
    if col != EXPECTED_DIM:
        raise AssertionError(f"built {col} columns, expected {EXPECTED_DIM}")

    diag = {"n_candidates": n, "n_observed_metrics": int(obs.sum()),
            "observed_metrics": "|".join(present),
            "n_unseen_algorithm_candidates": n_unseen,
            "n_unique_configs": int(pd.Series(cfgs).nunique())}
    return X, 227 - 3 * N_METRICS - len(FB.SOURCES) - len(FB.K_MODES) - 1, diag


def assert_no_hidden_exposure(X: np.ndarray, n_common: int,
                              metric_names: Sequence[str],
                              trial_log: pd.DataFrame) -> None:
    """Every metric absent from the run must be NaN in the value block."""
    blk = X[:, n_common:n_common + N_METRICS]
    for j, m in enumerate(metric_names):
        if f"{m}_norm" not in trial_log.columns:
            if not np.isnan(blk[:, j]).all():
                raise AssertionError(f"unobserved metric {m} is not NaN")
