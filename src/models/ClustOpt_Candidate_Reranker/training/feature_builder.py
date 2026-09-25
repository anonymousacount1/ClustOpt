"""Candidate-local feature construction for the 11 information conditions.

Stage 2B-1 deliberately restricts the model to **candidate-local** information.
Excluded on purpose: the 250 meta-features, both 60-dim utility vectors, metric
weights, objective J, weighting mode and search history. The question here is
whether observable candidate quality alone carries reranking signal.

The masking guarantee is structural, not procedural. For a deployable condition
the CVI column is built as::

    value = where(observed AND valid, cvi, NaN)

so an unobserved metric's value is destroyed while the matrix is assembled. No
code path downstream can reach it, because it is not present in ``X``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RICH_CONDITION = "RICH_60"
VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")
ALGORITHM_BASES: Tuple[str, ...] = (
    "agglomerative", "birch", "dbscan", "gmm", "hdbscan",
    "hdbscan_constrained", "kmeans", "minibatch_kmeans",
)
# Numeric hyperparameters that actually vary in the frozen 96-candidate grid.
NUMERIC_HP: Tuple[str, ...] = (
    "n_clusters", "n_components", "batch_size", "random_state", "threshold",
    "branching_factor", "eps", "min_samples", "min_cluster_size",
)
# Single-valued categorical hyperparameters: presence is the whole signal.
FLAG_HP: Tuple[str, ...] = ("init", "linkage", "covariance_type")
PARTITION_COLS: Tuple[str, ...] = (
    "part__n_clusters_wo_noise", "part__noise_ratio", "part__n_labels_unique",
    "part__valid", "part__has_exception",
)
SOURCES: Tuple[str, ...] = ("mlp", "knn")
K_MODES: Tuple[str, ...] = ("top1", "top3", "top5", "top10", "dynamic")


def condition_id(source: Optional[str], k_mode: Optional[str]) -> str:
    if source is None:
        return RICH_CONDITION
    return f"{source.upper()}_{k_mode.upper()}"


def all_conditions() -> List[Dict[str, Any]]:
    """RICH first, then the 10 deployable regimes in frozen order."""
    out = [{"condition_id": RICH_CONDITION, "kind": "RICH",
            "utility_source": None, "k_mode": None, "regime_id": None}]
    for s in SOURCES:
        for k in K_MODES:
            out.append({"condition_id": condition_id(s, k), "kind": "DEPLOYABLE",
                        "utility_source": s, "k_mode": k,
                        "regime_id": f"{s}__{k}"})
    if len(out) != 11:
        raise AssertionError(f"expected 11 conditions, built {len(out)}")
    return out


# ------------------------------------------------------------ common context
def build_catalogue_matrix(cat: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """(view, candidate_index) configuration block -> dense matrix + names.

    Rows are ordered by (view_id, candidate_index) so a candidate row can index
    into it with ``view_offset + candidate_index``.
    """
    cat = cat.sort_values(["view_id", "candidate_index"]).reset_index(drop=True)
    names: List[str] = []
    cols: List[np.ndarray] = []

    for a in ALGORITHM_BASES:
        cols.append((cat["algorithm_base"] == a).to_numpy(np.float32))
        names.append(f"algo__{a}")
    cols.append(pd.to_numeric(cat["cfg__constrained_target_k"],
                              errors="coerce").to_numpy(np.float32))
    names.append("cfg__constrained_target_k")
    cols.append(cat["cfgmask__constrained_target_k"].to_numpy(np.float32))
    names.append("cfgmask__constrained_target_k")
    for h in NUMERIC_HP:
        cols.append(pd.to_numeric(cat[f"cfg__{h}"],
                                  errors="coerce").to_numpy(np.float32))
        names.append(f"cfg__{h}")
        cols.append(cat[f"cfgmask__{h}"].to_numpy(np.float32))
        names.append(f"cfgmask__{h}")
    for h in FLAG_HP:
        cols.append(cat[f"cfgmask__{h}"].to_numpy(np.float32))
        names.append(f"cfgmask__{h}")
    return np.column_stack(cols).astype(np.float32), names


def catalogue_index(cat: pd.DataFrame) -> Dict[str, int]:
    """view_id -> offset of its 32-row block in the sorted catalogue."""
    order = sorted(cat["view_id"].unique())
    return {v: i * 32 for i, v in enumerate(order)}


def common_block(view_codes: np.ndarray, cat_rows: np.ndarray,
                 cat_matrix: np.ndarray, cat_names: Sequence[str],
                 partition: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """View one-hot + configuration + persisted partition descriptors."""
    n = len(view_codes)
    oh = np.zeros((n, len(VIEW_IDS)), dtype=np.float32)
    oh[np.arange(n), view_codes] = 1.0
    names = [f"view__{v}" for v in VIEW_IDS] + list(cat_names) + list(PARTITION_COLS)
    return np.hstack([oh, cat_matrix[cat_rows], partition.astype(np.float32)]), names


# ------------------------------------------------------------------ CVI block
def rich_block(cvi: np.ndarray, cvi_valid: np.ndarray,
               metric_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """All 60 CVI values (NaN where invalid) + 60 validity indicators."""
    v = np.where(cvi_valid.astype(bool), cvi, np.nan).astype(np.float32)
    names = ([f"cvi__{m}" for m in metric_names]
             + [f"cvi_valid__{m}" for m in metric_names])
    return np.hstack([v, cvi_valid.astype(np.float32)]), names


def deployable_block(cvi: np.ndarray, cvi_valid: np.ndarray,
                     observability: np.ndarray, metric_names: Sequence[str],
                     source: str, k_mode: str, actual_k: np.ndarray
                     ) -> Tuple[np.ndarray, List[str]]:
    """Masked CVI values + observability mask + observed-validity mask + context.

    ``observability[m] = 1`` iff the regime selected metric m for this slate.
    ``validity[m] = 1`` iff m is selected AND its stored CVI value is valid.
    The value column is NaN wherever either is 0 -- the hidden values never enter
    the matrix at all.
    """
    obs = observability.astype(bool)
    val = obs & cvi_valid.astype(bool)
    v = np.where(val, cvi, np.nan).astype(np.float32)

    n = len(cvi)
    src = np.zeros((n, len(SOURCES)), dtype=np.float32)
    src[:, SOURCES.index(source)] = 1.0
    km = np.zeros((n, len(K_MODES)), dtype=np.float32)
    km[:, K_MODES.index(k_mode)] = 1.0

    names = ([f"cvi__{m}" for m in metric_names]
             + [f"obsmask__{m}" for m in metric_names]
             + [f"validmask__{m}" for m in metric_names]
             + [f"source__{s}" for s in SOURCES]
             + [f"kmode__{k}" for k in K_MODES]
             + ["actual_k"])
    return (np.hstack([v, obs.astype(np.float32), val.astype(np.float32),
                       src, km, actual_k.reshape(-1, 1).astype(np.float32)]),
            names)


def assert_no_hidden_leakage(X: np.ndarray, cvi: np.ndarray,
                             observability: np.ndarray, n_common: int,
                             n_metrics: int = 60) -> None:
    """Every unobserved CVI position in X must be NaN, on every row.

    Checked on the assembled matrix rather than trusted from the constructor:
    this is the one guarantee that makes a deployable condition deployable.
    """
    block = X[:, n_common:n_common + n_metrics]
    hidden = ~observability.astype(bool)
    if hidden.any() and not np.isnan(block[hidden]).all():
        n_bad = int((~np.isnan(block[hidden])).sum())
        raise AssertionError(f"{n_bad} unobserved CVI cells are not NaN")
    shown = observability.astype(bool)
    if shown.any():
        finite_src = np.isfinite(cvi[shown])
        got = block[shown]
        if not np.array_equal(np.isfinite(got) & finite_src, np.isfinite(got)):
            raise AssertionError("observed CVI values diverge from the source")


def feature_schema_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


def build_X(condition: Dict[str, Any], *, view_codes, cat_rows, cat_matrix,
            cat_names, partition, cvi, cvi_valid, metric_names,
            observability=None, actual_k=None
            ) -> Tuple[np.ndarray, List[str], int]:
    """Assemble the full design matrix for one condition. Returns (X, names, n_common)."""
    C, cnames = common_block(view_codes, cat_rows, cat_matrix, cat_names,
                             partition)
    if condition["kind"] == "RICH":
        B, bnames = rich_block(cvi, cvi_valid, metric_names)
    else:
        if observability is None or actual_k is None:
            raise ValueError("deployable condition needs observability + actual_k")
        B, bnames = deployable_block(cvi, cvi_valid, observability, metric_names,
                                     condition["utility_source"],
                                     condition["k_mode"], actual_k)
    X = np.hstack([C, B]).astype(np.float32)
    return X, cnames + bnames, C.shape[1]
