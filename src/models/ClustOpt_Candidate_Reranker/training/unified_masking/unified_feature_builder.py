"""C2 design matrix with PER-SLATE regime identity.

Stage 2B-2's builder hard-codes one regime for the whole matrix. Unified training
needs the source/K one-hot and ``actual_k`` to vary per slate, because each
training slate carries its own assigned regime.

The column layout is byte-identical to Stage-2B2 ``C2_LOCAL_UTIL`` -- same names,
same order, same width (347) -- so the clean MLP_TOP10 unified-vs-dedicated
comparison differs only in which rows were trained on, never in the schema.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import feature_builder as FB
from ..context_ablation import context_feature_builder as CFB

CONDITION = "C2_LOCAL_UTIL"
EXPECTED_DIM = 347                  # 227 local + 120 utilities
FILL_CHUNK = CFB.FILL_CHUNK


def feature_names(metric_names: Sequence[str], cat_names: Sequence[str],
                  util_cols: Sequence[str]) -> List[str]:
    return CFB.feature_names(CONDITION, metric_names, cat_names, [], util_cols)


def build_unified_X(*, view_codes: np.ndarray, cat_rows: np.ndarray,
                    cat_matrix: np.ndarray, partition: np.ndarray,
                    cvi: np.ndarray, cvi_valid: np.ndarray,
                    observability: np.ndarray, actual_k: np.ndarray,
                    source_code: np.ndarray, kmode_code: np.ndarray,
                    slate_pos: np.ndarray, util: np.ndarray
                    ) -> Tuple[np.ndarray, int]:
    """Preallocated matrix; regime identity varies per slate.

    ``observability``/``actual_k``/``source_code``/``kmode_code`` are SLATE-level
    and expanded through ``slate_pos``. A CVI value survives only where the
    metric is both observed AND valid -- identical to Stage 2B-1/2B-2.
    """
    n = len(view_codes)
    n_metrics = cvi.shape[1]
    n_common = len(FB.VIEW_IDS) + cat_matrix.shape[1] + partition.shape[1]
    X = np.empty((n, EXPECTED_DIM), dtype=np.float32)

    col = 0
    X[:, col:col + len(FB.VIEW_IDS)] = 0.0
    X[np.arange(n), col + view_codes] = 1.0
    col += len(FB.VIEW_IDS)

    w = cat_matrix.shape[1]
    for a in range(0, n, FILL_CHUNK):
        b = min(a + FILL_CHUNK, n)
        X[a:b, col:col + w] = cat_matrix[cat_rows[a:b]]
    col += w

    X[:, col:col + partition.shape[1]] = partition
    col += partition.shape[1]

    cvi_col = col
    for a in range(0, n, FILL_CHUNK):
        b = min(a + FILL_CHUNK, n)
        sp = slate_pos[a:b]
        obs = observability[sp].astype(bool)
        val = obs & cvi_valid[a:b].astype(bool)
        X[a:b, cvi_col:cvi_col + n_metrics] = np.where(val, cvi[a:b], np.nan)
        X[a:b, cvi_col + n_metrics:cvi_col + 2 * n_metrics] = obs
        X[a:b, cvi_col + 2 * n_metrics:cvi_col + 3 * n_metrics] = val
    col += 3 * n_metrics

    # ---- regime identity, PER SLATE ----------------------------------------
    ns, nk = len(FB.SOURCES), len(FB.K_MODES)
    X[:, col:col + ns] = 0.0
    X[:, col + ns:col + ns + nk] = 0.0
    for a in range(0, n, FILL_CHUNK):
        b = min(a + FILL_CHUNK, n)
        sp = slate_pos[a:b]
        rows = np.arange(a, b)
        X[rows, col + source_code[sp]] = 1.0
        X[rows, col + ns + kmode_code[sp]] = 1.0
        X[a:b, col + ns + nk] = actual_k[sp]
    col += ns + nk + 1

    if col != CFB.N_LOCAL:
        raise AssertionError(f"local block is {col} wide, expected "
                             f"{CFB.N_LOCAL}")
    if util.shape[1] != CFB.N_UTIL:
        raise AssertionError(f"UTIL block is {util.shape[1]}, expected 120")
    CFB._fill_from_slate(X, col, util, slate_pos)
    col += CFB.N_UTIL
    if col != EXPECTED_DIM:
        raise AssertionError(f"filled {col} columns, expected {EXPECTED_DIM}")
    return X, n_common


def assert_no_hidden_leakage(X: np.ndarray, observability: np.ndarray,
                             slate_pos: np.ndarray, n_common: int,
                             n_metrics: int = 60) -> None:
    CFB.assert_no_hidden_leakage(X, observability, slate_pos, n_common,
                                 n_metrics)


def assert_regime_identity(X: np.ndarray, slate_pos: np.ndarray,
                           source_code: np.ndarray, kmode_code: np.ndarray,
                           n_common: int, n_metrics: int = 60) -> None:
    """The encoded regime on each row must match its slate's assignment."""
    base = n_common + 3 * n_metrics
    ns, nk = len(FB.SOURCES), len(FB.K_MODES)
    n = X.shape[0]
    for a in range(0, n, FILL_CHUNK):
        b = min(a + FILL_CHUNK, n)
        sp = slate_pos[a:b]
        src = X[a:b, base:base + ns]
        km = X[a:b, base + ns:base + ns + nk]
        if not np.array_equal(src.argmax(axis=1), source_code[sp]):
            raise AssertionError("source one-hot does not match the assignment")
        if not np.array_equal(km.argmax(axis=1), kmode_code[sp]):
            raise AssertionError("k-mode one-hot does not match the assignment")
        if not np.allclose(src.sum(axis=1), 1.0) or \
           not np.allclose(km.sum(axis=1), 1.0):
            raise AssertionError("regime one-hot is not exactly one-hot")
