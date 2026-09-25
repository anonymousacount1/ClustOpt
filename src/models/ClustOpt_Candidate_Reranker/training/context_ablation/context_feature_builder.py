"""Design matrices for the four Stage-2B2 context conditions.

The candidate-local block is byte-for-byte the Stage-2B1 ``MLP_TOP10``
representation; context is appended, never substituted.

**Memory is the design constraint here.** ``C3`` is 1,085,658 x 597 float32 =
2.47 GB, and building it with ``np.hstack`` would hold the operands and the
result simultaneously (~5 GB) on a machine with ~4 GB free. So the matrix is
preallocated once and filled block-by-block, and every slate->candidate expansion
is chunked so the temporary from fancy indexing stays around 100 MB rather than
scaling with the block.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import feature_builder as FB

LOCAL_CONDITION = "C0_LOCAL"
REGIME_SOURCE = "mlp"
REGIME_K_MODE = "top10"
REGIME_ID = "mlp__top10"
STAGE2B1_CONDITION = "MLP_TOP10"

N_META = 250
N_UTIL = 120
N_LOCAL = 227                      # 39 common + 60 cvi + 60 obs + 60 valid + 7 + 1
FILL_CHUNK = 100_000

CONDITIONS: Tuple[Dict[str, Any], ...] = (
    {"condition_id": "C0_LOCAL", "blocks": (), "reused": True,
     "description": "Stage-2B1 MLP_TOP10 candidate-local model, reused verbatim"},
    {"condition_id": "C1_LOCAL_META", "blocks": ("meta",), "reused": False,
     "description": "LOCAL + 250 frozen dataset/view meta-features"},
    {"condition_id": "C2_LOCAL_UTIL", "blocks": ("util",), "reused": False,
     "description": "LOCAL + 60 MLP + 60 KNN predicted utilities"},
    {"condition_id": "C3_LOCAL_META_UTIL", "blocks": ("meta", "util"),
     "reused": False, "description": "LOCAL + META_250 + UTILITIES_120"},
)
# Tie-break ordering by feature complexity (brief section 27).
COMPLEXITY_ORDER = {"C0_LOCAL": 0, "C2_LOCAL_UTIL": 1, "C1_LOCAL_META": 2,
                    "C3_LOCAL_META_UTIL": 3}


def all_conditions() -> List[Dict[str, Any]]:
    return [dict(c) for c in CONDITIONS]


def new_conditions() -> List[Dict[str, Any]]:
    return [dict(c) for c in CONDITIONS if not c["reused"]]


def expected_dim(condition_id: str) -> int:
    c = {x["condition_id"]: x for x in CONDITIONS}[condition_id]
    d = N_LOCAL
    if "meta" in c["blocks"]:
        d += N_META
    if "util" in c["blocks"]:
        d += N_UTIL
    return d


def feature_names(condition_id: str, metric_names: Sequence[str],
                  cat_names: Sequence[str], meta_cols: Sequence[str],
                  util_cols: Sequence[str]) -> List[str]:
    names = ([f"view__{v}" for v in FB.VIEW_IDS] + list(cat_names)
             + list(FB.PARTITION_COLS)
             + [f"cvi__{m}" for m in metric_names]
             + [f"obsmask__{m}" for m in metric_names]
             + [f"validmask__{m}" for m in metric_names]
             + [f"source__{s}" for s in FB.SOURCES]
             + [f"kmode__{k}" for k in FB.K_MODES] + ["actual_k"])
    c = {x["condition_id"]: x for x in CONDITIONS}[condition_id]
    if "meta" in c["blocks"]:
        names += [f"meta__{m}" for m in meta_cols]
    if "util" in c["blocks"]:
        names += [f"ctxutil__{u}" for u in util_cols]
    return names


def schema_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


def _fill_from_slate(out: np.ndarray, col0: int, block: np.ndarray,
                     slate_pos: np.ndarray) -> None:
    """``out[:, col0:col0+w] = block[slate_pos]`` in row chunks."""
    w = block.shape[1]
    n = out.shape[0]
    for a in range(0, n, FILL_CHUNK):
        b = min(a + FILL_CHUNK, n)
        out[a:b, col0:col0 + w] = block[slate_pos[a:b]]


def build_context_X(condition_id: str, *, view_codes: np.ndarray,
                    cat_rows: np.ndarray, cat_matrix: np.ndarray,
                    partition: np.ndarray, cvi: np.ndarray,
                    cvi_valid: np.ndarray, observability: np.ndarray,
                    actual_k: np.ndarray, slate_pos: np.ndarray,
                    meta: Optional[np.ndarray] = None,
                    util: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, int]:
    """Preallocated design matrix. Returns (X, n_common) for the leakage check.

    ``observability``/``actual_k`` are SLATE-level and expanded here through
    ``slate_pos``; the candidate-local semantics are identical to Stage 2B-1's
    ``deployable_block``, including the rule that a CVI value survives only where
    the metric is both observed AND valid.
    """
    c = {x["condition_id"]: x for x in CONDITIONS}[condition_id]
    n = len(view_codes)
    n_metrics = cvi.shape[1]
    n_common = len(FB.VIEW_IDS) + cat_matrix.shape[1] + partition.shape[1]
    d = expected_dim(condition_id)
    X = np.empty((n, d), dtype=np.float32)

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

    # ---- masked CVI values, then the two masks -----------------------------
    cvi_col = col
    for a in range(0, n, FILL_CHUNK):
        b = min(a + FILL_CHUNK, n)
        obs = observability[slate_pos[a:b]].astype(bool)
        val = obs & cvi_valid[a:b].astype(bool)
        X[a:b, cvi_col:cvi_col + n_metrics] = np.where(val, cvi[a:b], np.nan)
        X[a:b, cvi_col + n_metrics:cvi_col + 2 * n_metrics] = obs
        X[a:b, cvi_col + 2 * n_metrics:cvi_col + 3 * n_metrics] = val
    col += 3 * n_metrics

    X[:, col:col + len(FB.SOURCES)] = 0.0
    X[:, col + FB.SOURCES.index(REGIME_SOURCE)] = 1.0
    col += len(FB.SOURCES)
    X[:, col:col + len(FB.K_MODES)] = 0.0
    X[:, col + FB.K_MODES.index(REGIME_K_MODE)] = 1.0
    col += len(FB.K_MODES)
    for a in range(0, n, FILL_CHUNK):
        b = min(a + FILL_CHUNK, n)
        X[a:b, col] = actual_k[slate_pos[a:b]]
    col += 1

    if col != N_LOCAL:
        raise AssertionError(f"local block is {col} wide, expected {N_LOCAL}")

    if "meta" in c["blocks"]:
        if meta is None:
            raise ValueError(f"{condition_id} requires META_250")
        if meta.shape[1] != N_META:
            raise AssertionError(f"META block is {meta.shape[1]}, expected {N_META}")
        _fill_from_slate(X, col, meta, slate_pos)
        col += N_META
    if "util" in c["blocks"]:
        if util is None:
            raise ValueError(f"{condition_id} requires UTILITIES_120")
        if util.shape[1] != N_UTIL:
            raise AssertionError(f"UTIL block is {util.shape[1]}, expected {N_UTIL}")
        _fill_from_slate(X, col, util, slate_pos)
        col += N_UTIL

    if col != d:
        raise AssertionError(f"filled {col} columns, expected {d}")
    return X, n_common


def assert_no_hidden_leakage(X: np.ndarray, observability: np.ndarray,
                             slate_pos: np.ndarray, n_common: int,
                             n_metrics: int = 60) -> None:
    """Unobserved CVI positions must be NaN on every row, checked in chunks."""
    n = X.shape[0]
    bad = 0
    for a in range(0, n, FILL_CHUNK):
        b = min(a + FILL_CHUNK, n)
        hidden = ~observability[slate_pos[a:b]].astype(bool)
        blk = X[a:b, n_common:n_common + n_metrics]
        if hidden.any():
            bad += int((~np.isnan(blk[hidden])).sum())
    if bad:
        raise AssertionError(f"{bad} unobserved CVI cells are not NaN")


def assert_context_finite(meta: Optional[np.ndarray],
                          util: Optional[np.ndarray]) -> Dict[str, int]:
    """Frozen context blocks are expected finite; no imputation is invented."""
    out = {}
    for name, blk in (("meta", meta), ("util", util)):
        if blk is None:
            continue
        n_bad = int((~np.isfinite(blk)).sum())
        if n_bad:
            raise AssertionError(
                f"{n_bad} non-finite values in the frozen {name} context block; "
                f"the source schema defines no imputation -- stopping")
        out[name] = n_bad
    return out
