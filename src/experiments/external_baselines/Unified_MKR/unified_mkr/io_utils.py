"""IO + import-path helpers for the Unified MKR builder.

Reuses the project's long-path-safe atomic writers (from the AutoML4Clust
baseline, which mirrors ``experiment_execution/output_writer.py``) and adds a
single atomic Parquet writer. Also centralises ``sys.path`` wiring so both the
ClustOpt package (``models.ClustOpt``) and the external ML2DAC source
(``external/ml2dac/src``) are importable from worker processes.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk upward until the directory containing ``models/ClustOpt`` is found."""
    here = Path(start or __file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "models" / "ClustOpt").is_dir():
            return parent
    # Fallback: 4 levels up (unified_mkr/ -> Unified_MKR/ -> external_baselines/
    # -> experiments/ -> repo root).
    return here.parents[4]


def ensure_repo_on_path(repo_root: str | Path) -> Path:
    """Prepend the repo root to ``sys.path`` (idempotent)."""
    root = Path(repo_root).resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def ensure_ml2dac_on_path(repo_root: str | Path) -> Path:
    """Prepend ``external/ml2dac/src`` to ``sys.path`` (idempotent).

    ML2DAC uses top-level absolute imports (``from ClusterValidityIndices...``),
    so its ``src`` directory must be importable directly.
    """
    src = Path(repo_root).resolve() / "external" / "ml2dac" / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return src


# --- Re-export long-path-safe atomic IO helpers ----------------------------
# Importing lazily keeps this module usable even before the repo root is on the
# path (the runner calls ensure_repo_on_path first).
def _load_path_helpers():
    from experiments.external_baselines.AutoML4Clust.utils.path_utils import (
        ext,
        ensure_dir,
        path_exists,
        write_json_atomic,
        write_text_atomic,
        write_csv_atomic,
        write_npy_atomic,
        read_json,
    )
    return {
        "ext": ext,
        "ensure_dir": ensure_dir,
        "path_exists": path_exists,
        "write_json_atomic": write_json_atomic,
        "write_text_atomic": write_text_atomic,
        "write_csv_atomic": write_csv_atomic,
        "write_npy_atomic": write_npy_atomic,
        "read_json": read_json,
    }


def __getattr__(name: str):
    helpers = _load_path_helpers()
    if name in helpers:
        return helpers[name]
    raise AttributeError(name)


def write_parquet_atomic(path: str | Path, df: pd.DataFrame) -> Path:
    """Atomic, long-path-safe Parquet write (temp file + ``os.replace``)."""
    helpers = _load_path_helpers()
    ext = helpers["ext"]
    path = Path(path)
    os.makedirs(ext(path.parent), exist_ok=True)
    tmp = str(path) + ".tmp"
    df.to_parquet(ext(tmp), index=False, engine="pyarrow")
    os.replace(ext(tmp), ext(path))
    return path


def read_parquet(path: str | Path) -> Optional[pd.DataFrame]:
    """Long-path-safe Parquet read; returns ``None`` on any error."""
    helpers = _load_path_helpers()
    ext = helpers["ext"]
    try:
        return pd.read_parquet(ext(path), engine="pyarrow")
    except Exception:
        return None


def downcast_labels(labels: np.ndarray) -> np.ndarray:
    """Cast cluster labels to the smallest safe signed int dtype.

    Cluster ids are tiny (0..k-1 plus the -1 noise label), so int64 wastes ~4x
    disk over int16. Falls back to wider types if values somehow exceed the
    range (defensive; never lossy).
    """
    a = np.asarray(labels)
    if a.size == 0:
        return a.astype(np.int16)
    lo, hi = int(a.min()), int(a.max())
    for dtype in (np.int16, np.int32):
        info = np.iinfo(dtype)
        if lo >= info.min and hi <= info.max:
            return a.astype(dtype)
    return a.astype(np.int64)


def array_hash(arr: np.ndarray) -> str:
    """Stable SHA1 over an array's dtype, shape, and bytes."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha1()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()
