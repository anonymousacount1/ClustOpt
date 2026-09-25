r"""Path helpers for the AutoML4Clust baseline.

Two responsibilities:

1. **Long-path-safe, atomic IO** -- the experiment output tree
   ``<dataset>/experiments/split_XX/AutoML4Clust/<record>/<file>`` is deep enough
   to exceed the legacy Windows ``MAX_PATH`` (260) limit on this machine
   (``LongPathsEnabled=0``). Every file operation is routed through the
   extended-length ``\\?\`` prefix on Windows (a no-op elsewhere). Writes are
   atomic (temp file + ``os.replace``) so a crash mid-write never leaves a
   partial file that resume could mistake for a completed run. This mirrors
   ``experiment_execution/output_writer.py``.

2. **Locating the external AutoML4Clust source** and the record-type/view mapping
   shared across the baseline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Record-type <-> ClustOpt view-id mapping.
#
# The task uses record types ``1d_x`` / ``1d_y`` / ``2d``; ClustOpt's internal
# loader/view-builder uses ``x_only`` / ``y_only`` / ``xy_2d``. Keep both worlds
# in sync through this single mapping.
# ---------------------------------------------------------------------------
RECORD_TYPES: tuple[str, ...] = ("1d_x", "1d_y", "2d")

RECORD_TYPE_TO_VIEW_ID: dict[str, str] = {
    "1d_x": "x_only",
    "1d_y": "y_only",
    "2d": "xy_2d",
}
VIEW_ID_TO_RECORD_TYPE: dict[str, str] = {v: k for k, v in RECORD_TYPE_TO_VIEW_ID.items()}


def record_type_to_view_id(record_type: str) -> str:
    try:
        return RECORD_TYPE_TO_VIEW_ID[record_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown record_type '{record_type}'. Expected one of {RECORD_TYPES}."
        ) from exc


# ---------------------------------------------------------------------------
# Long-path-safe atomic IO.
# ---------------------------------------------------------------------------
def ext(path: str | Path) -> str:
    r"""Return a long-path-safe absolute path string (``\\?\`` prefix on win32)."""
    s = os.path.abspath(str(path))
    if sys.platform == "win32" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    os.makedirs(ext(p), exist_ok=True)
    return p


def path_exists(path: str | Path) -> bool:
    return os.path.exists(ext(path))


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    os.makedirs(ext(path.parent), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(ext(tmp), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=_json_default)
    os.replace(ext(tmp), ext(path))
    return path


def write_text_atomic(path: str | Path, text: str) -> Path:
    path = Path(path)
    os.makedirs(ext(path.parent), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(ext(tmp), "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(ext(tmp), ext(path))
    return path


def write_csv_atomic(path: str | Path, df: pd.DataFrame, *, index: bool = False) -> Path:
    path = Path(path)
    os.makedirs(ext(path.parent), exist_ok=True)
    tmp = str(path) + ".tmp"
    df.to_csv(ext(tmp), index=index)
    os.replace(ext(tmp), ext(path))
    return path


def write_npy_atomic(path: str | Path, array: np.ndarray) -> Path:
    path = Path(path)
    os.makedirs(ext(path.parent), exist_ok=True)
    tmp = str(path) + ".tmp.npy"
    np.save(ext(tmp), np.asarray(array))
    # np.save appends .npy if not present; we already gave .npy, so name matches.
    os.replace(ext(tmp), ext(path))
    return path


def read_json(path: str | Path) -> Optional[dict]:
    """Long-path-safe JSON read; returns None on any error."""
    try:
        with open(ext(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# External AutoML4Clust source location + sys.path wiring.
# ---------------------------------------------------------------------------
def external_automl4clust_src(repo_root: str | Path) -> Path:
    """Path to the external AutoML4Clust ``src`` directory (the import root)."""
    return Path(repo_root) / "external" / "Automl4Clust" / "src"


def external_automl4clust_exists(repo_root: str | Path) -> bool:
    src = external_automl4clust_src(repo_root)
    return (src / "Optimizer" / "Optimizer.py").is_file()


def ensure_automl4clust_on_path(repo_root: str | Path) -> Path:
    """Prepend the external AutoML4Clust ``src`` dir to ``sys.path`` (idempotent).

    AutoML4Clust uses top-level absolute imports (``from Optimizer.Optimizer
    import ...``), so its ``src`` directory must be importable directly.
    Returns the src path so callers can report it in diagnostics.
    """
    src = external_automl4clust_src(repo_root)
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src


def split_dir_name(split_id: int) -> str:
    return f"split_{int(split_id):02d}"


__all__ = [
    "RECORD_TYPES",
    "RECORD_TYPE_TO_VIEW_ID",
    "VIEW_ID_TO_RECORD_TYPE",
    "record_type_to_view_id",
    "ext",
    "ensure_dir",
    "path_exists",
    "write_json_atomic",
    "write_text_atomic",
    "write_csv_atomic",
    "write_npy_atomic",
    "read_json",
    "external_automl4clust_src",
    "external_automl4clust_exists",
    "ensure_automl4clust_on_path",
    "split_dir_name",
]
