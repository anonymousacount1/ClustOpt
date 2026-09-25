"""Atomic, long-path-safe output writers for experiment results.

The experiment output tree
(``<dataset>/experiments/split_XX/<method>/<view>/<file>``) is deep enough to
exceed the legacy Windows ``MAX_PATH`` (260) limit for the longer method names,
and this machine has ``LongPathsEnabled=0``. To stay robust we route every file
operation through the extended-length ``\\?\`` prefix on Windows (a no-op on
other platforms).

Writes are atomic (temp file + ``os.replace``) so a crash mid-write never
leaves a partial file that resume could mistake for a completed run. This is a
standalone implementation; the utility-generation writer is left untouched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _ext(path: str | Path) -> str:
    """Return a long-path-safe absolute path string.

    On Windows, prefixes the extended-length ``\\?\`` marker (requires an
    absolute, backslash-separated, normalised path). Elsewhere returns the
    plain absolute path.
    """
    s = os.path.abspath(str(path))
    if sys.platform == "win32" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    os.makedirs(_ext(p), exist_ok=True)
    return p


def _json_default(value: Any):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    os.makedirs(_ext(path.parent), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(_ext(tmp), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=_json_default)
    os.replace(_ext(tmp), _ext(path))
    return path


def write_csv_atomic(path: str | Path, df: pd.DataFrame, *, index: bool = False) -> Path:
    path = Path(path)
    os.makedirs(_ext(path.parent), exist_ok=True)
    tmp = str(path) + ".tmp"
    df.to_csv(_ext(tmp), index=index)
    os.replace(_ext(tmp), _ext(path))
    return path


def write_text_atomic(path: str | Path, text: str) -> Path:
    path = Path(path)
    os.makedirs(_ext(path.parent), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(_ext(tmp), "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(_ext(tmp), _ext(path))
    return path


def read_json(path: str | Path) -> Optional[dict]:
    """Long-path-safe JSON read; returns None on any error."""
    try:
        with open(_ext(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def path_exists(path: str | Path) -> bool:
    return os.path.exists(_ext(path))


__all__ = [
    "ensure_dir", "write_json_atomic", "write_csv_atomic", "write_text_atomic",
    "read_json", "path_exists",
]
