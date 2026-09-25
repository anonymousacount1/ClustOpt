"""Atomic file writers for utility-generation outputs.

Writes are performed to a sibling ``*.tmp`` file first and then renamed to
the final destination. Resume mode treats only the final file as a real
completion marker, so a crash partway through a write cannot fool the
checkpoint logic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _atomic_rename(src: Path, dst: Path) -> None:
    """Rename ``src`` over ``dst`` atomically (on the same filesystem).

    On Windows, ``os.replace`` is the recommended atomic replace and works
    even when ``dst`` already exists.
    """
    os.replace(src, dst)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=_json_default)
    _atomic_rename(tmp, path)
    return path


def write_csv_atomic(path: Path, df: pd.DataFrame, *, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=index)
    _atomic_rename(tmp, path)
    return path


def write_text_atomic(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
    _atomic_rename(tmp, path)
    return path


def _json_default(value: Any):
    # Make numpy / pandas / pathlib payloads JSON-serialisable.
    try:
        import numpy as np  # local import keeps the writer dep-light
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)
