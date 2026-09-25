"""Small atomic writers for split artifacts (JSON / CSV / text).

Mirrors the utility-generation atomic-write pattern: write to a sibling
``*.tmp`` file then ``os.replace`` it over the destination so a crash partway
through a write never leaves a partial file behind.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
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
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=_json_default)
    os.replace(tmp, path)
    return path


def write_csv_atomic(path: str | Path, df: pd.DataFrame, *, index: bool = False) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)
    return path


def write_text_atomic(path: str | Path, text: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return path
