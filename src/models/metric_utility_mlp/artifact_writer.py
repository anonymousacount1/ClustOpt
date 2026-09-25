"""
artifact_writer.py
==================

Centralised, defensive helpers for persisting run artifacts (JSON, CSV, text,
model architecture).  Keeping all writes here makes the directory layout in
:mod:`cross_validation` explicit and consistent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def write_csv(path: str | Path, df: pd.DataFrame, index: bool = False) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
