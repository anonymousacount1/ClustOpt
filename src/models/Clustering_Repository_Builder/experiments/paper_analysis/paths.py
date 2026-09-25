r"""Long-path-safe filesystem + IO helpers.

Every path in this analysis routinely exceeds Windows ``MAX_PATH`` (260): the
per-dataset run directories are ~270 chars before a filename is appended, and
long paths are **disabled** on the target machine. So *all* IO in this package
goes through :func:`ext` (the ``\\?\`` extended-length prefix).

This module is deliberately dependency-light (pandas only) so it can be
imported by the registries without pulling in matplotlib/scipy.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

__all__ = [
    "ext", "mkdirs", "exists", "isdir", "size_bytes", "listdir",
    "read_json", "write_json", "write_text", "read_text",
    "write_csv", "read_csv", "write_parquet", "read_table",
    "repo_root", "split_dir_name",
]


# --------------------------------------------------------------------------- #
# Path primitives
# --------------------------------------------------------------------------- #
def ext(path: str | Path) -> str:
    r"""Absolute path string, ``\\?\``-prefixed on Windows.

    Idempotent: an already-prefixed path is returned unchanged.
    """
    s = os.path.abspath(str(path))
    if sys.platform == "win32" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def mkdirs(path: str | Path) -> None:
    os.makedirs(ext(path), exist_ok=True)


def exists(path: str | Path) -> bool:
    return os.path.exists(ext(path))


def isdir(path: str | Path) -> bool:
    return os.path.isdir(ext(path))


def size_bytes(path: str | Path) -> int:
    try:
        return int(os.path.getsize(ext(path)))
    except OSError:
        return 0


def listdir(path: str | Path) -> List[str]:
    """``os.listdir`` that returns ``[]`` instead of raising."""
    try:
        return sorted(os.listdir(ext(path)))
    except OSError:
        return []


def repo_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[4]


def split_dir_name(split_id: int, suffix: str = "") -> str:
    base = f"split_{int(split_id):02d}"
    return f"{base}_{suffix}" if suffix else base


# --------------------------------------------------------------------------- #
# JSON / text
# --------------------------------------------------------------------------- #
def read_json(path: str | Path) -> Optional[Any]:
    """Parse a JSON file, returning ``None`` on any failure (missing/corrupt)."""
    try:
        with open(ext(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    mkdirs(Path(path).parent)
    with open(ext(path), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=_json_default,
                  ensure_ascii=False, sort_keys=False)


def _json_default(obj: Any) -> Any:
    """Make numpy / pandas scalars and Paths JSON-serialisable."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):          # numpy scalar
        try:
            return obj.item()
        except Exception:             # pragma: no cover
            pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def write_text(path: str | Path, text: str) -> None:
    mkdirs(Path(path).parent)
    with open(ext(path), "w", encoding="utf-8") as fh:
        fh.write(text)


def read_text(path: str | Path) -> Optional[str]:
    try:
        with open(ext(path), "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    write_text(path, "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Tabular
# --------------------------------------------------------------------------- #
def write_csv(path: str | Path, df: Optional[pd.DataFrame], *,
              index: Optional[bool] = None) -> Optional[Path]:
    """Long-path-safe CSV write. ``None``/empty frames still write a header row.

    Returns the written path (or ``None`` if ``df`` is ``None``), so callers can
    register the artifact in the manifest.
    """
    if df is None:
        return None
    p = Path(path)
    mkdirs(p.parent)
    if index is None:
        index = isinstance(df.index, pd.MultiIndex) or df.index.name is not None
    df.to_csv(ext(p), index=bool(index))
    return p


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ext(path), **kwargs)


def write_parquet(path: str | Path, df: pd.DataFrame) -> Optional[Path]:
    """Parquet write; falls back to CSV (same stem) if pyarrow is unavailable.

    Returns the path actually written.
    """
    p = Path(path)
    mkdirs(p.parent)
    try:
        df.to_parquet(ext(p), index=False)
        return p
    except Exception:                                    # pragma: no cover
        alt = p.with_suffix(".csv")
        df.to_csv(ext(alt), index=False)
        return alt


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a ``.parquet`` or ``.csv`` canonical frame by extension."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(ext(p))
    return pd.read_csv(ext(p), **kwargs)


def json_dumps(payload: Any, *, indent: int = 2) -> str:
    return json.dumps(payload, indent=indent, default=_json_default,
                      ensure_ascii=False)


def as_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame -> list of plain dicts with NaN mapped to ``None``."""
    return df.where(pd.notna(df), None).to_dict(orient="records")
