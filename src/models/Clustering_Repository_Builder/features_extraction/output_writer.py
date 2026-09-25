"""Per-dataset and subfamily-level output writers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .feature_schema import (
    ALL_FEATURE_COLUMNS,
    ALL_RECORD_COLUMNS,
    AUDIT_COLUMN_KINDS,
    AUDIT_COLUMNS,
    ensure_complete_feature_record,
    schema_metadata,
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def _records_to_dataframe(records: Sequence[dict]) -> pd.DataFrame:
    """Build a DataFrame in schema order with **no empty cells**.

    Each record is run through `ensure_complete_feature_record` defensively so
    that even legacy/buggy callers can't leak NaN or None into the output.
    Feature columns are coerced to float (0.0 as a safety net); audit columns
    keep their typed defaults from the schema.
    """
    if not records:
        return pd.DataFrame(columns=ALL_RECORD_COLUMNS)
    filled = [ensure_complete_feature_record(rec) for rec in records]
    df = pd.DataFrame.from_records(filled, columns=ALL_RECORD_COLUMNS)
    # Final safety net — replace any residual nan/None per column type.
    for col in ALL_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    for col, kind in AUDIT_COLUMN_KINDS.items():
        if kind == "string":
            df[col] = df[col].astype(object).where(df[col].notna(), "unknown")
            df[col] = df[col].apply(lambda v: v if (isinstance(v, str) and v.strip()) else "unknown")
        elif kind == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")
        elif kind == "bool":
            df[col] = df[col].fillna(False).astype(bool)
        elif kind == "list":
            df[col] = df[col].apply(lambda v: v if isinstance(v, list) else ([] if v is None else [v]))
    return df


@dataclass
class DatasetWriteResult:
    features_dir: Path
    csv_path: Path
    jsonl_path: Path
    metadata_path: Path
    log_path: Path


def write_dataset_outputs(
    dataset_dir: Path,
    records: Sequence[dict],
    timing: dict,
    *,
    overwrite: bool = False,
) -> DatasetWriteResult:
    features_dir = dataset_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    csv_path = features_dir / "features_records.csv"
    jsonl_path = features_dir / "features_records.jsonl"
    metadata_path = features_dir / "features_metadata.json"
    log_path = features_dir / "feature_extraction_log.json"

    df = _records_to_dataframe(records)
    df.to_csv(csv_path, index=False)

    # JSONL gets the schema-completed (no-empty-cell) records, not the raw input.
    filled_records = [ensure_complete_feature_record(rec) for rec in records]
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in filled_records:
            fh.write(json.dumps(rec, default=_json_default, allow_nan=False) + "\n")

    metadata = {
        "dataset_dir": str(dataset_dir),
        "n_records": len(records),
        "schema": {
            "n_features": schema_metadata()["n_features"],
            "blocks": schema_metadata()["blocks"],
        },
        "views": [r.get("view_mode") for r in records],
    }
    _write_json(metadata_path, metadata)
    _write_json(log_path, timing)
    return DatasetWriteResult(features_dir, csv_path, jsonl_path, metadata_path, log_path)


def write_subfamily_combined_csv(records: Sequence[dict], path: Path) -> Path:
    df = _records_to_dataframe(records)
    df.to_csv(path, index=False)
    return path


def write_subfamily_parquet(records: Sequence[dict], path: Path) -> tuple[bool, str | None]:
    df = _records_to_dataframe(records)
    try:
        df.to_parquet(path, index=False)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def write_failed_csv(failures: Iterable[dict], path: Path) -> Path:
    cols = ["dataset_dir", "dataset_id", "error_stage", "error_message", "traceback_short"]
    rows = list(failures)
    if not rows:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return path
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = "unknown"
    df = df[cols]
    # Fill any empty / NaN cells so the CSV has no blanks.
    df = df.fillna("unknown")
    for c in cols:
        df[c] = df[c].apply(lambda v: v if (isinstance(v, str) and v.strip()) else "unknown")
    df.to_csv(path, index=False)
    return path


def write_schema_file(path: Path) -> Path:
    _write_json(path, schema_metadata())
    return path
