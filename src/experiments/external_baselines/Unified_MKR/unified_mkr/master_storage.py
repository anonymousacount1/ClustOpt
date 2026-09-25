"""On-disk layout for the Master Unified Repository.

The master repository is a merged layer built on top of the per-subfamily
outputs (it does NOT move or duplicate them). It holds one canonical copy of
every table plus the readiness/validation reports::

    master/
        schema_version.json
        master_build_config.json
        records.parquet  metafeatures.parquet  configurations.parquet
        metric_registry.parquet  evaluations.parquet  failures.parquet
        reports/      *.md
        validation/   *.json

Labels are NOT copied here; ``evaluations.labels_path`` is rewritten to be
relative to the outputs root (``<family>/<subfamily>/labels/...``) so the
master stays a thin merged layer over the existing label files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import io_utils

MASTER_TABLES = (
    "records",
    "metafeatures",
    "configurations",
    "metric_registry",
    "evaluations",
    "failures",
)


class MasterStorage:
    def __init__(self, master_root: str | Path):
        self.root = Path(master_root)

    # --- directories -------------------------------------------------------
    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def validation_dir(self) -> Path:
        return self.root / "validation"

    def table_path(self, table: str) -> Path:
        return self.root / f"{table}.parquet"

    # --- setup -------------------------------------------------------------
    def ensure_dirs(self) -> None:
        io_utils.ensure_dir(self.root)
        io_utils.ensure_dir(self.reports_dir)
        io_utils.ensure_dir(self.validation_dir)

    # --- writers -----------------------------------------------------------
    def write_table(self, table: str, df: pd.DataFrame) -> Path:
        return io_utils.write_parquet_atomic(self.table_path(table), df)

    def read_table(self, table: str) -> Optional[pd.DataFrame]:
        return io_utils.read_parquet(self.table_path(table))

    def write_json(self, name: str, payload: dict) -> Path:
        return io_utils.write_json_atomic(self.root / name, payload)

    def write_validation(self, name: str, payload: dict) -> Path:
        io_utils.ensure_dir(self.validation_dir)
        return io_utils.write_json_atomic(self.validation_dir / name, payload)

    def write_report(self, name: str, text: str) -> Path:
        io_utils.ensure_dir(self.reports_dir)
        return io_utils.write_text_atomic(self.reports_dir / name, text)
