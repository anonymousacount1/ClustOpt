"""On-disk layout, per-dataset staging, and subfamily aggregation.

Each worker writes only its own dataset's staging directory (no shared-state
contention), plus label ``.npy`` files under the subfamily-level ``labels/``
tree (paths are disjoint because ``record_id`` embeds the ``dataset_id``).
The main process then aggregates all staged per-dataset Parquet shards into the
mandatory subfamily-level tables.

Output layout (subfamily root = ``<output_root>/<family>/<subfamily>``)::

    schema_version.json
    build_config.json
    records.parquet  metafeatures.parquet  configurations.parquet
    metric_registry.parquet  evaluations.parquet  failures.parquet
    labels/<record_id>/<config_id>.npy
    logs/build.log
    validation/{integrity_report,coverage_report,cache_report}.json
    _datasets/<dataset_id>/{records,metafeatures,evaluations,failures}.parquet
    _datasets/<dataset_id>/status.json
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import io_utils
from .schemas import SCHEMA_VERSION

_STAGED_TABLES = ("records", "metafeatures", "evaluations", "failures")


@dataclass
class DatasetArtifacts:
    dataset_id: str
    records: pd.DataFrame
    metafeatures: pd.DataFrame
    evaluations: pd.DataFrame
    failures: pd.DataFrame
    status: Dict[str, object]


class UnifiedMKRStorage:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)

    # --- paths -------------------------------------------------------------
    @property
    def labels_dir(self) -> Path:
        return self.out_dir / "labels"

    @property
    def datasets_dir(self) -> Path:
        return self.out_dir / "_datasets"

    @property
    def logs_dir(self) -> Path:
        return self.out_dir / "logs"

    @property
    def validation_dir(self) -> Path:
        return self.out_dir / "validation"

    def dataset_stage_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / dataset_id

    def label_path(self, record_id: str, config_id: str) -> Path:
        return self.labels_dir / record_id / f"{config_id}.npy"

    def label_path_rel(self, record_id: str, config_id: str) -> str:
        return f"labels/{record_id}/{config_id}.npy"

    # --- label IO ----------------------------------------------------------
    def write_labels(self, record_id: str, config_id: str, labels: np.ndarray) -> str:
        path = self.label_path(record_id, config_id)
        io_utils.write_npy_atomic(path, np.asarray(labels))
        return self.label_path_rel(record_id, config_id)

    def labels_exist(self, record_id: str, config_id: str) -> bool:
        return io_utils.path_exists(self.label_path(record_id, config_id))

    def load_labels(self, record_id: str, config_id: str) -> Optional[np.ndarray]:
        path = self.label_path(record_id, config_id)
        try:
            return np.load(io_utils.ext(path))
        except Exception:
            return None

    # --- per-dataset staging ----------------------------------------------
    def stage_dataset(self, art: DatasetArtifacts) -> None:
        stage = self.dataset_stage_dir(art.dataset_id)
        io_utils.ensure_dir(stage)
        io_utils.write_parquet_atomic(stage / "records.parquet", art.records)
        io_utils.write_parquet_atomic(stage / "metafeatures.parquet", art.metafeatures)
        io_utils.write_parquet_atomic(stage / "evaluations.parquet", art.evaluations)
        io_utils.write_parquet_atomic(stage / "failures.parquet", art.failures)
        io_utils.write_json_atomic(stage / "status.json", art.status)

    def read_dataset_status(self, dataset_id: str) -> Optional[Dict[str, object]]:
        return io_utils.read_json(self.dataset_stage_dir(dataset_id) / "status.json")

    def dataset_complete(self, dataset_id: str) -> bool:
        status = self.read_dataset_status(dataset_id)
        return bool(status and status.get("status") == "complete")

    def _read_staged(self, dataset_id: str, table: str) -> Optional[pd.DataFrame]:
        return io_utils.read_parquet(self.dataset_stage_dir(dataset_id) / f"{table}.parquet")

    # --- subfamily aggregation --------------------------------------------
    def aggregate(
        self,
        dataset_ids: List[str],
        *,
        configurations: pd.DataFrame,
        metric_registry: pd.DataFrame,
        build_config: Dict[str, object],
    ) -> Dict[str, int]:
        io_utils.ensure_dir(self.out_dir)
        collected: Dict[str, List[pd.DataFrame]] = {t: [] for t in _STAGED_TABLES}
        for ds_id in dataset_ids:
            for table in _STAGED_TABLES:
                df = self._read_staged(ds_id, table)
                if df is not None and not df.empty:
                    collected[table].append(df)

        counts: Dict[str, int] = {}
        for table in _STAGED_TABLES:
            frames = collected[table]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            io_utils.write_parquet_atomic(self.out_dir / f"{table}.parquet", df)
            counts[table] = int(len(df))

        io_utils.write_parquet_atomic(self.out_dir / "configurations.parquet", configurations)
        io_utils.write_parquet_atomic(self.out_dir / "metric_registry.parquet", metric_registry)
        counts["configurations"] = int(len(configurations))
        counts["metric_registry"] = int(len(metric_registry))

        io_utils.write_json_atomic(
            self.out_dir / "schema_version.json",
            {"schema_version": SCHEMA_VERSION},
        )
        io_utils.write_json_atomic(self.out_dir / "build_config.json", build_config)
        return counts
