"""Subfamily-level summary writers (CSV / JSON / Markdown / checkpoint)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .config import UtilityGenerationConfig
from .utility_output_writer import (
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from .utility_record_runner import RecordResult


SUMMARY_CSV_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "dataset_dir",
    "family",
    "subfamily",
    "specific_subfamily",
    "view_id",
    "status",
    "metric_mode",
    "sample_size",
    "map_name",
    "n_configs",
    "n_valid_configs",
    "runtime_sec",
    "fit_predict_sec",
    "cvi_eval_sec",
    "utility_compute_sec",
    "n_metrics",
    "top1_metric",
    "top1_utility",
    "top5_metrics",
    "error_message",
)


FAILURE_CSV_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "dataset_dir",
    "view_id",
    "status",
    "metric_mode",
    "map_name",
    "error_message",
)


def write_summary_csv(records: Iterable[RecordResult], path: Path) -> Path:
    rows = [r.as_summary_row() for r in records]
    df = pd.DataFrame(rows, columns=list(SUMMARY_CSV_COLUMNS))
    return write_csv_atomic(path, df)


def write_failures_csv(failures: Iterable[RecordResult], path: Path) -> Path:
    rows = [
        {col: r.as_summary_row().get(col, "") for col in FAILURE_CSV_COLUMNS}
        for r in failures
    ]
    df = pd.DataFrame(rows, columns=list(FAILURE_CSV_COLUMNS))
    return write_csv_atomic(path, df)


def write_summary_json(
    *,
    path: Path,
    cfg: UtilityGenerationConfig,
    tracker_summary: dict,
    family_id: str | None,
    subfamily_id: str | None,
    output_files: List[str],
) -> Path:
    payload = {
        "subfamily_dir": str(cfg.subfamily_dir),
        "family_id": family_id,
        "subfamily_id": subfamily_id,
        "config": cfg.to_summary_dict(),
        "tracker": tracker_summary,
        "output_files": list(output_files),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return write_json_atomic(path, payload)


def write_summary_md(
    *,
    path: Path,
    cfg: UtilityGenerationConfig,
    tracker_summary: dict,
    family_id: str | None,
    subfamily_id: str | None,
    output_files: List[str],
) -> Path:
    cfg_d = cfg.to_summary_dict()
    lines: list[str] = []
    lines.append(f"# Utility Generation Summary")
    lines.append("")
    lines.append(f"- Subfamily directory: `{cfg.subfamily_dir}`")
    lines.append(f"- Family: `{family_id or ''}`")
    lines.append(f"- Subfamily: `{subfamily_id or ''}`")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for k, v in cfg_d.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Tracker totals")
    lines.append("")
    for k, v in tracker_summary.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Output files")
    for f in output_files:
        lines.append(f"- `{f}`")
    lines.append("")
    return write_text_atomic(path, "\n".join(lines))


def write_checkpoint(
    *,
    path: Path,
    cfg: UtilityGenerationConfig,
    records: Iterable[RecordResult],
) -> Path:
    by_dataset: dict[str, dict[str, str]] = {}
    for r in records:
        by_dataset.setdefault(r.dataset_id, {})[r.view_id] = r.status

    payload = {
        "subfamily_dir": str(cfg.subfamily_dir),
        "metric_mode": cfg.metric_mode,
        "sample_size": int(cfg.fast_metric_sample_size if cfg.metric_mode == "fast" else 0),
        "views": list(cfg.views),
        "datasets": by_dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return write_json_atomic(path, payload)
