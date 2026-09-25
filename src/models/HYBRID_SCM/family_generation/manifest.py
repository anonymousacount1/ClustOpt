"""Manifest, summary, and snapshot writers for a family generation run."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


MANIFEST_COLUMNS: List[str] = [
    "dataset_index",
    "dataset_id",
    "family_id",
    "family_name",
    "subfamily_id",
    "cluster_count",
    "difficulty",
    "family_seed",
    "dataset_seed",
    "n_points_requested",
    "n_points_generated",
    "output_dir",
    "status",
    "error_message",
]


@dataclass
class ManifestRow:
    dataset_index: int
    dataset_id: str
    family_id: str
    family_name: str
    subfamily_id: str
    cluster_count: int
    difficulty: str
    family_seed: int
    dataset_seed: int
    n_points_requested: int
    n_points_generated: Optional[int]
    output_dir: str
    status: str
    error_message: Optional[str] = None

    def as_csv_row(self) -> Dict[str, Any]:
        return {col: getattr(self, col) for col in MANIFEST_COLUMNS}


@dataclass
class SubfamilyReport:
    subfamily_id: str
    planned: int = 0
    generated: int = 0
    failed: int = 0
    by_cluster: Dict[int, Dict[str, int]] = field(default_factory=dict)

    def record(self, cluster_count: int, ok: bool) -> None:
        bucket = self.by_cluster.setdefault(
            int(cluster_count),
            {"planned": 0, "generated": 0, "failed": 0},
        )
        bucket["generated"] += 1 if ok else 0
        bucket["failed"] += 0 if ok else 1
        self.generated += 1 if ok else 0
        self.failed += 0 if ok else 1


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_manifest_csv(rows: List[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            row_dict = r.as_csv_row()
            for k, v in list(row_dict.items()):
                if v is None:
                    row_dict[k] = ""
            writer.writerow(row_dict)


def write_family_snapshot(raw_config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw_config, indent=2, default=_json_default), encoding="utf-8")


def write_summary_json(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


def write_summary_md(
    summary: Dict[str, Any],
    rows: List[ManifestRow],
    subfamily_reports: List[SubfamilyReport],
    path: Path,
) -> None:
    lines: List[str] = []
    lines.append(f"# Family generation report: {summary['family_name']}")
    lines.append("")
    lines.append(f"- **family_id**: `{summary['family_id']}`")
    lines.append(f"- **family_seed**: `{summary['family_seed']}`")
    lines.append(f"- **output_dir**: `{summary['output_dir']}`")
    lines.append(f"- **total_planned**: {summary['total_planned']}")
    lines.append(f"- **total_generated**: {summary['total_generated']}")
    lines.append(f"- **total_failed**: {summary['total_failed']}")
    lines.append(
        f"- **expected_future_repository_rows** (3 views): "
        f"{int(summary['total_generated']) * 3}"
    )
    lines.append("")
    lines.append("## Subfamily summary")
    lines.append("")
    lines.append("| Subfamily | Planned | Generated | Failed |")
    lines.append("|---|---:|---:|---:|")
    for rep in subfamily_reports:
        lines.append(
            f"| `{rep.subfamily_id}` | {rep.planned} | {rep.generated} | {rep.failed} |"
        )
    lines.append("")
    failures = [r for r in rows if r.status != "ok"]
    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.append("| dataset_index | subfamily | clusters | difficulty | error |")
        lines.append("|---:|---|---:|---|---|")
        for r in failures:
            err = (r.error_message or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {r.dataset_index} | `{r.subfamily_id}` | {r.cluster_count} | "
                f"{r.difficulty} | {err} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
