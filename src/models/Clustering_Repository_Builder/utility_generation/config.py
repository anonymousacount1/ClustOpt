"""Configuration dataclasses and constants for utility generation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# The three views the pipeline produces, in fixed execution order.
VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")

# Map view_id -> utility-map file name (resolved relative to repo-root /
# models/ClustOpt/configs/utilities_maps/).
UTILITY_MAP_FILES: dict[str, str] = {
    "x_only": "utility_map_x_only_32.json",
    "y_only": "utility_map_y_only_32.json",
    "xy_2d": "utility_map_xy_2d_32.json",
}


def _default_max_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(cpu - 1, 8))


@dataclass
class UtilityGenerationConfig:
    """Resolved CLI options for one subfamily utility-generation run."""

    repo_root: Path
    subfamily_dir: Path

    metric_mode: str = "fast"
    fast_metric_sample_size: int = 3000
    fast_metric_random_state: int = 42

    max_workers: int = field(default_factory=_default_max_workers)
    resume: bool = True
    overwrite: bool = False
    dry_run: bool = False

    limit_datasets: int | None = None
    dataset_id_filter: str | None = None
    views: Tuple[str, ...] = VIEW_IDS

    progress_every: int = 5

    # ------------------------------------------------------------------ helpers
    def utility_maps_root(self) -> Path:
        return self.repo_root / "models" / "ClustOpt" / "configs" / "utilities_maps"

    def map_path_for_view(self, view_id: str) -> Path:
        return self.utility_maps_root() / UTILITY_MAP_FILES[view_id]

    def runtime_block(self) -> dict[str, object]:
        """Runtime block to inject into ClustOpt builder."""
        block: dict[str, object] = {"metric_mode": self.metric_mode}
        if self.metric_mode == "fast":
            block["fast_metric_sample_size"] = int(self.fast_metric_sample_size)
            block["fast_metric_random_state"] = int(self.fast_metric_random_state)
        return block

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repo_root": str(self.repo_root),
            "subfamily_dir": str(self.subfamily_dir),
            "metric_mode": self.metric_mode,
            "fast_metric_sample_size": int(self.fast_metric_sample_size),
            "fast_metric_random_state": int(self.fast_metric_random_state),
            "max_workers": int(self.max_workers),
            "resume": bool(self.resume),
            "overwrite": bool(self.overwrite),
            "dry_run": bool(self.dry_run),
            "limit_datasets": self.limit_datasets,
            "dataset_id_filter": self.dataset_id_filter,
            "views": list(self.views),
        }
