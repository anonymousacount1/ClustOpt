"""Discover and validate dataset folders inside a subfamily directory."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


REQUIRED_DATASET_FILES: tuple[str, ...] = (
    "data_data.csv",
    "data_metadata.json",
    "data_replay_config.json",
    "data_structural_ground_truth.json",
)

REQUIRED_FEATURE_FILES: tuple[str, ...] = (
    "features/features_records.csv",
)


@dataclass
class DiscoveredDataset:
    dataset_dir: Path
    dataset_id: str
    missing_files: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.missing_files


def _check_files(dataset_dir: Path) -> list[str]:
    missing: list[str] = []
    for rel in REQUIRED_DATASET_FILES + REQUIRED_FEATURE_FILES:
        if not (dataset_dir / rel).is_file():
            missing.append(rel)
    return missing


def discover_datasets(
    subfamily_dir: Path,
    *,
    limit: Optional[int] = None,
    dataset_id_filter: Optional[str] = None,
) -> tuple[List[DiscoveredDataset], List[DiscoveredDataset]]:
    """Return ``(valid_datasets, invalid_datasets)``.

    A dataset folder is considered "valid" if all required files are present.
    Non-dataset entries (entries lacking ``data_data.csv``) are silently
    ignored — they are not returned as invalid; only directories that look
    like they should be datasets but are incomplete are listed.
    """
    if not subfamily_dir.is_dir():
        raise FileNotFoundError(f"Subfamily directory does not exist: {subfamily_dir}")

    valid: list[DiscoveredDataset] = []
    invalid: list[DiscoveredDataset] = []

    for entry in sorted(subfamily_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Heuristic: any subfolder containing data_data.csv counts as a dataset attempt.
        if not (entry / "data_data.csv").is_file():
            continue
        if dataset_id_filter and dataset_id_filter not in entry.name:
            continue
        missing = _check_files(entry)
        d = DiscoveredDataset(
            dataset_dir=entry, dataset_id=entry.name, missing_files=missing
        )
        if d.is_valid:
            valid.append(d)
        else:
            invalid.append(d)

    if limit is not None and limit > 0:
        valid = valid[:limit]

    return valid, invalid


def detect_family_subfamily(subfamily_dir: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Walk up the path to extract (family_id, subfamily_id, specific_subfamily).

    Assumes the canonical layout::

        .../analyzed_data/<family>/subfamilies/<subfamily>/[<dataset_dir>/...]

    ``specific_subfamily`` is the last directory segment (same as ``subfamily``
    in the canonical layout but kept separate so callers can override).
    """
    parts = subfamily_dir.resolve().parts
    family_id: Optional[str] = None
    subfamily_id: Optional[str] = None
    if "subfamilies" in parts:
        idx = parts.index("subfamilies")
        if idx + 1 < len(parts):
            subfamily_id = parts[idx + 1]
        if idx - 1 >= 0:
            family_id = parts[idx - 1]
    specific_subfamily = subfamily_dir.name
    return family_id, subfamily_id, specific_subfamily
