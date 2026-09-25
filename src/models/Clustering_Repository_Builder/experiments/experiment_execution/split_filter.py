"""Filter discovered dataset folders by their assigned split id.

Loads ``dataset_split_assignments.csv`` and returns the set of ``dataset_id``
values (and dataset dirs) belonging to a requested split id. Datasets are
matched primarily by ``dataset_id`` (the folder name), so the assignment file
stays valid even if the repository is moved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


class SplitAssignmentError(ValueError):
    pass


def load_split_assignments(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise SplitAssignmentError(f"split assignments file not found: {path}")
    df = pd.read_csv(path)
    for col in ("dataset_id", "split_id"):
        if col not in df.columns:
            raise SplitAssignmentError(
                f"split assignments missing column '{col}' (have {list(df.columns)})"
            )
    return df


def dataset_ids_for_split(assignments: pd.DataFrame, split_id: int) -> Set[str]:
    sub = assignments[assignments["split_id"].astype("Int64") == int(split_id)]
    return set(sub["dataset_id"].astype(str).tolist())


def split_id_lookup(assignments: pd.DataFrame) -> Dict[str, int]:
    return {
        str(r["dataset_id"]): int(r["split_id"])
        for _, r in assignments.iterrows()
        if pd.notna(r["split_id"])
    }


def filter_dataset_dirs_by_split(
    dataset_dirs: List[Path],
    assignments: pd.DataFrame,
    split_id: int,
) -> List[Path]:
    """Keep only dataset dirs whose folder name is assigned to ``split_id``."""
    wanted = dataset_ids_for_split(assignments, split_id)
    return [d for d in dataset_dirs if d.name in wanted]
