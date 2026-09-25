"""Load a dataset folder and build its three view records.

Reuses ClustOpt's ``record_loader.load_dataset_for_utility`` +
``view_builder.build_view`` so the partition / evaluation spaces are identical
to those ClustOpt itself uses (``x_only`` / ``y_only`` / ``xy_2d``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .schemas import VIEW_TYPES, make_record_id


@dataclass
class ViewRecord:
    record_id: str
    dataset_id: str
    view_type: str
    X_decision: np.ndarray
    X_full: np.ndarray
    y_true: np.ndarray
    n_samples: int
    n_features: int
    true_k: Optional[int]


@dataclass
class DatasetViews:
    dataset_id: str
    dataset_folder: str
    data_path: str
    metadata_path: str
    true_labels_path: str
    views: Dict[str, ViewRecord]


def _derive_true_k(loaded) -> Optional[int]:
    if loaded.cluster_count_from_generator is not None:
        return int(loaded.cluster_count_from_generator)
    y = np.asarray(loaded.y_true)
    uniq = np.unique(y[y != -1])
    return int(uniq.size) if uniq.size else None


def load_dataset_views(dataset_dir: str | Path) -> DatasetViews:
    from models.Clustering_Repository_Builder.utility_generation.record_loader import (
        load_dataset_for_utility,
    )
    from models.Clustering_Repository_Builder.utility_generation.view_builder import (
        build_view,
    )

    dataset_dir = Path(dataset_dir)
    loaded = load_dataset_for_utility(dataset_dir)
    true_k = _derive_true_k(loaded)

    views: Dict[str, ViewRecord] = {}
    for view_type in VIEW_TYPES:
        try:
            view = build_view(loaded, view_type)
        except Exception:
            continue  # genuinely missing/invalid view
        X_decision = np.asarray(view.X_decision, dtype=float)
        views[view_type] = ViewRecord(
            record_id=make_record_id(loaded.dataset_id, view_type),
            dataset_id=loaded.dataset_id,
            view_type=view_type,
            X_decision=X_decision,
            X_full=np.asarray(view.X_full, dtype=float),
            y_true=np.asarray(view.y_true),
            n_samples=int(X_decision.shape[0]),
            n_features=int(X_decision.shape[1]),
            true_k=true_k,
        )

    return DatasetViews(
        dataset_id=loaded.dataset_id,
        dataset_folder=str(dataset_dir),
        data_path=str(dataset_dir / "data_data.csv"),
        metadata_path=str(dataset_dir / "data_metadata.json"),
        true_labels_path=str(dataset_dir / "data_data.csv"),
        views=views,
    )


def available_view_types(views: DatasetViews) -> List[str]:
    return [v for v in VIEW_TYPES if v in views.views]
