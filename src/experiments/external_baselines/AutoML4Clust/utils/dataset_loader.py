"""Dataset loading for the AutoML4Clust baseline.

Reuses the canonical ClustOpt loader + view builder
(``models/Clustering_Repository_Builder/utility_generation``) so AutoML4Clust
sees *exactly* the same partition/evaluation spaces ClustOpt evaluated -- the
critical fairness guarantee. A self-contained fallback (reading ``data_data.csv``
directly) is provided so the baseline still loads datasets even if that package
is reorganised.

Record-type -> space mapping (identical to ClustOpt's view convention)::

    record_type = 1d_x:  partition = X[:, [0]]      evaluation = X[:, [0, 1]]
    record_type = 1d_y:  partition = X[:, [1]]      evaluation = X[:, [0, 1]]
    record_type = 2d:    partition = X[:, [0, 1]]   evaluation = X[:, [0, 1]]

The model is ALWAYS clustered on ``partition_space`` (record-type dependent),
while ``evaluation_space`` is the full 2-D space (used for reporting / any
geometry-aware metric and kept available for downstream consistency).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .path_utils import record_type_to_view_id

# Try the canonical ClustOpt loader first; fall back to a local reader.
try:  # pragma: no cover - import wiring
    from models.Clustering_Repository_Builder.utility_generation.record_loader import (
        load_dataset_for_utility as _clustopt_load,
    )
    from models.Clustering_Repository_Builder.utility_generation.view_builder import (
        build_view as _clustopt_build_view,
    )
    _HAVE_CLUSTOPT_LOADER = True
except Exception:  # noqa: BLE001
    _clustopt_load = None
    _clustopt_build_view = None
    _HAVE_CLUSTOPT_LOADER = False


@dataclass
class LoadedAutoML4ClustDataset:
    """Everything the AutoML4Clust runner needs for one dataset folder.

    Fairness note: ``true_k`` and ``y_true`` are for *external evaluation only*
    and are never passed to AutoML4Clust.
    """

    dataset_dir: Path
    dataset_id: str
    X: np.ndarray              # full feature matrix (no label column)
    feature_names: list
    y_true: np.ndarray         # ground-truth labels (external eval only)
    X_2d: np.ndarray           # full 2-D evaluation space [x, y]
    family: Optional[str] = None
    subfamily: Optional[str] = None
    difficulty: Optional[str] = None
    true_k: Optional[int] = None    # external eval only
    metadata: Dict[str, Any] = field(default_factory=dict)
    loader_backend: str = "clustopt"

    @property
    def n_points(self) -> int:
        return int(self.X.shape[0])

    def get_record(self, record_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(partition_space, evaluation_space)`` for a record type.

        ``partition_space`` is what AutoML4Clust clusters on; ``evaluation_space``
        is always the full 2-D space.
        """
        view_id = record_type_to_view_id(record_type)
        if _HAVE_CLUSTOPT_LOADER and self._loaded_obj is not None:
            view = _clustopt_build_view(self._loaded_obj, view_id)
            return (
                np.ascontiguousarray(view.X_decision),
                np.ascontiguousarray(view.X_full),
            )
        # Fallback: derive directly from X_2d.
        evaluation = np.ascontiguousarray(self.X_2d)
        if record_type == "1d_x":
            partition = evaluation[:, [0]]
        elif record_type == "1d_y":
            partition = evaluation[:, [1]]
        else:  # 2d
            partition = evaluation
        return np.ascontiguousarray(partition), evaluation

    # Internal handle to the ClustOpt loaded object (set by the loader).
    _loaded_obj: Any = None


def _infer_family_subfamily(dataset_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    parts = dataset_dir.resolve().parts
    if "subfamilies" not in parts:
        return None, None
    idx = parts.index("subfamilies")
    family_id = parts[idx - 1] if idx - 1 >= 0 else None
    subfamily_id = parts[idx + 1] if idx + 1 < len(parts) else None
    return family_id, subfamily_id


def _resolve_xy_columns(feature_names: list, X: np.ndarray) -> Tuple[int, int]:
    cols = [str(c).lower() for c in feature_names]
    if "x" in cols and "y" in cols:
        return cols.index("x"), cols.index("y")
    if X.shape[1] < 2:
        raise ValueError("Dataset has fewer than 2 feature columns; cannot build 2-D space.")
    return 0, 1


def _fallback_load(dataset_dir: Path) -> LoadedAutoML4ClustDataset:
    """Minimal self-contained loader reading ``data_data.csv`` directly."""
    csv_path = dataset_dir / "data_data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing data_data.csv in {dataset_dir}")
    df = pd.read_csv(csv_path)

    label_col = None
    for c in ("cluster_id", "label", "labels", "y_true"):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"No label column (cluster_id/label/labels/y_true) in {csv_path}")
    y_true = df[label_col].to_numpy()

    feature_cols = [c for c in df.columns if c != label_col]
    if not feature_cols:
        raise ValueError(f"No usable feature columns in {csv_path}")
    X = df[feature_cols].to_numpy(dtype=float, na_value=np.nan)

    meta_path = dataset_dir / "data_metadata.json"
    metadata: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    x_idx, y_idx = _resolve_xy_columns(feature_cols, X)
    X_2d = np.asarray(X[:, [x_idx, y_idx]], dtype=float)

    family_path, subfamily_path = _infer_family_subfamily(dataset_dir)
    tags = metadata.get("tags") if isinstance(metadata.get("tags"), dict) else {}
    family = metadata.get("family_id") or family_path
    subfamily = metadata.get("subfamily_id") or subfamily_path
    difficulty = tags.get("difficulty")
    true_k = None
    if isinstance(metadata.get("n_clusters"), int):
        true_k = int(metadata["n_clusters"])
    elif isinstance(tags.get("cluster_count"), int):
        true_k = int(tags["cluster_count"])

    return LoadedAutoML4ClustDataset(
        dataset_dir=dataset_dir,
        dataset_id=dataset_dir.name,
        X=X,
        feature_names=list(feature_cols),
        y_true=np.asarray(y_true),
        X_2d=X_2d,
        family=family,
        subfamily=subfamily,
        difficulty=difficulty,
        true_k=true_k,
        metadata=metadata,
        loader_backend="fallback",
    )


def load_dataset(
    dataset_dir: str | Path,
    *,
    true_k_override: Optional[int] = None,
) -> LoadedAutoML4ClustDataset:
    """Load one dataset folder for AutoML4Clust.

    Parameters
    ----------
    dataset_dir:
        Path to a dataset folder (containing ``data_data.csv`` etc.).
    true_k_override:
        Optional true K (e.g. the ``cluster_count`` column from the split
        assignment CSV). Used only for external evaluation; takes precedence
        over metadata-derived true K when provided.
    """
    dataset_dir = Path(dataset_dir)

    if _HAVE_CLUSTOPT_LOADER and _clustopt_load is not None:
        loaded = _clustopt_load(dataset_dir)
        tags = {}
        if isinstance(loaded.metadata, dict) and isinstance(loaded.metadata.get("tags"), dict):
            tags = loaded.metadata["tags"]
        x_idx, y_idx = _resolve_xy_columns(loaded.feature_names, loaded.X)
        X_2d = np.asarray(loaded.X[:, [x_idx, y_idx]], dtype=float)
        true_k = loaded.cluster_count_from_generator
        if true_k is None:
            true_k = tags.get("cluster_count")
        ds = LoadedAutoML4ClustDataset(
            dataset_dir=Path(loaded.dataset_dir),
            dataset_id=loaded.dataset_id,
            X=np.asarray(loaded.X, dtype=float),
            feature_names=list(loaded.feature_names),
            y_true=np.asarray(loaded.y_true),
            X_2d=X_2d,
            family=loaded.family_name or loaded.family_id,
            subfamily=loaded.subfamily_id,
            difficulty=tags.get("difficulty"),
            true_k=int(true_k) if true_k is not None else None,
            metadata=loaded.metadata if isinstance(loaded.metadata, dict) else {},
            loader_backend="clustopt",
        )
        ds._loaded_obj = loaded
    else:
        ds = _fallback_load(dataset_dir)

    if true_k_override is not None:
        ds.true_k = int(true_k_override)
    return ds


__all__ = ["LoadedAutoML4ClustDataset", "load_dataset"]
