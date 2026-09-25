"""Load one dataset folder produced by HYBRID_SCM into a `LoadedDataset`.

The loader exposes labels and structural ground truth as AUDIT artifacts only.
Feature blocks must never read those fields — they are stored on the dataclass
purely so the runner can fill audit columns and detect availability flags.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils.image_utils import load_image_gray


# Allowed unlabeled raster artifacts (matches plan §5).
ALLOWED_RENDERING_FILES: tuple[str, ...] = (
    "grayscale.png",
    "binary_mask.png",
    "distance_transform.png",
    "edge_map.png",
    "gradient_orientation_map.png",
    "local_thickness_map.png",
    "skeleton_mask.png",
)

# Strictly forbidden as X — loaded only via separate audit paths.
FORBIDDEN_RENDERING_FILES: tuple[str, ...] = (
    "cluster_id_raster.npy",
)


@dataclass
class LoadedDataset:
    dataset_dir: Path
    dataset_id: str
    data_df: pd.DataFrame
    X_original: np.ndarray
    labels: Optional[np.ndarray]
    feature_names: list[str]
    metadata: dict
    replay_config: dict
    structural_ground_truth: Optional[dict]
    render_summary: Optional[dict]
    rendering_images: dict[str, np.ndarray] = field(default_factory=dict)

    # Derived flags / audit helpers.
    family_id: Optional[str] = None
    family_name: Optional[str] = None
    subfamily_id: Optional[str] = None
    difficulty: Optional[str] = None
    cluster_count_from_generator: Optional[int] = None
    dataset_seed: Optional[int] = None
    family_seed: Optional[int] = None

    @property
    def n_points(self) -> int:
        return int(self.X_original.shape[0])

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    @property
    def has_rendering(self) -> bool:
        return bool(self.rendering_images) or self.render_summary is not None

    @property
    def has_structural_ground_truth(self) -> bool:
        return self.structural_ground_truth is not None


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _find_data_csv(dataset_dir: Path) -> Optional[Path]:
    """Locate the primary data CSV. Prefer `data_data.csv`, then any
    `*_data.csv`, then a single CSV in the folder.
    """
    primary = dataset_dir / "data_data.csv"
    if primary.is_file():
        return primary
    candidates = sorted(dataset_dir.glob("*_data.csv"))
    if candidates:
        return candidates[0]
    csvs = sorted(dataset_dir.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    for c in csvs:
        if "label" not in c.stem.lower():
            return c
    return None


def _split_label_column(df: pd.DataFrame, metadata: Optional[dict]) -> tuple[pd.DataFrame, Optional[np.ndarray], list[str]]:
    feat_names_meta = None
    if metadata and isinstance(metadata.get("feature_names"), list):
        feat_names_meta = [str(n) for n in metadata["feature_names"]]

    cols = list(df.columns)
    label_col = None
    for candidate in ("cluster_id", "label", "labels", "y_true"):
        if candidate in cols:
            label_col = candidate
            break

    if feat_names_meta is not None:
        feature_cols = [c for c in feat_names_meta if c in cols]
        if not feature_cols:
            # Metadata named features that aren't in the CSV — fall back to all non-label columns.
            feature_cols = [c for c in cols if c != label_col]
    else:
        feature_cols = [c for c in cols if c != label_col]

    labels = df[label_col].to_numpy() if label_col is not None else None
    return df[feature_cols], labels, feature_cols


def load_rendering(dataset_dir: Path) -> tuple[dict[str, np.ndarray], Optional[dict]]:
    rendering_dir = dataset_dir / "rendering"
    if not rendering_dir.is_dir():
        return {}, None

    summary = _read_json(rendering_dir / "render_summary.json")
    images: dict[str, np.ndarray] = {}
    for fname in ALLOWED_RENDERING_FILES:
        arr = load_image_gray(rendering_dir / fname)
        if arr is not None:
            images[fname] = arr
    return images, summary


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: object) -> Optional[str]:
    return None if value is None else str(value)


def _infer_family_and_subfamily(dataset_dir: Path) -> tuple[Optional[str], Optional[str]]:
    """Walk up: dataset_dir / subfamilies / <subfamily_id> / <family_id> / ..."""
    parts = dataset_dir.resolve().parts
    if "subfamilies" not in parts:
        return None, None
    idx = parts.index("subfamilies")
    subfamily_id = parts[idx + 1] if idx + 1 < len(parts) else None
    family_id = parts[idx - 1] if idx - 1 >= 0 else None
    return family_id, subfamily_id


def load_dataset(dataset_dir: Path, *, load_rasters: bool = True) -> LoadedDataset:
    dataset_dir = Path(dataset_dir)
    csv_path = _find_data_csv(dataset_dir)
    if csv_path is None:
        raise FileNotFoundError(f"No data CSV found in {dataset_dir}")

    df = pd.read_csv(csv_path)
    metadata = _read_json(dataset_dir / "data_metadata.json") or {}
    replay_config = _read_json(dataset_dir / "data_replay_config.json") or {}
    structural_gt = _read_json(dataset_dir / "data_structural_ground_truth.json")

    feature_df, labels, feature_names = _split_label_column(df, metadata)
    if feature_df.shape[1] == 0:
        raise ValueError(f"Dataset {dataset_dir} has no usable feature columns")
    X = feature_df.to_numpy(dtype=float, na_value=np.nan)

    rendering_images: dict[str, np.ndarray] = {}
    render_summary: Optional[dict] = None
    if load_rasters:
        rendering_images, render_summary = load_rendering(dataset_dir)

    family_id_from_path, subfamily_id_from_path = _infer_family_and_subfamily(dataset_dir)
    tags = metadata.get("tags") if isinstance(metadata.get("tags"), dict) else {}

    def _from_metadata_or_tags(*keys: str) -> object:
        """Return the first non-None value found at any of these keys, checking
        top-level metadata first and then `metadata['tags']` as a fallback."""
        for key in keys:
            v = metadata.get(key)
            if v is not None:
                return v
        for key in keys:
            v = tags.get(key) if isinstance(tags, dict) else None
            if v is not None:
                return v
        return None

    family_id = _coerce_str(_from_metadata_or_tags("family_id")) or family_id_from_path
    family_name = _coerce_str(_from_metadata_or_tags("family_name")) or family_id
    subfamily_id = _coerce_str(_from_metadata_or_tags("subfamily_id")) or subfamily_id_from_path

    difficulty = _coerce_str(_from_metadata_or_tags("difficulty"))
    cluster_count = _coerce_int(_from_metadata_or_tags("cluster_count", "n_clusters"))
    dataset_seed = _coerce_int(_from_metadata_or_tags("dataset_seed", "seed"))
    family_seed = _coerce_int(_from_metadata_or_tags("family_seed"))

    if structural_gt is not None and cluster_count is None:
        if "clusters" in structural_gt and isinstance(structural_gt["clusters"], list):
            cluster_count = len(structural_gt["clusters"])
        elif "n_clusters" in structural_gt:
            cluster_count = _coerce_int(structural_gt["n_clusters"])

    return LoadedDataset(
        dataset_dir=dataset_dir,
        dataset_id=dataset_dir.name,
        data_df=feature_df,
        X_original=X,
        labels=None if labels is None else np.asarray(labels),
        feature_names=feature_names,
        metadata=metadata,
        replay_config=replay_config,
        structural_ground_truth=structural_gt,
        render_summary=render_summary,
        rendering_images=rendering_images,
        family_id=family_id,
        family_name=family_name,
        subfamily_id=subfamily_id,
        difficulty=difficulty,
        cluster_count_from_generator=cluster_count,
        dataset_seed=dataset_seed,
        family_seed=family_seed,
    )
