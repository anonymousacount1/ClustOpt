"""Build the three view variants per dataset: x_only, y_only, xy_2d."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .dataset_loader import LoadedDataset


@dataclass
class DatasetView:
    view_mode: str
    view_dim: int
    X_view: np.ndarray
    selected_features: list[str]
    X_original: np.ndarray
    original_feature_names: list[str]


def build_views(loaded: LoadedDataset) -> List[DatasetView]:
    feature_names = list(loaded.feature_names)
    X_original = loaded.X_original
    if X_original.ndim != 2 or X_original.shape[1] == 0:
        raise ValueError(f"Dataset {loaded.dataset_dir} has invalid feature matrix shape {X_original.shape}")

    # x is column 0, y is column 1 if present.
    n_dims = X_original.shape[1]
    x_name = feature_names[0]
    y_name = feature_names[1] if n_dims > 1 else None

    x_only_X = X_original[:, [0]]
    views: list[DatasetView] = [
        DatasetView(
            view_mode="x_only",
            view_dim=1,
            X_view=x_only_X,
            selected_features=[x_name],
            X_original=X_original,
            original_feature_names=feature_names,
        )
    ]
    if y_name is not None:
        views.append(
            DatasetView(
                view_mode="y_only",
                view_dim=1,
                X_view=X_original[:, [1]],
                selected_features=[y_name],
                X_original=X_original,
                original_feature_names=feature_names,
            )
        )
        views.append(
            DatasetView(
                view_mode="xy_2d",
                view_dim=2,
                X_view=X_original[:, :2],
                selected_features=[x_name, y_name],
                X_original=X_original,
                original_feature_names=feature_names,
            )
        )
    return views
