"""Dataset loading for the AutoClust baseline.

Reuses the AutoML4Clust loader (which itself reuses the canonical ClustOpt loader
+ view builder), so AutoClust sees *exactly* the same partition/evaluation spaces
ClustOpt, AutoML4Clust, and ML2DAC evaluated. Record-type -> space map::

    1d_x: partition = X[:, [0]]      evaluation = X[:, [0, 1]]
    1d_y: partition = X[:, [1]]      evaluation = X[:, [0, 1]]
    2d:   partition = X[:, [0, 1]]   evaluation = X[:, [0, 1]]

true_k / y_true are for external evaluation only and are never passed to AutoClust.
"""
from __future__ import annotations

from experiments.external_baselines.AutoML4Clust.utils.dataset_loader import (  # noqa: F401
    LoadedAutoML4ClustDataset as LoadedDataset,
    load_dataset,
)

__all__ = ["LoadedDataset", "load_dataset"]
