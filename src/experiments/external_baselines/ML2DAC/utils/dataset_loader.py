"""Dataset loading for the ML2DAC baseline.

Reuses the AutoML4Clust loader (which itself reuses the canonical ClustOpt
loader + view builder), so ML2DAC sees *exactly* the same partition/evaluation
spaces ClustOpt and AutoML4Clust evaluated. Record-type -> space mapping::

    1d_x: partition = X[:, [0]]      evaluation = X[:, [0, 1]]
    1d_y: partition = X[:, [1]]      evaluation = X[:, [0, 1]]
    2d:   partition = X[:, [0, 1]]   evaluation = X[:, [0, 1]]

true_k / y_true are for external evaluation only and are never passed to ML2DAC.
"""
from __future__ import annotations

from experiments.external_baselines.AutoML4Clust.utils.dataset_loader import (  # noqa: F401
    LoadedAutoML4ClustDataset as LoadedDataset,
    load_dataset,
)

__all__ = ["LoadedDataset", "load_dataset"]
