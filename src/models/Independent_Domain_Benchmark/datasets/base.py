"""Dataset interfaces. Execution sees X only; labels live behind a separate door."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class ExecutionInput:
    """Everything a method is allowed to see. Deliberately has no label field."""

    dataset_id: str
    source: str
    view: str            # 1d_x | 1d_y | 2d
    X: np.ndarray
    representation_checksum: str


@dataclass(frozen=True)
class EvaluationLabels:
    """Ground truth. Loaded ONLY after predictions have been persisted."""

    dataset_id: str
    y: np.ndarray
    n_classes: int


class DatasetProvider:
    """Contract for the three corpus groups (fcps / sklearn_shapes / real_pca)."""

    def execution_input(self, dataset_id: str, view: str) -> ExecutionInput:
        raise NotImplementedError

    def evaluation_labels(self, dataset_id: str) -> EvaluationLabels:
        raise NotImplementedError

    def metadata(self, dataset_id: str) -> Dict[str, Any]:
        raise NotImplementedError
