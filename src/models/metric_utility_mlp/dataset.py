"""
dataset.py
==========

Thin PyTorch ``Dataset`` wrapping pre-scaled feature and target arrays plus an
optional integer row-index column used to recover metadata during evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class UtilityDataset(Dataset):
    """Returns ``(x, y, row_index)`` tensors.

    ``row_index`` carries the position into the *original* dataframe so that the
    evaluator can join predictions back to metadata (family, view, ...).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, row_index: np.ndarray) -> None:
        if not (len(x) == len(y) == len(row_index)):
            raise ValueError("x, y and row_index must have the same length.")
        self.x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float32))
        self.row_index = torch.as_tensor(np.asarray(row_index, dtype=np.int64))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.row_index[idx]
