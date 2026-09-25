"""Label access gate.

Labels may be read only after a prediction artifact exists on disk. This module
is the single door to ground truth; execution code must never import it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class PredictionsNotPersisted(RuntimeError):
    """Raised when offline scoring is attempted too early."""


def load_labels_for_scoring(labels_path: Path, predictions_path: Path) -> np.ndarray:
    """Hand over labels only once predictions are persisted."""
    if not Path(predictions_path).exists():
        raise PredictionsNotPersisted(
            "offline scoring attempted before predictions were persisted: %s"
            % predictions_path)
    return np.load(str(labels_path))
