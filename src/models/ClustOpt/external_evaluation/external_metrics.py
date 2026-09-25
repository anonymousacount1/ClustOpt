"""External (ground-truth based) evaluator for ClustOpt search runs.

Currently computes Adjusted Rand Index (ARI). The evaluator is attached at
runtime via :meth:`BaseSearchAlgorithm.set_ground_truth`, so the rest of
the pipeline keeps running unchanged when no ground truth is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import adjusted_rand_score


@dataclass
class ExternalEvaluationConfig:
    """Knobs that control how external metrics interpret noise labels."""

    noise_label: int = -1
    ignore_noise: bool = False


class ExternalEvaluator:
    """Compute external scores against a fixed ground truth vector."""

    def __init__(self, y_true: np.ndarray, config: Optional[ExternalEvaluationConfig] = None):
        if y_true is None:
            raise ValueError("y_true must not be None.")
        self.y_true = np.asarray(y_true).reshape(-1)
        if self.y_true.size == 0:
            raise ValueError("y_true must contain at least one element.")
        self.config = config or ExternalEvaluationConfig()

    def compute(self, labels_pred: Any) -> Dict[str, float]:
        if labels_pred is None:
            return {"ARI": float("nan")}

        y_pred = np.asarray(labels_pred).reshape(-1)
        if y_pred.size == 0:
            return {"ARI": float("nan")}
        if y_pred.shape[0] != self.y_true.shape[0]:
            raise ValueError(
                f"labels_pred length ({y_pred.shape[0]}) != y_true length ({self.y_true.shape[0]})."
            )

        if self.config.ignore_noise:
            mask = y_pred != self.config.noise_label
            if int(np.sum(mask)) < 2:
                return {"ARI": float("nan")}
            ari = adjusted_rand_score(self.y_true[mask], y_pred[mask])
        else:
            ari = adjusted_rand_score(self.y_true, y_pred)

        return {"ARI": float(ari)}
