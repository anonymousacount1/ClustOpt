"""Method adapter contract.

Adapters ADAPT frozen existing implementations; they must not re-implement any
scientific component. Every adapter loads its artifacts in ``initialise``
(measured as MODEL_INIT_RUNTIME) and does all inference in ``run`` (measured as
ONLINE_RUNTIME).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class MethodResult:
    labels: np.ndarray
    selected_algorithm: Optional[str]
    best_configuration: Dict[str, Any]
    n_evaluations: int
    extra: Dict[str, Any] = field(default_factory=dict)


class MethodAdapter:
    """One external method variant. Receives X only -- never labels, never K."""

    method_id: str = ""
    framework: str = ""

    def initialise(self) -> Dict[str, str]:
        """Load frozen artifacts. Returns {artifact_name: sha256_16}."""
        raise NotImplementedError

    def run(self, X: np.ndarray, *, seed: int, k_min: int, k_max: int,
            budget: int) -> MethodResult:
        raise NotImplementedError
