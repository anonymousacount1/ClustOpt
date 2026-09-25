"""Registry of available CVI implementations.

To register a new CVI implementation, add an entry here mapping the lowercase
type-key used in configs to its constructor.
"""
from __future__ import annotations

from typing import Callable, Dict

from models.ClustOpt.cluster_validity_indices.cvi_base import BaseCVI
from models.ClustOpt.cluster_validity_indices.cvi_implementations.cvi_generic import (
    GenericCVIEvaluator,
)
from models.ClustOpt.cluster_validity_indices.cvi_implementations.cvi_dynamic import (
    KNNDynamicCVIEvaluator,
    OracleDynamicCVIEvaluator,
    RegressorDynamicCVIEvaluator,
)


CVI_REGISTRY: Dict[str, Callable[..., BaseCVI]] = {
    "generic": GenericCVIEvaluator,
    # Dynamic CVI types resolve a concrete {metric: weight} dict per
    # dataset/view, then delegate to the generic evaluator unchanged.
    "regressor_dynamic": RegressorDynamicCVIEvaluator,
    "oracle_dynamic": OracleDynamicCVIEvaluator,
    "knn_dynamic": KNNDynamicCVIEvaluator,
}
