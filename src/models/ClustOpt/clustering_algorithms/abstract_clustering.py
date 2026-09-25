"""Abstract clustering-algorithm wrapper.

Concrete subclasses translate a per-parameter config block (``{type, low,
high, choices, ...}``) into a skopt search-space description, and instantiate
a fitted/unfitted estimator from a chosen set of parameters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from skopt.space import Categorical, Integer, Real


class ClusteringAlgorithm(ABC):
    def __init__(self, name: str, param_config: Dict[str, Any]):
        self.name = name
        self.param_config: Dict[str, Any] = dict(param_config or {})

    @abstractmethod
    def instantiate(self, params: Dict[str, Any]):
        """Return an estimator exposing ``fit_predict``."""

    def get_search_space(self) -> Dict[str, Any]:
        space: Dict[str, Any] = {}
        for param_name, definition in self.param_config.items():
            t = definition["type"]
            if t == "int":
                low, high = definition["low"], definition["high"]
                space[param_name] = Categorical([low]) if low == high else Integer(low, high)
            elif t == "float":
                low, high = definition["low"], definition["high"]
                space[param_name] = Categorical([low]) if low == high else Real(low, high)
            elif t == "categorical":
                space[param_name] = Categorical(definition["choices"])
            else:
                raise ValueError(
                    f"Unsupported parameter type '{t}' for parameter "
                    f"'{param_name}' in algorithm '{self.name}'."
                )
        return space
