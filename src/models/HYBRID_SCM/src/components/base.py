from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


class Component:
    name: str

    def sample_params(
        self,
        rng: np.random.Generator,
        n_features: int,
        param_spec: Dict[str, Any],
        parent_params: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def sample_points(
        self,
        rng: np.random.Generator,
        n_points: int,
        n_features: int,
        bounds_low: np.ndarray,
        bounds_high: np.ndarray,
        params_used: Dict[str, Any],
        parent_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        raise NotImplementedError


_REGISTRY: Dict[str, Component] = {}


def register(comp: Component) -> None:
    _REGISTRY[comp.name] = comp


def get_component(name: str) -> Component:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown component '{name}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]
