from __future__ import annotations

from typing import Any, Dict
import numpy as np


def sample_from_spec(rng: np.random.Generator, spec: Any) -> Any:
    """
    Sample a value from a prior spec, or return as-is if already concrete.

    Supported:
      {"dist":"uniform","low":-1,"high":1}
      {"dist":"normal","mu":0,"sigma":1}
      {"dist":"loguniform","low":1e-3,"high":1e-1}
      {"dist":"choice","values":[...]}
    """
    if isinstance(spec, dict) and "dist" in spec:
        dist = str(spec["dist"]).lower()

        if dist == "uniform":
            return float(rng.uniform(float(spec["low"]), float(spec["high"])))

        if dist == "normal":
            return float(rng.normal(float(spec.get("mu", 0.0)), float(spec.get("sigma", 1.0))))

        if dist == "loguniform":
            low = float(spec["low"])
            high = float(spec["high"])
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))

        if dist == "choice":
            vals = list(spec["values"])
            return vals[int(rng.integers(0, len(vals)))]

        raise ValueError(f"Unknown dist in prior spec: {dist}")

    return spec


def sample_param_dict(rng: np.random.Generator, params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: sample_from_spec(rng, v) for k, v in (params or {}).items()}
