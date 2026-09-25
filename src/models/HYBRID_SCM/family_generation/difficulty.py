"""Difficulty translation utilities.

Maps a difficulty label ("easy" | "medium" | "hard" | "very_hard") into
HYBRID_SCM-supported parameters:

  * a component-level noise sigma (added to each component's Gaussian noise)
  * a dict for ``DatasetConfig.dataset_perturbations``
  * a dict for ``DatasetConfig.global_difficulty``
  * a difficulty-aware total point count drawn from the family policy

Only keys already validated by ``io.config_io`` are produced.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


DIFFICULTIES: Tuple[str, str, str, str] = ("easy", "medium", "hard", "very_hard")


# Component-level Gaussian noise sigma applied via DatasetConfig.components[i].noise_params.
# Values are expressed in the same units as the dataset bounds; defaults are
# tuned for a unit-square ``bounds=[0,1]x[0,1]`` and rescaled by the runner
# proportionally to the actual bounding box.
_COMPONENT_NOISE_SIGMA: Dict[str, float] = {
    "easy": 0.006,
    "medium": 0.014,
    "hard": 0.028,
    "very_hard": 0.045,
}


_DATASET_PERTURBATIONS: Dict[str, Dict[str, Any]] = {
    "easy": {},
    "medium": {
        "point_dropout_rate": 0.03,
        "cluster_size_imbalance": 0.15,
    },
    "hard": {
        "point_dropout_rate": 0.07,
        "cluster_size_imbalance": 0.35,
        "density_imbalance": 0.30,
        "local_contamination_rate": 0.03,
        "local_contamination_sigma": 0.020,
        "duplicate_rate": 0.02,
        "duplicate_jitter_sigma": 0.008,
    },
    "very_hard": {
        "point_dropout_rate": 0.12,
        "cluster_size_imbalance": 0.55,
        "density_imbalance": 0.50,
        "local_contamination_rate": 0.06,
        "local_contamination_sigma": 0.035,
        "duplicate_rate": 0.04,
        "duplicate_jitter_sigma": 0.015,
        "bridge_segments": {"n_bridges": 1, "points_per_bridge": 60, "sigma": 0.012},
    },
}


_GLOBAL_DIFFICULTY: Dict[str, Dict[str, Any]] = {
    "easy": {},
    "medium": {"overlap_jitter_sigma": 0.004},
    "hard": {"overlap_jitter_sigma": 0.010, "label_flip_rate": 0.01},
    "very_hard": {"overlap_jitter_sigma": 0.020, "label_flip_rate": 0.03},
}


_FALLBACK_POINT_RANGES: Dict[str, Tuple[int, int]] = {
    "easy": (2200, 3000),
    "medium": (2600, 3600),
    "hard": (3000, 4200),
    "very_hard": (3500, 5000),
}


def _bounds_scale(bounds: Dict[str, Any] | None) -> float:
    """Scale factor relative to a 1.0-wide unit-square baseline."""
    if not bounds or "low" not in bounds or "high" not in bounds:
        return 1.0
    low = np.asarray(bounds["low"], dtype=float)
    high = np.asarray(bounds["high"], dtype=float)
    if low.size == 0:
        return 1.0
    span = np.maximum(high - low, 1e-9)
    return float(np.mean(span))


def difficulty_component_noise_sigma(
    difficulty: str,
    bounds: Dict[str, Any] | None = None,
) -> float:
    """Component-level Gaussian noise sigma scaled to the dataset bounds."""
    base = _COMPONENT_NOISE_SIGMA[difficulty]
    return float(base * _bounds_scale(bounds))


def difficulty_dataset_perturbations(
    difficulty: str,
    bounds: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Dataset-perturbations dict matching ``VALID_DATASET_PERTURBATION_KEYS``."""
    spec = dict(_DATASET_PERTURBATIONS[difficulty])
    scale = _bounds_scale(bounds)
    if "local_contamination_sigma" in spec:
        spec["local_contamination_sigma"] = float(spec["local_contamination_sigma"] * scale)
    if "duplicate_jitter_sigma" in spec:
        spec["duplicate_jitter_sigma"] = float(spec["duplicate_jitter_sigma"] * scale)
    bridge = spec.get("bridge_segments")
    if isinstance(bridge, dict) and "sigma" in bridge:
        bridge = dict(bridge)
        bridge["sigma"] = float(bridge["sigma"] * scale)
        spec["bridge_segments"] = bridge
    return spec


def difficulty_global_difficulty(
    difficulty: str,
    bounds: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Global-difficulty dict matching ``VALID_GLOBAL_DIFFICULTY_KEYS``."""
    spec = dict(_GLOBAL_DIFFICULTY[difficulty])
    scale = _bounds_scale(bounds)
    if "overlap_jitter_sigma" in spec:
        spec["overlap_jitter_sigma"] = float(spec["overlap_jitter_sigma"] * scale)
    return spec


def sample_total_points(
    rng: np.random.Generator,
    difficulty: str,
    n_points_policy: Dict[str, Any] | None,
) -> int:
    """Draw a total point count for a single raw dataset.

    Honours the per-family ``n_points_policy.difficulty_ranges`` overrides and
    clamps to the global ``min_total_points``/``max_total_points`` envelope.
    """
    policy = dict(n_points_policy or {})
    ranges = dict(policy.get("difficulty_ranges") or {})
    low, high = _FALLBACK_POINT_RANGES[difficulty]
    if difficulty in ranges:
        low, high = int(ranges[difficulty][0]), int(ranges[difficulty][1])
    min_total = int(policy.get("min_total_points", 2000))
    max_total = int(policy.get("max_total_points", 5000))
    low = max(min_total, int(low))
    high = min(max_total, int(high))
    if high <= low:
        return int(low)
    return int(rng.integers(low, high + 1))
