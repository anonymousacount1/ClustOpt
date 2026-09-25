"""Centralized constants and defaults for feature extraction."""
from __future__ import annotations

from dataclasses import dataclass

EPS: float = 1e-12

# Pairwise sampling caps for G block.
G_MAX_POINTS_FOR_PAIRWISE: int = 2000
G_MAX_PAIRWISE_PAIRS: int = 50_000
G_KNN_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
G_LOCAL_DENSITY_K: int = 10

# Density grid bin counts shared by A/B.
A_HIST_BINS_LIST: tuple[int, ...] = (32, 64)
B_DENSITY_GRID_BINS: tuple[int, ...] = (16, 32)
A_ENTROPY_PEAK_BINS: int = 32

# Landmarking caps.
L_MAX_POINTS_FOR_LANDMARKING: int = 2000
L_KMEANS_K_VALUES: tuple[int, ...] = (2, 3, 4, 5)
L_GMM_K_VALUES: tuple[int, ...] = (2, 3, 4, 5)
L_AGGLO_K_VALUES: tuple[int, ...] = (2, 3, 4, 5)

# Image processing knobs.
V_BINARY_THRESHOLD: int = 128
V_ORIENTATION_BINS: int = 16
V_BOX_COUNTING_SCALES: tuple[int, ...] = (2, 4, 8, 16, 32)


@dataclass(frozen=True)
class RunConfig:
    """Resolved CLI options for one extraction run."""

    subfamily_root: str
    overwrite: bool = False
    skip_landmarking: bool = False
    max_datasets: int | None = None
    progress_every: int = 10
    workers: int = 1
    write_parquet: bool = True
    random_seed: int = 0
