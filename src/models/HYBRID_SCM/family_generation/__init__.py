"""Family-level repository generation for HYBRID_SCM.

This subpackage adds family-level raw dataset generation on top of the
existing HYBRID_SCM model. It does not modify any existing flows: the
single-config, bundle, and replay runners in ``run_generate.py`` keep
working untouched.

Public entry point:

    python -m models.HYBRID_SCM.run_family_repository_generation \\
        --family-config models/HYBRID_SCM/configs/repository_generation/families/<family>.json
"""

from .planning import (
    FamilyPlan,
    FamilySubfamilyPlan,
    DatasetJob,
    load_family_plan,
    expand_jobs,
    split_difficulty,
    derive_dataset_seed,
)
from .difficulty import (
    DIFFICULTIES,
    difficulty_component_noise_sigma,
    difficulty_dataset_perturbations,
    difficulty_global_difficulty,
    sample_total_points,
)
from .dataset_builder import build_dataset_config

__all__ = [
    "FamilyPlan",
    "FamilySubfamilyPlan",
    "DatasetJob",
    "load_family_plan",
    "expand_jobs",
    "split_difficulty",
    "derive_dataset_seed",
    "DIFFICULTIES",
    "difficulty_component_noise_sigma",
    "difficulty_dataset_perturbations",
    "difficulty_global_difficulty",
    "sample_total_points",
    "build_dataset_config",
]
