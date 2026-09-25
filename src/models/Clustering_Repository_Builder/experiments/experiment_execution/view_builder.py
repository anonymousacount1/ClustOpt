"""View construction for experiments.

Reuses the utility-generation loader + view builder, which already implement the
critical partition/evaluation separation:

* ``X_decision`` = the partition/search space (1-D for x_only/y_only, 2-D for xy_2d),
* ``X_full``     = the full 2-D evaluation space (always [x, y]),
* ``y_true``     = ground-truth labels.

Re-using these guarantees the experiment pipeline computes geometry/image
metrics on the true 2-D structure even for 1-D partition views.
"""
from __future__ import annotations

from models.Clustering_Repository_Builder.utility_generation.record_loader import (  # noqa: F401
    LoadedDatasetForUtility,
    load_dataset_for_utility,
)
from models.Clustering_Repository_Builder.utility_generation.view_builder import (  # noqa: F401
    ClustOptView,
    build_view,
)

__all__ = [
    "LoadedDatasetForUtility",
    "load_dataset_for_utility",
    "ClustOptView",
    "build_view",
]
