"""THIN Stage-1C entry point -- delegates to the production experiment framework.

This script deliberately contains NO scientific logic. Stage-1B-1 originally
implemented the offline replay here; during Stage-1C-A that responsibility was
refactored into the repository's established experiment-execution architecture:

    objective / weighting   models/ClustOpt/.../dynamic_metric_selection (production resolvers)
    offline scoring         experiment_execution/offline_candidate_execution.py
    persistence             experiment_execution/offline_method_runner.py
    worker unit             experiment_execution/offline_dataset_worker.py
    one subfamily           experiment_execution/run_offline_subfamily.py
    serial sweep            experiment_execution/run_offline_2x3_all_subfamilies.py
    aggregation             experiment_execution/summary_writer.py (shared with online)
    statistics              paper_analysis/stats.py

All this file does is forward arguments.

Examples
--------
Smoke (NOT reportable)::

    .venv_clustopt/Scripts/python.exe scripts/stage1/run_offline_2x3.py \
        --only-subfamily arcs_circles_rings/annuli_variable_thickness \
        --limit-datasets 2 --workers 1 \
        --head14-model-run-dir results_analysis/mlp/<ts>__metric_utility_mlp_head14_split1_holdout

Full Stage-1C-B sweep (requires explicit approval)::

    .venv_clustopt/Scripts/python.exe scripts/stage1/run_offline_2x3.py \
        --workers 8 --head14-model-run-dir <head14 run dir>
"""
from __future__ import annotations

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.Clustering_Repository_Builder.experiments.experiment_execution import (  # noqa: E402
    run_offline_2x3_all_subfamilies as sweep,
)


def main() -> int:
    argv = list(sys.argv[1:])
    if not any(a == "--repo-root" or a.startswith("--repo-root=") for a in argv):
        argv = ["--repo-root", REPO] + argv
    return sweep.main(argv)


if __name__ == "__main__":
    sys.exit(main())
