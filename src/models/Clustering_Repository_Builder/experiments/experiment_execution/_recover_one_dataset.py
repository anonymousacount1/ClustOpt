"""Surgical recovery: regenerate specific Phase-D methods for ONE dataset,
writing directly into its canonical split_01 folder (split_dir_suffix="").

Used to repair a dataset that lost a few method folders during the split_01
merge. Writes only the requested (method, view) runs; touches no summaries.
"""
from __future__ import annotations

import sys
from pathlib import Path

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (
    ExperimentExecutionConfig, METHODS_PHASED, VIEW_IDS,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.experiment_config_resolver import (
    resolve_configs,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.view_builder import (
    build_view, load_dataset_for_utility,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.method_runner import (
    run_method_view,
)

configure_process()


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    dataset_dir = Path(sys.argv[2]).resolve()
    subfamily_dir = dataset_dir.parent
    methods_wanted = sys.argv[3].split(",")
    csv = (repo / "results_analysis" / "clustering_repository" / "analyzed_data"
           / "experiment_splits" / "dataset_split_assignments.csv")

    cfg = ExperimentExecutionConfig(
        repo_root=repo, subfamily_dir=subfamily_dir, split_assignments=csv,
        split_id=1, split_dir_suffix="", method_set="phaseD", max_workers=1)
    resolved = resolve_configs(
        cfg.experiments_config_root(), methods=cfg.active_registry(), validate=False)
    method_map = {m.name: m for m in METHODS_PHASED}

    loaded = load_dataset_for_utility(dataset_dir)
    meta = {"family": loaded.family_id, "subfamily": loaded.subfamily_id,
            "difficulty": None,
            "cluster_count": loaded.cluster_count_from_generator}
    for view_id in VIEW_IDS:
        view = build_view(loaded, view_id)
        for mname in methods_wanted:
            rec = run_method_view(
                loaded=loaded, view=view, method=method_map[mname],
                config_path=resolved.config_path(mname, view_id),
                dataset_meta=meta, cfg=cfg)
            print(f"  {mname}/{view_id}: {rec.status} ari={rec.ari} "
                  f"mc={rec.extra.get('selected_metric_count')}", flush=True)
    print("recovery done -> wrote into split_01", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
