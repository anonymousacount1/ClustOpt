"""Run one CLUSTOPT configuration on one dataset view.

A dataset is a folder produced by the generator (``data_data.csv`` with columns
x, y, cluster_id). Ground truth is used only to report ARI afterwards; the search
objective never reads it. Configurations that predict the utility profile
(``regressor_dynamic`` / ``knn_dynamic``) additionally need the dataset's
meta-features (``clustopt.meta_features.extract``) and the released predictor
(run ``python scripts/prepare_workspace.py`` once).
"""
from __future__ import annotations

from pathlib import Path

from ..common import ensure_src_on_path

ensure_src_on_path()


def load_view(dataset_dir, view_id):
    from models.Clustering_Repository_Builder.experiments.experiment_execution.view_builder import (
        build_view, load_dataset_for_utility)
    return build_view(load_dataset_for_utility(Path(dataset_dir)), view_id)


def run(config_path, dataset_dir, view_id="xy_2d", *, n_trials=None, runtime=None):
    """Return the implementation's ExperimentRunOutcome (selected partition in ``y_pred``)."""
    import json
    import tempfile
    from models.Clustering_Repository_Builder.experiments.experiment_execution.clustopt_execution import (
        run_experiment_clustopt)
    config_path = Path(config_path)
    if n_trials is not None:  # optional smaller budget for quick checks
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg["search_algorithm"].setdefault("params", {})["n_trials"] = int(n_trials)
        tmp = Path(tempfile.mkdtemp()) / config_path.name
        tmp.write_text(json.dumps(cfg), encoding="utf-8")
        config_path = tmp
    view = load_view(dataset_dir, view_id)
    return run_experiment_clustopt(config_path=config_path, dataset_dir=Path(dataset_dir), view_id=view_id,
                                   X_decision=view.X_decision, X_full=view.X_full, y_true=None,
                                   runtime_block=dict(runtime or {}), attach_profiler=False)


def adjusted_rand(dataset_dir, labels, view_id="xy_2d"):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(load_view(dataset_dir, view_id).y_true, labels))
