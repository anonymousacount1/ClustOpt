"""Shared synthetic fixtures.

The synthetic frames deliberately include the awkward cases: a dataset where a
method has no successful view, exact ARI ties that force a tie-break, and a
method with no metric-selection record.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    collector, frames,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
    METHOD_BY_NAME, VIEW_IDS,
)

FAMILIES = ("Arcs / circles / rings", "Standard clustering blobs",
            "Linear bands / parallel stripes")


def make_run_row(*, dataset_id: str, family: str, method_name: str, view_id: str,
                 ari: float | None, runtime: float, status: str = "success",
                 selected_k: int = 3, true_k: int = 3,
                 selected_metrics: Sequence[str] = (),
                 weights: Dict[str, float] | None = None) -> Dict:
    """One synthetic run-frame row carrying full registry provenance."""
    import json

    spec = METHOD_BY_NAME[method_name]
    row = {c: None for c in collector.RUN_FRAME_COLUMNS}
    row.update({
        "dataset_id": dataset_id, "dataset_dir": f"/synthetic/{dataset_id}",
        "split_id": 1, "family": family, "family_id": family.lower().replace(" ", "_"),
        "subfamily": f"{family}-sub", "difficulty": "easy", "cluster_count": true_k,
        "n_points": 1000, "view_id": view_id,
        "method_name": spec.method_name, "method_full_name": spec.display_name,
        "method_short_label": spec.short_label, "paper_label": spec.paper_label,
        "on_disk_name": spec.on_disk_name, "schema": spec.schema,
        "framework": spec.framework, "method_family": spec.method_family,
        "utility_source": spec.utility_source, "k_policy": spec.k_policy,
        "fixed_metric_count": spec.fixed_metric_count,
        "dynamic_k_source": spec.dynamic_k_source,
        "weighting_mode": spec.weighting_mode,
        "softmax_temperature": spec.softmax_temperature,
        "metric_inventory": spec.metric_inventory,
        "metric_inventory_size": spec.metric_inventory_size,
        "selection_capability": ("metric_weights" if spec.metric_selection_capable
                                else "none"),
        "oracle_level": spec.oracle_level,
        "uses_real_utility": spec.uses_real_utility,
        "uses_real_dynamic_k": spec.uses_real_dynamic_k,
        "is_oracle_assisted": spec.is_oracle_assisted,
        "is_utility_oracle": spec.is_utility_oracle,
        "is_meta_learning": spec.is_meta_learning,
        "is_external": spec.is_external, "is_paper_method": spec.is_paper_method,
        "same_search_space": spec.same_search_space,
        "search_budget": spec.search_budget,
        "status": status, "failure_reason": "" if status == "success" else "synthetic",
        "provenance_mismatch": "",
        "ari": ari, "nmi": ari, "ami": ari,
        "runtime_sec": runtime, "fit_predict_sec": runtime * 0.6,
        "cvi_eval_sec": runtime * 0.4, "n_cvi_evaluations": 50,
        "selected_k": selected_k, "true_k": true_k,
        "k_correct": selected_k == true_k,
        "k_abs_error": abs(selected_k - true_k),
        "k_signed_error": selected_k - true_k,
        "best_objective_score": 0.5,
        "n_trials_completed": 50,
        "run_dir": f"/synthetic/{dataset_id}/{method_name}/{view_id}",
        "trace_path": f"/synthetic/{dataset_id}/{method_name}/{view_id}/trace.csv",
    })
    if selected_metrics:
        w = weights or {m: 1.0 / len(selected_metrics) for m in selected_metrics}
        row.update({
            "selected_metrics": json.dumps(list(selected_metrics)),
            "selected_metric_weights": json.dumps(w),
            "selected_metric_utilities": json.dumps(
                {m: 0.9 - 0.05 * i for i, m in enumerate(selected_metrics)}),
            "selected_metric_count": len(selected_metrics),
            "metric_weight_entropy": 0.95,
            "top_metric": list(selected_metrics)[0],
        })
    return row


@pytest.fixture
def tiny_run_frame() -> pd.DataFrame:
    """A 3-method x 4-dataset x 3-view frame with edge cases baked in.

    * ``d1``/``regressor_top5_raw``: an exact ARI tie between two views with
      different runtimes -> the runtime tie-break must fire.
    * ``d4``/``knn_top5_raw``: every view failed -> the row must survive with a
      null Best-View ARI.
    * ``silhouette_single``: single-CVI, so no meaningful metric selection.
    """
    rows: List[Dict] = []
    metrics = ["silhouette", "dbcv", "neighborhood_purity", "hollow_score",
               "gridness_2d"]
    plan = {
        ("d1", "regressor_top5_raw"): [(0.80, 10.0), (0.80, 4.0), (0.60, 6.0)],
        ("d1", "knn_top5_raw"):       [(0.70, 5.0), (0.75, 5.0), (0.65, 5.0)],
        ("d1", "silhouette_single"):  [(0.50, 2.0), (0.55, 2.0), (0.40, 2.0)],
        ("d2", "regressor_top5_raw"): [(0.90, 8.0), (0.85, 8.0), (0.88, 8.0)],
        ("d2", "knn_top5_raw"):       [(0.95, 6.0), (0.60, 6.0), (0.70, 6.0)],
        ("d2", "silhouette_single"):  [(0.40, 2.0), (0.45, 2.0), (0.42, 2.0)],
        ("d3", "regressor_top5_raw"): [(0.30, 9.0), (0.35, 9.0), (0.33, 9.0)],
        ("d3", "knn_top5_raw"):       [(0.40, 7.0), (0.38, 7.0), (0.39, 7.0)],
        ("d3", "silhouette_single"):  [(0.60, 2.0), (0.20, 2.0), (0.25, 2.0)],
        ("d4", "regressor_top5_raw"): [(0.55, 11.0), (0.50, 11.0), (0.52, 11.0)],
        ("d4", "knn_top5_raw"):       [(None, 3.0), (None, 3.0), (None, 3.0)],
        ("d4", "silhouette_single"):  [(0.48, 2.0), (0.44, 2.0), (0.46, 2.0)],
    }
    for (dataset_id, method), per_view in plan.items():
        family = FAMILIES[int(dataset_id[-1]) % len(FAMILIES)]
        for view_id, (ari, runtime) in zip(VIEW_IDS, per_view):
            sel = metrics if method != "silhouette_single" else ["silhouette"]
            rows.append(make_run_row(
                dataset_id=dataset_id, family=family, method_name=method,
                view_id=view_id, ari=ari, runtime=runtime,
                status="success" if ari is not None else "failed",
                selected_k=3 if (ari or 0) > 0.5 else 4,
                selected_metrics=sel))
    return collector._coerce_types(
        pd.DataFrame(rows, columns=list(collector.RUN_FRAME_COLUMNS)))


@pytest.fixture
def tiny_best_view(tiny_run_frame: pd.DataFrame) -> pd.DataFrame:
    return frames.build_best_view_frame(tiny_run_frame, verbose=False)


@pytest.fixture
def pool_best_view() -> pd.DataFrame:
    """A best-view frame spanning a whole candidate pool, for selection/VBS tests.

    Method ``i`` gets a base ARI of ``0.50 + 0.01*i`` and a per-dataset wobble, so
    the ranking is known in advance and a deliberate exact tie is inserted between
    the top two methods' means.
    """
    from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
        CANDIDATE_POOL_BY_ID, names_where,
    )
    methods = list(names_where(method_family="clustopt_mlp")) \
        + list(names_where(method_family="clustopt_knn")) \
        + list(names_where(method_family="clustopt_real")) \
        + ["AutoClust_Extended_InDomain_same_search_space",
           "AutoClust_Original_InDomain_same_search_space",
           "ML2DAC_Extended_InDomain_same_search_space",
           "ML2DAC_Original_InDomain_same_search_space",
           "all_metrics_uniform", "uniform_classic_cvi", "silhouette_single",
           "dbcv_single", "davies_bouldin_single", "calinski_harabasz_single"]
    rng = np.random.default_rng(7)
    n_ds = 24
    rows: List[Dict] = []
    metrics = ["silhouette", "dbcv", "neighborhood_purity", "hollow_score",
               "gridness_2d", "arc_circle_fit"]
    for i, method in enumerate(methods):
        base = 0.50 + 0.01 * i
        # External adapters write no metric-selection record, so the fixture must
        # not invent one -- otherwise the capability preflight would look more
        # permissive here than it is on real data.
        selected = () if METHOD_BY_NAME[method].is_external \
            else metrics[: max(1, (i % 5) + 1)]
        for j in range(n_ds):
            dataset_id = f"ds{j:02d}"
            family = FAMILIES[j % len(FAMILIES)]
            ari = float(np.clip(base + rng.normal(0, 0.08), 0.0, 1.0))
            for view_id, delta in zip(VIEW_IDS, (0.0, -0.05, -0.03)):
                rows.append(make_run_row(
                    dataset_id=dataset_id, family=family, method_name=method,
                    view_id=view_id, ari=float(np.clip(ari + delta, 0.0, 1.0)),
                    runtime=5.0 + i, selected_k=3 if ari > 0.5 else 4,
                    selected_metrics=selected))
    run_frame = collector._coerce_types(
        pd.DataFrame(rows, columns=list(collector.RUN_FRAME_COLUMNS)))
    return frames.build_best_view_frame(run_frame, verbose=False)
