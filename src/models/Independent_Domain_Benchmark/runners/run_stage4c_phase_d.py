"""Stage 4C Phase D — frozen analysis over both tracks.

Consumes only Phase-C outputs. It changes no model, metric or protocol, and
nothing here may trigger a rerun.

Every aggregation rule was frozen before outcomes: the dataset is the inferential
unit; BestView is the maximum over SUCCESSFULLY produced views and a failed view
is never given a sentinel; the source-balanced macro averages within FCPS,
sklearn-shapes and real-PCA and then weights those three means equally, so the 28
real datasets cannot dominate by count; diagnostics stay out of primary metric
rankings.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
warnings.filterwarnings("ignore")

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
CFG = _ROOT / "configs"
SOURCES = ("fcps", "sklearn_shapes", "real_pca")
GENERALIST_CORE = ("noise_aware_silhouette", "silhouette", "gridness_fft_acf")


def source_balanced(df: pd.DataFrame, value: str, by: List[str]) -> pd.DataFrame:
    """Mean within each source, then the three source means weighted equally."""
    per = (df.dropna(subset=[value])
             .groupby(by + ["source"], dropna=False)[value].mean().reset_index())
    return (per.groupby(by, dropna=False)[value].mean().reset_index()
              .rename(columns={value: "%s_source_balanced" % value}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    src = RESULTS / args.run_id / "phase_c"
    out = RESULTS / args.run_id / "phase_d"
    out.mkdir(parents=True, exist_ok=True)

    method = pd.read_csv(src / "method_ari.csv.gz")
    f32 = pd.read_csv(src / "fixed32_candidate_ari.csv.gz")
    util = pd.read_csv(src / "metric_utility.csv.gz")
    registry = json.loads((CFG / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    primary_ids = {e["canonical_metric_id"] for e in registry["entries"]}
    diagnostics = set(registry["diagnostics"])

    # ---------------------------------------------------------- method track
    method.to_csv(out / "method_per_dataset_view.csv.gz", index=False)

    succ = method[method["status"] == "success"].dropna(subset=["ari"])
    best = (succ.sort_values("ari", ascending=False)
                .groupby(["dataset_id", "source", "method_id"], as_index=False)
                .first()[["dataset_id", "source", "method_id", "view_id", "ari"]]
                .rename(columns={"view_id": "best_view", "ari": "bestview_ari"}))
    # datasets where every view failed keep NO artificial value
    allv = method.groupby(["dataset_id", "source", "method_id"], as_index=False).agg(
        views_attempted=("view_id", "count"),
        views_successful=("status", lambda s: int((s == "success").sum())))
    best = allv.merge(best, on=["dataset_id", "source", "method_id"], how="left")
    best["dataset_method_status"] = np.where(best["views_successful"] > 0,
                                             "SUCCESS", "FAILURE")
    best.to_csv(out / "method_bestview_per_dataset.csv.gz", index=False)

    xy = succ[succ["view_id"] == "xy_2d"][
        ["dataset_id", "source", "method_id", "ari"]].rename(
        columns={"ari": "xy_2d_ari"})
    xy.to_csv(out / "method_xy2d_per_dataset.csv.gz", index=False)

    agg = best.groupby("method_id", as_index=False).agg(
        datasets=("dataset_id", "nunique"),
        dataset_success=("dataset_method_status",
                         lambda s: int((s == "SUCCESS").sum())),
        bestview_ari_macro=("bestview_ari", "mean"))
    agg = agg.merge(source_balanced(best, "bestview_ari", ["method_id"]),
                    on="method_id", how="left")
    xy_agg = xy.groupby("method_id", as_index=False).agg(
        xy_2d_ari_macro=("xy_2d_ari", "mean"))
    agg = agg.merge(xy_agg, on="method_id", how="left")
    agg = agg.merge(source_balanced(xy, "xy_2d_ari", ["method_id"]),
                    on="method_id", how="left")
    rates = method.groupby("method_id", as_index=False).agg(
        arm_records=("status", "count"),
        arm_success=("status", lambda s: int((s == "success").sum())))
    rates["arm_success_rate"] = rates["arm_success"] / rates["arm_records"]
    agg = agg.merge(rates, on="method_id", how="left")
    agg.sort_values("bestview_ari_macro", ascending=False).to_csv(
        out / "method_aggregate_summary.csv", index=False)

    fail = (method[method["status"] != "success"]
            .groupby(["method_id", "failure_kind"], as_index=False)
            .size().rename(columns={"size": "n"}))
    fail.to_csv(out / "method_failure_summary.csv", index=False)

    rt = method.dropna(subset=["method_runtime_equivalent"]).groupby(
        "method_id", as_index=False).agg(
        runtime_equiv_mean=("method_runtime_equivalent", "mean"),
        runtime_equiv_median=("method_runtime_equivalent", "median"),
        runtime_equiv_total=("method_runtime_equivalent", "sum"))
    rt.to_csv(out / "method_runtime_summary.csv", index=False)

    # --------------------------------------------------------- fixed32 track
    valid = f32.groupby(["dataset_id", "source", "view_id"], as_index=False).agg(
        valid_candidates=("candidate_index", "count"))
    valid.to_csv(out / "fixed32_candidate_validity.csv", index=False)
    f32.to_csv(out / "fixed32_candidate_ari.csv.gz", index=False)

    u = util[util["metric_id"].isin(primary_ids)].copy()
    u.to_csv(out / "metric_utility_per_dataset_view.csv.gz", index=False)

    uv = u[u["status"] == "valid"].dropna(subset=["utility"])
    magg = uv.groupby(["metric_id", "metric_group"], as_index=False).agg(
        utility_macro=("utility", "mean"),
        utility_median=("utility", "median"),
        n_units=("utility", "count"))
    magg = magg.merge(source_balanced(uv, "utility", ["metric_id"]),
                      on="metric_id", how="left")
    cov = u.groupby("metric_id", as_index=False).agg(
        units_total=("status", "count"),
        units_valid=("status", lambda s: int((s == "valid").sum())))
    cov["valid_fraction"] = cov["units_valid"] / cov["units_total"]
    magg = magg.merge(cov, on="metric_id", how="left")
    magg["rank_macro"] = magg["utility_macro"].rank(ascending=False, method="min")
    magg["rank_source_balanced"] = magg["utility_source_balanced"].rank(
        ascending=False, method="min")
    magg.sort_values("utility_macro", ascending=False).to_csv(
        out / "metric_aggregate_summary.csv", index=False)

    # top-k frequency per dataset/view
    topk = []
    for (d, v), g in uv.groupby(["dataset_id", "view_id"]):
        for mid in g.nlargest(5, "utility")["metric_id"]:
            topk.append({"dataset_id": d, "view_id": v, "metric_id": mid})
    tk = (pd.DataFrame(topk).groupby("metric_id", as_index=False).size()
          .rename(columns={"size": "top5_count"}) if topk else pd.DataFrame())
    if len(tk):
        tk.to_csv(out / "metric_top5_frequency.csv", index=False)

    grp = magg.groupby("metric_group", as_index=False).agg(
        metrics=("metric_id", "count"),
        utility_macro_mean=("utility_macro", "mean"),
        utility_source_balanced_mean=("utility_source_balanced", "mean"),
        valid_fraction_mean=("valid_fraction", "mean"))
    grp.to_csv(out / "metric_group_summary.csv", index=False)

    # ------------------------------------------- generalist / specialist
    gen = magg[magg["metric_id"].isin(GENERALIST_CORE)].copy()
    gen["hypothesis"] = "internal_generalist_core"
    gen.to_csv(out / "generalist_core_external.csv", index=False)

    spread = uv.groupby("metric_id", as_index=False).agg(
        utility_mean=("utility", "mean"), utility_std=("utility", "std"),
        utility_min=("utility", "min"), utility_max=("utility", "max"))
    spread["spread"] = spread["utility_max"] - spread["utility_min"]
    spread = spread.merge(magg[["metric_id", "metric_group"]], on="metric_id",
                          how="left")
    spread.sort_values("utility_std", ascending=False).to_csv(
        out / "metric_specialisation_spread.csv", index=False)

    # ---------------------------------------------- modern comparators
    modern = magg[magg["metric_group"] == "Modern"].copy()
    others = magg[magg["metric_group"] != "Modern"]
    mod_cmp = {
        "modern_primary": modern[["metric_id", "utility_macro",
                                  "utility_source_balanced", "valid_fraction",
                                  "rank_macro"]].to_dict("records"),
        "comparison_groups": {g: {
            "n": int((others["metric_group"] == g).sum()),
            "utility_macro_mean": float(
                others[others["metric_group"] == g]["utility_macro"].mean()),
            "best_metric": (others[others["metric_group"] == g]
                            .nlargest(1, "utility_macro")["metric_id"].tolist())}
            for g in sorted(others["metric_group"].dropna().unique())},
        "diagnostics_excluded": sorted(diagnostics),
        "mode_shopping": False,
        "note": ("primary remains the paper-faithful dcsi and cdbw; the "
                 "compatibility variants are sensitivity diagnostics and never "
                 "enter this comparison"),
    }
    (out / "modern_comparator_analysis.json").write_text(
        json.dumps(mod_cmp, indent=2, default=str), encoding="utf-8")

    # ---------------------------------------------- joint pipeline (descriptive)
    best_metric = (uv.sort_values("utility", ascending=False)
                     .groupby(["dataset_id", "view_id"], as_index=False).first()
                     [["dataset_id", "view_id", "metric_id", "utility"]]
                     .rename(columns={"metric_id": "best_fixed32_metric",
                                      "utility": "best_fixed32_utility"}))
    joint = succ.merge(best_metric, on=["dataset_id", "view_id"], how="left")
    joint = joint[["dataset_id", "source", "view_id", "method_id", "ari",
                   "selected_algorithm", "best_fixed32_metric",
                   "best_fixed32_utility"]]
    joint.to_csv(out / "joint_pipeline_per_unit.csv.gz", index=False)
    jsum = joint.groupby("method_id", as_index=False).agg(
        units=("ari", "count"), method_ari_mean=("ari", "mean"),
        best_available_metric_utility_mean=("best_fixed32_utility", "mean"))
    jsum["note"] = ("metric availability vs method capture: a high available "
                    "utility with low method ARI indicates the method did not "
                    "exploit an available signal")
    jsum.to_csv(out / "joint_pipeline_summary.csv", index=False)

    summary = {
        "run_id": args.run_id,
        "method_rows": int(len(method)),
        "method_successful": int((method["status"] == "success").sum()),
        "datasets": int(method["dataset_id"].nunique()),
        "methods": int(method["method_id"].nunique()),
        "fixed32_candidate_ari_rows": int(len(f32)),
        "metric_utility_rows": int(len(u)),
        "metric_utility_valid": int(len(uv)),
        "primary_metrics_with_any_utility": int(uv["metric_id"].nunique()),
        "aggregation_rules": {
            "inferential_unit": "dataset",
            "dataset_macro": "equal weight per dataset",
            "source_balanced_macro": ("mean within FCPS / sklearn_shapes / "
                                      "real_pca, then the three means weighted "
                                      "equally"),
            "bestview": "max over SUCCESSFUL views; all-view failure stays FAILURE",
            "no_sentinel_ari": True,
            "frozen_before_outcomes": True},
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    (out / "phase_d_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("  Phase D wrote %d files to %s" % (len(summary["outputs"]), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
