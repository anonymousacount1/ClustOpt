"""Stage 4C Phase D -- publication-authoritative metric analysis.

Amendment ``stage4c_utility_candidate_basis_correction``
(basis ``canonical_fixed32_candidate_basis_v1``).

Only metric-utility-dependent tables are built here. The method track is not
recomputed and not re-aggregated: Phase-A outputs, method ARI, BestView, xy_2d,
runtimes and failures stay exactly as produced and are referenced by path.

Coverage is reported as a separate axis and never folded into the utility. A
metric evaluable on part of the corpus is ranked on the units where it is
defined, so the ranking columns are named ``available_case_*`` to keep that
explicit. No coverage-adjusted score is introduced here; coverage-aware
interpretation belongs to scientific review.
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

from Independent_Domain_Benchmark.execution import hashing as H        # noqa: E402

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
CFG = _ROOT / "configs"
GENERALIST_CORE = ("noise_aware_silhouette", "silhouette", "gridness_fft_acf")
TOTAL_UNITS = 150


def source_balanced(df: pd.DataFrame, value: str, by: List[str]) -> pd.DataFrame:
    per = (df.dropna(subset=[value])
             .groupby(by + ["source"], dropna=False)[value].mean().reset_index())
    return (per.groupby(by, dropna=False)[value].mean().reset_index()
              .rename(columns={value: "%s_source_balanced" % value}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    run_dir = RESULTS / args.run_id
    out = run_dir / "phase_d_canonical_utility"
    out.mkdir(parents=True, exist_ok=True)

    registry = json.loads((CFG / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    amend = json.loads((CFG / "stage4c_utility_basis_amendment.json")
                       .read_text(encoding="utf-8"))
    dispatch = json.loads((CFG / "stage4c_metric_dispatch_amendment.json")
                          .read_text(encoding="utf-8"))
    primary = {e["canonical_metric_id"]: e for e in registry["entries"]}
    diagnostics = set(registry["diagnostics"])

    u = pd.read_csv(run_dir / "phase_c_canonical" / "metric_utility_canonical.csv.gz")
    u = u[u["metric_id"].isin(set(primary))]
    u.to_csv(out / "metric_utility_per_dataset_view.csv.gz", index=False)
    uv = u[u["status"] == "valid"].dropna(subset=["utility"])

    # ------------------------------------------------ aggregate, available-case
    magg = uv.groupby(["metric_id", "metric_group"], as_index=False).agg(
        available_case_utility_macro=("utility", "mean"),
        available_case_utility_median=("utility", "median"),
        units_with_valid_utility=("utility", "count"))
    sb = source_balanced(uv, "utility", ["metric_id"]).rename(
        columns={"utility_source_balanced":
                 "available_case_utility_source_balanced"})
    magg = magg.merge(sb, on="metric_id", how="left")
    magg["units_total"] = TOTAL_UNITS
    magg["coverage"] = magg["units_with_valid_utility"] / TOTAL_UNITS
    magg["available_case_macro_rank"] = magg[
        "available_case_utility_macro"].rank(ascending=False, method="min")
    magg["available_case_source_balanced_rank"] = magg[
        "available_case_utility_source_balanced"].rank(ascending=False,
                                                       method="min")
    magg["full_coverage"] = magg["units_with_valid_utility"] == TOTAL_UNITS
    magg["repaired_metric"] = magg["metric_id"].isin(dispatch["repair_set"])
    magg["ranking_basis"] = "available_case: ranked on units where the metric is defined"
    magg.sort_values("available_case_utility_macro", ascending=False).to_csv(
        out / "metric_aggregate_summary.csv", index=False)

    # ------------------------------------------------------- per-source table
    per_src = uv.groupby(["metric_id", "metric_group", "source"],
                         as_index=False).agg(
        utility_macro=("utility", "mean"), units=("utility", "count"))
    per_src.to_csv(out / "metric_source_summary.csv", index=False)

    # ----------------------------------------------------------- rank table
    ranks = magg[["metric_id", "metric_group",
                  "available_case_utility_macro", "available_case_macro_rank",
                  "available_case_utility_source_balanced",
                  "available_case_source_balanced_rank",
                  "units_with_valid_utility", "units_total", "coverage",
                  "full_coverage"]].sort_values("available_case_macro_rank")
    ranks.to_csv(out / "metric_rank_table.csv", index=False)

    full_only = magg[magg["full_coverage"]].copy()
    full_only["full_coverage_macro_rank"] = full_only[
        "available_case_utility_macro"].rank(ascending=False, method="min")
    full_only.sort_values("full_coverage_macro_rank").to_csv(
        out / "metric_rank_table_full_coverage_only.csv", index=False)

    # --------------------------------------------------------- top-k frequency
    topk = []
    for (d, v), g in uv.groupby(["dataset_id", "view_id"]):
        for mid in g.nlargest(5, "utility")["metric_id"]:
            topk.append({"dataset_id": d, "view_id": v, "metric_id": mid})
    if topk:
        (pd.DataFrame(topk).groupby("metric_id", as_index=False).size()
         .rename(columns={"size": "top5_count"})
         .sort_values("top5_count", ascending=False)
         .to_csv(out / "metric_top5_frequency.csv", index=False))

    # ------------------------------------------------------------- groups
    grp = magg.groupby("metric_group", as_index=False).agg(
        metrics=("metric_id", "count"),
        available_case_utility_macro_mean=("available_case_utility_macro", "mean"),
        available_case_source_balanced_mean=(
            "available_case_utility_source_balanced", "mean"),
        coverage_mean=("coverage", "mean"))
    grp.to_csv(out / "metric_group_summary.csv", index=False)

    # --------------------------------------------------- generalist / specialist
    gen = magg[magg["metric_id"].isin(GENERALIST_CORE)].copy()
    gen["hypothesis"] = "internal_generalist_core"
    gen.to_csv(out / "generalist_core_external.csv", index=False)

    spread = uv.groupby("metric_id", as_index=False).agg(
        utility_mean=("utility", "mean"), utility_std=("utility", "std"),
        utility_min=("utility", "min"), utility_max=("utility", "max"))
    spread["spread"] = spread["utility_max"] - spread["utility_min"]
    spread = spread.merge(
        magg[["metric_id", "metric_group", "coverage", "full_coverage"]],
        on="metric_id", how="left")
    spread.sort_values("utility_std", ascending=False).to_csv(
        out / "metric_specialisation_spread.csv", index=False)

    # ------------------------------------------------------ modern comparators
    modern = magg[magg["metric_group"] == "Modern"].copy()
    others = magg[magg["metric_group"] != "Modern"]
    mod = {
        "amendment_id": amend["amendment_id"],
        "candidate_basis_version": amend["candidate_basis_version"],
        "ranking_basis": "available_case",
        "modern_primary": modern[[
            "metric_id", "available_case_utility_macro",
            "available_case_utility_source_balanced", "coverage",
            "units_with_valid_utility", "units_total",
            "available_case_macro_rank"]].to_dict("records"),
        "comparison_groups": {g: {
            "n": int((others["metric_group"] == g).sum()),
            "available_case_utility_macro_mean": float(
                others[others["metric_group"] == g][
                    "available_case_utility_macro"].mean()),
            "best_metric": (others[others["metric_group"] == g]
                            .nlargest(1, "available_case_utility_macro")
                            ["metric_id"].tolist())}
            for g in sorted(others["metric_group"].dropna().unique())},
        "diagnostics_excluded": sorted(diagnostics),
        "mode_shopping": False,
        "coverage_caveat": (
            "utility is an available-case macro over the units where a metric is "
            "defined. A metric with partial coverage is NOT thereby superior to a "
            "full-coverage metric; no coverage-adjusted score is introduced here "
            "and none is implied. metric_rank_table_full_coverage_only.csv gives "
            "the restricted comparison."),
        "cvnn_note": ("CVNN is normalised across each fixed32 bank, the comparison "
                      "set fpc assumes, so its value is bank-relative by design."),
    }
    (out / "modern_comparator_analysis.json").write_text(
        json.dumps(mod, indent=2, default=str), encoding="utf-8")

    # --------------------------------- joint pipeline: metric availability only
    method = pd.read_csv(run_dir / "phase_c" / "method_ari.csv.gz")
    succ = method[method["status"] == "success"].dropna(subset=["ari"])
    best = (uv.sort_values("utility", ascending=False)
              .groupby(["dataset_id", "view_id"], as_index=False).first()
              [["dataset_id", "view_id", "metric_id", "utility"]]
              .rename(columns={"metric_id": "best_available_metric",
                               "utility": "best_available_utility"}))
    joint = succ.merge(best, on=["dataset_id", "view_id"], how="left")
    joint = joint[["dataset_id", "source", "view_id", "method_id", "ari",
                   "selected_algorithm", "best_available_metric",
                   "best_available_utility"]]
    joint.to_csv(out / "joint_pipeline_per_unit.csv.gz", index=False)
    jsum = joint.groupby("method_id", as_index=False).agg(
        units=("ari", "count"), method_ari_mean=("ari", "mean"),
        best_available_metric_utility_mean=("best_available_utility", "mean"))
    jsum["note"] = ("metric availability vs method capture; the method columns are "
                    "reused unchanged from the authoritative method track")
    jsum.to_csv(out / "joint_pipeline_summary.csv", index=False)

    # ------------------------------------------------------------- manifest
    zero = sorted(set(primary) - set(magg["metric_id"]))
    declared: Dict[str, int] = {}
    for e in registry["entries"]:
        declared[e["group"]] = declared.get(e["group"], 0) + 1
    cs = json.loads((run_dir / "phase_c_canonical"
                     / "canonical_utility_summary.json").read_text(encoding="utf-8"))

    manifest = {
        "package": "phase_d_canonical_utility",
        "publication_authoritative_for": "metric utility analysis",
        "run_id": args.run_id,
        "amendment_id": amend["amendment_id"],
        "candidate_basis_version": amend["candidate_basis_version"],
        "dispatch_amendment_id": dispatch["amendment_id"],
        "utility_source_hash": cs["utility_source_hash"],
        "utility_definition_changed": False,
        "canonical_candidates": cs["stats"]["canonical_candidates"],
        "logical_candidate_metric_combinations": cs["stats"]["logical_combinations"],
        "utility_cells": cs["stats"]["utility_cells_produced"],
        "utility_valid": cs["stats"]["utility_valid"],
        "utility_insufficient": cs["stats"]["utility_insufficient"],
        "declared_inventory": declared,
        "observed_inventory_with_usable_utility": {
            g: int((magg["metric_group"] == g).sum())
            for g in sorted(magg["metric_group"].dropna().unique())},
        "primary_total": len(primary),
        "primary_with_usable_utility": int(magg["metric_id"].nunique()),
        "zero_coverage_primary_metrics": zero,
        "method_track": {
            "recomputed": False,
            "authoritative_path": "phase_d/",
            "tables": ["method_aggregate_summary.csv",
                       "method_bestview_per_dataset.csv.gz",
                       "method_xy2d_per_dataset.csv.gz",
                       "method_runtime_summary.csv",
                       "method_failure_summary.csv"]},
        "superseded_packages": {
            "phase_d": "original; 61/67 metrics, modern comparators empty",
            "phase_d_corrected": "dispatch repaired but on the non-canonical basis"},
        "unchanged_layers": {
            "fixed32_banks": "physical/fixed32__*",
            "candidate_ari": "phase_c/fixed32_candidate_ari.csv.gz",
            "metric_values_61": "evaluation/metrics__*",
            "metric_values_6": "evaluation/metrics_v2__*"},
        "coverage_axis": "reported separately; never folded into utility",
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    txt = json.dumps(manifest, indent=2, default=str)
    (out / "package_manifest.json").write_text(txt, encoding="utf-8")
    print("  canonical Phase D -> %s" % out)
    print("  primary with usable utility: %d / %d"
          % (manifest["primary_with_usable_utility"], len(primary)))
    print("  inventory %s"
          % json.dumps(manifest["observed_inventory_with_usable_utility"]))
    print("  package manifest hash %s" % H.canonical_text_hash(txt.encode("utf-8")))
    if zero:
        print("  zero-coverage primary metrics: %s" % zero)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
