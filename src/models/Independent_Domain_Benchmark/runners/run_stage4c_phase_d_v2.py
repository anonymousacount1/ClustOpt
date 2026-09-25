"""Stage 4C Phase D (corrected) -- the intended 67-metric analysis.

Amendment ``stage4c_fixed32_metric_dispatch_repair``.

Only the metric-dependent tables are rebuilt. The method track is NOT recomputed
and NOT re-aggregated: Phase A outputs, method ARIs, BestView values and method
runtime summaries are untouched and are referenced, not regenerated. The
fixed32 track remains diagnostic; nothing here feeds back into any method,
model, policy or search.

The utility input is composed: the 61 untouched metrics keep their original
Phase-C utilities, and the six repaired metrics take their Phase-C-v2 values.
Phase C v2 verifies that recomputing the 61 on the combined frame reproduces
their stored values, so the composition equals the single intended run.

Aggregation rules are unchanged from the frozen Phase D: the dataset is the
inferential unit, a failed unit is never given a sentinel, and the
source-balanced macro averages within FCPS, sklearn-shapes and real-PCA before
weighting those three means equally.
"""
from __future__ import annotations

import argparse
import json
import shutil
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
GENERALIST_CORE = ("noise_aware_silhouette", "silhouette", "gridness_fft_acf")


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
    src = run_dir / "phase_c"
    out = run_dir / "phase_d_corrected"
    out.mkdir(parents=True, exist_ok=True)

    registry = json.loads((CFG / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    amend = json.loads((CFG / "stage4c_metric_dispatch_amendment.json")
                       .read_text(encoding="utf-8"))
    primary = {e["canonical_metric_id"]: e for e in registry["entries"]}
    diagnostics = set(registry["diagnostics"])
    repair = set(amend["repair_set"])

    # ------------------------------------------------ composed utility input
    u1 = pd.read_csv(src / "metric_utility.csv.gz")
    u2 = pd.read_csv(run_dir / "phase_c_v2" / "metric_utility_v2.csv.gz")
    u1 = u1[~u1["metric_id"].isin(repair)].copy()      # drop the failed v1 rows
    u2 = u2[u2["metric_id"].isin(repair)].copy()
    u1["utility_source"] = "phase_c_v1"
    u2["utility_source"] = "phase_c_v2_repaired"
    u = pd.concat([u1, u2], ignore_index=True)
    u = u[u["metric_id"].isin(set(primary))]
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
    magg["repaired"] = magg["metric_id"].isin(repair)
    magg["rank_macro"] = magg["utility_macro"].rank(ascending=False, method="min")
    magg["rank_source_balanced"] = magg["utility_source_balanced"].rank(
        ascending=False, method="min")
    magg.sort_values("utility_macro", ascending=False).to_csv(
        out / "metric_aggregate_summary.csv", index=False)

    topk = []
    for (d, v), g in uv.groupby(["dataset_id", "view_id"]):
        for mid in g.nlargest(5, "utility")["metric_id"]:
            topk.append({"dataset_id": d, "view_id": v, "metric_id": mid})
    if topk:
        (pd.DataFrame(topk).groupby("metric_id", as_index=False).size()
         .rename(columns={"size": "top5_count"})
         .sort_values("top5_count", ascending=False)
         .to_csv(out / "metric_top5_frequency.csv", index=False))

    grp = magg.groupby("metric_group", as_index=False).agg(
        metrics=("metric_id", "count"),
        utility_macro_mean=("utility_macro", "mean"),
        utility_source_balanced_mean=("utility_source_balanced", "mean"),
        valid_fraction_mean=("valid_fraction", "mean"))
    grp.to_csv(out / "metric_group_summary.csv", index=False)

    gen = magg[magg["metric_id"].isin(GENERALIST_CORE)].copy()
    gen["hypothesis"] = "internal_generalist_core"
    gen.to_csv(out / "generalist_core_external.csv", index=False)

    spread = uv.groupby("metric_id", as_index=False).agg(
        utility_mean=("utility", "mean"), utility_std=("utility", "std"),
        utility_min=("utility", "min"), utility_max=("utility", "max"))
    spread["spread"] = spread["utility_max"] - spread["utility_min"]
    spread = spread.merge(magg[["metric_id", "metric_group", "repaired"]],
                          on="metric_id", how="left")
    spread.sort_values("utility_std", ascending=False).to_csv(
        out / "metric_specialisation_spread.csv", index=False)

    # ----------------------------------------------- modern comparators
    modern = magg[magg["metric_group"] == "Modern"].copy()
    others = magg[magg["metric_group"] != "Modern"]
    mod_cmp = {
        "amendment_id": amend["amendment_id"],
        "modern_primary": modern[["metric_id", "utility_macro",
                                  "utility_source_balanced", "valid_fraction",
                                  "rank_macro", "n_units"]].to_dict("records"),
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
                 "enter this comparison. CVNN is normalised across each fixed32 "
                 "bank, which is the comparison set fpc assumes, so its value is "
                 "bank-relative by construction."),
    }
    (out / "modern_comparator_analysis.json").write_text(
        json.dumps(mod_cmp, indent=2, default=str), encoding="utf-8")

    # ----------------------------------------- joint pipeline (descriptive)
    method = pd.read_csv(src / "method_ari.csv.gz")
    succ = method[method["status"] == "success"].dropna(subset=["ari"])
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

    # -------------------------------------------------- inventory check
    inv = {g: int((magg["metric_group"] == g).sum())
           for g in sorted(magg["metric_group"].dropna().unique())}
    declared = {}
    for e in registry["entries"]:
        declared[e["group"]] = declared.get(e["group"], 0) + 1
    zero_coverage = sorted(set(primary) - set(magg["metric_id"]))

    summary = {
        "run_id": args.run_id,
        "amendment_id": amend["amendment_id"],
        "package": "phase_d_corrected",
        "supersedes": "phase_d",
        "original_package_preserved": True,
        "utility_composition": {
            "metrics_from_phase_c_v1": int(u1["metric_id"].nunique()),
            "metrics_from_phase_c_v2": int(u2["metric_id"].nunique()),
            "repaired_metrics": sorted(repair),
        },
        "declared_inventory": declared,
        "observed_inventory_with_usable_utility": inv,
        "primary_total_declared": len(primary),
        "primary_with_usable_utility": int(magg["metric_id"].nunique()),
        "zero_coverage_primary_metrics": zero_coverage,
        "method_track_recomputed": False,
        "method_tables_authoritative_path": "phase_d/",
        "aggregation_rules": {
            "inferential_unit": "dataset",
            "source_balanced_macro": ("mean within FCPS / sklearn_shapes / "
                                      "real_pca, then the three means weighted "
                                      "equally"),
            "no_sentinel_ari": True,
            "frozen_before_outcomes": True,
            "unchanged_from_original_phase_d": True},
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    (out / "phase_d_corrected_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("  corrected Phase D -> %s" % out)
    print("  primary metrics with usable utility: %d / %d"
          % (summary["primary_with_usable_utility"], len(primary)))
    print("  inventory %s" % json.dumps(inv))
    if zero_coverage:
        print("  zero-coverage primary metrics: %s" % zero_coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
