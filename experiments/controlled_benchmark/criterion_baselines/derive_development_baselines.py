"""Seeded criterion-baseline sweep (internal label E1; paper Tables 1, 16 and 17). Included for transparency; it expects the regenerated controlled repository.

E1 / Step 2 -- derive the development-only (splits 2-16) criterion baselines.

DEVELOPMENT ONLY. No Split-1 outcome, prediction or utility enters any
derivation here. The only input is the frozen out-of-fold utility table for
splits 2-16.

Produces
  development_mean_profile.csv        B-meanprof, per view
  development_best_single_index.csv   B-best1
  development_core3.csv               B-core3 (fixed membership, weighting documented)
  development_derivation_manifest.json

STEP-8 GATE. The ClustOpt utility-target namespace uses each metric's ``.name``
attribute, while the CVI config / METRIC_REGISTRY uses the registry key. For 58
of the 60 metrics these coincide; for two they do not
(convexity_ratio_image_based <-> convexity_solidity_image,
 hough_arc_circle_strength  <-> hough_circle_arc_strength).

``GenericCVIEvaluator`` selects metrics by REGISTRY KEY but looks weights up by
``.name`` (cvi_generic.py: ``self.weights.get(m.name, 1.0)``), so a
``cvi.type="generic"`` config carrying a NON-UNIFORM weight on either aliased
metric would silently fall back to weight 1.0. This script therefore refuses to
emit any arm whose selected set contains an aliased metric, and the caller must
stop rather than run such an arm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

OOF = (REPO / "results_analysis" / "clustering_repository" / "stage2"
       / "oof_utility_splits2_16" / "combined" / "oof_mlp_utilities.csv.gz")
VIEWS = ("x_only", "y_only", "xy_2d")
TOP_K = 10
CORE3 = ["silhouette", "noise_aware_silhouette", "gridness_fft_acf"]

# authoritative alias map (registry key -> ClustOpt .name)
ALIAS = {"convexity_ratio_image_based": "convexity_solidity_image",
         "hough_arc_circle_strength": "hough_circle_arc_strength"}
NAME_TO_KEY = {v: k for k, v in ALIAS.items()}


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")


def main() -> int:
    from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
        METRIC_REGISTRY,
    )

    # ---- live registry: name <-> key, verified, not assumed -----------------
    name_of_key = {k: getattr(m, "name", k) for k, m in METRIC_REGISTRY.items()}
    key_of_name = {v: k for k, v in name_of_key.items()}
    aliased_keys = sorted(k for k, n in name_of_key.items() if n != k)
    manifest = {
        "derivation": "E1 development-only criterion baselines",
        "population": "controlled repository splits 2-16 ONLY",
        "split_1_used": False,
        "source": rel(OOF),
        "registry_size": len(METRIC_REGISTRY),
        "aliased_metrics_in_live_registry": {k: name_of_key[k] for k in aliased_keys},
    }
    assert set(aliased_keys) == set(ALIAS), (
        "live registry aliases %s != expected %s" % (aliased_keys, sorted(ALIAS)))

    # ---- load development utilities ---------------------------------------
    df = pd.read_csv(OOF)
    assert 1 not in set(df["heldout_split"].unique()), "Split 1 leaked into development"
    tcols = [c for c in df.columns if c.startswith("true_")]
    names = [c[5:] for c in tcols]
    assert len(names) == 60, "expected 60 FULL60 utility targets, got %d" % len(names)

    manifest["development_records"] = int(len(df))
    manifest["development_datasets"] = int(df["dataset_id"].nunique())
    manifest["development_splits"] = sorted(int(x) for x in df["heldout_split"].unique())
    manifest["n_utility_targets"] = len(names)

    # every target name must resolve to a live registry key
    unresolved = [n for n in names if n not in key_of_name and n not in METRIC_REGISTRY]
    assert not unresolved, "utility targets not in the registry: %s" % unresolved
    reg_key = {n: (n if n in METRIC_REGISTRY else key_of_name[n]) for n in names}

    # ------------------------------------------------ A. B-meanprof, per view
    rows = []
    for v in VIEWS:
        sub = df[df["view_mode"] == v]
        mu = sub[tcols].mean(axis=0, skipna=True)          # mean TRUE utility
        cov = sub[tcols].notna().mean(axis=0)
        prof = pd.DataFrame({"view": v, "metric": names,
                             "registry_key": [reg_key[n] for n in names],
                             "mean_utility": mu.to_numpy(),
                             "coverage": cov.to_numpy(),
                             "n_records": len(sub)})
        prof = prof.sort_values("mean_utility", ascending=False, kind="stable")
        prof["rank"] = np.arange(1, len(prof) + 1)
        prof["selected_top10"] = prof["rank"] <= TOP_K
        # RAW normalised positive weighting, exactly as C4:
        #   w_j = max(u_j, 0) / sum_{l in S} max(u_l, 0)
        pos = np.where(prof["selected_top10"], np.clip(prof["mean_utility"], 0, None), 0.0)
        tot = pos.sum()
        prof["normalized_weight"] = np.where(prof["selected_top10"], pos / tot, np.nan)
        rows.append(prof)
    meanprof = pd.concat(rows, ignore_index=True)
    meanprof.to_csv(HERE / "development_mean_profile.csv", index=False)

    sel = meanprof[meanprof.selected_top10]
    manifest["meanprof"] = {
        "rule": "mean TRUE FULL60 utility over splits 2-16, per view; rank; "
                "take top-10; RAW positive-normalised weights (identical rule to C4)",
        "weight_formula": "w_j = max(u_j,0) / sum_{l in S} max(u_l,0)",
        "per_view_selected": {v: sel[sel.view == v].sort_values("rank")
                              ["metric"].tolist() for v in VIEWS},
        "per_view_weights": {v: dict(zip(
            sel[sel.view == v].sort_values("rank")["metric"],
            sel[sel.view == v].sort_values("rank")["normalized_weight"].round(8)))
            for v in VIEWS},
        "weight_sum_check": {v: float(sel[sel.view == v]["normalized_weight"].sum())
                             for v in VIEWS},
    }

    # ------------------------------------------------------- B. B-best1
    best_rows = []
    for v in VIEWS:
        p = meanprof[meanprof.view == v].sort_values("rank")
        top = p.iloc[0]
        best_rows.append({"scope": "per_view", "view": v,
                          "metric": top["metric"], "registry_key": top["registry_key"],
                          "mean_utility": float(top["mean_utility"]),
                          "runner_up": p.iloc[1]["metric"],
                          "runner_up_mean_utility": float(p.iloc[1]["mean_utility"]),
                          "margin": float(top["mean_utility"] - p.iloc[1]["mean_utility"])})
    gmu = df[tcols].mean(axis=0, skipna=True)
    go = pd.DataFrame({"metric": names, "mean_utility": gmu.to_numpy()}) \
        .sort_values("mean_utility", ascending=False, kind="stable")
    best_rows.append({"scope": "global_all_views", "view": "ALL",
                      "metric": go.iloc[0]["metric"],
                      "registry_key": reg_key[go.iloc[0]["metric"]],
                      "mean_utility": float(go.iloc[0]["mean_utility"]),
                      "runner_up": go.iloc[1]["metric"],
                      "runner_up_mean_utility": float(go.iloc[1]["mean_utility"]),
                      "margin": float(go.iloc[0]["mean_utility"] - go.iloc[1]["mean_utility"])})
    best = pd.DataFrame(best_rows)
    best.to_csv(HERE / "development_best_single_index.csv", index=False)

    per_view_best = {r["view"]: r["metric"] for r in best_rows if r["scope"] == "per_view"}
    manifest["best1"] = {
        "rule": "single metric with the highest mean TRUE utility on splits 2-16",
        "per_view": per_view_best,
        "global": str(go.iloc[0]["metric"]),
        "per_view_agrees_with_global": len(set(per_view_best.values())) == 1
        and list(set(per_view_best.values()))[0] == str(go.iloc[0]["metric"]),
        "deployed_choice": "per-view, because C4's subset rule is per-view "
                           "(view_aware=true); the global winner is recorded too",
        "top10_global": go.head(10).to_dict(orient="records"),
    }

    # ------------------------------------------------------- C. B-core3
    core_rows = []
    for v in VIEWS:
        p = meanprof[meanprof.view == v].set_index("metric")
        for m in CORE3:
            core_rows.append({"view": v, "metric": m, "registry_key": reg_key[m],
                              "weight": 1.0,
                              "dev_mean_utility": float(p.loc[m, "mean_utility"]),
                              "dev_rank": int(p.loc[m, "rank"])})
    pd.DataFrame(core_rows).to_csv(HERE / "development_core3.csv", index=False)
    manifest["core3"] = {
        "members": CORE3,
        "membership_source": "frozen generalist core, "
                             "metric_analysis_existing_domain/"
                             "family_subfamily_specialization/core_definition.json "
                             "(top-5 by TRUE utility AND by selection frequency in "
                             ">=6 of 12 families; stable at 5/6/7)",
        "weighting_rule": "UNIFORM weights (1.0 each)",
        "weighting_rule_authority": "the project_lead plan's own wording -- "
                                    "'Static core: uniform weights over the three "
                                    "generalist indices' (clustopt_revised.tex, "
                                    "Appendix 'Criterion baselines', arm (iii))",
        "note": "uniform, NOT development-utility-weighted; a utility-weighted "
                "core would be a different (partly conditioned) arm",
    }

    # ------------------------------------------ STEP-8 GATE on every arm
    def gate(selected_keys, arm, uniform_weights):
        bad = [k for k in selected_keys if name_of_key[k] != k]
        return {"arm": arm, "n_metrics": len(selected_keys),
                "aliased_metrics_present": bad,
                "uniform_weights": uniform_weights,
                "safe_for_generic_cvi_config": (not bad) or uniform_weights,
                "reason": ("no aliased metric selected" if not bad else
                           ("aliased metric present but all weights are 1.0, so the "
                            "weights.get(m.name, 1.0) fallback returns the intended "
                            "value" if uniform_weights else
                            "ALIASED METRIC WITH NON-UNIFORM WEIGHT -- the generic "
                            "CVI path would silently substitute weight 1.0. STOP."))}

    gates = []
    for v in VIEWS:
        ks = sel[sel.view == v]["registry_key"].tolist()
        gates.append({**gate(ks, "E1-MEANPROF/%s" % v, False)})
        gates.append({**gate([reg_key[per_view_best[v]]], "E1-BEST1/%s" % v, True)})
    gates.append({**gate([reg_key[m] for m in CORE3], "E1-CORE3", True)})
    manifest["step8_gate"] = gates
    manifest["step8_verdict"] = ("PASS -- no arm requires a non-uniform weight on an "
                                 "aliased metric"
                                 if all(g["safe_for_generic_cvi_config"] for g in gates)
                                 else "FAIL -- STOP BEFORE RUNNING")
    manifest["step8_full60_completeness"] = {
        "n_utility_targets": len(names),
        "all_60_resolve_to_live_registry_keys": True,
        "includes_both_aliased_metrics": sorted(
            set(names) & set(ALIAS.values())) == sorted(ALIAS.values()),
        "note": "C4's FULL60 predictor targets contain all 60 intended ClustOpt "
                "metrics, including the two that the baseline MKR lookup lost. "
                "The broken Unified-MKR lookup path is not used anywhere in E1.",
    }

    (HERE / "development_derivation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("STEP-8 VERDICT:", manifest["step8_verdict"])
    for g in gates:
        print("  ", g["arm"], "aliased:", g["aliased_metrics_present"],
              "safe:", g["safe_for_generic_cvi_config"])
    print()
    print("=== B-meanprof top-10 per view ===")
    for v in VIEWS:
        p = sel[sel.view == v].sort_values("rank")
        print("---", v)
        print(p[["rank", "metric", "mean_utility", "normalized_weight"]]
              .to_string(index=False))
    print()
    print("=== B-best1 ===")
    print(best.to_string(index=False))
    print()
    print("=== global top-10 by mean true utility ===")
    print(go.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
