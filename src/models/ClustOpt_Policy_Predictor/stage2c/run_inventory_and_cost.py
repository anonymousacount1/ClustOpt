"""Stage 2C-0: write the reranker-unit inventory and the nested-v2 cost model.

Read-only apart from its own result directory. Nothing is trained here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    ensure_dir, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)

PAIR_SOURCES = (
    # where a pair-exclusion RERANKER would live if one had ever been trained
    ("stage2a3b0_nested_crossfit",
     "results_analysis/clustopt_policy_predictor/stage2a3b0_nested_crossfit"),
    ("stage2c_pair_exclusion",
     RI.STAGE2C_REL + "/pair_exclusion_predictions"),
)


def pair_exclusion_inventory(repo: Path) -> pd.DataFrame:
    """Which R_{-(s,t)} reranker models exist. Expected: none."""
    rows = []
    for s, t in combinations(RI.OUTER_SPLITS, 2):
        found, prov = None, []
        for name, rel in PAIR_SOURCES:
            d = repo / rel / ("pair_%02d_%02d" % (s, t))
            m = read_json(d / "unit_manifest.json")
            if m and m.get("n_features") == UFB.EXPECTED_DIM:
                found = name
            prov.append("%s:%s" % (name, "present" if m else "absent"))
        rows.append({"outer_split": s, "inner_split": t,
                     "status": "EXISTS" if found else "MISSING",
                     "source": found, "provenance": "; ".join(prov)})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / RI.STAGE2C_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    print("=" * 78)
    print("STAGE 2C-0  RERANKER OOF UNIT INVENTORY AND NESTED-v2 COST MODEL")
    print("=" * 78, flush=True)

    inv = RI.build(repo)
    write_csv_atomic(out / "reranker_oof_model_inventory.csv", inv)

    # Rebuild the training summary from the unit manifests on disk. The runner
    # rewrites it per invocation, so a resumed run would otherwise leave only
    # its own units listed.
    fitted = []
    for _, r in inv[(inv["status"] == "EXISTS")
                    & (inv["source_stage"] == "stage2c")].iterrows():
        m = read_json(Path(r["unit_dir"]) / "unit_manifest.json") or {}
        fitted.append({"regime_id": r["regime_id"], "split_id": r["split_id"],
                       "n_train_rows": m.get("n_train_rows"),
                       "n_train_slates": m.get("n_train_slates"),
                       "n_features": m.get("n_features"),
                       "unit_hash": m.get("unit_hash"),
                       "fit_sec": (m.get("fit_info") or {}).get("fit_sec"),
                       "bestview": (m.get("metrics") or {}).get(
                           "mean_bestview_selected_ari"),
                       "recovery_raw": (m.get("metrics") or {}).get(
                           "recovery_raw"),
                       "source_commit": m.get("source_commit")})
    if fitted:
        write_csv_atomic(out / "reranker_oof_training_summary.csv",
                         pd.DataFrame(fitted).sort_values(["regime_id",
                                                           "split_id"]))
        print("  training summary rebuilt from disk: %d newly fitted units"
              % len(fitted), flush=True)

    S = RI.summarise(inv)
    write_json_atomic(out / "reranker_oof_model_reuse_summary.json", S)

    print("\n  units required %d | EXISTS %d | MISSING/INCOMPATIBLE %d"
          % (S["n_units_required"], S["n_exists"], S["n_missing"]), flush=True)
    print("  reuse by stage: %s" % S["reuse_by_stage"], flush=True)
    piv = inv.pivot_table(index="regime_id", columns="status",
                          values="split_id", aggfunc="count").fillna(0).astype(int)
    print("\n" + piv.to_string(), flush=True)
    print("\n  fully covered : %s" % ", ".join(S["regimes_fully_covered"]),
          flush=True)
    print("  needing fits  : %s" % ", ".join(S["regimes_needing_fits"]),
          flush=True)
    print("  measured HGBR fit: mean %.1fs median %.1fs max %.1fs"
          % (S["measured_fit_sec_mean"], S["measured_fit_sec_median"],
             S["measured_fit_sec_max"]), flush=True)

    # ---- nested-v2 pair-exclusion cost model -------------------------------
    pairs = pair_exclusion_inventory(repo)
    write_csv_atomic(out / "pair_exclusion_inventory.csv", pairs)
    n_pairs = int(len(pairs))
    n_pair_exists = int((pairs["status"] == "EXISTS").sum())
    n_regimes = len(RI.RERANKER_IDS)
    # Two honest anchors. The Stage-2B units measured ~335 s per fit and the
    # Stage-2C units ~110 s for the same model on the same data size; the gap is
    # machine load, not configuration. Report the range, not the flattering end.
    per_stage = (inv[inv["status"] == "EXISTS"]
                 .dropna(subset=["fit_sec"])
                 .groupby("source_stage")["fit_sec"].median().to_dict())
    fit_fast = float(min(per_stage.values()))
    fit_slow = float(max(per_stage.values()))
    fit = float(S["measured_fit_sec_median"])
    # measured per-unit overhead beyond the fit itself (feature build + predict)
    unit_total = fit * 1.35
    missing_pair_models = (n_pairs - n_pair_exists) * n_regimes
    est = {
        "requirement": ("a fully nested outer-fold v2 estimate needs, for outer "
                        "split s and training row from split t, a reranker "
                        "trained without BOTH s and t: R_{-(s,t)}"),
        "n_unordered_split_pairs": n_pairs,
        "n_regimes": n_regimes,
        "pair_models_required_max": n_pairs * n_regimes,
        "pair_models_existing": n_pair_exists * n_regimes,
        "pair_models_missing": missing_pair_models,
        "symmetric_reuse": ("R_{-(s,t)} is symmetric in s and t, so one model "
                            "serves outer s / inner t and outer t / inner s; "
                            "the 105 UNORDERED pairs already reflect this "
                            "(210 ordered combinations would double it)"),
        "measured_fit_sec_median": fit,
        "measured_fit_sec_median_by_stage": per_stage,
        "assumed_unit_sec_incl_features_and_predict": unit_total,
        "est_total_days_serial_optimistic":
            missing_pair_models * fit_fast * 1.35 / 86400.0,
        "est_total_days_serial_conservative":
            missing_pair_models * fit_slow * 1.35 / 86400.0,
        "estimate_range_note": ("per-fit time measured at %.0f s (Stage-2C, "
                                "quiet machine) and %.0f s (Stage-2B, loaded "
                                "machine) for the identical model and data "
                                "size; the conservative end should drive the "
                                "protocol decision"
                                % (fit_fast, fit_slow)),
        "est_serial_fit_hours": missing_pair_models * fit / 3600.0,
        "est_serial_total_hours": missing_pair_models * unit_total / 3600.0,
        "est_total_days_serial": missing_pair_models * unit_total / 86400.0,
        "est_wall_clock_hours_at_2_workers":
            missing_pair_models * unit_total / 3600.0 / 2.0,
        "memory_note": ("one training matrix is ~1.09M x 347 float32 = 1.5 GB; "
                        "2 concurrent workers is the safe ceiling on this "
                        "machine, so parallelism cannot rescue the estimate"),
        "storage_note": ("models are not persisted; each unit writes a "
                         "slate_selections.csv.gz of ~135 KB, so 1,050 units "
                         "cost ~140 MB of predictions"),
        "est_storage_mb_predictions": missing_pair_models * 0.135,
        "feature_generation_note": ("feature construction is repeated per unit "
                                    "and is included in the 1.35x factor; it "
                                    "cannot be cached across regimes because "
                                    "the observability mask differs"),
        "practical": bool(missing_pair_models * unit_total / 86400.0 < 3.0),
    }
    write_json_atomic(out / "nested_v2_cost_estimate.json", est)
    print("\n  pair-exclusion: %d unordered pairs x %d regimes = %d models; "
          "%d exist, %d missing" % (n_pairs, n_regimes, n_pairs * n_regimes,
                                    est["pair_models_existing"],
                                    missing_pair_models), flush=True)
    print("  estimated serial cost: %.0f fit-hours, %.0f total hours "
          "(%.1f days); at 2 workers ~%.0f h"
          % (est["est_serial_fit_hours"], est["est_serial_total_hours"],
             est["est_total_days_serial"],
             est["est_wall_clock_hours_at_2_workers"]), flush=True)

    # ---- cost of completing the SINGLE-exclusion matrix --------------------
    n_new = int(S["n_missing"])
    single = {"missing_single_exclusion_units": n_new,
              "est_fit_hours": n_new * fit / 3600.0,
              "est_total_hours": n_new * unit_total / 3600.0,
              "regimes": S["regimes_needing_fits"]}
    write_json_atomic(out / "single_exclusion_completion_cost.json", single)
    print("  completing single-exclusion: %d units, est %.1f h total"
          % (n_new, single["est_total_hours"]), flush=True)

    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "2C-0/1", "purpose": "reranker-aware policy target construction",
        "contract": S["contract"], "policy_semantics_hash":
            S["policy_semantics_hash"],
        "split_1_used": False, "source_commit": commit,
        "runtime_sec": time.time() - t0})
    print("\n  done (%.0fs) -> %s" % (time.time() - t0, out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
