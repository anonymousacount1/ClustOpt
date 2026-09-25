"""Stage 2C-2A: aggregate the screening, classify it, and plan the next step.

The GO thresholds below are transcribed verbatim from the Stage-2C2A brief and
are evaluated mechanically. They were fixed before any fold result was read and
are not adjusted here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E402,E501
    stats as S,
)
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI  # noqa: E402,E501

OUT_REL = "results_analysis/clustopt_policy_predictor/stage2c_v2_screening"
S2A_SEL = ("results_analysis/clustopt_policy_predictor/"
           "stage2a3b_model_selection")
STAGE2C1_GLOBAL_FIXED = 0.6196177489460544        # development reference only
STAGE2C1_V1_PLUS_R = 0.6151823448986208
STAGE2C1_VBS = 0.6432366127747722

# ---- pre-registered thresholds (Stage-2C2A brief section 15) ---------------
THRESHOLDS = {
    "STRONG_GO": {"mean_delta_gt_0": True, "ci_lo_gt_0": True,
                  "recovery_ge": 0.10, "positive_splits_ge": 12},
    "MODERATE_GO": {"mean_delta_gt_0": True, "recovery_gt": 0.05,
                    "positive_splits_ge": 10},
}


def paired(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    d = np.asarray(a, float) - np.asarray(b, float)
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    r = {"n": int(len(d)), "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
         "mean_delta": float(pt), "median_delta": float(np.median(d)),
         "ci_lo": float(lo), "ci_hi": float(hi),
         "wins_a": int((d > 1e-12).sum()), "wins_b": int((d < -1e-12).sum()),
         "ties": int((np.abs(d) <= 1e-12).sum()),
         "cohens_dz": float(S.cohens_dz(d))}
    try:
        from scipy import stats as sst
        nz = d[np.abs(d) > 1e-12]
        r["wilcoxon_p"] = float(sst.wilcoxon(nz).pvalue) if len(nz) else 1.0
    except Exception:                                              # noqa: BLE001
        r["wilcoxon_p"] = float("nan")
    return r


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    print("=" * 78)
    print("STAGE 2C-2A  SCREENING AGGREGATION AND CLASSIFICATION")
    print("=" * 78, flush=True)

    FOLDS = pd.read_csv(_ext(out / "outer_fold_results.csv"))
    PER = pd.read_csv(_ext(out / "per_dataset_results.csv.gz"))
    ROW = pd.concat([pd.read_csv(_ext(f)) for f in
                     sorted(out.glob("rowlevel_fold_*.csv.gz"))],
                    ignore_index=True)
    mapping = pd.DataFrame(PM.build_mapping()).set_index("policy_id")
    print("  %d folds | %d datasets | %d rows"
          % (len(FOLDS), len(PER), len(ROW)), flush=True)

    # ---- 12. four-level table ---------------------------------------------
    lv = {"A": "A_fixed", "B": "B_v1", "C": "C_v2", "D": "D_vbs"}
    names = {"A": "Outer-train-selected Fixed Policy + Reranker",
             "B": "Policy Predictor v1 + Reranker",
             "C": "Policy Predictor v2 + Reranker",
             "D": "VBS + Reranker"}
    A, B, C, D = (PER[lv[k]].to_numpy(float) for k in "ABCD")
    head_AD = float(D.mean() - A.mean())
    rows = []
    for k in "ABCD":
        x = PER[lv[k]].to_numpy(float)
        pt, lo, hi = S.bootstrap_ci(x, statistic="mean")
        rows.append({
            "level": k, "name": names[k], "mean_bestview": float(x.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "delta_vs_A": float(x.mean() - A.mean()),
            "delta_vs_B": float(x.mean() - B.mean()),
            "headroom_to_D": float(D.mean() - x.mean()),
            "recovery_of_A_to_D": float((x.mean() - A.mean()) / head_AD)
            if head_AD > 1e-12 else float("nan"),
            "status": "COMPUTED"})
    FOUR = pd.DataFrame(rows)
    write_csv_atomic(out / "four_level_screening_table.csv", FOUR)
    print("\n" + FOUR[["level", "name", "mean_bestview", "ci_lo", "ci_hi",
                       "delta_vs_A", "recovery_of_A_to_D"]]
          .round(6).to_string(index=False), flush=True)

    # ---- 13/14. primary contrasts -----------------------------------------
    vf = paired(C, A)
    vv = paired(C, B)
    fold_pos = int((FOLDS["delta_v2_minus_fixed"] > 0).sum())
    recovery = float((C.mean() - A.mean()) / head_AD) if head_AD > 1e-12 else np.nan
    write_csv_atomic(out / "v2_vs_fixed.csv", pd.DataFrame([{
        **vf, "contrast": "v2_minus_outer_train_fixed",
        "positive_outer_splits": fold_pos, "n_outer_splits": len(FOLDS),
        "headroom_recovery": recovery}]))
    write_csv_atomic(out / "v2_vs_v1.csv", pd.DataFrame([{
        **vv, "contrast": "v2_minus_v1",
        "positive_outer_splits": int((FOLDS["delta_v2_minus_v1"] > 0).sum())}]))
    print("\n  v2 - fixed : %+.7f [%.6f, %.6f] p=%.3g W/L/T %d/%d/%d | "
          "%d/%d splits positive | recovery %.4f"
          % (vf["mean_delta"], vf["ci_lo"], vf["ci_hi"], vf["wilcoxon_p"],
             vf["wins_a"], vf["wins_b"], vf["ties"], fold_pos, len(FOLDS),
             recovery), flush=True)
    print("  v2 - v1    : %+.7f [%.6f, %.6f] p=%.3g W/L/T %d/%d/%d"
          % (vv["mean_delta"], vv["ci_lo"], vv["ci_hi"], vv["wilcoxon_p"],
             vv["wins_a"], vv["wins_b"], vv["ties"]), flush=True)

    # ---- 17. selection diagnostics ----------------------------------------
    freq = (ROW.groupby("v2_selected_regime").size()
            .rename("n").reset_index())
    freq["fraction"] = freq["n"] / len(ROW)
    freq["source"] = freq["v2_selected_regime"].str.split("_").str[0]
    freq["k_mode"] = freq["v2_selected_regime"].str.split("_", n=1).str[1]
    write_csv_atomic(out / "v2_policy_selection_frequencies.csv",
                     freq.sort_values("n", ascending=False))
    pr = freq["fraction"].to_numpy()
    entropy = float(-(pr * np.log(np.maximum(pr, 1e-12))).sum())
    exact_hit = float((ROW["v2_selected_regime"]
                       == ROW["oracle_regime"]).mean())
    gap = ROW["row_oracle_ari"] - ROW["v2_selected_ari"]
    near = {("within_%s" % t): float((gap <= float(t)).mean())
            for t in ("0.001", "0.005", "0.01")}
    raw_sel = ROW["v2_selected_policy"].map(mapping["weighting_mode"])
    diag = {
        "n_rows": int(len(ROW)),
        "regime_entropy_nats": entropy,
        "regime_entropy_normalised": entropy / np.log(10),
        "n_distinct_regimes_selected": int(freq.shape[0]),
        "source_shares": ROW["v2_selected_regime"].str.split("_").str[0]
        .value_counts(normalize=True).to_dict(),
        "k_shares": ROW["v2_selected_regime"].str.split("_", n=1).str[1]
        .value_counts(normalize=True).to_dict(),
        "raw_softmax_shares": raw_sel.value_counts(normalize=True).to_dict(),
        "oracle_regime_exact_hit_rate": exact_hit,
        **near,
        "note": "top-1 policy accuracy is not the primary metric: the target is "
                "tie-heavy (RAW/SOFTMAX are exact duplicates and 12% of rows "
                "have all policies tied)",
    }
    write_json_atomic(out / "v2_selection_diagnostics.json", diag)
    print("\n  regimes selected %d | entropy %.3f nats | oracle-regime hit "
          "%.4f | within .001/.005/.01 = %.3f/%.3f/%.3f"
          % (diag["n_distinct_regimes_selected"], entropy, exact_hit,
             near["within_0.001"], near["within_0.005"], near["within_0.01"]),
          flush=True)

    # ---- 18. regret --------------------------------------------------------
    REG = pd.DataFrame([
        {"arm": "v2", "mean_row_regret": float(gap.mean()),
         "median_row_regret": float(gap.median())},
        {"arm": "fixed",
         "mean_row_regret": float((ROW["row_oracle_ari"]
                                   - ROW["fixed_ari"]).mean()),
         "median_row_regret": float((ROW["row_oracle_ari"]
                                     - ROW["fixed_ari"]).median())},
        {"arm": "v1",
         "mean_row_regret": float((ROW["row_oracle_ari"]
                                   - ROW["v1_ari"]).mean()),
         "median_row_regret": float((ROW["row_oracle_ari"]
                                     - ROW["v1_ari"]).median())}])
    REG["dataset_headroom_recovered"] = [
        recovery, 0.0, float((B.mean() - A.mean()) / head_AD)]
    write_csv_atomic(out / "v2_regret_analysis.csv", REG)

    # ---- 19. view-level ----------------------------------------------------
    vw = []
    for v, g in ROW.groupby("view_id"):
        vw.append({"view_id": v, "n_rows": int(len(g)),
                   "A_fixed": float(g["fixed_ari"].mean()),
                   "B_v1": float(g["v1_ari"].mean()),
                   "C_v2": float(g["v2_selected_ari"].mean()),
                   "D_oracle": float(g["row_oracle_ari"].mean()),
                   "delta_v2_minus_fixed": float((g["v2_selected_ari"]
                                                  - g["fixed_ari"]).mean())})
    VIEW = pd.DataFrame(vw)
    write_csv_atomic(out / "v2_view_analysis.csv", VIEW)
    print("\n" + VIEW.round(6).to_string(index=False), flush=True)

    # ---- 20. family / subfamily -------------------------------------------
    fam = ROW.groupby("family").apply(
        lambda g: pd.Series({
            "n_rows": len(g),
            "delta_v2_minus_fixed": float((g["v2_selected_ari"]
                                           - g["fixed_ari"]).mean()),
            "C_v2": float(g["v2_selected_ari"].mean()),
            "A_fixed": float(g["fixed_ari"].mean())}),
        include_groups=False).reset_index()
    write_csv_atomic(out / "v2_family_analysis.csv",
                     fam.sort_values("delta_v2_minus_fixed"))
    sub = ROW.groupby("subfamily").apply(
        lambda g: pd.Series({
            "n_rows": len(g),
            "delta_v2_minus_fixed": float((g["v2_selected_ari"]
                                           - g["fixed_ari"]).mean())}),
        include_groups=False).reset_index()
    sub = sub[sub["n_rows"] >= 90]
    write_csv_atomic(out / "v2_subfamily_analysis.csv",
                     sub.sort_values("delta_v2_minus_fixed"))

    # ---- 16. calibration control ------------------------------------------
    cal_path = out / "approx_calibration_folds.csv"
    if cal_path.exists():
        cal = pd.read_csv(_ext(cal_path))
        exact = pd.read_csv(_ext(repo / S2A_SEL / "all_configuration_fold_metrics.csv"))
        exact = exact[exact["config_id"]
                      == "EXTRATREES__UNIFIED__META_UTILITIES__CENTERED"]
        col = ("bestview_selector_ari" if "bestview_selector_ari" in exact.columns
               else "bestview")
        ex = exact.set_index("outer_split")[col] if "outer_split" in exact.columns \
            else exact.set_index(exact.columns[1])[col]
        m = cal.set_index("outer_split").join(ex.rename("exact"))
        m["delta"] = m["approx_old_v1_bestview"] - m["exact"]
        write_csv_atomic(out / "approx_calibration_folds.csv", m.reset_index())
        calib = {
            "available": True,
            "approx_old_v1_macro_bestview": float(
                m["approx_old_v1_bestview"].mean()),
            "exact_nested_old_v1_macro_bestview": float(m["exact"].mean()),
            "delta_approx_minus_exact": float(m["delta"].mean()),
            "delta_per_fold_min": float(m["delta"].min()),
            "delta_per_fold_max": float(m["delta"].max()),
            "n_folds": int(len(m)),
            "what_it_is": "the SAME simplification (non-fully-nested training "
                          "targets, identical nested input features and frozen "
                          "ExtraTrees) applied to the OLD pre-reranker problem, "
                          "where a fully nested answer already exists",
            "what_it_is_not": "a formal quantification of the Stage-2C2A bias",
        }
    else:
        calib = {"available": False,
                 "reason": "calibration folds were not generated"}
    write_json_atomic(out / "approx_protocol_calibration.json", calib)
    if calib["available"]:
        print("\n  calibration: approx %.6f vs exact nested %.6f -> delta %+.6f"
              % (calib["approx_old_v1_macro_bestview"],
                 calib["exact_nested_old_v1_macro_bestview"],
                 calib["delta_approx_minus_exact"]), flush=True)

    # ---- 15. pre-registered classification ---------------------------------
    strong = {
        "1_mean_delta_gt_0": bool(vf["mean_delta"] > 0),
        "2_ci_entirely_above_zero": bool(vf["ci_lo"] > 0),
        "3_recovery_ge_0.10": bool(recovery >= 0.10),
        "4_positive_splits_ge_12": bool(fold_pos >= 12)}
    moderate = {
        "1_mean_delta_gt_0": bool(vf["mean_delta"] > 0),
        "2_recovery_gt_0.05": bool(recovery > 0.05),
        "3_positive_splits_ge_10": bool(fold_pos >= 10)}
    nogo = {
        "1_mean_delta_le_0": bool(vf["mean_delta"] <= 0),
        "2_recovery_le_0": bool(recovery <= 0),
        "3_inconsistent_splits": bool(fold_pos < 10)}
    if all(strong.values()):
        verdict = "STRONG GO"
    elif all(moderate.values()):
        verdict = "MODERATE GO"
    elif any(nogo.values()):
        verdict = "NO GO"
    else:
        verdict = "MODERATE GO (partial)"
    cls = {"classification": verdict,
           "thresholds_preregistered": THRESHOLDS,
           "strong_go_criteria": strong, "moderate_go_criteria": moderate,
           "no_go_criteria": nogo,
           "mean_delta_v2_minus_fixed": vf["mean_delta"],
           "ci": [vf["ci_lo"], vf["ci_hi"]],
           "wilcoxon_p": vf["wilcoxon_p"],
           "headroom_recovery": recovery,
           "positive_outer_splits": fold_pos, "n_outer_splits": len(FOLDS),
           "protocol": "APPROXIMATE DEVELOPMENT SCREENING -- residual "
                       "optimistic-bias risk in training-target generation",
           "next_stage_auto_executed": False}
    write_json_atomic(out / "screening_classification.json", cls)
    print("\n  CLASSIFICATION: %s" % verdict, flush=True)
    for k, v in strong.items():
        print("    STRONG %-28s %s" % (k, v), flush=True)

    # ---- 23. sampled exact nesting plan (declaration only) -----------------
    splits = sorted(FOLDS["outer_split"].tolist())
    chosen = [splits[0], splits[len(splits) // 2], splits[-1]]
    n_pairs = len(list(combinations(splits, 2))) - len(
        list(combinations([s for s in splits if s not in chosen], 2)))
    cost = json.loads((repo / RI.STAGE2C_REL / "nested_v2_cost_estimate.json")
                      .read_text(encoding="utf-8"))
    fast = min(cost["measured_fit_sec_median_by_stage"].values())
    slow = max(cost["measured_fit_sec_median_by_stage"].values())
    plan = {
        "status": "PROPOSED_NOT_EXECUTED",
        "selection_rule": "deterministic and outcome-blind: the first, middle "
                          "and last outer split by ascending split id "
                          "(%s) -- chosen from the split inventory alone, "
                          "never from v2 performance" % chosen,
        "outer_splits": chosen,
        "pairs_touching_chosen_splits": n_pairs,
        "regimes": 10,
        "pair_exclusion_models_required": n_pairs * 10,
        "existing": 0,
        "est_serial_hours_optimistic": n_pairs * 10 * fast * 1.35 / 3600.0,
        "est_serial_hours_conservative": n_pairs * 10 * slow * 1.35 / 3600.0,
        "memory_plan": "serial, one worker; one training matrix is ~1.4 GB and "
                       "concurrency above 2 is unsafe on this machine",
        "confirmation_criterion": "on the three sampled outer splits, the exactly "
                                  "nested v2-minus-fixed delta keeps the sign and "
                                  "at least half the magnitude of the screening "
                                  "delta; a sign flip or collapse toward zero "
                                  "would attribute the screening result to the "
                                  "training-target shortcut",
        "do_not_execute_now": True,
    }
    write_json_atomic(out / "future_sampled_nesting_plan.json", plan)
    print("  sampled-nesting plan: splits %s -> %d pairs x 10 = %d models, "
          "%.0f-%.0f h serial" % (chosen, n_pairs, n_pairs * 10,
                                  plan["est_serial_hours_optimistic"],
                                  plan["est_serial_hours_conservative"]),
          flush=True)

    rt = {"fold_fit_sec_mean": float(FOLDS["fit_sec"].mean()),
          "fold_total_sec_mean": float(FOLDS["fold_sec"].mean()),
          "total_sec": float(FOLDS["fold_sec"].sum()),
          "n_folds": int(len(FOLDS)),
          "artifact_mb": float(sum(f.stat().st_size for f in out.glob("*"))
                               / 2**20)}
    write_json_atomic(out / "runtime_summary.json", rt)
    write_json_atomic(out / "stage2c2a_summary.json", {
        "four_level": FOUR.to_dict("records"),
        "v2_vs_fixed": vf, "v2_vs_v1": vv,
        "classification": cls, "diagnostics": diag,
        "calibration": calib, "plan": plan,
        "development_references": {
            "stage2c1_global_fixed": STAGE2C1_GLOBAL_FIXED,
            "stage2c1_v1_plus_r": STAGE2C1_V1_PLUS_R,
            "stage2c1_vbs": STAGE2C1_VBS},
        "source_commit": commit, "runtime_sec": time.time() - t0})
    print("\n  done (%.0fs)" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
