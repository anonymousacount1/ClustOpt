"""Stage 2C-2B: exact targets, three exact v2 fits, and the confirmation verdict.

Training targets for outer split ``s`` come from the pair-exclusion rerankers
``R_{-(s,t)}``; test targets keep the Stage-2C1 single-exclusion ``R_{-s}``
outcomes, which are already OOF-clean. Both the input features and the training
targets are therefore outer-fold clean -- the whole point of Stage 2C-2B.

The confirmation rules in :data:`RULES` are transcribed from the brief and were
fixed before any exact outcome was inspected.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, read_json, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E402,E501
    stats as S,
)
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.stage2c.run_pair_exclusion_rerankers import (  # noqa: E402,E501
    ALL_SPLITS, OUTER_SPLITS, OUT_REL, unit_dir,
)
from models.ClustOpt_Policy_Predictor.stage2c.run_v2_screening import (  # noqa: E402,E501
    FROZEN, NESTED_REL, S2A_DATA, V1_REL, align, bv, reranked_targets,
)
from models.ClustOpt_Policy_Predictor.training import evaluate_fold as EF  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import feature_sets as FSET  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import target_transforms as TT  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import train_fold as TFOLD  # noqa: E402,E501

APPROX_REL = "results_analysis/clustopt_policy_predictor/stage2c_v2_screening"
GZ = {"index": False, "compression": "gzip"}
KEY = ["dataset_id", "view_id"]

RULES = {
    "STRONG": ["pooled exact v2-fixed > 0", "pooled 95% CI entirely > 0",
               "3/3 outer splits positive",
               "exact effect retains >= 50% of the approximate effect"],
    "ADEQUATE": ["pooled exact v2-fixed > 0", ">= 2/3 outer splits positive",
                 "exact retains a positive meaningful fraction",
                 "gain not entirely explained by nesting bias"],
}


def exact_train_targets(repo: Path, s: int, pol_ids: Sequence[str]
                        ) -> pd.DataFrame:
    """(dataset, view) -> 20 policy columns, from R_{-(s,t)} for every t != s."""
    out = repo / OUT_REL
    mapping = pd.DataFrame(PM.build_mapping()).set_index("policy_id")
    frames = []
    for t in [x for x in ALL_SPLITS if x != s]:
        per_regime = {}
        for cid in RI.RERANKER_IDS:
            d = unit_dir(out, s, t, cid)
            f = pd.read_csv(_ext(d / "slate_selections.csv.gz"))
            per_regime[cid] = f.set_index(KEY)["selected_ari"]
        w = pd.DataFrame(per_regime)
        frames.append(w)
    W = pd.concat(frames)
    cols = {}
    for pid in pol_ids:
        cols["target__ari__%s" % pid] = W[mapping.loc[pid, "reranker_id"]]
    return pd.DataFrame(cols)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    approx = repo / APPROX_REL
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t_all = time.time()
    print("=" * 78)
    print("STAGE 2C-2B  EXACT NESTED v2 ON SPLITS %s" % list(OUTER_SPLITS))
    print("=" * 78, flush=True)

    order = json.loads((repo / S2A_DATA / "policy_target_order.json")
                       .read_text(encoding="utf-8"))
    pol_ids = [q["policy_id"] for q in sorted(order["policies"],
                                              key=lambda x: x["index"])]
    mapping = pd.DataFrame(PM.build_mapping()).set_index("policy_id")
    SINGLE = reranked_targets(repo, pol_ids)          # R_{-s} test-side targets
    root = repo / NESTED_REL

    fold_rows: List[Dict[str, Any]] = []
    per_rows: List[pd.DataFrame] = []
    bias_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for s in OUTER_SPLITS:
        t0 = time.time()
        F = FSET.load_fold(root, s)
        tri, tei = F["train_ident"], F["test_ident"]
        EX = exact_train_targets(repo, s, pol_ids)
        A_tr_exact = align(EX, tri)
        A_tr_approx = align(SINGLE, tri)
        A_te = align(SINGLE, tei)                     # OOF-clean, unchanged
        pd.DataFrame(A_tr_exact, columns=pol_ids).assign(
            dataset_id=tri["dataset_id"].to_numpy(),
            view_id=tri["view_id"].to_numpy()).to_csv(
            _ext(out / ("outer%d_exact_targets.csv.gz" % s)), **GZ)

        # ---- 19. target-landscape bias -----------------------------------
        cor = np.array([np.corrcoef(A_tr_exact[i], A_tr_approx[i])[0, 1]
                        if A_tr_exact[i].std() > 0 and A_tr_approx[i].std() > 0
                        else np.nan for i in range(len(A_tr_exact))])
        uniq = [i for i, q in enumerate(pol_ids)
                if mapping.loc[q, "weighting_mode"] == "RAW"]
        we, wa = A_tr_exact[:, uniq].argmax(1), A_tr_approx[:, uniq].argmax(1)
        src_e = np.array([mapping.loc[pol_ids[uniq[i]], "utility_source"]
                          for i in we])
        src_a = np.array([mapping.loc[pol_ids[uniq[i]], "utility_source"]
                          for i in wa])
        k_e = np.array([mapping.loc[pol_ids[uniq[i]], "k_mode"] for i in we])
        k_a = np.array([mapping.loc[pol_ids[uniq[i]], "k_mode"] for i in wa])
        ce = TT.transform(A_tr_exact, "CENTERED")
        ca = TT.transform(A_tr_approx, "CENTERED")
        bias_rows.append({
            "outer_split": s, "n_train_rows": int(len(A_tr_exact)),
            "mean_row_correlation": float(np.nanmean(cor)),
            "median_row_correlation": float(np.nanmedian(cor)),
            "mean_abs_target_shift": float(np.abs(A_tr_exact
                                                  - A_tr_approx).mean()),
            "max_abs_target_shift": float(np.abs(A_tr_exact
                                                 - A_tr_approx).max()),
            "mean_abs_centered_shift": float(np.abs(ce - ca).mean()),
            "regime_winner_switch_fraction": float((we != wa).mean()),
            "source_switch_fraction": float((src_e != src_a).mean()),
            "k_switch_fraction": float((k_e != k_a).mean()),
            "exact_mean_target": float(A_tr_exact.mean()),
            "approx_mean_target": float(A_tr_approx.mean())})

        # ---- 11. exact v2 --------------------------------------------------
        Fx = dict(F)
        Fx["train_ari"] = pd.DataFrame(A_tr_exact, columns=pol_ids)
        Fx["test_ari"] = pd.DataFrame(A_te, columns=pol_ids)
        res = TFOLD.run_fold(root, s, fold=Fx, **FROZEN)
        pr = res["predictions"]

        # ---- 12. exact fixed baseline (chosen on EXACT train targets) ------
        base = EF.fold_baselines(Fx, pol_ids)
        fx_idx = int(base["global_best_single_index"])
        ds_te = tei["dataset_id"].to_numpy()
        fixed_bv = bv(ds_te, A_te[:, fx_idx])

        # ---- 13/14. v1 and VBS ---------------------------------------------
        v1 = pd.read_csv(_ext(repo / V1_REL / ("fold_%02d_predictions.csv.gz" % s)))
        v1 = tei[KEY].merge(v1[KEY + ["selected_policy_index"]], on=KEY,
                            how="left", validate="one_to_one")
        v1_sel = v1["selected_policy_index"].to_numpy(int)
        v1_bv = bv(ds_te, A_te[np.arange(len(v1_sel)), v1_sel])
        v2_bv = bv(ds_te, pr["selected_ari"].to_numpy(float))
        vbs_bv = bv(ds_te, A_te.max(axis=1))
        idx = fixed_bv.index

        # ---- 20. prediction bias vs the approximate run --------------------
        ap = pd.read_csv(_ext(approx / ("rowlevel_fold_%02d.csv.gz" % s)))
        ap = tei[KEY].merge(ap[KEY + ["v2_selected_policy_index",
                                      "v2_selected_regime",
                                      "v2_selected_ari"]],
                            on=KEY, how="left", validate="one_to_one")
        ex_sel = pr["selected_policy_index"].to_numpy(int)
        ex_reg = np.array([mapping.loc[pol_ids[i], "reranker_id"]
                           for i in ex_sel])
        gap_ex = A_te.max(axis=1) - pr["selected_ari"].to_numpy(float)
        gap_ap = A_te.max(axis=1) - ap["v2_selected_ari"].to_numpy(float)
        pred_rows.append({
            "outer_split": s, "n_rows": int(len(ex_sel)),
            "selected_regime_switch_fraction": float(
                (ex_reg != ap["v2_selected_regime"].to_numpy()).mean()),
            "selected_policy_switch_fraction": float(
                (ex_sel != ap["v2_selected_policy_index"].to_numpy()).mean()),
            "mean_outcome_difference_exact_minus_approx": float(
                (pr["selected_ari"].to_numpy(float)
                 - ap["v2_selected_ari"].to_numpy(float)).mean()),
            "near_optimal_within_0.005_exact": float((gap_ex <= 0.005).mean()),
            "near_optimal_within_0.005_approx": float((gap_ap <= 0.005).mean())})

        d_fix = (v2_bv - fixed_bv).reindex(idx)
        fold_rows.append({
            "outer_split": s, "n_test_datasets": int(len(idx)),
            "exact_fixed_policy": pol_ids[fx_idx],
            "exact_fixed_regime": mapping.loc[pol_ids[fx_idx], "reranker_id"],
            "A_exact_fixed": float(fixed_bv.mean()),
            "B_exact_v1": float(v1_bv.mean()),
            "C_exact_v2": float(v2_bv.mean()),
            "D_exact_vbs": float(vbs_bv.mean()),
            "exact_delta_v2_minus_fixed": float(d_fix.mean()),
            "exact_delta_v2_minus_v1": float((v2_bv - v1_bv).reindex(idx).mean()),
            "fit_sec": float(res["fit_info"][0].get("fit_sec", np.nan)),
            "fold_sec": time.time() - t0})
        per_rows.append(pd.DataFrame({
            "dataset_id": idx, "outer_split": s,
            "A_exact_fixed": fixed_bv.reindex(idx).to_numpy(),
            "B_exact_v1": v1_bv.reindex(idx).to_numpy(),
            "C_exact_v2": v2_bv.reindex(idx).to_numpy(),
            "D_exact_vbs": vbs_bv.reindex(idx).to_numpy()}))
        print("   [outer %02d] A=%.5f (%s) B=%.5f C=%.5f D=%.5f  dC-A=%+.5f "
              "(%.0fs)" % (s, fixed_bv.mean(),
                           mapping.loc[pol_ids[fx_idx], "reranker_id"],
                           v1_bv.mean(), v2_bv.mean(), vbs_bv.mean(),
                           d_fix.mean(), time.time() - t0), flush=True)

    FOLDS = pd.DataFrame(fold_rows)
    write_csv_atomic(out / "per_outer_fold_results.csv", FOLDS)
    PER = pd.concat(per_rows, ignore_index=True)
    PER.to_csv(_ext(out / "per_dataset_exact_results.csv.gz"), **GZ)
    write_csv_atomic(out / "target_landscape_bias.csv", pd.DataFrame(bias_rows))
    write_csv_atomic(out / "prediction_bias.csv", pd.DataFrame(pred_rows))

    # ---- 17. pooled exact result ------------------------------------------
    A, B, C, D = (PER[c].to_numpy(float) for c in
                  ("A_exact_fixed", "B_exact_v1", "C_exact_v2", "D_exact_vbs"))
    head = float(D.mean() - A.mean())
    four = []
    for k, col, nm in (("A", "A_exact_fixed", "Exact Fixed + R"),
                       ("B", "B_exact_v1", "Exact v1 + R"),
                       ("C", "C_exact_v2", "Exact v2 + R"),
                       ("D", "D_exact_vbs", "Exact VBS + R")):
        x = PER[col].to_numpy(float)
        pt, lo, hi = S.bootstrap_ci(x, statistic="mean")
        four.append({"level": k, "name": nm, "mean_bestview": float(x.mean()),
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "delta_vs_A": float(x.mean() - A.mean()),
                     "recovery_of_A_to_D": float((x.mean() - A.mean()) / head)
                     if head > 1e-12 else float("nan")})
    FOUR = pd.DataFrame(four)
    write_csv_atomic(out / "four_level_exact_table.csv", FOUR)
    print("\n" + FOUR.round(6).to_string(index=False), flush=True)

    d = C - A
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    try:
        from scipy import stats as sst
        nz = d[np.abs(d) > 1e-12]
        wp = float(sst.wilcoxon(nz).pvalue) if len(nz) else 1.0
    except Exception:                                              # noqa: BLE001
        wp = float("nan")
    vf = {"contrast": "exact_v2_minus_exact_fixed", "n": int(len(d)),
          "mean_delta": float(pt), "median_delta": float(np.median(d)),
          "ci_lo": float(lo), "ci_hi": float(hi), "wilcoxon_p": wp,
          "wins_v2": int((d > 1e-12).sum()), "wins_fixed": int((d < -1e-12).sum()),
          "ties": int((np.abs(d) <= 1e-12).sum()),
          "split_macro_mean_delta": float(
              FOLDS["exact_delta_v2_minus_fixed"].mean()),
          "positive_splits": int((FOLDS["exact_delta_v2_minus_fixed"] > 0).sum())}
    write_csv_atomic(out / "exact_v2_vs_fixed.csv", pd.DataFrame([vf]))
    dv = C - B
    pv, vlo, vhi = S.bootstrap_ci(dv, statistic="mean")
    write_csv_atomic(out / "exact_v2_vs_v1.csv", pd.DataFrame([{
        "contrast": "exact_v2_minus_v1", "n": int(len(dv)),
        "mean_delta": float(pv), "ci_lo": float(vlo), "ci_hi": float(vhi),
        "wins_v2": int((dv > 1e-12).sum()),
        "wins_v1": int((dv < -1e-12).sum()),
        "ties": int((np.abs(dv) <= 1e-12).sum())}]))
    print("\n  exact v2 - fixed : %+.7f [%.6f, %.6f] p=%.3g W/L/T %d/%d/%d | "
          "%d/3 splits positive" % (vf["mean_delta"], vf["ci_lo"], vf["ci_hi"],
                                    wp, vf["wins_v2"], vf["wins_fixed"],
                                    vf["ties"], vf["positive_splits"]),
          flush=True)

    # ---- 15/16. approximate vs exact on the SAME splits --------------------
    ap_fold = pd.read_csv(_ext(approx / "outer_fold_results.csv"))
    ap_fold = ap_fold[ap_fold["outer_split"].isin(OUTER_SPLITS)]
    ap_per = pd.read_csv(_ext(approx / "per_dataset_results.csv.gz"))
    ap_per = ap_per[ap_per["outer_split"].isin(OUTER_SPLITS)]
    cmp_rows = []
    for _, r in ap_fold.iterrows():
        s = int(r["outer_split"])
        e = FOLDS[FOLDS["outer_split"] == s].iloc[0]
        cmp_rows.append({
            "outer_split": s,
            "approx_fixed": float(r["A_fixed_bestview"]),
            "exact_fixed": float(e["A_exact_fixed"]),
            "approx_v2": float(r["C_v2_bestview"]),
            "exact_v2": float(e["C_exact_v2"]),
            "approx_delta": float(r["delta_v2_minus_fixed"]),
            "exact_delta": float(e["exact_delta_v2_minus_fixed"]),
            "bias_delta": float(r["delta_v2_minus_fixed"]
                                - e["exact_delta_v2_minus_fixed"])})
    CMP = pd.DataFrame(cmp_rows)
    write_csv_atomic(out / "approx_vs_exact.csv", CMP)
    ap_delta_pool = float((ap_per["C_v2"] - ap_per["A_fixed"]).mean())
    retention = (float(vf["mean_delta"] / ap_delta_pool)
                 if abs(ap_delta_pool) > 1e-12 else float("nan"))
    summ = {
        "outer_splits": list(OUTER_SPLITS),
        "approx_pooled_delta": ap_delta_pool,
        "exact_pooled_delta": vf["mean_delta"],
        "bias_delta_pooled": ap_delta_pool - vf["mean_delta"],
        "effect_retention_ratio": retention,
        "approx_pooled_fixed": float(ap_per["A_fixed"].mean()),
        "exact_pooled_fixed": float(A.mean()),
        "approx_pooled_v2": float(ap_per["C_v2"].mean()),
        "exact_pooled_v2": float(C.mean()),
        "per_split": CMP.to_dict("records")}
    write_json_atomic(out / "approx_vs_exact_summary.json", summ)
    print("\n" + CMP.round(6).to_string(index=False), flush=True)
    print("  pooled approx delta %+.7f -> exact %+.7f | retention %.3f"
          % (ap_delta_pool, vf["mean_delta"], retention), flush=True)

    # ---- 18. exact headroom recovery --------------------------------------
    rec = float((C.mean() - A.mean()) / head) if head > 1e-12 else float("nan")
    write_json_atomic(out / "exact_headroom_recovery.json", {
        "label": "Sampled Exact Headroom Recovery",
        "outer_splits": list(OUTER_SPLITS),
        "exact_fixed": float(A.mean()), "exact_vbs": float(D.mean()),
        "exact_v2": float(C.mean()), "headroom": head,
        "recovery": rec,
        "caveat": "computed on the three sampled outer splits only; do NOT "
                  "generalise to all Splits 2-16"})

    # ---- 16. pre-registered confirmation ----------------------------------
    strong = {"1_pooled_delta_gt_0": bool(vf["mean_delta"] > 0),
              "2_ci_entirely_above_zero": bool(vf["ci_lo"] > 0),
              "3_all_three_splits_positive": bool(vf["positive_splits"] == 3),
              "4_retains_ge_50pct_of_approx": bool(retention >= 0.50)}
    adequate = {"1_pooled_delta_gt_0": bool(vf["mean_delta"] > 0),
                "2_at_least_2_of_3_positive": bool(vf["positive_splits"] >= 2),
                "3_positive_meaningful_fraction": bool(retention > 0.0),
                "4_not_entirely_bias": bool(vf["mean_delta"] > 0
                                            and retention > 0.0)}
    failed = {"1_pooled_delta_le_0": bool(vf["mean_delta"] <= 0),
              "2_at_most_one_split_positive": bool(vf["positive_splits"] <= 1),
              "3_effect_eliminated_or_reversed": bool(retention <= 0.0)}
    if all(strong.values()):
        verdict = "STRONG CONFIRMATION"
    elif any(failed.values()):
        verdict = "FAILED CONFIRMATION"
    elif all(adequate.values()):
        verdict = "ADEQUATE CONFIRMATION"
    else:
        verdict = "FAILED CONFIRMATION"
    cls = {"classification": verdict, "rules_preregistered": RULES,
           "strong_criteria": strong, "adequate_criteria": adequate,
           "failed_criteria": failed,
           "pooled_exact_delta": vf["mean_delta"], "ci": [vf["ci_lo"], vf["ci_hi"]],
           "wilcoxon_p": wp, "positive_splits": vf["positive_splits"],
           "effect_retention_ratio": retention,
           "sampled_exact_headroom_recovery": rec,
           "next_stage_auto_executed": False}
    write_json_atomic(out / "confirmation_classification.json", cls)
    print("\n  CONFIRMATION: %s" % verdict, flush=True)
    for k, v in strong.items():
        print("    STRONG %-32s %s" % (k, v), flush=True)

    # ---- 23. Stage 2C-3 plan (declaration only) ---------------------------
    write_json_atomic(out / "future_stage2c3_plan.json", {
        "status": "PROPOSED_NOT_EXECUTED",
        "A_final_v2": "one ExtraTrees fit on all Splits 2-16 using the Stage-2C1 "
                      "SINGLE-exclusion reranker-aware targets (no development "
                      "outer-test split needs excluding at final-training time)",
        "B_apply": "score Split-1 label-free dataset/view features",
        "C_select": "one of the frozen 20 policies / 10 regimes per row, frozen "
                    "tie-break",
        "D_lookup": "read the already-frozen Stage-2B5 reranker-assisted ONLINE "
                    "outcome; no clustering, Optuna, CVI or reranker inference",
        "E_compare": "Fixed + R, v1 + R, v2 + R, VBS + R on the frozen online "
                     "matrix",
        "estimated_compute": "single ExtraTrees fit (~60 s) plus a lookup join; "
                             "minutes, not hours",
        "labelling_requirement": "Split 1 is DEVELOPMENT-INFORMED for v2 because "
                                 "Stage-2B outcomes already steered the project; "
                                 "it is not an untouched final test",
        "raw_softmax_replay_diagnostic": {
            "status": "PROTOCOL_DEFINED_NOT_EVALUATED",
            "definition": "within the regime v2 selects, count how often the RAW "
                          "and SOFTMAX Stage-2B5 ONLINE outcomes differ",
            "must_not": "be used to train or tune the weighting-mode choice"},
        "do_not_execute_now": True})

    write_json_atomic(out / "stage2c2b_summary.json", {
        "four_level": FOUR.to_dict("records"), "exact_v2_vs_fixed": vf,
        "approx_vs_exact": summ, "classification": cls,
        "target_landscape_bias": bias_rows, "prediction_bias": pred_rows,
        "source_commit": commit, "runtime_sec": time.time() - t_all})
    print("\n  done (%.0fs)" % (time.time() - t_all), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
