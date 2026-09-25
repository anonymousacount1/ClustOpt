"""Stage 2C-3 phase 2: evaluate the frozen selections on the Stage-2B5 matrix.

Re-validates the pre-outcome freeze before reading a single outcome, then does a
pure lookup of ``RerankedARI(d, v, p)`` for policies chosen in phase 1. Nothing
is refitted and no selection can change.
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
from models.ClustOpt_Policy_Predictor.stage2c.run_final_v2_split1_replay import (  # noqa: E402,E501
    AC_BV, B5_REL, KEY, OUT_REL, S2A_DATA, sha_file,
)

GZ = {"index": False, "compression": "gzip"}
AC_MATCHED = "AutoClust_Extended_InDomain_same_search_space"
AC_NATIVE = "AutoClust_Extended_InDomain"
AC_MATCHED_REF = 0.6144995191007991
AC_NATIVE_REF = 0.6170597383643214
B5_VBS_REF = 0.6567603388046713
B5_HYBRID_REF = 0.673114272109875
B5_HINDSIGHT_FIXED = ("knn_top10_raw", 0.6160307817072919)


def paired(a, b, la: str, lb: str, name: str) -> Dict[str, Any]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    r = {"comparison": name, "arm_a": la, "arm_b": lb, "n": int(len(d)),
         "mean_a": float(a.mean()), "mean_b": float(b.mean()),
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


def verdict(r: Dict[str, Any]) -> str:
    if r["ci_lo"] > 0 and r["wilcoxon_p"] < 0.05:
        return "statistically outperforms"
    if r["ci_hi"] < 0 and r["wilcoxon_p"] < 0.05:
        return "statistically underperforms"
    if r["mean_delta"] > 0:
        return "numerically exceeds but is statistically indistinguishable"
    if r["mean_delta"] < 0:
        return "numerically below but statistically indistinguishable"
    return "exactly tied"


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
    print("STAGE 2C-3  SPLIT-1 ONLINE REPLAY EVALUATION")
    print("=" * 78, flush=True)

    man = read_json(out / "pre_outcome_selection_manifest.json")
    for k, f in (("split1_v1_selections_hash", "split1_v1_policy_selections.csv.gz"),
                 ("split1_v2_selections_hash", "split1_v2_policy_selections.csv.gz"),
                 ("development_fixed_policy_hash",
                  "development_selected_fixed_policy.json")):
        if man.get(k) != sha_file(out / f):
            raise SystemExit("FROZEN ARTIFACT CHANGED: %s" % f)
    print("  freeze validated -- outcome access authorised", flush=True)

    order = json.loads((repo / S2A_DATA / "policy_target_order.json")
                       .read_text(encoding="utf-8"))
    pol_ids = [q["policy_id"] for q in sorted(order["policies"],
                                              key=lambda x: x["index"])]
    mapping = pd.DataFrame(PM.build_mapping()).set_index("policy_id")
    fixed = read_json(out / "development_selected_fixed_policy.json")

    # ---- frozen Stage-2B5 online matrix -----------------------------------
    b5 = pd.read_csv(_ext(repo / B5_REL / "per_run_reranker_results.csv.gz"),
                     usecols=["dataset_id", "view_id", "policy_id",
                              "reranked_selected_ari", "original_selected_ari",
                              "visited_oracle_ari"])
    M = b5.pivot_table(index=KEY, columns="policy_id",
                       values="reranked_selected_ari")[pol_ids]
    print("  online matrix: %d rows x %d policies | %d datasets"
          % (M.shape[0], M.shape[1], M.index.get_level_values(0).nunique()),
          flush=True)
    if M.shape != (3165, 20) or M.isna().any().any():
        raise AssertionError("online matrix is %s with %d NaN"
                             % (M.shape, int(M.isna().sum().sum())))

    v1 = pd.read_csv(_ext(out / "split1_v1_policy_selections.csv.gz"))
    v2 = pd.read_csv(_ext(out / "split1_v2_policy_selections.csv.gz"))
    idx = pd.MultiIndex.from_arrays([v2["dataset_id"], v2["view_id"]])
    A = M.reindex(idx).to_numpy(float)
    ds = v2["dataset_id"].to_numpy()
    vw = v2["view_id"].to_numpy()
    fx_i = int(fixed["policy_index"])

    row = pd.DataFrame({
        "dataset_id": ds, "view_id": vw,
        "fixed_ari": A[:, fx_i],
        "v1_ari": A[np.arange(len(A)), v1["selected_policy_index"].to_numpy(int)],
        "v2_ari": A[np.arange(len(A)), v2["selected_policy_index"].to_numpy(int)],
        "row_oracle_ari": A.max(axis=1),
        "v2_selected_regime": v2["selected_regime_id"],
        "v1_selected_regime": v1["selected_regime_id"],
        "oracle_regime": [mapping.loc[pol_ids[i], "reranker_id"]
                          for i in A.argmax(axis=1)]})
    assign = pd.read_csv(_ext(repo / "results_analysis/clustering_repository/"
                              "analyzed_data/experiment_splits/"
                              "dataset_split_assignments.csv"),
                         usecols=["dataset_id", "family_id", "subfamily_id"])
    row = row.merge(assign.drop_duplicates("dataset_id"), on="dataset_id",
                    how="left")
    row.to_csv(_ext(out / "per_view_online_results.csv.gz"), **GZ)

    g = row.groupby("dataset_id")
    D = pd.DataFrame({
        "A_fixed": g["fixed_ari"].max(), "B_v1": g["v1_ari"].max(),
        "C_v2": g["v2_ari"].max(), "D_vbs": g["row_oracle_ari"].max()})
    D.reset_index().to_csv(_ext(out / "per_dataset_online_results.csv.gz"), **GZ)

    four = []
    head = float(D["D_vbs"].mean() - D["A_fixed"].mean())
    names = {"A_fixed": "Development-selected Fixed Policy + Reranker",
             "B_v1": "Final Policy Predictor v1 + Reranker",
             "C_v2": "Final Policy Predictor v2 + Reranker",
             "D_vbs": "Reranker-assisted online policy VBS"}
    for c, nm in names.items():
        x = D[c].to_numpy(float)
        pt, lo, hi = S.bootstrap_ci(x, statistic="mean")
        four.append({"level": c[0], "name": nm, "mean_bestview": float(x.mean()),
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "delta_vs_fixed": float(x.mean() - D["A_fixed"].mean()),
                     "recovery_of_fixed_to_vbs":
                         float((x.mean() - D["A_fixed"].mean()) / head)
                         if head > 1e-12 else float("nan")})
    FOUR = pd.DataFrame(four)
    write_csv_atomic(out / "four_level_online_table.csv", FOUR)
    print("\n" + FOUR.round(7).to_string(index=False), flush=True)

    vf = paired(D["C_v2"], D["A_fixed"], "v2+R", "Fixed+R", "v2_minus_fixed")
    vv = paired(D["C_v2"], D["B_v1"], "v2+R", "v1+R", "v2_minus_v1")
    write_csv_atomic(out / "v2_vs_fixed.csv", pd.DataFrame([vf]))
    write_csv_atomic(out / "v2_vs_v1.csv", pd.DataFrame([vv]))
    print("\n  v2 - fixed: %+.7f [%.6f, %.6f] p=%.3g W/L/T %d/%d/%d"
          % (vf["mean_delta"], vf["ci_lo"], vf["ci_hi"], vf["wilcoxon_p"],
             vf["wins_a"], vf["wins_b"], vf["ties"]), flush=True)
    print("  v2 - v1   : %+.7f [%.6f, %.6f] p=%.3g W/L/T %d/%d/%d"
          % (vv["mean_delta"], vv["ci_lo"], vv["ci_hi"], vv["wilcoxon_p"],
             vv["wins_a"], vv["wins_b"], vv["ties"]), flush=True)

    write_json_atomic(out / "online_headroom_recovery.json", {
        "fixed_online": float(D["A_fixed"].mean()),
        "vbs_online": float(D["D_vbs"].mean()),
        "hybrid_visited_oracle_reference": B5_HYBRID_REF,
        "policy_headroom": head,
        "v2_online": float(D["C_v2"].mean()),
        "v1_online": float(D["B_v1"].mean()),
        "v2_recovery": float((D["C_v2"].mean() - D["A_fixed"].mean()) / head),
        "v1_recovery": float((D["B_v1"].mean() - D["A_fixed"].mean()) / head),
        "v2_gap_to_vbs": float(D["D_vbs"].mean() - D["C_v2"].mean()),
        "vbs_reconstructed_vs_stage2b5_reference": {
            "reconstructed": float(D["D_vbs"].mean()),
            "stage2b5_reference": B5_VBS_REF,
            "abs_diff": abs(float(D["D_vbs"].mean()) - B5_VBS_REF)},
        "aggregation": "ratio of aggregate means, not mean of per-dataset ratios"})

    # ---- AutoClust, both arms separately -----------------------------------
    # The Stage-2B5 package stored only the NATIVE arms (that was the Stage-2B5D
    # finding); the matched arm lives in the provenance-audit artifact, which
    # carries all four arms with per-dataset Best-View.
    ac = pd.read_csv(_ext(repo / "results_analysis/clustering_repository/"
                          "autoclust_extended_result_provenance_audit/"
                          "stage2b5d_autoclust_arms_bestview.csv.gz"))
    ac = (ac.rename(columns={"directory_id": "method"})
          .dropna(subset=["bestview_ari"]))
    for arm, ref, fn in ((AC_MATCHED, AC_MATCHED_REF,
                          "autoclust_matched_comparison.csv"),
                         (AC_NATIVE, AC_NATIVE_REF,
                          "autoclust_native_comparison.csv")):
        s = ac[ac["method"] == arm].set_index("dataset_id")["bestview_ari"]
        common = D.index.intersection(s.index)
        r = paired(D.loc[common, "C_v2"], s.loc[common], "v2+R", arm,
                   "v2_minus_%s" % arm)
        r["autoclust_reference_value"] = ref
        r["reconstruction_abs_diff"] = abs(float(s.mean()) - ref)
        r["interpretation"] = verdict(r)
        r["search_space_role"] = ("matched (PRIMARY)" if arm == AC_MATCHED
                                  else "native (secondary)")
        write_csv_atomic(out / fn, pd.DataFrame([r]))
        print("  vs %-46s %+.7f [%.6f, %.6f] p=%.3g -> %s"
              % (arm, r["mean_delta"], r["ci_lo"], r["ci_hi"],
                 r["wilcoxon_p"], r["interpretation"]), flush=True)

    # ---- hindsight fixed diagnostic ---------------------------------------
    hp, hv = B5_HINDSIGHT_FIXED
    hb = row.assign(a=A[:, pol_ids.index(hp)]).groupby("dataset_id")["a"].max()
    write_json_atomic(out / "hindsight_fixed_diagnostic.json", {
        "development_selected_policy": fixed["policy_id"],
        "development_selected_online": float(D["A_fixed"].mean()),
        "stage2b5_hindsight_best_policy": hp,
        "stage2b5_hindsight_reference": hv,
        "hindsight_reconstructed_online": float(hb.mean()),
        "same_policy": bool(fixed["policy_id"] == hp),
        "note": "the hindsight policy's 'best' identity was established USING "
                "Split-1 outcomes; it is a diagnostic upper benchmark among "
                "fixed policies, never a deployable baseline"})

    # ---- RAW/SOFTMAX online diagnostic (does not alter any selection) ------
    rs = []
    for rid, gm in mapping.groupby("reranker_id"):
        raw = gm[gm["weighting_mode"] == "RAW"].index[0]
        sof = gm[gm["weighting_mode"] != "RAW"].index[0]
        ar, asf = A[:, pol_ids.index(raw)], A[:, pol_ids.index(sof)]
        sel = (row["v2_selected_regime"] == rid).to_numpy()
        rs.append({"regime_id": rid, "n_rows_selected_by_v2": int(sel.sum()),
                   "raw_wins": int((ar > asf + 1e-12).sum()),
                   "softmax_wins": int((asf > ar + 1e-12).sum()),
                   "ties": int((np.abs(ar - asf) <= 1e-12).sum()),
                   "mean_abs_online_difference": float(np.abs(ar - asf).mean()),
                   "weighting_oracle_gain": float(
                       (np.maximum(ar, asf) - ar).mean())})
    RS = pd.DataFrame(rs)
    write_csv_atomic(out / "raw_softmax_online_diagnostic.csv", RS)
    sel_i = v2["selected_policy_index"].to_numpy(int)
    twin = np.array([pol_ids.index(
        mapping[(mapping["reranker_id"] == mapping.loc[pol_ids[i], "reranker_id"])
                & (mapping["weighting_mode"]
                   != mapping.loc[pol_ids[i], "weighting_mode"])].index[0])
        for i in sel_i])
    chosen = A[np.arange(len(A)), sel_i]
    other = A[np.arange(len(A)), twin]
    wo_gain = float(pd.DataFrame({"d": ds, "a": np.maximum(chosen, other)})
                    .groupby("d")["a"].max().mean() - D["C_v2"].mean())
    write_json_atomic(out / "raw_softmax_online_summary.json", {
        "v2_raw_softmax_tie_rate": float(v2["raw_softmax_tie"].mean()),
        "rows_where_chosen_beats_twin": int((chosen > other + 1e-12).sum()),
        "rows_where_twin_beats_chosen": int((other > chosen + 1e-12).sum()),
        "rows_tied": int((np.abs(chosen - other) <= 1e-12).sum()),
        "mean_abs_online_difference": float(np.abs(chosen - other).mean()),
        "hypothetical_weighting_mode_oracle_gain_bestview": wo_gain,
        "selection_altered_by_this_diagnostic": False,
        "note": "diagnostic only; the weighting-mode oracle is NOT deployable "
                "because fixed-32 supervision cannot distinguish RAW from "
                "SOFTMAX"})

    # ---- selection behaviour / near-optimality ----------------------------
    freq = row.groupby("v2_selected_regime").size().rename("n").reset_index()
    freq["fraction"] = freq["n"] / len(row)
    freq["source"] = freq["v2_selected_regime"].str.split("_").str[0]
    freq["k_mode"] = freq["v2_selected_regime"].str.split("_", n=1).str[1]
    write_csv_atomic(out / "policy_selection_frequencies.csv",
                     freq.sort_values("n", ascending=False))
    gap = row["row_oracle_ari"] - row["v2_ari"]
    pr = freq["fraction"].to_numpy()
    write_csv_atomic(out / "online_near_optimality.csv", pd.DataFrame([{
        "arm": "v2", "exact_oracle_regime_hit": float(
            (row["v2_selected_regime"] == row["oracle_regime"]).mean()),
        "within_0.001": float((gap <= 0.001).mean()),
        "within_0.005": float((gap <= 0.005).mean()),
        "within_0.01": float((gap <= 0.01).mean()),
        "mean_row_regret": float(gap.mean()),
        "median_row_regret": float(gap.median()),
        "regime_entropy_nats": float(-(pr * np.log(np.maximum(pr, 1e-12))).sum()),
        "n_regimes_selected": int(len(freq))}]))

    vw_rows = []
    for v, gg in row.groupby("view_id"):
        vw_rows.append({"view_id": v, "n_rows": int(len(gg)),
                        "A_fixed": float(gg["fixed_ari"].mean()),
                        "B_v1": float(gg["v1_ari"].mean()),
                        "C_v2": float(gg["v2_ari"].mean()),
                        "D_oracle": float(gg["row_oracle_ari"].mean()),
                        "delta_v2_minus_fixed": float(
                            (gg["v2_ari"] - gg["fixed_ari"]).mean()),
                        "delta_v2_minus_v1": float(
                            (gg["v2_ari"] - gg["v1_ari"]).mean())})
    write_csv_atomic(out / "view_analysis.csv", pd.DataFrame(vw_rows))
    print("\n" + pd.DataFrame(vw_rows).round(6).to_string(index=False), flush=True)

    for col, fn, mn in (("family_id", "family_analysis.csv", 1),
                        ("subfamily_id", "subfamily_analysis.csv", 30)):
        t = row.groupby(col).apply(lambda q: pd.Series({
            "n_rows": len(q),
            "delta_v2_minus_fixed": float((q["v2_ari"] - q["fixed_ari"]).mean()),
            "delta_v2_minus_v1": float((q["v2_ari"] - q["v1_ari"]).mean()),
            "C_v2": float(q["v2_ari"].mean())}),
            include_groups=False).reset_index()
        write_csv_atomic(out / fn, t[t["n_rows"] >= mn]
                         .sort_values("delta_v2_minus_fixed"))

    rt2 = read_json(out / "split1_v2_inference_runtime.json")
    rt1 = read_json(out / "split1_v1_inference_runtime.json")
    b5rt = read_json(repo / B5_REL / "runtime_summary.json")
    write_json_atomic(out / "runtime_summary.json", {
        "policy_predictor_v2": {
            "predict_ms_per_dataset_view": rt2["predict_ms_per_row"],
            "argmax_ms_per_dataset_view": rt2["argmax_ms_per_row"],
            "total_ms_per_dataset_view": rt2["predict_ms_per_row"]
            + rt2["argmax_ms_per_row"],
            "n_rows": rt2["n_rows"]},
        "policy_predictor_v1": {
            "total_ms_per_dataset_view": rt1["predict_ms_per_row"]
            + rt1["argmax_ms_per_row"]},
        "candidate_reranker_context_ms_per_run": b5rt["per_run_incremental_ms"]["mean"],
        "excludes": "outcome-table lookup, ARI evaluation, report generation",
        "autoclust_comparability": "NOT_DIRECTLY_COMPARABLE",
        "note": "the Policy Predictor selects ONE policy per dataset/view, so "
                "its cost is per dataset/view; the reranker figure is per "
                "dataset-view-POLICY run and the two are not additive without "
                "stating the deployment configuration"})

    write_json_atomic(out / "stage2c3_final_summary.json", {
        "kind": "FROZEN DEVELOPMENT-INFORMED ONLINE REPLAY",
        "four_level": FOUR.to_dict("records"),
        "v2_vs_fixed": vf, "v2_vs_v1": vv,
        "development_fixed_policy": fixed["policy_id"],
        "post_replay_tuning": False,
        "source_commit": commit, "runtime_sec": time.time() - t0})
    print("\n  done (%.0fs)" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
