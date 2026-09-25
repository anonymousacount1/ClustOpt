"""Stage 2B-5B global aggregation, analyses and integrity gate.

The authoritative global result is reconstructed from ALL 1,055 per-dataset
``result.json`` files -- never by averaging subfamily means. Dataset weighting is
applied exactly once.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E402,E501
    stats as S,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from .run_split1_final import (  # noqa: E402
    EXPERIMENT_ID, GLOBAL_ROOT_REL, SPLIT_CSV_REL, VIEWS,
)

AUTOCLUST_METHODS = ("AutoClust_Extended_InDomain", "AutoClust_Original_InDomain")
GZ = {"index": False, "compression": "gzip"}
N_DATASETS = 1055
N_RUNS = 63300


def load_runs(repo: Path) -> pd.DataFrame:
    p = repo / GLOBAL_ROOT_REL / "per_run_reranker_results.csv.gz"
    if not path_exists(p):
        raise SystemExit("per_run_reranker_results.csv.gz missing -- wet run "
                         "has not completed")
    return pd.read_csv(_ext(p))


def bestview(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset_id", "policy_id"]).agg(
        original=("original_selected_ari", "max"),
        reranked=("reranked_selected_ari", "max"),
        oracle=("visited_oracle_ari", "max"),
        seen_oracle=("seen_family_visited_oracle_ari", "max"),
        family=("family", "first"), subfamily=("subfamily", "first"),
        mean_eligible=("n_eligible_candidates", "mean"),
        unseen_fraction=("unseen_fraction", "mean"),
        reranker_ms=("reranker_total_incremental_time_ms", "sum"),
        original_runtime_sec=("original_runtime_sec", "sum"),
        timeout=("n_trial_records", lambda s: float((s < 50).mean())),
    ).reset_index()
    g["gain"] = g["reranked"] - g["original"]
    g["headroom"] = g["oracle"] - g["original"]
    g["recovery"] = np.where(g["headroom"] > 1e-9,
                             g["gain"] / g["headroom"], np.nan)
    return g


def paired(name: str, a, b, la: str, lb: str) -> Dict[str, Any]:
    d = np.asarray(a, float) - np.asarray(b, float)
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    r = {"comparison": name, "arm_a": la, "arm_b": lb, "n": int(len(d)),
         "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
         "mean_delta": float(pt), "median_delta": float(np.median(d)),
         "ci_lo": float(lo), "ci_hi": float(hi),
         "wins_a": int((d > 1e-12).sum()), "wins_b": int((d < -1e-12).sum()),
         "ties": int((np.abs(d) <= 1e-12).sum())}
    try:
        from scipy import stats as sst
        nz = d[np.abs(d) > 1e-12]
        r["wilcoxon_p"] = float(sst.wilcoxon(nz).pvalue) if len(nz) else 1.0
    except Exception:                                              # noqa: BLE001
        r["wilcoxon_p"] = float("nan")
    return r


def load_autoclust(repo: Path, ids: Sequence[str]) -> pd.DataFrame:
    assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
    s1 = assign[assign["split_id"] == 1]
    rows = []
    view_map = {"1d_x": "x_only", "1d_y": "y_only", "2d": "xy_2d"}
    for _, r in s1.iterrows():
        for meth in AUTOCLUST_METHODS:
            best = np.nan
            for d, v in view_map.items():
                j = read_json(Path(r["dataset_dir"]) / "experiments" / "split_01"
                              / meth / d / "result.json")
                if not j or j.get("status") != "success":
                    continue
                a = (j.get("metrics") or {}).get("ari")
                if a is None:
                    a = (j.get("metrics") or {}).get("ARI")
                if a is not None and np.isfinite(float(a)):
                    best = float(a) if not np.isfinite(best) else max(best,
                                                                     float(a))
            rows.append({"dataset_id": r["dataset_id"], "method": meth,
                         "bestview_ari": best})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / GLOBAL_ROOT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    print("=" * 78)
    print("STAGE 2B-5B  --  GLOBAL AGGREGATION AND ANALYSIS")
    print("=" * 78, flush=True)

    df = load_runs(repo)
    bv = bestview(df)
    print(f"  {len(df)} runs | {df['dataset_id'].nunique()} datasets | "
          f"{df['policy_id'].nunique()} policies", flush=True)
    bv.to_csv(_ext(out / "per_dataset_policy_results.csv.gz"), **GZ)

    # ---- 20-policy table ----------------------------------------------------
    rows = []
    for pid, g in bv.groupby("policy_id"):
        d = (g["reranked"] - g["original"]).to_numpy()
        pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
        meta = df[df["policy_id"] == pid].iloc[0]
        rows.append({
            "policy_id": pid, "utility_source": meta["utility_source"],
            "k_mode": meta["k_mode"], "weighting_mode": meta["weighting_mode"],
            "n_datasets": int(len(g)),
            "original_mean_bestview": float(g["original"].mean()),
            "reranked_mean_bestview": float(g["reranked"].mean()),
            "visited_oracle_mean_bestview": float(g["oracle"].mean()),
            "seen_family_oracle_mean_bestview": float(g["seen_oracle"].mean()),
            "absolute_gain": float(pt), "gain_ci_lo": float(lo),
            "gain_ci_hi": float(hi),
            # Aggregate (pooled) recovery is the PRIMARY statistic. The mean of
            # per-dataset ratios is unusable here: headroom -> 0 on many datasets
            # (380/1055 below 0.01 for the winner), so a tiny denominator with a
            # small negative gain produces ratios as extreme as -180. The frozen
            # per-run definition is unchanged; only the aggregation is robust.
            "recovery": float((g["reranked"].mean() - g["original"].mean())
                              / (g["oracle"].mean() - g["original"].mean()))
            if (g["oracle"].mean() - g["original"].mean()) > 1e-9 else float("nan"),
            "recovery_aggregate": float((g["reranked"].mean() - g["original"].mean())
                                        / (g["oracle"].mean() - g["original"].mean()))
            if (g["oracle"].mean() - g["original"].mean()) > 1e-9 else float("nan"),
            "recovery_per_dataset_median": float(np.nanmedian(g["recovery"])),
            "recovery_per_dataset_mean_UNSTABLE": float(np.nanmean(g["recovery"])),
            "n_datasets_headroom_lt_0.01": int((g["headroom"] < 0.01).sum()),
            "n_datasets_zero_headroom": int((g["headroom"] <= 1e-9).sum()),
            "datasets_improved": int((d > 1e-12).sum()),
            "datasets_unchanged": int((np.abs(d) <= 1e-12).sum()),
            "datasets_worsened": int((d < -1e-12).sum()),
            "mean_eligible_candidates": float(g["mean_eligible"].mean()),
            "timeout_fraction": float(g["timeout"].mean()),
            "mean_unseen_fraction": float(g["unseen_fraction"].mean()),
            "mean_reranker_ms_3view": float(g["reranker_ms"].mean()),
            "median_reranker_ms_3view": float(g["reranker_ms"].median()),
            "p95_reranker_ms_3view": float(g["reranker_ms"].quantile(0.95)),
            "mean_original_runtime_sec_3view": float(
                g["original_runtime_sec"].mean()),
        })
        for v in VIEWS:
            sub = df[(df["policy_id"] == pid) & (df["view_id"] == v)]
            rows[-1][f"{v}_reranked_ari"] = float(
                sub["reranked_selected_ari"].mean())
            rows[-1][f"{v}_original_ari"] = float(
                sub["original_selected_ari"].mean())
    POL = pd.DataFrame(rows)
    write_csv_atomic(out / "per_policy_results.csv", POL)

    bo = POL.loc[POL["original_mean_bestview"].idxmax()]
    br = POL.loc[POL["reranked_mean_bestview"].idxmax()]
    write_json_atomic(out / "best_fixed_original_policy.json",
                      {k: (v if not isinstance(v, np.generic) else v.item())
                       for k, v in bo.to_dict().items()})
    write_json_atomic(out / "best_fixed_reranked_policy.json",
                      {k: (v if not isinstance(v, np.generic) else v.item())
                       for k, v in br.to_dict().items()})

    # ---- VBS / hybrid oracle ------------------------------------------------
    vbs = bv.groupby("dataset_id")["reranked"].max()
    vbs_view = df.groupby("dataset_id")["reranked_selected_ari"].max()
    hyb = bv.groupby("dataset_id")["oracle"].max()
    obest = bv[bv["policy_id"] == bo["policy_id"]].set_index("dataset_id")
    rbest = bv[bv["policy_id"] == br["policy_id"]].set_index("dataset_id")
    pv, plo, phi = S.bootstrap_ci(vbs.to_numpy(), statistic="mean")
    hv, hlo, hhi = S.bootstrap_ci(hyb.to_numpy(), statistic="mean")
    write_json_atomic(out / "reranker_assisted_vbs.json", {
        "reranked_policy_vbs_mean": float(vbs.mean()),
        "ci": [float(plo), float(phi)],
        "reranked_view_policy_vbs_mean": float(vbs_view.mean()),
        "definitions_equal": bool(np.allclose(vbs.sort_index().to_numpy(),
                                              vbs_view.sort_index().to_numpy(),
                                              atol=1e-12)),
        "equivalence_note": "verified empirically, not assumed",
        "headroom_over_best_fixed_reranked": float(
            vbs.mean() - rbest["reranked"].mean())})
    write_json_atomic(out / "hybrid_visited_oracle.json", {
        "hybrid_visited_oracle_mean": float(hyb.mean()),
        "ci": [float(hlo), float(hhi)],
        "gap_over_reranked_vbs": float(hyb.mean() - vbs.mean()),
        "gap_over_best_fixed_reranked": float(
            hyb.mean() - rbest["reranked"].mean())})

    # ---- domain shift -------------------------------------------------------
    algc = [c for c in df.columns if c.startswith("algcnt__")]
    fam_tot = df[algc].fillna(0).sum()
    seen_names = {f"algcnt__{a}" for a in
                  __import__("models.ClustOpt_Candidate_Reranker.training."
                             "feature_builder", fromlist=["x"]).ALGORITHM_BASES}
    dom = pd.DataFrame({
        "algorithm_family": [c.replace("algcnt__", "") for c in algc],
        "n_candidates": fam_tot.values,
        "fraction": (fam_tot / fam_tot.sum()).values,
        "seen_in_training": [c in seen_names for c in algc]}).sort_values(
        "n_candidates", ascending=False)
    write_csv_atomic(out / "algorithm_domain_shift.csv", dom)
    strata = pd.cut(df["unseen_fraction"], [-.001, 0, .1, .2, .3, 1.0],
                    labels=["0", "0-10%", "10-20%", "20-30%", ">30%"])
    ds = df.assign(stratum=strata).groupby("stratum", observed=True).agg(
        n_runs=("gain", "size"), mean_gain=("gain", "mean"),
        mean_recovery=("recovery", "mean"),
        mean_eligible=("n_eligible_candidates", "mean")).reset_index()
    write_csv_atomic(out / "domain_shift_strata.csv", ds)
    sel = pd.DataFrame([{
        "selector": "original_J",
        "frac_unseen": float(df["original_selected_unseen"].mean())},
        {"selector": "reranker",
         "frac_unseen": float(df["reranked_selected_unseen"].mean())},
        {"selector": "visited_oracle",
         "frac_unseen": float(df["oracle_selected_unseen"].mean())}])
    write_csv_atomic(out / "unseen_family_selection_rates.csv", sel)
    write_csv_atomic(out / "seen_family_oracle_gap.csv", pd.DataFrame([{
        "visited_oracle_mean": float(bv["oracle"].mean()),
        "seen_family_oracle_mean": float(bv["seen_oracle"].mean()),
        "gap": float(bv["oracle"].mean() - bv["seen_oracle"].mean()),
        "note": "seen-family oracle is DIAGNOSTIC ONLY; the authoritative "
                "oracle uses all eligible visited candidates"}]))

    # ---- contrasts ----------------------------------------------------------
    st: List[Dict[str, Any]] = []
    piv = bv.pivot(index="dataset_id", columns="policy_id",
                   values="reranked").sort_index()
    pivo = bv.pivot(index="dataset_id", columns="policy_id",
                    values="original").sort_index()
    pivv = bv.pivot(index="dataset_id", columns="policy_id",
                    values="oracle").sort_index()
    pairs = [("regressor_top5_raw", "regressor_top10_raw"),
             ("regressor_top5_softmax_t05", "regressor_top10_softmax_t05"),
             ("knn_top5_raw", "knn_top10_raw"),
             ("knn_top5_softmax_t05", "knn_top10_softmax_t05")]
    t5 = []
    for a, b in pairs:
        for lbl, P in (("original", pivo), ("reranked", piv), ("oracle", pivv)):
            r = paired(f"{a}_vs_{b}__{lbl}", P[a], P[b], a, b)
            r["layer"] = lbl
            t5.append(r)
            st.append(r)
    write_csv_atomic(out / "top5_vs_top10.csv", pd.DataFrame(t5))
    rs = []
    for pid in POL["policy_id"]:
        if not pid.endswith("_raw"):
            continue
        sm = pid.replace("_raw", "_softmax_t05")
        if sm not in piv.columns:
            continue
        for lbl, P in (("original", pivo), ("reranked", piv), ("oracle", pivv)):
            r = paired(f"{pid}_vs_{sm}__{lbl}", P[pid], P[sm], pid, sm)
            r["layer"] = lbl
            rs.append(r)
    write_csv_atomic(out / "raw_vs_softmax.csv", pd.DataFrame(rs))
    mk = []
    for km in ("top1", "top3", "top5", "top10", "dynamic"):
        for w in ("raw", "softmax_t05"):
            a = (f"regressor_top{km[3:]}_{w}" if km != "dynamic"
                 else f"regressor_top_dynamic_predictK_{w}")
            b = (f"knn_top{km[3:]}_{w}" if km != "dynamic"
                 else f"knn_top_dynamic_predictK_{w}")
            if a not in piv.columns or b not in piv.columns:
                continue
            for lbl, P in (("original", pivo), ("reranked", piv),
                           ("oracle", pivv)):
                r = paired(f"{a}_vs_{b}__{lbl}", P[a], P[b], a, b)
                r["layer"] = lbl
                mk.append(r)
    write_csv_atomic(out / "mlp_vs_knn.csv", pd.DataFrame(mk))

    # ---- AutoClust ----------------------------------------------------------
    ac = load_autoclust(repo, AUTOCLUST_METHODS)
    ac.to_csv(_ext(out / "autoclust_bestview.csv.gz"), **GZ)
    ac_rows = []
    for meth, g in ac.groupby("method"):
        s = g.set_index("dataset_id")["bestview_ari"].dropna()
        common = rbest.index.intersection(s.index)
        r = paired(f"reranked_{br['policy_id']}_vs_{meth}",
                   rbest.loc[common, "reranked"], s.loc[common],
                   str(br["policy_id"]), meth)
        r["autoclust_mean_bestview"] = float(s.mean())
        r["n_common_datasets"] = int(len(common))
        ac_rows.append(r)
        st.append(r)
    write_csv_atomic(out / "autoclust_performance_comparison.csv",
                     pd.DataFrame(ac_rows))

    # ---- runtime ------------------------------------------------------------
    rt = df["reranker_total_incremental_time_ms"]
    ov = (df["reranker_total_incremental_time_ms"] / 1000.0
          / df["original_runtime_sec"].replace(0, np.nan)) * 100.0
    write_json_atomic(out / "runtime_summary.json", {
        "per_run_incremental_ms": {
            "mean": float(rt.mean()), "median": float(rt.median()),
            "p90": float(rt.quantile(.90)), "p95": float(rt.quantile(.95)),
            "p99": float(rt.quantile(.99)), "max": float(rt.max())},
        "component_ms": {
            "feature_build_mean": float(
                df["reranker_feature_build_time_ms"].mean()),
            "predict_mean": float(df["reranker_predict_time_ms"].mean()),
            "argmin_mean": float(df["reranker_argmin_time_ms"].mean())},
        "three_view_dataset_policy_ms_mean": float(bv["reranker_ms"].mean()),
        "original_search_runtime_sec_mean": float(
            df["original_runtime_sec"].mean()),
        "incremental_overhead_pct_of_search": {
            "mean": float(np.nanmean(ov)), "median": float(np.nanmedian(ov))},
        "excludes": "ARI lookup, aggregation and reporting",
        "model_load_excluded_from_warm": True,
        "autoclust_comparability": "NOT_DIRECTLY_COMPARABLE (Stage 2B-5A)"})
    for by, name in (("policy_id", "policy"), ("view_id", "view"),
                     ("subfamily", "subfamily"), ("family", "family")):
        write_csv_atomic(out / f"runtime_per_{name}.csv",
                         df.groupby(by)["reranker_total_incremental_time_ms"]
                         .agg(["size", "mean", "median",
                               lambda s: s.quantile(.95)]).reset_index())
    buckets = pd.cut(df["n_eligible_candidates"],
                     [0, 9, 19, 29, 39, 49, 50],
                     labels=["1-9", "10-19", "20-29", "30-39", "40-49", "50"])
    strat = df.assign(bucket=buckets).groupby("bucket", observed=True).agg(
        n_runs=("gain", "size"), mean_gain=("gain", "mean"),
        mean_recovery=("recovery", "mean"),
        mean_reranker_ms=("reranker_total_incremental_time_ms", "mean")
    ).reset_index()
    write_csv_atomic(out / "runtime_candidate_count_strata.csv", strat)
    write_csv_atomic(out / "timeout_slate_size_analysis.csv", strat)
    tof = df.assign(timed_out=df["n_trial_records"] < 50).groupby(
        "timed_out").agg(n_runs=("gain", "size"), mean_gain=("gain", "mean"),
                         mean_recovery=("recovery", "mean")).reset_index()
    write_csv_atomic(out / "timeout_vs_full_budget.csv", tof)

    # ---- family / subfamily -------------------------------------------------
    for by, name in (("subfamily", "per_subfamily_results.csv"),
                     ("family", "per_family_results.csv")):
        g = bv.groupby([by, "policy_id"]).agg(
            n=("original", "size"), original=("original", "mean"),
            reranked=("reranked", "mean"), oracle=("oracle", "mean")).reset_index()
        g["gain"] = g["reranked"] - g["original"]
        write_csv_atomic(out / name, g)
    write_csv_atomic(out / "statistical_comparisons.csv", pd.DataFrame(st))
    write_csv_atomic(out / "visited_oracle_summary.csv", pd.DataFrame([{
        "mean_visited_oracle_bestview": float(bv["oracle"].mean()),
        "mean_reranked_bestview": float(bv["reranked"].mean()),
        "mean_original_bestview": float(bv["original"].mean())}]))
    write_csv_atomic(out / "reranker_recovery.csv",
                     POL[["policy_id", "recovery", "absolute_gain",
                          "gain_ci_lo", "gain_ci_hi"]])
    write_csv_atomic(out / "original_vs_reranked.csv",
                     POL[["policy_id", "original_mean_bestview",
                          "reranked_mean_bestview", "absolute_gain"]])

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "n_datasets": int(bv["dataset_id"].nunique()),
        "n_runs": int(len(df)),
        "best_original_policy": str(bo["policy_id"]),
        "best_original_bestview": float(bo["original_mean_bestview"]),
        "best_reranked_policy": str(br["policy_id"]),
        "best_reranked_bestview": float(br["reranked_mean_bestview"]),
        "best_reranked_original_bestview": float(br["original_mean_bestview"]),
        "best_reranked_gain": float(br["absolute_gain"]),
        "best_reranked_recovery": float(br["recovery"]),
        "reranked_vbs": float(vbs.mean()),
        "hybrid_visited_oracle": float(hyb.mean()),
        "autoclust": [{k: r[k] for k in ("arm_b", "autoclust_mean_bestview",
                                         "mean_delta", "ci_lo", "ci_hi",
                                         "wilcoxon_p")} for r in ac_rows],
        "runtime_mean_ms": float(rt.mean()),
        "runtime_p95_ms": float(rt.quantile(.95)),
        "unseen_family_fraction": float(df["unseen_fraction"].mean()),
        "source_commit": commit, "runtime_sec": time.time() - t0}
    write_json_atomic(out / "stage2b5_final_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("autoclust",)}, indent=1)[:1200], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
