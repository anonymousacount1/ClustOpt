"""Stage 4 -- assemble the scientific review package.

Reads authoritative artifacts only. Nothing is executed, recomputed or refitted:
no method, metric, ARI, utility, bank or model is touched. Every number here is
an aggregation of frozen outputs.

Authoritative sources
  methods : phase_c/method_ari.csv.gz (the offline-scored Phase-A layer)
  metrics : phase_d_canonical_utility/ (canonical candidate basis)
  corpus  : configs/dataset_manifest.json + publication manifest

phase_d/ and phase_d_corrected/ are read ONLY to document provenance, never as
metric-analysis sources.

Statistics follow configs/benchmark_protocol.json and nothing else: the dataset
is the inferential unit, the bootstrap is percentile and GROUP-AWARE (the three
sklearn difficulties from one generator are one dependence group and are
resampled together, never as three independent rows), the test is Wilcoxon
signed-rank where valid, and views are never pooled as independent observations.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

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

N_BOOT = 10000
BASE_SEED = 1234          # the project's frozen base seed
SOURCES = ("fcps", "sklearn_shapes", "real_pca")

FAMILY = {
    "IDB_ClustOpt_C0_MLP_TOP5_RAW_noR": ("ClustOpt", "MLP Top-5 RAW, no reranker"),
    "IDB_ClustOpt_C1_KNN_TOP10_RAW_noR": ("ClustOpt", "KNN Top-10 RAW, no reranker"),
    "IDB_ClustOpt_C2_PPv1_noR": ("ClustOpt", "Policy Predictor v1, no reranker"),
    "IDB_ClustOpt_C3_MLP_TOP5_RAW_R": ("ClustOpt", "MLP Top-5 RAW + reranker"),
    "IDB_ClustOpt_C4_KNN_TOP10_RAW_R": ("ClustOpt", "KNN Top-10 RAW + reranker"),
    "IDB_ClustOpt_C5_PPv2_R": ("ClustOpt", "Policy Predictor v2 + reranker"),
    "IDB_AutoClust_A0_original_cvis": ("AutoClust", "original CVIs"),
    "IDB_AutoClust_A1_original_plus_established": ("AutoClust", "original + Established"),
    "IDB_AutoClust_A2_original_plus_new46": ("AutoClust", "original + New46"),
    "IDB_AutoClust_A3_extended_cvis": ("AutoClust", "extended CVIs"),
    "IDB_ML2DAC_M0_original_cvis": ("ML2DAC", "original CVIs"),
    "IDB_ML2DAC_M1_original_plus_established": ("ML2DAC", "original + Established"),
    "IDB_ML2DAC_M2_original_plus_new46": ("ML2DAC", "original + New46"),
    "IDB_ML2DAC_M3_extended_cvis": ("ML2DAC", "extended CVIs"),
}
SHORT = {k: k.split("_")[2] for k in FAMILY}


# ----------------------------------------------------------------- statistics
def groups_for(datasets: List[str], gmap: Dict[str, str]) -> np.ndarray:
    return np.array([gmap[d] for d in datasets])


def group_bootstrap_ci(values: np.ndarray, groups: np.ndarray, *,
                       n_boot: int = N_BOOT, seed: int = BASE_SEED,
                       stat=np.mean) -> Dict[str, Optional[float]]:
    """Percentile bootstrap that resamples DEPENDENCE GROUPS, not rows."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    values, groups = values[ok], np.asarray(groups)[ok]
    if values.size == 0:
        return {"lo": None, "hi": None, "n": 0, "n_groups": 0}
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_group[g] for g in pick])
        out[b] = stat(values[sel])
    return {"lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)),
            "n": int(values.size), "n_groups": int(uniq.size)}


def wilcoxon_paired(delta: np.ndarray) -> Dict[str, Any]:
    from scipy.stats import wilcoxon
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    res: Dict[str, Any] = {"n_pairs": int(d.size), "n_nonzero": int(nz.size),
                           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                           "ties": int((d == 0).sum())}
    if nz.size < 1:
        res.update({"statistic": None, "p_value": None,
                    "rank_biserial": None, "valid": False,
                    "note": "no non-zero differences; Wilcoxon not valid"})
        return res
    try:
        st = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
        ranks = pd.Series(np.abs(nz)).rank().to_numpy()
        wp = float(ranks[nz > 0].sum())
        wm = float(ranks[nz < 0].sum())
        rb = (wp - wm) / (wp + wm) if (wp + wm) > 0 else None
        res.update({"statistic": float(st.statistic), "p_value": float(st.pvalue),
                    "rank_biserial": rb, "valid": True})
    except Exception as exc:  # noqa: BLE001
        res.update({"statistic": None, "p_value": None, "rank_biserial": None,
                    "valid": False, "note": str(exc)[:120]})
    return res


def source_balanced(vals: pd.Series, srcs: pd.Series) -> Optional[float]:
    df = pd.DataFrame({"v": vals, "s": srcs}).dropna()
    if df.empty:
        return None
    return float(df.groupby("s")["v"].mean().mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    run_dir = RESULTS / args.run_id
    out = run_dir / "scientific_review"
    fig = out / "figure_data"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    man = json.loads((CFG / "dataset_manifest.json").read_text(encoding="utf-8"))
    entries = {e["dataset_id"]: e for e in man["datasets"] if e.get("accepted", True)}
    # dependence groups: one sklearn generator = one group; others stand alone
    gmap = {d: (("skgen_%s" % e["generator"]) if e.get("generator") else d)
            for d, e in entries.items()}
    basis_amend = json.loads((CFG / "stage4c_utility_basis_amendment.json")
                             .read_text(encoding="utf-8"))
    disp_amend = json.loads((CFG / "stage4c_metric_dispatch_amendment.json")
                            .read_text(encoding="utf-8"))
    registry = json.loads((CFG / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    reg_hash = H.canonical_text_hash((CFG / "metric_analysis_registry.json")
                                     .read_bytes())
    corpus_hash = (RESULTS / "publication" / "corpus" / "manifest"
                   / "manifest_hash.txt").read_text(encoding="utf-8").strip()
    prim = {e["canonical_metric_id"]: e for e in registry["entries"]}

    PROV = {
        "run_id": args.run_id,
        "method_source": "phase_c/method_ari.csv.gz (Phase-A layer, unchanged)",
        "metric_source": "phase_d_canonical_utility/",
        "candidate_basis_version": basis_amend["candidate_basis_version"],
        "utility_basis_amendment_id": basis_amend["amendment_id"],
        "dispatch_amendment_id": disp_amend["amendment_id"],
        "metric_registry_hash": reg_hash,
        "corpus_publication_manifest_hash": corpus_hash,
        "statistics": {
            "inferential_unit": "dataset",
            "bootstrap": "percentile, group-aware, %d resamples" % N_BOOT,
            "dependence_groups": ("one sklearn generator (easy/medium/hard) is one "
                                  "group; FCPS and real-PCA datasets stand alone"),
            "n_dependence_groups": len(set(gmap.values())),
            "seed": BASE_SEED,
            "test": "Wilcoxon signed-rank where valid",
            "effect_size": "matched-pairs rank-biserial correlation",
            "no_view_pseudo_replication": True,
        },
    }
    (out / "provenance.json").write_text(json.dumps(PROV, indent=2),
                                         encoding="utf-8")

    # ================================================== 1. method master table
    m = pd.read_csv(run_dir / "phase_c" / "method_ari.csv.gz")
    m["source"] = m["dataset_id"].map(lambda d: entries[d]["source"])
    succ = m[m["status"] == "success"].dropna(subset=["ari"])
    best = (succ.sort_values("ari", ascending=False)
                .groupby(["dataset_id", "method_id"], as_index=False).first()
                [["dataset_id", "source", "method_id", "view_id", "ari"]]
                .rename(columns={"view_id": "best_view", "ari": "bestview_ari"}))
    xy = (succ[succ["view_id"] == "xy_2d"]
          [["dataset_id", "source", "method_id", "ari"]]
          .rename(columns={"ari": "xy_2d_ari"}))
    best.to_csv(out / "method_bestview_per_dataset.csv", index=False)

    rows = []
    for mid in sorted(FAMILY):
        b = best[best["method_id"] == mid]
        x = xy[xy["method_id"] == mid]
        bci = group_bootstrap_ci(b["bestview_ari"].to_numpy(),
                                 groups_for(list(b["dataset_id"]), gmap))
        xci = group_bootstrap_ci(x["xy_2d_ari"].to_numpy(),
                                 groups_for(list(x["dataset_id"]), gmap))
        rt = m[(m["method_id"] == mid)]["method_runtime_equivalent"].dropna()
        r = {"method_id": mid, "short": SHORT[mid],
             "family": FAMILY[mid][0], "arm_description": FAMILY[mid][1],
             "bestview_ari_mean": b["bestview_ari"].mean(),
             "bestview_ari_median": b["bestview_ari"].median(),
             "bestview_ari_std": b["bestview_ari"].std(ddof=1),
             "bestview_ci_lo": bci["lo"], "bestview_ci_hi": bci["hi"],
             "xy_2d_ari_mean": x["xy_2d_ari"].mean(),
             "xy_2d_ari_median": x["xy_2d_ari"].median(),
             "xy_2d_ari_std": x["xy_2d_ari"].std(ddof=1),
             "xy_2d_ci_lo": xci["lo"], "xy_2d_ci_hi": xci["hi"],
             "bestview_source_balanced": source_balanced(b["bestview_ari"],
                                                         b["source"]),
             "runtime_mean": rt.mean(), "runtime_median": rt.median(),
             "runtime_p90": rt.quantile(0.90),
             "datasets_successful": int(b["dataset_id"].nunique()),
             "datasets_total": 50}
        for s in SOURCES:
            r["bestview_mean_%s" % s] = b[b["source"] == s]["bestview_ari"].mean()
        rows.append(r)
    master = pd.DataFrame(rows).sort_values("bestview_ari_mean", ascending=False)
    master.to_csv(out / "method_master_table.csv", index=False)

    # ============================================== 3. best baseline arms
    def pick_best(prefix):
        sub = master[master["method_id"].str.contains(prefix)]
        row = sub.loc[sub["bestview_ari_mean"].idxmax()]
        return row["method_id"], float(row["bestview_ari_mean"]), \
            float(row["xy_2d_ari_mean"])

    best_ac, best_ac_bv, best_ac_xy = pick_best("AutoClust")
    best_ml, best_ml_bv, best_ml_xy = pick_best("ML2DAC")
    best_co, best_co_bv, best_co_xy = pick_best("ClustOpt")
    baselines = {
        "all_autoclust_arms": master[master["family"] == "AutoClust"][
            ["method_id", "bestview_ari_mean", "xy_2d_ari_mean",
             "bestview_source_balanced"]].to_dict("records"),
        "all_ml2dac_arms": master[master["family"] == "ML2DAC"][
            ["method_id", "bestview_ari_mean", "xy_2d_ari_mean",
             "bestview_source_balanced"]].to_dict("records"),
        "all_clustopt_arms": master[master["family"] == "ClustOpt"][
            ["method_id", "bestview_ari_mean", "xy_2d_ari_mean",
             "bestview_source_balanced"]].to_dict("records"),
        "best_autoclust": {"method_id": best_ac, "bestview_ari_mean": best_ac_bv,
                           "xy_2d_ari_mean": best_ac_xy},
        "best_ml2dac": {"method_id": best_ml, "bestview_ari_mean": best_ml_bv,
                        "xy_2d_ari_mean": best_ml_xy},
        "best_clustopt": {"method_id": best_co, "bestview_ari_mean": best_co_bv,
                          "xy_2d_ari_mean": best_co_xy},
        "selection_rule": ("highest BestView dataset-macro mean over the frozen "
                           "primary summary; all arms listed before selection"),
    }
    (out / "baseline_arm_selection.json").write_text(
        json.dumps(baselines, indent=2, default=str), encoding="utf-8")

    # ==================================================== 2/5. paired contrasts
    piv = best.pivot(index="dataset_id", columns="method_id", values="bestview_ari")
    piv_xy = xy.pivot(index="dataset_id", columns="method_id", values="xy_2d_ari")
    src = pd.Series({d: entries[d]["source"] for d in piv.index})

    def contrast(a, b, label, frame=piv):
        j = frame[[a, b]].dropna()
        d = (j[a] - j[b]).to_numpy()
        ds = list(j.index)
        ci = group_bootstrap_ci(d, groups_for(ds, gmap))
        w = wilcoxon_paired(d)
        rec = {"contrast": label, "method_a": a, "method_b": b,
               "n_paired_datasets": len(ds),
               "mean_delta": float(np.mean(d)), "median_delta": float(np.median(d)),
               "ci_lo": ci["lo"], "ci_hi": ci["hi"],
               "n_dependence_groups": ci["n_groups"],
               "wins": w["wins"], "ties": w["ties"], "losses": w["losses"],
               "wilcoxon_statistic": w["statistic"], "p_value": w["p_value"],
               "rank_biserial": w["rank_biserial"],
               "wilcoxon_valid": w["valid"]}
        ss = pd.Series(d, index=ds)
        for s in SOURCES:
            ids = [x for x in ds if entries[x]["source"] == s]
            rec["delta_%s" % s] = float(ss[ids].mean()) if ids else None
        rec["delta_source_balanced"] = source_balanced(
            ss, pd.Series({x: entries[x]["source"] for x in ds}))
        return rec

    C = {k: "IDB_ClustOpt_%s" % v for k, v in {
        "C0": "C0_MLP_TOP5_RAW_noR", "C1": "C1_KNN_TOP10_RAW_noR",
        "C2": "C2_PPv1_noR", "C3": "C3_MLP_TOP5_RAW_R",
        "C4": "C4_KNN_TOP10_RAW_R", "C5": "C5_PPv2_R"}.items()}

    contrasts = [
        contrast(C["C5"], C["C4"], "C5 PPv2_R vs C4 KNN_R"),
        contrast(C["C5"], C["C3"], "C5 PPv2_R vs C3 MLP_R"),
        contrast(C["C5"], C["C2"], "C5 PPv2_R vs C2 PPv1_noR"),
        contrast(C["C3"], C["C0"], "reranker: C3 vs C0 (MLP Top-5)"),
        contrast(C["C4"], C["C1"], "reranker: C4 vs C1 (KNN Top-10)"),
        contrast(C["C5"], C["C2"], "reranker+policy: C5 vs C2 (PP v2 vs v1)"),
        contrast(C["C5"], best_ac, "C5 vs best AutoClust (%s)" % SHORT[best_ac]),
        contrast(C["C5"], best_ml, "C5 vs best ML2DAC (%s)" % SHORT[best_ml]),
        contrast(C["C4"], best_ac, "C4 vs best AutoClust (%s)" % SHORT[best_ac]),
        contrast(C["C4"], best_ml, "C4 vs best ML2DAC (%s)" % SHORT[best_ml]),
    ]
    pd.DataFrame(contrasts).to_csv(out / "paired_comparisons.csv", index=False)

    # reranker deltas, per dataset and per view
    rer = []
    for lab, (a, b) in {"C3-C0": (C["C3"], C["C0"]), "C4-C1": (C["C4"], C["C1"]),
                        "C5-C2": (C["C5"], C["C2"])}.items():
        j = piv[[a, b]].dropna()
        for d_, v in (j[a] - j[b]).items():
            rer.append({"pair": lab, "dataset_id": d_,
                        "source": entries[d_]["source"], "level": "bestview",
                        "delta": float(v)})
        for view in ("x_only", "y_only", "xy_2d"):
            sv = succ[succ["view_id"] == view].pivot(
                index="dataset_id", columns="method_id", values="ari")
            if a in sv.columns and b in sv.columns:
                jj = sv[[a, b]].dropna()
                for d_, v in (jj[a] - jj[b]).items():
                    rer.append({"pair": lab, "dataset_id": d_,
                                "source": entries[d_]["source"], "level": view,
                                "delta": float(v)})
    rerank = pd.DataFrame(rer)
    rerank.to_csv(out / "reranker_deltas.csv", index=False)
    rsum = rerank.groupby(["pair", "level"], as_index=False).agg(
        n=("delta", "count"), mean_delta=("delta", "mean"),
        median_delta=("delta", "median"),
        wins=("delta", lambda s: int((s > 0).sum())),
        ties=("delta", lambda s: int((s == 0).sum())),
        losses=("delta", lambda s: int((s < 0).sum())),
        min_delta=("delta", "min"), max_delta=("delta", "max"))
    rsum.to_csv(out / "reranker_summary.csv", index=False)
    hurt = (rerank[(rerank["level"] == "bestview") & (rerank["delta"] < -0.05)]
            .sort_values("delta"))
    hurt.to_csv(out / "reranker_material_harm.csv", index=False)

    # ================================================ 4. per-dataset matrix
    mat = piv.copy()
    mat.columns = [SHORT[c] for c in mat.columns]
    mat = mat.reset_index()
    mat["source"] = mat["dataset_id"].map(lambda d: entries[d]["source"])
    mat["structural_tags"] = mat["dataset_id"].map(
        lambda d: "|".join(entries[d].get("structural_tags") or []))
    bv = best.pivot(index="dataset_id", columns="method_id", values="best_view")
    for c in bv.columns:
        mat["best_view_%s" % SHORT[c]] = mat["dataset_id"].map(bv[c])
    mat["C5_minus_bestAutoClust"] = mat["C5"] - mat[SHORT[best_ac]]
    mat["C5_minus_bestML2DAC"] = mat["C5"] - mat[SHORT[best_ml]]
    mat.to_csv(out / "per_dataset_method_matrix.csv", index=False)
    mat.nlargest(10, "C5_minus_bestAutoClust")[
        ["dataset_id", "source", "C5", SHORT[best_ac], SHORT[best_ml],
         "C5_minus_bestAutoClust", "C5_minus_bestML2DAC"]].to_csv(
        out / "top10_c5_advantages.csv", index=False)
    mat.nsmallest(10, "C5_minus_bestAutoClust")[
        ["dataset_id", "source", "C5", SHORT[best_ac], SHORT[best_ml],
         "C5_minus_bestAutoClust", "C5_minus_bestML2DAC"]].to_csv(
        out / "top10_c5_disadvantages.csv", index=False)

    # ================================================ 6. policy-selection view
    pol = []
    for lab, noR, withR in (("MLP_TOP5_RAW", C["C0"], C["C3"]),
                            ("KNN_TOP10_RAW", C["C1"], C["C4"]),
                            ("PolicyPredictor", C["C2"], C["C5"])):
        pol.append({"policy_family": lab,
                    "arm_without_reranker": SHORT[noR],
                    "bestview_mean_without_reranker": float(piv[noR].mean()),
                    "arm_with_reranker": SHORT[withR],
                    "bestview_mean_with_reranker": float(piv[withR].mean()),
                    "reranker_delta": float((piv[withR] - piv[noR]).mean()),
                    "note": ("C2 is PPv1 and C5 is PPv2, so the PolicyPredictor "
                             "row mixes a policy change with the reranker and is "
                             "not a clean reranker contrast")})
    pd.DataFrame(pol).to_csv(out / "policy_selection_summary.csv", index=False)

    # ==================================================== 7/8. metric tables
    P = run_dir / "phase_d_canonical_utility"
    magg = pd.read_csv(P / "metric_aggregate_summary.csv")
    ucell = pd.read_csv(P / "metric_utility_per_dataset_view.csv.gz")
    uv = ucell[ucell["status"] == "valid"].dropna(subset=["utility"])
    top5 = pd.read_csv(P / "metric_top5_frequency.csv") \
        if (P / "metric_top5_frequency.csv").is_file() else pd.DataFrame()
    t10 = []
    for (d, v), g in uv.groupby(["dataset_id", "view_id"]):
        for mid in g.nlargest(10, "utility")["metric_id"]:
            t10.append(mid)
    t10c = pd.Series(t10).value_counts().rename_axis("metric_id") \
        .reset_index(name="top10_count")
    mm = magg.merge(top5, on="metric_id", how="left") if len(top5) else magg.copy()
    mm = mm.merge(t10c, on="metric_id", how="left")
    mm["top5_count"] = mm.get("top5_count", pd.Series(dtype=float)).fillna(0)
    mm["top10_count"] = mm["top10_count"].fillna(0)
    mm["orientation"] = mm["metric_id"].map(lambda x: prim[x]["orientation"])
    mm["metric_space"] = mm["metric_id"].map(lambda x: prim[x]["metric_space"])
    mm["stage3_internal_comparable_field"] = "NONE (see review notes)"
    for s in SOURCES:
        sm = uv[uv["source"] == s].groupby("metric_id")["utility"].mean()
        mm["utility_%s" % s] = mm["metric_id"].map(sm)
    mm.sort_values("available_case_utility_macro", ascending=False).to_csv(
        out / "metric_master_table.csv", index=False)

    for tag, thr in (("all", 0.0), ("coverage_100", 1.0),
                     ("coverage_ge_095", 0.95), ("coverage_ge_090", 0.90)):
        sl = mm[mm["coverage"] >= thr].copy()
        sl["slice_rank"] = sl["available_case_utility_macro"].rank(
            ascending=False, method="min")
        sl.sort_values("slice_rank")[
            ["metric_id", "metric_group", "available_case_utility_macro",
             "available_case_utility_source_balanced", "coverage",
             "units_with_valid_utility", "slice_rank"]].to_csv(
            out / ("metric_slice_%s.csv" % tag), index=False)

    # ------------------------------------------------ 10. New46 specialists
    spec = []
    for (d, v), g in uv.groupby(["dataset_id", "view_id"]):
        gg = g.sort_values("utility", ascending=False).reset_index(drop=True)
        for pos, r in gg.iterrows():
            spec.append({"dataset_id": d, "view_id": v, "source": r["source"],
                         "metric_id": r["metric_id"], "rank": pos + 1})
    sp = pd.DataFrame(spec)
    rk = sp.groupby("metric_id", as_index=False).agg(
        rank1=("rank", lambda s: int((s == 1).sum())),
        top3=("rank", lambda s: int((s <= 3).sum())),
        top5=("rank", lambda s: int((s <= 5).sum())),
        top10=("rank", lambda s: int((s <= 10).sum())))
    n46 = mm[mm["metric_group"] == "New46"].merge(rk, on="metric_id", how="left")
    for s in SOURCES:
        sm = uv[uv["source"] == s].groupby("metric_id")["utility"].mean()
        n46["utility_%s" % s] = n46["metric_id"].map(sm)
    n46.sort_values("available_case_utility_macro", ascending=False).to_csv(
        out / "new46_specialist_table.csv", index=False)

    best_orig = mm[mm["metric_group"] == "Original"]["available_case_utility_macro"].max()
    best_est = mm[mm["metric_group"] == "Established"]["available_case_utility_macro"].max()
    n46_beats = {
        "best_original_utility": float(best_orig),
        "best_established_utility": float(best_est),
        "new46_beating_best_original_overall": int(
            (n46["available_case_utility_macro"] > best_orig).sum()),
        "new46_beating_best_established_overall": int(
            (n46["available_case_utility_macro"] > best_est).sum()),
    }
    for s in SOURCES:
        bo = uv[(uv["source"] == s) & (uv["metric_group"] == "Original")] \
            .groupby("metric_id")["utility"].mean().max()
        be = uv[(uv["source"] == s) & (uv["metric_group"] == "Established")] \
            .groupby("metric_id")["utility"].mean().max()
        n46_beats["new46_beating_best_original_%s" % s] = int(
            (n46["utility_%s" % s] > bo).sum())
        n46_beats["new46_beating_best_established_%s" % s] = int(
            (n46["utility_%s" % s] > be).sum())
    (out / "new46_vs_reference_groups.json").write_text(
        json.dumps(n46_beats, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------ 11. modern comparators
    store_rows = []
    import glob as _g
    inval = {}
    for f in _g.glob(str(run_dir / "evaluation" / "metrics_v2__*.json")):
        c = json.loads(Path(f).read_text(encoding="utf-8"))["content"]
        for r in c["rows"]:
            if r["metric_id"] in ("cvdd", "cvnn", "dcsi", "cdbw") \
                    and r["status"] != "valid":
                k = (r["metric_id"], str(r["failure_kind"]))
                inval[k] = inval.get(k, 0) + 1
    modern = mm[mm["metric_group"] == "Modern"].copy()
    modern_out = {
        "provenance": PROV,
        "metrics": modern[["metric_id", "available_case_utility_macro",
                           "available_case_utility_source_balanced", "coverage",
                           "units_with_valid_utility", "units_total",
                           "available_case_macro_rank", "full_coverage"]
                          + ["utility_%s" % s for s in SOURCES]].to_dict("records"),
        "invalid_reason_distribution": {
            "%s :: %s" % k: v for k, v in sorted(inval.items(),
                                                 key=lambda x: -x[1])},
        "cvdd_statement": (
            "CVDD is valid on 102 of 150 dataset x view units (coverage 0.68). Its "
            "rank is an AVAILABLE-CASE rank over the units where it is defined and "
            "does not establish that it is globally best; the full-coverage-only "
            "slice excludes it by construction."),
    }
    (out / "modern_comparator_review.json").write_text(
        json.dumps(modern_out, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------- 9. generalist core
    core = mm[mm["metric_id"].isin(
        ("noise_aware_silhouette", "silhouette", "gridness_fft_acf"))].copy()
    fullc = mm[mm["coverage"] >= 1.0].copy()
    fullc["fc_rank"] = fullc["available_case_utility_macro"].rank(
        ascending=False, method="min")
    core = core.merge(fullc[["metric_id", "fc_rank"]], on="metric_id", how="left")
    core.to_csv(out / "generalist_core_review.csv", index=False)

    # ------------------------------------------------------ 14. source domain
    srows = []
    for s in SOURCES:
        ds = [d for d in entries if entries[d]["source"] == s]
        for mid in sorted(FAMILY):
            b = best[(best["method_id"] == mid) & (best["source"] == s)]
            srows.append({"source": s, "n_datasets": len(ds), "method_id": mid,
                          "short": SHORT[mid], "family": FAMILY[mid][0],
                          "bestview_mean": b["bestview_ari"].mean(),
                          "bestview_median": b["bestview_ari"].median()})
    ssum = pd.DataFrame(srows)
    ssum.to_csv(out / "source_method_summary.csv", index=False)
    gsum = uv.groupby(["source", "metric_group"], as_index=False).agg(
        utility_mean=("utility", "mean"), n=("utility", "count"))
    gsum.to_csv(out / "source_metric_group_summary.csv", index=False)

    # ------------------------------------------------------ 13. joint pipeline
    jp = pd.read_csv(P / "joint_pipeline_per_unit.csv.gz")
    jsum = pd.read_csv(P / "joint_pipeline_summary.csv")
    jsum.to_csv(out / "joint_pipeline_summary.csv", index=False)
    c5 = jp[jp["method_id"] == C["C5"]].copy()
    if len(c5):
        um, am = c5["best_available_utility"].median(), c5["ari"].median()
        def quad(r):
            hi_u = r["best_available_utility"] >= um
            hi_a = r["ari"] >= am
            return ("A_strong_metric_strong_ari" if hi_u and hi_a else
                    "B_strong_metric_weak_ari" if hi_u and not hi_a else
                    "D_weak_metric_strong_ari" if hi_a else
                    "C_weak_metric_weak_ari")
        c5["quadrant"] = c5.apply(quad, axis=1)
        c5.to_csv(out / "joint_pipeline_c5_units.csv", index=False)
        c5.groupby("quadrant", as_index=False).agg(
            units=("ari", "count"), mean_ari=("ari", "mean"),
            mean_best_available_utility=("best_available_utility", "mean")).to_csv(
            out / "joint_pipeline_quadrants.csv", index=False)

    # ---------------------------------------------------------- 15. runtime
    rt = master[["method_id", "short", "family", "bestview_ari_mean",
                 "bestview_source_balanced", "runtime_median", "runtime_p90"]].copy()
    fastest = rt["runtime_median"].min()
    rt["runtime_median_relative_to_fastest"] = rt["runtime_median"] / fastest
    rt.sort_values("bestview_ari_mean", ascending=False).to_csv(
        out / "runtime_tradeoff.csv", index=False)

    # ------------------------------------------------------ 16. figure data
    best.to_csv(fig / "fig_method_bestview_distribution.csv", index=False)
    sc = piv[[C["C5"], best_ac, best_ml]].copy()
    sc.columns = ["C5", "best_autoclust", "best_ml2dac"]
    sc["source"] = [entries[d]["source"] for d in sc.index]
    sc.reset_index().to_csv(fig / "fig_paired_c5_vs_baselines.csv", index=False)
    rerank.to_csv(fig / "fig_reranker_delta_distribution.csv", index=False)
    mm[["metric_id", "metric_group", "available_case_utility_macro",
        "coverage", "units_with_valid_utility"]].to_csv(
        fig / "fig_metric_utility_vs_coverage.csv", index=False)
    sp.to_csv(fig / "fig_metric_rank_per_unit_heatmap.csv.gz", index=False)
    ssum.to_csv(fig / "fig_source_method_comparison.csv", index=False)
    rt.to_csv(fig / "fig_runtime_vs_ari.csv", index=False)

    print("  review package -> %s" % out)
    print("  best ClustOpt %s | best AutoClust %s | best ML2DAC %s"
          % (SHORT[best_co], SHORT[best_ac], SHORT[best_ml]))
    print("  dependence groups: %d" % len(set(gmap.values())))
    print("  files: %d + %d figure-data"
          % (len(list(out.glob('*'))) - 1, len(list(fig.glob('*')))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
