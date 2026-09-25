"""Stage-1E final aggregation and analysis (Split 1, online_fixed50).

Builds the Stage-1E result package from the extracted record population
(``online_2x3_record_results.csv``, produced by
``scripts/stage1/extract_stage1e_records.py`` from the persisted per-run
artifacts) -- never from console output, never from the live monitor.

Reuses:
  * Best-View          summary_writer._best_view_summary semantics
                       (groupby(method,dataset).ari.idxmax, first-wins ties)
  * paired statistics  paper_analysis/stats.py (bootstrap_ci, paired_test,
                       holm_correct, Hypothesis)

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/build_stage1e_analysis.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pathlib import Path  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    _ext, ensure_dir, path_exists, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.run_offline_2x3_all_subfamilies import (  # noqa: E402
    ARM_ORDER,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import stats as S  # noqa: E402

ARMS = [c for c, _m in ARM_ORDER]
METHOD_OF = dict(ARM_ORDER)

OUT = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50"
OFFLINE = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32"
DOCS = Path(REPO) / "docs/internal_reports/stage1"
RECORDS = OUT / "online_2x3_record_results.csv"

FULL60_MODEL = "20260615_231715"
HEAD14_MODEL = "20260823_195830"

# Six pre-specified confirmatory contrasts of the 2x3 factorial.
CONTRASTS = [
    ("HP-HU", "HP", "HU", "learned utility within Head-14"),
    ("HO-HU", "HO", "HU", "oracle utility within Head-14"),
    ("FP-FU", "FP", "FU", "learned utility within Full-60"),
    ("FU-HU", "FU", "HU", "New-46 under uniform weighting"),
    ("FP-HP", "FP", "HP", "New-46 under learned utility"),
    ("FO-HO", "FO", "HO", "New-46 under oracle utility"),
]


def best_view(records: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level Best-View rows, historical first-wins convention."""
    ok = records[records["status"] == "success"].copy()
    ok["ari"] = pd.to_numeric(ok["ari"], errors="coerce")
    ok = ok.dropna(subset=["ari"])
    idx = ok.groupby(["method_name", "dataset_id"])["ari"].idxmax()
    bv = ok.loc[idx].copy()
    return bv.rename(columns={"ari": "best_view_ari", "view_id": "best_view"})


def agg_block(g: pd.DataFrame, ari_col: str = "best_view_ari") -> dict:
    a = pd.to_numeric(g[ari_col], errors="coerce")
    ek = pd.to_numeric(g["exact_k"], errors="coerce")
    return {
        "n_datasets": int(g["dataset_id"].nunique()),
        "n_rows": int(len(g)),
        "mean_ari": float(a.mean()), "median_ari": float(a.median()),
        "std_ari": float(a.std()),
        "exact_k_rate": float(ek.mean()) if ek.notna().any() else np.nan,
    }


def main() -> int:
    print("=" * 78)
    print("STAGE-1E FINAL AGGREGATION + ANALYSIS (online_fixed50, Split 1)")
    print("=" * 78)
    ensure_dir(OUT); ensure_dir(DOCS)

    print("\n[1/9] loading extracted record population ...", flush=True)
    rec = pd.read_csv(_ext(RECORDS))
    for c in ("ari", "objective_J", "best_visited_ari", "requested_trials",
              "delivered_trials", "valid_trials", "failed_trials",
              "rank_of_best_ari_trial", "selected_k", "true_k", "runtime_sec",
              "n_selected_metrics", "objective_ari_spearman", "ari_at_rank1",
              "max_ari_in_top3", "max_ari_in_top5", "max_ari_in_top10",
              "best_in_top1", "best_in_top3", "best_in_top5", "best_in_top10"):
        if c in rec.columns:
            rec[c] = pd.to_numeric(rec[c], errors="coerce")
    rec["exact_k"] = (rec["selected_k"] == rec["true_k"]).astype(float)
    rec["regret"] = rec["best_visited_ari"] - rec["ari"]
    rec["selected_is_best_visited"] = (rec["regret"].abs() <= 1e-12).astype(float)
    print(f"  records: {len(rec):,}")

    # ------------------------------------------------------------ integrity
    print("\n[2/9] INTEGRITY GATE")
    checks, ok_all = [], True

    def chk(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    n_ds, n_fam, n_sub = (rec["dataset_id"].nunique(), rec["family"].nunique(),
                          rec["subfamily"].nunique())
    chk("1,055 datasets", n_ds == 1055, str(n_ds))
    chk("12 families", n_fam == 12, str(n_fam))
    chk("86 subfamilies", n_sub == 86, str(n_sub))
    chk("3,165 dataset/view keys",
        rec.groupby(["dataset_id", "view_id"]).ngroups == 3165,
        str(rec.groupby(["dataset_id", "view_id"]).ngroups))
    chk("18,990 arm-records", len(rec) == 18990, str(len(rec)))
    chk("no duplicate (dataset,view,arm)",
        not rec.duplicated(subset=["dataset_id", "view_id", "arm"]).any())
    chk("all runs success", (rec["status"] == "success").all(),
        f"{int((rec['status'] != 'success').sum())} non-success")
    chk("every run requested 50 trials",
        set(rec["requested_trials"].dropna().unique()) == {50},
        str(sorted(set(rec["requested_trials"].dropna().unique()))))
    chk("every run delivered 50 trials",
        set(rec["delivered_trials"].dropna().unique()) == {50},
        str(sorted(set(rec["delivered_trials"].dropna().unique()))))
    chk("zero failed trials sweep-wide", float(rec["failed_trials"].sum()) == 0.0,
        f"{rec['failed_trials'].sum():.0f}")
    seed_var = rec.groupby(["dataset_id", "view_id"])["search_seed"].nunique()
    chk("search seed identical across the six arms of a dataset/view",
        (seed_var == 1).all(), f"{int((seed_var != 1).sum())} violations")
    chk("regret never negative", (rec["regret"] >= -1e-12).all(),
        f"min={rec['regret'].min():.3e}")

    from models.metric_utility_mlp.head_groups import resolve_target_group
    h14 = set(resolve_target_group("head14"))
    head = rec[rec["metric_inventory"] == "head14"].dropna(subset=["selected_metrics"])
    bad_h = int(head["selected_metrics"].map(lambda s: not set(s.split("|")) <= h14).sum())
    chk("HU/HP/HO restricted to Head-14", bad_h == 0, f"{bad_h} violations")
    hp_m = rec[rec["arm"] == "HP"]["predictor"].dropna().unique()
    fp_m = rec[rec["arm"] == "FP"]["predictor"].dropna().unique()
    chk("HP bound to the Head-14 model", len(hp_m) == 1 and HEAD14_MODEL in hp_m[0],
        str(hp_m[:1]))
    chk("FP bound to the frozen Full-60 model", len(fp_m) == 1 and FULL60_MODEL in fp_m[0],
        str(fp_m[:1]))
    topk = set(rec[rec["utility_strategy"] != "uniform"]["resolver_top_k"].dropna().unique())
    chk("Top-5 selection for all non-uniform arms", topk == {5}, str(sorted(topk)))
    nsel = rec.groupby("utility_strategy")["n_selected_metrics"].agg(["min", "max"]).to_dict()
    chk("uniform arms use their full inventory",
        int(rec[rec["arm"] == "HU"]["n_selected_metrics"].max()) == 14
        and int(rec[rec["arm"] == "FU"]["n_selected_metrics"].max()) == 60,
        f"HU={rec[rec['arm']=='HU']['n_selected_metrics'].max():.0f} "
        f"FU={rec[rec['arm']=='FU']['n_selected_metrics'].max():.0f}")
    chk("no fallback resolver invocations",
        not rec["used_fallback"].astype(str).str.lower().eq("true").any(),
        f"{int(rec['used_fallback'].astype(str).str.lower().eq('true').sum())} runs")

    pd.DataFrame(checks).to_csv(_ext(OUT / "online_2x3_integrity_checks.csv"), index=False)
    if not ok_all:
        print("\n  INTEGRITY GATE FAILED -- no scientific conclusions produced.")
        return 1

    # --------------------------------------------------------- dataset level
    print("\n[3/9] dataset-level Best-View")
    bv = best_view(rec)
    bv.to_csv(_ext(OUT / "online_2x3_dataset_results.csv"), index=False)
    print(f"  Best-View rows: {len(bv):,} (expect {1055*6:,})")

    rows = []
    for code in ARMS:
        g = bv[bv["arm"] == code]
        d = agg_block(g); d["arm"] = code; d["method_id"] = METHOD_OF[code]
        d["metric_inventory"] = g["metric_inventory"].iloc[0]
        d["utility_strategy"] = g["utility_strategy"].iloc[0]
        d["deployable"] = bool(g["deployable"].iloc[0])
        pt, lo, hi = S.bootstrap_ci(pd.to_numeric(g["best_view_ari"]).to_numpy(float),
                                    statistic="mean")
        d["mean_ci_lo"], d["mean_ci_hi"] = lo, hi
        r = rec[rec["arm"] == code]
        d["mean_best_visited_ari"] = float(r["best_visited_ari"].mean())
        d["mean_regret"] = float(r["regret"].mean())
        d["selected_is_best_rate"] = float(r["selected_is_best_visited"].mean())
        d["mean_valid_trials"] = float(r["valid_trials"].mean())
        d["mean_runtime_sec"] = float(r["runtime_sec"].mean())
        d["record_exact_k_rate"] = float(r["exact_k"].mean())
        rows.append(d)
    overall = pd.DataFrame(rows)
    overall.to_csv(_ext(OUT / "online_2x3_overall_summary.csv"), index=False)

    # ------------------------------------------------------------ contrasts
    print("\n[4/9] paired contrasts (dataset-level, n=1,055)")
    dl = bv[["dataset_id", "method_name", "best_view_ari"]].copy()
    crows = []
    for cid, a, b, why in CONTRASTS:
        hyp = S.Hypothesis(hypothesis_id=cid, family="stage1e_primary",
                           method_a=METHOD_OF[a], method_b=METHOD_OF[b],
                           claim_level=S.CLAIM_CONFIRMATORY, is_primary=True,
                           rationale=why)
        res = S.paired_test(dl, hyp)
        va, vb, _ = S.paired_values(dl, METHOD_OF[a], METHOD_OF[b])
        delta = va - vb
        res.update({
            "contrast": cid, "rationale": why,
            "mean_ci_lo": res.get("mean_delta_ci_low"),
            "mean_ci_hi": res.get("mean_delta_ci_high"),
            "n": int(delta.size), "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "wins": int((delta > 1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()),
        })
        crows.append(res)
    contrasts = pd.DataFrame(crows)
    contrasts = S.holm_correct(contrasts, p_col="wilcoxon_p", family_col="hypothesis_family")

    # exploratory interaction, reported outside the Holm family
    def dmap(a, b):
        va, vb, ids = S.paired_values(dl, METHOD_OF[a], METHOD_OF[b])
        return dict(zip(ids, va - vb))
    fp_hp, fu_hu = dmap("FP", "HP"), dmap("FU", "HU")
    common = sorted(set(fp_hp) & set(fu_hu))
    inter = np.array([fp_hp[d] - fu_hu[d] for d in common], float)
    ipt, ilo, ihi = S.bootstrap_ci(inter, statistic="mean")
    iwp = np.nan
    if S.scipy_available():
        from scipy import stats as sst
        try:
            iwp = float(sst.wilcoxon(inter, alternative="two-sided").pvalue)
        except Exception:
            pass
    inter_row = pd.DataFrame([{
        "contrast": "(FP-HP)-(FU-HU)", "rationale": "utility x New-46 interaction",
        "hypothesis_family": "stage1e_exploratory", "n": int(inter.size),
        "mean_delta": float(inter.mean()), "median_delta": float(np.median(inter)),
        "mean_ci_lo": ilo, "mean_ci_hi": ihi, "wilcoxon_p": iwp,
        "wins": int((inter > 1e-12).sum()), "ties": int((np.abs(inter) <= 1e-12).sum()),
        "losses": int((inter < -1e-12).sum()),
    }])
    contrasts = pd.concat([contrasts, inter_row], ignore_index=True)
    contrasts.to_csv(_ext(OUT / "online_2x3_paired_contrasts.csv"), index=False)

    # -------------------------------------------------------- family / view
    print("\n[5/9] family, subfamily and per-view tables")
    fam_rows, sub_rows = [], []
    for col, sink in (("family", fam_rows), ("subfamily", sub_rows)):
        for name, gg in bv.groupby(col):
            for code in ARMS:
                g = gg[gg["arm"] == code]
                if g.empty:
                    continue
                d = agg_block(g); d[col] = name; d["arm"] = code
                rr = rec[(rec["arm"] == code) & (rec[col] == name)]
                d["mean_regret"] = float(rr["regret"].mean())
                d["mean_best_visited_ari"] = float(rr["best_visited_ari"].mean())
                if col == "subfamily":
                    d["family"] = g["family"].iloc[0]
                sink.append(d)
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(_ext(OUT / "online_2x3_family_summary.csv"), index=False)
    pd.DataFrame(sub_rows).to_csv(_ext(OUT / "online_2x3_subfamily_summary.csv"), index=False)

    pv = []
    for code in ARMS:
        for view in ("x_only", "y_only", "xy_2d"):
            g = rec[(rec["arm"] == code) & (rec["view_id"] == view)]
            d = agg_block(g, ari_col="ari"); d["arm"] = code; d["view_id"] = view
            d["mean_regret"] = float(g["regret"].mean())
            d["mean_best_visited_ari"] = float(g["best_visited_ari"].mean())
            d["selected_is_best_rate"] = float(g["selected_is_best_visited"].mean())
            pv.append(d)
    perview = pd.DataFrame(pv)
    perview.to_csv(_ext(OUT / "online_2x3_per_view_summary.csv"), index=False)

    bvsel = bv.groupby("arm")["best_view"].value_counts(normalize=True).unstack().fillna(0)
    bvsel.to_csv(_ext(OUT / "online_2x3_best_view_choice.csv"))

    # ----------------------------------------- search / selection decomposition
    print("\n[6/9] search-vs-selection decomposition")
    dec = []
    for code in ARMS:
        g = rec[rec["arm"] == code]
        pt, lo, hi = S.bootstrap_ci(g["best_visited_ari"].to_numpy(float), statistic="mean")
        rpt, rlo, rhi = S.bootstrap_ci(g["regret"].to_numpy(float), statistic="mean")
        dec.append({
            "arm": code, "n_runs": int(len(g)),
            "mean_selected_ari": float(g["ari"].mean()),
            "mean_best_visited_ari": float(g["best_visited_ari"].mean()),
            "best_visited_ci_lo": lo, "best_visited_ci_hi": hi,
            "mean_regret": float(g["regret"].mean()),
            "regret_ci_lo": rlo, "regret_ci_hi": rhi,
            "median_regret": float(g["regret"].median()),
            "selected_is_best_rate": float(g["selected_is_best_visited"].mean()),
            "mean_objective_ari_spearman": float(g["objective_ari_spearman"].mean()),
            "share_of_gap_from_selection": float(
                g["regret"].mean() / g["best_visited_ari"].mean()),
        })
    decomp = pd.DataFrame(dec)
    decomp.to_csv(_ext(OUT / "online_2x3_search_selection_decomposition.csv"), index=False)

    # pairwise best-visited contrasts: is the SEARCH different across arms?
    rec_view = rec[["dataset_id", "view_id", "arm", "best_visited_ari"]].copy()
    piv_bvis = rec_view.pivot_table(index=["dataset_id", "view_id"], columns="arm",
                                    values="best_visited_ari")
    srch = []
    for cid, a, b, _why in CONTRASTS:
        d = (piv_bvis[a] - piv_bvis[b]).dropna().to_numpy(float)
        pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
        p = np.nan
        if S.scipy_available():
            from scipy import stats as sst
            try:
                p = float(sst.wilcoxon(d, alternative="two-sided").pvalue)
            except Exception:
                pass
        srch.append({"contrast": cid, "n_runs": int(d.size),
                     "mean_delta_best_visited": float(d.mean()),
                     "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p})
    pd.DataFrame(srch).to_csv(
        _ext(OUT / "online_2x3_best_visited_contrasts.csv"), index=False)

    # ------------------------------------------------------- trial accounting
    print("\n[7/9] trial accounting")
    tr = []
    for code in ARMS:
        g = rec[rec["arm"] == code]
        tr.append({
            "arm": code, "n_runs": int(len(g)),
            "requested_trials_total": int(g["requested_trials"].sum()),
            "delivered_trials_total": int(g["delivered_trials"].sum()),
            "valid_trials_total": int(g["valid_trials"].sum()),
            "failed_trials_total": int(g["failed_trials"].sum()),
            "mean_valid_trials": float(g["valid_trials"].mean()),
            "min_valid_trials": int(g["valid_trials"].min()),
            "valid_trial_rate": float(g["valid_trials"].sum() / g["delivered_trials"].sum()),
            "mean_runtime_sec": float(g["runtime_sec"].mean()),
            "total_runtime_h": float(g["runtime_sec"].sum() / 3600),
        })
    trials = pd.DataFrame(tr)
    trials.to_csv(_ext(OUT / "online_2x3_trial_accounting.csv"), index=False)

    # ------------------------------------------------------ reranker evidence
    print("\n[8/9] reranker feasibility evidence")
    rr = []
    for code in ARMS:
        g = rec[rec["arm"] == code].dropna(subset=["rank_of_best_ari_trial"])
        d = {"arm": code, "n_runs": int(len(g)),
             "mean_rank_of_best_ari_trial": float(g["rank_of_best_ari_trial"].mean()),
             "median_rank_of_best_ari_trial": float(g["rank_of_best_ari_trial"].median())}
        for k in (1, 3, 5, 10):
            d[f"best_in_top{k}_rate"] = float(g[f"best_in_top{k}"].mean())
            d[f"mean_max_ari_in_top{k}"] = float(g[f"max_ari_in_top{k}"].mean())
            d[f"headroom_top{k}"] = float(g[f"max_ari_in_top{k}"].mean() - g["ari"].mean())
        d["mean_selected_ari"] = float(g["ari"].mean())
        d["mean_best_visited_ari"] = float(g["best_visited_ari"].mean())
        d["oracle_rerank_headroom"] = float(g["best_visited_ari"].mean() - g["ari"].mean())
        rr.append(d)
    rerank = pd.DataFrame(rr)
    rerank.to_csv(_ext(OUT / "online_2x3_reranker_evidence.csv"), index=False)

    # ---------------------------------------------------- online vs offline
    print("\n[9/9] online-vs-offline comparison")
    cmp_rows = []
    off_path = OFFLINE / "offline_2x3_dataset_results.csv"
    have_offline = path_exists(off_path)
    if have_offline:
        off = pd.read_csv(_ext(off_path))
        off["best_view_ari"] = pd.to_numeric(off["best_view_ari"], errors="coerce")
        on_map = bv.set_index(["arm", "dataset_id"])["best_view_ari"]
        off_map = off.set_index(["arm", "dataset_id"])["best_view_ari"]
        for code in ARMS:
            o = on_map.loc[code]; f = off_map.loc[code]
            common_ds = sorted(set(o.index) & set(f.index))
            d = (o.loc[common_ds].to_numpy(float) - f.loc[common_ds].to_numpy(float))
            pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
            p = np.nan
            if S.scipy_available():
                from scipy import stats as sst
                try:
                    p = float(sst.wilcoxon(d, alternative="two-sided").pvalue)
                except Exception:
                    pass
            cmp_rows.append({
                "arm": code, "n_datasets": len(common_ds),
                "offline_fixed32_mean": float(f.loc[common_ds].mean()),
                "online_fixed50_mean": float(o.loc[common_ds].mean()),
                "mean_delta_online_minus_offline": float(d.mean()),
                "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p,
                "wins": int((d > 1e-12).sum()), "losses": int((d < -1e-12).sum()),
            })
        pd.DataFrame(cmp_rows).to_csv(
            _ext(OUT / "online_vs_offline_comparison.csv"), index=False)
    else:
        print("  [WARN] Stage-1C dataset results not found; comparison skipped")

    # winners
    piv = bv.pivot_table(index="dataset_id", columns="arm", values="best_view_ari")

    def winners(cols):
        sub = piv[cols].dropna(); mx = sub.max(axis=1)
        return ({c: int(((sub[c] - mx).abs() <= 1e-12).sum()) for c in cols},
                int((sub.eq(mx, axis=0).sum(axis=1) > 1).sum()), len(sub))
    aw, at, an = winners(ARMS)
    dw, dt, dn = winners(["HU", "HP", "FU", "FP"])
    pd.DataFrame([{"scope": "all_arms", "n_datasets": an, "n_tied_wins": at, **aw},
                  {"scope": "deployable_only", "n_datasets": dn, "n_tied_wins": dt, **dw}]
                 ).to_csv(_ext(OUT / "online_2x3_winner_counts.csv"), index=False)

    # ------------------------------------------------------------- manifest
    mpath = OUT / "experiment_manifest.json"
    man = json.load(open(_ext(mpath), encoding="utf-8")) if path_exists(mpath) else {}
    man.update({
        "final_source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True).stdout.strip(),
        "population": {"datasets": int(n_ds), "families": int(n_fam),
                       "subfamilies": int(n_sub), "dataset_view_keys": 3165,
                       "arm_records": int(len(rec)),
                       "optuna_trials": int(rec["delivered_trials"].sum())},
        "integrity_gate": "PASS",
        "total_search_runtime_h": float(rec["runtime_sec"].sum() / 3600),
        "best_view_source": "summary_writer._best_view_summary semantics "
                            "(groupby(method,dataset).ari.idxmax, first-wins)",
        "statistics_source": "paper_analysis/stats.py",
        "records_built_by": "scripts/stage1/extract_stage1e_records.py",
        "analysis_built_by": "scripts/stage1/build_stage1e_analysis.py",
    })
    write_json_atomic(mpath, man)

    # ---------------------------------------------------------------- print
    def hdr(t):
        print("\n" + "=" * 78); print(t); print("=" * 78)

    hdr("ONLINE 2x3 (dataset-weighted mean over 1,055 Best-View ARIs)")
    sh = overall.set_index("arm")
    print(f"{'arm':4} {'inv':7} {'utility':10} {'dep':5} {'mean':>8} {'median':>8} "
          f"{'CI95':>19} {'bestVis':>8} {'regret':>8} {'selBest':>8} {'exactK':>7}")
    for c in ARMS:
        r = sh.loc[c]
        print(f"{c:4} {r.metric_inventory:7} {r.utility_strategy:10} "
              f"{str(bool(r.deployable)):5} {r.mean_ari:8.4f} {r.median_ari:8.4f} "
              f"[{r.mean_ci_lo:7.4f},{r.mean_ci_hi:7.4f}] {r.mean_best_visited_ari:8.4f} "
              f"{r.mean_regret:8.4f} {r.selected_is_best_rate:8.4f} {r.exact_k_rate:7.4f}")
    print("\nMACRO-FAMILY mean (unweighted mean of the 12 family means):")
    print("  " + "   ".join(
        f"{c}:{fam[fam.arm == c]['mean_ari'].mean():.4f}" for c in ARMS))

    hdr("PAIRED CONTRASTS (n=1,055 datasets; Holm within stage1e_primary)")
    print(f"{'contrast':18} {'mean_d':>8} {'med_d':>8} {'CI95':>19} "
          f"{'W/T/L':>17} {'p_raw':>10} {'p_holm':>10}")
    for _, r in contrasts.iterrows():
        ci = f"[{r.get('mean_ci_lo', np.nan):7.4f},{r.get('mean_ci_hi', np.nan):7.4f}]"
        ph = r.get("holm_p", np.nan)
        ph = ph if isinstance(ph, (int, float)) and not pd.isna(ph) else float("nan")
        print(f"{r['contrast']:18} {r['mean_delta']:8.4f} {r['median_delta']:8.4f} "
              f"{ci:>19} {int(r['wins']):5d}/{int(r['ties']):4d}/{int(r['losses']):5d} "
              f"{r.get('wilcoxon_p', float('nan')):10.2e} {ph:10.2e}")

    hdr("FAMILY TABLE (mean Best-View ARI)")
    p = fam.pivot_table(index="family", columns="arm", values="mean_ari")[ARMS]
    p.insert(0, "N", fam[fam.arm == "HU"].set_index("family")["n_datasets"])
    p["HP-HU"] = p["HP"] - p["HU"]; p["FU-HU"] = p["FU"] - p["HU"]
    p["FP-HP"] = p["FP"] - p["HP"]; p["FO-HO"] = p["FO"] - p["HO"]
    print(p.round(4).to_string())
    p.round(6).to_csv(_ext(OUT / "online_2x3_family_table.csv"))

    hdr("PER-VIEW TABLE (record-level mean ARI)")
    pvp = perview.pivot_table(index="view_id", columns="arm", values="mean_ari")[ARMS]
    pvr = perview.pivot_table(index="view_id", columns="arm", values="mean_regret")[ARMS]
    print("mean ARI"); print(pvp.round(4).to_string())
    print("\nmean regret"); print(pvr.round(4).to_string())

    hdr("SEARCH vs SELECTION")
    print(f"{'arm':4} {'selARI':>8} {'bestVis':>8} {'bestVisCI95':>19} {'regret':>8} "
          f"{'selBest':>8} {'rho(J,ARI)':>11} {'gapShare':>9}")
    for _, r in decomp.iterrows():
        print(f"{r['arm']:4} {r['mean_selected_ari']:8.4f} "
              f"{r['mean_best_visited_ari']:8.4f} "
              f"[{r['best_visited_ci_lo']:7.4f},{r['best_visited_ci_hi']:7.4f}] "
              f"{r['mean_regret']:8.4f} {r['selected_is_best_rate']:8.4f} "
              f"{r['mean_objective_ari_spearman']:11.4f} "
              f"{r['share_of_gap_from_selection']:9.4f}")

    hdr("TRIAL ACCOUNTING")
    print(trials.round(4).to_string(index=False))

    hdr("RERANKER EVIDENCE (objective-descending rank of the best-ARI trial)")
    print(f"{'arm':4} {'meanRank':>9} {'medRank':>8} {'top1':>7} {'top3':>7} "
          f"{'top5':>7} {'top10':>7} {'selARI':>8} {'top5ARI':>8} {'head5':>8} "
          f"{'headMax':>8}")
    for _, r in rerank.iterrows():
        print(f"{r['arm']:4} {r['mean_rank_of_best_ari_trial']:9.2f} "
              f"{r['median_rank_of_best_ari_trial']:8.1f} "
              f"{r['best_in_top1_rate']:7.4f} {r['best_in_top3_rate']:7.4f} "
              f"{r['best_in_top5_rate']:7.4f} {r['best_in_top10_rate']:7.4f} "
              f"{r['mean_selected_ari']:8.4f} {r['mean_max_ari_in_top5']:8.4f} "
              f"{r['headroom_top5']:8.4f} {r['oracle_rerank_headroom']:8.4f}")

    if have_offline:
        hdr("ONLINE (fixed-50 search) vs OFFLINE (fixed-32 candidates)")
        print(pd.DataFrame(cmp_rows).round(4).to_string(index=False))

    hdr("WINNER COUNTS")
    print(f"  all arms       (n={an}): " + "  ".join(f"{c}:{aw[c]}" for c in ARMS)
          + f"   tied:{at}")
    print(f"  deployable only(n={dn}): "
          + "  ".join(f"{c}:{dw[c]}" for c in ['HU', 'HP', 'FU', 'FP']) + f"   tied:{dt}")

    print(f"\n  package -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
