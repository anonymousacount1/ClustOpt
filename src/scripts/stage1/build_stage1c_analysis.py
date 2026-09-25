"""Stage-1C final aggregation and first analysis (Split 1, offline_fixed32).

Builds the global result package from the PERSISTED per-run
``offline_result.json`` population -- never from console output -- and runs the
first descriptive/inferential analysis.

Reuses:
  * Best-View            summary_writer._best_view_summary semantics (idxmax on
                         status=='success', first-wins ties)
  * paired statistics    paper_analysis/stats.py (bootstrap_ci, paired_test,
                         holm_correct, Hypothesis)
  * record loading       experiment_execution.run_offline_2x3_all_subfamilies
                         .load_offline_records

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/build_stage1c_analysis.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pathlib import Path  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (  # noqa: E402
    ExperimentExecutionConfig,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    ensure_dir, path_exists, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.run_offline_2x3_all_subfamilies import (  # noqa: E402
    ARM_ORDER, discover_subfamilies, load_offline_records,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import stats as S  # noqa: E402

ARMS = [c for c, _m in ARM_ORDER]
METHOD_OF = dict(ARM_ORDER)
ARM_OF = {m: c for c, m in ARM_ORDER}
HEAD14 = "results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout"

OUT = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32"
DOCS = Path(REPO) / "docs/internal_reports/stage1"

CONTRASTS = [
    ("HP-HU", "HP", "HU", "learned utility within Head-14"),
    ("HO-HU", "HO", "HU", "oracle ceiling within Head-14"),
    ("FU-HU", "FU", "HU", "New-46 under uniform weighting"),
    ("FP-HP", "FP", "HP", "New-46 under learned utility"),
    ("FO-HO", "FO", "HO", "New-46 under oracle utility"),
]


def load_population() -> pd.DataFrame:
    """Every offline record across all 86 subfamilies."""
    analyzed = Path(REPO) / "results_analysis/clustering_repository/analyzed_data"
    split_csv = analyzed / "experiment_splits/dataset_split_assignments.csv"
    frames = []
    subs = discover_subfamilies(analyzed)
    for i, sub in enumerate(subs, 1):
        cfg = ExperimentExecutionConfig(
            repo_root=Path(REPO), subfamily_dir=sub, split_assignments=split_csv,
            split_id=1, split_dir_suffix="offline_fixed32",
            method_set="stage1_2x3", max_workers=8)
        df = load_offline_records(cfg)
        if df is not None and not df.empty:
            frames.append(df)
        if i % 20 == 0:
            print(f"    loaded {i}/{len(subs)} subfamilies", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def best_view(records: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level Best-View rows using the historical first-wins convention."""
    ok = records[records["status"] == "success"].copy()
    ok["ari"] = pd.to_numeric(ok["ari"], errors="coerce")
    ok = ok.dropna(subset=["ari"])
    idx = ok.groupby(["method_name", "dataset_id"])["ari"].idxmax()
    bv = ok.loc[idx].copy()
    bv = bv.rename(columns={"ari": "best_view_ari", "view_id": "best_view"})
    return bv


def agg_block(g: pd.DataFrame) -> dict:
    a = pd.to_numeric(g["best_view_ari"], errors="coerce")
    r = pd.to_numeric(g["ari_regret"], errors="coerce")
    ek = pd.to_numeric(g["exact_K"], errors="coerce")
    ch = g["ceiling_hit"].map(lambda v: bool(v) if v is not None else np.nan)
    return {
        "n_datasets": int(g["dataset_id"].nunique()),
        "mean_best_view_ari": float(a.mean()), "median_best_view_ari": float(a.median()),
        "std_best_view_ari": float(a.std()),
        "mean_regret": float(r.mean()) if r.notna().any() else np.nan,
        "median_regret": float(r.median()) if r.notna().any() else np.nan,
        "ceiling_hit_rate": float(pd.to_numeric(ch, errors="coerce").mean()),
        "exact_k_rate": float(ek.mean()) if ek.notna().any() else np.nan,
    }


def main() -> int:
    print("=" * 78); print("STAGE-1C FINAL AGGREGATION + FIRST ANALYSIS"); print("=" * 78)
    ensure_dir(OUT); ensure_dir(DOCS)

    print("\n[1/7] loading persisted record population ...", flush=True)
    rec = load_population()
    print(f"  records: {len(rec):,}")

    # ------------------------------------------------------- integrity gate
    print("\n[2/7] INTEGRITY GATE")
    subs = discover_subfamilies(Path(REPO) / "results_analysis/clustering_repository/analyzed_data")
    checks, ok_all = [], True

    def chk(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok); checks.append({"check": name, "status": "PASS" if ok else "FAIL",
                                           "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    n_ds = rec["dataset_id"].nunique()
    n_fam = rec["family"].nunique()
    n_sub = rec["subfamily"].nunique()
    chk("1,055 datasets", n_ds == 1055, f"{n_ds}")
    chk("12 families", n_fam == 12, f"{n_fam}")
    chk("86 subfamilies", n_sub == 86, f"{n_sub}")
    chk("3,165 dataset/view keys",
        rec.groupby(["dataset_id", "view_id"]).ngroups == 3165,
        f"{rec.groupby(['dataset_id','view_id']).ngroups}")
    chk("18,990 arm-records", len(rec) == 18990, f"{len(rec)}")
    chk("no duplicate (dataset,view,arm)",
        not rec.duplicated(subset=["dataset_id", "view_id", "method_name"]).any())
    chk("all records success", (rec["status"] == "success").all(),
        f"{(rec['status']!='success').sum()} non-success")
    chk("32 candidates everywhere", set(rec["candidate_count"].unique()) == {32},
        f"{sorted(set(rec['candidate_count'].unique()))}")
    fp = rec.groupby(["dataset_id", "view_id"])["candidate_set_fingerprint"].nunique()
    chk("candidate fingerprint identical across arms", (fp == 1).all(),
        f"{int((fp!=1).sum())} violations")
    ce = rec.groupby(["dataset_id", "view_id"])["best_possible_ari_in_candidate_set"].nunique()
    chk("candidate ceiling identical across arms", (ce == 1).all(),
        f"{int((ce!=1).sum())} violations")
    head_arms = rec[rec["metric_inventory"] == "head14"]
    from models.metric_utility_mlp.head_groups import METRIC_HEAD_GROUPS, resolve_target_group
    h14 = set(resolve_target_group("head14"))
    bad_h = head_arms["selected_metrics"].map(lambda s: not set(s.split("|")) <= h14).sum()
    chk("HP/HO/HU restricted to Head-14", bad_h == 0, f"{bad_h} violations")
    hp = rec[rec["method_name"] == METHOD_OF["HP"]]["predictor"].unique()
    fp_ = rec[rec["method_name"] == METHOD_OF["FP"]]["predictor"].unique()
    chk("HP bound to Head-14 model", len(hp) == 1 and "head14" in hp[0], str(hp[:1]))
    chk("FP bound to frozen Full-60", len(fp_) == 1 and "20260615_231715" in fp_[0], str(fp_[:1]))
    chk("regret never negative",
        (pd.to_numeric(rec["ari_regret"], errors="coerce") >= -1e-12).all())
    chk("execution_mode offline_fixed32", (rec["execution_mode"] == "offline_fixed32").all())
    chk("selected J always == max J",
        np.allclose(pd.to_numeric(rec["objective_J"]), pd.to_numeric(rec["max_objective_J"]),
                    atol=1e-12))
    chk("smoke trees excluded (records come only from split_01_offline_fixed32)", True,
        "loader reads split_01_offline_fixed32 only")

    pd.DataFrame(checks).to_csv(OUT / "offline_2x3_integrity_checks.csv", index=False)
    if not ok_all:
        print("\n  INTEGRITY GATE FAILED -- no scientific conclusions produced.")
        return 1

    # ------------------------------------------------------------- outputs
    print("\n[3/7] writing record + dataset level results")
    rec.to_csv(OUT / "offline_2x3_record_results.csv", index=False)
    bv = best_view(rec)
    bv["arm"] = bv["method_name"].map(ARM_OF)
    bv.to_csv(OUT / "offline_2x3_dataset_results.csv", index=False)
    print(f"  records {len(rec):,} -> dataset Best-View rows {len(bv):,} "
          f"(expect {1055*6})")

    # ---------------------------------------------------------- summaries
    print("\n[4/7] overall / per-view / family / subfamily summaries")
    rows = []
    for code in ARMS:
        g = bv[bv["arm"] == code]
        d = agg_block(g); d["arm"] = code; d["method_id"] = METHOD_OF[code]
        d["metric_inventory"] = g["metric_inventory"].iloc[0]
        d["utility_strategy"] = g["utility_strategy"].iloc[0]
        d["deployable"] = bool(g["deployable"].iloc[0])
        # bootstrap_ci returns (point_estimate, ci_low, ci_high) -- NOT (lo, hi, pt).
        pt, lo, hi = S.bootstrap_ci(pd.to_numeric(g["best_view_ari"]).to_numpy(float),
                                    statistic="mean")
        d["mean_ci_lo"], d["mean_ci_hi"] = lo, hi
        rows.append(d)
    overall = pd.DataFrame(rows)
    overall.to_csv(OUT / "offline_2x3_overall_summary.csv", index=False)

    pv = []
    for code in ARMS:
        for view in ("x_only", "y_only", "xy_2d"):
            g = rec[(rec["method_name"] == METHOD_OF[code]) & (rec["view_id"] == view)].copy()
            g = g.rename(columns={"ari": "best_view_ari"})
            d = agg_block(g); d["arm"] = code; d["view_id"] = view
            d["tie_rate"] = float((pd.to_numeric(g["n_tied_at_max"]) > 1).mean())
            pv.append(d)
    pd.DataFrame(pv).to_csv(OUT / "offline_2x3_per_view_summary.csv", index=False)

    fam_rows, sub_rows = [], []
    for key, out_list, col in (("family", fam_rows, "family"), ("subfamily", sub_rows, "subfamily")):
        for name, gg in bv.groupby(col):
            for code in ARMS:
                g = gg[gg["arm"] == code]
                if g.empty:
                    continue
                d = agg_block(g); d[col] = name; d["arm"] = code
                if col == "subfamily":
                    d["family"] = g["family"].iloc[0]
                out_list.append(d)
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(OUT / "offline_2x3_family_summary.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(OUT / "offline_2x3_subfamily_summary.csv", index=False)

    # ---------------------------------------------------------- contrasts
    print("\n[5/7] paired contrasts (dataset-level, n=1,055)")
    dataset_level = bv[["dataset_id", "method_name", "best_view_ari"]].copy()
    crows = []
    for cid, a, b, why in CONTRASTS:
        hyp = S.Hypothesis(hypothesis_id=cid, family="stage1c_primary",
                           method_a=METHOD_OF[a], method_b=METHOD_OF[b],
                           claim_level=S.CLAIM_CONFIRMATORY, is_primary=True,
                           rationale=why)
        res = S.paired_test(dataset_level, hyp)
        va, vb, _ids = S.paired_values(dataset_level, METHOD_OF[a], METHOD_OF[b])
        delta = va - vb
        res.update({
            "contrast": cid, "rationale": why,
            "mean_ci_lo": res.get("mean_delta_ci_low"),
            "mean_ci_hi": res.get("mean_delta_ci_high"),
            "n": int(delta.size),
            "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
            "wins": int((delta > 1e-12).sum()), "ties": int((np.abs(delta) <= 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()),
        })
        crows.append(res)

    # interaction: per-dataset composite delta
    fp_hp = dict(zip(*[S.paired_values(dataset_level, METHOD_OF["FP"], METHOD_OF["HP"])[2],
                       (lambda a, b, i: a - b)(*S.paired_values(dataset_level, METHOD_OF["FP"], METHOD_OF["HP"]))]))
    fu_hu = dict(zip(*[S.paired_values(dataset_level, METHOD_OF["FU"], METHOD_OF["HU"])[2],
                       (lambda a, b, i: a - b)(*S.paired_values(dataset_level, METHOD_OF["FU"], METHOD_OF["HU"]))]))
    common = sorted(set(fp_hp) & set(fu_hu))
    inter = np.array([fp_hp[d] - fu_hu[d] for d in common], dtype=float)
    pt, lo, hi = S.bootstrap_ci(inter, statistic="mean")
    wil_p = np.nan
    if S.scipy_available():
        from scipy import stats as sst
        try:
            wil_p = float(sst.wilcoxon(inter, alternative="two-sided").pvalue)
        except Exception:
            wil_p = np.nan
    crows.append({
        "contrast": "(FP-HP)-(FU-HU)", "rationale": "utility x New-46 interaction",
        "n": int(inter.size), "mean_delta": float(inter.mean()),
        "median_delta": float(np.median(inter)),
        "wins": int((inter > 1e-12).sum()), "ties": int((np.abs(inter) <= 1e-12).sum()),
        "losses": int((inter < -1e-12).sum()),
        "mean_ci_lo": lo, "mean_ci_hi": hi, "wilcoxon_p": wil_p,
        "hypothesis_family": "stage1c_primary",
    })
    contrasts = pd.DataFrame(crows)
    if "wilcoxon_p" in contrasts.columns:
        contrasts = S.holm_correct(contrasts, p_col="wilcoxon_p",
                                   family_col="hypothesis_family")
    contrasts.to_csv(OUT / "offline_2x3_paired_contrasts.csv", index=False)

    # --------------------------------------------------------- ties/winners
    print("\n[6/7] tie + winner analysis")
    tie = []
    for code in ARMS:
        g = rec[rec["method_name"] == METHOD_OF[code]]
        n = pd.to_numeric(g["n_tied_at_max"], errors="coerce")
        tie.append({"arm": code, "n_records": len(g),
                    "tie_rate": float((n > 1).mean()), "mean_tie_size": float(n.mean()),
                    "max_tie_size": int(n.max()),
                    "records_tied": int((n > 1).sum())})
    pd.DataFrame(tie).to_csv(OUT / "offline_2x3_tie_analysis.csv", index=False)

    piv = bv.pivot_table(index="dataset_id", columns="arm", values="best_view_ari")
    def winners(cols):
        sub = piv[cols].dropna()
        mx = sub.max(axis=1)
        counts = {c: int(((sub[c] - mx).abs() <= 1e-12).sum()) for c in cols}
        n_tie = int((sub.eq(mx, axis=0).sum(axis=1) > 1).sum())
        return counts, n_tie, len(sub)
    all_w, all_tie, all_n = winners(ARMS)
    dep = ["HU", "HP", "FU", "FP"]
    dep_w, dep_tie, dep_n = winners(dep)
    pd.DataFrame([
        {"scope": "all_arms", "n_datasets": all_n, "n_tied_wins": all_tie, **all_w},
        {"scope": "deployable_only", "n_datasets": dep_n, "n_tied_wins": dep_tie, **dep_w},
    ]).to_csv(OUT / "offline_2x3_winner_counts.csv", index=False)

    # ------------------------------------------------------------ manifest
    print("\n[7/7] manifest")
    mpath = OUT / "experiment_manifest.json"
    man = json.load(open(mpath, encoding="utf-8")) if path_exists(mpath) else {}
    import subprocess
    man.update({
        "final_source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True).stdout.strip(),
        "population": {"datasets": int(n_ds), "families": int(n_fam),
                       "subfamilies": int(n_sub), "dataset_view_keys": 3165,
                       "arm_records": int(len(rec)),
                       "candidate_scores": int(len(rec) * 32)},
        "integrity_gate": "PASS",
        "best_view_source": "summary_writer._best_view_summary semantics "
                            "(groupby(method,dataset).ari.idxmax, first-wins)",
        "statistics_source": "paper_analysis/stats.py",
        "analysis_built_by": "scripts/stage1/build_stage1c_analysis.py",
    })
    write_json_atomic(mpath, man)

    # -------------------------------------------------------------- print
    print("\n" + "=" * 78)
    print("OVERALL 2x3 (dataset-weighted micro mean over 1,055 Best-View ARIs)")
    print("=" * 78)
    show = overall.set_index("arm")
    print(f"{'arm':4} {'N':>5} {'mean':>8} {'median':>8} {'SD':>7} "
          f"{'CI95':>18} {'regret':>8} {'ceilHit':>8} {'exactK':>7}")
    for c in ARMS:
        r = show.loc[c]
        print(f"{c:4} {int(r.n_datasets):5d} {r.mean_best_view_ari:8.4f} "
              f"{r.median_best_view_ari:8.4f} {r.std_best_view_ari:7.4f} "
              f"[{r.mean_ci_lo:7.4f},{r.mean_ci_hi:7.4f}] {r.mean_regret:8.4f} "
              f"{r.ceiling_hit_rate:8.4f} {r.exact_k_rate:7.4f}")
    print("\nMACRO-FAMILY mean (mean of 12 family means), reported separately:")
    for c in ARMS:
        print(f"  {c}: {fam[fam.arm == c]['mean_best_view_ari'].mean():.4f}")

    print("\n" + "=" * 78); print("PAIRED CONTRASTS (n=1,055 datasets)"); print("=" * 78)
    cs = contrasts.set_index("contrast")
    print(f"{'contrast':18} {'mean_d':>8} {'med_d':>8} {'CI95':>18} "
          f"{'W/T/L':>16} {'p_holm':>9}")
    for cid in list(cs.index):
        r = cs.loc[cid]
        ci = (f"[{r.get('mean_ci_lo', float('nan')):7.4f},"
              f"{r.get('mean_ci_hi', float('nan')):7.4f}]")
        ph = r.get("holm_p", np.nan)
        print(f"{cid:18} {r['mean_delta']:8.4f} {r['median_delta']:8.4f} {ci:>18} "
              f"{int(r['wins']):5d}/{int(r['ties']):4d}/{int(r['losses']):5d} "
              f"{(ph if isinstance(ph,(int,float)) else float('nan')):9.2e}")

    print("\n" + "=" * 78); print("FAMILY TABLE (mean Best-View ARI)"); print("=" * 78)
    p = fam.pivot_table(index="family", columns="arm", values="mean_best_view_ari")
    n = fam[fam.arm == "HU"].set_index("family")["n_datasets"]
    p = p[ARMS]
    p.insert(0, "N", n)
    p["HP-HU"] = p["HP"] - p["HU"]; p["FU-HU"] = p["FU"] - p["HU"]
    p["FP-HP"] = p["FP"] - p["HP"]
    print(p.round(4).to_string())
    p.round(6).to_csv(OUT / "offline_2x3_family_table.csv")

    print(f"\n  package -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
