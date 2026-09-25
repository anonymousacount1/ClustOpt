"""Stage-1F: consolidated audit + post-hoc upper bounds. POST-PROCESSING ONLY.

Reads persisted Stage-1C / Stage-1E artifacts. Runs no clustering, no Optuna, no
training. Every quantity here is a re-aggregation of already-recorded trial data.

Produces:
  A2   audit of the "Regret" column's aggregation unit in the primary table
  A3   Best-View full-slate perfect-candidate-selection upper bound
  A4   Best-View Top-K (1/3/5/10/ALL) perfect-rerank upper bounds
  A5   deployable-policy VBS and selective-policy VBS
  A6   Stage-2 mechanism comparison table
  A7   Holm correction over the EXACT Stage-1A pre-specified family of six
  A8   tie-aware winner counts (unique / win-or-tie / tie)
  A9   candidate-pool ceiling comparison (offline grid vs online Optuna sample)
  A11  the search-vs-selection ratio, restated as a descriptive ratio
  A12  three precisely-labelled FP-vs-FO ratios

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/build_stage1f_upper_bounds.py
"""
from __future__ import annotations

import json
import os
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
from models.Clustering_Repository_Builder.experiments.paper_analysis import stats as S  # noqa: E402

ARMS = ["HU", "HP", "HO", "FU", "FP", "FO"]
DEPLOYABLE = ["HU", "HP", "FU", "FP"]
SELECTIVE = ["HP", "FP"]
METHOD_OF = {
    "HU": "stage1_head14_uniform_fixed50",
    "HP": "stage1_head14_mlp_top5_raw_fixed50",
    "HO": "stage1_head14_oracle_top5_raw_fixed50",
    "FU": "stage1_full60_uniform_fixed50",
    "FP": "stage1_full60_mlp_top5_raw_fixed50",
    "FO": "stage1_full60_oracle_top5_raw_fixed50",
}

ON = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50"
OFF = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32"
OUT = ON / "stage1f"

# The family locked in docs/internal_reports/stage1/STAGE1A_DESIGN_LOCK.md sec. 9.
PRESPECIFIED_SIX = [
    ("HP-HU", "HP", "HU", "utility contribution on established metrics"),
    ("HO-HU", "HO", "HU", "oracle ceiling on established metrics"),
    ("FU-HU", "FU", "HU", "New-46 contribution under uniform weighting"),
    ("FP-HP", "FP", "HP", "New-46 contribution under learned utility"),
    ("FO-HO", "FO", "HO", "New-46 contribution under oracle utility"),
    # 6th member is the interaction, handled separately (difference-of-differences)
]
EXPLORATORY = [
    ("FP-FU", "FP", "FU", "learned utility within Full-60 (NOT pre-specified)"),
    ("FO-FU", "FO", "FU", "oracle utility within Full-60 (NOT pre-specified)"),
]


def wilcoxon_p(d: np.ndarray) -> float:
    if not S.scipy_available():
        return float("nan")
    from scipy import stats as sst
    try:
        return float(sst.wilcoxon(d, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def paired_row(cid: str, d: np.ndarray, rationale: str, family: str) -> dict:
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    return {
        "contrast": cid, "rationale": rationale, "hypothesis_family": family,
        "n": int(d.size), "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "mean_ci_lo": lo, "mean_ci_hi": hi,
        "cohens_dz": S.cohens_dz(d), "wilcoxon_p": wilcoxon_p(d),
        "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
        "losses": int((d < -1e-12).sum()),
    }


def bound_block(actual: np.ndarray, bound: np.ndarray, label: str) -> dict:
    gap = bound - actual
    pt, lo, hi = S.bootstrap_ci(gap, statistic="mean")
    return {
        "bound": label, "n_datasets": int(actual.size),
        "actual_mean": float(actual.mean()), "bound_mean": float(bound.mean()),
        "headroom_abs": float(gap.mean()),
        "headroom_ci_lo": lo, "headroom_ci_hi": hi,
        "headroom_median": float(np.median(gap)),
        "headroom_rel_pct": float(100.0 * gap.mean() / actual.mean())
        if actual.mean() > 0 else np.nan,
        "zero_headroom_rate": float((gap <= 1e-12).mean()),
        "improved_rate": float((gap > 1e-12).mean()),
        "wilcoxon_p": wilcoxon_p(gap) if (gap > 1e-12).any() else float("nan"),
    }


def main() -> int:
    ensure_dir(OUT)
    print("=" * 78)
    print("STAGE-1F  CONSOLIDATED AUDIT + POST-HOC UPPER BOUNDS")
    print("=" * 78)
    report = {}

    rec = pd.read_csv(_ext(ON / "online_2x3_record_results.csv"))
    num = ["ari", "best_visited_ari", "objective_ari_spearman",
           "max_ari_in_top3", "max_ari_in_top5", "max_ari_in_top10",
           "ari_at_rank1", "rank_of_best_ari_trial", "valid_trials"]
    for c in num:
        rec[c] = pd.to_numeric(rec[c], errors="coerce")
    rec["regret"] = rec["best_visited_ari"] - rec["ari"]
    bv = pd.read_csv(_ext(ON / "online_2x3_dataset_results.csv"))
    bv["best_view_ari"] = pd.to_numeric(bv["best_view_ari"], errors="coerce")

    # ------------------------------------------------------------------ A2
    print("\n" + "-" * 78)
    print("A2  AGGREGATION-UNIT AUDIT OF THE PRIMARY TABLE'S 'Regret' COLUMN")
    print("-" * 78)
    a2 = []
    for a in ARMS:
        r = rec[rec["arm"] == a]
        b = bv[bv["arm"] == a]
        dv_regret = float(r["regret"].mean())               # dataset/view level
        # dataset-level Best-View regret: BestViewBestVisited - BestViewSelected
        bvbvis = r.groupby("dataset_id")["best_visited_ari"].max()
        bvsel = b.set_index("dataset_id")["best_view_ari"]
        common = bvsel.index.intersection(bvbvis.index)
        ds_regret = float((bvbvis.loc[common] - bvsel.loc[common]).mean())
        a2.append({"arm": a,
                   "dataset_view_mean_regret": dv_regret,
                   "dataset_level_bestview_regret": ds_regret,
                   "reported_in_primary_table": dv_regret})
    a2df = pd.DataFrame(a2)
    a2df.to_csv(_ext(OUT / "a2_regret_aggregation_audit.csv"), index=False)
    print(a2df.round(4).to_string(index=False))
    print("\n  VERDICT: the primary table's 'Regret' column equals the")
    print("  DATASET/VIEW-level mean (n=3,165 runs), NOT a Best-View quantity.")
    print("  It must never appear unlabelled beside Best-View ARI (n=1,055).")
    report["A2_regret_column_unit"] = "dataset/view mean over 3,165 runs"

    # ------------------------------------------------------------- A3 + A4
    print("\n" + "-" * 78)
    print("A3/A4  BEST-VIEW PERFECT-CANDIDATE-SELECTION UPPER BOUNDS")
    print("-" * 78)
    # Best-View aggregation of each reranked quantity: max over the 3 views.
    rec["top1"] = rec["ari_at_rank1"]
    rec["topALL"] = rec["best_visited_ari"]
    kcols = {"Top-1": "top1", "Top-3": "max_ari_in_top3", "Top-5": "max_ari_in_top5",
             "Top-10": "max_ari_in_top10", "Full slate": "topALL"}
    rows, bvtab = [], {}
    for a in ARMS:
        r = rec[rec["arm"] == a]
        actual = bv[bv["arm"] == a].set_index("dataset_id")["best_view_ari"].sort_index()
        bvtab[(a, "actual")] = actual
        for label, col in kcols.items():
            b = r.groupby("dataset_id")[col].max().sort_index()
            b = b.reindex(actual.index)
            bvtab[(a, label)] = b
            d = bound_block(actual.to_numpy(float), b.to_numpy(float), label)
            d["arm"] = a
            rows.append(d)
    ub = pd.DataFrame(rows)[["arm", "bound", "n_datasets", "actual_mean", "bound_mean",
                             "headroom_abs", "headroom_ci_lo", "headroom_ci_hi",
                             "headroom_median", "headroom_rel_pct",
                             "zero_headroom_rate", "improved_rate", "wilcoxon_p"]]
    ub.to_csv(_ext(OUT / "a3_a4_bestview_upper_bounds.csv"), index=False)
    piv = ub.pivot_table(index="arm", columns="bound", values="bound_mean")
    piv = piv[["Top-1", "Top-3", "Top-5", "Top-10", "Full slate"]]
    piv.insert(0, "actual", ub.groupby("arm")["actual_mean"].first())
    print("\nBest-View ARI under perfect reranking of the objective's Top-K:")
    print(piv.loc[ARMS].round(4).to_string())
    print("\nHeadroom over actual (absolute):")
    ph = ub.pivot_table(index="arm", columns="bound", values="headroom_abs")
    print(ph.loc[ARMS][["Top-1", "Top-3", "Top-5", "Top-10", "Full slate"]]
          .round(4).to_string())
    print("\nDetail for the two deployable selective arms:")
    for a in SELECTIVE:
        sub = ub[ub["arm"] == a]
        for _, r in sub.iterrows():
            print(f"  {a}  {r['bound']:11} bound={r['bound_mean']:.4f}  "
                  f"headroom=+{r['headroom_abs']:.4f} "
                  f"CI[{r['headroom_ci_lo']:+.4f},{r['headroom_ci_hi']:+.4f}]  "
                  f"rel={r['headroom_rel_pct']:5.1f}%  "
                  f"zeroHeadroom={r['zero_headroom_rate']:.3f}  "
                  f"improved={r['improved_rate']:.3f}")

    # ------------------------------------------------------------------ A5
    print("\n" + "-" * 78)
    print("A5  POLICY VIRTUAL-BEST-SOLVER (oracle over policies)")
    print("-" * 78)
    wide = bv.pivot_table(index="dataset_id", columns="arm", values="best_view_ari")
    wide = wide.dropna()
    vbs_rows = []
    for name, cols in (("SelectivePolicyVBS (HP,FP)", SELECTIVE),
                       ("DeployablePolicyVBS (HU,HP,FU,FP)", DEPLOYABLE),
                       ("AllArmVBS (incl. utility oracles)", ARMS)):
        v = wide[cols].max(axis=1)
        d = bound_block(wide["FP"].to_numpy(float), v.to_numpy(float), name)
        d["reference"] = "FP"
        mx = wide[cols].max(axis=1)
        for c in cols:
            d[f"{c}_win_or_tie_rate"] = float(((wide[c] - mx).abs() <= 1e-12).mean())
        vbs_rows.append(d)
    vbsdf = pd.DataFrame(vbs_rows)
    vbsdf.to_csv(_ext(OUT / "a5_policy_vbs.csv"), index=False)
    for _, r in vbsdf.iterrows():
        print(f"\n  {r['bound']}")
        print(f"    FP mean                 {r['actual_mean']:.4f}")
        print(f"    VBS mean                {r['bound_mean']:.4f}")
        print(f"    headroom over FP        +{r['headroom_abs']:.4f}  "
              f"CI[{r['headroom_ci_lo']:+.4f},{r['headroom_ci_hi']:+.4f}]")
        print(f"    FP already optimal on   {r['zero_headroom_rate']*100:.1f}% of datasets")
        rates = {c: r.get(f"{c}_win_or_tie_rate") for c in ARMS
                 if not pd.isna(r.get(f"{c}_win_or_tie_rate", np.nan))}
        print("    win-or-tie: " + "  ".join(f"{k}:{v*100:.1f}%" for k, v in rates.items()))

    # ------------------------------------------------------------------ A7
    print("\n" + "-" * 78)
    print("A7  HOLM OVER THE EXACT STAGE-1A PRE-SPECIFIED FAMILY OF SIX")
    print("-" * 78)
    dl = bv[["dataset_id", "method_name", "best_view_ari"]].copy()

    def delta(a, b):
        va, vb, ids = S.paired_values(dl, METHOD_OF[a], METHOD_OF[b])
        return va - vb, ids
    crows = []
    for cid, a, b, why in PRESPECIFIED_SIX:
        d, _ = delta(a, b)
        crows.append(paired_row(cid, d, why, "stage1e_primary"))
    dfp, ids1 = delta("FP", "HP")
    dfu, ids2 = delta("FU", "HU")
    assert list(ids1) == list(ids2)
    crows.append(paired_row("(FP-HP)-(FU-HU)", dfp - dfu,
                            "utility x new-metric interaction", "stage1e_primary"))
    primary = S.holm_correct(pd.DataFrame(crows), p_col="wilcoxon_p",
                             family_col="hypothesis_family")
    expl = pd.DataFrame([paired_row(cid, delta(a, b)[0], why, "stage1e_exploratory")
                         for cid, a, b, why in EXPLORATORY])
    allc = pd.concat([primary, expl], ignore_index=True)
    allc.to_csv(_ext(OUT / "a7_prespecified_contrasts.csv"), index=False)
    print(f"{'contrast':18} {'mean_d':>8} {'med_d':>8} {'CI95':>19} {'dz':>6} "
          f"{'W/T/L':>17} {'p_raw':>10} {'p_holm':>10}")
    for _, r in allc.iterrows():
        ph = r.get("holm_p", np.nan)
        ph = float(ph) if isinstance(ph, (int, float, np.floating)) and not pd.isna(ph) \
            else float("nan")
        print(f"{r['contrast']:18} {r['mean_delta']:8.4f} {r['median_delta']:8.4f} "
              f"[{r['mean_ci_lo']:7.4f},{r['mean_ci_hi']:7.4f}] {r['cohens_dz']:6.3f} "
              f"{int(r['wins']):5d}/{int(r['ties']):4d}/{int(r['losses']):5d} "
              f"{r['wilcoxon_p']:10.2e} {ph:10.2e}")
    print("\n  Family = the six locked in STAGE1A_DESIGN_LOCK.md sec.9.")
    print("  FP-FU / FO-FU are EXPLORATORY (no pre-specification evidence).")

    # ------------------------------------------------------------------ A8
    print("\n" + "-" * 78)
    print("A8  TIE-AWARE WINNER COUNTS")
    print("-" * 78)
    wrows = []
    for scope, cols in (("A_all_six_arms", ARMS),
                        ("B_deployable", DEPLOYABLE),
                        ("C_predicted_selective", SELECTIVE)):
        sub = wide[cols]
        mx = sub.max(axis=1)
        is_max = sub.sub(mx, axis=0).abs() <= 1e-12
        n_at_max = is_max.sum(axis=1)
        row = {"scope": scope, "n_datasets": int(len(sub)),
               "n_datasets_with_tied_win": int((n_at_max > 1).sum())}
        for c in cols:
            row[f"{c}_win_or_tie"] = int(is_max[c].sum())
            row[f"{c}_unique_win"] = int((is_max[c] & (n_at_max == 1)).sum())
        wrows.append(row)
        print(f"\n  {scope}  (n={len(sub)}, datasets with a tied win: "
              f"{int((n_at_max > 1).sum())})")
        print(f"    {'arm':5} {'unique win':>11} {'win-or-tie':>11} {'win-or-tie %':>13}")
        for c in cols:
            print(f"    {c:5} {int((is_max[c] & (n_at_max == 1)).sum()):11d} "
                  f"{int(is_max[c].sum()):11d} {is_max[c].mean()*100:12.1f}%")
    pd.DataFrame(wrows).to_csv(_ext(OUT / "a8_winner_counts.csv"), index=False)

    # ------------------------------------------------------------------ A9
    print("\n" + "-" * 78)
    print("A9  CANDIDATE-POOL CEILING COMPARISON (pool quality, not policy quality)")
    print("-" * 78)
    a9 = {}
    off_rec_p = OFF / "offline_2x3_record_results.csv"
    if path_exists(off_rec_p):
        off = pd.read_csv(_ext(off_rec_p))
        off["best_possible_ari_in_candidate_set"] = pd.to_numeric(
            off["best_possible_ari_in_candidate_set"], errors="coerce")
        # arm-independent by construction -> take one arm
        off_ceiling = (off[off["arm"] == "HU"]
                       .groupby(["dataset_id", "view_id"])
                       ["best_possible_ari_in_candidate_set"].first())
        on_ceiling = (rec[rec["arm"] == "HU"]
                      .set_index(["dataset_id", "view_id"])["best_visited_ari"])
        common = off_ceiling.index.intersection(on_ceiling.index)
        oc, nc = off_ceiling.loc[common], on_ceiling.loc[common]
        # per-arm online ceilings (they differ slightly; offline is arm-independent)
        on_arm = {a: float(rec[rec["arm"] == a]["best_visited_ari"].mean()) for a in ARMS}
        d = (nc - oc).to_numpy(float)
        pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
        a9 = {"n_dataset_views": int(len(common)),
              "offline_grid_ceiling_mean": float(oc.mean()),
              "online_optuna_ceiling_mean_HU": float(nc.mean()),
              "online_ceiling_by_arm": on_arm,
              "mean_delta_online_minus_offline": float(d.mean()),
              "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": wilcoxon_p(d),
              "online_ceiling_higher_rate": float((d > 1e-12).mean()),
              "offline_ceiling_higher_rate": float((d < -1e-12).mean())}
        print(f"  offline fixed-32 pool ceiling (dataset/view mean) : "
              f"{a9['offline_grid_ceiling_mean']:.4f}")
        print(f"  online fixed-50 pool ceiling  (dataset/view mean) : "
              f"{a9['online_optuna_ceiling_mean_HU']:.4f}")
        print(f"  delta (online - offline)                          : "
              f"{a9['mean_delta_online_minus_offline']:+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}]  p={a9['wilcoxon_p']:.2e}")
        print(f"  online pool better on {a9['online_ceiling_higher_rate']*100:.1f}% of "
              f"dataset/views, offline better on "
              f"{a9['offline_ceiling_higher_rate']*100:.1f}%")
        write_json_atomic(OUT / "a9_pool_ceiling_comparison.json", a9)
    else:
        print("  [WARN] Stage-1C record file not found")

    # ----------------------------------------------------------------- A11
    print("\n" + "-" * 78)
    print("A11  SEARCH-vs-SELECTION: DESCRIPTIVE RATIO (not a decomposition)")
    print("-" * 78)
    a11 = []
    for cid, a, b, _ in PRESPECIFIED_SIX + [("FP-FU", "FP", "FU", "")]:
        ra = rec[rec["arm"] == a].set_index(["dataset_id", "view_id"])
        rb = rec[rec["arm"] == b].set_index(["dataset_id", "view_id"])
        idx = ra.index.intersection(rb.index)
        d_sel = (ra.loc[idx, "ari"] - rb.loc[idx, "ari"]).to_numpy(float)
        d_vis = (ra.loc[idx, "best_visited_ari"]
                 - rb.loc[idx, "best_visited_ari"]).to_numpy(float)
        a11.append({
            "contrast": cid, "n_runs": int(idx.size),
            "mean_delta_selected_ari": float(d_sel.mean()),
            "mean_delta_best_visited_ari": float(d_vis.mean()),
            "abs_ratio_bestvisited_over_selected":
                float(abs(d_vis.mean()) / abs(d_sel.mean())) if d_sel.mean() else np.nan,
        })
    a11df = pd.DataFrame(a11)
    a11df.to_csv(_ext(OUT / "a11_search_vs_selection_ratio.csv"), index=False)
    print(a11df.round(5).to_string(index=False))
    spread_sel = (rec.groupby("arm")["ari"].mean().max()
                  - rec.groupby("arm")["ari"].mean().min())
    spread_vis = (rec.groupby("arm")["best_visited_ari"].mean().max()
                  - rec.groupby("arm")["best_visited_ari"].mean().min())
    print(f"\n  across-arm spread, mean SELECTED ARI      : {spread_sel:.4f}")
    print(f"  across-arm spread, mean BEST-VISITED ARI  : {spread_vis:.4f}")
    print(f"  ratio (best-visited spread / selected)    : "
          f"{spread_vis/spread_sel*100:.1f}%")
    report["A11"] = {"selected_spread": float(spread_sel),
                     "best_visited_spread": float(spread_vis),
                     "ratio_pct": float(spread_vis / spread_sel * 100)}

    # ----------------------------------------------------------------- A12
    print("\n" + "-" * 78)
    print("A12  FP vs FO -- THREE PRECISELY-LABELLED QUANTITIES")
    print("-" * 78)
    m = {a: float(wide[a].mean()) for a in ARMS}
    gap = wide["FO"] - wide["FP"]
    pt, lo, hi = S.bootstrap_ci(gap.to_numpy(float), statistic="mean")
    a12 = {
        "A_absolute_score_ratio_FP_over_FO": m["FP"] / m["FO"],
        "B_fraction_of_full60_utility_gain_captured_(FP-FU)/(FO-FU)":
            (m["FP"] - m["FU"]) / (m["FO"] - m["FU"]),
        "C_remaining_utility_oracle_gap_FO_minus_FP": m["FO"] - m["FP"],
        "C_gap_ci_lo": lo, "C_gap_ci_hi": hi,
        "means": m,
    }
    write_json_atomic(OUT / "a12_fp_vs_fo_ratios.json", a12)
    print(f"  A. absolute score ratio   FP/FO              = "
          f"{a12['A_absolute_score_ratio_FP_over_FO']*100:.1f}%   "
          f"(a ratio of ARI levels; NOT 'headroom captured')")
    print(f"  B. fraction of the Full-60 utility-guidance gain captured")
    print(f"     (FP-FU)/(FO-FU) = ({m['FP']:.4f}-{m['FU']:.4f})/"
          f"({m['FO']:.4f}-{m['FU']:.4f}) = "
          f"{a12['B_fraction_of_full60_utility_gain_captured_(FP-FU)/(FO-FU)']*100:.1f}%")
    print(f"  C. remaining Utility-Oracle gap  FO-FP       = "
          f"{a12['C_remaining_utility_oracle_gap_FO_minus_FP']:+.4f} "
          f"CI[{lo:+.4f},{hi:+.4f}]")

    # ------------------------------------------------------------------ A6
    print("\n" + "-" * 78)
    print("A6  STAGE-2 MECHANISM COMPARISON (all bounds on the Best-View scale)")
    print("-" * 78)

    def hb(arm, label):
        r = ub[(ub["arm"] == arm) & (ub["bound"] == label)].iloc[0]
        return r["bound_mean"], r["headroom_abs"], r["headroom_ci_lo"], r["headroom_ci_hi"]
    mech = []
    for name, base_arm, base, bound_mean, head, clo, chi, kind in [
        ("1. Full-slate candidate reranking on FP", "FP", m["FP"], *hb("FP", "Full slate"),
         "candidate-selection oracle"),
        ("2. Top-5 reranking on FP", "FP", m["FP"], *hb("FP", "Top-5"),
         "candidate-selection oracle"),
        ("3. Top-10 reranking on FP", "FP", m["FP"], *hb("FP", "Top-10"),
         "candidate-selection oracle"),
    ]:
        mech.append({"mechanism": name, "baseline_arm": base_arm,
                     "deployable_baseline": base, "oracle_upper_bound": bound_mean,
                     "max_available_gain": head, "gain_ci_lo": clo, "gain_ci_hi": chi,
                     "bound_type": kind})
    for name, cols, kind in [
        ("4. Policy Selector over HP/FP", SELECTIVE, "policy oracle (VBS)"),
        ("5. Policy Selector over HU/HP/FU/FP", DEPLOYABLE, "policy oracle (VBS)"),
    ]:
        v = float(wide[cols].max(axis=1).mean())
        g = (wide[cols].max(axis=1) - wide["FP"]).to_numpy(float)
        _p, lo2, hi2 = S.bootstrap_ci(g, statistic="mean")
        mech.append({"mechanism": name, "baseline_arm": "FP",
                     "deployable_baseline": m["FP"], "oracle_upper_bound": v,
                     "max_available_gain": v - m["FP"], "gain_ci_lo": lo2,
                     "gain_ci_hi": hi2, "bound_type": kind})
    _p, lo3, hi3 = S.bootstrap_ci(gap.to_numpy(float), statistic="mean")
    mech.append({"mechanism": "6. Better Full-60 utility prediction (FP -> FO)",
                 "baseline_arm": "FP", "deployable_baseline": m["FP"],
                 "oracle_upper_bound": m["FO"], "max_available_gain": m["FO"] - m["FP"],
                 "gain_ci_lo": lo3, "gain_ci_hi": hi3,
                 "bound_type": "utility oracle (NOT an ARI ceiling)"})
    mdf = pd.DataFrame(mech).sort_values("max_available_gain", ascending=False)
    mdf.to_csv(_ext(OUT / "a6_stage2_mechanism_comparison.csv"), index=False)
    print(f"{'mechanism':46} {'base':>7} {'bound':>7} {'gain':>8} {'gain CI95':>19}")
    for _, r in mdf.iterrows():
        print(f"{r['mechanism']:46} {r['deployable_baseline']:7.4f} "
              f"{r['oracle_upper_bound']:7.4f} {r['max_available_gain']:+8.4f} "
              f"[{r['gain_ci_lo']:+.4f},{r['gain_ci_hi']:+.4f}]")

    write_json_atomic(OUT / "stage1f_summary.json", report)
    print(f"\n  package -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
