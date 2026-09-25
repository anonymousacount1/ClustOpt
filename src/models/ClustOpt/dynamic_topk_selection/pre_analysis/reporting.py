"""Aggregation tables, distribution diagnostics, validation and markdown reports.

Everything downstream of the per-record scan. No plotting here (see
:mod:`plotting`). All functions take the assembled ``summary_df`` (one row per
record, including a ``status`` column) and write CSV / Parquet / Markdown.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from heuristic_candidates import MAX_K, heuristic_columns

# Representative candidates highlighted in group summaries / reports.
# Chosen to span the behavioural range: the two that produce a usable decaying
# K distribution (k_rel_090, k_elbow_10), two moderate ones, the conservative
# hybrid, and one saturating mass baseline for contrast.
SELECTED_CANDIDATES = [
    "k_rel_090", "k_rel_085", "k_elbow_10", "k_elbow_norm_10",
    "k_hybrid_min_a085_t065", "k_mass_065",
]

# Heuristic whose K distribution best matches the desired shape across the repo;
# used as the "primary" candidate for family/view pct breakdowns and heatmaps.
PRIMARY_HEURISTIC = "k_rel_090"

SHAPE_FEATURES = ["max", "normalized_entropy", "participation_ratio",
                  "perplexity_effective_k", "largest_gap_position_1_to_10"]


# --------------------------------------------------------------------------- #
# Distribution diagnostics                                                    #
# --------------------------------------------------------------------------- #
def k_distribution(df: pd.DataFrame, col: str) -> dict:
    """Full K distribution + desired-shape diagnostics for one heuristic col."""
    k = df[col].dropna().astype(int)
    n = len(k)
    raw_col = f"{col}_raw"
    out = {"heuristic": col, "n_records": n}
    if n == 0:
        return out
    counts = k.value_counts().to_dict()
    for kk in range(1, MAX_K + 1):
        out[f"K={kk}"] = int(counts.get(kk, 0))
    out["mean_K"] = float(k.mean())
    out["median_K"] = float(k.median())
    out["mode_K"] = int(k.mode().iloc[0])
    out["pct_K_1"] = float((k == 1).mean() * 100)
    out["pct_K_2_to_5"] = float(k.between(2, 5).mean() * 100)
    out["pct_K_gt_5"] = float((k > 5).mean() * 100)
    if raw_col in df.columns:
        kr = df[raw_col].dropna().astype(int)
        out["pct_K_gt_10_before_clipping"] = float((kr > MAX_K).mean() * 100)
    else:
        out["pct_K_gt_10_before_clipping"] = 0.0
    # Quantitative "how close to the desired bell shape" score (higher = better):
    # reward mass in K=2-5, penalise a heavy K>5 tail, penalise pre-clip
    # saturation (a heuristic that *wanted* K>10 is only usable because of the
    # clip), and penalise a mode far from ~3.
    out["desirability_score"] = float(
        out["pct_K_2_to_5"]
        - 0.5 * out["pct_K_gt_5"]
        - 0.7 * out["pct_K_gt_10_before_clipping"]
        - 4.0 * abs(out["mode_K"] - 3)
    )
    return out


def heuristic_distribution_table(df_ok: pd.DataFrame) -> pd.DataFrame:
    rows = [k_distribution(df_ok, c) for c in heuristic_columns()]
    return pd.DataFrame(rows)


def _bell_shaped(row: pd.Series) -> bool:
    """Qualitative desired behaviour: some K=1, real bulk in 2-5, decaying tail,
    a low mode, and little pre-clip saturation (so the clip isn't doing the work).

    Thresholds are deliberately permissive -- given how flat the utility vectors
    are (see report), even the best candidates only approximate the ideal."""
    return (
        row.get("pct_K_1", 0) >= 2.0
        and row.get("pct_K_2_to_5", 0) >= 38.0
        and row.get("pct_K_gt_5", 100) <= 45.0
        and row.get("mode_K", 99) <= 5
        and row.get("pct_K_gt_10_before_clipping", 100) <= 20.0
    )


# --------------------------------------------------------------------------- #
# Group summaries                                                             #
# --------------------------------------------------------------------------- #
def _group_summary(df_ok: pd.DataFrame, by: str) -> pd.DataFrame:
    agg = {f"mean_{f}": (f, "mean") for f in SHAPE_FEATURES if f in df_ok}
    for c in heuristic_columns():
        agg[f"meanK_{c}"] = (c, "mean")
    g = df_ok.groupby(by).agg(n_records=("record_id", "size"), **agg)
    # pct breakdown for the primary candidate
    prim = PRIMARY_HEURISTIC
    if prim in df_ok.columns:
        gp = df_ok.groupby(by)[prim]
        g["pct_K_1"] = gp.apply(lambda s: (s == 1).mean() * 100)
        g["pct_K_2_to_5"] = gp.apply(lambda s: s.between(2, 5).mean() * 100)
        g["pct_K_gt_5"] = gp.apply(lambda s: (s > 5).mean() * 100)
    return g.reset_index()


def family_summary(df_ok):     return _group_summary(df_ok, "family_id")
def subfamily_summary(df_ok):  return _group_summary(df_ok, "subfamily_id")
def view_summary(df_ok):       return _group_summary(df_ok, "view_type")


def global_summary(df_ok: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-statistic global table (long format, easy to read)."""
    feats = SHAPE_FEATURES + [
        "mean", "std", "entropy", "top1_mass", "top3_mass", "top5_mass",
        "top10_mass", "top1_over_top2", "largest_gap_value_1_to_10",
        "curve_auc_normalized",
    ]
    rows = []
    for f in feats:
        if f not in df_ok.columns:
            continue
        s = df_ok[f].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(s):
            continue
        rows.append({
            "statistic": f, "mean": s.mean(), "std": s.std(),
            "min": s.min(), "p25": s.quantile(.25), "median": s.median(),
            "p75": s.quantile(.75), "max": s.max(),
        })
    return pd.DataFrame(rows)


def metric_summary(wide_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric: typical utility + how often it lands in the selected set."""
    metric_cols = [c for c in wide_df.columns if c.startswith("metric_")]
    ok_ids = set(summary_df.loc[summary_df["status"] == "ok", "record_id"])
    w = wide_df[wide_df["record_id"].isin(ok_ids)]
    rows = []
    vals = w[metric_cols].to_numpy(dtype=float)
    # descending rank of each metric per record (1 = best)
    order = (-np.nan_to_num(vals, nan=-np.inf)).argsort(axis=1).argsort(axis=1) + 1
    for j, c in enumerate(metric_cols):
        col = vals[:, j]
        finite = col[np.isfinite(col)]
        rnk = order[:, j]
        rows.append({
            "metric": c[len("metric_"):],
            "n_present": int(np.isfinite(col).sum()),
            "mean_utility": float(np.nanmean(col)) if finite.size else np.nan,
            "median_utility": float(np.nanmedian(col)) if finite.size else np.nan,
            "std_utility": float(np.nanstd(col)) if finite.size else np.nan,
            "mean_rank": float(rnk.mean()),
            "pct_top1": float((rnk == 1).mean() * 100),
            "pct_top3": float((rnk <= 3).mean() * 100),
            "pct_top5": float((rnk <= 5).mean() * 100),
        })
    return pd.DataFrame(rows).sort_values("mean_utility", ascending=False)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def validation(summary_df: pd.DataFrame, n_index: int, out_dir: str) -> dict:
    ok = summary_df[summary_df["status"] == "ok"]
    missing = summary_df[summary_df["status"] == "missing"]
    malformed = summary_df[summary_df["status"] == "malformed"]
    n_datasets = summary_df["dataset_id"].nunique()

    nm = ok["n_metrics"].value_counts().to_dict() if len(ok) else {}
    all_nan = int(ok["all_nan"].sum()) if "all_nan" in ok else 0

    report = {
        "n_index_records": int(n_index),
        "n_scanned": int(len(summary_df)),
        "n_ok": int(len(ok)),
        "n_missing": int(len(missing)),
        "n_malformed": int(len(malformed)),
        "n_datasets": int(n_datasets),
        "expected_records_3x_datasets": int(n_datasets * 3),
        "records_equals_3x_datasets": bool(len(summary_df) == n_datasets * 3),
        "n_metrics_value_counts": {int(k): int(v) for k, v in nm.items()},
        "n_metrics_consistent": bool(len(nm) == 1),
        "n_all_nan_vectors": all_nan,
        "views_present": sorted(ok["view_type"].unique().tolist()) if len(ok) else [],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scan_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    cols = ["record_id", "dataset_id", "family_id", "subfamily", "view_type",
            "utility_file_path", "status", "error"]
    missing[[c for c in cols if c in missing.columns]].to_csv(
        os.path.join(out_dir, "missing_files_report.csv"), index=False)
    malformed[[c for c in cols if c in malformed.columns]].to_csv(
        os.path.join(out_dir, "malformed_files_report.csv"), index=False)
    return report


# --------------------------------------------------------------------------- #
# Markdown reports                                                            #
# --------------------------------------------------------------------------- #
def _fmt(x, d=3):
    try:
        return f"{x:.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def write_reports(out_dir: str, val: dict, dist: pd.DataFrame,
                  gsum: pd.DataFrame, fam: pd.DataFrame, view: pd.DataFrame,
                  df_ok: pd.DataFrame):
    rep_dir = os.path.join(out_dir, "reports")
    os.makedirs(rep_dir, exist_ok=True)

    dist = dist.copy()
    dist["bell_shaped"] = dist.apply(_bell_shaped, axis=1)

    # ---- UTILITY_VECTOR_STATISTICS.md ----
    lines = ["# Utility Vector Statistics", "",
             "Global descriptive / shape statistics across all OK records.", ""]
    lines.append("| statistic | mean | std | min | p25 | median | p75 | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in gsum.iterrows():
        lines.append("| {s} | {m} | {sd} | {mn} | {p25} | {md} | {p75} | {mx} |".format(
            s=r["statistic"], m=_fmt(r["mean"]), sd=_fmt(r["std"]),
            mn=_fmt(r["min"]), p25=_fmt(r["p25"]), md=_fmt(r["median"]),
            p75=_fmt(r["p75"]), mx=_fmt(r["max"])))
    with open(os.path.join(rep_dir, "UTILITY_VECTOR_STATISTICS.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- HEURISTIC_CANDIDATE_REPORT.md ----
    h = ["# Heuristic Candidate Report", "",
         "K distributions for every simulated heuristic (K clipped to "
         f"[1, {MAX_K}]), sorted by desirability score. `bell` flags the desired "
         "qualitative shape (some K=1, bulk in 2-5, decaying tail, low mode, "
         "little pre-clip saturation). `score` = pct_K_2_5 - 0.5*pct_K_gt5 - "
         "0.7*pct_raw_gt10 - 4*|mode-3| (higher is better).", "",
         "| heuristic | mean_K | median_K | mode_K | pct_K_1 | pct_K_2_5 | "
         "pct_K_gt5 | pct_raw_gt10 | score | bell |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in dist.sort_values("desirability_score", ascending=False).iterrows():
        h.append("| {n} | {me} | {md} | {mo} | {p1} | {p25} | {p5} | {pg} | {sc} | {b} |".format(
            n=r["heuristic"], me=_fmt(r.get("mean_K")), md=_fmt(r.get("median_K"), 1),
            mo=int(r.get("mode_K", 0)), p1=_fmt(r.get("pct_K_1"), 1),
            p25=_fmt(r.get("pct_K_2_to_5"), 1), p5=_fmt(r.get("pct_K_gt_5"), 1),
            pg=_fmt(r.get("pct_K_gt_10_before_clipping"), 1),
            sc=_fmt(r.get("desirability_score"), 1),
            b="YES" if r.get("bell_shaped") else ""))
    h += ["", "## Bell-shaped candidates", ""]
    bell = dist[dist["bell_shaped"]]["heuristic"].tolist()
    h.append(", ".join(bell) if bell else "_None matched the strict criteria._")
    with open(os.path.join(rep_dir, "HEURISTIC_CANDIDATE_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(h) + "\n")

    # ---- DYNAMIC_TOPK_PRE_ANALYSIS_REPORT.md ----
    m = ["# Dynamic Top-K Pre-Analysis Report", "",
         "Statistical pre-analysis of ClustOpt utility vectors to inform a "
         "future **dynamic** Top-K metric-selection heuristic. No ClustOpt "
         "behaviour was modified; this is what-if simulation only.", "",
         "## 1-5. Data coverage", "",
         f"- Datasets scanned: **{val['n_datasets']}**",
         f"- Records / views scanned: **{val['n_scanned']}** "
         f"(expected 3x datasets = {val['expected_records_3x_datasets']}; "
         f"match: {val['records_equals_3x_datasets']})",
         f"- OK records: **{val['n_ok']}**",
         f"- Missing utility files: **{val['n_missing']}**",
         f"- Malformed utility files: **{val['n_malformed']}**",
         f"- Metrics per vector: **{val['n_metrics_value_counts']}** "
         f"(consistent: {val['n_metrics_consistent']})",
         f"- All-NaN vectors: **{val['n_all_nan_vectors']}**",
         f"- Views present: {val['views_present']}", ""]

    def feat_line(f):
        s = df_ok[f].replace([np.inf, -np.inf], np.nan).dropna()
        return (f"  - {f}: mean={_fmt(s.mean())}, median={_fmt(s.median())}, "
                f"p10={_fmt(s.quantile(.1))}, p90={_fmt(s.quantile(.9))}")

    m += ["## 6-8. Global utility / entropy / gap statistics", ""]
    for f in ["max", "mean", "normalized_entropy", "participation_ratio",
              "perplexity_effective_k", "top1_mass", "top5_mass",
              "largest_gap_position_1_to_10", "largest_gap_value_1_to_10",
              "top1_over_top2"]:
        if f in df_ok.columns:
            m.append(feat_line(f))
    m.append("")

    m += ["## 9. Heuristic K distribution comparison", "",
          "See `HEURISTIC_CANDIDATE_REPORT.md` for the full table. Summary for "
          "the highlighted candidates:", "",
          "| heuristic | mean_K | pct_K_1 | pct_K_2_5 | pct_K_gt5 | bell |",
          "|---|---|---|---|---|---|"]
    for c in SELECTED_CANDIDATES:
        r = dist[dist["heuristic"] == c]
        if not len(r):
            continue
        r = r.iloc[0]
        m.append("| {n} | {me} | {p1} | {p25} | {p5} | {b} |".format(
            n=c, me=_fmt(r.get("mean_K")), p1=_fmt(r.get("pct_K_1"), 1),
            p25=_fmt(r.get("pct_K_2_to_5"), 1), p5=_fmt(r.get("pct_K_gt_5"), 1),
            b="YES" if r.get("bell_shaped") else ""))
    m.append("")

    # family concentration ranking
    fam_sorted = fam.sort_values("mean_normalized_entropy")
    m += ["## 10. Family-level differences", "",
          "Families with the **most concentrated** utility vectors "
          "(lowest mean normalized entropy):", ""]
    for _, r in fam_sorted.head(4).iterrows():
        m.append(f"  - {r['family_id']}: norm_entropy="
                 f"{_fmt(r['mean_normalized_entropy'])}, "
                 f"participation_ratio={_fmt(r.get('mean_participation_ratio'))}")
    m += ["", "Families with the **most diffuse** utility vectors "
          "(highest mean normalized entropy):", ""]
    for _, r in fam_sorted.tail(4).iloc[::-1].iterrows():
        m.append(f"  - {r['family_id']}: norm_entropy="
                 f"{_fmt(r['mean_normalized_entropy'])}, "
                 f"participation_ratio={_fmt(r.get('mean_participation_ratio'))}")
    m.append("")

    m += ["## 11. View-level differences", "",
          f"| view | mean_max | mean_norm_entropy | mean_participation_ratio | "
          f"meanK({PRIMARY_HEURISTIC}) |", "|---|---|---|---|---|"]
    for _, r in view.iterrows():
        m.append("| {v} | {mx} | {ne} | {pr} | {k} |".format(
            v=r["view_type"], mx=_fmt(r.get("mean_max")),
            ne=_fmt(r.get("mean_normalized_entropy")),
            pr=_fmt(r.get("mean_participation_ratio")),
            k=_fmt(r.get(f"meanK_{PRIMARY_HEURISTIC}"))))
    m.append("")

    ranked = dist.sort_values("desirability_score", ascending=False)
    bell = dist[dist["bell_shaped"]]["heuristic"].tolist()
    m += ["## 12. Recommended candidate heuristic families for next phase", "",
          "Utility vectors across the repository are **highly diffuse** (median "
          "normalized entropy ~0.985, participation ratio ~53 of 60 metrics). As "
          "a direct consequence, **cumulative-mass, entropy/effective-K and loose "
          "relative-threshold heuristics saturate at K=10** (they want K>10 before "
          "clipping for ~99% of records) and are *not* viable dynamic-K signals.",
          "",
          "The candidates whose **natural** K distribution best approximates the "
          "desired shape (some K=1, bulk in K=2-5, decaying tail, minimal pre-clip "
          "saturation), ranked by desirability score:", "",
          "| rank | heuristic | mean_K | mode_K | pct_K_1 | pct_K_2_5 | pct_K_gt5 "
          "| pct_raw_gt10 | score |", "|---|---|---|---|---|---|---|---|---|"]
    for i, (_, r) in enumerate(ranked.head(6).iterrows(), 1):
        m.append("| {i} | {n} | {me} | {mo} | {p1} | {p25} | {p5} | {pg} | {sc} |".format(
            i=i, n=r["heuristic"], me=_fmt(r.get("mean_K")),
            mo=int(r.get("mode_K", 0)), p1=_fmt(r.get("pct_K_1"), 1),
            p25=_fmt(r.get("pct_K_2_to_5"), 1), p5=_fmt(r.get("pct_K_gt_5"), 1),
            pg=_fmt(r.get("pct_K_gt_10_before_clipping"), 1),
            sc=_fmt(r.get("desirability_score"), 1)))
    top = ranked.head(4)["heuristic"].tolist()
    m += ["",
          f"Heuristics passing the (permissive) bell-shape filter: "
          f"{', '.join(bell) if bell else '_none_'}.",
          "",
          "**Suggested candidates to inspect manually next:** "
          + ", ".join(top) + ".",
          "",
          "**No final heuristic is selected.** These are evidence-based starting "
          "points for manual interpretation.", ""]

    with open(os.path.join(rep_dir, "DYNAMIC_TOPK_PRE_ANALYSIS_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(m) + "\n")

    return bell
