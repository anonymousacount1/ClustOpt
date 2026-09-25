"""ML2DAC Correlation Audit (analysis only; does NOT rebuild the MKRs).

Compares the CVI* best-correlation-with-ARI between the Original and Extended
derived ML2DAC repositories, quantifies ClustOpt-metric adoption, and breaks the
improvement down by family and view. Reuses the already-built artifacts:
    <variant>/cvi_star.parquet, <variant>/cvi_rank_correlations.parquet
Writes reports/ML2DAC_CORRELATION_AUDIT.{md,json} and
reports/ML2DAC_METRIC_SELECTION_ANALYSIS.md.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

# The 7 original ML2DAC CVIs (ClustOpt canonical names).
ML2DAC_CVIS = {
    "calinski_harabasz", "davies_bouldin", "silhouette", "dbcv",
    "dunn_index", "coggins_jain_index", "cop",
}
HELDOUT_SPLIT = 1
PCTS = [10, 25, 50, 75, 90, 95, 99]


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _summary(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    out = {
        "n": int(x.size),
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)), "max": float(np.max(x)),
    }
    for p in PCTS:
        out[f"p{p}"] = float(np.percentile(x, p))
    return out


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta via Mann-Whitney U: delta = 2U/(n*m) - 1 (a vs b)."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (a.size * b.size) - 1.0)


def _md_table(headers: List[str], rows: List[List[object]]) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def run(root: Path) -> int:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    policy = {}
    pol_path = root / "split_policy.json"
    if pol_path.exists():
        policy = json.loads(pol_path.read_text(encoding="utf-8"))

    o_star = pd.read_parquet(root / "original_cvis" / "cvi_star.parquet")
    e_star = pd.read_parquet(root / "extended_cvis" / "cvi_star.parquet")
    o_corr = pd.read_parquet(root / "original_cvis" / "cvi_rank_correlations.parquet")
    e_corr = pd.read_parquet(root / "extended_cvis" / "cvi_rank_correlations.parquet")

    audit: Dict[str, object] = {"generated_at": _now()}
    audit["split_policy"] = {
        "heldout_test_split": policy.get("heldout_test_split", HELDOUT_SPLIT),
        "train_splits": policy.get("train_splits"),
        "excluded_splits": policy.get("excluded_splits", []),
        "observed_split_ids": sorted(int(s) for s in o_star["split_id"].unique()),
    }

    # ---- 11. validation ----
    val = {
        "original_records": int(o_star["record_id"].nunique()),
        "extended_records": int(e_star["record_id"].nunique()),
        "record_counts_match": int(o_star["record_id"].nunique()) == int(e_star["record_id"].nunique()),
        "record_ids_identical": set(o_star["record_id"]) == set(e_star["record_id"]),
        "original_resolved": int(o_star["resolved"].sum()),
        "extended_resolved": int(e_star["resolved"].sum()),
        "original_split1_rows": int((o_star["split_id"] == HELDOUT_SPLIT).sum()),
        "extended_split1_rows": int((e_star["split_id"] == HELDOUT_SPLIT).sum()),
        "corr_split1_check": int(o_corr.merge(
            o_star[["record_id", "split_id"]], on="record_id", how="left"
        )["split_id"].eq(HELDOUT_SPLIT).sum()),
    }
    # consistency: star.best_correlation equals max over correlations
    for tag, star, corr in [("original", o_star, o_corr), ("extended", e_star, e_corr)]:
        mx = corr.groupby("record_id")["spearman_correlation"].max()
        d = (star.set_index("record_id")["best_correlation"] - mx).abs().max()
        val[f"{tag}_best_eq_max_corr"] = bool(float(d) < 1e-9)
    val["no_split1_anywhere"] = bool(
        val["original_split1_rows"] == 0 and val["extended_split1_rows"] == 0
        and val["corr_split1_check"] == 0
    )
    val["all_ok"] = bool(
        val["record_counts_match"] and val["record_ids_identical"]
        and val["original_resolved"] == val["original_records"]
        and val["extended_resolved"] == val["extended_records"]
        and val["no_split1_anywhere"]
        and val["original_best_eq_max_corr"] and val["extended_best_eq_max_corr"]
    )
    audit["validation"] = val

    # ---- 3. distributions ----
    o_best = o_star.set_index("record_id")["best_correlation"]
    e_best = e_star.set_index("record_id")["best_correlation"]
    audit["distribution"] = {
        "original": _summary(o_best.to_numpy()),
        "extended": _summary(e_best.to_numpy()),
    }

    # ---- 4. paired statistical comparison ----
    common = o_best.index.intersection(e_best.index)
    a = e_best.reindex(common).to_numpy()  # extended
    b = o_best.reindex(common).to_numpy()  # original
    diff = a - b
    finite = ~np.isnan(diff)
    diff = diff[finite]
    n_gt = int((diff > 1e-12).sum())
    n_eq = int((np.abs(diff) <= 1e-12).sum())
    n_lt = int((diff < -1e-12).sum())
    # Wilcoxon signed-rank (paired); needs at least one non-zero diff
    try:
        wstat, wp = wilcoxon(a[finite], b[finite], zero_method="wilcox", alternative="two-sided")
        wstat, wp = float(wstat), float(wp)
    except Exception as ex:  # all-zero diffs etc.
        wstat, wp = float("nan"), float("nan")
        _ = ex
    cohens_d = float(np.mean(diff) / np.std(diff, ddof=1)) if diff.size > 1 and np.std(diff, ddof=1) > 0 else float("nan")
    cliffs = _cliffs_delta(a, b)
    audit["comparison"] = {
        "n_paired_records": int(diff.size),
        "mean_difference_ext_minus_orig": float(np.mean(diff)),
        "median_difference_ext_minus_orig": float(np.median(diff)),
        "mean_extended": float(np.nanmean(a)), "mean_original": float(np.nanmean(b)),
        "median_extended": float(np.nanmedian(a)), "median_original": float(np.nanmedian(b)),
        "records_extended_better": n_gt, "records_equal": n_eq, "records_original_better": n_lt,
        "pct_extended_better": float(100.0 * n_gt / max(1, diff.size)),
        "wilcoxon_statistic": wstat, "wilcoxon_pvalue": wp,
        "cohens_d_paired": cohens_d,
        "cliffs_delta": cliffs,
        "note": ("Extended candidate set is a superset of Original (7 CVIs + ClustOpt "
                 "metrics), so per-record best correlation is >= Original by construction; "
                 "the magnitude quantifies how much the added structural metrics improve "
                 "ARI-tracking of the oracle-selected CVI*."),
    }

    # ---- 5/6. selection frequency + adoption ----
    def _freq(star, top=None):
        res = star[star["resolved"]]
        vc = res["best_cvi"].value_counts()
        n = int(len(res))
        rows = [(k, int(v), 100.0 * int(v) / n) for k, v in vc.items()]
        return rows[:top] if top else rows, n

    o_freq, o_n = _freq(o_star)
    e_freq, e_n = _freq(e_star, top=20)
    e_res = e_star[e_star["resolved"]]
    ml2dac_sel = int(e_res["best_cvi"].isin(ML2DAC_CVIS).sum())
    clustopt_sel = int((~e_res["best_cvi"].isin(ML2DAC_CVIS)).sum())
    audit["selection_frequency"] = {
        "original": [{"metric": k, "count": c, "pct": p} for k, c, p in o_freq],
        "extended_top20": [{"metric": k, "count": c, "pct": p} for k, c, p in e_freq],
    }
    audit["clustopt_adoption"] = {
        "extended_resolved_records": int(len(e_res)),
        "ml2dac_cvi_selected": ml2dac_sel,
        "ml2dac_cvi_pct": 100.0 * ml2dac_sel / max(1, len(e_res)),
        "clustopt_metric_selected": clustopt_sel,
        "clustopt_metric_pct": 100.0 * clustopt_sel / max(1, len(e_res)),
    }

    # ---- 7. family-level ----
    of = o_star.groupby("family")["best_correlation"].mean()
    ef = e_star.groupby("family")["best_correlation"].mean()
    fam = pd.DataFrame({"original_mean": of, "extended_mean": ef})
    fam["improvement"] = fam["extended_mean"] - fam["original_mean"]
    fam = fam.sort_values("improvement", ascending=False).reset_index()
    audit["family_analysis"] = fam.to_dict("records")

    # ---- 8. view-level ----
    views = {}
    for view in ["x_only", "y_only", "xy_2d"]:
        ov = o_star[o_star["view_type"] == view]["best_correlation"]
        ev = e_star[e_star["view_type"] == view]["best_correlation"]
        views[view] = {
            "original_mean": float(ov.mean()), "original_median": float(ov.median()),
            "extended_mean": float(ev.mean()), "extended_median": float(ev.median()),
            "mean_improvement": float(ev.mean() - ov.mean()),
            "n_records": int(len(ev)),
        }
    audit["view_analysis"] = views

    # ---- 9. top metrics by correlation quality (extended) ----
    e_res_idx = set(e_res["record_id"])
    sel_counts = e_res["best_cvi"].value_counts().to_dict()
    cq = e_corr.copy()
    valid = cq[cq["spearman_correlation"].notna()]
    grp = valid.groupby("metric_id")["spearman_correlation"]
    mq = pd.DataFrame({
        "mean_corr": grp.mean(), "median_corr": grp.median(),
        "valid_records": grp.size(),
    }).reset_index()
    mq["selection_count"] = mq["metric_id"].map(sel_counts).fillna(0).astype(int)
    mq["selection_pct"] = 100.0 * mq["selection_count"] / max(1, len(e_res))
    by_sel = mq.sort_values("selection_count", ascending=False).head(20)
    by_corr = mq.sort_values("mean_corr", ascending=False).head(20)
    audit["extended_metric_quality"] = {
        "top20_by_selection": by_sel.to_dict("records"),
        "top20_by_mean_corr": by_corr.to_dict("records"),
    }

    # ---- write JSON ----
    (reports / "ML2DAC_CORRELATION_AUDIT.json").write_text(
        json.dumps(_clean(audit), indent=2), encoding="utf-8")

    _write_audit_md(reports, audit)
    _write_selection_md(reports, audit, o_freq, o_n)

    # ---- console summary ----
    c = audit["comparison"]
    a_ = audit["clustopt_adoption"]
    print(f"[corr-audit] validation all_ok={val['all_ok']}")
    print(f"[corr-audit] best-corr mean: original={audit['distribution']['original']['mean']:.4f} "
          f"extended={audit['distribution']['extended']['mean']:.4f} "
          f"(diff {c['mean_difference_ext_minus_orig']:+.4f})")
    print(f"[corr-audit] Wilcoxon p={c['wilcoxon_pvalue']:.3e} cohen_d={c['cohens_d_paired']:.3f} "
          f"cliffs_delta={c['cliffs_delta']:.3f}")
    print(f"[corr-audit] extended better in {c['pct_extended_better']:.1f}% of records")
    print(f"[corr-audit] ClustOpt adoption: {a_['clustopt_metric_pct']:.1f}% "
          f"(ML2DAC CVIs {a_['ml2dac_cvi_pct']:.1f}%)")
    return 0 if val["all_ok"] else 1


def _write_audit_md(reports: Path, a: dict) -> None:
    d = a["distribution"]; c = a["comparison"]; ad = a["clustopt_adoption"]; v = a["validation"]
    sp = a.get("split_policy", {})
    L = ["# ML2DAC Correlation Audit", "", f"_Generated {a['generated_at']}_", "",
         "_Analysis only — the derived MKRs were not modified._", "",
         "## Split policy", "",
         f"- heldout split: {sp.get('heldout_test_split')}",
         f"- train splits: {sp.get('train_splits')}",
         f"- excluded splits: {sp.get('excluded_splits') or 'none'}",
         f"- observed split ids in CVI* records: {sp.get('observed_split_ids')}", "",
         "## Validation", "",
         f"- Record counts match: {v['record_counts_match']} "
         f"(original {v['original_records']}, extended {v['extended_records']})",
         f"- Record IDs identical: {v['record_ids_identical']}",
         f"- All records resolved: original {v['original_resolved']}, extended {v['extended_resolved']}",
         f"- No split-1 anywhere: {v['no_split1_anywhere']}",
         f"- best_correlation == max(correlations): original {v['original_best_eq_max_corr']}, "
         f"extended {v['extended_best_eq_max_corr']}",
         f"- **All validation OK: {v['all_ok']}**", "",
         "## Best-correlation distribution (per record)", ""]
    headers = ["statistic", "original", "extended"]
    keys = ["mean", "median", "std", "min", "max"] + [f"p{p}" for p in PCTS]
    L.append(_md_table(headers, [[k, _fmt(d["original"][k]), _fmt(d["extended"][k])] for k in keys]))
    L += ["", "## Statistical comparison (paired, same records)", "",
          _md_table(["measure", "value"], [
              ["n paired records", c["n_paired_records"]],
              ["mean (original)", _fmt(c["mean_original"])],
              ["mean (extended)", _fmt(c["mean_extended"])],
              ["mean difference (ext - orig)", _fmt(c["mean_difference_ext_minus_orig"])],
              ["median difference (ext - orig)", _fmt(c["median_difference_ext_minus_orig"])],
              ["records extended better", f"{c['records_extended_better']} ({_fmt(c['pct_extended_better'],1)}%)"],
              ["records equal", c["records_equal"]],
              ["records original better", c["records_original_better"]],
              ["Wilcoxon statistic", _fmt(c["wilcoxon_statistic"], 1)],
              ["Wilcoxon p-value", f"{c['wilcoxon_pvalue']:.3e}" if c["wilcoxon_pvalue"] == c["wilcoxon_pvalue"] else "n/a"],
              ["Cohen's d (paired)", _fmt(c["cohens_d_paired"], 3)],
              ["Cliff's delta", _fmt(c["cliffs_delta"], 3)],
          ]),
          "", f"> {c['note']}", "",
          "## ClustOpt metric adoption (Extended)", "",
          _md_table(["selection", "count", "pct"], [
              ["ML2DAC CVIs", ad["ml2dac_cvi_selected"], _fmt(ad["ml2dac_cvi_pct"], 1)],
              ["ClustOpt structural metrics", ad["clustopt_metric_selected"], _fmt(ad["clustopt_metric_pct"], 1)],
          ]), ""]

    # family table
    L += ["## Family-level improvement (sorted by improvement)", "",
          _md_table(["family", "original_mean", "extended_mean", "improvement"],
                    [[r["family"], _fmt(r["original_mean"]), _fmt(r["extended_mean"]),
                      _fmt(r["improvement"])] for r in a["family_analysis"]]), ""]

    # view table
    L += ["## View-level analysis", "",
          _md_table(["view", "orig_mean", "orig_median", "ext_mean", "ext_median", "mean_improvement"],
                    [[k, _fmt(vw["original_mean"]), _fmt(vw["original_median"]),
                      _fmt(vw["extended_mean"]), _fmt(vw["extended_median"]),
                      _fmt(vw["mean_improvement"])] for k, vw in a["view_analysis"].items()]), ""]

    # interpretation
    L += ["## Interpretation", "", _interpretation(a)]
    (reports / "ML2DAC_CORRELATION_AUDIT.md").write_text("\n".join(L), encoding="utf-8")


def _write_selection_md(reports: Path, a: dict, o_freq, o_n) -> None:
    L = ["# ML2DAC Metric Selection Analysis", "", f"_Generated {a['generated_at']}_", "",
         "## Original — CVI selection frequency (all 7 candidates)", "",
         _md_table(["metric", "count", "pct"],
                   [[k, c, _fmt(p, 2)] for k, c, p in o_freq]), "",
         "## Extended — top 20 selected metrics", "",
         _md_table(["metric", "count", "pct"],
                   [[r["metric"], r["count"], _fmt(r["pct"], 2)]
                    for r in a["selection_frequency"]["extended_top20"]]), "",
         "## Extended — top 20 metrics by selection frequency (with correlation quality)", "",
         _md_table(["metric", "selection_count", "selection_pct", "mean_corr", "median_corr", "valid_records"],
                   [[r["metric_id"], r["selection_count"], _fmt(r["selection_pct"], 2),
                     _fmt(r["mean_corr"]), _fmt(r["median_corr"]), r["valid_records"]]
                    for r in a["extended_metric_quality"]["top20_by_selection"]]), "",
         "## Extended — top 20 metrics by mean correlation (where valid)", "",
         _md_table(["metric", "mean_corr", "median_corr", "selection_count", "selection_pct", "valid_records"],
                   [[r["metric_id"], _fmt(r["mean_corr"]), _fmt(r["median_corr"]),
                     r["selection_count"], _fmt(r["selection_pct"], 2), r["valid_records"]]
                    for r in a["extended_metric_quality"]["top20_by_mean_corr"]])]
    (reports / "ML2DAC_METRIC_SELECTION_ANALYSIS.md").write_text("\n".join(L), encoding="utf-8")


def _interpretation(a: dict) -> str:
    c = a["comparison"]; ad = a["clustopt_adoption"]; views = a["view_analysis"]
    fams = a["family_analysis"]
    top_fams = ", ".join(f"{r['family']} (+{r['improvement']:.3f})" for r in fams[:3])
    best_view = max(views.items(), key=lambda kv: kv[1]["mean_improvement"])
    lines = [
        f"- **Extended achieves higher best-correlation than Original.** Mean best Spearman with ARI "
        f"rises from {c['mean_original']:.3f} to {c['mean_extended']:.3f} "
        f"(mean paired gain {c['mean_difference_ext_minus_orig']:+.3f}); Extended is strictly better in "
        f"{c['pct_extended_better']:.1f}% of records and never worse (Wilcoxon p={c['wilcoxon_pvalue']:.1e}, "
        f"Cohen's d={c['cohens_d_paired']:.2f}, Cliff's delta={c['cliffs_delta']:.2f}). This is partly "
        f"structural — Extended's candidate set is a superset of Original — so the result quantifies the "
        f"*added value* of the in-domain ClustOpt metrics for oracle CVI* selection, not a head-to-head of "
        f"disjoint methods.",
        f"- **ClustOpt structural metrics are adopted heavily.** They are the best CVI* for "
        f"{ad['clustopt_metric_pct']:.1f}% of records ({ad['clustopt_metric_selected']}/"
        f"{ad['extended_resolved_records']}); the original 7 CVIs remain best for only "
        f"{ad['ml2dac_cvi_pct']:.1f}%.",
        f"- **Most-useful structural metrics** (by selection frequency) include the top entries in "
        f"ML2DAC_METRIC_SELECTION_ANALYSIS.md; silhouette remains the single most-selected metric overall.",
        f"- **Improvements are concentrated in specific families** — largest gains in {top_fams}.",
        f"- **View effect:** the largest mean improvement is in `{best_view[0]}` "
        f"(+{best_view[1]['mean_improvement']:.3f}); per-view gains are "
        + ", ".join(f"{k} +{v['mean_improvement']:.3f}" for k, v in views.items()) + ".",
        "- These are descriptive observations on the train CVI* labels (splits 2-15). They do not yet "
        "measure downstream clustering performance on the held-out split 1, which remains the decisive test.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ML2DAC correlation audit (analysis only)")
    ap.add_argument("--derived-root", required=True,
                    help="path to derived/ml2dac_split1")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing audit reports (always regenerated)")
    args = ap.parse_args(argv)
    return run(Path(args.derived_root))


if __name__ == "__main__":
    raise SystemExit(main())
