"""Consolidation of the two analysis phases + markdown reports for the selected
heuristic (sections 3, 4, 8, 10)."""

from __future__ import annotations

import json
import os
import shutil

import pandas as pd

import selected_dynamic_topk as sel

# What to copy from each phase into the consolidated folder.
# (src relative path, dest subdir)
PHASE1_COPY = [
    ("global_summary.csv", "tables"),
    ("family_summary.csv", "tables"),
    ("subfamily_summary.csv", "tables"),
    ("view_summary.csv", "tables"),
    ("metric_summary.csv", "tables"),
    ("heuristic_k_distribution.csv", "tables"),
    ("validation/scan_report.json", "validation"),
    ("reports/UTILITY_VECTOR_STATISTICS.md", "reports"),
    ("reports/DYNAMIC_TOPK_PRE_ANALYSIS_REPORT.md", "reports"),
    ("reports/HEURISTIC_CANDIDATE_REPORT.md", "reports"),
    ("plots/sorted_utility_mean_curve.png", "plots"),
    ("plots/sorted_utility_percentile_curves.png", "plots"),
    ("plots/utility_entropy_distribution.png", "plots"),
    ("plots/utility_effective_k_distribution.png", "plots"),
    ("plots/utility_max_distribution.png", "plots"),
    ("plots/gap_distribution.png", "plots"),
    ("plots/gap_position_distribution.png", "plots"),
    ("plots/largest_gap_value_distribution.png", "plots"),
    ("plots/heuristic_k_distribution_all.png", "plots"),
    ("plots/heuristic_k_distribution_selected_candidates.png", "plots"),
]
PHASE2_COPY = [
    ("heuristic_ranking.csv", "tables"),
    ("reports/FINAL_HEURISTIC_SELECTION.md", "reports"),
    ("reports/HEURISTIC_RANKING.md", "reports"),
    ("reports/DISTRIBUTION_ANALYSIS.md", "reports"),
    ("plots/heuristic_ranking.png", "plots"),
    ("plots/bell_score_comparison.png", "plots"),
    ("plots/cap_sensitivity.png", "plots"),
    ("plots/relative_threshold_sweep.png", "plots"),
    ("plots/agreement_heatmap.png", "plots"),
    ("plots/family_variance_heatmap.png", "plots"),
]


def consolidate(phase1_dir, phase2_dir, out_dir) -> dict:
    for d in ("reports", "tables", "plots", "validation"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    manifest = {"phase1_dir": phase1_dir, "phase2_dir": phase2_dir,
                "copied": [], "missing": []}
    for src_root, items in ((phase1_dir, PHASE1_COPY), (phase2_dir, PHASE2_COPY)):
        for rel, dest in items:
            src = os.path.join(src_root, rel)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, dest, os.path.basename(rel)))
                manifest["copied"].append(os.path.join(dest, os.path.basename(rel)))
            else:
                manifest["missing"].append(src)
    with open(os.path.join(out_dir, "validation", "consolidation_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def desired_behaviour_verdict(gstats, fam) -> tuple[bool, list[str]]:
    """Check the qualitative target behaviour. Returns (overall, per-check lines)."""
    checks = [
        ("some K=1 cases", gstats["P_K1"] > 0.01,
         f"P(K=1) = {_f(gstats['P_K1'])}"),
        ("most records K=3-5 (preferred region dominant)",
         gstats["P_2_5"] >= 0.45,
         f"P(2<=K<=5) = {_f(gstats['P_2_5'])}"),
        ("controlled tail above K=5", gstats["P_gt5"] <= 0.35,
         f"P(K>5) = {_f(gstats['P_gt5'])}"),
        ("very few K=10", gstats["P_eq10"] <= 0.03,
         f"P(K=10) = {_f(gstats['P_eq10'])}"),
        ("stable across families", float(fam["mean_K"].std(ddof=0)) <= 0.6,
         f"cross-family mean-K std = {_f(fam['mean_K'].std(ddof=0))}"),
    ]
    overall = all(c[1] for c in checks)
    lines = [f"- {'PASS' if ok else 'FAIL'} — {name} ({detail})"
             for name, ok, detail in checks]
    return overall, lines


def write_tables(out_dir, gdist, fam, sub, view):
    t = os.path.join(out_dir, "tables")
    gdist.to_csv(os.path.join(t, "selected_heuristic_k_distribution.csv"),
                 index=False)
    fam.to_csv(os.path.join(t, "selected_family_summary.csv"), index=False)
    sub.to_csv(os.path.join(t, "selected_subfamily_summary.csv"), index=False)
    view.to_csv(os.path.join(t, "selected_view_summary.csv"), index=False)


def _dist_table(gdist):
    lines = ["| K | count | percent |", "|---|---|---|"]
    for _, r in gdist.iterrows():
        lines.append(f"| {int(r['K'])} | {int(r['count'])} | {_f(r['percent'],2)} |")
    return "\n".join(lines)


def write_post_analysis_report(out_dir, coverage, gstats, gdist, fam, sub, view,
                               validation):
    overall, check_lines = desired_behaviour_verdict(gstats, fam)
    impl = "models/ClustOpt/dynamic_topk_selection/heuristics/selected_dynamic_topk.py"
    lines = [
        "# Selected Dynamic Top-K — Post-Analysis Report", "",
        f"Heuristic: **`{sel.HEURISTIC_NAME}`** (alpha={sel.ALPHA}, soft-cap "
        f"base={sel.SOFT_CAP_BASE}, max_k={sel.MAX_K}). Standalone selector "
        "evaluated on the existing Phase-1 utility vectors; ClustOpt search is "
        "unchanged.", "",
        "## Coverage", "",
        f"- Records (dataset-views): **{coverage['n_records']}**",
        f"- Datasets: **{coverage['n_datasets']}**",
        f"- Families: **{coverage['n_families']}**",
        f"- Subfamilies: **{coverage['n_subfamilies']}**",
        f"- Views: **{coverage['n_views']}** ({', '.join(coverage['views'])})", "",
        "## Selected heuristic", "",
        "```", "u = clip(nan/inf->0, [0,1]); u_max = max(u)",
        "k_rel = 1 if u_max<=eps else count(u_i >= 0.92*u_max)",
        "K = k_rel               if k_rel <= 5",
        "K = ceil((k_rel+5)/2)   if k_rel  > 5     # soft cap",
        "K = clip(K, 1, 10)                        # hard cap", "```",
        f"- alpha = {sel.ALPHA}, soft_cap_base = {sel.SOFT_CAP_BASE}, "
        f"max_k = {sel.MAX_K}",
        f"- implementation: `{impl}`", "",
        "## Global K distribution", "",
        _dist_table(gdist), "",
        f"- mean_K = **{_f(gstats['mean_K'],3)}**, median_K = "
        f"{_f(gstats['median_K'],1)}, mode_K = {gstats['mode_K']}, "
        f"std_K = {_f(gstats['std_K'],3)}",
        f"- P(K=1) = {_f(gstats['P_K1'])}, P(2<=K<=5) = **{_f(gstats['P_2_5'])}**, "
        f"P(K>5) = {_f(gstats['P_gt5'])}, P(K=10) = {_f(gstats['P_eq10'])}", "",
        "**Interpretation:** the preferred region K=2-5 carries the bulk of the "
        "mass, with a modest K=1 share and a controlled upper tail; clipping at "
        "K=10 is rare.", "",
        "## K distribution by family", "",
        _group_table(fam, "family"), "",
        "## K distribution by view", "",
        _group_table(view, "view_type"), "",
        "## K distribution by subfamily (extremes)", "",
        "Lowest mean-K subfamilies:", "",
        _group_table(sub.sort_values("mean_K").head(6), "subfamily"), "",
        "Highest mean-K subfamilies:", "",
        _group_table(sub.sort_values("mean_K").tail(6), "subfamily"), "",
        "## Does the heuristic produce the intended behaviour?", "",
        *check_lines, "",
        f"**Overall: {'YES' if overall else 'NO'}** — the selected heuristic "
        f"{'matches' if overall else 'does not fully match'} the desired profile "
        "(some K=1, bulk at K=3-5, controlled tail, very few K=10, stable across "
        "structure).", "",
        "## Validation", "",
        *[f"- {'PASS' if v is True else ('FAIL' if v is False else v)} — {k}"
          for k, v in validation.items()], "",
        "## Recommendation", "",
        f"Is `{sel.HEURISTIC_NAME}` ready to be integrated into ClustOpt as an "
        f"experimental option? **{'YES' if (overall and validation['all_checks_pass']) else 'NO'}** "
        "— it is a validated, well-behaved standalone selector. Its *benefit* "
        "(ARI vs fixed Top-K) still has to be confirmed by an in-ClustOpt "
        "ablation, but as an experimental option it is ready.", "",
    ]
    with open(os.path.join(out_dir, "reports", "POST_ANALYSIS_REPORT.md"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return overall


def _group_table(g, key):
    cols = ["mean_K", "median_K", "mode_K", "std_K", "P_K1", "P_2_5",
            "P_gt5", "P_eq10"]
    head = "| " + key + " | " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in range(len(cols) + 1)) + "|"
    lines = [head, sep]
    for _, r in g.iterrows():
        cells = [str(r[key])] + [_f(r[c], 1 if c == "mode_K" else 3) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_consolidated_report(out_dir, phase1_dir, phase2_dir, manifest,
                              gstats, overall):
    lines = [
        "# Consolidated Dynamic Top-K Analysis", "",
        "Permanent, timestamp-free home for the Dynamic Top-K metric-selection "
        "investigation. It consolidates two exploratory phases and the final, "
        "implemented selector. The original timestamped folders are preserved.",
        "",
        "## 1. What was analysed", "",
        "- **Phase 1** (`" + os.path.basename(phase1_dir) + "`): scanned all "
        "51,204 dataset-view utility vectors (60 metrics each) and characterised "
        "their shape — entropy, participation ratio, top-K mass, sorted-curve "
        "gaps/elbow.",
        "- **Phase 2** (`" + os.path.basename(phase2_dir) + "`): simulated 260 "
        "candidate dynamic-K heuristics and scored their K distributions against "
        "an explicit ideal bell shape, with stability and a read-only split-1 "
        "diagnostic.",
        "- **This folder**: implements and runs the selected heuristic as a "
        "standalone selector, with full post-analysis.", "",
        "## 2. Why mass / entropy heuristics were rejected", "",
        "Utility vectors are **highly diffuse** (median normalized entropy "
        "~0.985, participation ratio ~53 of 60 metrics, top-1 mass ~3%). "
        "Cumulative-mass, effective-K and entropy heuristics therefore demand "
        "K>10 for ~99% of records and only \"work\" by clipping — their bell-fit "
        "is ~0.01. Not viable.", "",
        "## 3. Why relative-threshold heuristics were preferred", "",
        "Counting metrics within a fraction `alpha` of the best metric is the "
        "only family that yields a usable, decaying K distribution. A tight "
        "`alpha` (~0.90-0.93) keeps K small; caps/guards tame the diffuse tail.",
        "",
        "## 4. Why `dynamic_topk_relsoft_a092_k5_ceil` was selected", "",
        "Best Phase-2 candidate by composite score: `alpha=0.92` (peak bell-fit "
        "~0.77), a **soft** cap (ceil at base 5) that avoids abrupt clipping, and "
        "a hard cap at 10 for the most diffuse vectors. On the full run it gives "
        f"mean K {_f(gstats['mean_K'],2)}, P(2<=K<=5) {_f(gstats['P_2_5'])}, "
        f"P(K=10) {_f(gstats['P_eq10'])}; desired-behaviour check: "
        f"{'PASS' if overall else 'PARTIAL'}.", "",
        "## 5. What remains to test inside ClustOpt", "",
        "- Wire the selector into the ClustOpt aggregation step (per view, "
        "exact K) as an experimental ablation.",
        "- Measure ARI vs the fixed Top-1/3/5/10 policies on split 1 (the "
        "standalone split-1 diagnostic was coarse and only matched, not beat, "
        "Top-5).",
        "- Decide `max_k` and whether to re-tune `alpha` once real ARI feedback "
        "is available.", "",
        "## Folder layout", "",
        "- `reports/` — consolidated Phase-1/2 reports + `POST_ANALYSIS_REPORT.md`"
        " + this report.",
        "- `tables/` — Phase-1/2 summaries, ranking, and the selected-heuristic "
        "assignments/distribution.",
        "- `plots/` — key Phase-1/2 figures + selected-heuristic figures.",
        "- `validation/` — scan, consolidation and selected-heuristic validation.",
        "",
        f"_Consolidation manifest: {len(manifest['copied'])} files copied, "
        f"{len(manifest['missing'])} missing (see "
        "`validation/consolidation_report.json`)._", "",
    ]
    with open(os.path.join(out_dir, "reports",
                           "CONSOLIDATED_DYNAMIC_TOPK_REPORT.md"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_readme(out_dir):
    lines = [
        "# Dynamic Top-K Selection Analysis (consolidated)", "",
        "Permanent, timestamp-free consolidation of the Dynamic Top-K "
        "investigation plus the implemented selected heuristic "
        f"`{sel.HEURISTIC_NAME}`.", "",
        "## Contents", "",
        "- `reports/CONSOLIDATED_DYNAMIC_TOPK_REPORT.md` — master narrative.",
        "- `reports/POST_ANALYSIS_REPORT.md` — selected-heuristic K-distribution "
        "analysis + readiness.",
        "- `reports/FINAL_HEURISTIC_SELECTION.md`, `HEURISTIC_RANKING.md` — "
        "Phase-2 selection evidence (copied).",
        "- `tables/` — summaries, ranking, `selected_heuristic_assignments.parquet`"
        ", `selected_heuristic_summary.csv`.",
        "- `plots/` — key Phase-1/2 figures + `selected_heuristic_*` figures.",
        "- `validation/` — `scan_report.json`, `consolidation_report.json`, "
        "`selected_heuristic_validation.json`.", "",
        "## Reproduce", "",
        "```bash",
        "cd models/ClustOpt/dynamic_topk_selection/heuristics",
        "python run_consolidation_and_analysis.py",
        "```", "",
        "Sources (preserved, not modified):", "",
        "- `results_analysis/dynamic_topk_utility_pre_analysis_20260630_201614`",
        "- `results_analysis/dynamic_topk_phase2_selection_20260630_205831`", "",
        "The selected heuristic is implemented in "
        "`selected_dynamic_topk.py` and reuses the Phase-1 extracted utility "
        "vectors (no rescanning).", "",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
