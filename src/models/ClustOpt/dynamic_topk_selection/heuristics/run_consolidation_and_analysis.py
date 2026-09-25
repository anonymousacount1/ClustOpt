"""Orchestrator: consolidate the two Dynamic Top-K phases, run the selected
heuristic on every record, and produce the post-analysis.

Reuses Phase-1 extracted utility vectors (no rescan). Writes a permanent,
timestamp-free folder ``results_analysis/dynamic_topk_selection_analysis`` and
leaves the original timestamped phase folders untouched.

Usage (from this directory):
    python run_consolidation_and_analysis.py
    python run_consolidation_and_analysis.py --phase1-dir <d> --phase2-dir <d>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotting  # noqa: E402
import post_analysis as pa  # noqa: E402
import reporting  # noqa: E402
import run_selected_heuristic as rsh  # noqa: E402

P1_DEFAULT = "dynamic_topk_utility_pre_analysis_20260630_201614"
P2_DEFAULT = "dynamic_topk_phase2_selection_20260630_205831"


def default_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def _resolve(repo, given, default_name, pattern):
    if given:
        return given
    p = os.path.join(repo, "results_analysis", default_name)
    if os.path.exists(p):
        return p
    cands = sorted(glob.glob(os.path.join(repo, "results_analysis", pattern)))
    if not cands:
        raise FileNotFoundError(f"no dir matching {pattern}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=default_repo_root())
    ap.add_argument("--phase1-dir", default=None)
    ap.add_argument("--phase2-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    repo = args.repo_root
    p1 = _resolve(repo, args.phase1_dir, P1_DEFAULT,
                  "dynamic_topk_utility_pre_analysis_*")
    p2 = _resolve(repo, args.phase2_dir, P2_DEFAULT,
                  "dynamic_topk_phase2_selection_*")
    out_dir = args.out_dir or os.path.join(
        repo, "results_analysis", "dynamic_topk_selection_analysis")
    print(f"[sel] phase1={p1}\n[sel] phase2={p2}\n[sel] out={out_dir}", flush=True)

    # ---- 1. consolidate -------------------------------------------------
    manifest = reporting.consolidate(p1, p2, out_dir)
    print(f"[sel] consolidated: {len(manifest['copied'])} copied, "
          f"{len(manifest['missing'])} missing", flush=True)

    # expected record count from Phase-1 scan report
    with open(os.path.join(p1, "validation", "scan_report.json"),
              encoding="utf-8") as fh:
        n_expected = json.load(fh)["n_ok"]

    # ---- 2. run selected heuristic --------------------------------------
    assign = rsh.run(p1)
    rsh.save(assign, os.path.join(out_dir, "tables"))
    print(f"[sel] heuristic applied to {len(assign)} records", flush=True)

    # ---- 3. post-analysis -----------------------------------------------
    gstats = pa.k_stats(assign["selected_k"])
    gdist = pa.global_distribution(assign)
    fam = pa.grouped_summary(assign, "family")
    sub = pa.grouped_summary(assign, "subfamily")
    view = pa.grouped_summary(assign, "view_type")
    reporting.write_tables(out_dir, gdist, fam, sub, view)

    # ---- 4. validation --------------------------------------------------
    validation = pa.validate(assign, n_expected=n_expected)
    with open(os.path.join(out_dir, "validation",
                           "selected_heuristic_validation.json"),
              "w", encoding="utf-8") as fh:
        json.dump(validation, fh, indent=2)
    print(f"[sel] validation all_checks_pass={validation['all_checks_pass']}",
          flush=True)

    # ---- 5. plots -------------------------------------------------------
    plotting.make_all(os.path.join(out_dir, "plots"), assign)

    # ---- 6. reports -----------------------------------------------------
    coverage = {
        "n_records": int(len(assign)),
        "n_datasets": int(assign["dataset_id"].nunique()),
        "n_families": int(assign["family"].nunique()),
        "n_subfamilies": int(assign["subfamily"].nunique()),
        "n_views": int(assign["view_type"].nunique()),
        "views": sorted(assign["view_type"].unique().tolist()),
    }
    overall = reporting.write_post_analysis_report(
        out_dir, coverage, gstats, gdist, fam, sub, view, validation)
    reporting.write_consolidated_report(out_dir, p1, p2, manifest, gstats, overall)
    reporting.write_readme(out_dir)

    print(f"[sel] global: meanK={gstats['mean_K']:.3f} P(2-5)={gstats['P_2_5']:.3f} "
          f"P(K=1)={gstats['P_K1']:.3f} P(K=10)={gstats['P_eq10']:.3f}", flush=True)
    print(f"[sel] desired-behaviour overall={'YES' if overall else 'NO'}", flush=True)
    print(f"[sel] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
