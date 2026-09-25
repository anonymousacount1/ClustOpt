"""Phase-2 orchestrator: final-heuristic selection analysis for Dynamic Top-K.

Reuses the Phase-1 record tables (no rescanning), adds the new Relative-Threshold
candidate heuristics, scores every heuristic's K distribution, measures stability
and agreement, runs the optional read-only split-1 diagnostic, and writes new
reports / plots into a fresh output dir (Phase-1 outputs are left untouched).

Usage (from this directory):
    python run_phase2_analysis.py
    python run_phase2_analysis.py --phase1-dir <dir>
    python run_phase2_analysis.py --out-dir <existing dir>   # rewrite into it
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import distribution_metrics as dm  # noqa: E402
import heuristic_comparison as hc  # noqa: E402
import heuristic_selection_report as hsr  # noqa: E402
import phase2_plotting as pp  # noqa: E402
from heuristic_refinement import shortlist_columns  # noqa: E402


def default_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=default_repo_root())
    ap.add_argument("--phase1-dir", default=None,
                    help="Phase-1 output dir (default: latest auto-detected)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    repo = args.repo_root
    phase1_dir = hc.find_phase1_dir(repo, args.phase1_dir)
    print(f"[p2] reusing Phase-1 outputs: {phase1_dir}", flush=True)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(repo, "results_analysis",
                               f"dynamic_topk_phase2_selection_{ts}")
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "validation"), exist_ok=True)
    print(f"[p2] output dir: {out_dir}", flush=True)

    # ---- assemble heuristics (reuse + new) --------------------------------
    full, p1cols, newcols = hc.load_and_assemble(phase1_dir)
    heuristics = p1cols + newcols
    print(f"[p2] heuristics: {len(p1cols)} Phase-1 + {len(newcols)} new "
          f"= {len(heuristics)} total over {len(full)} records", flush=True)

    # ---- validation (section 14) ------------------------------------------
    val = hc.validate(full, p1cols, newcols, n_expected=len(full))
    with open(os.path.join(out_dir, "validation", "phase2_validation.json"),
              "w", encoding="utf-8") as fh:
        json.dump(val, fh, indent=2)
    print(f"[p2] validation: {val}", flush=True)

    # ---- distribution metrics + stability ---------------------------------
    metrics = hc.all_metrics(full, heuristics)

    # ---- split-1 diagnostic (optional, read-only) -------------------------
    split1, baselines = hc.split1_diagnostic(full, heuristics, repo)
    has_split1 = split1 is not None
    if has_split1:
        metrics = metrics.merge(split1, on="heuristic", how="left")
        with open(os.path.join(out_dir, "split1_diagnostic_baselines.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(baselines, fh, indent=2)
    print(f"[p2] split-1 diagnostic: {'done' if has_split1 else 'skipped'}",
          flush=True)

    # ---- agreement (shortlist) --------------------------------------------
    shortlist = [c for c in shortlist_columns() if c in full.columns]
    mad, exact = dm.agreement_matrices(full, shortlist)

    # ---- persist tables ---------------------------------------------------
    metrics.to_csv(os.path.join(out_dir, "heuristic_metrics_full.csv"), index=False)
    ranking = metrics.sort_values("composite_score", ascending=False)
    ranking.to_csv(os.path.join(out_dir, "heuristic_ranking.csv"), index=False)
    mad.to_csv(os.path.join(out_dir, "agreement_mean_abs_diff.csv"))
    exact.to_csv(os.path.join(out_dir, "agreement_exact_pct.csv"))
    if has_split1:
        split1.to_csv(os.path.join(out_dir, "split1_diagnostic.csv"), index=False)

    # ---- plots (section 12) -----------------------------------------------
    pp.make_all(os.path.join(out_dir, "plots"), metrics, full, exact, shortlist)

    # ---- reports (section 13) ---------------------------------------------
    hsr.write_distribution_report(out_dir, metrics)
    hsr.write_ranking_report(out_dir, ranking, baselines)
    recommended = hsr.pick_recommendations(ranking, k=3)
    hsr.write_final_selection(out_dir, ranking, baselines, recommended, has_split1)

    print("[p2] recommended:", [r["heuristic"] for r in recommended], flush=True)
    print(f"[p2] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
