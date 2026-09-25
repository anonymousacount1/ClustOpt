"""Phase-D orchestrator: run the 28-method ablation across all subfamilies.

A *thin* wrapper over the existing per-subfamily runner (:func:`run_subfamily_
experiments.run`). It discovers every subfamily under the repository, runs them
**sequentially** (each with its own 8-worker dataset pool), streams a per-
subfamily report as soon as each finishes, and never modifies the underlying
pipeline. Resume is automatic (the per-run ``status.json`` skip logic).

Run from the repo root::

    python -m models.Clustering_Repository_Builder.experiments.experiment_execution.run_phaseD_all_subfamilies \
        --repo-root . --split-id 1 --max-workers 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

from .config import ExperimentExecutionConfig, METHODS_PHASED
from .run_subfamily_experiments import run as run_subfamily
from .summary_writer import summary_dir

configure_process()

PHASED_SUFFIX = "phaseD_no_earlystop_50trials_dynamic_ablation"
PHASED_METHODS = tuple(m.name for m in METHODS_PHASED)


def discover_subfamilies(analyzed_data: Path) -> List[Path]:
    """All ``analyzed_data/<family>/subfamilies/<subfamily>`` dirs, sorted."""
    out: List[Path] = []
    for fam in sorted(analyzed_data.iterdir()):
        sub_root = fam / "subfamilies"
        if not sub_root.is_dir():
            continue
        for sub in sorted(sub_root.iterdir()):
            if sub.is_dir():
                out.append(sub)
    return out


def _subfamily_report(cfg: ExperimentExecutionConfig) -> str:
    """One-line-per-method rollup from the just-written execution summary."""
    sd = summary_dir(cfg)
    csv = sd / "execution_summary.csv"
    if not csv.is_file():
        return "  (no execution_summary.csv written)"
    df = pd.read_csv(csv)
    if df.empty:
        return "  (empty summary)"
    lines = []
    n_ds = df["dataset_id"].nunique()
    counts = df["status"].value_counts().to_dict()
    lines.append(f"  datasets={n_ds} runs: " + " ".join(
        f"{k}={v}" for k, v in sorted(counts.items())))
    ok = df[df["status"] == "success"].copy()
    if not ok.empty:
        ok["ari"] = pd.to_numeric(ok["ari"], errors="coerce")
        best = ok.loc[ok.groupby(["method_name", "dataset_id"])["ari"].idxmax()]
        bv = (best.groupby("method_name")
              .agg(best_view_mean_ari=("ari", "mean"),
                   n=("dataset_id", "nunique"))
              .sort_values("best_view_mean_ari", ascending=False))
        lines.append("  best-view mean ARI by method (top 8):")
        for name, r in bv.head(8).iterrows():
            lines.append(f"    {name:<42} {r['best_view_mean_ari']:.4f}  (n={int(r['n'])})")
        # dynamic-method selected metric-count distribution
        if "selected_metric_count" in ok.columns:
            dyn = ok[ok["method_name"].str.contains("dynamic")]
            if not dyn.empty:
                dist = dyn["selected_metric_count"].value_counts().sort_index().to_dict()
                lines.append(f"  dynamic selected_metric_count dist: {dist}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--split-id", type=int, default=1)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--split-dir-suffix", default=PHASED_SUFFIX)
    p.add_argument("--limit-subfamilies", type=int, default=None,
                   help="process only the first N subfamilies (smoke/debug).")
    p.add_argument("--limit-datasets", type=int, default=None,
                   help="cap datasets per subfamily (smoke/debug).")
    p.add_argument("--methods", nargs="+", default=list(PHASED_METHODS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    analyzed = repo / "results_analysis" / "clustering_repository" / "analyzed_data"
    split_csv = analyzed / "experiment_splits" / "dataset_split_assignments.csv"

    subfamilies = discover_subfamilies(analyzed)
    if args.limit_subfamilies:
        subfamilies = subfamilies[: args.limit_subfamilies]
    print(f"[phaseD] {len(subfamilies)} subfamilies | split={args.split_id} | "
          f"{len(args.methods)} methods x 3 views | workers={args.max_workers}",
          flush=True)

    t0 = time.perf_counter()
    done = 0
    for i, sub in enumerate(subfamilies, 1):
        family = sub.parent.parent.name
        cfg = ExperimentExecutionConfig(
            repo_root=repo, subfamily_dir=sub, split_assignments=split_csv,
            split_id=int(args.split_id), split_dir_suffix=str(args.split_dir_suffix),
            method_set="phaseD", methods=tuple(args.methods),
            max_workers=int(args.max_workers), resume=not args.overwrite,
            overwrite=bool(args.overwrite), limit_datasets=args.limit_datasets,
        )
        ts = time.perf_counter()
        print(f"\n[{i}/{len(subfamilies)}] {family}/{sub.name} ...", flush=True)
        try:
            rc = run_subfamily(cfg)
            done += 1
            elapsed = time.perf_counter() - ts
            print(f"  rc={rc} runtime={elapsed:.0f}s", flush=True)
            print(_subfamily_report(cfg), flush=True)
        except Exception as exc:  # never let one subfamily stop the sweep
            print(f"  [ERROR] {type(exc).__name__}: {exc}", flush=True)

    print(f"\n[phaseD] swept {done}/{len(subfamilies)} subfamilies in "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
