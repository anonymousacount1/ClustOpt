"""CLI: build a balanced 16-way split over all raw datasets.

Run as a module::

    python -m models.Clustering_Repository_Builder.experiments.data_splitting.run_create_splits \
        --repo-root <PATH> \
        --analyzed-data-root <PATH>\\results_analysis\\clustering_repository\\analyzed_data \
        --n-splits 16 \
        --seed 42

Outputs are written under
``<analyzed-data-root>/experiment_splits/``.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .artifact_writer import write_csv_atomic, write_json_atomic, write_text_atomic
from .dataset_index_builder import build_dataset_index
from .split_report import (
    build_distribution_tables,
    render_summary_md,
    split_summary_frame,
)
from .split_validator import validate_assignments
from .stratified_splitter import assign_splits

OUTPUT_SUBDIR = "experiment_splits"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create balanced experiment splits over all raw datasets.",
    )
    p.add_argument("--repo-root", required=True, type=str)
    p.add_argument("--analyzed-data-root", required=True, type=str,
                   help="Path to results_analysis/clustering_repository/analyzed_data")
    p.add_argument("--n-splits", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output dir (default: <analyzed-data-root>/experiment_splits).")
    return p


def run(args: argparse.Namespace) -> int:
    analyzed_root = Path(args.analyzed_data_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else analyzed_root / OUTPUT_SUBDIR

    print("=" * 70, flush=True)
    print("AutoClustering experiment — create balanced splits", flush=True)
    print("=" * 70, flush=True)
    print(f"Analyzed-data root: {analyzed_root}", flush=True)
    print(f"Output dir:         {output_dir}", flush=True)
    print(f"n_splits={args.n_splits} seed={args.seed}", flush=True)
    print(flush=True)

    index_df, stats = build_dataset_index(analyzed_root)
    if index_df.empty:
        print("[error] no datasets found under analyzed-data root.", file=sys.stderr)
        return 3

    assigned = assign_splits(index_df, n_splits=args.n_splits, seed=args.seed)
    report = validate_assignments(assigned, n_splits=args.n_splits, raise_on_error=False)

    # ---- write artifacts ----
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_cols = [c for c in (
        "dataset_id", "dataset_dir", "family", "family_id", "subfamily",
        "subfamily_id", "specific_subfamily", "difficulty", "cluster_count",
        "n_points", "split_id",
    ) if c in assigned.columns]
    assignments_out = assigned[assignment_cols].copy()

    write_csv_atomic(output_dir / "dataset_split_assignments.csv", assignments_out)
    write_json_atomic(
        output_dir / "dataset_split_assignments.json",
        {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "analyzed_data_root": str(analyzed_root),
                "n_splits": int(args.n_splits),
                "seed": int(args.seed),
                "n_datasets": int(len(assignments_out)),
            },
            "assignments": assignments_out.to_dict(orient="records"),
        },
    )

    summary_df = split_summary_frame(assigned)
    write_csv_atomic(output_dir / "split_summary.csv", summary_df)

    tables = build_distribution_tables(assigned)
    for name, table in tables.items():
        if not table.empty:
            write_csv_atomic(output_dir / f"{name}.csv", table)

    md = render_summary_md(
        assigned, summary_df, n_splits=args.n_splits, seed=args.seed,
        validation_report=report,
    )
    write_text_atomic(output_dir / "split_summary.md", md)

    # ---- console summary ----
    print(flush=True)
    print(f"Datasets indexed: {len(assigned)}", flush=True)
    print(f"Split sizes: {report.split_sizes}", flush=True)
    print(f"Max size imbalance: {report.max_size_imbalance}", flush=True)
    print(f"Validation: {'OK' if report.ok else 'FAILED'}", flush=True)
    for e in report.errors:
        print(f"  ERROR: {e}", flush=True)
    for w in report.warnings:
        print(f"  warn: {w}", flush=True)
    print(flush=True)
    print(f"Wrote outputs under: {output_dir}", flush=True)
    return 0 if report.ok else 4


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
