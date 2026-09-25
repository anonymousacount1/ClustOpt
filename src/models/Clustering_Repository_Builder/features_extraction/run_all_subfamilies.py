"""Sequentially run feature extraction on every subfamily under an
analyzed-data root, then write a consolidated JSON + Markdown report.

Usage::

    python -m models.Clustering_Repository_Builder.features_extraction.run_all_subfamilies \\
        --analyzed-data-root results_analysis/clustering_repository/analyzed_data \\
        --overwrite \\
        --progress-every 1000 \\
        --report-dir results_analysis/clustering_repository/feature_extraction_runs/<ts>

The orchestrator calls :func:`run_subfamily` directly (same process), wrapping
each subfamily call in try/except so a single fatal error does not abort the
whole batch. A per-subfamily log file is also written via stdout redirection.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from .config import RunConfig
from .run_extract_subfamily_features import run_subfamily


def _now_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def discover_subfamilies(analyzed_data_root: Path) -> list[Path]:
    subs: list[Path] = []
    for family_dir in sorted(analyzed_data_root.iterdir()):
        sub_root = family_dir / "subfamilies"
        if not sub_root.is_dir():
            continue
        for sub in sorted(sub_root.iterdir()):
            if sub.is_dir():
                subs.append(sub)
    return subs


def _fmt_hms(seconds: float) -> str:
    if seconds < 0:
        return "00:00:00"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run feature extraction on every subfamily under analyzed_data/.",
    )
    parser.add_argument("--analyzed-data-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-landmarking", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-datasets", type=int, default=None,
                        help="Per-subfamily cap (debug only).")
    parser.add_argument("--report-dir", required=True, type=Path,
                        help="Where to write the consolidated report and per-subfamily logs.")
    args = parser.parse_args(argv)

    analyzed_root = args.analyzed_data_root.resolve()
    if not analyzed_root.is_dir():
        print(f"ERROR: analyzed_data root does not exist: {analyzed_root}", file=sys.stderr)
        return 2

    report_dir: Path = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = report_dir / "subfamily_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    subfamilies = discover_subfamilies(analyzed_root)
    print(f"Discovered {len(subfamilies)} subfamilies under {analyzed_root}", flush=True)
    print(f"Report directory: {report_dir}", flush=True)
    print(f"Per-subfamily logs: {logs_dir}", flush=True)
    print(f"Overwrite: {args.overwrite}  Skip landmarking: {args.skip_landmarking}", flush=True)

    overall_start = time.perf_counter()
    results: list[dict] = []

    for idx, sub_dir in enumerate(subfamilies, start=1):
        family_id = sub_dir.parent.parent.name
        subfamily_id = sub_dir.name
        rel_label = f"{family_id}/{subfamily_id}"
        print()
        print("#" * 70, flush=True)
        print(f"# [{idx:02d}/{len(subfamilies)}] {rel_label}", flush=True)
        print("#" * 70, flush=True)

        cfg = RunConfig(
            subfamily_root=str(sub_dir),
            overwrite=bool(args.overwrite),
            skip_landmarking=bool(args.skip_landmarking),
            progress_every=int(args.progress_every),
            max_datasets=args.max_datasets,
            write_parquet=not args.no_parquet,
        )

        log_path = logs_dir / f"{idx:02d}__{family_id}__{subfamily_id}.log"
        sub_start = time.perf_counter()
        payload: Optional[dict] = None
        fatal_error: Optional[str] = None
        fatal_traceback: Optional[str] = None

        # Redirect stdout/stderr of this subfamily into its own log file. The
        # orchestrator-level prints (the "#" banners above) still go to the
        # main log so the batch progress remains readable.
        with log_path.open("w", encoding="utf-8") as log_fh:
            with contextlib.redirect_stdout(log_fh), contextlib.redirect_stderr(log_fh):
                try:
                    payload = run_subfamily(cfg)
                except Exception as exc:
                    fatal_error = f"{type(exc).__name__}: {exc}"
                    fatal_traceback = traceback.format_exc()

        elapsed = time.perf_counter() - sub_start
        if payload is not None:
            block_rates_raw = _safe_get(
                payload, "missing_value_summary", "raw_missing_before_fill", "block_rates", default={}
            )
            block_rates_after = _safe_get(
                payload, "missing_value_summary", "missing_after_fill", "block_rates", default={}
            )
            block_means = _safe_get(payload, "block_timing_summary", "means_sec", default={})
            results.append({
                "idx": idx,
                "family_id": family_id,
                "subfamily_id": subfamily_id,
                "subfamily_root": str(sub_dir),
                "log_path": str(log_path),
                "status": "ok",
                "datasets_found": int(payload.get("dataset_folders_found", 0)),
                "processed_successfully": int(payload.get("processed_successfully", 0)),
                "failed": int(payload.get("failed", 0)),
                "skipped": int(payload.get("skipped", 0)),
                "records_created": int(payload.get("records_created", 0)),
                "runtime_total_sec": float(payload.get("runtime_total_sec", elapsed)),
                "wallclock_sec": float(elapsed),
                "avg_time_per_dataset_sec": float(payload.get("avg_time_per_dataset_sec", 0.0)),
                "throughput_datasets_per_min": float(payload.get("throughput_datasets_per_min", 0.0)),
                "parquet_written": bool(payload.get("parquet_written", False)),
                "parquet_error": payload.get("parquet_error"),
                "raw_missing_block_rates": dict(block_rates_raw),
                "missing_after_fill_block_rates": dict(block_rates_after),
                "block_means_sec": dict(block_means),
                "dataset_level_failures": list(payload.get("failures", [])),
            })
            print(
                f"  -> ok: {payload.get('processed_successfully', 0)}/{payload.get('dataset_folders_found', 0)} "
                f"datasets, {payload.get('records_created', 0)} records, "
                f"elapsed {_fmt_hms(elapsed)}",
                flush=True,
            )
        else:
            results.append({
                "idx": idx,
                "family_id": family_id,
                "subfamily_id": subfamily_id,
                "subfamily_root": str(sub_dir),
                "log_path": str(log_path),
                "status": "fatal_error",
                "wallclock_sec": float(elapsed),
                "error_message": fatal_error,
                "traceback": fatal_traceback,
            })
            print(f"  -> FATAL: {fatal_error} (see {log_path})", flush=True)

        # Persist intermediate progress so a crash leaves something usable.
        intermediate = {
            "analyzed_data_root": str(analyzed_root),
            "subfamily_count": len(subfamilies),
            "completed": idx,
            "wallclock_so_far_sec": float(time.perf_counter() - overall_start),
            "wallclock_so_far_str": _fmt_hms(time.perf_counter() - overall_start),
            "overwrite": bool(args.overwrite),
            "skip_landmarking": bool(args.skip_landmarking),
            "results_so_far": results,
        }
        with (report_dir / "batch_run_report.partial.json").open("w", encoding="utf-8") as fh:
            json.dump(intermediate, fh, indent=2, default=str)

    overall_elapsed = time.perf_counter() - overall_start

    # Aggregate stats.
    total_datasets_found = sum(r.get("datasets_found", 0) for r in results if r["status"] == "ok")
    total_ok = sum(r.get("processed_successfully", 0) for r in results if r["status"] == "ok")
    total_failed = sum(r.get("failed", 0) for r in results if r["status"] == "ok")
    total_skipped = sum(r.get("skipped", 0) for r in results if r["status"] == "ok")
    total_records = sum(r.get("records_created", 0) for r in results if r["status"] == "ok")
    fatal_subfamilies = [r for r in results if r["status"] == "fatal_error"]

    report = {
        "analyzed_data_root": str(analyzed_root),
        "subfamily_count": len(subfamilies),
        "overall_runtime_sec": float(overall_elapsed),
        "overall_runtime_str": _fmt_hms(overall_elapsed),
        "overwrite": bool(args.overwrite),
        "skip_landmarking": bool(args.skip_landmarking),
        "totals": {
            "total_dataset_folders": int(total_datasets_found),
            "processed_successfully": int(total_ok),
            "dataset_level_failures": int(total_failed),
            "skipped": int(total_skipped),
            "records_created": int(total_records),
            "fatal_subfamily_count": int(len(fatal_subfamilies)),
        },
        "results": results,
    }
    with (report_dir / "batch_run_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    # Markdown rollup.
    md_lines: list[str] = []
    md_lines.append("# Batch feature-extraction report")
    md_lines.append("")
    md_lines.append(f"- analyzed_data root: `{analyzed_root}`")
    md_lines.append(f"- subfamilies: {len(subfamilies)}")
    md_lines.append(f"- overall wallclock: {_fmt_hms(overall_elapsed)}")
    md_lines.append(f"- overwrite: {args.overwrite}")
    md_lines.append(f"- skip_landmarking: {args.skip_landmarking}")
    md_lines.append("")
    md_lines.append("## Totals across subfamilies")
    md_lines.append("")
    md_lines.append(f"- dataset folders found: {total_datasets_found}")
    md_lines.append(f"- processed_successfully: {total_ok}")
    md_lines.append(f"- dataset-level failures: {total_failed}")
    md_lines.append(f"- skipped: {total_skipped}")
    md_lines.append(f"- records_created: {total_records}")
    md_lines.append(f"- fatal subfamily errors: {len(fatal_subfamilies)}")
    md_lines.append("")
    md_lines.append("## Per-subfamily summary")
    md_lines.append("")
    md_lines.append("| # | family | subfamily | status | datasets | ok | failed | records | wallclock | avg s/dataset |")
    md_lines.append("|---:|---|---|---|---:|---:|---:|---:|---|---:|")
    for r in results:
        if r["status"] == "ok":
            md_lines.append(
                f"| {r['idx']} | {r['family_id']} | {r['subfamily_id']} | ok | "
                f"{r['datasets_found']} | {r['processed_successfully']} | {r['failed']} | "
                f"{r['records_created']} | {_fmt_hms(r['wallclock_sec'])} | "
                f"{r['avg_time_per_dataset_sec']:.2f} |"
            )
        else:
            md_lines.append(
                f"| {r['idx']} | {r['family_id']} | {r['subfamily_id']} | **fatal** | - | - | - | - | "
                f"{_fmt_hms(r['wallclock_sec'])} | - |"
            )
    md_lines.append("")
    if fatal_subfamilies:
        md_lines.append("## Fatal subfamily errors")
        md_lines.append("")
        for r in fatal_subfamilies:
            md_lines.append(f"### {r['family_id']} / {r['subfamily_id']}")
            md_lines.append("")
            md_lines.append(f"- log: `{r['log_path']}`")
            md_lines.append(f"- error: `{r.get('error_message')}`")
            md_lines.append("")
    with (report_dir / "batch_run_report.md").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    print()
    print(f"Wrote consolidated JSON: {report_dir / 'batch_run_report.json'}", flush=True)
    print(f"Wrote consolidated MD:   {report_dir / 'batch_run_report.md'}", flush=True)
    print(f"Per-subfamily logs in:   {logs_dir}", flush=True)
    print(f"Total wallclock: {_fmt_hms(overall_elapsed)} ({overall_elapsed/3600:.2f} hours)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
