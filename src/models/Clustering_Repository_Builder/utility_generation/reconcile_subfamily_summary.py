"""Rebuild the subfamily-level summary from per-view ``utility_record_summary.json``.

Useful when the runtime CSV labels rows as ``skipped`` or ``failed`` even
though the on-disk per-view outputs are actually complete and successful
(e.g. a dataset succeeded in an earlier verification run and the later
full-subfamily run with ``--resume`` skipped it).

The script:
1. Walks every dataset folder under the subfamily directory.
2. For each view (``x_only``, ``y_only``, ``xy_2d``) reads the on-disk
   per-view summary JSON (and ``timing_summary.json`` / ``run_log.json``
   to fill in timing columns).
3. Rebuilds ``utility_generation_summary.csv`` / ``.json`` / ``_report.md`` /
   ``utility_generation_failures.csv`` / ``utility_generation_checkpoint.json``
   from that on-disk truth.

No clustering or utility computation is re-executed.

Usage::

    python -m models.Clustering_Repository_Builder.utility_generation.reconcile_subfamily_summary \
        --subfamily-dir "...annuli_variable_thickness"
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import UTILITY_MAP_FILES, VIEW_IDS
from .dataset_discovery import detect_family_subfamily, discover_datasets
from .summary_report import (
    SUMMARY_CSV_COLUMNS,
    write_checkpoint as _write_checkpoint_writer,  # not used directly
    write_failures_csv,
    write_summary_csv,
)
from .utility_output_writer import (
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from .utility_record_runner import RecordResult


def _safe_read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _read_record(dataset_dir: Path, view_id: str) -> RecordResult:
    """Reconstruct a :class:`RecordResult` from the on-disk per-view JSONs."""
    view_dir = dataset_dir / "utility" / view_id
    summary = _safe_read_json(view_dir / "utility_record_summary.json")
    timing = _safe_read_json(view_dir / "timing_summary.json") or {}

    family, subfamily, specific_subfamily = detect_family_subfamily(dataset_dir.parent)

    # Status: must be 'success' AND the three required output files must exist.
    required = ("metric_utilities.csv", "utility_vector.json", "utility_record_summary.json")
    files_ok = all((view_dir / f).is_file() for f in required)
    if summary is None or not files_ok:
        return RecordResult(
            dataset_id=dataset_dir.name,
            dataset_dir=str(dataset_dir),
            family=family,
            subfamily=subfamily,
            specific_subfamily=specific_subfamily,
            view_id=view_id,
            status="missing",
            metric_mode="",
            sample_size=0,
            map_name=UTILITY_MAP_FILES.get(view_id, ""),
            error_message="per-view outputs missing or unreadable",
        )

    status = str(summary.get("status", "")).lower() or "unknown"
    top5 = list(summary.get("top5_metrics") or [])

    return RecordResult(
        dataset_id=str(summary.get("dataset_id") or dataset_dir.name),
        dataset_dir=str(dataset_dir),
        family=family,
        subfamily=subfamily,
        specific_subfamily=specific_subfamily,
        view_id=str(summary.get("view_id") or view_id),
        status=status,
        metric_mode=str(summary.get("metric_mode", "")),
        sample_size=int(summary.get("sample_size", 0) or 0),
        map_name=str(summary.get("map_name", UTILITY_MAP_FILES.get(view_id, ""))),
        n_configs=int(summary.get("n_configs", 0) or 0),
        n_valid_configs=int(summary.get("n_valid_configs", 0) or 0),
        runtime_sec=float(summary.get("runtime_sec", timing.get("runtime_sec", 0.0)) or 0.0),
        fit_predict_sec=float(timing.get("fit_predict_sec", 0.0) or 0.0),
        cvi_eval_sec=float(timing.get("cvi_eval_sec", 0.0) or 0.0),
        utility_compute_sec=float(timing.get("utility_compute_sec", 0.0) or 0.0),
        n_metrics=int(summary.get("n_metrics", 0) or 0),
        top1_metric=summary.get("top1_metric"),
        top1_utility=float(summary.get("top1_utility", 0.0) or 0.0),
        top5_metrics=[str(m) for m in top5],
        error_message=str(summary.get("error_message", "") or ""),
    )


def reconcile(subfamily_dir: Path) -> dict[str, object]:
    if not subfamily_dir.is_dir():
        raise FileNotFoundError(f"Subfamily directory does not exist: {subfamily_dir}")

    valid, invalid = discover_datasets(subfamily_dir)
    if not valid:
        raise RuntimeError(f"No valid dataset folders found under {subfamily_dir}")

    family_id, subfamily_id, _ = detect_family_subfamily(subfamily_dir)

    records: list[RecordResult] = []
    counts: dict[str, int] = {"success": 0, "failed": 0, "skipped": 0, "missing": 0, "other": 0}

    for d in valid:
        for view_id in VIEW_IDS:
            rec = _read_record(d.dataset_dir, view_id)
            records.append(rec)
            slot = rec.status if rec.status in counts else "other"
            counts[slot] += 1

    # ---- write CSVs (atomic) ----
    summary_csv = subfamily_dir / "utility_generation_summary.csv"
    failures_csv = subfamily_dir / "utility_generation_failures.csv"
    write_summary_csv(records, summary_csv)
    write_failures_csv([r for r in records if r.status not in ("success", "skipped")], failures_csv)

    # ---- summary.json (preserve the original 'config' block if present) ----
    summary_json = subfamily_dir / "utility_generation_summary.json"
    prev = _safe_read_json(summary_json) or {}
    prev_config = prev.get("config", {})

    success_rate = (counts["success"] / len(records)) if records else 0.0
    tracker_payload = {
        "datasets_found": len(valid),
        "datasets_processed": len(valid),
        "records_expected": len(records),
        "records_success": counts["success"],
        "records_failed": counts["failed"] + counts["missing"] + counts["other"],
        "records_skipped": counts["skipped"],
        "records_success_rate": success_rate,
        "runtime_total_sec": float(prev.get("tracker", {}).get("runtime_total_sec", 0.0) or 0.0),
        "runtime_total_str": str(prev.get("tracker", {}).get("runtime_total_str", "")),
        "reconciled_at": datetime.now(timezone.utc).isoformat(),
    }
    output_files = [
        "utility_generation_summary.csv",
        "utility_generation_summary.json",
        "utility_generation_report.md",
        "utility_generation_failures.csv",
        "utility_generation_checkpoint.json",
    ]
    payload = {
        "subfamily_dir": str(subfamily_dir),
        "family_id": family_id,
        "subfamily_id": subfamily_id,
        "config": prev_config,
        "tracker": tracker_payload,
        "output_files": output_files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reconciled_from_disk": True,
    }
    write_json_atomic(summary_json, payload)

    # ---- report.md ----
    lines: list[str] = []
    lines.append("# Utility Generation Summary (reconciled from on-disk per-view outputs)")
    lines.append("")
    lines.append(f"- Subfamily directory: `{subfamily_dir}`")
    lines.append(f"- Family: `{family_id or ''}`")
    lines.append(f"- Subfamily: `{subfamily_id or ''}`")
    lines.append(f"- Reconciled at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- datasets: **{len(valid)}**")
    lines.append(f"- records expected: **{len(records)}**")
    lines.append(f"- records success: **{counts['success']}**")
    lines.append(f"- records skipped (still): **{counts['skipped']}**")
    lines.append(f"- records failed: **{counts['failed']}**")
    lines.append(f"- records missing on disk: **{counts['missing']}**")
    lines.append(f"- success rate: **{success_rate * 100:.2f}%**")
    lines.append("")
    lines.append("## Output files")
    for f in output_files:
        lines.append(f"- `{f}`")
    write_text_atomic(subfamily_dir / "utility_generation_report.md", "\n".join(lines) + "\n")

    # ---- checkpoint.json ----
    by_dataset: dict[str, dict[str, str]] = {}
    for r in records:
        by_dataset.setdefault(r.dataset_id, {})[r.view_id] = r.status
    write_json_atomic(
        subfamily_dir / "utility_generation_checkpoint.json",
        {
            "subfamily_dir": str(subfamily_dir),
            "datasets": by_dataset,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "datasets": len(valid),
        "records_total": len(records),
        "counts": counts,
        "success_rate": success_rate,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuild the subfamily-level summary from per-view JSONs on disk.",
    )
    p.add_argument("--subfamily-dir", required=True, type=str,
                   help="Path to the subfamily directory to reconcile.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        result = reconcile(Path(args.subfamily_dir).resolve())
    except Exception:
        traceback.print_exc()
        return 1

    print(f"Reconciled subfamily summary from disk:")
    print(f"  datasets:        {result['datasets']}")
    print(f"  records:         {result['records_total']}")
    print(f"  success:         {result['counts']['success']}")
    print(f"  skipped (still): {result['counts']['skipped']}")
    print(f"  failed:          {result['counts']['failed']}")
    print(f"  missing on disk: {result['counts']['missing']}")
    print(f"  success rate:    {result['success_rate'] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
