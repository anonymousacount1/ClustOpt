"""Subfamily-level summary writers (JSON + Markdown)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .feature_schema import ALL_FEATURE_COLUMNS, BLOCK_COLUMNS, schema_metadata


def _json_default(value):  # type: ignore
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _count_block_and_column_nans(records: list[dict]) -> dict[str, object]:
    """Count NaN / None / non-finite values per block and per column.

    Returns a dict with `total_records`, `block_rates`, and `column_rates`.
    A value is considered missing if it is None, NaN, +/-Inf, or not coercible
    to a finite float. Strings are treated as missing.
    """
    if not records:
        return {"total_records": 0, "block_rates": {}, "column_rates": {}}
    block_rates: dict[str, float] = {}
    for block_name, cols in BLOCK_COLUMNS.items():
        n_nan = 0
        for rec in records:
            for c in cols:
                v = rec.get(c, np.nan)
                try:
                    if v is None or not np.isfinite(float(v)):
                        n_nan += 1
                except (TypeError, ValueError):
                    n_nan += 1
        block_rates[block_name] = n_nan / (len(records) * len(cols))

    column_rates: dict[str, float] = {}
    for c in ALL_FEATURE_COLUMNS:
        n_nan = 0
        for rec in records:
            v = rec.get(c, np.nan)
            try:
                if v is None or not np.isfinite(float(v)):
                    n_nan += 1
            except (TypeError, ValueError):
                n_nan += 1
        column_rates[c] = n_nan / len(records)

    return {
        "total_records": len(records),
        "block_rates": block_rates,
        "column_rates": column_rates,
    }


def compute_missing_value_summary(
    raw_records: Iterable[dict] | None,
    filled_records: Iterable[dict] | None = None,
) -> dict[str, object]:
    """Return both pre-fill and post-fill missing-value statistics.

    `raw_records` are the unfilled 250-feature dicts captured by
    `feature_extractor.extract_features_for_view` (NaN preserved).
    `filled_records` are the schema-complete records actually written to disk
    (should have 0% missing after the 0.0-fill policy).

    The structure makes the source of each stat explicit:

        {
          "raw_missing_before_fill": {total_records, block_rates, column_rates},
          "missing_after_fill":      {total_records, block_rates, column_rates},
        }
    """
    raw_list = list(raw_records) if raw_records is not None else []
    filled_list = list(filled_records) if filled_records is not None else []
    return {
        "raw_missing_before_fill": _count_block_and_column_nans(raw_list),
        "missing_after_fill": _count_block_and_column_nans(filled_list),
    }


def build_summary_payload(
    *,
    subfamily_root: Path,
    family_id: str | None,
    subfamily_id: str | None,
    tracker_summary: dict,
    expected_records: int,
    parquet_written: bool,
    parquet_error: str | None,
    skip_landmarking: bool,
    overwrite: bool,
    missing_value_summary: dict,
    output_files: list[str],
) -> dict[str, object]:
    schema = schema_metadata()
    payload = {
        "subfamily_root": str(subfamily_root),
        "family_id": family_id,
        "subfamily_id": subfamily_id,
        "dataset_folders_found": tracker_summary["datasets_found"],
        "processed_successfully": tracker_summary["processed_successfully"],
        "failed": tracker_summary["failed"],
        "skipped": tracker_summary["skipped"],
        "records_created": tracker_summary["records_created"],
        "expected_records": int(expected_records),
        "feature_count": schema["n_features"],
        "block_counts": schema["blocks"],
        "runtime_total_sec": tracker_summary["runtime_total_sec"],
        "runtime_total_str": tracker_summary["runtime_total_str"],
        "avg_time_per_dataset_sec": tracker_summary["avg_time_per_dataset_sec"],
        "throughput_datasets_per_min": tracker_summary["throughput_datasets_per_min"],
        "parquet_written": bool(parquet_written),
        "parquet_error": parquet_error,
        "skip_landmarking": bool(skip_landmarking),
        "overwrite": bool(overwrite),
        "missing_value_summary": missing_value_summary,
        "block_timing_summary": {
            "totals_sec": tracker_summary["block_timing_totals_sec"],
            "means_sec": tracker_summary["block_timing_means_sec"],
        },
        "failures": tracker_summary["failures"],
        "output_files": output_files,
    }
    return payload


def write_summary_json(payload: dict, path: Path) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def write_summary_md(payload: dict, path: Path) -> Path:
    lines: list[str] = []
    lines.append(f"# Subfamily feature-extraction summary")
    lines.append("")
    lines.append(f"- Subfamily root: `{payload['subfamily_root']}`")
    lines.append(f"- Family id: `{payload.get('family_id')}`")
    lines.append(f"- Subfamily id: `{payload.get('subfamily_id')}`")
    lines.append(f"- Dataset folders found: {payload['dataset_folders_found']}")
    lines.append(f"- Processed successfully: {payload['processed_successfully']}")
    lines.append(f"- Failed: {payload['failed']}")
    lines.append(f"- Skipped: {payload['skipped']}")
    lines.append(f"- Records created: {payload['records_created']}")
    lines.append(f"- Expected records (3 per OK dataset): {payload['expected_records']}")
    lines.append(f"- Total feature columns per record: {payload['feature_count']}")
    lines.append("")
    lines.append("## Block counts")
    lines.append("")
    for block, count in payload["block_counts"].items():
        lines.append(f"- {block}: {count}")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Total elapsed: {payload['runtime_total_str']}")
    lines.append(f"- Avg time per dataset (sec): {payload['avg_time_per_dataset_sec']:.3f}")
    lines.append(f"- Throughput (datasets/min): {payload['throughput_datasets_per_min']:.2f}")
    lines.append("")
    lines.append("### Block timing (total, mean per dataset)")
    lines.append("")
    lines.append("| Block | total_sec | mean_per_dataset_sec |")
    lines.append("|---|---:|---:|")
    totals = payload["block_timing_summary"]["totals_sec"]
    means = payload["block_timing_summary"]["means_sec"]
    for key in sorted(totals.keys()):
        lines.append(f"| {key} | {totals[key]:.2f} | {means[key]:.3f} |")
    lines.append("")
    lines.append("## Parquet")
    lines.append("")
    lines.append(f"- Written: {payload['parquet_written']}")
    if payload.get("parquet_error"):
        lines.append(f"- Parquet error: `{payload['parquet_error']}`")
    lines.append("")
    lines.append("## Missing values")
    lines.append("")
    mv = payload.get("missing_value_summary", {})
    raw = mv.get("raw_missing_before_fill", {}) or {}
    after = mv.get("missing_after_fill", {}) or {}
    lines.append(f"- Records analysed (raw, pre-fill): {raw.get('total_records', 0)}")
    lines.append(f"- Records analysed (post-fill, written to CSV/Parquet): {after.get('total_records', 0)}")
    lines.append(
        f"- Fill policy: features → 0.0; audit strings → \"unknown\"; "
        f"audit ints → -1; audit bools → False; audit lists → []"
    )
    lines.append("")
    lines.append("### Per-block NaN rate — raw (BEFORE fill)")
    lines.append("")
    lines.append("| Block | NaN rate |")
    lines.append("|---|---:|")
    for block, rate in raw.get("block_rates", {}).items():
        lines.append(f"| {block} | {rate:.3f} |")
    lines.append("")
    lines.append("### Per-block NaN rate — post-fill (AFTER 0.0-fill, should all be 0.000)")
    lines.append("")
    lines.append("| Block | NaN rate |")
    lines.append("|---|---:|")
    for block, rate in after.get("block_rates", {}).items():
        lines.append(f"| {block} | {rate:.3f} |")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for f in payload.get("output_files", []):
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Failed datasets")
    lines.append("")
    if payload.get("failures"):
        lines.append("| dataset_id | error_stage | error_message |")
        lines.append("|---|---|---|")
        for f in payload["failures"]:
            lines.append(f"| `{f['dataset_id']}` | {f['error_stage']} | {f['error_message']} |")
    else:
        lines.append("None.")
    lines.append("")

    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path
