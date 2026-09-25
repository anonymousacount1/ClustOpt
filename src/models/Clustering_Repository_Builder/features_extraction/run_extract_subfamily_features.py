"""Run feature extraction on a single subfamily directory.

Usage::

    python -m models.Clustering_Repository_Builder.features_extraction.run_extract_subfamily_features \
        --subfamily-root <path> [--overwrite] [--skip-landmarking] \
        [--max-datasets N] [--progress-every N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from .config import RunConfig
from .dataset_loader import LoadedDataset, load_dataset
from .feature_extractor import extract_features_for_view
from .feature_schema import ALL_FEATURE_COLUMNS
from .output_writer import (
    write_dataset_outputs,
    write_failed_csv,
    write_schema_file,
    write_subfamily_combined_csv,
    write_subfamily_parquet,
)
from .progress_tracker import DatasetTiming, ProgressTracker
from .summary_writer import (
    build_summary_payload,
    compute_missing_value_summary,
    write_summary_json,
    write_summary_md,
)
from .view_builder import build_views


SUBFAMILY_OUTPUT_FILES = [
    "subfamily_features_all.csv",
    "subfamily_features_all.parquet",
    "subfamily_feature_extraction_summary.json",
    "subfamily_feature_extraction_summary.md",
    "subfamily_feature_schema.json",
    "failed_datasets.csv",
]


def _discover_datasets(subfamily_root: Path) -> list[Path]:
    if not subfamily_root.is_dir():
        raise FileNotFoundError(f"Subfamily root does not exist: {subfamily_root}")
    candidates: list[Path] = []
    for entry in sorted(subfamily_root.iterdir()):
        if entry.is_dir() and (entry / "data_data.csv").is_file():
            candidates.append(entry)
    return candidates


def _detect_family_subfamily(subfamily_root: Path) -> tuple[Optional[str], Optional[str]]:
    parts = subfamily_root.resolve().parts
    if "subfamilies" not in parts:
        return None, subfamily_root.name
    idx = parts.index("subfamilies")
    subfamily_id = parts[idx + 1] if idx + 1 < len(parts) else subfamily_root.name
    family_id = parts[idx - 1] if idx - 1 >= 0 else None
    return family_id, subfamily_id


def _dataset_features_already_written(dataset_dir: Path) -> bool:
    csv_path = dataset_dir / "features" / "features_records.csv"
    return csv_path.is_file()


def _per_dataset_log(timing: DatasetTiming, block_errors: dict) -> dict[str, object]:
    out = timing.to_dict()
    out["block_errors"] = dict(block_errors)
    return out


def run_subfamily(cfg: RunConfig) -> dict[str, object]:
    subfamily_root = Path(cfg.subfamily_root).resolve()
    family_id, subfamily_id = _detect_family_subfamily(subfamily_root)
    dataset_dirs = _discover_datasets(subfamily_root)
    if cfg.max_datasets is not None and cfg.max_datasets > 0:
        dataset_dirs = dataset_dirs[: cfg.max_datasets]

    tracker = ProgressTracker(total_datasets=len(dataset_dirs), progress_every=cfg.progress_every)
    tracker.start(
        subfamily_root=subfamily_root,
        metadata={
            "family_id": family_id,
            "subfamily_id": subfamily_id,
            "datasets_found": len(dataset_dirs),
            "expected_records": len(dataset_dirs) * 3,
            "feature_count": len(ALL_FEATURE_COLUMNS),
            "overwrite": cfg.overwrite,
            "skip_landmarking": cfg.skip_landmarking,
            "output_files": SUBFAMILY_OUTPUT_FILES,
        },
    )

    rng = np.random.default_rng(cfg.random_seed)
    all_records: list[dict] = []
    all_raw_feature_records: list[dict] = []

    for i, dataset_dir in enumerate(dataset_dirs, start=1):
        tracker.start_dataset(i, dataset_dir.name)

        if not cfg.overwrite and _dataset_features_already_written(dataset_dir):
            tracker.skip_dataset(i, dataset_dir.name, reason="features/ already exists; pass --overwrite to recompute")
            continue

        dataset_start = time.perf_counter()
        timing = DatasetTiming()
        block_errors_per_view: dict[str, dict[str, str]] = {}

        # ---- load ----
        load_t0 = time.perf_counter()
        try:
            loaded: LoadedDataset = load_dataset(dataset_dir)
        except Exception as exc:
            tracker.finish_dataset_failed(i, dataset_dir.name, stage="loading", error=exc, dataset_dir=str(dataset_dir))
            continue
        timing.load_time_sec = time.perf_counter() - load_t0
        tracker.stage("loading artifacts", ok=True, elapsed=timing.load_time_sec)

        # ---- build views ----
        view_t0 = time.perf_counter()
        try:
            views = build_views(loaded)
        except Exception as exc:
            tracker.finish_dataset_failed(i, dataset_dir.name, stage="view_building", error=exc, dataset_dir=str(dataset_dir))
            continue
        timing.view_build_time_sec = time.perf_counter() - view_t0
        tracker.stage(
            "building views",
            ok=True,
            detail=", ".join(v.view_mode for v in views),
            elapsed=timing.view_build_time_sec,
        )

        # ---- extract features ----
        feat_t0 = time.perf_counter()
        view_records: list[dict] = []
        view_raw_records: list[dict] = []
        try:
            for v in views:
                rec, view_timing, raw_rec = extract_features_for_view(
                    loaded=loaded,
                    view=v,
                    skip_landmarking=cfg.skip_landmarking,
                    rng=rng,
                )
                view_records.append(rec)
                view_raw_records.append(raw_rec)
                for key in ("block_A_time_sec", "block_B_time_sec", "block_G_time_sec",
                            "block_V_time_sec", "block_R_time_sec", "block_L_time_sec"):
                    setattr(timing, key, getattr(timing, key) + float(view_timing.get(key, 0.0) or 0.0))
                if view_timing.get("block_errors"):
                    block_errors_per_view[v.view_mode] = view_timing["block_errors"]
        except Exception as exc:
            tracker.finish_dataset_failed(i, dataset_dir.name, stage="feature_extraction", error=exc, dataset_dir=str(dataset_dir))
            continue
        feat_elapsed = time.perf_counter() - feat_t0
        tracker.stage("extracting A/B/G/V/R/L", ok=True, elapsed=feat_elapsed)

        # ---- save per-dataset ----
        save_t0 = time.perf_counter()
        try:
            write_dataset_outputs(
                dataset_dir=dataset_dir,
                records=view_records,
                timing=_per_dataset_log(timing, block_errors_per_view),
                overwrite=cfg.overwrite,
            )
        except Exception as exc:
            tracker.finish_dataset_failed(i, dataset_dir.name, stage="saving_features", error=exc, dataset_dir=str(dataset_dir))
            continue
        timing.save_time_sec = time.perf_counter() - save_t0
        tracker.stage("saving dataset features", ok=True, elapsed=timing.save_time_sec)

        timing.total_time_sec = time.perf_counter() - dataset_start
        all_records.extend(view_records)
        all_raw_feature_records.extend(view_raw_records)
        tracker.finish_dataset_ok(i, dataset_dir.name, records=len(view_records), timing=timing.to_dict())

    # ----------- subfamily-level outputs -----------
    csv_path = subfamily_root / "subfamily_features_all.csv"
    parquet_path = subfamily_root / "subfamily_features_all.parquet"
    schema_path = subfamily_root / "subfamily_feature_schema.json"
    summary_json_path = subfamily_root / "subfamily_feature_extraction_summary.json"
    summary_md_path = subfamily_root / "subfamily_feature_extraction_summary.md"
    failed_csv_path = subfamily_root / "failed_datasets.csv"

    write_subfamily_combined_csv(all_records, csv_path)
    parquet_written, parquet_error = (False, "skipped")
    if cfg.write_parquet:
        parquet_written, parquet_error = write_subfamily_parquet(all_records, parquet_path)
    write_schema_file(schema_path)
    write_failed_csv([f.__dict__ for f in tracker.failures], failed_csv_path)

    tracker_summary = tracker.finish()
    payload = build_summary_payload(
        subfamily_root=subfamily_root,
        family_id=family_id,
        subfamily_id=subfamily_id,
        tracker_summary=tracker_summary,
        expected_records=tracker.processed_ok * 3,
        parquet_written=parquet_written,
        parquet_error=parquet_error,
        skip_landmarking=cfg.skip_landmarking,
        overwrite=cfg.overwrite,
        missing_value_summary=compute_missing_value_summary(
            raw_records=all_raw_feature_records,
            filled_records=all_records,
        ),
        output_files=SUBFAMILY_OUTPUT_FILES,
    )
    write_summary_json(payload, summary_json_path)
    write_summary_md(payload, summary_md_path)

    tracker.print_final_summary(
        subfamily_id=subfamily_id or subfamily_root.name,
        output_files=SUBFAMILY_OUTPUT_FILES,
        expected_records=tracker.processed_ok * 3,
        parquet_status="ok" if parquet_written else f"failed ({parquet_error})",
    )

    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clustering Repository Builder — feature extraction for one subfamily.",
    )
    parser.add_argument("--subfamily-root", required=True, type=str, help="Path to a single subfamily directory.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even when features/ already exists.")
    parser.add_argument("--skip-landmarking", action="store_true", help="Fill L_* features with NaN.")
    parser.add_argument("--max-datasets", type=int, default=None, help="Optional cap on number of datasets to process.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print verbose stage lines every N datasets.")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future parallel extraction.")
    parser.add_argument("--no-parquet", action="store_true", help="Skip Parquet output (CSV still written).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = RunConfig(
        subfamily_root=args.subfamily_root,
        overwrite=bool(args.overwrite),
        skip_landmarking=bool(args.skip_landmarking),
        max_datasets=args.max_datasets,
        progress_every=int(args.progress_every),
        workers=int(args.workers),
        write_parquet=not args.no_parquet,
        random_seed=int(args.seed),
    )
    try:
        run_subfamily(cfg)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
