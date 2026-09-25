"""Build a single unified training CSV from all per-dataset features + utilities.

For every dataset under
    analyzed_data/<family>/subfamilies/<subfamily>/<dataset>/
this script:
  1. reads features/features_records.csv  (one row per view: x_only/y_only/xy_2d)
  2. for each row, reads utility/<view>/utility_vector.json and pulls the
     'utilities' dict (the 60 final per-metric utility scores — no helper cols)
  3. merges them into a single wide row

All merged rows are concatenated into one CSV ready to be sent to a regressor.

Output (under <analyzed_data>/unified_training_dataset/):
  - unified_training_dataset.csv   — the main training table
  - build_metadata.json            — counts, column lists, timings
  - missing_artifacts.csv          — only written when something was skipped

Re-running fully overwrites the output directory's products.

CLI:
  python -m models.Clustering_Repository_Builder.training_dataset_builder.build_training_dataset
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from models.Clustering_Repository_Builder.training_dataset_builder.dataset_assembly import (
    MissingArtifact,
    assemble_per_dataset,
    atomic_write_csv,
    atomic_write_json,
    iter_datasets,
)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("training_dataset_builder")
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


def _detect_repo_root() -> Path:
    # this file lives at:
    #   <repo>/models/Clustering_Repository_Builder/training_dataset_builder/build_training_dataset.py
    return Path(__file__).resolve().parents[3]


def _default_analyzed_data_dir(repo_root: Path) -> Path:
    return repo_root / "results_analysis" / "clustering_repository" / "analyzed_data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="Repo root (default: auto-detect from this file).")
    parser.add_argument("--analyzed-data-dir", type=Path, default=None,
                        help="Override analyzed_data dir (default: "
                             "<repo>/results_analysis/clustering_repository/analyzed_data).")
    parser.add_argument("--output-name", type=str, default="unified_training_dataset",
                        help="Subfolder name (and base filename) under analyzed_data. "
                             "Default: unified_training_dataset.")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="Log progress every N datasets (default: 500).")
    args = parser.parse_args(argv)

    logger = _setup_logging()
    repo_root = (args.repo_root or _detect_repo_root()).resolve()
    analyzed_data_dir = (args.analyzed_data_dir or _default_analyzed_data_dir(repo_root)).resolve()
    if not analyzed_data_dir.is_dir():
        logger.error("analyzed_data dir not found: %s", analyzed_data_dir)
        return 2

    out_dir = analyzed_data_dir / args.output_name
    out_csv = out_dir / f"{args.output_name}.csv"
    out_metadata = out_dir / "build_metadata.json"
    out_missing = out_dir / "missing_artifacts.csv"

    logger.info("Repo root:        %s", repo_root)
    logger.info("Analyzed data:    %s", analyzed_data_dir)
    logger.info("Output directory: %s", out_dir)

    started_at = datetime.now(timezone.utc)
    t0 = time.time()

    missing: list[MissingArtifact] = []
    all_rows: list[dict] = []
    families_seen: set[str] = set()
    subfamilies_seen: set[tuple[str, str]] = set()
    n_datasets = 0

    for family, subfamily, ds_dir in iter_datasets(analyzed_data_dir):
        n_datasets += 1
        families_seen.add(family)
        subfamilies_seen.add((family, subfamily))
        rows = assemble_per_dataset(family, subfamily, ds_dir, missing)
        all_rows.extend(rows)
        if n_datasets % args.progress_every == 0:
            logger.info("processed %d datasets (%d rows so far, %d missing)",
                        n_datasets, len(all_rows), len(missing))

    if not all_rows:
        logger.error("no rows assembled — refusing to write empty CSV")
        return 3

    df = pd.DataFrame(all_rows)

    # Final column ordering: feature/identifier columns first (in their
    # original order), then utility__* targets (sorted) at the end.
    utility_cols = sorted(c for c in df.columns if c.startswith("utility__"))
    non_utility_cols = [c for c in df.columns if c not in set(utility_cols)]
    df = df[non_utility_cols + utility_cols]

    logger.info("assembled %d rows × %d columns (%d utility targets)",
                len(df), df.shape[1], len(utility_cols))

    atomic_write_csv(df, out_csv)
    logger.info("wrote %s", out_csv)

    if missing:
        pd.DataFrame([asdict(m) for m in missing]).to_csv(out_missing, index=False)
        logger.warning("%d artifacts missing — see %s", len(missing), out_missing)
    else:
        # remove any prior missing report so a clean rerun reflects clean state
        if out_missing.is_file():
            out_missing.unlink()

    finished_at = datetime.now(timezone.utc)
    metadata = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "runtime_sec": round(time.time() - t0, 3),
        "repo_root": str(repo_root),
        "analyzed_data_dir": str(analyzed_data_dir),
        "output_csv": str(out_csv),
        "n_families": len(families_seen),
        "n_subfamilies": len(subfamilies_seen),
        "n_datasets": n_datasets,
        "n_rows": int(len(df)),
        "n_columns_total": int(df.shape[1]),
        "n_feature_and_identifier_columns": len(non_utility_cols),
        "n_utility_targets": len(utility_cols),
        "utility_columns": utility_cols,
        "n_missing_artifacts": len(missing),
        "families": sorted(families_seen),
    }
    atomic_write_json(metadata, out_metadata)
    logger.info("wrote %s", out_metadata)

    logger.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
