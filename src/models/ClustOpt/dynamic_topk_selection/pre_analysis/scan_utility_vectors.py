"""Discover and load every utility vector in the clustering repository.

Scan index
----------
The authoritative list of datasets is ``experiment_splits/
dataset_split_assignments.csv`` (one row per dataset, 17,068 rows). It carries
``dataset_dir`` plus family / subfamily / split / difficulty metadata, so we
never have to walk the whole tree. Each dataset has three views; the utility
vector lives at::

    <dataset_dir>/utility/<view>/utility_vector.json

with a ``utilities`` dict keyed ``utility__<metric_name>``.

A fallback filesystem walk is provided for robustness if the index is missing.

Each record is reduced to a flat stats/heuristics row plus a wide
``metric_<name> -> value`` mapping. Loading is parallel (multiprocessing) and
resumable via per-shard parquet checkpoints.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Keep sibling-module imports working under Windows 'spawn' worker processes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heuristic_candidates import compute_heuristics
from utility_vector_stats import clean_vector, compute_vector_stats, sorted_clean

VIEWS = ("x_only", "y_only", "xy_2d")
UTILITY_PREFIX = "utility__"


@dataclass
class ScanPaths:
    repo_root: str

    @property
    def analyzed_data(self) -> str:
        return os.path.join(
            self.repo_root, "results_analysis", "clustering_repository",
            "analyzed_data",
        )

    @property
    def split_assignments(self) -> str:
        return os.path.join(
            self.analyzed_data, "experiment_splits",
            "dataset_split_assignments.csv",
        )


def build_task_index(paths: ScanPaths) -> pd.DataFrame:
    """One row per (dataset, view) record to scan.

    Uses the split-assignment CSV when present, else falls back to a glob walk.
    """
    if os.path.exists(paths.split_assignments):
        df = pd.read_csv(paths.split_assignments)
        rows = []
        for _, r in df.iterrows():
            for v in VIEWS:
                rows.append({
                    "record_id": f"{r['dataset_id']}__{v}",
                    "dataset_id": r["dataset_id"],
                    "family": r.get("family", r["family_id"]),
                    "family_id": r["family_id"],
                    "subfamily": r["subfamily"],
                    "subfamily_id": r.get("subfamily_id", r["subfamily"]),
                    "view_type": v,
                    "dataset_dir": r["dataset_dir"],
                    "split_id": r.get("split_id", -1),
                    "difficulty": r.get("difficulty", ""),
                    "cluster_count": r.get("cluster_count", -1),
                    "n_points": r.get("n_points", -1),
                    "utility_file_path": os.path.join(
                        r["dataset_dir"], "utility", v, "utility_vector.json"),
                })
        return pd.DataFrame(rows)

    # -- fallback: walk the tree -------------------------------------------
    rows = []
    pattern = os.path.join(paths.analyzed_data, "*", "subfamilies", "*", "*",
                           "utility", "*", "utility_vector.json")
    for fp in glob.glob(pattern):
        parts = fp.replace("\\", "/").split("/")
        view = parts[-2]
        dataset_id = parts[-4]
        subfamily = parts[-6]
        family_id = parts[-8]
        rows.append({
            "record_id": f"{dataset_id}__{view}",
            "dataset_id": dataset_id, "family": family_id,
            "family_id": family_id, "subfamily": subfamily,
            "subfamily_id": subfamily, "view_type": view,
            "dataset_dir": os.path.dirname(os.path.dirname(os.path.dirname(fp))),
            "split_id": -1, "difficulty": "", "cluster_count": -1,
            "n_points": -1, "utility_file_path": fp,
        })
    return pd.DataFrame(rows)


def _parse_utilities(blob: dict) -> tuple[list[str], np.ndarray]:
    """Extract (metric_names, values) from a utility_vector.json dict."""
    u = blob.get("utilities", {})
    names, vals = [], []
    for k, v in u.items():
        name = k[len(UTILITY_PREFIX):] if k.startswith(UTILITY_PREFIX) else k
        names.append(name)
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return names, np.asarray(vals, dtype=float)


# Module-level so multiprocessing can pickle it.
def process_record(task: dict) -> dict:
    """Load one record, compute all stats + heuristics. Never raises."""
    base = {k: task[k] for k in (
        "record_id", "dataset_id", "family", "family_id", "subfamily",
        "subfamily_id", "view_type", "dataset_dir", "split_id", "difficulty",
        "cluster_count", "n_points", "utility_file_path")}
    fp = task["utility_file_path"]

    if not os.path.exists(fp):
        return {**base, "status": "missing", "error": "file not found"}
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - want any read/parse failure logged
        return {**base, "status": "malformed", "error": f"{type(exc).__name__}: {exc}"}

    names, vals = _parse_utilities(blob)
    if vals.size == 0:
        return {**base, "status": "malformed", "error": "empty utilities dict"}

    stats = compute_vector_stats(vals)
    s = sorted_clean(vals)
    heur = compute_heuristics(
        s, stats.get("participation_ratio", float("nan")),
        stats.get("perplexity_effective_k", float("nan")))

    row = {**base, "status": "ok",
           "n_metrics_declared": blob.get("n_metrics", len(names)),
           "metric_names": "|".join(names), **stats, **heur}
    # wide metric columns
    wide = {f"metric_{n}": float(v) for n, v in zip(names, vals)}
    row["_wide"] = wide
    return row


def scan(paths: ScanPaths, n_workers: int, shard_dir: str,
         shard_size: int = 4000, limit: int | None = None,
         progress_every: int = 5000):
    """Scan all records with checkpointed shards; returns (summary_df, wide_df).

    Resumability: each completed shard is written to ``shard_dir`` as parquet.
    On rerun, finished shards are loaded instead of recomputed.
    """
    import multiprocessing as mp

    os.makedirs(shard_dir, exist_ok=True)
    tasks = build_task_index(paths)
    if limit is not None:
        tasks = tasks.iloc[:limit].copy()
    task_list = tasks.to_dict("records")
    n = len(task_list)
    print(f"[scan] {n} records to process (workers={n_workers})", flush=True)

    summary_rows: list[dict] = []
    wide_rows: list[dict] = []

    n_shards = (n + shard_size - 1) // shard_size
    for si in range(n_shards):
        shard_path = os.path.join(shard_dir, f"shard_{si:04d}.parquet")
        wide_path = os.path.join(shard_dir, f"wide_{si:04d}.parquet")
        if os.path.exists(shard_path) and os.path.exists(wide_path):
            summary_rows.extend(pd.read_parquet(shard_path).to_dict("records"))
            wide_rows.extend(pd.read_parquet(wide_path).to_dict("records"))
            print(f"[scan] shard {si + 1}/{n_shards} loaded from cache", flush=True)
            continue

        chunk = task_list[si * shard_size:(si + 1) * shard_size]
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                results = pool.map(process_record, chunk, chunksize=64)
        else:
            results = [process_record(t) for t in chunk]

        shard_summary, shard_wide = [], []
        for r in results:
            wide = r.pop("_wide", {})
            shard_summary.append(r)
            shard_wide.append({"record_id": r["record_id"], **wide})
        pd.DataFrame(shard_summary).to_parquet(shard_path, index=False)
        pd.DataFrame(shard_wide).to_parquet(wide_path, index=False)
        summary_rows.extend(shard_summary)
        wide_rows.extend(shard_wide)
        print(f"[scan] shard {si + 1}/{n_shards} done "
              f"({len(summary_rows)}/{n})", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    wide_df = pd.DataFrame(wide_rows)
    return summary_df, wide_df
