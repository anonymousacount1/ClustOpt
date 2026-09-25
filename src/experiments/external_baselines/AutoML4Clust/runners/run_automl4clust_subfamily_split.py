"""Final runner: AutoML4Clust over one subfamily and one split id.

* Scans dataset folders under ``--subfamily-root``.
* Keeps only datasets assigned to ``--split-id`` (via the split-assignment CSV);
  the CSV's ``cluster_count`` becomes the per-dataset true-K override (external
  eval only).
* Processes one dataset per worker (``ProcessPoolExecutor``); no nested
  multiprocessing. A crash in one dataset is captured and never stops the rest.
* Prints per-dataset progress (index, id, record statuses, best ARI so far,
  elapsed, ETA).
* Writes a run-level summary under
  ``results_analysis/clustering_repository/external_baselines/AutoML4Clust/<ts>_split_XX_<subfamily>/``.
  Primary per-dataset results still live inside each dataset folder.

CLI example::

    python -m experiments.external_baselines.AutoML4Clust.runners.run_automl4clust_subfamily_split ^
      --subfamily-root "<path_to_subfamily_folder>" ^
      --split-assignment-csv "<path_to_dataset_split_assignment_csv>" ^
      --split-id 1 ^
      --config "experiments/external_baselines/AutoML4Clust/configs/automl4clust_smac_2d.json" ^
      --record-types 1d_x 1d_y 2d ^
      --workers 8 ^
      --repo-root "."
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from experiments.external_baselines.AutoML4Clust.runners.run_automl4clust_dataset import (
    load_config,
    process_single_dataset,
)
from experiments.external_baselines.AutoML4Clust.utils.logging_utils import configure_process
from experiments.external_baselines.AutoML4Clust.utils.path_utils import (
    RECORD_TYPES,
    ensure_dir,
    path_exists,
    split_dir_name,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)

configure_process()


# --------------------------------------------------------------------- discovery
def discover_dataset_dirs(subfamily_root: Path, *, max_depth: int = 3) -> List[Path]:
    """Return dataset folders (containing ``data_data.csv``) under a subfamily root."""
    subfamily_root = Path(subfamily_root)
    found: List[Path] = []
    if not subfamily_root.is_dir():
        return found
    root_depth = len(subfamily_root.resolve().parts)
    for dirpath, dirnames, filenames in os.walk(subfamily_root):
        p = Path(dirpath)
        depth = len(p.resolve().parts) - root_depth
        if "data_data.csv" in filenames:
            found.append(p)
            dirnames[:] = []  # don't descend into a dataset folder
            continue
        if depth >= max_depth:
            dirnames[:] = []
    return sorted(found, key=lambda d: d.name)


def load_split_assignments(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"split assignment CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "dataset_id" not in df.columns or "split_id" not in df.columns:
        raise ValueError(
            f"split assignment CSV missing dataset_id/split_id (have {list(df.columns)})"
        )
    return df


def filter_by_split(
    dataset_dirs: List[Path], assignments: pd.DataFrame, split_id: int
) -> Tuple[List[Path], Dict[str, Optional[int]]]:
    """Keep dirs whose folder name is assigned to split_id; map id->true_k."""
    sub = assignments[assignments["split_id"].astype("Int64") == int(split_id)]
    wanted = set(sub["dataset_id"].astype(str).tolist())
    true_k_map: Dict[str, Optional[int]] = {}
    if "cluster_count" in sub.columns:
        for _, r in sub.iterrows():
            try:
                true_k_map[str(r["dataset_id"])] = (
                    int(r["cluster_count"]) if pd.notna(r["cluster_count"]) else None
                )
            except (ValueError, TypeError):
                true_k_map[str(r["dataset_id"])] = None
    selected = [d for d in dataset_dirs if d.name in wanted]
    return selected, true_k_map


# ----------------------------------------------------------------- multiprocessing
def _worker_entry(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Top-level picklable worker. Unpacks args and runs one dataset."""
    configure_process()
    (dataset_dir, split_id, config, record_types, repo_root, overwrite, true_k) = args
    try:
        return process_single_dataset(
            dataset_dir,
            split_id=split_id,
            config=config,
            record_types=record_types,
            repo_root=repo_root,
            overwrite=overwrite,
            true_k_override=true_k,
            echo=False,
        )
    except Exception as exc:  # pragma: no cover - top-level safety net
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {
            "dataset_id": Path(dataset_dir).name,
            "dataset_dir": str(dataset_dir),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "record_statuses": {},
            "best_ari": None,
            "total_runtime_sec": 0.0,
        }


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ------------------------------------------------------------------- run summary
def run_summary_dir(repo_root: Path, split_id: int, subfamily_name: str, ts: str) -> Path:
    return (
        Path(repo_root)
        / "results_analysis" / "clustering_repository" / "external_baselines"
        / "AutoML4Clust"
        / f"{ts}_{split_dir_name(split_id)}_{subfamily_name}"
    )


def _records_row(result: Dict[str, Any]) -> Dict[str, Any]:
    statuses = result.get("record_statuses", {}) or {}
    return {
        "dataset_id": result.get("dataset_id"),
        "dataset_dir": result.get("dataset_dir"),
        "family": result.get("family"),
        "subfamily": result.get("subfamily"),
        "difficulty": result.get("difficulty"),
        "true_k": result.get("true_k"),
        "status_1d_x": statuses.get("1d_x"),
        "status_1d_y": statuses.get("1d_y"),
        "status_2d": statuses.get("2d"),
        "best_record_type": result.get("best_record_type"),
        "best_ari": result.get("best_ari"),
        "total_runtime_sec": result.get("total_runtime_sec"),
        "fatal_error": result.get("fatal_error"),
    }


def write_run_summaries(
    out_dir: Path,
    *,
    results: List[Dict[str, Any]],
    run_config: Dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    rows = [_records_row(r) for r in results]
    df = pd.DataFrame(rows)
    write_csv_atomic(out_dir / "run_summary.csv", df)

    failed = [r for r in results if r.get("fatal_error")]
    failed_df = pd.DataFrame([_records_row(r) for r in failed]) if failed else pd.DataFrame()
    write_csv_atomic(out_dir / "failed_datasets.csv", failed_df)

    def _all_skipped(r: Dict[str, Any]) -> bool:
        st = r.get("record_statuses", {}) or {}
        return bool(st) and all(v == "skipped" for v in st.values())

    skipped = [r for r in results if _all_skipped(r)]
    skipped_df = pd.DataFrame([_records_row(r) for r in skipped]) if skipped else pd.DataFrame()
    write_csv_atomic(out_dir / "skipped_datasets.csv", skipped_df)

    write_json_atomic(out_dir / "run_config.json", run_config)

    n = len(results)
    n_fatal = len(failed)
    n_skipped = len(skipped)
    aris = [r.get("best_ari") for r in results
            if r.get("best_ari") is not None and not r.get("fatal_error")]
    mean_ari = (sum(aris) / len(aris)) if aris else None
    md = [
        "# AutoML4Clust run summary",
        "",
        f"- Subfamily: **{run_config.get('subfamily_name')}**",
        f"- Split id: **{run_config.get('split_id')}**",
        f"- Datasets processed: **{n}** (fatal={n_fatal}, all-skipped={n_skipped})",
        f"- Record types: {run_config.get('record_types')}",
        f"- Optimizer / Metric: {run_config.get('config', {}).get('optimizer')} / "
        f"{run_config.get('config', {}).get('metric')}",
        f"- Workers: {run_config.get('workers')}",
        f"- Mean best-ARI (successful datasets): {mean_ari}",
        f"- Total wall-clock: {run_config.get('wallclock_str')}",
        "",
        "Primary per-dataset results live inside each dataset folder under "
        "`experiments/split_XX/AutoML4Clust/`. This folder holds only the "
        "run-level rollup.",
        "",
        "## Files",
        "- `run_summary.csv` -- one row per dataset (statuses + best ARI)",
        "- `failed_datasets.csv` -- datasets with a fatal error",
        "- `skipped_datasets.csv` -- datasets where every record was skipped",
        "- `run_config.json` -- exact run configuration",
    ]
    write_text_atomic(out_dir / "README_RUN_SUMMARY.md", "\n".join(md) + "\n")


# --------------------------------------------------------------------------- main
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run AutoML4Clust over one subfamily and one split id.",
    )
    p.add_argument("--subfamily-root", required=True, type=str)
    p.add_argument("--split-assignment-csv", required=True, type=str)
    p.add_argument("--split-id", required=True, type=int)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--record-types", nargs="+", default=list(RECORD_TYPES),
                   choices=list(RECORD_TYPES))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--repo-root", required=True, type=str)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-datasets", type=int, default=None,
                   help="Process at most N datasets (debugging).")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover + filter only; do not run AutoML4Clust.")
    return p


def run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    subfamily_root = Path(args.subfamily_root).resolve()
    config = load_config(args.config)
    subfamily_name = subfamily_root.name

    print("=" * 70, flush=True)
    print("AutoML4Clust -- subfamily / split", flush=True)
    print("=" * 70, flush=True)
    print(f"Subfamily root: {subfamily_root}", flush=True)
    print(f"Split id:       {args.split_id}", flush=True)
    print(f"Record types:   {args.record_types}", flush=True)
    print(f"Workers:        {args.workers}", flush=True)
    print(f"Overwrite:      {args.overwrite}", flush=True)

    all_dirs = discover_dataset_dirs(subfamily_root)
    if not all_dirs:
        print(f"[error] no dataset folders (with data_data.csv) under {subfamily_root}",
              file=sys.stderr, flush=True)
        return 3

    assignments = load_split_assignments(args.split_assignment_csv)
    selected, true_k_map = filter_by_split(all_dirs, assignments, args.split_id)
    if args.limit_datasets:
        selected = selected[: args.limit_datasets]

    print(f"Datasets discovered: {len(all_dirs)} | selected for split "
          f"{args.split_id}: {len(selected)}", flush=True)
    if not selected:
        print("[info] no datasets assigned to this split; nothing to do.", flush=True)
        return 0

    if args.dry_run:
        print("[dry-run] would process:", flush=True)
        for d in selected:
            print(f"  {d.name} (true_k={true_k_map.get(d.name)})", flush=True)
        return 0

    work_args = [
        (str(d), int(args.split_id), config, list(args.record_types),
         str(repo_root), bool(args.overwrite), true_k_map.get(d.name))
        for d in selected
    ]

    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []
    best_ari_so_far: Optional[float] = None
    n_total = len(work_args)
    dataset_times: List[float] = []

    def _track(idx: int, result: Dict[str, Any]) -> None:
        nonlocal best_ari_so_far
        results.append(result)
        dataset_times.append(float(result.get("total_runtime_sec", 0.0) or 0.0))
        bari = result.get("best_ari")
        if bari is not None and (best_ari_so_far is None or bari > best_ari_so_far):
            best_ari_so_far = bari
        avg = sum(dataset_times) / len(dataset_times) if dataset_times else 0.0
        eta = avg * max(n_total - idx, 0)
        elapsed = time.perf_counter() - t_start
        tag = "FATAL " if result.get("fatal_error") else ""
        print(
            f"[dataset {idx:>4}/{n_total}] {tag}{result.get('dataset_id')} "
            f"({result.get('total_runtime_sec', 0.0):.1f}s) "
            f"statuses={result.get('record_statuses')}",
            flush=True,
        )
        print(
            f"  best_ari_so_far={best_ari_so_far} | "
            f"elapsed={_fmt_elapsed(elapsed)} | ETA={_fmt_elapsed(eta)}",
            flush=True,
        )

    if args.workers <= 1 or n_total <= 1:
        for idx, wa in enumerate(work_args, start=1):
            _track(idx, _worker_entry(wa))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            fut_to_dir = {pool.submit(_worker_entry, wa): wa[0] for wa in work_args}
            for idx, fut in enumerate(as_completed(fut_to_dir), start=1):
                try:
                    result = fut.result()
                except Exception as exc:  # pragma: no cover
                    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    dd = fut_to_dir[fut]
                    result = {
                        "dataset_id": Path(dd).name, "dataset_dir": dd,
                        "fatal_error": f"{type(exc).__name__}: {exc}", "traceback": tb,
                        "record_statuses": {}, "best_ari": None, "total_runtime_sec": 0.0,
                    }
                _track(idx, result)

    wallclock = time.perf_counter() - t_start
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = run_summary_dir(repo_root, args.split_id, subfamily_name, ts)
    run_config = {
        "method": "AutoML4Clust",
        "subfamily_root": str(subfamily_root),
        "subfamily_name": subfamily_name,
        "split_id": int(args.split_id),
        "split_assignment_csv": str(Path(args.split_assignment_csv).resolve()),
        "record_types": list(args.record_types),
        "workers": int(args.workers),
        "overwrite": bool(args.overwrite),
        "config_path": str(Path(args.config).resolve()),
        "config": config,
        "n_datasets": n_total,
        "wallclock_sec": wallclock,
        "wallclock_str": _fmt_elapsed(wallclock),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        write_run_summaries(out_dir, results=results, run_config=run_config)
    except Exception as exc:
        print(f"[warn] could not write run summaries: {exc}", flush=True)

    n_fatal = sum(1 for r in results if r.get("fatal_error"))
    print(flush=True)
    print("=" * 70, flush=True)
    print(f"AutoML4Clust subfamily/split complete: {n_total} datasets "
          f"(fatal={n_fatal})", flush=True)
    print(f"Best ARI overall: {best_ari_so_far}", flush=True)
    print(f"Wall-clock: {_fmt_elapsed(wallclock)}", flush=True)
    print(f"Run summary: {out_dir}", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        return run(args)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
