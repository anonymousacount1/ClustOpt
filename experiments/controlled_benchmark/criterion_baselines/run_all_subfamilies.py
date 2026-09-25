"""Seeded criterion-baseline sweep (internal label E1; paper Tables 1, 16 and 17). Included for transparency; it expects the regenerated controlled repository.

E1 orchestrator: run the five conditioning arms over all Split-1 subfamilies.

Execution is delegated to the established, unmodified
``run_subfamily_experiments.run``, exactly as the Stage-1 online sweep does, so
dataset discovery, the worker pool, method_runner, ClustOpt/Optuna execution,
per-candidate persistence, resume and Best-View all come from the existing
framework.

Isolation: results land in ``experiments/split_01_e1_conditioning/`` via the
established ``split_dir_suffix`` mechanism. The historical ``split_01``,
``split_01_online_fixed50`` and ``split_01_offline_fixed32`` trees are never
touched and cannot be confused with these.

The arm registry is injected into ``config.METHOD_SETS`` at runtime by this
module, so no tracked registry file is edited.

Run from the repo root::

    python experiments/controlled_benchmark/criterion_baselines/run_all_subfamilies.py \
        --repo-root src --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (  # noqa: E402
    configure_process,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution import (  # noqa: E402
    config as exec_config,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (  # noqa: E402
    ExperimentExecutionConfig, MethodSpec, VIEW_IDS,
)

configure_process()

SPLIT_DIR_SUFFIX = "e1_conditioning"
E1_ARMS = ("e1_c4_seed", "e1_meanprof", "e1_core3", "e1_best1", "e1_random")
E1_GROUP = {"e1_c4_seed": "regressor", "e1_meanprof": "baseline",
            "e1_core3": "baseline", "e1_best1": "baseline",
            "e1_random": "baseline"}

# ---- runtime registration: additive, no tracked registry file is edited ----
METHODS_E1 = tuple(
    MethodSpec(a, E1_GROUP[a], f"e1_conditioning/{a}") for a in E1_ARMS)
exec_config.METHOD_SETS["e1_conditioning"] = METHODS_E1

from models.Clustering_Repository_Builder.experiments.experiment_execution.method_runner import (  # noqa: E402
    is_run_complete, run_output_dir,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    ensure_dir, path_exists, read_json, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.run_offline_2x3_all_subfamilies import (  # noqa: E402
    _split1_dataset_dirs, discover_subfamilies,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.run_subfamily_experiments import (  # noqa: E402
    run as run_subfamily_online,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.summary_writer import (  # noqa: E402
    summary_dir,
)

REQUIRED_FILES = (
    "clustopt_results.csv", "best_result.json", "external_metrics.json",
    "selected_metrics.json", "timing_summary.json", "run_log.json",
    "status.json", "config_snapshot.json",
)


def run_is_complete(dataset_dir: Path, split_dir: str, method: str, view: str) -> bool:
    out = run_output_dir(dataset_dir, split_dir, method, view)
    if not is_run_complete(out):
        return False
    return all(path_exists(out / f) for f in REQUIRED_FILES)


def subfamily_is_complete(cfg, dataset_dirs, methods):
    split_dir = cfg.split_dir_name()
    missing: List[str] = []
    for d in dataset_dirs:
        for view in VIEW_IDS:
            for m in methods:
                if not run_is_complete(Path(d), split_dir, m, view):
                    missing.append(f"{Path(d).name}/{view}/{m}")
    sd = summary_dir(cfg)
    summaries_ok = path_exists(sd / "execution_summary.csv")
    return (not missing and summaries_ok), {
        "expected_runs": len(dataset_dirs) * len(VIEW_IDS) * len(methods),
        "missing_runs": len(missing), "missing_examples": missing[:5],
        "summaries_present": summaries_ok, "datasets": len(dataset_dirs),
    }


def _progress_path(name: str = "e1_progress.json") -> Path:
    return HERE / "run_state" / name


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--split-id", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--parallelism", default="record",
                   choices=("dataset", "record"))
    p.add_argument("--limit-subfamilies", type=int, default=None)
    p.add_argument("--limit-datasets", type=int, default=None)
    p.add_argument("--only-subfamily", default=None)
    p.add_argument("--order", default="stratified",
                   choices=("canonical", "stratified"),
                   help="stratified = deterministic round-robin across the 12 "
                        "families, so any interrupted prefix stays representative")
    p.add_argument("--arms", default=",".join(E1_ARMS))
    p.add_argument("--split-dir-suffix", default=SPLIT_DIR_SUFFIX)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dataset-allowlist", default=None,
                   help="CSV with a dataset_id column. Restricts execution to "
                        "those datasets WITHOUT changing the output location, so "
                        "the runs remain reusable by the later full sweep.")
    p.add_argument("--progress-name", default="e1_progress.json",
                   help="progress file under run_state/ (use a distinct name for "
                        "a restricted run so it cannot be mistaken for full-sweep "
                        "progress)")
    args = p.parse_args(argv)

    # ---- optional dataset allowlist -------------------------------------
    # Applied by wrapping the framework's own split filter, so dataset
    # DISCOVERY, output paths, configs, seeds and the completion barrier are
    # all unchanged; only the set of datasets scheduled shrinks.
    allow: Optional[set] = None
    if args.dataset_allowlist:
        import csv as _csv
        with open(args.dataset_allowlist, "r", encoding="utf-8") as fh:
            allow = {row["dataset_id"].strip()
                     for row in _csv.DictReader(fh) if row.get("dataset_id")}
        from models.Clustering_Repository_Builder.experiments.experiment_execution import (
            split_filter as _sf, run_subfamily_experiments as _rse,
        )
        _orig = _sf.filter_dataset_dirs_by_split

        def _filtered(dirs, assignments, split_id, _o=_orig, _a=allow):
            return [d for d in _o(dirs, assignments, split_id)
                    if Path(d).name in _a]

        _sf.filter_dataset_dirs_by_split = _filtered
        _rse.filter_dataset_dirs_by_split = _filtered
        print("[allowlist] restricted to %d datasets from %s"
              % (len(allow), args.dataset_allowlist), flush=True)

    repo = Path(args.repo_root).resolve()
    analyzed = repo / "results_analysis" / "clustering_repository" / "analyzed_data"
    # the canonical split table used by every historical sweep
    splits = analyzed / "experiment_splits" / "dataset_split_assignments.csv"
    if not path_exists(splits):
        print("[error] %s not found" % splits, file=sys.stderr)
        return 2

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for a in arms:
        assert a in E1_ARMS, "unknown arm %s" % a

    subfams = discover_subfamilies(analyzed)
    if args.order == "stratified":
        # Deterministic round-robin across families, so that ANY prefix of the
        # sweep spans all 12 structural families roughly evenly. This matters
        # because the run is long and resumable: a partial result under the
        # canonical (family-alphabetical) order would be one or two families
        # only, and therefore unrepresentative. Ordering cannot affect any
        # scientific value -- every (dataset, view, arm) run is independent and
        # its seed depends only on (dataset_id, view_id).
        by_fam: Dict[str, List[Path]] = {}
        for s in subfams:
            by_fam.setdefault(s.parent.parent.name, []).append(s)
        fams = sorted(by_fam)
        interleaved: List[Path] = []
        i = 0
        while any(len(by_fam[f]) > i for f in fams):
            for f in fams:
                if len(by_fam[f]) > i:
                    interleaved.append(by_fam[f][i])
            i += 1
        subfams = interleaved
    if args.only_subfamily:
        subfams = [s for s in subfams if s.name == args.only_subfamily]
    if args.limit_subfamilies:
        subfams = subfams[: args.limit_subfamilies]

    ensure_dir(HERE / "run_state")
    prog_path = _progress_path(args.progress_name)
    progress: Dict[str, Any] = read_json(prog_path) or {
        "run_id": "e1_%s" % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "split_dir_suffix": args.split_dir_suffix,
        "dataset_allowlist": args.dataset_allowlist,
        "n_allowlisted": (len(allow) if allow else None),
        "arms": list(arms), "completed": {}, "subfamilies": {},
    }
    progress.setdefault("completed", {})
    progress.setdefault("subfamilies", {})

    t0 = time.time()
    total = len(subfams)
    for i, sub in enumerate(subfams, start=1):
        key = f"{sub.parent.parent.name}/{sub.name}"
        cfg = ExperimentExecutionConfig(
            repo_root=repo, subfamily_dir=sub, split_assignments=splits,
            split_id=args.split_id, max_workers=args.workers,
            resume=not args.overwrite, overwrite=args.overwrite,
            method_set="e1_conditioning", methods=arms,
            parallelism=args.parallelism,
            split_dir_suffix=args.split_dir_suffix,
            limit_datasets=args.limit_datasets,
        )
        dirs = _split1_dataset_dirs(cfg)
        if not dirs:
            progress["completed"][key] = True
            continue

        done, status = subfamily_is_complete(cfg, dirs, arms)
        if done and not args.overwrite:
            progress["completed"][key] = True
            print(f"[{i}/{total}] SKIP (complete) {key}  datasets={len(dirs)}",
                  flush=True)
            write_json_atomic(prog_path, progress)
            continue

        t1 = time.time()
        rc = run_subfamily_online(cfg)
        done, status = subfamily_is_complete(cfg, dirs, arms)
        progress["completed"][key] = bool(done)
        progress["subfamilies"][key] = {
            "rc": rc, "datasets": len(dirs), "barrier_complete": bool(done),
            "missing_runs": status["missing_runs"],
            "runtime_sec": round(time.time() - t1, 2),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(prog_path, progress)
        n_done = sum(1 for v in progress["completed"].values() if v)
        print(f"[{i}/{total}] {key}  datasets={len(dirs)}  rc={rc}  "
              f"complete={done}  missing={status['missing_runs']}  "
              f"{(time.time()-t1)/60:.1f} min  |  cumulative {n_done}/{total} "
              f"subfamilies, elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    n_done = sum(1 for v in progress["completed"].values() if v)
    progress["finished_at"] = datetime.now(timezone.utc).isoformat()
    progress["total_runtime_sec"] = round(time.time() - t0, 2)
    write_json_atomic(prog_path, progress)
    print(f"E1 sweep: {n_done}/{total} subfamilies complete in "
          f"{(time.time()-t0)/3600:.2f} h", flush=True)
    return 0 if n_done == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
