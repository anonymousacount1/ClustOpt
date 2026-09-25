"""Stage 2A-2 orchestrator: OOF policy replay across Splits 2-16.

Execution contract, mirroring the established Stage-1 runners:

    for split k in 2..16:                       # SERIAL
        load the split's Stage-2A-1 OOF utilities once
        for family -> subfamily in canonical repository order:   # SERIAL
            run the subfamily's split-k datasets with 8 worker PROCESSES
            each worker capped at 1 numerical compute thread
            wait, persist progress, report, then advance

Resumable at (dataset, policy, view) granularity: a unit is skipped only when its
artifact exists, is marked success, and carries this run's experiment version,
policy-semantics hash and held-out split.

ZERO clustering: only the persisted fixed-32 pools are scored.

Run from the repo root::

    .venv_clustopt/Scripts/python.exe -m models.Clustering_Repository_Builder\
.experiments.experiment_execution.run_oof_policy20_all_splits --repo-root .
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

configure_process(thread_limit=1)

import pandas as pd  # noqa: E402

from .oof_policy_replay import (  # noqa: E402
    DEV_SPLITS, EXECUTION_MODE, EXPERIMENT_VERSION, SPLIT_DIR_TEMPLATE, VIEW_IDS,
    build_policy_set, load_oof_index, policy_semantics_hash, replay_dataset,
)
from .output_writer import _ext, ensure_dir, read_json, write_json_atomic  # noqa: E402

OOF_ROOT_REL = "results_analysis/clustering_repository/stage2/oof_utility_splits2_16"
OUT_REL = ("results_analysis/clustering_repository/stage2/"
           "offline_oof_policy20_splits2_16")

_WORKER: Dict[str, Any] = {}


def _init_worker(payload: Dict[str, Any]) -> None:
    """Runs once per worker process: cap threads, hold the split's OOF index."""
    configure_process(thread_limit=1)
    _WORKER.update(payload)


def _run_one_dataset(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    configure_process(thread_limit=1)
    return replay_dataset(
        dataset_dir=Path(task["dataset_dir"]), dataset_id=task["dataset_id"],
        split_id=task["split_id"], family=task["family"],
        subfamily=task["subfamily"], policies=_WORKER["policies"],
        oof=_WORKER["oof"], semantics_hash=_WORKER["semantics_hash"],
        source_commit=_WORKER["source_commit"], overwrite=_WORKER["overwrite"])


def _metric_names(repo: Path) -> List[str]:
    schema = read_json(repo / "results_analysis/metric_utility_knn/phase_e1/models"
                       / "utility_metric_schema.json") or {}
    return list(schema["metric_names"])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--splits", type=int, nargs="+", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-subfamilies", type=int, default=None)
    p.add_argument("--limit-datasets", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    oof_root = repo / OOF_ROOT_REL
    out_root = ensure_dir(repo / OUT_REL)
    analyzed = repo / "results_analysis/clustering_repository/analyzed_data"
    split_csv = analyzed / "experiment_splits/dataset_split_assignments.csv"

    policies = build_policy_set()
    shash = policy_semantics_hash(policies)
    metric_names = _metric_names(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()

    # Frozen policy manifest -- written before any execution.
    pm = pd.DataFrame([x.to_record() for x in policies])
    pm.to_csv(_ext(out_root / "policy_manifest.csv"), index=False)

    assign = pd.read_csv(_ext(split_csv))
    splits = args.splits or list(DEV_SPLITS)
    if 1 in splits:
        raise SystemExit("Split 1 must never be processed in Stage 2A-2")

    print("=" * 78, flush=True)
    print("STAGE 2A-2  --  OFFLINE OOF POLICY REPLAY (20 learned deployable)", flush=True)
    print("=" * 78, flush=True)
    print(f"  policies {len(policies)} (10 MLP + 10 KNN) | semantics {shash}", flush=True)
    print(f"  splits {splits[0]}..{splits[-1]} | workers {args.workers} x 1 thread", flush=True)
    print(f"  experiment dir <dataset>/experiments/"
          f"{SPLIT_DIR_TEMPLATE.format(split=0)[:-2]}XX", flush=True)

    progress_path = out_root / "execution_progress.csv"
    prog_rows: List[Dict[str, Any]] = []
    if progress_path.exists() and not args.overwrite:
        try:
            prog_rows = pd.read_csv(_ext(progress_path)).to_dict("records")
        except Exception:                                             # noqa: BLE001
            prog_rows = []
    done_keys = {(r["split_id"], r["family"], r["subfamily"]) for r in prog_rows}

    t_all = time.time()
    cum_ds = cum_rec = cum_fail = 0
    for split_id in splits:
        sub = assign[assign["split_id"].astype("Int64") == int(split_id)]
        if args.limit_datasets:
            sub = sub.head(int(args.limit_datasets))
        if sub.empty:
            continue
        print(f"\n{'='*60}\nSPLIT {split_id:02d}  --  {len(sub)} datasets\n{'='*60}",
              flush=True)
        oof = load_oof_index(oof_root, split_id, metric_names)
        payload = {"policies": policies, "oof": oof, "semantics_hash": shash,
                   "source_commit": commit, "overwrite": bool(args.overwrite)}

        groups = list(sub.groupby(["family_id", "subfamily_id"], sort=True))
        if args.limit_subfamilies:
            groups = groups[: int(args.limit_subfamilies)]
        s_ds = s_rec = s_fail = 0
        t_split = time.time()

        # One pool per SPLIT, not per subfamily: the worker initialiser ships the
        # split's whole OOF index, so rebuilding the pool 86 times per split would
        # re-pickle it 86 x 8 times. Subfamilies are still executed strictly in
        # order, one at a time, so the family/subfamily execution controls and the
        # per-subfamily barrier are unchanged.
        pool = None
        if args.workers > 1:
            pool = ProcessPoolExecutor(max_workers=int(args.workers),
                                       initializer=_init_worker, initargs=(payload,))
        else:
            _init_worker(payload)
        try:
          for gi, ((fam, subfam), grp) in enumerate(groups, 1):
            if (split_id, fam, subfam) in done_keys and not args.overwrite:
                continue
            tasks = [{"dataset_dir": r["dataset_dir"], "dataset_id": r["dataset_id"],
                      "split_id": int(split_id), "family": fam, "subfamily": subfam}
                     for _, r in grp.iterrows()]
            t0 = time.time()
            rows: List[Dict[str, Any]] = []
            if pool is not None and len(tasks) > 1:
                for res in pool.map(_run_one_dataset, tasks):
                    rows.extend(res)
            else:
                if pool is None:
                    for t in tasks:
                        rows.extend(_run_one_dataset(t))
                else:
                    for res in pool.map(_run_one_dataset, tasks):
                        rows.extend(res)

            frame = pd.DataFrame(rows)
            n_fail = int((frame["status"] != "success").sum())
            shard = ensure_dir(out_root / "_shards") / \
                f"split{split_id:02d}__{fam}__{subfam}.csv.gz"
            frame.to_csv(_ext(shard), index=False, compression="gzip")

            s_ds += len(tasks); s_rec += len(frame); s_fail += n_fail
            cum_ds += len(tasks); cum_rec += len(frame); cum_fail += n_fail
            prog_rows.append({
                "split_id": split_id, "family": fam, "subfamily": subfam,
                "n_datasets": len(tasks), "n_records": len(frame),
                "n_failed": n_fail, "runtime_sec": round(time.time() - t0, 2),
                "shard": shard.name,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            pd.DataFrame(prog_rows).to_csv(_ext(progress_path), index=False)

            print("=" * 60, flush=True)
            print("STAGE 2A-2 -- OFFLINE OOF POLICY REPLAY", flush=True)
            print(f"\nSplit: {split_id:02d}\nFamily: {fam}\nSubfamily: {subfam}\n",
                  flush=True)
            print(f"Datasets complete:  {len(tasks)}", flush=True)
            print(f"Policies:           {len(policies)}", flush=True)
            print(f"Views:              {len(VIEW_IDS)}", flush=True)
            print(f"Replay records:     {len(frame)}", flush=True)
            print(f"Failures:           {n_fail}\n", flush=True)
            print(f"Workers:            {args.workers} processes", flush=True)
            print("                    1 numerical compute thread / worker\n",
                  flush=True)
            print(f"Cumulative split:   {s_ds}/{len(sub)} datasets  "
                  f"{s_rec} records  {(time.time()-t_split)/60:.1f} min", flush=True)
            print("=" * 60, flush=True)
            if args.preflight and gi >= 1:
                break
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        print("=" * 60, flush=True)
        print(f"SPLIT {split_id:02d} COMPLETE\n", flush=True)
        print(f"Datasets:                          {s_ds}", flush=True)
        print(f"Expected dataset/view/policy rows: {s_ds * 3 * len(policies)}",
              flush=True)
        print(f"Successful:                        {s_rec - s_fail}", flush=True)
        print(f"Failed:                            {s_fail}", flush=True)
        print(f"Integrity:                         "
              f"{'PASS' if s_fail == 0 else 'FAIL'}", flush=True)
        print(f"Elapsed:                           "
              f"{(time.time()-t_split)/60:.1f} min\n", flush=True)
        print(f"Cumulative splits:                 "
              f"{splits.index(split_id)+1} / {len(splits)}", flush=True)
        print(f"Cumulative datasets:               {cum_ds}", flush=True)
        print("=" * 60, flush=True)
        if args.preflight:
            break

    write_json_atomic(out_root / "experiment_manifest.json", {
        "stage": "Stage 2A-2 -- offline OOF policy replay (20 learned deployable)",
        "execution_mode": EXECUTION_MODE,
        "experiment_version": EXPERIMENT_VERSION,
        "policy_semantics_hash": shash,
        "n_policies": len(policies),
        "policy_ids": [x.policy_id for x in policies],
        "splits": splits, "workers": int(args.workers), "threads_per_worker": 1,
        "oof_source_root": OOF_ROOT_REL,
        "candidate_pool": "utility/<view_id>/clustopt_results.csv (fixed 32)",
        "clustering_executed": False,
        "per_dataset_experiment_dir": SPLIT_DIR_TEMPLATE,
        "source_commit": commit,
        "environment_lock": "environment/stage1_clustopt_requirements.lock.txt",
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"\n[stage2a2] {cum_ds} datasets | {cum_rec} records | {cum_fail} failures "
          f"| {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0 if cum_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
