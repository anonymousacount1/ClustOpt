"""Stage 2A-2 one-off migration: six-file layout -> consolidated ``result.json``.

Folds every already-written (dataset, policy, view) unit into a single
authoritative record, then removes the six legacy files. Write-then-verify order
means an interruption can never lose a completed unit; re-running is safe.

Scientific semantics, OOF bindings, policy computation and barriers are untouched
-- only the on-disk artifact layout changes.

Run::

    .venv_clustopt/Scripts/python.exe -m models.Clustering_Repository_Builder\
.experiments.experiment_execution.migrate_oof_policy20_results --repo-root .
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

configure_process(thread_limit=1)

import pandas as pd  # noqa: E402

from .oof_policy_replay import (  # noqa: E402
    DEV_SPLITS, SPLIT_DIR_TEMPLATE, build_policy_set, migrate_dataset_tree,
)
from .output_writer import _ext, path_exists  # noqa: E402

_P: List[Any] = []


def _init() -> None:
    configure_process(thread_limit=1)
    if not _P:
        _P.extend(build_policy_set())


def _one(task: Dict[str, Any]) -> Dict[str, int]:
    configure_process(thread_limit=1)
    if not _P:
        _P.extend(build_policy_set())
    return migrate_dataset_tree(Path(task["dataset_dir"]), int(task["split_id"]), _P)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    analyzed = repo / "results_analysis/clustering_repository/analyzed_data"
    assign = pd.read_csv(
        _ext(analyzed / "experiment_splits/dataset_split_assignments.csv"))

    tasks = []
    for _, r in assign.iterrows():
        s = int(r["split_id"])
        if s not in DEV_SPLITS:
            continue
        tree = (Path(r["dataset_dir"]) / "experiments"
                / SPLIT_DIR_TEMPLATE.format(split=s))
        if path_exists(tree):
            tasks.append({"dataset_dir": r["dataset_dir"], "split_id": s})

    print("=" * 78)
    print("STAGE 2A-2  --  MIGRATE TO CONSOLIDATED result.json")
    print("=" * 78)
    print(f"  dataset trees found: {len(tasks)}")
    if not tasks:
        print("  nothing to migrate.")
        return 0

    total: Counter = Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=int(args.workers), initializer=_init) as ex:
        for i, res in enumerate(ex.map(_one, tasks), 1):
            total.update(res)
            if i % 200 == 0:
                print(f"    {i}/{len(tasks)} datasets ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"\n  units: {dict(total)}")
    print(f"  elapsed {(time.time()-t0)/60:.1f} min")
    if total.get("incomplete"):
        print(f"  [WARN] {total['incomplete']} incomplete legacy units left in place; "
              f"they will simply be rebuilt by the replay runner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
