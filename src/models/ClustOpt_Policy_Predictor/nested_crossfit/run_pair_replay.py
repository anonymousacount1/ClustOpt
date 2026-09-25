"""Stage 2A-3B0 phase 2: nested 20-policy replay over the pair-exclusion utilities.

Pivoted by TARGET SPLIT: each dataset/view's 32-candidate pool is loaded once and
scored against all 14 pair-utility vectors x 20 policies. Zero clustering.

Output: one compact row per (excluded_pair, target_split, dataset_id, view_id)
carrying the full 20-ARI + 20-regret vector and tie diagnostics -- 672,546 rows
representing 13,450,920 logical policy outcomes, without a single extra JSON file.

Resumable per target split. Run::

    .venv_clustopt/Scripts/python.exe -m models.ClustOpt_Policy_Predictor\
.nested_crossfit.run_pair_replay --repo-root . [--splits 2 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

configure_process(thread_limit=1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_json_atomic,
)

from ..data import feature_schema as FS  # noqa: E402
from ..data import target_schema as TS  # noqa: E402
from . import pair_policy_replay as PR  # noqa: E402
from .pair_utility_builder import SPLITS, VIEWS  # noqa: E402
from .run_nested_crossfit import OUT_REL  # noqa: E402

_W: Dict[str, Any] = {}


def _init(payload: Dict[str, Any]) -> None:
    configure_process(thread_limit=1)
    _W.update(payload)
    _W["specs"] = PR.policy_specs()


def _one_dataset(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    configure_process(thread_limit=1)
    mn = _W["metric_names"]
    rows: List[Dict[str, Any]] = []
    for view in VIEWS:
        util_by_pair: Dict[Any, Dict[str, Dict[str, float]]] = {}
        for pair, rec in task["utils"].items():
            if view not in rec:
                continue
            m, k = rec[view]
            util_by_pair[pair] = {
                "mlp": {name: float(v) for name, v in zip(mn, m)},
                "knn": {name: float(v) for name, v in zip(mn, k)},
            }
        if not util_by_pair:
            continue
        for r in PR.replay_rows(Path(task["dataset_dir"]), view, mn,
                                util_by_pair, _W["specs"]):
            r.update({"dataset_id": task["dataset_id"], "view_id": view,
                      "target_split": task["split_id"]})
            rows.append(r)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    root = ensure_dir(repo / OUT_REL)
    out = ensure_dir(root / "pair_replay")
    metric_names = FS.load_metric_names(repo)
    specs = PR.policy_specs()

    assign = pd.read_csv(
        _ext(repo / "results_analysis/clustering_repository/analyzed_data/"
             "experiment_splits/dataset_split_assignments.csv"),
        usecols=["dataset_id", "dataset_dir", "split_id"])

    print("=" * 78, flush=True)
    print("STAGE 2A-3B0  --  NESTED 20-POLICY REPLAY (pair-exclusion utilities)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  policies {len(specs)} | order hash {TS.target_schema_hash()[:16]}",
          flush=True)
    print("  pivoted by target split: each candidate pool loaded once", flush=True)

    splits = args.splits or list(SPLITS)
    t_all = time.time()
    total_rows = 0
    for si, t in enumerate(splits, 1):
        shard = out / f"replay_target_split_{t:02d}.csv.gz"
        if path_exists(shard):
            n = int(pd.read_csv(_ext(shard), usecols=["dataset_id"]).shape[0])
            total_rows += n
            print(f"[{si}/{len(splits)}] split {t:02d}: complete "
                  f"({n:,} rows), skipping", flush=True)
            continue

        sub = assign[assign["split_id"] == t]
        utils = PR.load_pair_utilities(root, t, metric_names)
        # index utilities per dataset/view for cheap per-task slicing
        mcols = [f"mlp__{m}" for m in metric_names]
        kcols = [f"knn__{m}" for m in metric_names]
        idx: Dict[str, Dict[Any, Dict[str, Any]]] = {}
        for pair, df in utils.items():
            for ds, vw, mv, kv in zip(df["dataset_id"], df["view_id"],
                                      df[mcols].to_numpy(np.float64),
                                      df[kcols].to_numpy(np.float64)):
                idx.setdefault(ds, {}).setdefault(pair, {})[vw] = (mv, kv)

        tasks = [{"dataset_id": r["dataset_id"], "dataset_dir": r["dataset_dir"],
                  "split_id": t, "utils": idx.get(r["dataset_id"], {})}
                 for _, r in sub.iterrows()]
        t0 = time.time()
        rows: List[Dict[str, Any]] = []
        payload = {"metric_names": list(metric_names)}
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=_init,
                                 initargs=(payload,)) as ex:
            for res in ex.map(_one_dataset, tasks, chunksize=4):
                rows.extend(res)
        df = pd.DataFrame(rows)
        df.to_csv(_ext(shard), index=False, compression="gzip")
        total_rows += len(df)
        exp = len(sub) * 3 * 14
        print(f"[{si}/{len(splits)}] split {t:02d}: {len(df):,} rows "
              f"(expect {exp:,}) | {len(sub):,} datasets x 3 views x 14 pairs "
              f"| {(time.time()-t0)/60:.1f} min", flush=True)
        if len(df) != exp:
            print(f"  [STOP] row-count mismatch for split {t}", flush=True)
            return 1

    write_json_atomic(root / "replay_state.json", {
        "total_rows": int(total_rows), "expected_rows": 48039 * 14,
        "logical_policy_outcomes": int(total_rows) * 20,
        "policy_order_hash": TS.target_schema_hash(),
        "clustering_executed": False,
    })
    print(f"\n[replay] {total_rows:,} rows "
          f"({total_rows*20:,} logical policy outcomes) in "
          f"{(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
