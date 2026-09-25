"""Stage 2A-3B0 orchestrator: build the 105 pair-exclusion utility sources.

No Policy Predictor is trained. No clustering is run. Split 1 is never read.

Execution: pairs are processed serially by default. The MLP is the cost driver
(~3.7 min each on this machine), so ``--workers`` may run a small number of pair
trainings concurrently; the safe level on 4 physical cores is 2 processes x 2
compute threads. Resume is per pair and validates the whole manifest, never mere
file existence.

Run::

    .venv_clustopt/Scripts/python.exe -m models.ClustOpt_Policy_Predictor\
.nested_crossfit.run_nested_crossfit --repo-root . [--preflight] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

configure_process(thread_limit=2)

import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, read_json, write_json_atomic,
)

from ..data import feature_schema as FS  # noqa: E402
from ..data import target_schema as TS  # noqa: E402
from . import pair_utility_builder as PU  # noqa: E402

OUT_REL = "results_analysis/clustopt_policy_predictor/stage2a3b0_nested_crossfit"


def recipe_hash(repo: Path) -> str:
    """Identity of the frozen MLP/KNN recipes + policy order.

    A completed pair is only reusable if this matches, so a semantics change
    invalidates the cache rather than silently mixing recipes.
    """
    mlp_cfg = json.load(open(_ext(repo / PU.MLP_FROZEN_REL / "config.json"),
                             encoding="utf-8"))
    knn_meta = read_json(repo / PU.KNN_FROZEN_REL / "final_knn_metadata.json") or {}
    return PU._hash({
        "mlp_model": mlp_cfg["model"], "mlp_loss": mlp_cfg["loss"],
        "mlp_training": mlp_cfg["training"],
        "knn_config": knn_meta.get("config"),
        "policy_order_hash": TS.target_schema_hash(),
        "schema": PU.PAIR_SCHEMA_VERSION,
    })


def build_one_pair(repo: Path, root: Path, a: int, b: int, data,
                   metric_names, rhash: str, commit: str) -> Dict[str, Any]:
    t0 = time.time()
    mlp = PU.build_pair_mlp(repo, root, a, b, metric_names)
    knn = PU.build_pair_knn(repo, root, a, b, data, metric_names)
    gates = PU.pair_leakage_gates(a, b, mlp, knn)
    ok = bool((gates["status"] == "PASS").all())
    d = PU.pair_dir(root, a, b)
    gates.to_csv(_ext(d / "leakage_checks.csv"), index=False)
    man = {
        "schema_version": PU.PAIR_SCHEMA_VERSION,
        "excluded_pair": [a, b],
        "training_splits": PU.training_splits(a, b),
        "n_training_splits": 13,
        "recipe_hash": rhash,
        "mlp": mlp, "knn": knn,
        "leakage": "PASS" if ok else "FAIL",
        "status": "complete" if ok else "failed_gates",
        "runtime_sec": time.time() - t0,
        "source_commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(d / "manifest.json", man)
    return man


_W: Dict[str, Any] = {}


def _init_worker(payload: Dict[str, Any]) -> None:
    """One-time per worker: cap threads, load the shared read-only KNN data."""
    configure_process(thread_limit=int(payload["thread_limit"]))
    from models.metric_utility_knn.data_loader import load_unified_data
    _W["repo"] = Path(payload["repo"])
    _W["root"] = Path(payload["root"])
    _W["metric_names"] = payload["metric_names"]
    _W["rhash"] = payload["rhash"]
    _W["commit"] = payload["commit"]
    _W["data"] = load_unified_data()


def _build_pair_task(pair: List[int]) -> Dict[str, Any]:
    a, b = int(pair[0]), int(pair[1])
    configure_process(thread_limit=2)
    try:
        man = build_one_pair(_W["repo"], _W["root"], a, b, _W["data"],
                             _W["metric_names"], _W["rhash"], _W["commit"])
        return {"pair": [a, b], "ok": man["leakage"] == "PASS", "manifest": man}
    except Exception as exc:                                        # noqa: BLE001
        return {"pair": [a, b], "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent pair builds; 2 is the safe level on 4 cores")
    ap.add_argument("--thread-limit", type=int, default=2,
                    help="numerical compute threads per training process")
    ap.add_argument("--pairs", type=str, default=None,
                    help="comma list like 2-3,2-4")
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    root = ensure_dir(repo / OUT_REL)
    metric_names = FS.load_metric_names(repo)
    rhash = recipe_hash(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()

    pairs = PU.all_pairs()
    if args.pairs:
        want = {tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",")}
        pairs = [p for p in pairs if p in want]
    if args.preflight:
        pairs = pairs[:1]
    if args.limit:
        pairs = pairs[: int(args.limit)]

    print("=" * 78, flush=True)
    print("STAGE 2A-3B0  --  PAIR-EXCLUSION UTILITY SOURCES", flush=True)
    print("=" * 78, flush=True)
    print(f"  unordered pairs      : {len(PU.all_pairs())} (C(15,2))", flush=True)
    print(f"  this run             : {len(pairs)}", flush=True)
    print(f"  training population  : S \\ {{a,b}} = 13 splits", flush=True)
    print(f"  recipe hash          : {rhash[:16]}", flush=True)
    print(f"  Split 1              : never read", flush=True)

    # ---- fully idempotent resume: validate every pair, keep only the todo set --
    print("\n  scanning the 105 pair manifests ...", flush=True)
    todo, cached = [], 0
    for (a, b) in pairs:
        if PU.pair_is_complete(root, a, b, rhash):
            cached += 1
        else:
            todo.append([a, b])
    print(f"    valid/complete {cached} | to build {len(todo)}", flush=True)
    if not todo:
        print("  nothing to do -- all requested pairs are complete.", flush=True)

    # Resume-cycle bookkeeping, so the report can state how many invocations the
    # environment's process termination actually forced.
    state = read_json(root / "pair_run_state.json") or {}
    cycles = int(state.get("resume_cycles", 0)) + 1

    t_all = time.time()
    done = failed = 0
    fail_detail: List[Dict[str, Any]] = []
    if todo:
        payload = {"repo": str(repo), "root": str(root),
                   "metric_names": list(metric_names), "rhash": rhash,
                   "commit": commit, "thread_limit": int(args.thread_limit)}
        n_workers = max(1, int(args.workers))
        print(f"  building with {n_workers} concurrent trainings x "
              f"{args.thread_limit} compute threads", flush=True)
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(payload,)) as ex:
            for res in ex.map(_build_pair_task, todo):
                a, b = res["pair"]
                idx = PU.all_pairs().index((a, b)) + 1
                done += 1
                if not res["ok"]:
                    failed += 1
                    fail_detail.append(res)
                    print(f"  [FAIL] pair {{{a},{b}}}: "
                          f"{res.get('error', 'leakage gate')}", flush=True)
                    if "error" not in res:
                        print("  [STOP] leakage/schema assertion failed.",
                              flush=True)
                        return 1
                    continue
                man = res["manifest"]
                el = time.time() - t_all
                print("=" * 60, flush=True)
                print(f"STAGE 2A-3B0 -- PAIR EXCLUSION {idx} / 105\n", flush=True)
                print(f"Excluded pair:     {{{a}, {b}}}", flush=True)
                print(f"Training splits:   {man['training_splits']}", flush=True)
                print(f"MLP status:        trained "
                      f"({man['mlp']['epochs_run']} epochs, "
                      f"{man['mlp']['runtime_sec']/60:.1f} min)", flush=True)
                print(f"KNN status:        built "
                      f"({man['knn']['runtime_sec']:.1f}s)\n", flush=True)
                print("Predictions:", flush=True)
                print(f"  split {a:02d} rows     {man['mlp']['n_pred_a']:,} MLP / "
                      f"{man['knn']['n_pred_a']:,} KNN", flush=True)
                print(f"  split {b:02d} rows     {man['mlp']['n_pred_b']:,} MLP / "
                      f"{man['knn']['n_pred_b']:,} KNN\n", flush=True)
                print(f"Leakage:           {man['leakage']}\n", flush=True)
                print(f"Cumulative pairs:  {cached + done} / {len(pairs)}",
                      flush=True)
                print(f"Elapsed:           {el/60:.1f} min", flush=True)
                print("=" * 60, flush=True)

    n_complete = sum(1 for (a, b) in PU.all_pairs()
                     if PU.pair_is_complete(root, a, b, rhash))
    write_json_atomic(root / "pair_run_state.json", {
        "recipe_hash": rhash, "n_pairs_total": 105,
        "n_complete": n_complete,
        "cached_at_start_of_run": cached,
        "built_this_run": done - failed, "failed_this_run": failed,
        "failures": fail_detail,
        "resume_cycles": cycles,
        "cumulative_build_seconds": float(state.get("cumulative_build_seconds", 0.0))
        + (time.time() - t_all),
        "workers": int(args.workers), "thread_limit": int(args.thread_limit),
        "source_commit": commit,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"\n[stage2a3b0] {done - failed} built / {cached} cached / {failed} failed "
          f"| {n_complete}/105 complete | cycle {cycles} | "
          f"{(time.time()-t_all)/60:.1f} min", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
