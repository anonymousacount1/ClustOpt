"""Stage 4C parallel execution driver — Phases A and B.

Parallelism is by DATASET, never within a dataset/view unit, so nothing about the
frozen scientific semantics changes: each worker builds the same physical groups
in the same order for its own datasets and writes disjoint store keys. Atomic
COMPLETE writes make concurrent workers safe, and resume means a worker that dies
costs only its in-flight unit.

Worker count is bounded by MEMORY, not by cores. Each worker loads the ClustOpt
utility/policy/reranker artifacts, the AutoClust predictors and the ML2DAC
classifiers plus the 1.5M-row warmstart repository -- about 2 GB resident. On a
16 GB machine that is roughly 4 workers, not 8. Datasets are handed out in chunks
so that artifact loading is paid once per worker rather than once per dataset.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
VIEW_IDS = ("x_only", "y_only", "xy_2d")


def _driver():
    spec = importlib.util.spec_from_file_location(
        "r4c", str(_ROOT / "runners" / "run_stage4c_benchmark.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------- workers
def _worker_phase_a(payload: Tuple[str, List[Dict[str, Any]], Any]) -> Dict[str, Any]:
    run_id, entries, budget = payload
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "models"))
    sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.INFO)
    D = _driver()
    from Independent_Domain_Benchmark.execution import store as ST
    from Independent_Domain_Benchmark.execution.common_runner import (
        AutoClustExecutor, BenchmarkRunner, ClustOptExecutor, ML2DACExecutor)
    from Independent_Domain_Benchmark.execution.dataset_view import build_dataset_view

    store = ST.ResultStore(RESULTS, run_id)
    execs = {"AutoClust": AutoClustExecutor(), "ML2DAC": ML2DACExecutor(),
             "ClustOpt": ClustOptExecutor()}
    runner = BenchmarkRunner(store=store, budget=budget)
    done, failures = [], []
    for e in entries:
        t0 = time.perf_counter()
        try:
            X2, _y = D.materialise(e)
            for v in VIEW_IDS:
                view = build_dataset_view(dataset_id=e["dataset_id"],
                                          source=e["source"], view_id=v, X_full=X2)
                runner.run_view(view, execs)
            done.append({"dataset_id": e["dataset_id"],
                         "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as exc:  # noqa: BLE001
            failures.append({"dataset_id": e["dataset_id"],
                             "error": "%s: %s" % (type(exc).__name__, str(exc)[:220])})
    return {"pid": os.getpid(), "done": done, "failures": failures,
            "counters": dict(runner.counters)}


def _worker_phase_b(payload: Tuple[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    run_id, entries = payload
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "models"))
    sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.INFO)
    D = _driver()
    from Independent_Domain_Benchmark.analysis import fixed32_track as F32
    from Independent_Domain_Benchmark.execution import store as ST
    from Independent_Domain_Benchmark.execution.common_runner import apply_thread_controls
    from Independent_Domain_Benchmark.execution.dataset_view import build_dataset_view
    from Independent_Domain_Benchmark.execution import hashing as H

    apply_thread_controls()
    store = ST.ResultStore(RESULTS, run_id)
    registry = json.loads((_ROOT / "configs" / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    reg_hash = H.canonical_text_hash(
        (_ROOT / "configs" / "metric_analysis_registry.json").read_bytes())
    banks, evals, failures = 0, 0, []
    for e in entries:
        try:
            X2, _y = D.materialise(e)
        except Exception as exc:  # noqa: BLE001
            failures.append({"dataset_id": e["dataset_id"],
                             "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
            continue
        for v in VIEW_IDS:
            key = "%s__%s" % (e["dataset_id"], v)
            view = build_dataset_view(dataset_id=e["dataset_id"],
                                      source=e["source"], view_id=v, X_full=X2)
            prov = {"track": "fixed32", "dataset_id": e["dataset_id"], "view_id": v,
                    "fixed32_version": F32.FIXED32_VERSION,
                    "fixed32_config_hash": F32.fixed32_config_hash(v),
                    "metric_registry_hash": reg_hash,
                    "decision_hash": view.decision_hash, "full_hash": view.full_hash,
                    "sample_identity_hash": view.sample_identity_hash}
            if store.try_reuse(ST.Layer.PHYSICAL, "fixed32__" + key, prov).reusable \
                    and store.try_reuse(ST.Layer.EVALUATION, "metrics__" + key,
                                        prov).reusable:
                banks += 1
                continue
            try:
                t0 = time.perf_counter()
                bank = F32.build_fixed32_bank(view)
                bank_secs = time.perf_counter() - t0
                store.complete(
                    ST.Layer.PHYSICAL, "fixed32__" + key, provenance=prov,
                    content={"bank_id": bank.bank_id,
                             "candidate_count": bank.candidate_count,
                             "valid_candidates": len(bank.valid_indices()),
                             "content_hash": bank.content_hash(),
                             "runtime_seconds": round(bank_secs, 3),
                             "candidates": [
                                 {"candidate_index": c.candidate_index,
                                  "candidate_id": c.candidate_id,
                                  "algorithm": c.algorithm,
                                  "configuration": dict(c.configuration),
                                  "status": c.status, "failure_kind": c.failure_kind,
                                  "raw_labels_hash": c.raw_labels_hash,
                                  "canonical_partition_hash": c.canonical_partition_hash}
                                 for c in bank.candidates]})
                t1 = time.perf_counter()
                rows = F32.evaluate_bank_metrics(bank, view, registry)
                store.complete(
                    ST.Layer.EVALUATION, "metrics__" + key, provenance=prov,
                    content={"bank_id": bank.bank_id, "n_rows": len(rows),
                             "runtime_seconds": round(time.perf_counter() - t1, 3),
                             "rows": rows})
                # labels are needed for Phase C; persist them next to the bank
                np_dir = RESULTS / run_id / "labels"
                np_dir.mkdir(parents=True, exist_ok=True)
                import numpy as np
                np.savez_compressed(
                    np_dir / ("fixed32__%s.npz" % key),
                    **{"c%d" % c.candidate_index: np.asarray(c.labels)
                       for c in bank.candidates if c.labels is not None})
                banks += 1
                evals += len(rows)
            except Exception as exc:  # noqa: BLE001
                failures.append({"dataset_id": e["dataset_id"], "view": v,
                                 "error": "%s: %s" % (type(exc).__name__,
                                                      str(exc)[:200])})
    return {"pid": os.getpid(), "banks": banks, "metric_rows": evals,
            "failures": failures}


# ----------------------------------------------------------------- main
def chunk(items: List[Any], n: int) -> List[List[Any]]:
    """Round-robin so heavy datasets spread across workers."""
    out: List[List[Any]] = [[] for _ in range(n)]
    for i, it in enumerate(items):
        out[i % n].append(it)
    return [c for c in out if c]


def main(argv=None) -> int:
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["A", "B"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--limit-datasets", type=int, default=0)
    ap.add_argument("--dataset-ids", default="",
                    help="comma-separated dataset ids to process, in the given "
                         "order. Used to work a disjoint slice of the corpus in "
                         "a second pinned process; the store is atomic and "
                         "try_reuse skips anything already COMPLETE, so an "
                         "overlap costs time but cannot corrupt a record.")
    ap.add_argument("--run-id", default="",
                    help="pin this phase to an EXISTING run namespace. "
                         "The derived run id is a function of git_commit, "
                         "so committing code between phases silently opens "
                         "a new namespace and splits one benchmark across "
                         "two run directories. Pin it to keep phases joined.")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "models"))
    D = _driver()
    checks = D.preflight()
    derived_run_id = D.run_id_for(checks)
    run_id = args.run_id or derived_run_id
    if args.run_id and args.run_id != derived_run_id:
        # Legitimate only because preflight has already verified every
        # frozen SCIENTIFIC identity against its expected value; the only
        # remaining input to the derived id is git_commit.
        if not (RESULTS / args.run_id).is_dir():
            raise SystemExit("pinned --run-id %s does not exist; refusing "
                             "to create a namespace by typo" % args.run_id)
        print("  run id PINNED to %s (derived %s; differs by git_commit "
              "only -- all frozen scientific identities verified by "
              "preflight)" % (args.run_id, derived_run_id), flush=True)
    entries = [e for e in D._manifest()["datasets"] if e.get("accepted", True)]
    if args.dataset_ids:
        want = [d.strip() for d in args.dataset_ids.split(",") if d.strip()]
        by_id = {e["dataset_id"]: e for e in entries}
        missing = [d for d in want if d not in by_id]
        if missing:
            raise SystemExit("unknown dataset ids: %s" % ", ".join(missing))
        entries = [by_id[d] for d in want]
    if args.limit_datasets:
        entries = entries[:args.limit_datasets]
    budget = args.budget or None

    print("Stage 4C phase %s | run_id %s | %d datasets | %d workers"
          % (args.phase, run_id, len(entries), args.workers), flush=True)

    # The run namespace must say which environment produced it. Layer.RUN was
    # reserved for exactly this and nothing ever wrote to it, so finished runs
    # carried no record of their interpreter, dependency versions or thread
    # pinning. Written once per phase invocation, before any execution.
    from Independent_Domain_Benchmark.execution import manifest as MF
    from Independent_Domain_Benchmark.execution import store as ST0
    from Independent_Domain_Benchmark.execution.common_runner import (
        apply_thread_controls as _atc)
    _store = ST0.ResultStore(RESULTS, run_id)
    _key = "manifest__phase_%s" % args.phase.lower()
    _man = MF.build_manifest(run_id, thread_controls=_atc(), extra={
        "phase": args.phase, "workers": args.workers,
        "datasets": len(entries), "derived_run_id": derived_run_id,
        "run_id_pinned": bool(args.run_id),
        "captured_at": "phase_start_before_execution"})
    _store.complete(ST0.Layer.RUN, _key, provenance={
        "phase": args.phase, "run_id": run_id,
        "manifest_version": _man["manifest_version"],
        "git_commit": _man["git_commit"]}, content=_man)
    print("  environment manifest %s -> run/%s"
          % (_man["manifest_hash"], _key), flush=True)
    chunks = chunk(entries, args.workers)
    fn = _worker_phase_a if args.phase == "A" else _worker_phase_b
    payloads = [((run_id, c, budget) if args.phase == "A" else (run_id, c))
                for c in chunks]

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(chunks)) as pool:
        results = pool.map(fn, payloads)
    secs = time.perf_counter() - t0

    summary = {"phase": args.phase, "run_id": run_id,
               "derived_run_id": derived_run_id,
               "run_id_pinned": bool(args.run_id),
               "workers": len(chunks),
               "datasets": len(entries), "wall_clock_seconds": round(secs, 1),
               "worker_results": results,
               "failures": [f for r in results for f in r["failures"]]}
    out = RESULTS / run_id / ("phase_%s_parallel_summary.json" % args.phase.lower())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("  phase %s finished in %.1fs | failures %d"
          % (args.phase, secs, len(summary["failures"])), flush=True)
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
