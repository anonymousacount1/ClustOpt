"""Stage 4C -- targeted recompute of the six metrics the v1 harness never executed.

Amendment ``stage4c_fixed32_metric_dispatch_repair``. Nothing is re-clustered and
nothing is overwritten:

* the 150 frozen banks are REBUILT FROM THEIR STORED RECORDS plus the stored
  label vectors, never by re-running ``build_fixed32_bank``. Each rebuilt bank
  must reproduce the stored ``bank_id`` and ``content_hash``, and every label
  vector must reproduce its stored raw and canonical hash, before a single
  metric is evaluated;
* only the frozen six metrics are evaluated, through the repaired
  implementation-kind dispatch;
* results are written to NEW keys ``metrics_v2__<dataset>__<view>``; the v1
  records keep their content and their hashes.

Label-free: ground truth is never loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
sys.path.insert(0, str(_REPO / "models/ClustOpt"))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")
import logging                                                        # noqa: E402
logging.disable(logging.INFO)

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
CFG = _ROOT / "configs"
VIEW_IDS = ("x_only", "y_only", "xy_2d")


class BankReconstructionError(RuntimeError):
    """A stored bank could not be reproduced exactly from its record."""


def _rebuild_bank(brec, npz_path, F32):
    """Rebuild the frozen bank object. No clustering is performed."""
    c = brec["content"]
    z = np.load(npz_path)
    cands = []
    for cd in c["candidates"]:
        labels = None
        nm = "c%d" % cd["candidate_index"]
        if cd["status"] == "valid":
            if nm not in z:
                raise BankReconstructionError("%s absent from label file" % nm)
            labels = np.asarray(z[nm], dtype=np.int64)
            raw = hashlib.sha256(labels.tobytes()).hexdigest()[:16]
            if raw != cd["raw_labels_hash"]:
                raise BankReconstructionError("%s raw label hash mismatch" % nm)
            from Independent_Domain_Benchmark.methods.clustopt_trace import (
                partition_hash)
            if partition_hash(labels) != cd["canonical_partition_hash"]:
                raise BankReconstructionError("%s canonical hash mismatch" % nm)
        cands.append(F32.Fixed32Candidate(
            candidate_index=cd["candidate_index"], candidate_id=cd["candidate_id"],
            algorithm=cd["algorithm"],
            configuration=MappingProxyType(dict(cd["configuration"])),
            status=cd["status"], failure_kind=cd["failure_kind"],
            raw_labels_hash=cd["raw_labels_hash"],
            canonical_partition_hash=cd["canonical_partition_hash"],
            labels=labels))
    prov = brec["provenance"]
    bank = F32.Fixed32CandidateBank(
        dataset_id=prov["dataset_id"], view_id=prov["view_id"],
        fixed32_version=prov["fixed32_version"],
        fixed32_config_hash=prov["fixed32_config_hash"],
        decision_hash=prov["decision_hash"], full_hash=prov["full_hash"],
        candidate_count=c["candidate_count"], candidates=tuple(cands),
        bank_id=c["bank_id"], provenance=MappingProxyType({}))
    if bank.bank_id != c["bank_id"]:
        raise BankReconstructionError("bank_id changed")
    if bank.content_hash() != c["content_hash"]:
        raise BankReconstructionError(
            "content hash %s != stored %s" % (bank.content_hash(), c["content_hash"]))
    return bank


def _worker(payload: Tuple[str, List[str], List[str]]) -> Dict[str, Any]:
    run_id, keys, repair_set = payload
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "models"))
    sys.path.insert(0, str(_REPO / "models/ClustOpt"))
    sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
    warnings.filterwarnings("ignore")
    import logging as _lg
    _lg.disable(_lg.INFO)

    from Independent_Domain_Benchmark.analysis import fixed32_track as F32
    from Independent_Domain_Benchmark.datasets import corpus_archive as CA
    from Independent_Domain_Benchmark.execution import hashing as H
    from Independent_Domain_Benchmark.execution import store as ST
    from Independent_Domain_Benchmark.execution.common_runner import (
        apply_thread_controls)
    from Independent_Domain_Benchmark.execution.dataset_view import (
        build_dataset_view)

    apply_thread_controls()
    store = ST.ResultStore(RESULTS, run_id)
    registry = json.loads((CFG / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    amend = json.loads((CFG / "stage4c_metric_dispatch_amendment.json")
                       .read_text(encoding="utf-8"))
    man = json.loads((CFG / "dataset_manifest.json").read_text(encoding="utf-8"))
    entries = {e["dataset_id"]: e for e in man["datasets"] if e.get("accepted", True)}

    done, failures = [], []
    cache: Dict[str, np.ndarray] = {}
    for key in keys:
        body = key[len("fixed32__"):]
        ds, view_id = body.rsplit("__", 1)
        t0 = time.perf_counter()
        try:
            brec = store.load(ST.Layer.PHYSICAL, key)
            if brec is None or brec.get("state") != ST.State.COMPLETE.value:
                raise BankReconstructionError("bank record not COMPLETE")
            store.check_integrity(ST.Layer.PHYSICAL, key, brec)
            npz = RESULTS / run_id / "labels" / ("%s.npz" % key)
            bank = _rebuild_bank(brec, npz, F32)

            if ds not in cache:
                X2, arec = CA.load_scientific_input(ds)
                if arec["representation_checksum"] != \
                        entries[ds]["representation_checksum"]:
                    raise BankReconstructionError("archived representation drift")
                cache[ds] = X2
            view = build_dataset_view(dataset_id=ds, source=entries[ds]["source"],
                                      view_id=view_id, X_full=cache[ds])
            if view.decision_hash != brec["provenance"]["decision_hash"] or \
                    view.full_hash != brec["provenance"]["full_hash"]:
                raise BankReconstructionError("view hashes differ from the bank")

            rows = F32.evaluate_bank_metrics(
                bank, view, registry, include_diagnostics=False,
                only_metrics=repair_set)

            prov = dict(brec["provenance"])
            prov.update({
                "metric_evaluation_semantics": F32.METRIC_EVALUATION_SEMANTICS,
                "amendment_id": amend["amendment_id"],
                "repair_set_hash": amend["repair_set_hash"],
                "parent_bank_id": brec["content"]["bank_id"],
                "parent_bank_content_hash": brec["content"]["content_hash"],
                "parent_metric_record": "metrics__" + body,
            })
            store.complete(
                ST.Layer.EVALUATION, "metrics_v2__" + body, provenance=prov,
                content={"bank_id": bank.bank_id, "n_rows": len(rows),
                         "metric_evaluation_semantics":
                             F32.METRIC_EVALUATION_SEMANTICS,
                         "amendment_id": amend["amendment_id"],
                         "repaired_metrics": list(repair_set),
                         "runtime_seconds": round(time.perf_counter() - t0, 3),
                         "rows": rows})
            done.append({"key": key, "rows": len(rows),
                         "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": key, "error": "%s: %s"
                             % (type(exc).__name__, str(exc)[:220])})
    return {"pid": os.getpid(), "done": done, "failures": failures}


def main(argv=None) -> int:
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    from Independent_Domain_Benchmark.execution import store as ST
    amend = json.loads((CFG / "stage4c_metric_dispatch_amendment.json")
                       .read_text(encoding="utf-8"))
    repair_set = list(amend["repair_set"])
    store = ST.ResultStore(RESULTS, args.run_id)

    keys = [k for k in store.keys(ST.Layer.PHYSICAL) if k.startswith("fixed32__")]
    pending = [k for k in keys
               if store.load(ST.Layer.EVALUATION,
                             "metrics_v2__" + k[len("fixed32__"):]) is None]
    if args.limit:
        pending = pending[:args.limit]

    print("Stage 4C metric repair | run %s" % args.run_id)
    print("  amendment    %s" % amend["amendment_id"])
    print("  repair set   %s" % ", ".join(repair_set))
    print("  banks        %d total, %d pending" % (len(keys), len(pending)))
    if not pending:
        print("  nothing to do")
        return 0

    chunks: List[List[str]] = [[] for _ in range(max(1, args.workers))]
    for i, k in enumerate(pending):
        chunks[i % len(chunks)].append(k)
    chunks = [c for c in chunks if c]

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(chunks)) as pool:
        results = pool.map(_worker, [(args.run_id, c, repair_set) for c in chunks])
    secs = time.perf_counter() - t0

    done = [d for r in results for d in r["done"]]
    fails = [f for r in results for f in r["failures"]]
    summary = {"amendment_id": amend["amendment_id"], "run_id": args.run_id,
               "repair_set": repair_set, "banks_repaired": len(done),
               "failures": fails, "wall_clock_seconds": round(secs, 1),
               "clustering_reruns": 0, "banks_regenerated": 0}
    out = RESULTS / args.run_id / "metric_repair_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("  repaired %d banks in %.1fs | failures %d"
          % (len(done), secs, len(fails)))
    for f in fails[:5]:
        print("    FAIL %s %s" % (f["key"], f["error"]))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
