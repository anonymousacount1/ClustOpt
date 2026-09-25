"""Stage 4C — amended run: recover, freeze the rerun set, then rerun only that set.

Three sub-commands, in order:

``recover``
    For every SUCCESSFUL parent arm, materialise the final partition from the
    configuration the parent already selected and verify it cryptographically.
    Verified arms are migrated into the amended run with
    ``result_origin = VERIFIED_PARENT_PARTITION_RECOVERY``. Everything else --
    hash mismatch, non-reconstructible, error -- lands in the targeted-rerun set.
    Legitimate parent FAILURES are migrated as failures and never rerun.

``freeze``
    Write the TARGETED_RERUN_SET, hashed, before any rerun executes. The set also
    includes arms the parent never produced at all.

``rerun``
    Execute only the frozen set as ordinary frozen scientific arms under the
    patched v2 contract. One execution each: no repetition, no seed shopping, no
    forcing of the parent configuration, no ground truth.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")
import logging                                                        # noqa: E402
logging.disable(logging.INFO)

from Independent_Domain_Benchmark.execution import hashing as H        # noqa: E402
from Independent_Domain_Benchmark.execution import recovery as RC      # noqa: E402
from Independent_Domain_Benchmark.execution import store as ST         # noqa: E402
from Independent_Domain_Benchmark.execution.dataset_view import (      # noqa: E402
    build_dataset_view)

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
PARENT_RUN = "stage4c_62ed0d75d565"
VIEW_IDS = ("x_only", "y_only", "xy_2d")
CFG = _ROOT / "configs"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "r4c", str(_ROOT / "runners" / "run_stage4c_benchmark.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def amended_run_id(parent: str) -> str:
    from Independent_Domain_Benchmark.execution import common_runner as CR
    return "stage4c_v2_%s" % H.semantic_object_hash(
        {"parent": parent, "contract": CR.SCIENTIFIC_ARM_CONTRACT,
         "amendment": "final_labels_persistence"})[:10]


def _views_for(entry, D):
    X2, _y = D.materialise(entry)
    return {v: build_dataset_view(dataset_id=entry["dataset_id"],
                                  source=entry["source"], view_id=v, X_full=X2)
            for v in VIEW_IDS}


# ------------------------------------------------------------------ recover
def cmd_recover(args) -> int:
    D = _driver()
    man = {e["dataset_id"]: e for e in D._manifest()["datasets"]}
    parent = ST.ResultStore(RESULTS, PARENT_RUN)
    amended_id = amended_run_id(PARENT_RUN)
    amended = ST.ResultStore(RESULTS, amended_id)

    keys = parent.keys(ST.Layer.SCIENTIFIC)
    by_ds: Dict[str, List[str]] = {}
    for k in keys:
        by_ds.setdefault(k.split("__")[0], []).append(k)

    stats = {"parent_arm_records": len(keys), "successes": 0, "failures": 0,
             "verified": 0, "migrated_failures": 0, "rerun": [],
             "by_outcome": {}, "verified_by_convention": {}}
    rows = []
    for ds in sorted(by_ds):
        if ds not in man:
            continue
        views = None
        for k in sorted(by_ds[ds]):
            rec = parent.load(ST.Layer.SCIENTIFIC, k)
            c, pr = rec["content"], rec["provenance"]
            mid, view_id = pr["method_id"], pr["view_id"]
            if c["status"] != "success":
                stats["failures"] += 1
                amended.complete(
                    ST.Layer.SCIENTIFIC, k, provenance=pr,
                    content={**c, "result_contract": "scientific_arm_result_v2",
                             "result_origin": "MIGRATED_PARENT_FAILURE",
                             "parent_run_id": PARENT_RUN})
                stats["migrated_failures"] += 1
                continue
            stats["successes"] += 1
            if views is None:
                views = _views_for(man[ds], D)
            res = RC.recover_partition(c, mid, views[view_id])
            stats["by_outcome"][res.outcome] = stats["by_outcome"].get(res.outcome, 0) + 1
            if res.outcome == RC.RecoveryOutcome.VERIFIED:
                stats["verified"] += 1
                stats["verified_by_convention"][res.matched_convention] = \
                    stats["verified_by_convention"].get(res.matched_convention, 0) + 1
                amended.complete(
                    ST.Layer.SCIENTIFIC, k, provenance=pr,
                    content={**RC.migrate_record(rec, res, views[view_id]),
                             "parent_run_id": PARENT_RUN})
            else:
                stats["rerun"].append({
                    "key": k, "dataset_id": ds, "view_id": view_id,
                    "method_id": mid, "parent_status": c["status"],
                    "selected_algorithm": c.get("selected_algorithm"),
                    "reason": res.outcome, "detail": res.detail,
                    "parent_final_labels_hash": c.get("final_labels_hash")})
            rows.append({"key": k, "outcome": res.outcome,
                         "convention": res.matched_convention,
                         "algorithm": c.get("selected_algorithm")})
        print("   [recover] %-26s parent=%d verified=%d rerun=%d"
              % (ds, len(by_ds[ds]), stats["verified"], len(stats["rerun"])),
              flush=True)

    stats["amended_run_id"] = amended_id
    out = RESULTS / amended_id / "recovery_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**stats, "rows": rows}, indent=2, default=str),
                   encoding="utf-8")
    print("  parent arms %d | successes %d | verified %d | rerun %d | failures migrated %d"
          % (stats["parent_arm_records"], stats["successes"], stats["verified"],
             len(stats["rerun"]), stats["migrated_failures"]))
    return 0


# ------------------------------------------------------------------- freeze
def cmd_freeze(args) -> int:
    D = _driver()
    man = [e for e in D._manifest()["datasets"] if e.get("accepted", True)]
    amended_id = amended_run_id(PARENT_RUN)
    rep = json.loads((RESULTS / amended_id / "recovery_report.json")
                     .read_text(encoding="utf-8"))
    amended = ST.ResultStore(RESULTS, amended_id)
    have = set(amended.keys(ST.Layer.SCIENTIFIC))

    from Independent_Domain_Benchmark.execution import groups as G
    expected = [("%s__%s__%s" % (e["dataset_id"], v, m), e["dataset_id"], v, m)
                for e in man for v in VIEW_IDS for m in G.frozen_method_ids()]
    never = [{"key": k, "dataset_id": d, "view_id": v, "method_id": m,
              "parent_status": "ABSENT", "selected_algorithm": None,
              "reason": "NEVER_PRODUCED_IN_PARENT", "detail": "",
              "parent_final_labels_hash": None}
             for k, d, v, m in expected
             if k not in have and not any(r["key"] == k for r in rep["rerun"])]

    rerun_set = sorted(rep["rerun"] + never, key=lambda r: r["key"])
    import collections
    payload = {
        "set": "TARGETED_RERUN_SET", "amended_run_id": amended_id,
        "parent_run_id": PARENT_RUN, "n": len(rerun_set),
        "frozen_before_execution": True,
        "selection_rule": ("reconstruct the parent-selected configuration, verify the "
                           "partition hash, and rerun ONLY where verification fails or "
                           "no parent record exists. Outcomes/ARI play no part."),
        "ground_truth_used_to_choose_set": False,
        "by_reason": dict(collections.Counter(r["reason"] for r in rerun_set)),
        "by_method": dict(collections.Counter(r["method_id"] for r in rerun_set)),
        "by_algorithm": dict(collections.Counter(str(r["selected_algorithm"])
                                                 for r in rerun_set)),
        "by_view": dict(collections.Counter(r["view_id"] for r in rerun_set)),
        "by_dataset": dict(collections.Counter(r["dataset_id"] for r in rerun_set)),
        "entries": rerun_set,
    }
    payload["set_hash"] = H.semantic_object_hash(
        {"keys": [r["key"] for r in rerun_set]})
    p = RESULTS / amended_id / "targeted_rerun_set.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("  TARGETED_RERUN_SET n=%d hash=%s" % (payload["n"], payload["set_hash"]))
    print("  by reason  :", json.dumps(payload["by_reason"]))
    print("  by method  :", json.dumps(payload["by_method"]))
    print("  by algorithm:", json.dumps(payload["by_algorithm"]))
    print("  by view    :", json.dumps(payload["by_view"]))
    return 0


# -------------------------------------------------------------------- rerun
def cmd_rerun(args) -> int:
    from Independent_Domain_Benchmark.execution.common_runner import (
        AutoClustExecutor, BenchmarkRunner, ClustOptExecutor, ML2DACExecutor)
    D = _driver()
    man = {e["dataset_id"]: e for e in D._manifest()["datasets"]}
    amended_id = amended_run_id(PARENT_RUN)
    spec = json.loads((RESULTS / amended_id / "targeted_rerun_set.json")
                      .read_text(encoding="utf-8"))
    store = ST.ResultStore(RESULTS, amended_id)
    execs = {"AutoClust": AutoClustExecutor(), "ML2DAC": ML2DACExecutor(),
             "ClustOpt": ClustOptExecutor()}
    runner = BenchmarkRunner(store=store, budget=None)

    units: Dict[Tuple[str, str], List[str]] = {}
    for e in spec["entries"]:
        units.setdefault((e["dataset_id"], e["view_id"]), []).append(e["method_id"])
    todo = sorted(units)
    if args.limit:
        todo = todo[:args.limit]
    parent = ST.ResultStore(RESULTS, PARENT_RUN)
    done, diffs = 0, []
    for ds, view_id in todo:
        t0 = time.perf_counter()
        X2, _y = D.materialise(man[ds])
        view = build_dataset_view(dataset_id=ds, source=man[ds]["source"],
                                  view_id=view_id, X_full=X2)
        res = runner.run_view(view, execs)
        for mid in units[(ds, view_id)]:
            out = res.get(mid)
            if out is None:
                continue
            done += 1
            prec = parent.load(ST.Layer.SCIENTIFIC,
                               "%s__%s__%s" % (ds, view_id, mid))
            if prec and prec["content"].get("status") == "success":
                same = (json.dumps(prec["content"].get("selected_configuration"),
                                   sort_keys=True)
                        == json.dumps(out.selected_configuration, sort_keys=True))
                if not same:
                    diffs.append({"key": "%s__%s__%s" % (ds, view_id, mid),
                                  "parent_algorithm":
                                      prec["content"].get("selected_algorithm"),
                                  "rerun_algorithm": out.selected_algorithm})
        print("   [rerun] %-26s %-7s %d arms  %.1fs"
              % (ds, view_id, len(units[(ds, view_id)]), time.perf_counter() - t0),
              flush=True)
    rep = {"amended_run_id": amended_id, "units": len(todo), "arms_rerun": done,
           "selected_config_differs_from_parent": diffs,
           "n_differs": len(diffs),
           "note": ("a difference is provenance only. The documented floating-point "
                    "limitation permits a near-tie to resolve differently; the rerun "
                    "result is authoritative for that arm and was never repeated or "
                    "chosen by outcome."),
           "repeated_runs": 0, "seed_shopping": False}
    (RESULTS / amended_id / "targeted_rerun_report.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print("  reran %d arms over %d units | selection differs on %d"
          % (done, len(todo), len(diffs)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["recover", "freeze", "rerun"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    return {"recover": cmd_recover, "freeze": cmd_freeze,
            "rerun": cmd_rerun}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
