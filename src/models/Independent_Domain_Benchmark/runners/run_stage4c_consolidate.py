"""Stage 4C -- verified consolidation of a split run namespace.

WHY THIS EXISTS
---------------
``run_id_for`` hashes the preflight identity set, and that set includes
``git_commit``. Committing runner/archive code between Phase A and Phase B
therefore opens a NEW run namespace even though every frozen SCIENTIFIC identity
(plan, corpus, metric registry, numerical policy, utility source, search space,
external/ml2dac commit) is unchanged and was verified by preflight. Phase A
landed in one directory and Phase B in another, and Phase C needs both.

This module does NOT paper over that. It re-admits each Phase-B record into the
Phase-A namespace only if the record proves, cryptographically, that it is the
same work:

  1. the source record is COMPLETE and passes ``store.check_integrity``;
  2. its stored provenance equals a provenance rebuilt HERE from the archived
     corpus -- fixed32 version, fixed32 config hash, metric registry hash, and
     the decision/full/sample-identity hashes of a freshly constructed view;
  3. every valid candidate's stored labels reproduce BOTH the stored
     ``raw_labels_hash`` and the stored ``canonical_partition_hash``;
  4. the bank and its metric record refer to the same ``bank_id``;
  5. re-finalising the record into the target namespace reproduces the source
     ``provenance_hash`` and ``content_hash`` exactly.

Records are re-finalised through ``store.complete`` rather than copied, so the
``run_id`` field inside each record is truthful. A record already present in the
target is never overwritten: it must match, or it is reported as a CONFLICT.

Label-free: this runner loads features only, via ``load_scientific_input``. The
Phase-C ground-truth boundary stays closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")
import logging                                                        # noqa: E402
logging.disable(logging.INFO)

from Independent_Domain_Benchmark.analysis import fixed32_track as F32  # noqa: E402
from Independent_Domain_Benchmark.datasets import corpus_archive as CA  # noqa: E402
from Independent_Domain_Benchmark.execution import hashing as H         # noqa: E402
from Independent_Domain_Benchmark.execution import store as ST          # noqa: E402
from Independent_Domain_Benchmark.execution.dataset_view import (       # noqa: E402
    build_dataset_view)
from Independent_Domain_Benchmark.methods.clustopt_trace import (       # noqa: E402
    partition_hash)

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
CFG = _ROOT / "configs"
VIEW_IDS = ("x_only", "y_only", "xy_2d")

CONSOLIDATION_CONTRACT = "stage4c_run_namespace_consolidation_v1"


class Outcome:
    CONSOLIDATED = "CONSOLIDATED"
    ALREADY_PRESENT_IDENTICAL = "ALREADY_PRESENT_IDENTICAL"
    CONFLICT_TARGET_DIFFERS = "CONFLICT_TARGET_DIFFERS"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"
    LABEL_HASH_MISMATCH = "LABEL_HASH_MISMATCH"
    LABELS_MISSING = "LABELS_MISSING"
    INCOMPLETE_PAIR = "INCOMPLETE_PAIR"
    NOT_COMPLETE = "NOT_COMPLETE"
    ERROR = "ERROR"


def _expected_provenance(dataset_id, source, view_id, X2, reg_hash):
    view = build_dataset_view(dataset_id=dataset_id, source=source,
                              view_id=view_id, X_full=X2)
    return {"track": "fixed32", "dataset_id": dataset_id, "view_id": view_id,
            "fixed32_version": F32.FIXED32_VERSION,
            "fixed32_config_hash": F32.fixed32_config_hash(view_id),
            "metric_registry_hash": reg_hash,
            "decision_hash": view.decision_hash, "full_hash": view.full_hash,
            "sample_identity_hash": view.sample_identity_hash}


def _verify_labels(bank_content, npz_path):
    """Every valid candidate must reproduce both stored hashes."""
    if not npz_path.is_file():
        return {"ok": False, "outcome": Outcome.LABELS_MISSING,
                "detail": npz_path.name, "candidates_verified": 0}
    z = np.load(npz_path)
    checked = mismatched = 0
    bad: List[str] = []
    for c in bank_content["candidates"]:
        if c["status"] != "valid":
            continue
        nm = "c%d" % c["candidate_index"]
        if nm not in z:
            bad.append("%s absent" % nm)
            mismatched += 1
            continue
        a = np.asarray(z[nm], dtype=np.int64)
        raw = hashlib.sha256(a.tobytes()).hexdigest()[:16]
        can = partition_hash(a)
        checked += 1
        if raw != c["raw_labels_hash"] or can != c["canonical_partition_hash"]:
            mismatched += 1
            bad.append("%s raw=%s/%s canon=%s/%s"
                       % (nm, raw, c["raw_labels_hash"], can,
                          c["canonical_partition_hash"]))
    return {"ok": mismatched == 0, "candidates_verified": checked,
            "mismatched": mismatched, "detail": "; ".join(bad[:4]),
            "outcome": (Outcome.LABEL_HASH_MISMATCH if mismatched else None)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-id", required=True)
    ap.add_argument("--target-run-id", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="without this the runner only verifies and reports")
    args = ap.parse_args(argv)

    src_dir = RESULTS / args.source_run_id
    tgt_dir = RESULTS / args.target_run_id
    if not src_dir.is_dir():
        raise SystemExit("source run %s not found" % args.source_run_id)
    if not tgt_dir.is_dir():
        raise SystemExit("target run %s not found" % args.target_run_id)

    src = ST.ResultStore(RESULTS, args.source_run_id)
    tgt = ST.ResultStore(RESULTS, args.target_run_id)
    reg_hash = H.canonical_text_hash(
        (CFG / "metric_analysis_registry.json").read_bytes())
    man = json.loads((CFG / "dataset_manifest.json").read_text(encoding="utf-8"))
    entries = {e["dataset_id"]: e for e in man["datasets"]
               if e.get("accepted", True)}

    bank_keys = [k for k in src.keys(ST.Layer.PHYSICAL)
                 if k.startswith("fixed32__")]
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    cache: Dict[str, np.ndarray] = {}
    t0 = time.perf_counter()

    def record(rec):
        rows.append(rec)
        counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
        print("   [consolidate] %-40s %s" % (rec["key"], rec["outcome"]),
              flush=True)

    for key in bank_keys:
        rec: Dict[str, Any] = {"key": key}
        try:
            body = key[len("fixed32__"):]
            ds, view_id = body.rsplit("__", 1)
            if ds not in entries or view_id not in VIEW_IDS:
                raise ValueError("unrecognised key")
            brec = src.load(ST.Layer.PHYSICAL, key)
            mrec = src.load(ST.Layer.EVALUATION, "metrics__" + body)
            if brec is None or mrec is None:
                rec["outcome"] = Outcome.INCOMPLETE_PAIR
                rec["detail"] = "bank=%s metrics=%s" % (brec is not None,
                                                        mrec is not None)
                record(rec)
                continue
            if brec.get("state") != ST.State.COMPLETE.value or \
                    mrec.get("state") != ST.State.COMPLETE.value:
                rec["outcome"] = Outcome.NOT_COMPLETE
                record(rec)
                continue

            src.check_integrity(ST.Layer.PHYSICAL, key, brec)
            src.check_integrity(ST.Layer.EVALUATION, "metrics__" + body, mrec)

            if ds not in cache:
                X2, arec = CA.load_scientific_input(ds)
                if arec["representation_checksum"] != \
                        entries[ds]["representation_checksum"]:
                    raise ValueError("archived representation != frozen manifest")
                cache[ds] = X2
            exp = _expected_provenance(ds, entries[ds]["source"], view_id,
                                       cache[ds], reg_hash)
            exp_hash = H.semantic_object_hash(exp)
            if exp_hash != brec["provenance_hash"] or \
                    exp_hash != mrec["provenance_hash"]:
                rec["outcome"] = Outcome.PROVENANCE_MISMATCH
                rec["detail"] = sorted(
                    k for k in set(exp) | set(brec["provenance"])
                    if exp.get(k) != brec["provenance"].get(k))
                record(rec)
                continue

            if brec["content"]["bank_id"] != mrec["content"]["bank_id"]:
                rec["outcome"] = Outcome.INCOMPLETE_PAIR
                rec["detail"] = "bank_id disagreement"
                record(rec)
                continue

            npz = src_dir / "labels" / ("%s.npz" % key)
            lab = _verify_labels(brec["content"], npz)
            rec["candidates_verified"] = lab.get("candidates_verified")
            if not lab["ok"]:
                rec["outcome"] = lab["outcome"]
                rec["detail"] = lab["detail"]
                record(rec)
                continue

            t_b = tgt.load(ST.Layer.PHYSICAL, key)
            t_m = tgt.load(ST.Layer.EVALUATION, "metrics__" + body)
            if t_b is not None or t_m is not None:
                same = (t_b is not None and t_m is not None
                        and t_b.get("content_hash") == brec["content_hash"]
                        and t_b.get("provenance_hash") == brec["provenance_hash"]
                        and t_m.get("content_hash") == mrec["content_hash"]
                        and t_m.get("provenance_hash") == mrec["provenance_hash"])
                rec["outcome"] = (Outcome.ALREADY_PRESENT_IDENTICAL if same
                                  else Outcome.CONFLICT_TARGET_DIFFERS)
                record(rec)
                continue

            rec["bank_content_hash"] = brec["content_hash"]
            rec["metrics_content_hash"] = mrec["content_hash"]
            rec["provenance_hash"] = brec["provenance_hash"]
            rec["metric_rows"] = mrec["content"]["n_rows"]
            rec["valid_candidates"] = brec["content"]["valid_candidates"]

            if args.apply:
                nb = tgt.complete(ST.Layer.PHYSICAL, key,
                                  provenance=brec["provenance"],
                                  content=brec["content"])
                nm2 = tgt.complete(ST.Layer.EVALUATION, "metrics__" + body,
                                   provenance=mrec["provenance"],
                                   content=mrec["content"])
                if nb["content_hash"] != brec["content_hash"] or \
                        nb["provenance_hash"] != brec["provenance_hash"] or \
                        nm2["content_hash"] != mrec["content_hash"] or \
                        nm2["provenance_hash"] != mrec["provenance_hash"]:
                    raise ValueError("re-finalisation changed a hash")
                dst = tgt_dir / "labels"
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(npz, dst / npz.name)
                if hashlib.sha256((dst / npz.name).read_bytes()).hexdigest() != \
                        hashlib.sha256(npz.read_bytes()).hexdigest():
                    raise ValueError("label file copy is not byte-identical")
                relab = _verify_labels(brec["content"], dst / npz.name)
                if not relab["ok"]:
                    raise ValueError("copied labels fail the hash contract")
            rec["outcome"] = Outcome.CONSOLIDATED
        except Exception as exc:  # noqa: BLE001
            rec["outcome"] = Outcome.ERROR
            rec["detail"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        record(rec)

    report = {
        "contract": CONSOLIDATION_CONTRACT,
        "source_run_id": args.source_run_id,
        "target_run_id": args.target_run_id,
        "applied": bool(args.apply),
        "reason": ("run_id_for hashes git_commit; committing runner code between "
                   "phases opened a second namespace while every frozen scientific "
                   "identity stayed equal and preflight-verified"),
        "acceptance_rule": (
            "COMPLETE + store integrity + provenance rebuilt from the archived "
            "corpus + per-candidate raw and canonical label-hash reproduction + "
            "bank_id agreement + hash-preserving re-finalisation"),
        "ground_truth_loaded": False,
        "banks_examined": len(bank_keys),
        "outcomes": counts,
        "rows": rows,
        "wall_clock_seconds": round(time.perf_counter() - t0, 1),
    }
    out = tgt_dir / "consolidation_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n  examined %d | %s" % (len(bank_keys), json.dumps(counts)))
    print("  report: %s" % out)
    bad = sum(v for k, v in counts.items()
              if k not in (Outcome.CONSOLIDATED,
                           Outcome.ALREADY_PRESENT_IDENTICAL))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
