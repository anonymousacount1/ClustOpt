"""Materialise and verify the durable Stage-4 corpus archive.

Runs UNPINNED on purpose: the frozen manifest checksums were produced with
multi-threaded BLAS, and preparation is outside the T0..T1 runtime boundary, so
its numerical environment is separable from scientific execution. Every prepared
representation is checked against the frozen manifest before it is archived --
a mismatch stops the build rather than being archived as the new truth.
"""
from __future__ import annotations

import argparse
import json
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
warnings.filterwarnings("ignore")

from Independent_Domain_Benchmark.datasets import corpus_archive as CA   # noqa: E402
from Independent_Domain_Benchmark.execution import hashing as H         # noqa: E402

CFG = _ROOT / "configs"
PUB = CA.ARCHIVE_ROOT.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    man = json.loads((CFG / "dataset_manifest.json").read_text(encoding="utf-8"))
    entries = [e for e in man["datasets"] if e.get("accepted", True)]
    if args.limit:
        entries = entries[:args.limit]

    CA.ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ("manifest", "source", "prepared", "ground_truth"):
        (PUB / "corpus" / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("results", "protocols", "code_reference"):
        (PUB / sub).mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for e in entries:
        try:
            rec = CA.prepare_and_archive(e, overwrite=args.overwrite)
            records.append(rec)
            print("   [archive] %-26s %s n=%d checksum=%s %s"
                  % (e["dataset_id"], e["source"], rec["n_samples"],
                     rec["representation_checksum"],
                     "MATCH" if rec["checksum_matches_frozen_manifest"] else "MISMATCH"),
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"dataset_id": e["dataset_id"],
                             "error": "%s: %s" % (type(exc).__name__, str(exc)[:220])})
            print("   [archive] %-26s FAILED %s" % (e["dataset_id"],
                                                    str(exc)[:110]), flush=True)

    by_redist: Dict[str, int] = {}
    for r in records:
        k = r["redistribution"]["raw_redistribution"]
        by_redist[k] = by_redist.get(k, 0) + 1
    pca_n = sum(1 for r in records if r["preparation"]["pca_applied"])

    pub = {
        "manifest": "stage4_external_corpus_publication_manifest",
        "archive_contract": CA.ARCHIVE_CONTRACT, "version": 1,
        "n_datasets": len(records),
        "frozen_corpus_manifest_hash": man["manifest_sha256_16"],
        "preparation_environment": CA.PREPARATION_ENVIRONMENT,
        "preparation_environment_note": (
            "the frozen manifest checksums were produced with multi-threaded "
            "BLAS. Preparation is outside the T0..T1 runtime boundary, so it is "
            "archived once here and never recomputed during scientific execution; "
            "scientific workers run with numeric threads pinned and load the "
            "archived array instead of re-running PCA."),
        "view_definitions": dict(CA.VIEW_DEFINITIONS),
        "checksums_match_frozen_manifest": sum(
            1 for r in records if r["checksum_matches_frozen_manifest"]),
        "pca_reduced_datasets": pca_n,
        "redistribution_summary": by_redist,
        "manual_review_required": sorted(
            r["dataset_id"] for r in records
            if r["redistribution"]["manual_review_required"]),
        "contains_outcome_information": False,
        "entries": [{
            "dataset_id": r["dataset_id"], "source_group": r["source_group"],
            "source_identity": r["source_identity"],
            "acquisition_provenance": r["acquisition_provenance"],
            "redistribution": r["redistribution"],
            "prepared_artifact": r["artifacts"]["stage4_X_full_2d"],
            "ground_truth_artifact": r["artifacts"]["ground_truth"],
            "source_raw_artifact": r["artifacts"].get("source_raw"),
            "prepared_pre_pca_artifact": r["artifacts"].get("prepared_pre_pca"),
            "representation_checksum": r["representation_checksum"],
            "frozen_representation_checksum": r["frozen_representation_checksum"],
            "n_samples": r["n_samples"],
            "prepared_dimensionality": r["prepared_dimensionality"],
            "preparation": r["preparation"],
            "pca_state": r["pca_state"],
            "view_definitions": r["view_definitions"],
            "library_versions": r["library_versions"],
        } for r in sorted(records, key=lambda x: x["dataset_id"])],
        "failures": failures,
        "wall_clock_seconds": round(time.perf_counter() - t0, 1),
    }
    txt = json.dumps(pub, indent=2, default=str)
    out = PUB / "corpus" / "manifest" / "stage4_external_corpus_publication_manifest.json"
    out.write_text(txt, encoding="utf-8")
    pub_hash = H.canonical_text_hash(txt)
    (PUB / "corpus" / "manifest" / "manifest_hash.txt").write_text(
        pub_hash, encoding="utf-8")

    print("\n  archived %d/%d | checksums matching frozen manifest %d | PCA-reduced %d"
          % (len(records), len(entries), pub["checksums_match_frozen_manifest"],
             pca_n))
    print("  redistribution: %s" % json.dumps(by_redist))
    print("  manual review required: %d" % len(pub["manual_review_required"]))
    print("  publication manifest hash: %s" % pub_hash)
    print("  archive root: %s" % CA.ARCHIVE_ROOT)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
