"""Reconstruct the external benchmark datasets locally from their public sources.

No external data are shipped. For each entry of
``data/external_benchmark/dataset_manifest.json`` this fetches the source (FCPS
file, OpenML data id, the scikit-learn loader, or the seeded scikit-learn
generator), applies the recorded preparation (numeric cleaning, standardisation,
PCA to two dimensions where needed) and verifies the result against the recorded
``representation_checksum``. A mismatch is reported, never silently accepted.

    python scripts/reconstruct_external_datasets.py              # all 50
    python scripts/reconstruct_external_datasets.py --only sk_blobs_easy real_seeds
    python scripts/reconstruct_external_datasets.py --verify-only  # do not write arrays

Arrays are written locally to ``src/results_analysis/independent_domain_benchmark/
publication/corpus/<dataset_id>/`` (never commit them). The preparation contains a
PCA: on a different BLAS / hardware configuration the projection of a
high-dimensional source can differ in the last bits, which the checksum detects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import clustopt  # noqa: E402,F401  (puts src/ on the path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args(argv)
    from Independent_Domain_Benchmark.datasets import corpus_archive as CA
    from Independent_Domain_Benchmark.datasets import preprocessing as P
    man = json.loads((ROOT / "data/external_benchmark/dataset_manifest.json").read_text(encoding="utf-8"))
    entries = [e for e in man["datasets"] if e.get("accepted", True)]
    if a.only:
        entries = [e for e in entries if e["dataset_id"] in set(a.only)]
    bad = 0
    for e in entries:
        try:
            if a.verify_only:
                got = CA.acquire(e)
                prep = P.prepare(got["X_raw"], e["dataset_id"], e["source"])
                ok = prep.feature_checksum == e["representation_checksum"]
                msg = prep.feature_checksum
            else:
                CA.prepare_and_archive(e)
                ok, msg = True, "archived"
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, "%s: %s" % (type(exc).__name__, exc)
        bad += not ok
        print("%-34s %-5s %s (expected %s)" % (e["dataset_id"], "OK" if ok else "FAIL", msg, e["representation_checksum"]))
    print("%d/%d reconstructed and verified" % (len(entries) - bad, len(entries)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
