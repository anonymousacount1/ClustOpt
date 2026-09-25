"""Regression test: cluster VALUES must not depend on cluster NAMES.

The shipped CDbw wrapper relabels clusters to contiguous 0-based ids because the
MIT ``cdbw`` package requires them; a 1-based fixture raised ``TypeError`` before
that wrapper existed. Relabelling is only defensible if it is semantically
neutral, so this test pins that: the same partition, named three different ways,
must yield the identical index value.

Naming schemes exercised per fixture:

* ``contiguous``      -- 0,1,2,... exactly as the package wants
* ``non_contiguous``  -- 17, 42, 103, 255 (large, sparse, arbitrary ids)
* ``permuted``        -- the contiguous ids permuted, so cluster ORDER changes

This test found a material defect: PyPI ``cdbw`` 0.2 is NOT order-invariant --
the same partition, renumbered, moved the index by up to 0.725 absolute. CDbw is
therefore called through a frozen canonical cluster ordering
(``external_cvis.cdbw_canonical_labels``), which is what makes this test pass;
the raw magnitude of the artifact is recorded in
``cdbw_label_order_sensitivity.json``.

The same check is applied to CVDD and DCSI, which do their own class handling,
and to CVNN, whose value is defined relative to a compared candidate set and is
therefore relabelled consistently across the whole set.

Run directly to (re)write ``label_invariance_evidence.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT.parent))

from Independent_Domain_Benchmark.metric_validation import external_cvis as CV  # noqa: E402
from Independent_Domain_Benchmark.validation import cvi_fixtures as FX          # noqa: E402

#: sparse, non-contiguous ids required by the acceptance gate
SPARSE_IDS = [17, 42, 103, 255, 511, 1024]
#: All four indices are held to their declared numerical tolerance. Renaming
#: clusters changes the order in which per-cluster terms are summed, so the last
#: bits can move; nothing else can. The tolerances are many orders of magnitude
#: below any value difference that could change a ranking.
#:
#: CDbw is covered far more aggressively by
#: test_cdbw_permutation_invariance.py (8 geometries x 23 relabellings),
#: which exists because the reference package FAILS this property outright.
ATOL = {m: FX.TOLERANCE[m]["atol"] for m in ("cdbw", "cvdd", "cvnn", "dcsi")}


def _schemes(y: np.ndarray):
    """(scheme_name, relabelled_y) for one label vector. Noise (-1) is kept."""
    y = np.asarray(y).astype(int)
    real = y >= 0
    uniq, comp = np.unique(y[real], return_inverse=True)
    contig = y.copy(); contig[real] = comp
    sparse = y.copy(); sparse[real] = np.array(SPARSE_IDS[:len(uniq)])[comp]
    perm = np.array(SPARSE_IDS[:len(uniq)])[::-1]
    permuted = y.copy(); permuted[real] = perm[comp]
    return [("contiguous", contig), ("non_contiguous", sparse),
            ("permuted", permuted)]


def _record(metric_id, fid, vals):
    """vals: {scheme: (valid, value_or_None, reason)}"""
    base_valid, base_val, base_reason = vals["contiguous"]
    row = {"metric": metric_id, "fixture_id": fid,
           "contiguous": base_val, "contiguous_valid": base_valid,
           "contiguous_reason": base_reason}
    worst = 0.0
    for s, (v, val, rsn) in vals.items():
        if s == "contiguous":
            continue
        row[s] = val
        row[s + "_valid"] = v
        if v != base_valid or rsn != base_reason:
            row["status"] = "FAIL"
            row["detail"] = "validity/reason differs under %s" % s
            return row
        if base_valid:
            worst = max(worst, abs(float(val) - float(base_val)))
    row["max_abs_diff"] = float(worst)
    row["atol"] = ATOL[metric_id]
    row["status"] = "PASS" if worst <= ATOL[metric_id] else "FAIL"
    return row


def run() -> dict:
    rows = []
    for fid, (X, y) in FX.build().items():
        schemes = _schemes(y)
        for mid, fn in (("cdbw", CV.cdbw), ("cvdd", CV.cvdd),
                        ("dcsi", CV.dcsi)):
            vals = {}
            for name, yy in schemes:
                r = fn(X, yy)
                vals[name] = (bool(r.valid),
                              None if r.value is None else float(r.value),
                              r.invalid_reason)
            rows.append(_record(mid, fid, vals))
        # CVNN is set-relative: relabel every member of the compared set
        sets = FX.comparison_sets()[fid]
        vals = {}
        for i, (name, _) in enumerate(schemes):
            relabelled = [dict(_schemes(a))[name] for a in sets]
            res = CV.cvnn_index(X, relabelled)
            r = res[0]
            vals[name] = (bool(r.valid),
                          None if r.value is None else float(r.value),
                          r.invalid_reason)
        rows.append(_record("cvnn", fid, vals))

    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    diffs = [r.get("max_abs_diff", 0.0) for r in rows
             if r.get("max_abs_diff") is not None]
    return {"test": "cvi_cluster_label_invariance",
            "claim_under_test": "index values are invariant to how clusters "
                                "are named (contiguous / sparse / permuted)",
            "schemes": ["contiguous", "non_contiguous", "permuted"],
            "sparse_ids": SPARSE_IDS, "atol_by_metric": ATOL,
            "stronger_cdbw_suite": "test_cdbw_permutation_invariance.py",
            "n_checks": len(rows), "n_fail": n_fail,
            "max_abs_diff_overall": max(diffs) if diffs else 0.0,
            "rows": rows,
            "status": "PASS" if n_fail == 0 else "FAIL"}


def main() -> int:
    ev = run()
    out = _ROOT / "validation" / "label_invariance_evidence.json"
    out.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    for r in ev["rows"]:
        if r["status"] == "FAIL":
            print("  FAIL %-6s %-18s %s" % (r["metric"], r["fixture_id"],
                                            r.get("detail", r.get("max_abs_diff"))))
    print("checks=%d fail=%d max_abs_diff=%s -> %s"
          % (ev["n_checks"], ev["n_fail"], ev["max_abs_diff_overall"],
             ev["status"]))
    return 0 if ev["status"] == "PASS" else 1


def test_label_invariance():
    assert run()["status"] == "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
