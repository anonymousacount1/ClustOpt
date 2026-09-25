"""CDbw validation: the diagnostic tracks the package, the primary does not.

Stage 4B-1b changed what "validated" means for CDbw. The package is not a
correct oracle for the published index -- it is order-dependent, and its own
docstrings contradict its aggregations -- so agreeing with it is no longer the
goal. This runner therefore records three separate things:

1. **Diagnostic fidelity.** ``cdbw_package_compat`` must reproduce the raw
   package exactly, so the package's behaviour stays available for comparison.
2. **Primary values.** ``cdbw`` values and components on every fixture, with the
   package value alongside, so the size of the correction is on record.
3. **Order sensitivity.** How far the raw package moves when only the cluster
   names change, measured over every ordering.

Correctness evidence for the primary implementation lives in
``audit_cdbw_paper_semantics.py``: algebraic identities against the package on
inputs where the defects' effect is exactly predictable, plus an independent
permutation-invariance cross-check against ``fpc::cdbw``.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT.parent))

from Independent_Domain_Benchmark.metric_validation import external_cvis as CV  # noqa: E402
from Independent_Domain_Benchmark.validation import cvi_fixtures as FX          # noqa: E402

MAX_K_FOR_FULL_PERMUTATION = 5


def main(argv=None) -> int:
    from cdbw import CDbw
    out_dir = _ROOT / "validation"
    tol = FX.TOLERANCE["cdbw"]
    params = dict(CV.FROZEN_PARAMS["cdbw"])
    print("=" * 78)
    print("CDbw VALIDATION  atol=%g" % tol["atol"])
    print("  primary implementation : paper-faithful correction layer")
    print("  primary label ordering : %s" % CV.CDBW_LABEL_ORDER_POLICY)
    print("=" * 78, flush=True)

    rows, order = [], []
    for fid, (X, y) in FX.build().items():
        ours = CV.cdbw(X, y)
        compat = CV.cdbw_package_compat(X, y)
        canon = CV.cdbw_canonical_labels(X, y)
        K = int(len(np.unique(canon[canon >= 0])))
        rec = {"fixture_id": fid, "K": K,
               "cdbw_paper": ours.value, "cdbw_paper_valid": bool(ours.valid),
               "cdbw_paper_invalid_reason": ours.invalid_reason,
               "cdbw_package_compat": compat.value,
               "cdbw_package_compat_valid": bool(compat.valid)}
        if compat.valid:
            raw = float(CDbw(X, canon, **params))
            rec["package_raw"] = raw
            rec["diagnostic_abs_diff"] = abs(float(compat.value) - raw)
            rec["diagnostic_status"] = (
                "PASS" if rec["diagnostic_abs_diff"] <= tol["atol"] else "FAIL")
        else:
            rec["package_raw"] = None
            rec["diagnostic_abs_diff"] = None
            rec["diagnostic_status"] = "COMPAT_INVALID:%s" % compat.invalid_reason
        if ours.valid:
            c = ours.provenance["components"]
            rec.update({"compactness": c["compactness"],
                        "cohesion": c["cohesion"], "separation": c["separation"],
                        "stdev_paper": c["stdev_paper"],
                        "inter_dens": c["inter_dens"], "dist_m": c["dist_m"]})
            if compat.valid:
                rec["correction_ratio"] = float(ours.value) / float(compat.value)
        rows.append(rec)
        print("  %-18s paper=%-18s package=%-18s %s"
              % (fid,
                 ("%.9g" % ours.value) if ours.valid
                 else "INVALID:" + ours.invalid_reason,
                 ("%.9g" % compat.value) if compat.valid
                 else "INVALID:" + compat.invalid_reason,
                 rec["diagnostic_status"]), flush=True)

        if not compat.valid or K < 2 or K > MAX_K_FOR_FULL_PERMUTATION:
            continue
        vals = [float(CDbw(X, np.array(perm)[canon], **params))
                for perm in itertools.permutations(range(K))]
        v = np.array(vals)
        order.append({"fixture_id": fid, "K": K, "n_orderings": len(vals),
                      "min": float(v.min()), "max": float(v.max()),
                      "canonical": float(vals[0]),
                      "spread_abs": float(v.max() - v.min()),
                      "spread_rel": float((v.max() - v.min())
                                          / max(abs(v.mean()), 1e-300)),
                      "n_distinct": int(len(np.unique(np.round(v, 12))))})

    T = pd.DataFrame(rows)
    T.to_csv(out_dir / "cdbw_reference_validation.csv", index=False)
    comp = T[T.diagnostic_status.isin(["PASS", "FAIL"])]
    n_fail = int((comp.diagnostic_status == "FAIL").sum())
    summary = {
        "metric": "cdbw",
        "primary_implementation": "paper-faithful correction layer over PyPI "
                                  "cdbw==0.2 (MIT), reused with attribution",
        "primary_is_validated_by": "validation/cdbw_paper_semantics_audit.json "
                                   "(algebraic identities against the package "
                                   "plus an independent fpc cross-check) and "
                                   "validation/cdbw_permutation_invariance_"
                                   "evidence.json",
        "why_not_validated_by_agreement_with_the_package": "the package is not "
            "a correct oracle for the published index: it is order-dependent "
            "and its aggregations contradict its own docstrings, so agreeing "
            "with it would be evidence of reproducing a defect",
        "diagnostic_fidelity": {
            "metric_id": "cdbw_package_compat",
            "cases_compared": int(len(comp)),
            "n_pass": int((comp.diagnostic_status == "PASS").sum()),
            "n_fail": n_fail,
            "max_abs_diff": (float(comp["diagnostic_abs_diff"].max())
                             if len(comp) else None),
        },
        "tolerance": tol,
        "label_order_policy_primary": CV.CDBW_LABEL_ORDER_POLICY,
        "label_order_policy_diagnostic": CV.CDBW_COMPAT_LABEL_ORDER_POLICY,
        "non_compared": T[~T.diagnostic_status.isin(["PASS", "FAIL"])]
                         .diagnostic_status.value_counts().to_dict(),
        "overall": "PASS" if (n_fail == 0 and len(comp) > 0) else "FAIL",
    }
    (out_dir / "cdbw_reference_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    O = pd.DataFrame(order)
    O.to_csv(out_dir / "cdbw_label_order_sensitivity.csv", index=False)
    sens = {
        "finding": "PyPI cdbw 0.2 is NOT invariant to the integer ordering of "
                   "cluster labels",
        "status": "ROOT-CAUSED AND CORRECTED in Stage 4B-1b",
        "root_cause": "validation/cdbw_paper_semantics_audit.json",
        "classification": "A -- package implementation defect",
        "is_it_randomness": False,
        "determinism_evidence": "repeated calls with an identical label vector "
                                "return bit-identical values",
        "affected_components": "compactness, cohesion and separation, all via "
                               "the mis-aggregated neighbourhood radius; "
                               "separation additionally via a minimum taken "
                               "over lower-indexed clusters only",
        "max_spread_abs": float(O["spread_abs"].max()) if len(O) else None,
        "max_spread_rel": float(O["spread_rel"].max()) if len(O) else None,
        "per_fixture": order,
        "resolution": {
            "primary": "the aggregations are corrected, so CDbw is "
                       "permutation-invariant by construction and no ordering "
                       "convention is used",
            "diagnostic": "the raw package remains available as "
                          "cdbw_package_compat under a canonical ordering, "
                          "purely so this comparison stays reproducible",
        },
        "reporting_requirement": "CDbw values produced by the raw package "
                                 "without a stated cluster ordering are not "
                                 "comparable to ours, and are not comparable "
                                 "to each other either",
    }
    (out_dir / "cdbw_label_order_sensitivity.json").write_text(
        json.dumps(sens, indent=2), encoding="utf-8")

    print("\n  diagnostic fidelity: compared %d | PASS %d | FAIL %d | max abs %s"
          % (summary["diagnostic_fidelity"]["cases_compared"],
             summary["diagnostic_fidelity"]["n_pass"], n_fail,
             summary["diagnostic_fidelity"]["max_abs_diff"]))
    print("  package order sensitivity: max abs spread %s over %d fixtures"
          % (sens["max_spread_abs"], len(order)))
    print("  OVERALL: %s" % summary["overall"])
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
