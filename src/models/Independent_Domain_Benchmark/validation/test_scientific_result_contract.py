"""Result-contract regression: a finished run must be evaluable offline.

Stage 4B-4 proved storage integrity -- atomic completion, provenance, resume --
but never proved that the payload the offline evaluator needs was actually
persisted. A whole Phase-A execution therefore finished without the final label
vectors, and ARI could not be computed from it.

The gate here is property-based, not source-string based: build a real arm
record, then compute ARI from *stored data alone* with every scientific execution
path replaced by a raising stub. If evaluation needs the method to run again, the
test fails.

INTERNAL / NON_SCIENTIFIC fixtures only.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_ROOT.parent))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")
import logging                                                        # noqa: E402
logging.disable(logging.INFO)

from Independent_Domain_Benchmark.execution import store as ST         # noqa: E402
from Independent_Domain_Benchmark.execution.common_runner import (     # noqa: E402
    AutoClustExecutor, BenchmarkRunner, ML2DACExecutor,
    SCIENTIFIC_ARM_CONTRACT)
from Independent_Domain_Benchmark.execution.dataset_view import (      # noqa: E402
    build_dataset_view)

SMOKE = "NON_SCIENTIFIC_SMOKE_TEST"
BUDGET = 6


def main() -> int:
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"test": name, "status": "PASS" if ok else "FAIL",
                     "detail": str(detail)[:300]})
        print("  %-46s %s  %s" % (name, rows[-1]["status"], str(detail)[:50]),
              flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="idb_contract_"))
    try:
        r = np.random.default_rng(20260914)
        X = np.vstack([r.normal([0, 0], .35, (60, 2)), r.normal([5, 0], .35, (60, 2)),
                       r.normal([2.5, 5], .35, (60, 2))])
        y_true = np.repeat([0, 1, 2], 60)
        view = build_dataset_view(dataset_id="INVENTED_%s_CONTRACT" % SMOKE,
                                  source="internal_non_scientific_fixture",
                                  view_id="xy_2d", X_full=X)
        store = ST.ResultStore(tmp, "contract_run")
        runner = BenchmarkRunner(store=store, budget=BUDGET)
        res = runner.run_view(view, {"AutoClust": AutoClustExecutor(),
                                     "ML2DAC": ML2DACExecutor()})

        recs = {k: store.load(ST.Layer.SCIENTIFIC, k)
                for k in store.keys(ST.Layer.SCIENTIFIC)}
        succ = {k: v for k, v in recs.items() if v["content"]["status"] == "success"}
        chk("arms_persisted", len(recs) == 8, "%d records" % len(recs))
        chk("contract_version_is_v2",
            all(v["content"].get("result_contract") == SCIENTIFIC_ARM_CONTRACT
                for v in recs.values()), SCIENTIFIC_ARM_CONTRACT)
        chk("every_success_has_final_labels",
            all("final_labels" in v["content"] for v in succ.values()),
            "%d successes" % len(succ))
        chk("label_vector_covers_every_sample",
            all(len(v["content"]["final_labels"]) == view.sample_count
                for v in succ.values()), view.sample_count)

        import hashlib
        from Independent_Domain_Benchmark.methods.clustopt_trace import partition_hash
        ok_raw = ok_canon = True
        for v in succ.values():
            a = np.asarray(v["content"]["final_labels"], dtype=np.int64)
            ok_raw &= hashlib.sha256(a.tobytes()).hexdigest()[:16] \
                == v["content"]["final_labels_raw_hash"]
            ok_canon &= partition_hash(a) == v["content"]["canonical_partition_hash"]
        chk("stored_labels_match_stored_hashes", ok_raw and ok_canon,
            "raw and canonical both verified")

        # ---- the decisive property: evaluate with execution made impossible
        from Independent_Domain_Benchmark.methods import autoclust as AC
        from Independent_Domain_Benchmark.methods import autoclust_scoring as AS
        from Independent_Domain_Benchmark.methods import ml2dac as ML
        from Independent_Domain_Benchmark.methods import ml2dac_scoring as MS
        from unified_mkr import clustering as CL
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise AssertionError("scientific execution invoked during evaluation")

        keep = (AC.build_autoclust_candidate_trace, ML.build_ml2dac_candidate_trace,
                AS.score_trace, MS.score_trace, CL.run_configuration)
        (AC.build_autoclust_candidate_trace, ML.build_ml2dac_candidate_trace,
         AS.score_trace, MS.score_trace, CL.run_configuration) = (boom,) * 5
        try:
            from sklearn.metrics import adjusted_rand_score
            aris = {k: float(adjusted_rand_score(
                y_true, np.asarray(v["content"]["final_labels"])))
                for k, v in succ.items()}
            evaluable, err = True, ""
        except Exception as exc:  # noqa: BLE001
            aris, evaluable = {}, False
            err = "%s: %s" % (type(exc).__name__, str(exc)[:160])
        finally:
            (AC.build_autoclust_candidate_trace, ML.build_ml2dac_candidate_trace,
             AS.score_trace, MS.score_trace, CL.run_configuration) = keep
        chk("offline_evaluation_needs_no_re_execution",
            evaluable and calls["n"] == 0 and len(aris) == len(succ),
            "%d ARIs computed, %d execution calls. %s" % (len(aris), calls["n"], err))

        # ---- a success without labels must be refused, not silently stored
        from Independent_Domain_Benchmark.execution import common_runner as CR
        try:
            CR._label_payload(None, view)
            refused = False
        except CR.RunnerContractError:
            refused = True
        chk("success_without_labels_is_refused", refused,
            "RunnerContractError raised")
        try:
            CR._label_payload(np.zeros(7, dtype=np.int64), view)
            refused2 = False
        except CR.RunnerContractError:
            refused2 = True
        chk("wrong_length_label_vector_is_refused", refused2,
            "sample-count invariant enforced")

        blob = json.dumps([v["content"] for v in recs.values()]).lower()
        chk("no_ground_truth_in_arm_records",
            not any(b in blob for b in ('"ari"', "y_true", "true_k")), "clean")

        n_fail = sum(1 for x in rows if x["status"] == "FAIL")
        out = {"suite": "scientific_result_contract", "stage": "4C amendment",
               "smoke_tag": SMOKE, "contract": SCIENTIFIC_ARM_CONTRACT,
               "external_50_used": False, "external_ari_inspected": False,
               "n": len(rows), "n_fail": n_fail, "rows": rows,
               "status": "PASS" if n_fail == 0 else "FAIL"}
        (_ROOT / "validation" / "scientific_result_contract_tests.json").write_text(
            json.dumps(out, indent=2, default=str), encoding="utf-8")
        print("  %d tests | %d fail -> %s" % (len(rows), n_fail, out["status"]))
        return 0 if n_fail == 0 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
