"""Stage-4B-3A2a tests for the AutoClust arm-specific scoring layer.

The important gates here are BEHAVIOURAL, not source-string based: the frozen
predictors' ``fit`` and the Stage-4 modern comparators are replaced with objects
that raise, and real scoring is then executed. A guard that greps for ``.fit(``
can pass while the code trains; one that makes training explode cannot.
"""
from __future__ import annotations

import inspect, json, sys, warnings
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_ROOT.parent))
warnings.filterwarnings("ignore")

from Independent_Domain_Benchmark.methods import autoclust as AC             # noqa: E402
from Independent_Domain_Benchmark.methods import autoclust_scoring as SC     # noqa: E402

SMOKE = "NON_SCIENTIFIC_SMOKE_TEST"
EXPECT = {"IDB_AutoClust_A0_original_cvis": (7, "c1f4e857e0a7a7ad"),
          "IDB_AutoClust_A1_original_plus_established": (17, "6c0975fb360c3481"),
          "IDB_AutoClust_A2_original_plus_new46": (51, "751aaec7d687ccb4"),
          "IDB_AutoClust_A3_extended_cvis": (61, "252e4fca8d7399b9")}


def _X(n=90):
    r = np.random.default_rng(20260911)
    return np.vstack([r.normal([0, 0], .35, (n, 2)), r.normal([4, 0], .35, (n, 2)),
                      r.normal([2, 4], .35, (n, 2))])


def _mk(i, v, st="scored"):
    return SC.AutoClustCandidateScore("m", "t", i, "h", "vh", "ph", v, st, None, 0)


def run() -> dict:
    X = _X()
    trace = AC.build_autoclust_candidate_trace(
        X, "INVENTED_%s_DS" % SMOKE, "xy_2d", budget=6)
    h0 = trace.content_hash()
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"test": name, "status": "PASS" if ok else "FAIL",
                     "detail": str(detail)})
        print("  %-40s %s" % (name, rows[-1]["status"]), flush=True)

    # ---- schemas and artifacts
    arms, ok_schema, ok_hash, ok_forbid = {}, True, True, True
    for m, (n, h) in EXPECT.items():
        a = SC.load_arm(m)
        arms[m] = a
        ok_schema &= len(a["feature_columns"]) == n
        ok_schema &= len(set(a["feature_columns"])) == n
        ok_hash &= a["predictor_hash"] == h
        ok_forbid &= not (set(c.lower() for c in a["feature_columns"])
                          & SC.FORBIDDEN_FEATURES)
    chk("schemas_7_17_51_61_no_duplicates", ok_schema,
        {m: len(a["feature_columns"]) for m, a in arms.items()})
    chk("predictor_artifact_hashes_exact", ok_hash,
        {m: a["predictor_hash"] for m, a in arms.items()})
    chk("no_modern_comparator_in_schema", ok_forbid)

    # ---- behavioural: no training, no modern comparator
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from Independent_Domain_Benchmark.metric_validation import external_cvis as MC
    calls = {"fit": 0, "predict": 0, "modern": 0}
    of, op, pf = MLPRegressor.fit, MLPRegressor.predict, Pipeline.fit

    def boom_fit(*a, **k):
        calls["fit"] += 1
        raise AssertionError("fit invoked")

    def spy(self, *a, **k):
        calls["predict"] += 1
        return op(self, *a, **k)

    def boom_mod(*a, **k):
        calls["modern"] += 1
        raise AssertionError("modern comparator invoked")

    keep = {k: getattr(MC, k) for k in ("cdbw", "cvnn", "cvdd", "dcsi")}
    MLPRegressor.fit, Pipeline.fit, MLPRegressor.predict = boom_fit, boom_fit, spy
    for k in keep:
        setattr(MC, k, boom_mod)
    try:
        # V2 contract: X_full is explicit. This fixture is xy_2d, so the
        # decision view and the full evaluation space coincide.
        res = SC.score_all_arms(trace, X_decision=X, X_full=X)
        exec_ok, exec_err = True, ""
    except Exception as e:
        res, exec_ok, exec_err = {}, False, "%s: %s" % (type(e).__name__, str(e)[:150])
    finally:
        MLPRegressor.fit, Pipeline.fit, MLPRegressor.predict = of, pf, op
        for k, v in keep.items():
            setattr(MC, k, v)
    chk("scoring_executes_under_guards", exec_ok, exec_err)
    chk("behavioural_no_training", calls["fit"] == 0 and calls["predict"] > 0,
        "fit=%d predict=%d" % (calls["fit"], calls["predict"]))
    chk("behavioural_no_modern_comparator", calls["modern"] == 0,
        "modern=%d" % calls["modern"])

    # ---- trace untouched, four results
    chk("trace_unchanged_after_four_arms", trace.content_hash() == h0)
    chk("four_arm_results", len(res) == 4 and all(
        r.status == "success" for r in res.values()),
        {m: r.selected_candidate_index for m, r in res.items()})
    chk("one_shared_trace_id",
        len({r.trace_id for r in res.values()}) == 1 and
        all(r.trace_id == trace.trace_id for r in res.values()), trace.trace_id)

    # ---- label safety
    sig = list(inspect.signature(SC.score_trace).parameters)
    bad = [p for p in sig if p.lower() in
           ("y", "y_true", "target", "true_k", "ari", "nmi", "ami", "fmi")]
    chk("scoring_api_label_free", not bad, "(%s)" % ", ".join(sig))
    import inspect as _i
    _xf = _i.signature(SC.score_trace).parameters.get("X_full")
    chk("x_full_required_no_silent_alias",
        _xf is not None and _xf.default is _i.Parameter.empty
        and "Xd if X_full is None" not in _i.getsource(SC.score_trace),
        "X_full is a required keyword")
    blob = json.dumps({m: {k: str(v) for k, v in vars(r).items() if k != "scores"}
                       for m, r in res.items()}).lower()
    hits = [b for b in ('"ari"', '"nmi"', '"ami"', '"fmi"', "true_k", "y_true")
            if b in blob]
    chk("no_ground_truth_in_results", not hits, hits)

    # ---- selection semantics
    chk("selection_unique_max", SC.select_final((_mk(0, .1), _mk(1, .9), _mk(2, .4))) == 1)
    chk("selection_tie_lowest_index",
        SC.select_final((_mk(0, .5), _mk(1, .5), _mk(2, .4))) == 0)
    chk("selection_skips_ineligible",
        SC.select_final((_mk(0, None, "ineligible"), _mk(1, .3))) == 1)
    chk("selection_skips_nan", SC.select_final((_mk(0, float("nan")), _mk(1, .2))) == 1)
    chk("selection_all_invalid_none",
        SC.select_final((_mk(0, None, "ineligible"), _mk(1, None, "failed"))) is None)

    # ---- NaN reaches the imputer rather than being zeroed
    src = inspect.getsource(SC.score_trace)
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    chk("nan_not_pre_zeroed", "nan_to_num" not in code and "fillna" not in code)
    chk("failure_kinds_frozen", len(SC.AutoClustScoreFailure.ALL) >= 6,
        "%d kinds" % len(SC.AutoClustScoreFailure.ALL))

    nf = sum(1 for r in rows if r["status"] == "FAIL")
    return {"suite": "autoclust_scoring_layer", "smoke_tag": SMOKE,
            "trace_id": trace.trace_id, "budget": trace.budget,
            "external_50_used": False, "labels_used": False,
            "arm_results": {m: {"selected": r.selected_candidate_index,
                                "algorithm": r.selected_algorithm,
                                "n_eligible": r.n_eligible,
                                "schema_hash": r.feature_schema_hash,
                                "predictor_hash": r.predictor_artifact_hash,
                                "inventory": r.metric_inventory_id}
                            for m, r in res.items()},
            "selection_rule": SC.SELECTION_RULE,
            "n": len(rows), "n_fail": nf, "rows": rows,
            "status": "PASS" if nf == 0 else "FAIL"}


def main() -> int:
    ev = run()
    (_ROOT / "validation" / "autoclust_scoring_layer_tests.json").write_text(
        json.dumps(ev, indent=2, default=str), encoding="utf-8")
    print("  %d tests | %d fail -> %s" % (ev["n"], ev["n_fail"], ev["status"]))
    return 0 if ev["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
