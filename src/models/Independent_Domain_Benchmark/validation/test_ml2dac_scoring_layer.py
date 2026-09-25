"""Stage-4B-3B3 tests for the corrected ML2DAC arm scoring layer.

The gates that matter are BEHAVIOURAL: the legacy defective scorer, the frozen
estimators' ``fit`` and the Stage-4 modern comparators are replaced by objects
that raise, and real scoring is then executed. A guard that greps for
``_selected_cvi_value`` can pass while the code calls it; one that makes the call
explode cannot.
"""
from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_ROOT.parent))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")

from Independent_Domain_Benchmark.methods import ml2dac as ML           # noqa: E402
from Independent_Domain_Benchmark.methods import ml2dac_scoring as SC   # noqa: E402

SMOKE = "NON_SCIENTIFIC_SMOKE_TEST"
BUDGET = 10


def _X(seed=20260910, n=80):
    r = np.random.default_rng(seed)
    return np.vstack([r.normal([0, 0], .35, (n, 2)), r.normal([5, 0], .35, (n, 2)),
                      r.normal([2.5, 5], .35, (n, 2))])


def _raises_without_x_full(trace, X):
    """X_full must never silently default to the view: 46 of 61 classes need it."""
    try:
        SC.score_all_arms(trace, X_decision=X, X_full=None, repo=_REPO)
        return False
    except SC.ML2DACScoreError:
        return True
    except TypeError:
        return True


def _mk(i, v, st="scored", cvi="silhouette", d="MAXIMIZE"):
    return SC.ML2DACCandidateScore("m", "t", i, cvi, None, d, v, st, None)


def run() -> dict:
    X = _X()
    trace = ML.build_ml2dac_candidate_trace(
        X, "INVENTED_%s_SCORE" % SMOKE, "xy_2d", budget=BUDGET)
    h0 = trace.content_hash()
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"test": name, "status": "PASS" if ok else "FAIL",
                     "detail": str(detail)[:400]})
        print("  %-40s %s" % (name, rows[-1]["status"]), flush=True)

    # ---- registry completeness and no default
    reg = SC.load_direction_registry(_REPO)
    dirs = [e["orientation"] for e in reg["entries"]]
    chk("registry_61_classes_no_default",
        len(reg["entries"]) == 61 and set(dirs) == {"MAXIMIZE", "MINIMIZE"},
        "%d entries, %d MAXIMIZE" % (len(dirs), dirs.count("MAXIMIZE")))
    try:
        SC.direction_of("not_a_real_metric_id", _REPO)
        ok = False
        det = "unknown metric silently resolved"
    except SC.ML2DACScoreError as exc:
        ok = exc.kind == SC.ML2DACScoreFailure.CVI_CLASS_UNKNOWN
        det = exc.kind
    chk("unknown_class_raises_not_defaults", ok, det)
    hits = sorted(set(e["canonical_metric_id"] for e in reg["entries"])
                  & SC.FORBIDDEN_METRICS)
    chk("no_modern_comparator_in_class_set", not hits, hits)

    # ---- ML2DAC-only canonical raw values (defect 3 must be gone)
    lab = np.repeat([0, 1, 2], 80)
    tight = SC.canonical_ml2dac_only(X, lab, repo=_REPO)
    r2 = np.random.default_rng(4)
    Xl = np.vstack([r2.normal([0, 0], 2., (80, 2)), r2.normal([1, 0], 2., (80, 2)),
                    r2.normal([.5, 1], 2., (80, 2))])
    loose = SC.canonical_ml2dac_only(Xl, lab, repo=_REPO)
    good = {}
    for m in ("dunn_index", "coggins_jain_index", "cop"):
        d = SC.direction_of(m, _REPO)
        good[m] = (tight[m] > loose[m]) if d == "MAXIMIZE" else (tight[m] < loose[m])
    chk("ml2dac_only_sign_corrected", all(good.values()),
        {k: (round(tight[k], 4), round(loose[k], 4), good[k]) for k in good})

    # ---- selection semantics
    chk("selection_unique_max", SC.select_final((_mk(0, .1), _mk(1, .9), _mk(2, .4))) == 1)
    chk("selection_tie_lowest_index",
        SC.select_final((_mk(0, .5), _mk(1, .5), _mk(2, .4))) == 0)
    chk("selection_skips_ineligible_first",
        SC.select_final((_mk(0, None, "ineligible"), _mk(1, .3))) == 1)
    chk("selection_skips_ineligible_last",
        SC.select_final((_mk(0, .3), _mk(1, None, "ineligible"))) == 0)
    chk("selection_skips_non_finite",
        SC.select_final((_mk(0, float("nan")), _mk(1, float("-inf")), _mk(2, .2))) == 2)
    chk("selection_all_invalid_none",
        SC.select_final((_mk(0, None, "ineligible"), _mk(1, None, "failed"))) is None)
    chk("orientation_applied_exactly_once",
        SC.orient("silhouette", 0.7, _REPO) == 0.7
        and SC.orient("davies_bouldin", 0.7, _REPO) == -0.7,
        "sil %.2f dbi %.2f" % (SC.orient("silhouette", .7, _REPO),
                               SC.orient("davies_bouldin", .7, _REPO)))

    # ---- behavioural: legacy scorer, training and modern comparators all raise
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from experiments.external_baselines.ML2DAC.indomain import phase as LEGACY
    from Independent_Domain_Benchmark.metric_validation import external_cvis as MC
    calls = {"legacy": 0, "fit": 0, "modern": 0, "predict": 0}

    def boom(k):
        def f(*a, **kw):
            calls[k] += 1
            raise AssertionError("%s invoked" % k)
        return f

    real_pred = RandomForestClassifier.predict

    def spy(self, *a, **k):
        calls["predict"] += 1
        return real_pred(self, *a, **k)

    keep = (LEGACY._selected_cvi_value, LEGACY._oriented, RandomForestClassifier.fit,
            SimpleImputer.fit, SimpleImputer.fit_transform,
            RandomForestClassifier.predict,
            {n: getattr(MC, n) for n in ("cdbw", "cvnn", "cvdd", "dcsi")})
    LEGACY._selected_cvi_value = boom("legacy")
    LEGACY._oriented = boom("legacy")
    RandomForestClassifier.fit = boom("fit")
    SimpleImputer.fit = boom("fit")
    SimpleImputer.fit_transform = boom("fit")
    RandomForestClassifier.predict = spy
    for n in keep[6]:
        setattr(MC, n, boom("modern"))
    try:
        res = SC.score_all_arms(trace, X_decision=X, X_full=X, repo=_REPO)
        ok, err = True, ""
    except Exception as exc:  # noqa: BLE001
        res, ok, err = {}, False, "%s: %s" % (type(exc).__name__, str(exc)[:200])
    finally:
        (LEGACY._selected_cvi_value, LEGACY._oriented, RandomForestClassifier.fit,
         SimpleImputer.fit, SimpleImputer.fit_transform,
         RandomForestClassifier.predict) = keep[:6]
        for n, v in keep[6].items():
            setattr(MC, n, v)
    chk("scoring_executes_under_guards", ok, err)
    chk("behavioural_no_legacy_scorer", calls["legacy"] == 0, calls["legacy"])
    chk("behavioural_no_training",
        calls["fit"] == 0 and calls["predict"] >= 4,
        "fit=%d predict=%d" % (calls["fit"], calls["predict"]))
    chk("behavioural_no_modern_comparator", calls["modern"] == 0, calls["modern"])

    # ---- four arms, one trace, no mutation
    chk("trace_unchanged_after_four_arms", trace.content_hash() == h0)
    chk("four_arm_results", len(res) == 4, sorted(res))
    chk("one_shared_trace_id",
        len({r.shared_physical_trace_id for r in res.values()}) == 1
        and all(r.shared_physical_trace_id == trace.trace_id for r in res.values()),
        trace.trace_id)
    chk("semantics_id_on_every_result",
        all(r.scoring_semantics_id == "ml2dac_corrected_cvi_scoring_v1"
            for r in res.values()))
    chk("arm_specific_decision_not_in_trace",
        not any(k.startswith("predicted") or "classifier" in k for k in vars(trace)),
        sorted(vars(trace))[:6])

    # ---- output contract
    blob = json.dumps({m: {k: str(v) for k, v in vars(r).items() if k != "scores"}
                       for m, r in res.items()}).lower()
    bad = [b for b in ('"ari"', '"nmi"', '"ami"', '"fmi"', "true_k", "y_true",
                       "cvi_star", "cvi*") if b in blob]
    chk("no_evaluation_outcome_fields", not bad, bad)
    sig = list(inspect.signature(SC.score_trace).parameters)
    chk("scoring_api_label_free",
        not [p for p in sig if p.lower() in ("y", "y_true", "true_k", "ari")],
        "(%s)" % ", ".join(sig))
    chk("failure_kinds_frozen", len(SC.ML2DACScoreFailure.ALL) >= 8,
        "%d kinds" % len(SC.ML2DACScoreFailure.ALL))

    # ---- applicability: structural failure is DISTINCT from candidate failure
    chk("applicability_61_of_61_both_dims",
        SC.load_applicability_policy(_REPO)["applicable_1d"] == 61
        and SC.load_applicability_policy(_REPO)["applicable_2d"] == 61,
        "1d=%d 2d=%d" % (SC.load_applicability_policy(_REPO)["applicable_1d"],
                         SC.load_applicability_policy(_REPO)["applicable_2d"]))
    chk("is_applicable_true_for_every_class",
        all(SC.is_applicable(e["canonical_metric_id"], d, _REPO)
            for e in reg["entries"] for d in (1, 2)))

    # force ONE class structurally inapplicable and prove the early failure fires
    pol = SC.load_applicability_policy(_REPO)
    victim = res["IDB_ML2DAC_M0_original_cvis"].decision.predicted_cvi_metric_id
    entry = SC._APP_CACHE["by_id"][victim]
    keep1d, keep2d = entry["applicable_1d"], entry["applicable_2d"]
    entry["applicable_1d"] = entry["applicable_2d"] = False
    try:
        r_in = SC.score_trace(trace, "IDB_ML2DAC_M0_original_cvis",
                              X_decision=X, X_full=X, repo=_REPO)
    finally:
        entry["applicable_1d"], entry["applicable_2d"] = keep1d, keep2d
    chk("structural_inapplicability_fails_early",
        r_in.status == "failed"
        and r_in.failure_kind == SC.ML2DACScoreFailure.PREDICTED_CVI_INAPPLICABLE_FOR_VIEW
        and r_in.selected_candidate_index is None and len(r_in.scores) == 0,
        "%s / %d scores" % (r_in.failure_kind, len(r_in.scores)))
    chk("structural_failure_scores_no_candidate", len(r_in.scores) == 0)

    # an APPLICABLE cvi whose every candidate is ineligible -> the other kind
    class _AllNaN:
        X_decision = X

    keep_raw = SC.canonical_raw_value
    SC.canonical_raw_value = lambda *a, **k: float("nan")
    try:
        r_ne = SC.score_trace(trace, "IDB_ML2DAC_M0_original_cvis",
                              X_decision=X, X_full=X, repo=_REPO)
    finally:
        SC.canonical_raw_value = keep_raw
    chk("candidate_level_failure_is_no_eligible_candidate",
        r_ne.status == "failed"
        and r_ne.failure_kind == SC.ML2DACScoreFailure.NO_ELIGIBLE_CANDIDATE
        and len(r_ne.scores) == trace.candidate_count,
        "%s / %d scores" % (r_ne.failure_kind, len(r_ne.scores)))
    chk("two_failure_kinds_are_distinct",
        r_in.failure_kind != r_ne.failure_kind,
        "%s vs %s" % (r_in.failure_kind, r_ne.failure_kind))
    chk("no_fallback_after_either_failure",
        r_in.selected_candidate_index is None
        and r_ne.selected_candidate_index is None
        and r_in.selected_algorithm is None and r_ne.selected_algorithm is None)
    chk("x_full_is_required",
        _raises_without_x_full(trace, X))

    nf = sum(1 for r in rows if r["status"] == "FAIL")
    return {"suite": "ml2dac_scoring_layer", "smoke_tag": SMOKE,
            "scoring_semantics_id": SC.SCORING_SEMANTICS_ID,
            "trace_id": trace.trace_id, "budget": trace.budget,
            "budget_is_non_scientific": True,
            "external_50_used": False, "labels_used": False,
            "arm_results": {m: {"status": r.status,
                                "predicted_cvi": r.decision.predicted_cvi_metric_id,
                                "direction": r.decision.direction,
                                "selected": r.selected_candidate_index}
                            for m, r in res.items()},
            "selection_rule": SC.SELECTION_RULE,
            "n": len(rows), "n_fail": nf, "rows": rows,
            "status": "PASS" if nf == 0 else "FAIL"}


def main() -> int:
    ev = run()
    (_ROOT / "validation" / "ml2dac_scoring_layer_tests.json").write_text(
        json.dumps(ev, indent=2, default=str), encoding="utf-8")
    print("  %d tests | %d fail -> %s" % (ev["n"], ev["n_fail"], ev["status"]))
    return 0 if ev["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
