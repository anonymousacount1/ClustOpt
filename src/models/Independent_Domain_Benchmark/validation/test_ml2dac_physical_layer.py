"""Stage-4B-3B2 tests for the ML2DAC physical execution layer.

The gates that matter here are BEHAVIOURAL, not source-string based: the CVI
classifier, the fallback meta-feature extractor and the native optimizer sampler
are each replaced by objects that raise, and a real trace is then built. A guard
that greps for ``predict(`` can pass while the code predicts; one that makes
prediction explode cannot.
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

from Independent_Domain_Benchmark.methods import ml2dac as ML          # noqa: E402
from Independent_Domain_Benchmark.search import protocol as PROTO      # noqa: E402

SMOKE = "NON_SCIENTIFIC_SMOKE_TEST"
BANNED = ("y", "y_true", "labels_true", "target", "true_k", "ari", "nmi",
          "ami", "fmi", "ground_truth", "class_distribution")
BUDGET = 8          # NON_SCIENTIFIC reduced budget


def _X(seed=20260910, n=70):
    r = np.random.default_rng(seed)
    return np.vstack([r.normal([0, 0], .35, (n, 2)), r.normal([4, 0], .35, (n, 2)),
                      r.normal([2, 4], .35, (n, 2))])


def t_api_label_free():
    p = list(inspect.signature(ML.build_ml2dac_candidate_trace).parameters)
    bad = [x for x in p if x.lower() in BANNED]
    assert not bad, "label parameter(s) present: %s" % bad
    return {"signature": p, "banned_present": bad}


def t_no_repository_membership(tr):
    assert tr.dataset_id.startswith("INVENTED_"), tr.dataset_id
    return {"dataset_id": tr.dataset_id, "membership_required": False}


def t_seed_identity(tr):
    seeds = {m: PROTO.derive_seed(tr.dataset_id, tr.view_id, m)
             for m in ML.ARM_METHOD_IDS}
    assert len(set(seeds.values())) == 1, seeds
    assert tr.resolved_seed == next(iter(seeds.values()))
    assert tr.seed_identity == "ml2dac_shared_candidate_slate"
    return {"seeds": seeds, "identity": tr.seed_identity}


def t_trace_id_arm_independent(tr):
    src = inspect.getsource(ML.build_ml2dac_candidate_trace)
    blob = src[src.index("ident = json.dumps"):src.index("trace_id = hashlib")]
    for bad in ("method_id", "ARM_METHOD_IDS", "predicted_cvi", "classifier"):
        assert bad not in blob, "trace identity mentions %s" % bad
    tr2 = ML.build_ml2dac_candidate_trace(_X(), tr.dataset_id, tr.view_id,
                                          budget=tr.budget)
    assert tr2.trace_id == tr.trace_id
    assert tr2.content_hash() == tr.content_hash()
    return {"trace_id": tr.trace_id, "reproducible": True,
            "depends_on_method_id": False}


def t_schema(tr):
    assert len(tr.meta_feature_names) == ML.N_META_FEATURES
    assert tr.meta_feature_schema_hash == ML.META_FEATURE_SCHEMA_HASH
    assert tr.meta_feature_vector.shape == (ML.N_META_FEATURES,)
    assert dict(tr.meta_feature_env) == dict(ML.FROZEN_MF_ENV)
    return {"n_features": len(tr.meta_feature_names),
            "schema_hash": tr.meta_feature_schema_hash,
            "env": dict(tr.meta_feature_env)}


def t_common_state(tr):
    assert tr.imputer_artifact_hash == ML.IMPUTER_HASH
    assert tr.nn_index_artifact_hash == ML.NN_INDEX_HASH
    assert tr.warmstart_repository_semantic_hash == ML.WARMSTART_SEMANTIC_HASH
    assert tr.nearest_neighbor_id and "__" in tr.nearest_neighbor_id
    return {"imputer": tr.imputer_artifact_hash, "nn_index": tr.nn_index_artifact_hash,
            "warmstart_semantic": tr.warmstart_repository_semantic_hash,
            "neighbour": tr.nearest_neighbor_id}


def t_budget(tr):
    assert tr.candidate_count == len(tr.candidates)
    assert tr.n_warmstart_slots + tr.n_sampled_slots == tr.candidate_count
    assert tr.n_sampled_slots == tr.budget // 2, "sampled fill must be half the budget"
    idx = [c.candidate_index for c in tr.candidates]
    assert idx == list(range(tr.candidate_count)), "no resampling / no gaps"
    origins = [c.origin for c in tr.candidates]
    assert origins == sorted(origins, key=lambda o: 0 if o == "warmstart" else 1), \
        "warmstarts must precede sampled fill"
    return {"budget": tr.budget, "warmstart_slots": tr.n_warmstart_slots,
            "sampled_slots": tr.n_sampled_slots,
            "failed_kept": sum(1 for c in tr.candidates if c.status != "valid")}


def t_reachable_algorithms(tr):
    got = {c.algorithm for c in tr.candidates}
    extra = got - set(ML.REACHABLE_ALGORITHMS)
    assert not extra, "unreachable algorithm generated: %s" % sorted(extra)
    return {"generated": sorted(got), "outside_reachable_set": sorted(extra)}


def t_immutable(tr):
    before = tr.content_hash()
    out = {}
    for name, fn in (
            ("metadata", lambda: setattr(tr, "dataset_id", "X")),
            ("collection", lambda: tr.candidates.append(None)),
            ("warmstarts", lambda: tr.warmstarts.append(None)),
            ("meta_feature_vector", lambda: tr.meta_feature_vector.__setitem__(0, 1.0)),
            ("configuration", lambda: dict.__setitem__(
                tr.candidates[0].configuration, "n_clusters", 99)),
            ("warmstart_config", lambda: dict.__setitem__(
                tr.warmstarts[0].configuration, "n_clusters", 99))):
        try:
            fn()
            out[name] = "MUTATED"
        except Exception as exc:  # noqa: BLE001
            out[name] = type(exc).__name__
    lab = next((c.labels for c in tr.candidates if c.labels is not None), None)
    try:
        lab[0] = 12345
        out["labels"] = "MUTATED"
    except Exception as exc:  # noqa: BLE001
        out["labels"] = type(exc).__name__
    assert "MUTATED" not in out.values(), out
    assert tr.content_hash() == before
    return out


def t_four_arm_consumption(tr):
    before = tr.content_hash()
    seen = []
    for _ in ML.ARM_METHOD_IDS:
        seen.append((len(tr.get_candidates_for_scoring()), sum(tr.availability_mask())))
    assert tr.content_hash() == before
    assert len(set(seen)) == 1
    return {"arms": len(seen), "hash_stable": True, "availability_identical": True}


def t_no_ground_truth_fields(tr):
    blob = json.dumps({
        "trace": {k: str(v) for k, v in vars(tr).items()
                  if k not in ("candidates", "warmstarts")},
        "cand": [{k: str(v) for k, v in vars(c).items()} for c in tr.candidates[:5]],
        "ws": [{k: str(v) for k, v in vars(w).items()} for w in tr.warmstarts[:5]],
    }).lower()
    hits = [b for b in ('"ari"', '"nmi"', '"ami"', '"fmi"', "true_k", "y_true",
                        "selected_cvi", "predicted_cvi", "cvi_classifier",
                        "rank_by_ari") if b in blob]
    assert not hits, hits
    return {"forbidden_found": hits}


def t_no_scoring_state(tr):
    fields = set(vars(tr))
    bad = fields & {"selected_cvi", "predicted_cvi", "cvi_classifier_hash",
                    "candidate_scores", "selected_candidate_index",
                    "metric_inventory_id"}
    assert not bad, bad
    return {"scoring_fields_present": sorted(bad)}


def t_failure_kinds():
    assert len(ML.ML2DACFailure.ALL) >= 10
    for k in ("METAFEATURE_ENV_MISMATCH", "METAFEATURE_FAILED",
              "METAFEATURE_SCHEMA_MISMATCH", "NEIGHBOUR_LOOKUP_FAILED",
              "WARMSTART_LOOKUP_FAILED", "SEARCH_SPACE_MISMATCH",
              "CONFIG_GENERATION_FAILED", "CANDIDATE_EXECUTION_FAILED",
              "NO_VALID_CANDIDATE"):
        assert k in ML.ML2DACFailure.ALL, k
    return {"kinds": len(ML.ML2DACFailure.ALL)}


def t_env_gate_is_enforced():
    """Corrupt the expected environment; extraction must refuse, not proceed."""
    keep = ML.FROZEN_MF_ENV
    from types import MappingProxyType
    ML.FROZEN_MF_ENV = MappingProxyType({**dict(keep), "pymfe": "0.0.0-not-real"})
    try:
        ML.extract_authoritative_metafeatures(_X(), repo=_REPO)
        raised = None
    except ML.ML2DACTraceError as exc:
        raised = exc.kind
    finally:
        ML.FROZEN_MF_ENV = keep
    assert raised == ML.ML2DACFailure.METAFEATURE_ENV_MISMATCH, raised
    return {"raised": raised}


# --------------------------------------------------------------- behavioural
def t_behavioural_guards():
    """Build a REAL trace with the classifier, the fallback extractor and the
    native sampler all replaced by raising stubs."""
    import subprocess as _sp
    from sklearn.ensemble import RandomForestClassifier
    from unified_mkr import metafeatures as MF
    from experiments.external_baselines.ML2DAC.indomain import search as S

    calls = {"classifier": 0, "fallback": 0, "native": 0, "constrained": 0,
             "mfenv_subprocess": 0, "classifier_artifact_loaded": 0}

    def boom_classifier(*a, **k):
        calls["classifier"] += 1
        raise AssertionError("CVI classifier invoked inside the physical layer")

    def boom_fallback(*a, **k):
        calls["fallback"] += 1
        raise AssertionError("fallback extract_ml2dac invoked")

    def boom_native(*a, **k):
        calls["native"] += 1
        raise AssertionError("native optimizer sampler invoked")

    real_constrained = S.sample_optimizer_configs_constrained

    def spy_constrained(*a, **k):
        calls["constrained"] += 1
        return real_constrained(*a, **k)

    real_run = ML.subprocess.run

    def spy_run(cmd, *a, **k):
        joined = " ".join(str(c) for c in cmd)
        if ".mfenv" in joined and "pymfe_worker" in joined:
            calls["mfenv_subprocess"] += 1
        return real_run(cmd, *a, **k)

    import joblib
    real_load = joblib.load

    def spy_load(p, *a, **k):
        if "cvi_classifier" in str(p):
            calls["classifier_artifact_loaded"] += 1
            raise AssertionError("cvi_classifier.pkl loaded in the physical layer")
        return real_load(p, *a, **k)

    old = (RandomForestClassifier.predict, MF.extract_ml2dac,
           S.sample_optimizer_configs, S._sample_params,
           S.sample_optimizer_configs_constrained, ML.subprocess.run, joblib.load)
    RandomForestClassifier.predict = boom_classifier
    MF.extract_ml2dac = boom_fallback
    S.sample_optimizer_configs = boom_native
    S._sample_params = boom_native
    S.sample_optimizer_configs_constrained = spy_constrained
    ML.subprocess.run = spy_run
    joblib.load = spy_load
    try:
        tr = ML.build_ml2dac_candidate_trace(
            _X(), "INVENTED_%s_GUARDED" % SMOKE, "xy_2d", budget=BUDGET)
        ok, err = True, ""
    except Exception as exc:  # noqa: BLE001
        tr, ok, err = None, False, "%s: %s" % (type(exc).__name__, str(exc)[:200])
    finally:
        (RandomForestClassifier.predict, MF.extract_ml2dac,
         S.sample_optimizer_configs, S._sample_params,
         S.sample_optimizer_configs_constrained, ML.subprocess.run,
         joblib.load) = old
    assert ok, "trace construction failed under guards: %s" % err
    assert calls["classifier"] == 0, calls
    assert calls["classifier_artifact_loaded"] == 0, calls
    assert calls["fallback"] == 0, calls
    assert calls["native"] == 0, calls
    assert calls["constrained"] >= 1, calls
    assert calls["mfenv_subprocess"] >= 1, calls
    assert tr.candidate_count == BUDGET
    return {**calls, "trace_id": tr.trace_id, "candidates": tr.candidate_count}


def main() -> int:
    print("building physical trace (%s, budget %d)..." % (SMOKE, BUDGET), flush=True)
    tr = ML.build_ml2dac_candidate_trace(
        _X(), "INVENTED_%s_DS" % SMOKE, "xy_2d", budget=BUDGET)
    print("  trace %s | nn %s | seed %d | candidates %d"
          % (tr.trace_id, tr.nearest_neighbor_id, tr.resolved_seed,
             tr.candidate_count), flush=True)
    checks = [("api_label_free", t_api_label_free, ()),
              ("no_repository_membership", t_no_repository_membership, (tr,)),
              ("production_seed_identity", t_seed_identity, (tr,)),
              ("trace_id_arm_independent", t_trace_id_arm_independent, (tr,)),
              ("metafeature_schema_and_env", t_schema, (tr,)),
              ("common_generation_state", t_common_state, (tr,)),
              ("budget_and_no_resampling", t_budget, (tr,)),
              ("only_reachable_algorithms", t_reachable_algorithms, (tr,)),
              ("immutability", t_immutable, (tr,)),
              ("four_arm_consumption", t_four_arm_consumption, (tr,)),
              ("no_ground_truth_fields", t_no_ground_truth_fields, (tr,)),
              ("no_arm_scoring_state", t_no_scoring_state, (tr,)),
              ("failure_kinds_frozen", t_failure_kinds, ()),
              ("environment_gate_enforced", t_env_gate_is_enforced, ()),
              ("behavioural_guards", t_behavioural_guards, ())]
    rows, fail = [], 0
    for name, fn, args in checks:
        try:
            rows.append({"test": name, "status": "PASS", "detail": fn(*args)})
        except Exception as exc:  # noqa: BLE001
            fail += 1
            rows.append({"test": name, "status": "FAIL",
                         "error": "%s: %s" % (type(exc).__name__, str(exc)[:220])})
        print("  %-30s %s" % (name, rows[-1]["status"]), flush=True)
    out = {"suite": "ml2dac_physical_layer", "smoke_tag": SMOKE,
           "trace": {"trace_id": tr.trace_id, "dataset_id": tr.dataset_id,
                     "view_id": tr.view_id, "seed": tr.resolved_seed,
                     "nearest_neighbor": tr.nearest_neighbor_id,
                     "candidates": tr.candidate_count,
                     "valid": sum(tr.availability_mask()),
                     "content_hash": tr.content_hash(),
                     "search_space_fingerprint": tr.search_space_fingerprint,
                     "budget_is_non_scientific": True},
           "external_50_used": False, "labels_used": False,
           "n": len(rows), "n_fail": fail, "rows": rows,
           "status": "PASS" if fail == 0 else "FAIL"}
    (_ROOT / "validation" / "ml2dac_physical_layer_tests.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("  %d tests | %d fail -> %s" % (len(rows), fail, out["status"]))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
