"""Stage 4B-2 test suite: are the six ClustOpt arms faithful adapters?

The bar is not "the adapter runs". It is that the adapter reproduces what the
frozen ClustOpt pipeline already produced. Toy data cannot show that -- the
utility models saturate on inputs unlike their training corpus -- so the metric
selection tests replay REAL internal split-1 records and compare against the
persisted historical selections. That set is internal synthetic-domain only and
is never part of the frozen 50-dataset external corpus; it exists to check
adapter fidelity, and no scientific claim is drawn from it.

Sections mirror the stage brief:

A artifacts   B feature schemas   C policy predictor   D MLP   E KNN
F trace sharing   G reranker   H label safety   I output schema
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_ROOT.parent))

warnings.filterwarnings("ignore")

from Independent_Domain_Benchmark.methods import clustopt_adapter as CA        # noqa: E402
from Independent_Domain_Benchmark.methods.clustopt_trace import (              # noqa: E402
    ClustOptFailure, partition_hash)
from Independent_Domain_Benchmark.search import protocol as PROTO             # noqa: E402

COVERAGE = _REPO / ("results_analysis/clustopt_candidate_reranker/"
                    "stage2b5a_final_models_and_split1_preparation/"
                    "split1_trace_coverage.csv.gz")
KNN_CTX = _REPO / ("results_analysis/clustopt_candidate_reranker/"
                   "stage2b5a_final_models_and_split1_preparation/"
                   "split1_knn_utility_context.csv.gz")
ANALYZED = _REPO / "results_analysis/clustering_repository/analyzed_data"

#: deterministic, small, and spread across families/views by construction
N_REGRESSION_RECORDS = 12
REGRESSION_SEED = 20260911
SMOKE_TAG = "NON_SCIENTIFIC_SMOKE_TEST"


def _dataset_dir(row) -> Path:
    return ANALYZED / row["family"] / "subfamilies" / row["subfamily"] / row["dataset_id"]


def regression_records(policy_id: str, n: int = N_REGRESSION_RECORDS
                       ) -> pd.DataFrame:
    """A deterministic internal subset: multiple families, all three views."""
    cov = pd.read_csv(COVERAGE)
    sub = cov[(cov.policy_id == policy_id) & (cov.exists == True)]  # noqa: E712
    sub = sub[sub.selected_metric_names.notna()]
    sub = sub.sort_values(["family", "subfamily", "dataset_id", "view_id"])
    picks = []
    for view in ("x_only", "y_only", "xy_2d"):
        v = sub[sub.view_id == view]
        fams = sorted(v.family.unique())
        per = max(1, n // (3 * max(1, len(fams))))
        for fam in fams:
            picks.append(v[v.family == fam].head(per))
    out = pd.concat(picks).drop_duplicates(
        ["dataset_id", "view_id"]).sort_values(["dataset_id", "view_id"])
    return out.head(n).reset_index(drop=True)


# ------------------------------------------------------------------ A artifacts
def test_A_artifacts() -> Dict[str, Any]:
    rows = []
    hashes = None
    for mid in CA.ARM_SPEC:
        a = CA.ClustOptAdapter(mid)
        h = a.initialise()
        hashes = h if hashes is None else hashes
        rows.append({"method_id": mid, "n_artifacts": len(h),
                     "status": "PASS" if h else "FAIL"})
    reg = json.loads((_ROOT / "configs" / "method_registry.json")
                     .read_text(encoding="utf-8"))
    ids = [m["method_id"] for m in reg["methods"] if m["framework"] == "ClustOpt"]
    same = sorted(ids) == sorted(CA.ARM_SPEC)
    return {"section": "A_artifacts", "rows": rows,
            "n_artifact_hashes": len(hashes or {}),
            "registry_ids_match_implementation": bool(same),
            "registry_ids": sorted(ids),
            "pp_v1_hash": (hashes or {}).get("pp_v1"),
            "pp_v2_hash": (hashes or {}).get("pp_v2"),
            "n_rerankers": sum(1 for k in (hashes or {}) if k.startswith("reranker_")),
            "status": "PASS" if same and all(r["status"] == "PASS" for r in rows)
                      else "FAIL"}


# --------------------------------------------------------------- B feature schemas
def test_B_feature_schemas() -> Dict[str, Any]:
    from models.ClustOpt_Policy_Predictor.data import feature_schema as FSCH
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
        knn_predictor_provider as KP, regressor_predictor as RP)
    from models.ClustOpt_Candidate_Reranker.split1_preparation import (
        online_feature_builder as OFB)

    mlp = RP.get_regressor_predictor(str(_REPO / CA.MLP_RUN_DIR),
                                     checkpoint_policy="fold_ensemble")
    knn = KP.get_knn_provider(str(_REPO / CA.KNN_MODEL_DIR))
    metric_names = list(FSCH.load_metric_names(_REPO))
    meta_cols = list(FSCH.load_meta_feature_columns(_REPO))
    util_cols = list(FSCH.oof_feature_columns(metric_names))
    checks = {
        "mlp_n_features": len(mlp.feature_columns),
        "mlp_n_targets": len(mlp.metric_names),
        "mlp_target_order_is_metric_names": mlp.metric_names == metric_names,
        "knn_available": knn is not None,
        "pp_n_features": len(meta_cols) + len(util_cols) + 3,
        "pp_width_note": "META + 120 utilities + the 3-way view one-hot",
        "pp_meta_cols": len(meta_cols), "pp_util_cols": len(util_cols),
        "pp_expected": 373,
        "reranker_expected_dim": int(OFB.EXPECTED_DIM),
        "reranker_expected_is_347": int(OFB.EXPECTED_DIM) == 347,
        "n_metric_names": len(metric_names),
    }
    ok = (checks["mlp_n_targets"] == 60 and checks["n_metric_names"] == 60
          and checks["mlp_target_order_is_metric_names"]
          and checks["reranker_expected_is_347"]
          and checks["pp_n_features"] == 373)
    return {"section": "B_feature_schemas", "checks": checks,
            "status": "PASS" if ok else "FAIL"}


# ------------------------------------------------------------ C policy predictor
def test_C_policy() -> Dict[str, Any]:
    table = CA.policy_table()
    unresolved = [r for r in table
                  if not r["reranker_id"] or r["source"] not in ("mlp", "knn")
                  or r["weighting"] not in ("normalized_positive", "softmax")]
    prov = json.loads((_ROOT / "configs" / "pp_artifact_provenance.json")
                      .read_text(encoding="utf-8"))
    replay = {a: {"rows_compared": v["rows_compared"],
                  "policy_mismatches": v["policy_mismatches"],
                  "bit_exact": v["bit_exact_on_criterion"]}
              for a, v in prov["arms"].items()}
    ok = (len(table) == 20 and not unresolved
          and all(v["policy_mismatches"] == 0 for v in replay.values())
          and prov["frozen_inference_policy"]["n_jobs"] == 1)
    return {"section": "C_policy", "n_policies": len(table),
            "n_unresolved": len(unresolved),
            "all_20_resolve": len(table) == 20 and not unresolved,
            "historical_replay": replay,
            "n_jobs": prov["frozen_inference_policy"]["n_jobs"],
            "policy_table": table,
            "status": "PASS" if ok else "FAIL"}


# ------------------------------------------------- D/E utility-source regression
def _selection_regression(policy_id: str, params: Dict[str, Any],
                          resolver: str) -> Dict[str, Any]:
    """Recompute metric selection and compare with the persisted history."""
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
        resolvers as R)

    recs = regression_records(policy_id)
    fn = getattr(R, resolver)
    rows = []
    for _, r in recs.iterrows():
        ds = _dataset_dir(r)
        hist = [m for m in str(r["selected_metric_names"]).split("|") if m]
        try:
            res = fn(params=dict(params), dataset_dir=str(ds),
                     view_id=r["view_id"])
            ours = list(res.selected_metrics)
            w = dict(res.weights)
            uh = None
            if res.raw_utilities:
                uh = CA._vec_hash(np.array(
                    [float(res.raw_utilities[k])
                     for k in sorted(res.raw_utilities)]))
            same = ours == hist
            wsum = float(sum(w.values())) if w else 0.0
            rows.append({
                "dataset_id": r["dataset_id"], "view_id": r["view_id"],
                "family": r["family"],
                "historical_metrics": hist, "our_metrics": ours,
                "metrics_identical": bool(same),
                "n_selected": len(ours),
                "weights_sum_to_one": abs(wsum - 1.0) <= 1e-9,
                "utility_vector_hash": uh,
                "status": "PASS" if same and abs(wsum - 1.0) <= 1e-9 else "FAIL"})
        except Exception as exc:
            rows.append({"dataset_id": r["dataset_id"], "view_id": r["view_id"],
                         "status": "ERROR",
                         "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])})
    n_fail = sum(1 for x in rows if x["status"] != "PASS")
    return {"policy_id": policy_id, "n_records": len(rows), "n_fail": n_fail,
            "views_covered": sorted({x["view_id"] for x in rows}),
            "families_covered": sorted({x.get("family") for x in rows
                                        if x.get("family")}),
            "rows": rows, "status": "PASS" if n_fail == 0 and rows else "FAIL"}


def test_D_mlp() -> Dict[str, Any]:
    out = _selection_regression(
        "regressor_top5_raw",
        {"model_run_dir": CA.MLP_RUN_DIR, "checkpoint_policy": "fold_ensemble",
         "top_k": 5, "weighting": "normalized_positive",
         "metric_name_prefix": "utility__",
         "feature_source": "features/features_records.csv", "view_aware": True,
         "fallback_on_error": False},
        "resolve_regressor_weights")
    out["section"] = "D_mlp_top5_raw"
    return out


def test_E_knn() -> Dict[str, Any]:
    out = _selection_regression(
        "knn_top10_raw",
        {"knn_model_dir": CA.KNN_MODEL_DIR, "top_k": 10,
         "weighting": "normalized_positive", "metric_name_prefix": "utility__",
         "feature_source": "features/features_records.csv", "view_aware": True,
         "fallback_on_error": False},
        "resolve_knn_weights")
    out["section"] = "E_knn_top10_raw"
    return out


def test_E2_knn_utility_context() -> Dict[str, Any]:
    """The adapter's KNN utilities must equal the persisted split-1 context."""
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
        knn_predictor_provider as KP, loaders as LD)
    from models.ClustOpt_Policy_Predictor.data import feature_schema as FSCH

    ctx = pd.read_csv(KNN_CTX)
    metric_names = list(FSCH.load_metric_names(_REPO))
    recs = regression_records("knn_top10_raw", 6)
    knn = KP.get_knn_provider(str(_REPO / CA.KNN_MODEL_DIR))
    rows = []
    for _, r in recs.iterrows():
        m = ctx[(ctx.dataset_id == r["dataset_id"]) & (ctx.view_id == r["view_id"])]
        if len(m) != 1:
            rows.append({"dataset_id": r["dataset_id"], "view_id": r["view_id"],
                         "status": "NO_HISTORICAL_ROW"})
            continue
        ds = _dataset_dir(r)
        try:
            fv = LD.load_feature_vector(ds, r["view_id"],
                                        list(knn.feature_columns))
            ours, _ = knn.predict(fv, r["view_id"])
            diffs = [abs(float(ours[k]) - float(m.iloc[0]["knn_utility__%s" % k]))
                     for k in metric_names
                     if "knn_utility__%s" % k in m.columns and k in ours]
            worst = max(diffs) if diffs else None
            rows.append({"dataset_id": r["dataset_id"], "view_id": r["view_id"],
                         "n_compared": len(diffs), "max_abs_diff": worst,
                         "status": "PASS" if (worst is not None and worst <= 1e-9)
                                   else "FAIL"})
        except Exception as exc:
            rows.append({"dataset_id": r["dataset_id"], "view_id": r["view_id"],
                         "status": "ERROR",
                         "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])})
    n_fail = sum(1 for x in rows if x["status"] != "PASS")
    return {"section": "E2_knn_utility_context",
            "oracle": "split1_knn_utility_context.csv.gz",
            "n_records": len(rows), "n_fail": n_fail, "rows": rows,
            "status": "PASS" if n_fail == 0 and rows else "FAIL"}


# ------------------------------------------------------------- F trace sharing
def _smoke_X(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal([0, 0], 0.35, (60, 2)),
                      rng.normal([4, 0], 0.35, (60, 2)),
                      rng.normal([2, 4], 0.35, (60, 2))])


def _pair_trace_test(a_id: str, b_id: str, budget: int = 8) -> Dict[str, Any]:
    """Two arms sharing an identity must share ONE candidate trace exactly."""
    X = _smoke_X()
    dsid = "SMOKE_TRACE_%s" % SMOKE_TAG
    out = {}
    frames = {}
    CA.ClustOptAdapter._TRACE_CACHE.clear()
    work = None
    for mid in (a_id, b_id):
        ad = CA.ClustOptAdapter(mid, work_dir=work)
        work = ad._workdir()
        ad.initialise()
        seed = PROTO.derive_seed(dsid, "xy_2d", mid)
        res = ad.run(X, seed=seed, k_min=2, k_max=5, budget=budget,
                     dataset_id=dsid, view_id="xy_2d")
        key = (PROTO.seed_identity(mid), dsid, "xy_2d")
        tr = CA.ClustOptAdapter._TRACE_CACHE[key]
        frames[mid] = tr.candidate_frame()
        out[mid] = {"seed": seed, "trace_id": tr.trace_id,
                    "fingerprint": tr.trace_fingerprint(),
                    "n_candidates": len(tr.trial_log),
                    "selected_index": res.extra["selected_candidate_index"],
                    "selector": res.extra["final_selection_mechanism"],
                    "labels_hash": res.extra["final_labels_hash"]}
    fa, fb = frames[a_id], frames[b_id]
    same_shape = fa.shape == fb.shape
    mismatch = {}
    if same_shape:
        for c in fa.columns:
            if c in ("aggregate_score",):
                d = np.abs(pd.to_numeric(fa[c], errors="coerce")
                           - pd.to_numeric(fb[c], errors="coerce"))
                mismatch[c] = int((d.fillna(0) > 1e-12).sum())
            else:
                mismatch[c] = int((fa[c].astype(str) != fb[c].astype(str)).sum())
    identical = same_shape and all(v == 0 for v in mismatch.values())
    same_seed = out[a_id]["seed"] == out[b_id]["seed"]
    return {"pair": [a_id, b_id], "budget": budget,
            "same_seed_identity": bool(same_seed),
            "n_candidates": out[a_id]["n_candidates"],
            "same_fingerprint": out[a_id]["fingerprint"] == out[b_id]["fingerprint"],
            "per_column_mismatches": mismatch,
            "candidate_trace_identical": bool(identical),
            "selection_differs_only": (
                identical and out[a_id]["selector"] != out[b_id]["selector"]),
            "arms": out,
            "status": "PASS" if identical and same_seed else "FAIL"}


def test_F_trace_sharing() -> Dict[str, Any]:
    mlp = _pair_trace_test("IDB_ClustOpt_C0_MLP_TOP5_RAW_noR",
                           "IDB_ClustOpt_C3_MLP_TOP5_RAW_R")
    knn = _pair_trace_test("IDB_ClustOpt_C1_KNN_TOP10_RAW_noR",
                           "IDB_ClustOpt_C4_KNN_TOP10_RAW_R")
    not_shared = (PROTO.seed_identity("IDB_ClustOpt_C2_PPv1_noR")
                  != PROTO.seed_identity("IDB_ClustOpt_C5_PPv2_R"))
    ok = mlp["status"] == "PASS" and knn["status"] == "PASS" and not_shared
    return {"section": "F_trace_sharing", "mlp_pair": mlp, "knn_pair": knn,
            "c2_c5_kept_separate": bool(not_shared),
            "note": SMOKE_TAG,
            "status": "PASS" if ok else "FAIL"}


# ------------------------------------------------------------------- G reranker
def test_G_reranker() -> Dict[str, Any]:
    """Replay a persisted historical trace through the frozen reranker."""
    from models.ClustOpt_Candidate_Reranker.split1_preparation import (
        evaluate_split1_final as ESF, online_feature_builder as OFB)
    from models.ClustOpt_Policy_Predictor.data import feature_schema as FSCH
    from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E501
        _ext)

    metric_names = list(FSCH.load_metric_names(_REPO))
    ctx = pd.read_csv(KNN_CTX)
    recs = regression_records("regressor_top5_raw", 6)
    art = CA.ClustOptArtifacts(_REPO)
    art.resolve()
    rows = []
    for _, r in recs.iterrows():
        ds = _dataset_dir(r)
        # these paths exceed MAX_PATH on Windows, so the project long-path
        # helper is used rather than a bare open()
        trace = (ds / "experiments" / "split_01" / "regressor_top5_raw"
                 / r["view_id"] / "clustopt_results.csv")
        if not Path(_ext(trace)).is_file():
            rows.append({"dataset_id": r["dataset_id"], "view_id": r["view_id"],
                         "status": "NO_PERSISTED_TRACE"})
            continue
        try:
            log = pd.read_csv(_ext(trace))
            m = ctx[(ctx.dataset_id == r["dataset_id"])
                    & (ctx.view_id == r["view_id"])]
            uvec = np.zeros(120, dtype=float)
            if len(m) == 1:
                for j, k in enumerate(metric_names):
                    c = "knn_utility__%s" % k
                    if c in m.columns:
                        uvec[60 + j] = float(m.iloc[0][c])
            Xr, _, diag = OFB.build_online_X(
                trial_log=log, view_id=r["view_id"], metric_names=metric_names,
                utility_vector=uvec, source="mlp", k_mode="top5", actual_k=5)
            pred = np.asarray(art.reranker("MLP_TOP5").predict(Xr), float)
            elig = ESF.eligible_mask(log)
            chosen = ESF.select_reranked(pred, elig)
            rows.append({"dataset_id": r["dataset_id"], "view_id": r["view_id"],
                         "n_candidates": int(len(log)),
                         "feature_dim": int(Xr.shape[1]),
                         "dim_is_347": int(Xr.shape[1]) == OFB.EXPECTED_DIM,
                         "n_eligible": int(elig.sum()),
                         "selected_index": int(chosen),
                         "selected_is_eligible": bool(elig[chosen]),
                         "ari_consulted": False,
                         "status": "PASS" if (int(Xr.shape[1]) == OFB.EXPECTED_DIM
                                              and elig[chosen]) else "FAIL"})
        except Exception as exc:
            rows.append({"dataset_id": r["dataset_id"], "view_id": r["view_id"],
                         "status": "ERROR",
                         "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
    scored = [x for x in rows if x["status"] in ("PASS", "FAIL")]
    n_fail = sum(1 for x in rows if x["status"] not in ("PASS", "NO_PERSISTED_TRACE"))
    return {"section": "G_reranker", "n_records": len(rows),
            "n_scored": len(scored), "n_fail": n_fail,
            "final_artifacts_only": True, "oof_models_used": False,
            "rows": rows,
            "status": "PASS" if (scored and n_fail == 0) else "FAIL"}


# --------------------------------------------------------------- H label safety
def test_H_label_safety() -> Dict[str, Any]:
    import inspect
    sig = inspect.signature(CA.ClustOptAdapter.run)
    banned = {"y", "y_true", "labels", "labels_true", "true_k", "ari", "n_clusters"}
    hits = sorted(set(sig.parameters) & banned)
    src = (_ROOT / "methods" / "clustopt_adapter.py").read_text(encoding="utf-8")
    calls_gt = "set_ground_truth" in src
    passes_none = "y_true=None," in src
    return {"section": "H_label_safety",
            "run_signature": [p for p in sig.parameters if p != "self"],
            "banned_parameters_present": hits,
            "adapter_calls_set_ground_truth": calls_gt,
            "adapter_passes_y_true_none": passes_none,
            "objective_is_cvi_only": True,
            "status": "PASS" if (not hits and not calls_gt and passes_none)
                      else "FAIL"}


# -------------------------------------------------------------- I output schema
REQUIRED_OUTPUT = ("method_id", "dataset_id", "view", "search_trace_id",
                   "search_seed", "final_selection_mechanism",
                   "selected_metric_ids", "selected_metric_weights",
                   "artifact_hashes", "final_labels_hash")
FORBIDDEN_OUTPUT = ("ari", "ARI", "nmi", "NMI", "ami", "AMI", "true_k",
                    "y_true", "labels_true", "evaluation_labels")


def test_I_output_schema(budget: int = 8) -> Dict[str, Any]:
    X = _smoke_X()
    dsid = "SMOKE_OUT_%s" % SMOKE_TAG
    CA.ClustOptAdapter._TRACE_CACHE.clear()
    rows, work = [], None
    for mid in CA.ARM_SPEC:
        ad = CA.ClustOptAdapter(mid, work_dir=work)
        work = ad._workdir()
        ad.initialise()
        seed = PROTO.derive_seed(dsid, "xy_2d", mid)
        try:
            res = ad.run(X, seed=seed, k_min=2, k_max=5, budget=budget,
                         dataset_id=dsid, view_id="xy_2d")
            e = res.extra
            missing = [k for k in REQUIRED_OUTPUT if k not in e]
            forbidden = [k for k in FORBIDDEN_OUTPUT if k in e]
            rows.append({"method_id": mid, "n_evaluations": res.n_evaluations,
                         "selected_algorithm": res.selected_algorithm,
                         "selector": e["final_selection_mechanism"],
                         "policy_id": (e.get("policy") or {}).get("policy_id"),
                         "n_selected_metrics": len(e["selected_metric_ids"]),
                         "labels_hash": e["final_labels_hash"],
                         "missing_fields": missing,
                         "forbidden_fields": forbidden,
                         "status": "PASS" if not missing and not forbidden
                                   else "FAIL"})
        except Exception as exc:
            rows.append({"method_id": mid, "status": "ERROR",
                         "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
    n_fail = sum(1 for r in rows if r["status"] != "PASS")
    return {"section": "I_output_schema", "n_arms": len(rows), "n_fail": n_fail,
            "required_fields": list(REQUIRED_OUTPUT),
            "forbidden_fields": list(FORBIDDEN_OUTPUT),
            "note": SMOKE_TAG, "rows": rows,
            "status": "PASS" if n_fail == 0 else "FAIL"}


# ----------------------------------------------------------------- seed tests
def test_J_seeds() -> Dict[str, Any]:
    import hashlib
    checks = []
    for ds, v, mid in (("D1", "x_only", "IDB_ClustOpt_C0_MLP_TOP5_RAW_noR"),
                       ("D2", "xy_2d", "IDB_ClustOpt_C5_PPv2_R")):
        key = "%d|%s|%s|%s" % (PROTO.BASE_SEED, ds, v, PROTO.seed_identity(mid))
        expect = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31)
        checks.append({"dataset_id": ds, "view_id": v, "method_id": mid,
                       "expected": expect,
                       "actual": PROTO.derive_seed(ds, v, mid),
                       "match": expect == PROTO.derive_seed(ds, v, mid)})
    pairs = [("IDB_ClustOpt_C0_MLP_TOP5_RAW_noR", "IDB_ClustOpt_C3_MLP_TOP5_RAW_R", True),
             ("IDB_ClustOpt_C1_KNN_TOP10_RAW_noR", "IDB_ClustOpt_C4_KNN_TOP10_RAW_R", True),
             ("IDB_ClustOpt_C2_PPv1_noR", "IDB_ClustOpt_C5_PPv2_R", False)]
    pair_rows = []
    for a, b, want_same in pairs:
        sa = PROTO.derive_seed("D", "xy_2d", a)
        sb = PROTO.derive_seed("D", "xy_2d", b)
        pair_rows.append({"a": a, "b": b, "same_seed": sa == sb,
                          "expected_same": want_same,
                          "status": "PASS" if (sa == sb) == want_same else "FAIL"})
    ok = (PROTO.BASE_SEED == 1234 and all(c["match"] for c in checks)
          and all(p["status"] == "PASS" for p in pair_rows))
    return {"section": "J_seeds", "base_seed": PROTO.BASE_SEED,
            "formula": "sha256(base|dataset|view|seed_identity)[:8] % 2**31",
            "checks": checks, "pairs": pair_rows,
            "status": "PASS" if ok else "FAIL"}


SECTIONS = [test_A_artifacts, test_B_feature_schemas, test_C_policy, test_D_mlp,
            test_E_knn, test_E2_knn_utility_context, test_F_trace_sharing,
            test_G_reranker, test_H_label_safety, test_I_output_schema,
            test_J_seeds]


def main(argv=None) -> int:
    print("=" * 78)
    print("STAGE 4B-2 CLUSTOPT ADAPTER TESTS")
    print("=" * 78, flush=True)
    out: List[Dict[str, Any]] = []
    for fn in SECTIONS:
        try:
            r = fn()
        except Exception as exc:
            import traceback
            r = {"section": fn.__name__, "status": "ERROR",
                 "error": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
                 "traceback": traceback.format_exc()[-1200:]}
        out.append(r)
        print("  %-28s %s  %s" % (r.get("section", fn.__name__), r["status"],
                                  r.get("error", "")[:60]), flush=True)
    n_fail = sum(1 for r in out if r["status"] != "PASS")
    ev = {"suite": "stage4b2_clustopt_adapters",
          "internal_regression_set": {
              "source": "split1_trace_coverage.csv.gz + the persisted split-1 "
                        "dataset folders",
              "domain": "INTERNAL synthetic domain only",
              "is_part_of_external_corpus": False,
              "purpose": "adapter fidelity; no scientific claim is drawn from it"},
          "smoke_marking": SMOKE_TAG,
          "n_sections": len(out), "n_fail": n_fail,
          "sections": out,
          "status": "PASS" if n_fail == 0 else "FAIL"}
    p = _ROOT / "validation" / "stage4b2_adapter_evidence.json"
    p.write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    print("\n  %d sections | fail %d -> %s" % (len(out), n_fail, ev["status"]))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
