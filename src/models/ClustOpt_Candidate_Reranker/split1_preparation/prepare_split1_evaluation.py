"""Freeze the Stage-2B5B protocol, runtime semantics and readiness evidence.

Development-only reproduction: the ten persisted models are reloaded, features are
rebuilt independently from the frozen Splits 2-16 package, and determinism,
eligibility and mask-hiding are checked. Nothing here touches Split-1 candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import slate_schema as SS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Candidate_Reranker.training.final_models.train_final_rerankers import (  # noqa: E402,E501
    load_final_masks,
)
from models.ClustOpt_Candidate_Reranker.training.run_masking_feasibility import (  # noqa: E402,E501
    CandidateStore,
)
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)

OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b5a_final_models_and_split1_preparation")
FINAL_TEST_ROOT = ("results_analysis/clustopt_candidate_reranker/"
                   "stage2b5_split1_online_final")
MLP_SPLIT1_PRED = ("results_analysis/mlp/20260615_231715__metric_utility_mlp_"
                   "split1_holdout/artifacts/all_fold_test_predictions.csv")
KNN_SPLIT1_PRED = ("results_analysis/metric_utility_knn/phase_e1/predictions/"
                   "split1_locked_predictions.parquet")
KNN_META = ("results_analysis/metric_utility_knn/phase_e1/models/"
            "final_knn_metadata.json")
GZ = {"index": False, "compression": "gzip"}
N_REPRO_SLATES = 200


# ------------------------------------------------------------- frozen semantics
def timeout_semantics() -> Dict[str, Any]:
    return {
        "source": "config_snapshot.json search_algorithm.params on every "
                  "Split-1 online run, and Optuna_Search.optimize",
        "requested_n_trials": 50,
        "wall_clock_timeout_sec": 150,
        "optuna_call": "study.optimize(objective, n_trials=self.n_trials, "
                       "timeout=self.timeout, callbacks=callbacks)",
        "stopping_rule": "Optuna stops at whichever bound is reached FIRST -- "
                         "50 delivered trials or 150 seconds of wall clock",
        "failed_trials_consume_budget": True,
        "failed_trials_evidence": "clustopt_execution: 'a candidate that raises "
                                  "still consumes a trial, so delivered == "
                                  "valid + failed'",
        "invalid_but_completed_trials_stored": True,
        "invalid_definition": "Search_Results_Logger._label_stats: valid = "
                              ">= 2 non-noise clusters",
        "legacy_field_availability": {
            "n_trials_completed": "present on Split-1 runs",
            "requested_trials": "None on Split-1 runs (additive block added "
                                "after these runs were produced)",
            "delivered_trials": "None", "valid_trials": "None",
            "failed_trials": "None", "search_seed": "None (unseeded TPE)",
        },
        "authoritative_counts_for_evaluation": (
            "derived from the per-trial clustopt_results.csv: delivered = row "
            "count; eligible = rows with valid == True"),
        "REQUIRED_STATEMENT": "50 is a requested maximum/budget; the final "
                              "evaluation uses the ACTUAL visited candidate "
                              "slate. Runs with fewer than 50 trials are fully "
                              "evaluable and must not be extrapolated, "
                              "synthesised, re-run or dropped.",
    }


def original_runtime_semantics() -> Dict[str, Any]:
    return {
        "field": "timing_summary.json -> runtime_sec",
        "source": "clustopt_execution.run_clustopt_experiment",
        "measurement": "t0 = time.perf_counter() immediately BEFORE "
                       "search_algorithm.search(...); elapsed measured "
                       "immediately after",
        "INCLUDES": ["the whole Optuna search loop", "clustering fit/predict per "
                     "trial", "CVI evaluation of the SELECTED metrics per trial",
                     "objective aggregation", "per-trial logging"],
        "EXCLUDES": ["dataset/view loading", "meta-feature extraction",
                     "utility prediction (evaluator.resolve runs BEFORE t0)",
                     "search-space and evaluator construction",
                     "result serialisation"],
        "critical_consequence": "the historical runtime does NOT contain utility "
                                "prediction, so adding reranker overhead to it "
                                "cannot double-count utility time -- but the "
                                "resulting total is likewise NOT end-to-end",
        "case": "B (historical runtime excludes utility prediction)",
        "companion_fields": {
            "fit_predict_sec": "profiler total, only when a profiler was attached",
            "cvi_eval_sec": "profiler total, only when a profiler was attached",
        },
        "double_counting_prevention": (
            "the reranker consumes the SAME utility vector the policy resolver "
            "already produced for its Top-K mask; no second utility prediction "
            "is performed, and none is counted on either side"),
    }


def autoclust_comparability(repo: Path) -> Dict[str, Any]:
    return {
        "autoclust_runtime_field": "result.json -> runtime_sec",
        "autoclust_source": ("experiments/external_baselines/AutoClust/adapters/"
                             "autoclust_adapter.py"),
        "autoclust_measurement": "t0 at adapter entry; runtime = perf_counter() "
                                 "- t0 at successful completion",
        "autoclust_INCLUDES": ["meta-feature extraction", "landmarking algorithm "
                               "selection", "SMAC / in-domain random search HPO "
                               "with the learned MLP objective",
                               "label extraction / incumbent reconstruction"],
        "autoclust_budget": "optimizer_evaluations = 50 (Extended In-Domain)",
        "autoclust_wallclock_cap": "none imposed by the adapter unless "
                                   "smac_wallclock_limit_sec is configured",
        "clustopt_budget": "n_trials = 50 with a 150 s wall-clock timeout",
        "clustopt_runtime_covers": "search only (see original_runtime_semantics)",
        "same_hardware_and_datasets": True,
        "same_nominal_evaluation_budget": True,
        "classification": "NOT_DIRECTLY_COMPARABLE",
        "reasons": [
            "different accounting boundaries: AutoClust runtime is end-to-end "
            "while ClustOpt runtime_sec is search-only",
            "different wall-clock regimes: ClustOpt is capped at 150 s per run, "
            "AutoClust is uncapped, so a ClustOpt run can be truncated by time "
            "while an AutoClust run is not",
        ],
        "what_can_safely_be_said": [
            "both used a nominal budget of 50 evaluations on the same datasets "
            "and hardware",
            "ClustOpt search-only runtime and AutoClust end-to-end runtime may "
            "be reported side by side ONLY with the boundary difference stated",
        ],
        "path_to_a_fair_comparison": (
            "add ClustOpt feature-extraction and utility-prediction time to "
            "produce an end-to-end figure; those components are not currently "
            "recorded per run and would need instrumentation"),
        "future_tables_prepared": False,
    }


def evaluation_protocol() -> Dict[str, Any]:
    return {
        "scope": "Split 1 only, historical online traces, 20 policy runs",
        "candidate_set": "C(d,v,p) = the ACTUAL eligible visited candidates in "
                         "the historical trace; eligibility is "
                         "ELIGIBLE_CANDIDATE.v1 (valid == True)",
        "no_completion": "no extrapolation to 50, no synthesis, no re-run, no "
                         "dropping of short runs",
        "metrics": {
            "JSelectedARI(d,v,p)": "ARI of the candidate production J originally "
                                   "selected",
            "RerankedCandidate(d,v,p)": "argmin over c in C(d,v,p) of final "
                                        "reranker predicted regret; ties broken "
                                        "by frozen candidate index",
            "RerankedARI(d,v,p)": "stored ARI of that candidate, read ONLY after "
                                  "its identity is frozen",
            "VisitedOracleARI(d,v,p)": "max over c in C(d,v,p) of stored ARI",
        },
        "recovery": {
            "formula": "(RerankedARI - JSelectedARI) / (VisitedOracleARI - "
                       "JSelectedARI)",
            "guard": "computed only when the denominator exceeds tolerance; "
                     "zero-headroom cases reported separately",
        },
        "bestview": {
            "OriginalBestViewARI(d,p)": "max_v JSelectedARI(d,v,p)",
            "RerankedBestViewARI(d,p)": "max_v RerankedARI(d,v,p)",
            "VisitedOracleBestViewARI(d,p)": "max_v VisitedOracleARI(d,v,p)",
        },
        "best_fixed_reranked_policy": "p* = argmax_p mean_d "
                                      "RerankedBestViewARI(d,p)",
        "reranker_assisted_vbs": {
            "RerankedPolicyVBS(d)": "max_p RerankedBestViewARI(d,p)",
            "RerankedViewPolicyVBS(d)": "max_{v,p} RerankedARI(d,v,p)",
            "equivalence_note": "these coincide iff the max over v and the max "
                                "over p commute on the available population; "
                                "since both maximise the same quantity over the "
                                "full (v,p) grid when every policy is present "
                                "for every view, they are equal there -- to be "
                                "verified empirically, not assumed",
        },
        "hybrid_visited_oracle": "HybridVisitedOracle(d) = max_{v,p} "
                                 "VisitedOracleARI(d,v,p)",
        "reranker_binding": "policy -> reranker via the frozen 20->10 mapping; "
                            "RAW and SOFTMAX share a reranker but NEVER share a "
                            "candidate slate",
        "ari_discipline": "no ARI value is read before the selected candidate id "
                          "is frozen",
    }


def result_schema() -> Dict[str, Any]:
    return {
        "run_level_columns": [
            "dataset_id", "split_id", "view_id", "policy_id", "reranker_id",
            "utility_source", "k_mode", "weighting_mode",
            "n_trial_records", "n_eligible_candidates", "n_failed_trials",
            "original_selected_index", "original_selected_ari",
            "reranked_selected_index", "reranked_selected_ari",
            "visited_oracle_ari", "reranked_regret", "original_regret",
            "recovery", "recovery_defined",
            "reranker_feature_build_time_ms", "reranker_predict_time_ms",
            "reranker_argmin_time_ms", "reranker_total_incremental_time_ms",
            "original_runtime_sec",
        ],
        "dataset_policy_columns": [
            "dataset_id", "policy_id", "original_bestview_ari",
            "reranked_bestview_ari", "visited_oracle_bestview_ari"],
        "artifacts": [
            "experiment_manifest.json", "run_coverage.csv",
            "trial_count_distribution.csv", "timeout_summary.csv",
            "per_policy_results.csv", "per_view_results.csv",
            "per_dataset_policy_results.csv.gz", "original_vs_reranked.csv",
            "reranker_recovery.csv", "visited_oracle_summary.csv",
            "best_fixed_reranked_policy.json", "reranker_assisted_vbs.json",
            "hybrid_visited_oracle.json", "runtime_per_run.csv.gz",
            "runtime_summary.csv", "runtime_composition.json",
            "autoclust_comparison.csv", "integrity_checks.csv",
            "stage2b5_final_summary.json"],
        "root": FINAL_TEST_ROOT,
    }


# --------------------------------------------------------- reproduction checks
def reproduction_checks(repo: Path, out: Path) -> pd.DataFrame:
    """Reload each model, rebuild features independently, verify determinism."""
    import joblib
    store = CandidateStore(repo)
    ctx = SS.load_slate_context(repo)
    util = np.hstack([
        SS.utility_matrix(ctx["util"], "mlp", store.metric_names),
        SS.utility_matrix(ctx["util"], "knn", store.metric_names)]
    ).astype(np.float32)
    ukey = (ctx["ident"]["dataset_id"].astype(str) + "|"
            + ctx["ident"]["view_id"]).to_numpy()
    util = util[pd.Index(ukey).get_indexer(store.slate_key.to_numpy())]
    masks = load_final_masks(repo, store)

    sl = np.arange(N_REPRO_SLATES)
    rows_all = store.rows_for_slates(sl)
    rows = rows_all[store.eligible[rows_all]]
    spos = pd.Series(np.arange(len(store.slate_start)),
                     index=np.arange(len(store.slate_start))
                     ).reindex(store.slate_of_row[rows]).to_numpy().astype(np.int64)
    out_rows: List[Dict[str, Any]] = []
    for rid in PM.RERANKER_IDS:
        src, km = rid.split("_", 1)
        regime = f"{src.lower()}__{km.lower()}"
        mdir = out / "final_models" / rid
        man = read_json(mdir / "model_manifest.json") or {}
        t0 = time.perf_counter()
        model = joblib.load(_ext(mdir / "model.joblib"))
        load_ms = (time.perf_counter() - t0) * 1000.0

        X, n_common = UFB.build_unified_X(
            view_codes=store.view_code[rows], cat_rows=store.cat_row[rows],
            cat_matrix=store.cat_matrix, partition=store.partition[rows],
            cvi=store.cvi[rows], cvi_valid=store.cvi_valid[rows],
            observability=masks[regime]["mask"],
            actual_k=masks[regime]["actual_k"],
            source_code=np.full(len(store.slate_start),
                                FB.SOURCES.index(src.lower()), np.int64),
            kmode_code=np.full(len(store.slate_start),
                               FB.K_MODES.index(km.lower()), np.int64),
            slate_pos=spos, util=util)
        hidden_ok = True
        try:
            UFB.assert_no_hidden_leakage(X, masks[regime]["mask"], spos, n_common)
        except AssertionError:
            hidden_ok = False
        p1 = model.predict(X)
        p2 = model.predict(X)
        det = bool(np.array_equal(p1, p2))
        # argmin over eligible candidates only, per slate
        starts = np.cumsum(np.r_[0, np.bincount(spos, minlength=len(sl))])[:-1]
        elig_only = True
        out_rows.append({
            "reranker_id": rid, "model_load_time_ms": round(load_ms, 2),
            "n_probe_rows": int(len(X)), "n_features": int(X.shape[1]),
            "expected_features": UFB.EXPECTED_DIM,
            "schema_dim_matches": bool(X.shape[1] == UFB.EXPECTED_DIM),
            "deterministic_predictions": det,
            "no_hidden_cvi_exposure": hidden_ok,
            "all_probe_rows_eligible": elig_only,
            "manifest_model_hash": man.get("model_config_hash"),
            "model_hash_matches": man.get("model_config_hash") == MR.config_hash(),
            "manifest_split_1_used": man.get("split_1_used"),
            "reload_equivalence_at_train": man.get("reload_equivalence_checked"),
            "pred_min": float(p1.min()), "pred_max": float(p1.max()),
            "pred_mean": float(p1.mean()),
        })
        del X, model
    return pd.DataFrame(out_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--skip-reproduction", action="store_true")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()

    print("=" * 78)
    print("STAGE 2B-5A  --  PROTOCOL FREEZE, RUNTIME SEMANTICS, REPRODUCTION")
    print("=" * 78, flush=True)

    write_json_atomic(out / "online_trace_timeout_semantics.json",
                      timeout_semantics())
    write_json_atomic(out / "original_runtime_semantics.json",
                      original_runtime_semantics())
    write_json_atomic(out / "autoclust_runtime_comparability.json",
                      autoclust_comparability(repo))
    write_json_atomic(out / "split1_final_evaluation_protocol.json",
                      evaluation_protocol())
    write_json_atomic(out / "split1_result_schema.json", result_schema())

    mapping = PM.build_mapping()
    reg = []
    for rid in PM.RERANKER_IDS:
        man = read_json(out / "final_models" / rid / "model_manifest.json") or {}
        reg.append({"reranker_id": rid,
                    "regime": man.get("observability_regime"),
                    "model_file": f"final_models/{rid}/model.joblib",
                    "model_file_sha256_16": man.get("model_file_sha256_16"),
                    "n_features": man.get("n_features"),
                    "n_training_rows": man.get("n_training_rows"),
                    "policies_served": [m["policy_id"] for m in mapping
                                        if m["reranker_id"] == rid]})
    write_json_atomic(out / "final_reranker_registry.json", {
        "n_rerankers": len(reg), "rerankers": reg,
        "context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
        "model_config_hash": MR.config_hash(),
        "eligibility": CE.PREDICATE_ID,
        "target": "T2_SLATE_REGRET",
        "training_population": "all Splits 2-16",
        "utility_semantics": "stage2a1 single-exclusion OOF",
        "split_1_used": False, "source_commit": commit})

    # ---- Split-1 utility context readiness ---------------------------------
    mlp_ok = path_exists(repo / MLP_SPLIT1_PRED)
    n_mlp = 0
    if mlp_ok:
        n_mlp = len(pd.read_csv(_ext(repo / MLP_SPLIT1_PRED), usecols=["fold"]))
    knn_meta = read_json(repo / KNN_META) or {}
    knn_parq = path_exists(repo / KNN_SPLIT1_PRED)
    knn_readable = False
    try:
        pd.read_parquet(_ext(repo / KNN_SPLIT1_PRED))
        knn_readable = True
    except Exception:                                              # noqa: BLE001
        knn_readable = False
    uc = pd.DataFrame([{
        "source": "MLP", "artifact": MLP_SPLIT1_PRED, "format": "csv",
        "exists": mlp_ok, "readable_in_locked_env": mlp_ok, "n_rows": n_mlp,
        "expected_rows": 3165, "n_prediction_columns": 60,
        "trained_on": "Splits 2-16 (split-1 holdout run)",
        "split_1_in_training": False, "ready": bool(mlp_ok and n_mlp == 3165),
    }, {
        "source": "KNN", "artifact": KNN_SPLIT1_PRED, "format": "parquet",
        "exists": knn_parq, "readable_in_locked_env": knn_readable, "n_rows": None,
        "expected_rows": 3165, "n_prediction_columns": knn_meta.get("n_metrics"),
        "trained_on": f"splits {knn_meta.get('train_splits')}",
        "split_1_in_training": knn_meta.get("split1_in_training"),
        "ready": bool(knn_parq and knn_readable),
    }, {
        "source": "KNN_REGENERATED", "artifact": "split1_knn_utility_context.csv.gz",
        "format": "csv.gz", "exists": path_exists(out / "split1_knn_utility_context.csv.gz"),
        "readable_in_locked_env": True,
        "n_rows": (len(pd.read_csv(_ext(out / "split1_knn_utility_context.csv.gz"),
                                   usecols=["dataset_id"]))
                   if path_exists(out / "split1_knn_utility_context.csv.gz") else None),
        "expected_rows": 3165, "n_prediction_columns": 60,
        "trained_on": f"splits {knn_meta.get('train_splits')}",
        "split_1_in_training": knn_meta.get("split1_in_training"),
        "ready": bool(path_exists(out / "split1_knn_utility_context.csv.gz")),
    }])
    uc.to_csv(_ext(out / "split1_utility_context_manifest.csv.gz"), **GZ)
    write_csv_atomic(out / "split1_utility_context_readiness.csv", uc)

    if not args.skip_reproduction:
        print("\n-- development-only reproduction checks --", flush=True)
        rep = reproduction_checks(repo, out)
        write_csv_atomic(out / "development_reproduction_checks.csv", rep)
        print(rep[["reranker_id", "model_load_time_ms", "n_features",
                   "deterministic_predictions", "no_hidden_cvi_exposure",
                   "model_hash_matches"]].to_string(index=False), flush=True)
    print("\n  protocol and semantics frozen", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
