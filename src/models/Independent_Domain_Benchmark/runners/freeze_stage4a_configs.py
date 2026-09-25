"""Stage 4A: emit the frozen configuration set for the independent-domain benchmark.

Writes configuration only. No dataset is clustered, no model is fitted or loaded
for inference, and no result is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
CFG = _ROOT / "configs"
REPO = _ROOT.parents[1]
DERIVED = "experiments/external_baselines/Unified_MKR/derived"
RERANK = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b5a_final_models_and_split1_preparation")
PPREPLAY = ("results_analysis/clustopt_policy_predictor/"
            "stage2c_final_v2_split1_online_replay")
MLP_RUN = "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout"
KNN_RUN = "results_analysis/metric_utility_knn/phase_e2"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "MISSING"


def w(name: str, obj: Any) -> str:
    txt = json.dumps(obj, indent=2, default=str)
    (CFG / name).write_text(txt, encoding="utf-8")
    h = hashlib.sha256(txt.encode()).hexdigest()[:16]
    print("  wrote %-38s sha=%s" % (name, h), flush=True)
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else REPO
    CFG.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    sub = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo / "external/ml2dac",
                         capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 4A  FREEZE CONFIGURATION")
    print("=" * 78, flush=True)

    D = repo / DERIVED
    RR = repo / RERANK

    # =================================================== method registry (14)
    clustopt_shared = {
        "utility_model_run_dir": MLP_RUN,
        "utility_model_config_sha256_16": sha(repo / MLP_RUN / "config.json"),
        "utility_model_fold0_sha256_16": sha(repo / MLP_RUN / "folds/fold_0/best_model.pt"),
        "checkpoint_policy": "fold_ensemble",
        "training_splits": list(range(2, 17)), "heldout_split": 1,
        "retrained_for_external": False,
    }
    rr = json.loads((RR / "final_reranker_registry.json").read_text("utf-8"))
    rr_by = {x["reranker_id"]: x for x in rr["rerankers"]}

    def co(mid, label, source, k, weighting, reranker, pp, note):
        d = {"method_id": mid, "framework": "ClustOpt", "label": label,
             "utility_source": source, "top_k": k, "weighting": weighting,
             "reranker_id": reranker, "policy_predictor": pp,
             "retrained_for_external": False, "notes": note}
        d.update(clustopt_shared)
        if source == "knn":
            d["utility_model_run_dir"] = KNN_RUN
        if reranker == "per_policy":
            # PP v2's 20-way argmax may land on any policy, so every frozen
            # reranker regime must be loadable; the one actually applied is
            # chosen per row by the policy the predictor selects.
            d["reranker_model_file"] = "%s/final_models/<regime>/model.joblib" % RERANK
            d["reranker_registry"] = "%s/final_reranker_registry.json" % RERANK
            d["reranker_sha256_16"] = {x["reranker_id"]: x["model_file_sha256_16"]
                                       for x in rr["rerankers"]}
            d["reranker_n_features"] = sorted({x["n_features"] for x in rr["rerankers"]})
            d["n_rerankers_required"] = rr["n_rerankers"]
        elif reranker:
            d["reranker_model_file"] = "%s/final_models/%s/model.joblib" % (RERANK, reranker)
            d["reranker_sha256_16"] = rr_by[reranker]["model_file_sha256_16"]
            d["reranker_n_features"] = rr_by[reranker]["n_features"]
        if pp:
            man = "final_v1_model_manifest.json" if pp == "PP_v1" else "final_v2_model_manifest.json"
            m = json.loads((repo / PPREPLAY / man).read_text("utf-8"))
            d["pp_manifest"] = "%s/%s" % (PPREPLAY, man)
            d["pp_config_hash"] = m["config_hash"]
            d["pp_target_semantics"] = m["target_semantics"]
            d["pp_seed"] = m["seed"]
            d["pp_training_splits"] = m["training_splits"]
            d["pp_input_dim"] = m["input_dim"]
            d["pp_output_dim"] = m["output_dim"]
            d["pp_model_binary_persisted"] = False
            d["pp_blocker"] = ("BLOCKER: the final PP model binary was never "
                               "dumped by any pipeline script. It is deterministically "
                               "reconstructible (ExtraTrees, random_state=42, frozen "
                               "config hash, frozen training splits 2-16) and "
                               "reconstruction is bit-exactly verifiable against the "
                               "stored split1_%s_policy_selections.csv.gz."
                               % pp.split("_")[1])
        return d

    methods: List[Dict[str, Any]] = [
        co("IDB_ClustOpt_C0_MLP_TOP5_RAW_noR", "C0 MLP Top-5 RAW, no reranker",
           "mlp", 5, "normalized_positive", None, None,
           "continuity with the frozen Stage-1 arm stage1_full60_mlp_top5_raw_fixed50"),
        co("IDB_ClustOpt_C1_KNN_TOP10_RAW_noR", "C1 KNN Top-10 RAW, no reranker",
           "knn", 10, "normalized_positive", None, None,
           "Phase-E2 KNN utility predictor; K=10 matches the KNN_TOP10 reranker regime"),
        co("IDB_ClustOpt_C2_PPv1_noR", "C2 Policy Predictor v1, no reranker",
           "policy", None, "policy_selected", None, "PP_v1",
           "PP v1 target semantics are ORIGINAL policy ARI -> the scientifically "
           "aligned no-reranker PP arm"),
        co("IDB_ClustOpt_C3_MLP_TOP5_RAW_R", "C3 MLP Top-5 RAW + reranker",
           "mlp", 5, "normalized_positive", "MLP_TOP5", None,
           "shares its search trace with C0; only final candidate selection differs"),
        co("IDB_ClustOpt_C4_KNN_TOP10_RAW_R", "C4 KNN Top-10 RAW + reranker",
           "knn", 10, "normalized_positive", "KNN_TOP10", None,
           "shares its search trace with C1; only final candidate selection differs"),
        co("IDB_ClustOpt_C5_PPv2_R", "C5 Policy Predictor v2 + reranker",
           "policy", None, "policy_selected", "per_policy", "PP_v2",
           "PP v2 target semantics are RERANKER-AWARE -> the final deployed system; "
           "the reranker used is whichever regime PP v2 selects, so all 10 frozen "
           "rerankers must be loadable"),
    ]
    for tag, v, live in (("M0", "original_cvis", 7),
                         ("M1", "original_plus_established", 17),
                         ("M2", "original_plus_new46", 51),
                         ("M3", "extended_cvis", 61)):
        d = D / "ml2dac_split1" / v
        methods.append({
            "method_id": "IDB_ML2DAC_%s_%s" % (tag, v), "framework": "ML2DAC",
            "arm": tag, "derived_repository": "%s/ml2dac_split1/%s" % (DERIVED, v),
            "live_cvi_count": live,
            "cvi_classifier_sha256_16": sha(d / "cvi_classifier.pkl"),
            "imputer_sha256_16": sha(d / "imputer.pkl"),
            "nearest_neighbor_index_sha256_16": sha(d / "nearest_neighbor_index.pkl"),
            "warmstart_repository_sha256_16": sha(d / "warmstart_repository.parquet"),
            "training_splits": list(range(2, 17)), "heldout_split": 1, "seed": 1234,
            "retrained_for_external": False, "external_recalibration": False,
            "learned_candidate_set_changed": False,
            "note": ("the two internally all-NaN New46 metrics are NOT added as new "
                     "classifier classes; the learned candidate set is frozen"),
        })
    for tag, v in (("A0", "original_cvis"), ("A1", "original_plus_established"),
                   ("A2", "original_plus_new46"), ("A3", "extended_cvis")):
        d = D / "autoclust_split1" / v
        methods.append({
            "method_id": "IDB_AutoClust_%s_%s" % (tag, v), "framework": "AutoClust",
            "arm": tag, "model_root": "%s/autoclust_split1/%s" % (DERIVED, v),
            "ari_predictor_sha256_16": sha(d / "ari_predictor.pkl"),
            "algorithm_selector_sha256_16": sha(d / "algorithm_selector.pkl"),
            "training_splits": list(range(2, 17)), "heldout_split": 1, "seed": 1234,
            "retrained_for_external": False,
            "note": ("selector binaries differ byte-wise between arms but were "
                     "verified functionally identical on all 51,204 master "
                     "metafeature rows in Stage 3A-2; the CVI-dependent component "
                     "is the ARI predictor alone"),
        })
    mh = w("method_registry.json", {
        "n_methods": len(methods),
        "expected": {"ClustOpt": 6, "ML2DAC": 4, "AutoClust": 4, "total": 14},
        "frozen_before_any_external_execution": True,
        "no_method_may_be_added_after_outcomes": True,
        "source_commit": commit, "external_submodule_commit": sub,
        "methods": methods})

    # ==================================================== benchmark protocol
    bp = w("benchmark_protocol.json", {
        "protocol_id": "idb_bench_v1", "source_commit": commit,
        "search": {"budget_evaluations": 50, "k_min": 2, "k_max": 5,
                   "search_space_source": "clustopt_phaseE2_map",
                   "externally_tuned": False,
                   "note": ("the same previously frozen matched-search-space "
                            "protocol and common budget; this does NOT claim every "
                            "framework can instantiate every identical candidate")},
        "reachable_portfolio_differences": {
            "ClustOpt": "full phaseE2 grid incl. hdbscan_constrained variants",
            "AutoClust": "kmeans, minibatch_kmeans, gmm, agglomerative, birch, hdbscan "
                         "(observed selected-algorithm support in Stage 3A)",
            "ML2DAC": "warmstart-driven; algorithm reduction may exclude families "
                      "per dataset by design",
            "policy": "differences are recorded, never equalised by modification"},
        "density_methods": {"DBSCAN_HDBSCAN": "prior behaviour preserved",
                            "ground_truth_K_forced": False,
                            "note": "ground-truth K is evaluation metadata only"},
        "views": {"1d_x": "x_only", "1d_y": "y_only", "2d": "xy_2d"},
        "seed_protocol": {
            "scheme": "SHA256-based, reusing the project convention",
            "formula": ("seed = int(sha256(f'{base_seed}|{dataset_id}|{view_id}|"
                        "{seed_identity}').hexdigest()[:8], 16) % 2**31"),
            "base_seed": 1234,
            "seed_identity": ("the SEARCH identity, not the method id: arms that "
                              "share a search trace share seed_identity so their "
                              "traces are bit-identical"),
            "trace_groups": {"clustopt_mlp_top5": ["C0", "C3"],
                             "clustopt_knn_top10": ["C1", "C4"],
                             "clustopt_ppv1": ["C2"], "clustopt_ppv2": ["C5"],
                             "ml2dac_per_arm": ["M0", "M1", "M2", "M3"],
                             "autoclust_per_arm": ["A0", "A1", "A2", "A3"]},
            "derived_from_outcomes": False},
        "trace_reuse": {
            "shared": [["C0", "C3"], ["C1", "C4"]],
            "justification": ("the reranker changes only FINAL candidate selection; "
                              "the utility source, K, weighting, search space, budget "
                              "and seed are identical, so the visited slate is "
                              "identical by construction"),
            "not_shared": [["C2"], ["C5"]],
            "not_shared_reason": ("PP v1 and PP v2 may select different policies, so "
                                  "their searches legitimately differ"),
            "runtime_rule": ("a reranker arm's reported runtime = the shared search "
                             "cost + its own final-selection cost; NEVER the "
                             "incremental reranker time alone"),
            "physical_traces_for_six_outputs": 4},
        "endpoints": {"P1": "BestView ARI (historical continuity)",
                      "P2": "xy_2d ARI (deployable external representation)",
                      "both_always_reported": True,
                      "secondary": ["NMI", "AMI", "FMI", "purity", "k_error"]},
        "primary_comparisons": {
            "headline": [["IDB_ClustOpt_C5_PPv2_R", "IDB_AutoClust_A3_extended_cvis"],
                         ["IDB_ClustOpt_C5_PPv2_R", "IDB_ML2DAC_M3_extended_cvis"]],
            "clustopt_mechanism": [["C3", "C0"], ["C4", "C1"], ["C5", "C2"]],
            "autoclust_metric_transfer": [["A1", "A0"], ["A2", "A0"], ["A3", "A1"],
                                          ["A3", "A2"], ["A3", "A0"]],
            "ml2dac_metric_transfer": [["M1", "M0"], ["M2", "M0"], ["M3", "M1"],
                                       ["M3", "M2"], ["M3", "M0"]],
            "exhaustive_pairwise_is_primary": False},
        "reporting": {"per_source": ["fcps", "sklearn_shapes", "real_pca"],
                      "dataset_macro": True,
                      "source_balanced_macro": "(mean_fcps + mean_sklearn + mean_real)/3",
                      "both_macros_always_shown": True},
        "statistics": {"inferential_unit": "dataset",
                       "bootstrap": "percentile, group-aware",
                       "grouping": ("sklearn easy/medium/hard from one generator are "
                                    "one dependence group; bootstrap resamples "
                                    "generator groups, not the 15 rows"),
                       "test": "Wilcoxon signed-rank where valid",
                       "report": ["mean paired delta", "median", "95% CI",
                                  "wins/losses/ties", "effect size"],
                       "no_view_pseudo_replication": True},
    })

    # ====================================================== runtime protocol
    rt = w("runtime_protocol.json", {
        "protocol_id": "idb_runtime_v1",
        "headline": "ONLINE_RUNTIME = T1 - T0",
        "T0": ("immediately before method-specific online inference/search, with the "
               "frozen X for one view already materialised in memory"),
        "T1": "when final cluster labels have been selected and materialised",
        "included": {
            "ClustOpt": ["method-required meta-feature extraction", "utility inference",
                         "policy inference where applicable", "metric/policy selection",
                         "clustering search", "CVI evaluation", "final J selection",
                         "reranker inference where applicable"],
            "AutoClust": ["method-required meta/landmark features",
                          "frozen algorithm-selector inference", "candidate clustering",
                          "CVI vector computation", "frozen ARI-predictor inference",
                          "final selection"],
            "ML2DAC": ["method-required meta-feature extraction",
                       "nearest-neighbour lookup", "frozen CVI classifier",
                       "warmstart selection", "clustering search",
                       "selected-CVI evaluation", "final selection"]},
        "excluded": ["downloading", "parsing raw files",
                     "imputation/scaling/PCA benchmark preparation",
                     "model-file loading from disk", "offline ground-truth metrics",
                     "result serialization", "plotting"],
        "MODEL_INIT_RUNTIME": "separately measured cost of loading frozen artifacts",
        "TOTAL_COLD_RUNTIME": "model init + online; reported separately, never mixed "
                              "with the headline",
        "workers": 8,
        "worker_policy": ("exactly 8; if 8 workers are unsafe on the target machine "
                          "execution must STOP rather than silently reduce"),
        "thread_policy": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                          "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                          "reason": "prevent nested oversubscription under 8 workers"},
        "watchdog": {
            "common_seconds": 1800,
            "identical_across_methods": True,
            "purpose": "emergency hang/resource-failure catch ONLY",
            "evidence": ("historical per-(dataset,view) worst cases: AutoClust A2 "
                         "p95 total-all-views 719 s (~240 s per view); ML2DAC "
                         "materially faster. 1800 s per (dataset, view) is >7x the "
                         "observed p95 per-view cost and cannot truncate normal "
                         "execution"),
            "replaces": ("the historical 150 s ClustOpt-only online cap, which was "
                         "NOT comparable across frameworks and is not carried over"),
            "scientific_budget": "50 evaluations for every method"},
    })

    # ================================================= domain-shift protocol
    ds = w("domain_shift_protocol.json", {
        "protocol_id": "idb_ood_v1",
        "single_score_declared_in_advance": True,
        "reference_population": ("Unified_MKR master metafeatures.parquet, training "
                                 "splits 2-16 only"),
        "external_features": ("the same label-free meta-feature extractor the "
                              "training repository used, applied to the frozen 2-D "
                              "external representation"),
        "score": {
            "name": "standardized_nn_distance",
            "definition": ("z-score each meta-feature using the TRAINING mean/std, "
                           "then take the Euclidean distance from the external row to "
                           "its nearest training row in that standardized space"),
            "aggregate": "median over the dataset's three views",
            "reported_alongside": ["fraction_of_metafeatures_outside_training_range",
                                   "per_feature_percentile"]},
        "only_one_ood_score": ("exactly one aggregate score is declared here; "
                               "additional descriptive quantities are reported but "
                               "no alternative aggregate may be introduced later and "
                               "chosen for being favourable"),
        "must_not_affect": ["dataset acceptance", "model selection", "hyperparameters",
                            "method execution", "any design decision"],
        "analysis_only": True})

    # ============================================ external metric hypotheses
    hy = w("external_metric_hypotheses.json", {
        "source_stage": "3C", "status": "HYPOTHESES, not truths",
        "H1_core_reproduction": {
            "internal_core": ["noise_aware_silhouette", "silhouette", "gridness_fft_acf"],
            "internal_evidence": "Top-5 by utility AND selection in >=6/12 families; "
                                 "identical at 5/12, 6/12, 7/12",
            "external_test": "do these remain broadly useful on the external corpus?"},
        "H2_specialist_reproduction": {
            "named_candidates": ["connected_components_count", "contour_graph_connectivity",
                                 "axial_symmetry", "ribbon_thickness", "fractal_dimension",
                                 "parallel_bands", "convexity_ratio",
                                 "corner_junction_density", "gridness_fft_acf"],
            "rule": "carried forward in full; none may be deleted after outcomes"},
        "H3_fine_resolution": {
            "internal_evidence": "27 of 42 Tier-A subfamily specialists (64%) were "
                                 "invisible at family level",
            "external_requirement": "do not rely on a single global average"},
        "H4_image_vs_pattern": {
            "internal": "Image 8 family / 26 subfamily Tier-A vs Pattern 6 / 16",
            "external_test": "does the Image>Pattern ordering survive?"},
        "H5_gridness_interpretation": {
            "warning": "do NOT assume gridness_fft_acf is only a grid detector",
            "internal_evidence": "it is a core metric (9/12 families) and led on "
                                 "texture and radial/spiral families",
            "treat_as": "broadly useful frequency/periodicity-derived signal"},
        "H6_ladder_counterexample": {
            "internal": "gridness_fft_acf was a Tier-A specialist in ladder/grid, yet "
                        "that family had the lowest New46 oracle-win rate (0.370) and "
                        "the only negative Full60-Head14 gain (-0.0156)",
            "principle": "metric usefulness != guaranteed end-to-end gain"},
        "H7_capture_vs_existence": {
            "internal": "ML2DAC Tier-A specialist recall <=0.40 and exactly 0.000 in "
                        "two families where those metrics were the offline optimum",
            "external_requirement": "record not only whether a locally useful metric "
                                    "exists but whether each method can capture it"},
    })

    # ============================================== metric registry + fixed32
    mr = w("metric_registry.json", {
        "existing_inventory": {
            "original": 7, "established": 10, "new46_designed": 46,
            "new46_live_internally": 44, "extended_live_internally": 61,
            "internally_dead": ["convexity_ratio_image_based", "hough_arc_circle_strength"],
            "internally_dead_external_policy": (
                "ATTEMPTED externally as ANALYSIS metrics: dead on the internal "
                "repository does not imply dead on a different corpus. They are NOT "
                "added to any frozen learned baseline model.")},
        "utility_definition": {
            "reuse": "EXACT canonical ClustOpt utility from Stage 3B",
            "source": "models/ClustOpt/external_evaluation/compute_metric_utility.py",
            "function": "compute_metric_utilities",
            "invented_for_stage4": False},
        "fixed32": {
            "reuse_source": ("models/Clustering_Repository_Builder/experiments/"
                             "experiment_execution/offline_method_runner.py "
                             "(EXECUTION_MODE = 'offline_fixed32')"),
            "n_candidates": 32,
            "per": "(dataset, view)",
            "independent_of_method_search": True,
            "purpose": "prevents metric-quality analysis being confounded by any "
                       "method's search policy",
            "executed_in_stage4a": False},
        "endpoints": ["global utility", "rank", "top1", "top3", "top5", "top10",
                      "oracle_win_frequency", "best_established_comparator_difference",
                      "best_modern_comparator_difference",
                      "complementarity_unique_win_coverage", "source_level_utility",
                      "challenge_tag_utility", "runtime_cost_where_available"],
        "multiplicity": {"method": "Benjamini-Hochberg",
                         "families": ["per-index vs established comparator pool",
                                      "per-index vs modern comparator pool",
                                      "structural-tag uplift", "group contrasts"],
                         "report": ["raw p", "adjusted q", "test family"]},
    })

    # ================================================ modern comparator audit
    cv = w("external_cvi_comparator_audit.json", {
        "audit_date": "stage 4A", "implemented_in_stage4a": False,
        "policy": ("audit only; implementation is authorised in Stage 4B ONLY for "
                   "comparators whose definition and numerics are fully resolved. "
                   "No missing comparator may be silently substituted."),
        "already_available_in_inventory": ["dbcv", "s_dbw", "silhouette",
                                           "noise_aware_silhouette", "calinski_harabasz",
                                           "davies_bouldin", "dunn_index", "cop",
                                           "coggins_jain_index"],
        "installed_packages_checked": {"validclust": "NOT installed",
                                       "s_dbw": "NOT installed (metric already native)",
                                       "clusteval": "NOT installed",
                                       "cdbw": "NOT installed",
                                       "pyclustering": "NOT installed"},
        "CDbw": {
            "reference": "Halkidi & Vazirgiannis (2008), 'A density-based cluster "
                         "validity approach using multi-representatives'",
            "direction": "maximize",
            "inputs": "X + labels; needs multiple representatives per cluster",
            "k_assumption": "K>=2; undefined for K=1",
            "noise_handling": "no native noise class; DBSCAN/HDBSCAN noise must be "
                              "handled by an explicit, declared rule",
            "python_in_repo": False,
            "external_package": "PyPI 'cdbw' v0.2, MIT",
            "edge_cases": "shrinking factor s and representative count r are free "
                          "parameters; results vary with them",
            "status": "AVAILABLE BUT UNVERIFIED — parameters must be pinned and the "
                      "implementation numerically checked before use"},
        "CVNN": {
            "reference": "Liu, Li, Xiong, Gao, Wu (2013), 'Understanding and "
                         "Enhancement of Internal Clustering Validation Measures'",
            "direction": "minimize",
            "inputs": "X + labels + neighbourhood size k_nn",
            "k_assumption": "K>=2",
            "noise_handling": "undeclared in the source; needs an explicit rule",
            "python_in_repo": False,
            "external_package": "no trustworthy pinned PyPI implementation confirmed",
            "edge_cases": "separation term is normalised across the compared "
                          "partition set, so values are comparable only within a "
                          "fixed candidate bank — which fixed32 provides",
            "status": "NEEDS IMPLEMENTATION in Stage 4B from the paper, with the "
                      "k_nn value pinned in advance"},
        "CVDD": {
            "reference": "Hu & Zhong (2019), 'Internal validity index for fuzzy/"
                         "density-involved clustering (CVDD)'",
            "direction": "maximize",
            "inputs": "X + labels; density-aware path/relative distances",
            "k_assumption": "K>=2",
            "noise_handling": "not specified for an explicit noise label",
            "python_in_repo": False,
            "external_package": "no trustworthy pinned implementation confirmed",
            "edge_cases": "requires a path-based distance whose complexity can be "
                          "superlinear; cost must be measured before adoption",
            "status": "NEEDS IMPLEMENTATION in Stage 4B; correctness NOT yet resolved"},
        "DCSI": {
            "reference": "Gauss, Scheipl, Herrmann (2024), 'DCSI - An improved "
                         "measure of cluster separability based on separation and "
                         "connectedness'",
            "direction": "maximize",
            "inputs": "X + labels; density-based separation and connectedness",
            "k_assumption": "K>=2",
            "noise_handling": "paper discusses noise explicitly",
            "python_in_repo": False,
            "external_package": "reference implementation published with the paper; "
                                "licence must be checked before vendoring",
            "edge_cases": "depends on a minPts-like parameter that must be pinned",
            "status": "NEEDS LICENCE CHECK + IMPLEMENTATION in Stage 4B"},
        "consequence_if_unresolved": ("any comparator not implemented and verified "
                                      "cannot be claimed against; the metrics-paper "
                                      "criterion that requires comparison against "
                                      "modern CVIs would then remain unmet"),
    })

    # =========================================================== result paths
    rp = w("result_hierarchy.json", {
        "root": "results_analysis/independent_domain_benchmark",
        "not_reusing": "results_analysis/clustering_repository/analyzed_data",
        "layout": {
            "manifests/": ["dataset_manifest.json", "method_manifest.json",
                           "metric_manifest.json", "benchmark_protocol.json",
                           "provenance.json"],
            "datasets/<source>/<dataset_id>/": [
                "dataset_metadata.json", "preparation/preparation_report.json",
                "experiments/external_validation/<METHOD_ID>/{1d_x,1d_y,2d}/result.json",
                "experiments/external_validation/<METHOD_ID>/dataset_summary.json",
                "metric_validation/fixed32/{1d_x,1d_y,2d}/",
                "evaluation/offline_metrics.json"],
            "aggregations/": ["per_dataset/", "per_source/", "global/", "paired/",
                              "runtime/", "metric_validation/", "domain_shift/"],
            "reports/": []},
        "authoritative_results_created_in_stage4a": False})

    # ========================================================= result schemas
    sc = w("result_schemas.json", {
        "method_view_result": {
            "file": "experiments/external_validation/<METHOD_ID>/<view>/result.json",
            "fields": ["dataset_id", "source", "view", "method_id", "status",
                       "failure_kind", "labels_artifact", "selected_algorithm",
                       "best_configuration", "n_evaluations", "seed",
                       "search_space_id", "search_space_hash", "model_artifact_hashes",
                       "online_runtime_sec", "init_runtime_sec", "environment",
                       "protocol_hash", "code_commit", "created_at"],
            "must_not_contain": ["labels", "ground_truth_k", "any ARI or offline score"]},
        "offline_evaluation": {
            "file": "evaluation/offline_metrics.json",
            "separate_from_execution": True,
            "written_only_after": "predictions are persisted",
            "fields": ["dataset_id", "method_id", "view", "ari", "nmi", "ami", "fmi",
                       "purity", "predicted_k", "true_k", "k_error", "k_correct"]},
        "metric_validation": {
            "file": "metric_validation/fixed32/<view>/candidates.json",
            "separate_schema": True,
            "fields": ["dataset_id", "view", "candidate_id", "algorithm",
                       "hyperparameters", "predicted_k", "labels_artifact",
                       "cvi_values", "runtime_sec"],
            "ari_added_offline": True}})

    # ======================================================= resume semantics
    rs = w("resume_semantics.json", {
        "complete_iff_all_hold": [
            "result.json readable", "status == success",
            "dataset representation checksum matches",
            "METHOD_ID matches", "all model artifact hashes match",
            "view matches", "search-space hash matches", "budget matches",
            "seed matches", "benchmark-protocol hash matches",
            "code/protocol provenance matches"],
        "directory_existence_is_not_completion": True,
        "compatible_success_never_overwritten": True,
        "incomplete_or_incompatible": "may be rerun ONLY under the frozen configuration",
        "lineage": "mirrors the strict predicate added to AutoClust in Stage 3A-2, "
                   "which replaced a predicate that silently reused FAILED units"})

    hashes = {"method_registry.json": mh, "benchmark_protocol.json": bp,
              "runtime_protocol.json": rt, "domain_shift_protocol.json": ds,
              "external_metric_hypotheses.json": hy, "metric_registry.json": mr,
              "external_cvi_comparator_audit.json": cv, "result_hierarchy.json": rp,
              "result_schemas.json": sc, "resume_semantics.json": rs}
    w("config_hashes.json", {"source_commit": commit,
                             "external_submodule_commit": sub,
                             "hashes": hashes})
    print("\n  %d config files frozen" % (len(hashes) + 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
