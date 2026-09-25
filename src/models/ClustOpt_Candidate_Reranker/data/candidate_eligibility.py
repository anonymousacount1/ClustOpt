"""Frozen ELIGIBLE_CANDIDATE predicate and the candidate-oracle reconciliation.

The oracle becomes the denominator of every reranker recovery metric, so it must
name a candidate the deployed system is actually allowed to return.

Production selection (``offline_candidate_execution.run_arm_offline``)::

    eligible = [i for i, (j, v) in enumerate(zip(js, valid_mask))
                if v and j is not None]

Two conditions: the candidate is ``valid``, and its objective ``J`` is computable.
The second is implied by the first. ``load_candidate_set`` enforces that every
normalised CVI is finite on valid candidates, and that contract was verified on
the whole Splits 2-16 population: **0 of 1,161,522 valid candidates has fewer
than 60 finite normalised CVIs**. Since ``score_candidates`` returns ``None`` only
when some selected metric is non-finite, a valid candidate scores finitely under
EVERY metric subset. Eligibility is therefore policy-independent:

    ELIGIBLE_CANDIDATE(c)  <=>  valid(c)  <=>  |clusters(c) \\ {noise}| >= 2

ARI plays no part. 37,532 eligible candidates (3.23 %) have negative ARI and
remain eligible -- a target value may never gate eligibility.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import numpy as np

PREDICATE_ID = "ELIGIBLE_CANDIDATE.v1"
PREDICATE = "valid == True  (>=2 non-noise clusters)"

DEFINITION: Dict[str, Any] = {
    "predicate_id": PREDICATE_ID,
    "predicate": PREDICATE,
    "source_of_truth": ("models/Clustering_Repository_Builder/experiments/"
                        "experiment_execution/offline_candidate_execution.py::"
                        "run_arm_offline"),
    "production_expression": "valid_mask[i] and js[i] is not None",
    "valid_definition": ("Search_Results_Logger._label_stats: "
                         "valid = len(unique(labels) \\ {noise}) > 1"),
    "second_condition_is_implied": True,
    "implication_evidence": {
        "contract": ("offline_candidate_execution.load_candidate_set raises "
                     "OfflineCandidateError on any NaN/Inf normalised CVI for a "
                     "valid candidate"),
        "population_verification": ("0 of 1,161,522 valid candidates across all "
                                    "48,039 Splits 2-16 slates has fewer than 60 "
                                    "finite normalised CVIs"),
        "consequence": "J is finite for every metric subset, so eligibility does "
                       "not depend on the policy, the K regime or the weighting",
    },
    "policy_independent": True,
    "ari_participates_in_eligibility": False,
    "negative_ari_candidates_are_eligible": True,
    "n_negative_ari_eligible": 37532,
    "non_finite_objective_handling": ("a candidate whose J is non-finite is "
                                      "dropped from the eligible set; if no "
                                      "eligible candidate remains, "
                                      "run_arm_offline raises rather than "
                                      "silently falling back"),
    "oracle": "CandidateOracleARI(d,v) = max ARI over ELIGIBLE candidates",
    "bestview_oracle": "CandidateOracleBestView(d) = max_v CandidateOracleARI(d,v)",
    "excluded_and_why": ("invalid candidates -- a partition with <2 non-noise "
                         "clusters is never returnable by production, so it may "
                         "not define the oracle even when its ARI is higher"),
}

RECONCILIATION: Dict[str, Any] = {
    "question": "fixed32 dataset/view candidate oracle: 0.4979 vs 0.4949",
    "verdict": "NOT a definitional discrepancy -- the two numbers describe "
               "DIFFERENT POPULATIONS under the SAME definition",
    "old": {
        "value": 0.4979,
        "reported_in": ("docs/internal_reports/stage1/"
                        "STAGE1E_ONLINE_FIXED50_REPORT.md and "
                        "STAGE1_OFFLINE_ONLINE_PROJECT_LEAD_REVIEW.md"),
        "field": "best_possible_ari_in_candidate_set",
        "lineage": ("offline_method_runner -> OfflineRunOutcome.best_possible_ari "
                    "-> run_arm_offline ceiling = max ARI over VALID candidates"),
        "population": "SPLIT 1 only -- build_stage1c_analysis.load_population "
                      "constructs every config with split_id=1",
        "n_datasets": 1055, "n_slates": 3165,
    },
    "new": {
        "value": 0.494919,
        "reported_in": ("results_analysis/clustopt_candidate_reranker/"
                        "stage2b0_data_protocol_audit"),
        "field": "candidate_oracle_ari_valid",
        "lineage": "max ARI over candidates with part__valid == 1",
        "population": "SPLITS 2-16",
        "n_datasets": 16013, "n_slates": 48039,
    },
    "definitions_identical": True,
    "difference_decomposition": {
        "definitional_component": 0.0,
        "population_component": "the entire ~0.003 gap",
        "explanation": ("both values are the mean over dataset/view slates of the "
                        "max ARI over VALID candidates. Split 1 is a disjoint "
                        "sample of 1,055 datasets; Splits 2-16 hold the other "
                        "16,013. The per-split spread of the Splits 2-16 oracle "
                        "is itself 0.4877-0.5061, so a 0.0030 offset for a "
                        "16th held-out sample sits comfortably inside ordinary "
                        "between-split variation."),
        "per_split_range_splits_2_16": [0.4877, 0.5061],
        "split_1_reread": False,
        "note": "the 0.4979 figure is quoted from an already-published Stage-1 "
                "report; no Split-1 artifact was reopened to reconcile it",
    },
    "frozen": {
        "predicate_id": PREDICATE_ID,
        "dataset_view_oracle": 0.494919,
        "bestview_oracle": 0.659524,
        "population": "Splits 2-16",
        "changed_from_stage2b0": False,
        "reason": "Stage 2B-0 already computed the oracle over valid candidates "
                  "only, which is exactly the frozen predicate; no Stage-2B0 "
                  "number required revision",
    },
    "exception_slates": {
        "n": 33, "share": 0.000687, "all_view": "x_only",
        "description": ("slates where every ELIGIBLE candidate has a slightly "
                        "negative ARI (max -2.01e-06) while a degenerate INVALID "
                        "partition scores exactly 0"),
        "resolution": ("keep the eligible-only optimum. The invalid candidate is "
                       "unreturnable by production, so using it would define an "
                       "oracle the deployed system cannot reach and would make "
                       "reranker regret unattainable by construction. The "
                       "affected slates keep a marginally NEGATIVE oracle."),
        "numerical_impact": ("shifts the dataset/view oracle by < 1e-6; the two "
                             "conventions agree to 6 decimal places"),
        "artifact": "valid_optimum_exception_slates.csv",
    },
}


def is_eligible(valid_flag) -> np.ndarray:
    """The frozen predicate, applied to a partition-descriptor valid column."""
    return np.asarray(valid_flag).astype(bool)


def definition_hash() -> str:
    return hashlib.sha256(
        json.dumps({"predicate_id": PREDICATE_ID, "predicate": PREDICATE,
                    "policy_independent": True, "ari_gates_eligibility": False},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def candidate_oracle(ari, valid) -> float:
    """Oracle over eligible candidates for one slate."""
    a = np.asarray(ari, dtype=np.float64)
    m = is_eligible(valid)
    if not m.any():
        return float("nan")
    return float(np.nanmax(a[m]))
