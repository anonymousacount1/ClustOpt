"""CV semantics for the Candidate Reranker: what a future loader must obey.

Two facts drive everything here.

**The atomic unit is the dataset, not the row.** A dataset contributes 3 views x
32 candidates = 96 unique candidate rows, and each of those is observed under 10
masking regimes and 20 policy contexts. Those are re-descriptions of the same
underlying candidate, not independent samples. All of them must land in the same
outer fold; a random row split would place the SAME candidate on both sides.

**Utility-derived inputs need cross-fitting, raw candidate inputs do not.**
Meta-features, candidate configuration, partition descriptors and the offline
CVI matrix are label-free measurements of a dataset/candidate -- no model was
fitted to produce them, so ordinary split CV is sufficient. Masks, weights,
Dynamic-K and the utility vectors ARE model outputs, so for those the outer
training rows must come from a source that never saw the outer test split:

    outer split s, training row from split t  ->  pair source S \\ {s, t}
    outer split s, TEST row                   ->  single-exclusion source for s

Both source families already exist (Stage 2A-3B0 / Stage 2A-1). Stage 2B-0 emits
the mapping only; it retrains nothing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

SPLITS: Tuple[int, ...] = tuple(range(2, 17))
FORBIDDEN_SPLIT = 1

FINAL_TRAINING_MODE = "final_training_single_exclusion"
NESTED_CV_MODE = "nested_cv_pair_exclusion"

SINGLE_EXCLUSION_REL = ("results_analysis/clustering_repository/stage2/"
                        "oof_utility_splits2_16")
PAIR_EXCLUSION_REL = ("results_analysis/clustopt_policy_predictor/"
                      "stage2a3b0_nested_crossfit")

# Feature blocks that require cross-fitted utility sources, vs those that do not.
UTILITY_DERIVED_BLOCKS: Tuple[str, ...] = (
    "FEATURE_SLATE_CONTEXT/oof utility vectors",
    "FEATURE_MASK/metric availability masks",
    "FEATURE_POLICY_CONTEXT/metric weights, actual K, Dynamic-K",
)
MODEL_FREE_BLOCKS: Tuple[str, ...] = (
    "FEATURE_SLATE_CONTEXT/250 meta-features",
    "FEATURE_CANDIDATE_CONFIG/algorithm + hyperparameters",
    "FEATURE_PARTITION_DESCRIPTOR/observed K, noise ratio, label counts",
    "FEATURE_CVI_RICH/60 offline CVI values",
)


def single_exclusion_source(split_id: int) -> Dict[str, Any]:
    if split_id == FORBIDDEN_SPLIT:
        raise ValueError("Split 1 is held out and must never be sourced")
    return {"mode": FINAL_TRAINING_MODE, "heldout_split": int(split_id),
            "root": SINGLE_EXCLUSION_REL,
            "fold_dir": f"holdout_split_{split_id:02d}",
            "training_splits": [s for s in SPLITS if s != split_id]}


def pair_exclusion_source(outer_split: int, row_split: int) -> Dict[str, Any]:
    if outer_split == row_split:
        raise ValueError("pair source requires two distinct splits")
    for s in (outer_split, row_split):
        if s not in SPLITS:
            raise ValueError(f"split {s} outside 2..16")
    a, b = sorted((int(outer_split), int(row_split)))
    return {"mode": NESTED_CV_MODE, "outer_split": int(outer_split),
            "row_split": int(row_split), "pair": [a, b],
            "pair_id": f"pair_{a:02d}_{b:02d}", "root": PAIR_EXCLUSION_REL,
            "training_splits": [s for s in SPLITS if s not in (a, b)]}


def nested_cv_plan() -> List[Dict[str, Any]]:
    """Every (outer split, row split) source binding a future loader needs.

    15 outer x 14 training row-splits = 210 training bindings, plus 15 test
    bindings. The 210 bindings resolve onto only C(15,2) = 105 distinct pair
    sources because the exclusion set is unordered.
    """
    rows: List[Dict[str, Any]] = []
    for s in SPLITS:
        for t in SPLITS:
            if t == s:
                continue
            src = pair_exclusion_source(s, t)
            rows.append({"outer_split": s, "row_split": t, "role": "TRAIN",
                         "source_mode": src["mode"], "pair_id": src["pair_id"],
                         "source_root": src["root"],
                         "excluded_splits": f"{src['pair'][0]},{src['pair'][1]}"})
        src = single_exclusion_source(s)
        rows.append({"outer_split": s, "row_split": s, "role": "TEST",
                     "source_mode": src["mode"],
                     "pair_id": src["fold_dir"], "source_root": src["root"],
                     "excluded_splits": str(s)})
    return rows


def assert_group_integrity(dataset_ids, split_ids) -> None:
    """A dataset must never appear under two split ids."""
    seen: Dict[Any, Any] = {}
    for d, s in zip(dataset_ids, split_ids):
        if d in seen and seen[d] != s:
            raise ValueError(f"dataset {d} maps to splits {seen[d]} and {s}")
        seen[d] = s
