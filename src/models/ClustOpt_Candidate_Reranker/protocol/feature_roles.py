"""Column/block roles for the Candidate Reranker, deny-by-default.

Leakage control here is SEMANTIC, never substring-based. Stage 1F showed why:
``"ari" in name`` flags legitimate geometric meta-features such as
``G_linearity_pca_ratio`` and ``L_landmark_score_variance``. Classification is by
exact name, explicit prefix, or an explicit whole-token deny list.

Anything that cannot be positively classified is ``FORBIDDEN``: an unrecognised
column is never silently admitted as a feature.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Set, Tuple

ROLES: Tuple[str, ...] = (
    "FEATURE_SLATE_CONTEXT",
    "FEATURE_CANDIDATE_CONFIG",
    "FEATURE_PARTITION_DESCRIPTOR",
    "FEATURE_CVI_RICH",
    "FEATURE_CVI_MASKED",
    "FEATURE_MASK",
    "FEATURE_POLICY_CONTEXT",
    "FEATURE_OPTIONAL",
    "TARGET_ONLY",
    "METADATA_ONLY",
    "FORBIDDEN",
)

# ---------------------------------------------------------------- deny lists
# Whole tokens that are unambiguous markers of a target or a ground-truth leak.
#
# Deliberately NARROW. Stage 1F was burned by substring matching ("ari" inside
# G_linearity_pca_ratio); token matching fixes that but has its own trap, and it
# fired here: ``V_texture_glcm_homogeneity_approx`` is a grey-level co-occurrence
# texture statistic of a rasterised scatter image (block_v_visual_raster.py), with
# nothing to do with the clustering homogeneity score. Bare words that are common
# in geometry/imaging vocabulary -- homogeneity, completeness, purity, accuracy,
# rank, label -- are therefore NOT tokens here. External clustering metrics are
# caught by EXTERNAL_METRIC_RE below, which requires the metric's full identifier.
FORBIDDEN_TOKENS: Set[str] = {
    "ari", "adjustedrand", "nmi", "ami",
    "ytrue", "groundtruth", "truelabels", "truek",
    "regret", "oracle", "bestcandidate",
}
# Full external-metric identifiers, matched on the whole lowered column name.
EXTERNAL_METRIC_RE = re.compile(
    r"(adjusted_rand|rand_index|homogeneity_score|completeness_score|"
    r"v_measure|fowlkes_mallows|mutual_info|purity_score|cluster_accuracy|"
    r"true_k|true_labels|y_true|ground_truth)")
# Exact names that are targets rather than leaks: legitimate to store, never to
# feed a model as input.
TARGET_NAMES: Set[str] = {
    "ari", "candidate_ari", "slate_regret", "rank_group", "rank_dense",
    "slate_oracle_ari", "is_slate_best",
}
METADATA_NAMES: Set[str] = {
    "dataset_id", "split_id", "family", "subfamily", "family_id",
    "subfamily_id", "dataset_dir", "oof_heldout_split", "candidate_key",
    "candidate_index", "config_str", "exception", "row_id", "record_id",
    "view_mode", "specific_subfamily", "difficulty",
}
# ``view_id`` is metadata as a string but a legitimate FEATURE once encoded.
VIEW_NAMES: Set[str] = {"view_id"}

PREFIX_ROLES: Tuple[Tuple[str, str], ...] = (
    ("oof_mlp_utility__", "FEATURE_SLATE_CONTEXT"),
    ("oof_knn_utility__", "FEATURE_SLATE_CONTEXT"),
    ("oof_mlp_dynamic_k", "FEATURE_SLATE_CONTEXT"),
    ("oof_knn_dynamic_k", "FEATURE_SLATE_CONTEXT"),
    ("oof_mlp_k_rel_raw", "FEATURE_SLATE_CONTEXT"),
    ("oof_knn_k_rel_raw", "FEATURE_SLATE_CONTEXT"),
    ("cvi__", "FEATURE_CVI_RICH"),
    ("cvi_valid__", "FEATURE_MASK"),
    ("mask__", "FEATURE_MASK"),
    ("w_raw_", "FEATURE_POLICY_CONTEXT"),
    ("w_soft_", "FEATURE_POLICY_CONTEXT"),
    ("sel_metric_", "FEATURE_POLICY_CONTEXT"),
    ("cfg__", "FEATURE_CANDIDATE_CONFIG"),
    ("cfgmask__", "FEATURE_CANDIDATE_CONFIG"),
    ("part__", "FEATURE_PARTITION_DESCRIPTOR"),
    ("slate__", "FEATURE_OPTIONAL"),
)
# Meta-feature blocks carry a single-letter prefix followed by ``_``.
META_PREFIX_RE = re.compile(r"^[A-Z]_")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(name: str) -> Set[str]:
    return set(_TOKEN_RE.findall(name.lower()))


def classify_column(name: str, *, meta_allowlist: Optional[Set[str]] = None
                    ) -> str:
    """Role of one column. Deny-by-default: unknown names are FORBIDDEN.

    ``meta_allowlist`` is the authoritative frozen meta-feature schema (the 250
    A/B/G/L/R/V columns from ``feature_columns.json``, audited label-free in
    Stage 2A-3A). Membership in it is POSITIVE evidence and outranks the name
    screen, because the screen keeps producing false positives on legitimate
    geometry/imaging vocabulary -- ``V_texture_glcm_homogeneity_approx`` is a
    grey-level co-occurrence texture statistic, and
    ``R_mutual_information_xy_hist_16`` is mutual information between the x and y
    COORDINATE axes, not between predicted and true labels. Neither ever sees a
    label: ``feature_extractor.extract_features_for_view`` receives the view
    matrix only. Name screening remains in force for every column outside the
    allowlist.
    """
    if meta_allowlist and name in meta_allowlist:
        return "FEATURE_SLATE_CONTEXT"
    low = name.lower()
    if EXTERNAL_METRIC_RE.search(low):
        return "FORBIDDEN"
    if low in TARGET_NAMES:
        return "TARGET_ONLY"
    if low in METADATA_NAMES:
        return "METADATA_ONLY"
    if low in VIEW_NAMES:
        return "FEATURE_SLATE_CONTEXT"
    for prefix, role in PREFIX_ROLES:
        if name.startswith(prefix):
            return role
    if META_PREFIX_RE.match(name):
        # Meta-features are label-free by construction (Stage 2A-3A audit), but
        # still screened: a meta-feature whose name carries a forbidden token
        # would be a schema error, not something to admit.
        return "FORBIDDEN" if tokens(name) & FORBIDDEN_TOKENS else "FEATURE_SLATE_CONTEXT"
    if low in ("algorithm",):
        return "FEATURE_CANDIDATE_CONFIG"
    if low in ("valid", "n_clusters_wo_noise", "noise_ratio", "n_labels_unique"):
        return "FEATURE_PARTITION_DESCRIPTOR"
    if low in ("policy_id", "source", "k_mode", "actual_k", "k_rel_raw",
               "weighting", "weighting_mode", "objective_j"):
        return "FEATURE_POLICY_CONTEXT"
    if tokens(name) & FORBIDDEN_TOKENS:
        return "FORBIDDEN"
    return "FORBIDDEN"


def assert_no_forbidden_features(columns, *, allow: Set[str] = frozenset(),
                                 meta_allowlist: Optional[Set[str]] = None
                                 ) -> Dict[str, str]:
    """Classify columns; raise if any column would enter a model illegitimately."""
    roles = {c: classify_column(c, meta_allowlist=meta_allowlist)
             for c in columns if c not in allow}
    bad = {c: r for c, r in roles.items() if r == "FORBIDDEN"}
    if bad:
        raise ValueError(f"{len(bad)} unclassifiable/forbidden columns, "
                         f"e.g. {list(bad)[:5]}")
    return roles


def feature_roles() -> Tuple[str, ...]:
    return tuple(r for r in ROLES if r.startswith("FEATURE_"))
