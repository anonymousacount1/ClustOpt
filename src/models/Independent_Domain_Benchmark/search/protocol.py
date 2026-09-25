"""Frozen external search protocol and deterministic seed derivation."""
from __future__ import annotations

import hashlib

BUDGET_EVALUATIONS = 50
K_MIN, K_MAX = 2, 5
SEARCH_SPACE_SOURCE = "clustopt_phaseE2_map"
BASE_SEED = 1234

#: Arms that share a search trace share a seed identity, so their visited slates
#: are identical by construction and the pair is exactly matched.
TRACE_IDENTITY = {
    "IDB_ClustOpt_C0_MLP_TOP5_RAW_noR": "clustopt_mlp_top5",
    "IDB_ClustOpt_C3_MLP_TOP5_RAW_R": "clustopt_mlp_top5",
    "IDB_ClustOpt_C1_KNN_TOP10_RAW_noR": "clustopt_knn_top10",
    "IDB_ClustOpt_C4_KNN_TOP10_RAW_R": "clustopt_knn_top10",
    "IDB_ClustOpt_C2_PPv1_noR": "clustopt_ppv1",
    "IDB_ClustOpt_C5_PPv2_R": "clustopt_ppv2",
    #: Stage-4 pre-run protocol amendment (2026-09-09, project_lead-authorised).
    #: A0-A3 decompose the CVI inventory, so they draw a COMMON candidate
    #: slate: candidate generation was proven pre-sampled and independent of
    #: the inventory, so four independent slates would add Monte-Carlo noise
    #: to the A1-A0 / A2-A0 / A3-A1 / A3-A2 / A3-A0 contrasts for no gain.
    #: The four scientific METHOD_IDs are unchanged. See
    #: docs/internal_reports/stage4/STAGE4_PROTOCOL_AMENDMENT_AUTOCLUST_COMMON_SEED.md
    "IDB_AutoClust_A0_original_cvis": "autoclust_shared_candidate_slate",
    "IDB_AutoClust_A1_original_plus_established": "autoclust_shared_candidate_slate",
    "IDB_AutoClust_A2_original_plus_new46": "autoclust_shared_candidate_slate",
    "IDB_AutoClust_A3_extended_cvis": "autoclust_shared_candidate_slate",
    #: Stage-4 pre-run protocol amendment (2026-09-10, project_lead-authorised).
    #: M0-M3 decompose the CVI inventory exactly as A0-A3 do. Stage 4B-3B1
    #: proved the ML2DAC physical candidate chain -- meta-features, nearest
    #: neighbour, warmstarts, ws_algorithms, constrained sampling -- is
    #: arm-independent, and that the predicted CVI is consumed only by
    #: scoring. Four independent slates would therefore add Monte-Carlo noise
    #: to the M1-M0 / M2-M0 / M3-M1 / M3-M2 / M3-M0 contrasts for no gain.
    #: The four scientific METHOD_IDs are unchanged. See
    #: docs/internal_reports/stage4/STAGE4_PROTOCOL_AMENDMENT_ML2DAC_COMMON_SEED.md
    "IDB_ML2DAC_M0_original_cvis": "ml2dac_shared_candidate_slate",
    "IDB_ML2DAC_M1_original_plus_established": "ml2dac_shared_candidate_slate",
    "IDB_ML2DAC_M2_original_plus_new46": "ml2dac_shared_candidate_slate",
    "IDB_ML2DAC_M3_extended_cvis": "ml2dac_shared_candidate_slate",
}


def seed_identity(method_id: str) -> str:
    """The SEARCH identity, deliberately coarser than the method id."""
    return TRACE_IDENTITY.get(method_id, method_id)


def derive_seed(dataset_id: str, view_id: str, method_id: str,
                base_seed: int = BASE_SEED) -> int:
    """Deterministic SHA256 seed. Never derived from any outcome."""
    key = "%d|%s|%s|%s" % (base_seed, dataset_id, view_id, seed_identity(method_id))
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)
