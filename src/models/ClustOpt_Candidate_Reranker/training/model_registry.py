"""The single frozen Stage-2B1 model family. No search, no comparison.

``HistGradientBoostingRegressor`` is fixed in advance for reasons that are about
the data, not about measured performance: ~1.5 M candidate rows, a scalar
continuous regret target, and **native NaN handling** -- which the masked CVI
representation requires, since an unobserved metric must stay missing rather than
be imputed to a meaningful zero.

``early_stopping=False`` is deliberate (brief section 15). HGBR's automatic early
stopping carves a RANDOM row validation split, which would put candidates from the
same slate on both sides and add a second leakage-sensitive split for no benefit.
A fixed ``max_iter`` avoids that entirely.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import sklearn

MODEL_FAMILY = "HistGradientBoostingRegressor"
GLOBAL_SEED = 42

HGBR_CONFIG: Dict[str, Any] = {
    "loss": "squared_error",
    "learning_rate": 0.08,
    "max_iter": 150,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": GLOBAL_SEED,
}
API_NOTES = {
    "sklearn_version": sklearn.__version__,
    "api_deviation": "none -- every frozen key is a supported constructor "
                     "argument in this sklearn version",
    "early_stopping_rationale": "avoids HGBR's random-row validation split, "
                                "which would separate candidates of one slate",
    "missing_value_handling": "native; NaN denotes an unobserved or invalid CVI "
                              "and is never imputed",
}


def build_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**HGBR_CONFIG)


def config_hash() -> str:
    return hashlib.sha256(
        json.dumps({"family": MODEL_FAMILY, **HGBR_CONFIG},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def unit_hash(condition_id: str, outer_split: int, feature_schema_hash: str,
              eligibility_hash: str, mask_source: str) -> str:
    """Identity of one (condition x outer fold) execution unit."""
    payload = {
        "condition": condition_id, "outer_split": int(outer_split),
        "feature_schema": feature_schema_hash, "eligibility": eligibility_hash,
        "mask_source": mask_source, "model": config_hash(),
        "target": "T2_SLATE_REGRET", "seed": GLOBAL_SEED,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]
