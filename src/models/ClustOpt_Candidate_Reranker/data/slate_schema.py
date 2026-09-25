"""Table A -- slate context, one row per (dataset, view). 48,039 rows.

Composed from the Stage-2A-3A view-level tables rather than rebuilt. Those files
are already row-aligned, audited, and bound to the same single-exclusion OOF
utilities the Stage-2A-2 replay used; regenerating them here would risk a silent
divergence between two supposedly identical representations for no benefit.
They are opened read-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E501
    _ext,
)

STAGE2A3A_REL = ("results_analysis/clustopt_policy_predictor/"
                 "stage2a3a_view_level_training_data")

IDENT_FILE = "view_level_identifiers.csv.gz"
META_FILE = "view_level_meta_features.csv.gz"
UTIL_FILE = "view_level_oof_utility_features.csv.gz"
DYNK_FILE = "view_level_dynamic_k_features.csv.gz"

N_SLATES = 48039
N_META = 250
N_UTILITY = 120
MLP_PREFIX = "oof_mlp_utility__"
KNN_PREFIX = "oof_knn_utility__"


def load_slate_context(repo: Path) -> Dict[str, pd.DataFrame]:
    """Identifiers / meta / utilities / dynamic-K, row-aligned."""
    root = Path(repo) / STAGE2A3A_REL
    out = {
        "ident": pd.read_csv(_ext(root / IDENT_FILE)),
        "meta": pd.read_csv(_ext(root / META_FILE)),
        "util": pd.read_csv(_ext(root / UTIL_FILE)),
        "dynk": pd.read_csv(_ext(root / DYNK_FILE)),
    }
    n = len(out["ident"])
    if n != N_SLATES:
        raise ValueError(f"expected {N_SLATES} slates, got {n}")
    for k, d in out.items():
        if len(d) != n:
            raise ValueError(f"block {k} has {len(d)} rows, expected {n}")
    if out["meta"].shape[1] != N_META:
        raise ValueError(f"expected {N_META} meta-features, "
                         f"got {out['meta'].shape[1]}")
    if out["util"].shape[1] != N_UTILITY:
        raise ValueError(f"expected {N_UTILITY} utility features, "
                         f"got {out['util'].shape[1]}")
    return out


def utility_matrix(util: pd.DataFrame, source: str,
                   metric_names: List[str]):
    """The 60-dim OOF utility vector block for one source, in canonical order."""
    prefix = MLP_PREFIX if source == "mlp" else KNN_PREFIX
    cols = [f"{prefix}{m}" for m in metric_names]
    missing = [c for c in cols if c not in util.columns]
    if missing:
        raise ValueError(f"{len(missing)} {source} utility columns missing, "
                         f"e.g. {missing[:3]}")
    return util[cols].to_numpy(float)


def slate_key(ident: pd.DataFrame) -> pd.Series:
    return ident["dataset_id"].astype(str) + "|" + ident["view_id"].astype(str)


def slate_columns() -> Tuple[str, ...]:
    return ("dataset_id", "split_id", "family", "subfamily", "view_id",
            "oof_heldout_split")
