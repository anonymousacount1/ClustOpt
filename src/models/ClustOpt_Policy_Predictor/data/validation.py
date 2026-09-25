"""Leakage gates and integrity checks for the view-conditional training data.

Every gate uses explicit semantic column sets. Substring matching is deliberately
avoided: Stage 2A-2 recorded three false positives from an ``ari`` substring scan
(``G_linearity_pca_ratio``, ``G_planarity_or_2d_spread``,
``L_landmark_score_variance``), all of which are label-free geometry features.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from . import feature_schema as FS
from . import target_schema as TS

TOL = 1e-9


class Check:
    """Accumulates named PASS/FAIL checks."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.ok = True

    def add(self, name: str, cond: bool, detail: str = "") -> bool:
        cond = bool(cond)
        self.ok &= cond
        self.rows.append({"check": name, "status": "PASS" if cond else "FAIL",
                          "detail": str(detail)})
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""), flush=True)
        return cond

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def leakage_audit(meta_cols: Sequence[str], oof_cols: Sequence[str],
                  optional_cols: Sequence[str],
                  target_cols: Sequence[str],
                  metadata_cols: Sequence[str]) -> pd.DataFrame:
    """Per-column role assignment for every column in the package."""
    rows = []
    for group, cols in (("meta", meta_cols), ("oof_utility", oof_cols),
                        ("optional", optional_cols), ("target", target_cols),
                        ("metadata", metadata_cols)):
        for c in cols:
            role = FS.classify_column(c, meta=meta_cols, oof=oof_cols)
            expected = {"meta": FS.ROLE_META, "oof_utility": FS.ROLE_OOF_UTILITY,
                        "optional": FS.ROLE_OPTIONAL, "target": FS.ROLE_TARGET,
                        "metadata": FS.ROLE_METADATA}[group]
            rows.append({"column": c, "declared_block": group, "role": role,
                         "expected_role": expected,
                         "agrees": role == expected,
                         "predictive_allowed": role in (
                             FS.ROLE_META, FS.ROLE_OOF_UTILITY, FS.ROLE_OPTIONAL)})
    return pd.DataFrame(rows)


def assert_no_target_in_features(feature_cols: Sequence[str],
                                 target_cols: Sequence[str],
                                 metadata_cols: Sequence[str]) -> List[str]:
    """Semantic intersection test -- returns offending columns (empty == clean)."""
    forbidden = set(target_cols) | set(metadata_cols) | set(FS.FORBIDDEN_COLUMNS)
    bad = [c for c in feature_cols
           if c in forbidden
           or c.startswith(TS.ARI_PREFIX) or c.startswith(TS.REGRET_PREFIX)
           or c.lower() in FS.FORBIDDEN_COLUMNS]
    return bad


def regret_contract(ari: np.ndarray, regret: np.ndarray) -> Dict[str, Any]:
    """regret = rowmax(ari) - ari, non-negative, with >=1 exact zero per row."""
    vbs = ari.max(axis=1, keepdims=True)
    expected = vbs - ari
    return {
        "max_abs_deviation": float(np.abs(regret - expected).max()),
        "min_regret": float(regret.min()),
        "n_negative_rows": int((regret < -TOL).any(axis=1).sum()),
        "n_rows_without_zero_regret": int((~np.isclose(regret, 0.0, atol=TOL)
                                           ).all(axis=1).sum()),
    }


def oof_split_binding(df: pd.DataFrame) -> Dict[str, Any]:
    """Every row's OOF utility source must be the fold that held out its split."""
    bad = int((df["oof_heldout_split"] != df["split_id"]).sum())
    return {"n_rows": int(len(df)), "n_mismatched": bad,
            "split1_present": bool((df["split_id"] == 1).any())}
