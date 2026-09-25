"""Paired statistical comparisons (regressor vs baseline) on dataset-level ARI.

Optional: if SciPy is unavailable the module degrades gracefully -- callers get
an empty DataFrame and ``scipy_available() is False`` so the report can note it.
Pairing is by ``dataset_id`` using the dataset-level (best-view) ARI, matching
the win-rate analysis.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - presence depends on the environment
    from scipy import stats as _scipy_stats
    _SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _SCIPY = False

from .winrate_analysis import baselines_present, regressors_present


def scipy_available() -> bool:
    return _SCIPY


def _paired(dataset_level: pd.DataFrame, method: str, baseline: str):
    m = dataset_level[dataset_level["method_name"] == method][["dataset_id", "best_ari"]]
    b = dataset_level[dataset_level["method_name"] == baseline][["dataset_id", "best_ari"]]
    merged = m.merge(b, on="dataset_id", suffixes=("_m", "_b"))
    a = pd.to_numeric(merged["best_ari_m"], errors="coerce").to_numpy()
    c = pd.to_numeric(merged["best_ari_b"], errors="coerce").to_numpy()
    mask = np.isfinite(a) & np.isfinite(c)
    return a[mask], c[mask]


def _test_row(dataset_level: pd.DataFrame, method: str, baseline: str) -> dict:
    a, b = _paired(dataset_level, method, baseline)
    n = int(a.size)
    diff = a - b
    row = {
        "method": method, "baseline": baseline, "n_paired": n,
        "mean_diff": float(np.mean(diff)) if n else float("nan"),
        "median_diff": float(np.median(diff)) if n else float("nan"),
        "wilcoxon_p": float("nan"), "ttest_p": float("nan"), "cohens_dz": float("nan"),
    }
    if n == 0:
        return row
    sd = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    row["cohens_dz"] = float(np.mean(diff) / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    if _SCIPY and n >= 2:
        try:
            if np.any(diff != 0):
                row["wilcoxon_p"] = float(_scipy_stats.wilcoxon(a, b).pvalue)
        except Exception:
            pass
        try:
            row["ttest_p"] = float(_scipy_stats.ttest_rel(a, b).pvalue)
        except Exception:
            pass
    return row


def paired_significance(
    dataset_level: pd.DataFrame,
    *,
    group_cols: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    baselines: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Paired tests for each (group x method x baseline).

    ``methods`` / ``baselines`` default to dynamic detection from the data
    (regressors vs whichever non-regressor methods are present) -- matching the
    win-rate analysis -- so it works for both the phase-A (fixed-CVI baselines)
    and phase-B (external baselines) comparisons regardless of the displayed
    method names.
    """
    if dataset_level.empty:
        return pd.DataFrame()
    if methods is None:
        methods = regressors_present(dataset_level)
    if baselines is None:
        baselines = baselines_present(dataset_level)
    group_cols = [c for c in (group_cols or []) if c in dataset_level.columns]
    if not group_cols:
        present = set(dataset_level["method_name"].unique())
        rows = [
            _test_row(dataset_level, m, b)
            for m in methods if m in present
            for b in baselines if b in present and b != m
        ]
        return pd.DataFrame(rows)

    out = []
    for key, g in dataset_level.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        present = set(g["method_name"].unique())
        for m in methods:
            if m not in present:
                continue
            for b in baselines:
                if b not in present or b == m:
                    continue
                row = {c: v for c, v in zip(group_cols, key_tuple)}
                row.update(_test_row(g, m, b))
                out.append(row)
    return pd.DataFrame(out)
