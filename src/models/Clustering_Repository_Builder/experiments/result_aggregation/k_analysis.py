"""K-selection analysis: accuracy summary and per-method confusion matrices."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def k_accuracy_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-method K accuracy / error / over-under rates (successful runs)."""
    if raw.empty:
        return pd.DataFrame()
    success = raw[raw["status"] == "success"].copy()
    rows = []
    for method, g in success.groupby("method_name"):
        signed = pd.to_numeric(g["k_signed_error"], errors="coerce").dropna()
        absr = pd.to_numeric(g["k_abs_error"], errors="coerce").dropna()
        correct = g["k_correct"].dropna().astype(bool)
        n = int(signed.size)
        rows.append({
            "method_name": method,
            "n": n,
            "k_accuracy": float(correct.mean()) if len(correct) else float("nan"),
            "mean_k_abs_error": float(absr.mean()) if len(absr) else float("nan"),
            "median_k_abs_error": float(absr.median()) if len(absr) else float("nan"),
            "mean_k_signed_error": float(signed.mean()) if n else float("nan"),
            "overcluster_rate": float((signed > 0).sum() / n) if n else float("nan"),
            "undercluster_rate": float((signed < 0).sum() / n) if n else float("nan"),
            "exact_k_rate": float((signed == 0).sum() / n) if n else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("k_accuracy", ascending=False).reset_index(drop=True)


def k_confusion_matrix(raw: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """Confusion matrix (rows=true_k, cols=selected_k) for one method."""
    success = raw[(raw["status"] == "success") & (raw["method_name"] == method_name)]
    tk = pd.to_numeric(success["true_k"], errors="coerce")
    sk = pd.to_numeric(success["selected_k"], errors="coerce")
    mask = tk.notna() & sk.notna()
    if int(mask.sum()) == 0:
        return pd.DataFrame()
    cm = pd.crosstab(tk[mask].astype(int), sk[mask].astype(int))
    cm.index.name = "true_k"
    cm.columns.name = "selected_k"
    return cm


def all_k_confusion_matrices(raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if raw.empty:
        return {}
    out = {}
    for method in sorted(raw["method_name"].dropna().unique()):
        cm = k_confusion_matrix(raw, method)
        if not cm.empty:
            out[method] = cm
    return out
