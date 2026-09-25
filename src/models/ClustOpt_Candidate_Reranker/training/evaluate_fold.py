"""Selection, native-J baselines and fold metrics.

Selection discipline is structural: a candidate is chosen from predictions (or
from J) alone, its identity is frozen, and only then is its ARI read. ARI never
participates in any argmin/argmax and never breaks a tie.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TIE_TOL = 1e-12
VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")


# --------------------------------------------------------------- selection
def select_per_slate(score: np.ndarray, eligible: np.ndarray,
                     slate_starts: np.ndarray, slate_sizes: np.ndarray,
                     *, minimise: bool) -> np.ndarray:
    """Index (within the row block) of the chosen candidate for each slate.

    Ties are broken by the lowest candidate index, mirroring production's
    first-in-stored-order rule. Never consults ARI.
    """
    out = np.empty(len(slate_starts), dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    for i, (a, n) in enumerate(zip(slate_starts, slate_sizes)):
        sl = slice(a, a + n)
        elig = np.flatnonzero(eligible[sl])
        if elig.size == 0:
            out[i] = a
            continue
        v = s[sl][elig]
        best = v.min() if minimise else v.max()
        tied = elig[np.abs(v - best) <= TIE_TOL]
        out[i] = a + int(tied[0])
    return out


def native_j_scores(cvi: np.ndarray, metric_idx: np.ndarray,
                    weights: np.ndarray) -> np.ndarray:
    """J = sum_m w_m * <metric>_norm, vectorised per row.

    ``metric_idx`` and ``weights`` are per-ROW (each slate's regime resolves its
    own metric set), padded with -1 / 0.0.
    """
    n, kmax = metric_idx.shape
    out = np.zeros(n, dtype=np.float64)
    rows = np.arange(n)
    for j in range(kmax):
        idx = metric_idx[:, j]
        ok = idx >= 0
        if not ok.any():
            continue
        vals = np.zeros(n, dtype=np.float64)
        vals[ok] = cvi[rows[ok], idx[ok]]
        out += weights[:, j] * vals
    return out


def verify_production_equivalence(cvi_block: np.ndarray,
                                  frame_cols: Sequence[str],
                                  metric_names: Sequence[str],
                                  sel_names: Sequence[str],
                                  sel_weights: Sequence[float]) -> Dict[str, Any]:
    """Compare the vectorised J against production ``score_candidates`` exactly."""
    import pandas as _pd
    from models.Clustering_Repository_Builder.experiments.experiment_execution.offline_candidate_execution import (  # noqa: E501
        score_candidates,
    )
    frame = _pd.DataFrame(cvi_block, columns=[f"{m}_norm" for m in metric_names])
    wmap = {n: float(w) for n, w in zip(sel_names, sel_weights)}
    ref = np.asarray(score_candidates(frame, wmap), dtype=np.float64)
    idx = np.array([[metric_names.index(n) for n in sel_names]] * len(frame))
    w = np.array([list(sel_weights)] * len(frame), dtype=np.float64)
    got = native_j_scores(cvi_block, idx, w)
    dev = float(np.nanmax(np.abs(ref - got)))
    return {"n_rows": int(len(frame)), "max_abs_deviation": dev,
            "equivalent": bool(dev <= 1e-12)}


# ------------------------------------------------------------------ metrics
def _bestview(dataset_ids: np.ndarray, vals: np.ndarray) -> pd.Series:
    return pd.DataFrame({"d": dataset_ids, "a": vals}).groupby("d")["a"].max()


def slate_metrics(sel_ari: np.ndarray, oracle: np.ndarray,
                  view_ids: np.ndarray, dataset_ids: np.ndarray
                  ) -> Dict[str, Any]:
    """View-level and dataset BestView metrics for one arm on one fold."""
    reg = oracle - sel_ari
    m: Dict[str, Any] = {
        "n_slates": int(len(sel_ari)),
        "mean_selected_ari": float(np.mean(sel_ari)),
        "mean_candidate_oracle_ari": float(np.mean(oracle)),
        "mean_oracle_regret": float(np.mean(reg)),
        "median_oracle_regret": float(np.median(reg)),
        "frac_exact_oracle": float(np.mean(reg <= TIE_TOL)),
    }
    for thr, lab in ((0.001, "001"), (0.005, "005"), (0.01, "01")):
        m[f"frac_within_{lab}"] = float(np.mean(reg <= thr))
    for v in VIEW_IDS:
        k = view_ids == v
        if k.any():
            m[f"{v}__mean_selected_ari"] = float(np.mean(sel_ari[k]))
            m[f"{v}__mean_oracle"] = float(np.mean(oracle[k]))
            m[f"{v}__mean_regret"] = float(np.mean(reg[k]))
            m[f"{v}__frac_exact"] = float(np.mean(reg[k] <= TIE_TOL))
    bv_sel = _bestview(dataset_ids, sel_ari)
    bv_orc = _bestview(dataset_ids, oracle)
    m["n_datasets"] = int(len(bv_sel))
    m["mean_bestview_selected_ari"] = float(bv_sel.mean())
    m["mean_bestview_candidate_oracle"] = float(bv_orc.mean())
    m["mean_bestview_regret"] = float((bv_orc - bv_sel).mean())
    return m


def bestview_frame(dataset_ids: np.ndarray, sel_ari: np.ndarray,
                   oracle: np.ndarray) -> pd.DataFrame:
    f = pd.DataFrame({"dataset_id": dataset_ids, "sel": sel_ari, "orc": oracle})
    g = f.groupby("dataset_id").agg(bestview_selected=("sel", "max"),
                                    bestview_oracle=("orc", "max"))
    return g.reset_index()


def recovery(selector: float, baseline: float, oracle: float,
             tol: float = 1e-9) -> float:
    """(S - B) / (O - B); NaN when the baseline already sits at the oracle."""
    den = oracle - baseline
    return float((selector - baseline) / den) if den > tol else float("nan")
