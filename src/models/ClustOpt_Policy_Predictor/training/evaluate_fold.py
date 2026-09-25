"""Fold-level metrics: BestView selector ARI, baselines, oracle, recoveries.

Baseline identities are chosen from OUTER-TRAINING targets only and then applied
to outer test. Nothing here consults outer-test values to make a choice.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from . import feature_sets as FSET

TOL = 1e-12


def bestview_from_rows(df: pd.DataFrame, ari_col: str) -> pd.Series:
    """Dataset-level BestView = max over the dataset's three view rows."""
    return df.groupby("dataset_id")[ari_col].max()


def fold_baselines(fold: Dict[str, pd.DataFrame], pids) -> Dict[str, Any]:
    """B1 global and B2 per-view, both selected on TRAINING targets only."""
    tri, tei = fold["train_ident"], fold["test_ident"]
    A_tr = fold["train_ari"].to_numpy(np.float64)
    A_te = fold["test_ari"].to_numpy(np.float64)

    # ---- B1: one global policy maximising TRAIN dataset-level BestView -------
    tr_ds = tri["dataset_id"].to_numpy()
    best_p, best_score = 0, -np.inf
    for p in range(A_tr.shape[1]):
        s = pd.DataFrame({"d": tr_ds, "a": A_tr[:, p]}).groupby("d")["a"].max().mean()
        if s > best_score:
            best_p, best_score = p, s
    te_df = pd.DataFrame({"dataset_id": tei["dataset_id"].to_numpy(),
                          "a": A_te[:, best_p]})
    b_global = float(te_df.groupby("dataset_id")["a"].max().mean())

    # ---- B2: best policy per view, each chosen on its own TRAIN view ---------
    per_view, sel_te = {}, np.empty(len(tei))
    for v in FSET.VIEW_IDS:
        mtr = (tri["view_id"] == v).to_numpy()
        mte = (tei["view_id"] == v).to_numpy()
        pv = int(np.argmax(A_tr[mtr].mean(axis=0)))
        per_view[v] = pids[pv]
        sel_te[mte] = A_te[mte][:, pv]
    bv_df = pd.DataFrame({"dataset_id": tei["dataset_id"].to_numpy(), "a": sel_te})
    b_view = float(bv_df.groupby("dataset_id")["a"].max().mean())

    # ---- oracle ceiling (diagnostic only) ------------------------------------
    o_df = pd.DataFrame({"dataset_id": tei["dataset_id"].to_numpy(),
                         "a": A_te.max(axis=1)})
    oracle = float(o_df.groupby("dataset_id")["a"].max().mean())

    return {"global_best_single_policy": pids[best_p],
            "global_best_single_index": int(best_p),
            "b_global": b_global,
            "per_view_policies": per_view, "b_view": b_view,
            "oracle_bestview": oracle,
            "b_global_by_dataset": te_df.groupby("dataset_id")["a"].max(),
            "b_view_by_dataset": bv_df.groupby("dataset_id")["a"].max(),
            "oracle_by_dataset": o_df.groupby("dataset_id")["a"].max()}


def fold_metrics(pred: pd.DataFrame, base: Dict[str, Any]) -> Dict[str, Any]:
    """Primary + secondary metrics for one configuration on one fold."""
    sel_bv = bestview_from_rows(pred, "selected_ari")
    S = float(sel_bv.mean())
    bg, bv, V = base["b_global"], base["b_view"], base["oracle_bestview"]
    rg = (S - bg) / (V - bg) if (V - bg) > TOL else np.nan
    rv = (S - bv) / (V - bv) if (V - bv) > TOL else np.nan
    m: Dict[str, Any] = {
        "bestview_selector_ari": S, "b_global": bg, "b_view": bv,
        "oracle_bestview": V, "recovery_global": rg, "recovery_view": rv,
        "beats_b_global": bool(S > bg), "beats_b_view": bool(S > bv),
        "n_test_datasets": int(len(sel_bv)),
    }
    for v in FSET.VIEW_IDS:                       # view-level diagnostics
        q = pred[pred["view_id"] == v]
        m[f"{v}__mean_selected_ari"] = float(q["selected_ari"].mean())
        m[f"{v}__mean_regret"] = float(q["view_regret"].mean())
        for thr, lab in ((0.001, "001"), (0.005, "005"), (0.01, "01")):
            m[f"{v}__within_0_{lab}"] = float((q["view_regret"] < thr).mean())
    return m


def prediction_quality(pred: pd.DataFrame, ari_te: np.ndarray,
                       target: str) -> Dict[str, Any]:
    """Secondary regression diagnostics -- never used for model selection."""
    from . import target_transforms as TT
    P = pred[[f"pred_{j:02d}" for j in range(20)]].to_numpy(np.float64)
    Y = TT.transform(ari_te, target).astype(np.float64)
    err = P - Y
    out = {"pred_mae": float(np.abs(err).mean()),
           "pred_rmse": float(np.sqrt((err ** 2).mean()))}
    try:
        from scipy import stats as sst
        rs = [sst.spearmanr(P[i], Y[i]).statistic for i in range(0, len(P), 25)
              if np.ptp(Y[i]) > 0 and np.ptp(P[i]) > 0]
        out["pred_spearman"] = float(np.nanmean(rs)) if rs else float("nan")
    except Exception:                                              # noqa: BLE001
        out["pred_spearman"] = float("nan")
    win = np.isclose(ari_te, ari_te.max(axis=1, keepdims=True), atol=TOL)
    sel = pred["selected_policy_index"].to_numpy()
    out["selected_in_winner_set"] = float(
        win[np.arange(len(sel)), sel].mean())
    return out
