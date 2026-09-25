"""Stage 3C data layer: dataset-level utility, prediction and selection matrices.

ANALYSIS ONLY. Nothing is fitted, searched or rebuilt.

The single most important design decision is the **unit of analysis**. Canonical
utility is defined per (dataset, view, metric), but the three views of a dataset
are not independent observations. Every inferential quantity in Stage 3C is
therefore computed on **dataset-level** matrices, obtained by averaging over the
dataset's valid views (utility, predicted utility) or by taking the fraction of
valid views in which a metric is selected (selection rate). View-level matrices
are retained separately and used only descriptively, in the view-specificity
analysis.

Vocabulary note carried over from Stage 3B: the ClustOpt utility vocabulary is
Full-60. Three of the seven ORIGINAL baseline CVIs (``dunn_index``, ``cop``,
``coggins_jain_index``) are ML2DAC-side indices with no ClustOpt utility; they
are emitted in every output table with an explicit NA reason rather than being
dropped, so all 61 live metrics are accounted for.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import common as C

TRUE_P = "true_"
PRED_P = "pred_"
#: frozen ClustOpt deployment semantics, verified against the Stage-1 arm
#: `stage1_full60_mlp_top5_raw_fixed50` (resolver_top_k = 5, predictor = the
#: metric-utility MLP). Stage 3C does not invent a selection rule.
TOP_K = C.CLUSTOPT_TOP_K


@dataclass
class SpecData:
    metrics: List[str]                 # the 60 utility-space metrics
    all_live: List[str]                # all 61 live metrics (atlas order)
    no_utility: List[str]              # live metrics with no ClustOpt utility
    ds: pd.DataFrame                   # dataset_id, family, subfamily (16,013)
    TRUE: np.ndarray                   # (n_datasets, 60) mean true utility
    PRED: np.ndarray                   # (n_datasets, 60) mean predicted utility
    SEL: np.ndarray                    # (n_datasets, 60) selection rate in [0,1]
    view_true: Dict[str, np.ndarray]   # per view, (n_datasets, 60)
    view_sel: Dict[str, np.ndarray]
    view_index: Dict[str, np.ndarray]  # per view, dataset row indices present
    atlas: pd.DataFrame
    fam_of: np.ndarray                 # family per dataset row
    sub_of: np.ndarray                 # subfamily per dataset row
    parent: Dict[str, str]             # subfamily -> family
    provenance: Dict[str, object]


def _select_topk(P: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Boolean mask of the Top-k metrics by predicted utility, per row."""
    order = np.argsort(-np.nan_to_num(P, nan=-np.inf), axis=1)
    sel = np.zeros_like(P, dtype=bool)
    sel[np.arange(P.shape[0])[:, None], order[:, :k]] = True
    return sel


def load(repo: Path, log, knn: bool = False) -> SpecData:
    oof = C.load_oof(repo, knn=knn)
    tcols = C.metric_cols(oof, TRUE_P)
    metrics = C.strip(tcols, TRUE_P)
    pcols = [PRED_P + m for m in metrics]
    missing = [c for c in pcols if c not in oof.columns]
    if missing:
        raise AssertionError("missing predicted columns: %s" % missing[:5])

    atlas = pd.read_csv(C.ext(repo / C.OUT_REL / "canonical_metric_atlas.csv"))
    live = atlas[atlas.is_live]
    all_live = list(live["metric_id"])
    no_utility = [m for m in all_live if m not in set(metrics)]

    oof = oof.sort_values(["dataset_id", "view_mode"]).reset_index(drop=True)
    ds = (oof[["dataset_id", "family_id", "subfamily_id"]]
          .drop_duplicates("dataset_id").reset_index(drop=True)
          .rename(columns={"family_id": "family", "subfamily_id": "subfamily"}))
    pos = {d: i for i, d in enumerate(ds["dataset_id"])}
    n, p = len(ds), len(metrics)

    T = oof[tcols].to_numpy(np.float32)
    P = oof[pcols].to_numpy(np.float32)
    SELv = _select_topk(P)                       # per (dataset, view)
    row = oof["dataset_id"].map(pos).to_numpy()
    view = oof["view_mode"].to_numpy()

    # ---- dataset-level aggregation: mean over valid views / selection rate ----
    def _acc(M: np.ndarray, valid: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, np.ndarray]:
        ok = np.isfinite(M) if valid is None else valid
        s = np.zeros((n, p), np.float64)
        c = np.zeros((n, p), np.float64)
        np.add.at(s, row, np.where(ok, np.nan_to_num(M, nan=0.0), 0.0))
        np.add.at(c, row, ok.astype(np.float64))
        return s, c

    st, ct = _acc(T)
    sp, cp = _acc(P)
    with np.errstate(invalid="ignore", divide="ignore"):
        TRUE = np.where(ct > 0, st / np.maximum(ct, 1), np.nan).astype(np.float32)
        PRED = np.where(cp > 0, sp / np.maximum(cp, 1), np.nan).astype(np.float32)
    # selection rate: fraction of the dataset's views in which m is selected
    ss = np.zeros((n, p), np.float64)
    nv = np.zeros(n, np.float64)
    np.add.at(ss, row, SELv.astype(np.float64))
    np.add.at(nv, row, 1.0)
    SEL = (ss / nv[:, None]).astype(np.float32)

    view_true, view_sel, view_index = {}, {}, {}
    for v in C.VIEWS:
        m = view == v
        idx = row[m]
        vt = np.full((n, p), np.nan, np.float32)
        vs = np.zeros((n, p), np.float32)
        vt[idx] = T[m]
        vs[idx] = SELv[m]
        view_true[v], view_sel[v], view_index[v] = vt, vs, idx

    parent = (oof[["subfamily_id", "family_id"]].drop_duplicates()
              .set_index("subfamily_id")["family_id"].to_dict())
    prov = {
        "utility_table": C.OOF_REL,
        "population": "Splits 2-16 (Split 1 absent)",
        "n_datasets": int(n), "n_view_records": int(len(oof)),
        "n_metrics_utility_space": int(p),
        "n_live_metrics": len(all_live),
        "live_without_clustopt_utility": no_utility,
        "aggregation": {
            "utility": "mean over the dataset's valid views",
            "predicted_utility": "mean over the dataset's valid views",
            "selection": "fraction of the dataset's views in which the metric is "
                         "in the Top-%d by predicted utility" % TOP_K},
        "selection_semantics": {
            "mechanism": "Full60 MLP Top-%d, normalised-positive (RAW) weighting" % TOP_K,
            "frozen_arm": C.CLUSTOPT_ARM_ID,
            "K": TOP_K,
            "prediction_artifact": C.OOF_KNN_REL if knn else C.OOF_REL,
            "utility_artifact": C.OOF_REL,
            "rule": "argsort of stored predicted utility; no model executed",
            "verified_against": ("stage1_2x3_aggregation/online_fixed50/"
                                 "online_2x3_dataset_results.csv "
                                 "(resolver_top_k=5, arm FP)")},
        "inferential_unit": "dataset",
        "view_level_use": "descriptive only (view-specificity section)",
    }
    log("data: %d datasets x %d metrics (%d view records); families %d, subfamilies %d"
        % (n, p, len(oof), ds["family"].nunique(), ds["subfamily"].nunique()))
    log("live metrics without a ClustOpt utility (NA with reason): %s" % no_utility)
    return SpecData(metrics=metrics, all_live=all_live, no_utility=no_utility,
                    ds=ds, TRUE=TRUE, PRED=PRED, SEL=SEL, view_true=view_true,
                    view_sel=view_sel, view_index=view_index, atlas=atlas,
                    fam_of=ds["family"].to_numpy(), sub_of=ds["subfamily"].to_numpy(),
                    parent=parent, provenance=prov)


def group_counts(D: SpecData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    f = (D.ds.groupby("family").size().rename("n_datasets").reset_index())
    s = (D.ds.groupby(["family", "subfamily"]).size()
         .rename("n_datasets").reset_index())
    s["eligible_for_inference"] = s["n_datasets"] >= 8
    return f, s


def nanmean_rows(M: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Column means of the selected rows, ignoring NaN."""
    if not mask.any():
        return np.full(M.shape[1], np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(M[mask], axis=0)


def ranks_desc(v: np.ndarray) -> np.ndarray:
    """Dense competition ranks, 1 = largest; NaN stays NaN."""
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.any():
        order = np.argsort(-v[ok], kind="stable")
        r = np.empty(ok.sum())
        r[order] = np.arange(1, ok.sum() + 1)
        out[ok] = r
    return out
