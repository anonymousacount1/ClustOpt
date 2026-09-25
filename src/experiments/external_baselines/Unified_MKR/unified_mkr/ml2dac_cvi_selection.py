"""CVI* selection for the derived ML2DAC MKRs.

Mirrors ML2DAC's LearningPhase.get_best_cvi_for_dataset: for each dataset
(here: each per-view record) the best CVI is the one whose Spearman rank
correlation with ARI over that record's configurations is highest.

The one substantive adaptation vs. the original code is metric-direction
handling. ML2DAC's CVICollection stores every CVI in a single "loss"
(lower-is-better) convention *including ARI*, so the sign cancels and a raw
spearman(ARI_loss, CVI_loss) is already meaningful. Our master repository
stores RAW metric values in mixed conventions, so we orient each metric to
higher-is-better before correlating with (raw, higher-is-better) ARI:

* the 7 ML2DAC CVIs use their explicit raw direction from metric_registry;
* the ClustOpt structural metrics carry direction sentinel ``clustopt_utility``
  and are oriented by probing ClustOpt's own ``normalize()`` monotonicity
  (verified independent of cluster count, so a single sign per metric is exact
  for per-record Spearman — no n_clusters reordering occurs).

Spearman is invariant to monotone-increasing rescaling, so orienting by sign is
identical to correlating against the full ClustOpt utility transform.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr  # noqa: F401  (spearmanr kept for parity/reference)

# The 7 ML2DAC CVIs and their explicit raw direction (ClustOpt canonical names).
ML2DAC_CVI_DIRECTION: Dict[str, str] = {
    "calinski_harabasz": "maximize",
    "davies_bouldin": "minimize",
    "silhouette": "maximize",
    "dbcv": "maximize",
    "dunn_index": "maximize",
    "coggins_jain_index": "maximize",
    "cop": "minimize",
}
ML2DAC_CVI_IDS: Tuple[str, ...] = tuple(ML2DAC_CVI_DIRECTION.keys())

MIN_VALID_PAIRS = 5


def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        log(msg)


# --- direction resolution --------------------------------------------------

def _probe_clustopt_direction(metric_obj) -> str:
    """maximize/minimize for a ClustOpt metric by probing normalize() sign."""
    from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext
    raws = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    signs = []
    for ns, nc in [(500, 3), (2000, 8), (5000, 15)]:
        ctx = MetricContext(n_samples=ns, n_clusters=nc)
        vals = []
        for r in raws:
            try:
                vals.append(float(metric_obj.normalize(float(r), ctx)))
            except Exception:
                vals.append(np.nan)
        vals = np.asarray(vals, dtype=float)
        ok = ~np.isnan(vals)
        if ok.sum() < 3:
            continue
        c = np.corrcoef(rankdata(np.asarray(raws)[ok]), rankdata(vals[ok]))[0, 1]
        if not np.isnan(c) and abs(c) > 1e-9:
            signs.append(np.sign(c))
    if not signs:
        return "maximize"  # default; flagged by caller if needed
    return "maximize" if np.mean(signs) > 0 else "minimize"


def resolve_directions(
    candidate_ids: List[str],
    metric_registry: pd.DataFrame,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Resolve each candidate metric to 'maximize'/'minimize' (higher-is-better)."""
    try:
        from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
            METRIC_REGISTRY as CLUSTOPT_REG,
        )
    except Exception:  # pragma: no cover
        CLUSTOPT_REG = {}

    reg_dir = dict(zip(metric_registry["metric_id"], metric_registry["direction"]))
    out: Dict[str, str] = {}
    for mid in candidate_ids:
        if mid in ML2DAC_CVI_DIRECTION:
            out[mid] = ML2DAC_CVI_DIRECTION[mid]
        elif reg_dir.get(mid) in ("maximize", "minimize"):
            out[mid] = reg_dir[mid]
        elif mid in CLUSTOPT_REG:
            out[mid] = _probe_clustopt_direction(CLUSTOPT_REG[mid])
        else:
            out[mid] = "maximize"
            _log(log, f"[cvi] WARNING: no direction for {mid}; defaulting to maximize")
    return out


# --- CVI* computation ------------------------------------------------------

def _spearman_pairwise(a: np.ndarray, b: np.ndarray, min_pairs: int) -> Tuple[float, int]:
    m = ~np.isnan(a) & ~np.isnan(b)
    n = int(m.sum())
    if n < min_pairs:
        return float("nan"), n
    ra = rankdata(a[m])
    rb = rankdata(b[m])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan"), n
    c = float(np.corrcoef(ra, rb)[0, 1])
    return c, n


def compute_cvi_star(
    evaluations: pd.DataFrame,
    candidate_ids: List[str],
    directions: Dict[str, str],
    records: pd.DataFrame,
    *,
    min_valid_pairs: int = MIN_VALID_PAIRS,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cvi_star_df, correlations_df).

    For every record, orient each candidate metric to higher-is-better and take
    the Spearman correlation with ARI over the record's configurations; the best
    CVI is the argmax correlation (>= min_valid_pairs valid pairs required).
    """
    cols = ["record_id", "ari"] + candidate_ids
    ev = evaluations[cols].copy()
    # Orient: minimize -> negate so higher is always better (matches ARI).
    for mid in candidate_ids:
        if directions.get(mid) == "minimize":
            ev[mid] = -ev[mid]

    meta = records.set_index("record_id")[
        ["split_id", "family", "subfamily", "view_type"]
    ].to_dict("index")

    star_rows: List[dict] = []
    corr_rows: List[dict] = []
    n_done = 0
    for rid, g in ev.groupby("record_id", sort=False):
        a = g["ari"].to_numpy(dtype=float)
        best_cvi, best_corr = "", -np.inf
        n_valid_configs = int((~np.isnan(a)).sum())
        for mid in candidate_ids:
            c, npairs = _spearman_pairwise(a, g[mid].to_numpy(dtype=float), min_valid_pairs)
            corr_rows.append({
                "record_id": rid, "metric_id": mid,
                "spearman_correlation": c, "n_valid_pairs": npairs,
            })
            if not np.isnan(c) and c > best_corr:
                best_corr, best_cvi = c, mid
        m = meta.get(rid, {})
        resolved = best_cvi != ""
        star_rows.append({
            "record_id": rid,
            "split_id": m.get("split_id"),
            "family": m.get("family"),
            "subfamily": m.get("subfamily"),
            "view_type": m.get("view_type"),
            "best_cvi": best_cvi if resolved else None,
            "best_correlation": float(best_corr) if resolved else float("nan"),
            "n_valid_configs": n_valid_configs,
            "resolved": resolved,
        })
        n_done += 1
        if log and n_done % 5000 == 0:
            _log(log, f"[cvi] CVI* {n_done} records")

    return pd.DataFrame(star_rows), pd.DataFrame(corr_rows)
