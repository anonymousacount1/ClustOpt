"""Phase-2 comparison engine: assemble every heuristic (Phase-1 + new), score
their K distributions, measure stability and agreement, and run the optional
read-only split-1 diagnostic.

Nothing here trains or tunes anything. The split-1 step is a *sanity diagnostic
only* (``split1_diagnostic_only``): it maps each heuristic's selected K onto the
already-computed fixed Top-K results and reports what ARI that would have yielded.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import distribution_metrics as dm
from heuristic_refinement import build_new_heuristics, new_heuristic_columns

# Existing Phase-1 heuristics (kept, per section 4).
from heuristic_candidates import heuristic_columns as phase1_heuristic_columns

META_COLS = ["record_id", "dataset_id", "family_id", "subfamily_id",
             "view_type", "split_id"]
GAP_COLS = [f"gap_{i}" for i in range(1, 11)] + ["largest_gap_value_1_to_10"]

# split-1 diagnostic: fixed Top-K methods in the result package.
TOPK_METHODS = {
    "regressor_top1_softmax_t05": "TOP1",
    "regressor_top3_softmax_t05": "TOP3",
    "regressor_top5_softmax_t05": "TOP5",
    "regressor_top10_softmax_t05": "TOP10",
}
SPLIT1_RESULTS = ("results_analysis/clustering_repository/experiment_result_packages/"
                  "20260630_100348__split1_phaseC_regressors_vs_indomain/"
                  "11_Appendix_Raw_Files/raw/dataset_level_results.csv")


# --------------------------------------------------------------------------- #
# Loading / assembly                                                          #
# --------------------------------------------------------------------------- #
def find_phase1_dir(repo_root: str, override: str | None) -> str:
    if override:
        return override
    pat = os.path.join(repo_root, "results_analysis",
                       "dynamic_topk_utility_pre_analysis_*")
    dirs = sorted(glob.glob(pat))
    if not dirs:
        raise FileNotFoundError(f"no Phase-1 output dir matching {pat}")
    return dirs[-1]


def load_and_assemble(phase1_dir: str):
    """Load Phase-1 record tables and merge into one wide working frame."""
    summary = pd.read_parquet(os.path.join(phase1_dir,
                                            "records_utility_summary.parquet"))
    wide = pd.read_parquet(os.path.join(phase1_dir,
                                         "records_utility_vectors.parquet"))
    p1cols = [c for c in phase1_heuristic_columns() if c in summary.columns]
    keep = META_COLS + GAP_COLS + ["k_elbow_10", "max_clean"] + p1cols
    seen: set[str] = set()
    keep = [c for c in keep if c in summary.columns
            and not (c in seen or seen.add(c))]   # dedupe, preserve order
    base = summary[keep].merge(wide, on="record_id", how="left")
    new = build_new_heuristics(base)
    confidence = new.pop("_confidence")
    full = pd.concat([base, new], axis=1)
    full["_confidence"] = confidence
    return full, p1cols, list(new.columns)


# --------------------------------------------------------------------------- #
# Distribution metrics + stability for all heuristics                         #
# --------------------------------------------------------------------------- #
def all_metrics(full: pd.DataFrame, heuristics: list[str]) -> pd.DataFrame:
    rows = []
    for c in heuristics:
        k = full[c].to_numpy()
        rec = {"heuristic": c, "family": _family_of(c)}
        rec.update(dm.k_distribution_metrics(k))
        rec.update(dm.stability(full, c))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["composite_score"] = out.apply(dm.composite_score, axis=1)
    return out


def _family_of(col: str) -> str:
    if col.startswith("k_relcap_"):
        return "relative+hardcap"
    if col.startswith("k_relsoft_"):
        return "relative+softcap"
    if col.startswith("k_relelbow_"):
        return "relative+elbowguard"
    if col.startswith("k_relconf_"):
        return "relative+gapconf"
    if col.startswith("k_rel_a"):
        return "relative_fine"
    if col.startswith("k_rel_"):
        return "relative(phase1)"
    if col.startswith("k_elbow"):
        return "elbow(phase1)"
    if col.startswith("k_mass") or col.startswith("k_eff"):
        return "mass/effK(phase1)"
    if col.startswith("k_hybrid"):
        return "hybrid(phase1)"
    return "other"


# --------------------------------------------------------------------------- #
# Split-1 diagnostic (read-only, no training/tuning)                          #
# --------------------------------------------------------------------------- #
def _bucket_index(k: np.ndarray) -> np.ndarray:
    """Map K -> fixed-Top-K column index: 1->TOP1, {2,3}->TOP3, {4,5}->TOP5,
    {>=6}->TOP10 (section 9; 6-9 approximated by Top10)."""
    k = np.asarray(k, dtype=float)
    idx = np.full(k.shape, 3, dtype=int)      # default TOP10
    idx[k <= 1] = 0
    idx[(k >= 2) & (k <= 3)] = 1
    idx[(k >= 4) & (k <= 5)] = 2
    return idx


def split1_diagnostic(full: pd.DataFrame, heuristics: list[str],
                      repo_root: str):
    """Return (per-heuristic diagnostic DataFrame, baseline dict) or (None, None).

    Per dataset we reduce the (<=3) per-view K to a representative K via the
    rounded median across views, map it to a fixed Top-K bucket, and read off the
    already-computed ARI. The heuristic's diagnostic ARI is the mean across the
    1,055 split-1 datasets. STRICTLY a sanity check -- never an experiment.
    """
    path = os.path.join(repo_root, SPLIT1_RESULTS)
    if not os.path.exists(path):
        return None, None
    res = pd.read_csv(path)
    res = res[res["method_name"].isin(TOPK_METHODS)].copy()
    res["topk"] = res["method_name"].map(TOPK_METHODS)
    piv = res.pivot_table(index="dataset_id", columns="topk",
                          values="best_ari", aggfunc="first")
    order = ["TOP1", "TOP3", "TOP5", "TOP10"]
    piv = piv[order].dropna()
    ari = piv.to_numpy()                       # datasets x 4
    n = len(piv)

    baselines = {t: float(piv[t].mean()) for t in order}
    baselines["ORACLE_top_k"] = float(piv.max(axis=1).mean())
    baselines["n_datasets"] = n

    # per-dataset representative K for every heuristic at once
    s1 = full[full["split_id"] == 1]
    med = s1.groupby("dataset_id")[heuristics].median()
    med = med.reindex(piv.index)               # align to ARI rows
    rows = []
    best_fixed = max(baselines["TOP1"], baselines["TOP3"],
                     baselines["TOP5"], baselines["TOP10"])
    for c in heuristics:
        kvals = np.rint(med[c].to_numpy())
        bidx = _bucket_index(kvals)
        sel_ari = ari[np.arange(n), bidx]
        # bucket usage distribution
        bd = np.bincount(bidx, minlength=4) / n * 100
        rows.append({
            "heuristic": c,
            "split1_diag_ari": float(np.nanmean(sel_ari)),
            "split1_delta_vs_best_fixed": float(np.nanmean(sel_ari) - best_fixed),
            "split1_mean_repr_K": float(np.nanmean(kvals)),
            "pct_bucket_TOP1": float(bd[0]), "pct_bucket_TOP3": float(bd[1]),
            "pct_bucket_TOP5": float(bd[2]), "pct_bucket_TOP10": float(bd[3]),
        })
    return pd.DataFrame(rows), baselines


# --------------------------------------------------------------------------- #
# Validation (section 14)                                                     #
# --------------------------------------------------------------------------- #
def validate(full: pd.DataFrame, p1cols: list[str], newcols: list[str],
             n_expected: int) -> dict:
    rep = {
        "n_records": int(len(full)),
        "n_records_expected": int(n_expected),
        "record_count_matches": bool(len(full) == n_expected),
        "duplicate_record_ids": int(full["record_id"].duplicated().sum()),
        "n_phase1_heuristics_present": int(sum(c in full.columns for c in p1cols)),
        "n_phase1_heuristics_expected": len(p1cols),
        "all_phase1_present": bool(all(c in full.columns for c in p1cols)),
        "n_new_heuristics": len(newcols),
        "new_heuristics_no_nan": bool(full[newcols].notna().all().all()),
        "new_heuristics_in_range": bool(
            ((full[newcols] >= 1) & (full[newcols] <= 10)).all().all()),
    }
    return rep
