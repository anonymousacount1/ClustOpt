"""ARI trace-best / opportunity-gap analysis.

For every successful ``dataset x view x method`` run, ClustOpt evaluated many
candidate configurations and recorded them in ``clustopt_results.csv``. The
*selected* clustering is the one the internal objective (``aggregate_score``)
chose; some non-selected candidate in the trace may have had a higher external
ARI. The gap between the best ARI in the trace and the selected ARI measures
how much performance the objective left on the table:

    trace_best_ari            = max ARI among all trace candidates
    selected_ari              = ARI of the objective-selected candidate
    ari_gap_to_trace_best     = trace_best_ari - selected_ari
    selected_is_trace_best    = |gap| <= EPS
    relative_ari_gap          = gap / max(|trace_best_ari|, EPS)

Interpretation:
 * low gap  -> the objective picked nearly the best partition the search found.
 * high gap -> the search *found* a good partition but the objective failed to
   select it (objective failure, not search-space failure).

Robust to missing ``clustopt_results.csv`` / missing ARI columns: trace fields
become NaN and a warning counter is incremented (never crashes aggregation).
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .method_registry import ondisk_name

EPS = 1e-9

# Robust column detection (inspect actual files before trusting any single name).
_ARI_COL_CANDIDATES = ("ARI", "ari", "external_ari", "adjusted_rand_index")
_OBJ_COL_CANDIDATES = (
    "aggregate_score", "score", "objective", "best_objective_score",
    "aggregate_objective", "cvi_score",
)
_VALID_COL_CANDIDATES = ("valid",)

# Columns merged into the raw_results table.
TRACE_RAW_COLUMNS = [
    "trace_best_ari", "selected_ari", "ari_gap_to_trace_best", "relative_ari_gap",
    "selected_is_trace_best", "trace_best_rank_by_internal_objective",
    "selected_rank_by_ari", "n_trace_candidates", "trace_best_candidate_index",
    "selected_candidate_index", "trace_best_selected_algorithm",
    "trace_best_internal_objective",
]


def _ext(path: str | Path) -> str:
    r"""Long-path-safe absolute path string (``\\?\`` prefix on Windows)."""
    s = os.path.abspath(str(path))
    if sys.platform == "win32" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _split_dir_name(split_id: int) -> str:
    return f"split_{int(split_id):02d}"


def _run_dir_for(row: pd.Series) -> Path:
    # method_name in the frame is the trimmed display name; map it back to the
    # phase-tagged on-disk folder to locate clustopt_results.csv.
    #
    # Ambiguity: some Phase-D methods reuse the *untagged* baseline/regressor
    # folder names (e.g. ``silhouette_single``, ``regressor_top5_softmax_t05``)
    # that happen to equal Phase-A/B *display* names, so ``ondisk_name`` maps
    # them to the phase-tagged folder (``silhouette_single_phase_A``) which the
    # Phase-D merge deleted. Prefer the mapped folder only when it exists on disk
    # (keeps Phase-A/B resolution byte-identical), else fall back to the literal
    # display name -- which for Phase-C/D *is* the on-disk folder name.
    base = (Path(str(row["dataset_dir"])) / "experiments"
            / _split_dir_name(int(row["split_id"])))
    view = str(row["view_id"])
    mapped_dir = base / ondisk_name(str(row["method_name"])) / view
    if os.path.exists(_ext(mapped_dir)):
        return mapped_dir
    literal_dir = base / str(row["method_name"]) / view
    if os.path.exists(_ext(literal_dir)):
        return literal_dir
    return mapped_dir  # neither exists -> preserve the 'missing' warning path


def compute_trace_best_for_run(
    run_dir: Path, *, selected_ari: Optional[float], selected_objective: Optional[float],
) -> Tuple[Dict, Optional[str]]:
    """Return (trace fields dict, warning-or-None) for one run directory."""
    nan_fields = {c: float("nan") for c in TRACE_RAW_COLUMNS}
    nan_fields["selected_is_trace_best"] = float("nan")
    nan_fields["trace_best_selected_algorithm"] = None
    csv_path = run_dir / "clustopt_results.csv"
    if not os.path.exists(_ext(csv_path)):
        return nan_fields, "missing_clustopt_results"
    try:
        trace = pd.read_csv(_ext(csv_path))
    except Exception as exc:  # unreadable / empty file
        return nan_fields, f"unreadable_clustopt_results:{type(exc).__name__}"
    if trace.empty:
        return nan_fields, "empty_clustopt_results"

    ari_col = _find_col(trace, _ARI_COL_CANDIDATES)
    obj_col = _find_col(trace, _OBJ_COL_CANDIDATES)
    if ari_col is None:
        return nan_fields, "no_ari_column"

    valid_col = _find_col(trace, _VALID_COL_CANDIDATES)
    work = trace.copy()
    work["_ari"] = pd.to_numeric(work[ari_col], errors="coerce")
    if obj_col is not None:
        work["_obj"] = pd.to_numeric(work[obj_col], errors="coerce")
    else:
        work["_obj"] = np.nan
    if valid_col is not None:
        valid_mask = work[valid_col].astype("object").apply(
            lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes")
        )
        # Keep valid rows; if that removes everything, fall back to all rows.
        if int(valid_mask.sum()) > 0:
            work = work[valid_mask]

    cand = work[np.isfinite(work["_ari"])].reset_index(drop=True)
    if cand.empty:
        return nan_fields, "no_finite_ari"

    n_cand = int(len(cand))
    # Best-by-ARI candidate (trace-best opportunity).
    best_ari_idx = int(cand["_ari"].idxmax())
    trace_best_ari = float(cand.loc[best_ari_idx, "_ari"])

    # Selected candidate = max internal objective (matches best_objective_score).
    sel_idx: Optional[int]
    if cand["_obj"].notna().any():
        sel_idx = int(cand["_obj"].idxmax())
    else:
        sel_idx = None

    # selected_ari: trust the run's external ARI if provided, else trace-derived.
    sel_ari_val = (
        float(selected_ari)
        if selected_ari is not None and np.isfinite(selected_ari)
        else (float(cand.loc[sel_idx, "_ari"]) if sel_idx is not None else float("nan"))
    )

    gap = trace_best_ari - sel_ari_val if np.isfinite(sel_ari_val) else float("nan")
    rel_gap = (
        gap / max(abs(trace_best_ari), EPS)
        if np.isfinite(gap) else float("nan")
    )
    is_best = float(abs(gap) <= EPS) if np.isfinite(gap) else float("nan")

    # Ranks (1-based). Objective rank of the trace-best-ARI candidate; ARI rank
    # of the selected candidate. Higher value -> better -> rank 1.
    trace_best_rank_obj = float("nan")
    if cand["_obj"].notna().any():
        obj_order = cand["_obj"].rank(ascending=False, method="min")
        trace_best_rank_obj = float(obj_order.iloc[best_ari_idx])
    ari_order = cand["_ari"].rank(ascending=False, method="min")
    selected_rank_ari = (
        float(ari_order.iloc[sel_idx]) if sel_idx is not None else float("nan")
    )

    algo_col = _find_col(cand, ("algorithm", "selected_algorithm"))
    trace_best_algo = (
        str(cand.loc[best_ari_idx, algo_col]) if algo_col is not None else None
    )

    fields = {
        "trace_best_ari": trace_best_ari,
        "selected_ari": sel_ari_val,
        "ari_gap_to_trace_best": gap,
        "relative_ari_gap": rel_gap,
        "selected_is_trace_best": is_best,
        "trace_best_rank_by_internal_objective": trace_best_rank_obj,
        "selected_rank_by_ari": selected_rank_ari,
        "n_trace_candidates": float(n_cand),
        "trace_best_candidate_index": float(best_ari_idx),
        "selected_candidate_index": float(sel_idx) if sel_idx is not None else float("nan"),
        "trace_best_selected_algorithm": trace_best_algo,
        "trace_best_internal_objective": (
            float(cand.loc[best_ari_idx, "_obj"]) if cand["_obj"].notna().any() else float("nan")
        ),
    }
    return fields, None


def _trace_best_for_row(r: pd.Series) -> Tuple[Dict, Optional[str]]:
    """Compute (row dict incl. keys, warning) for one successful raw row."""
    run_dir = _run_dir_for(r)
    sel_ari = pd.to_numeric(pd.Series([r.get("ari")]), errors="coerce").iloc[0]
    sel_obj = pd.to_numeric(pd.Series([r.get("best_objective_score")]), errors="coerce").iloc[0]
    fields, warn = compute_trace_best_for_run(
        run_dir, selected_ari=sel_ari, selected_objective=sel_obj
    )
    row = {
        "dataset_id": r.get("dataset_id"), "split_id": r.get("split_id"),
        "view_id": r.get("view_id"), "method_name": r.get("method_name"),
        **fields,
    }
    return row, warn


def compute_trace_best_table(
    raw: pd.DataFrame, *, verbose: bool = True, max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Per-run trace-best fields for every successful raw row.

    Returns (table keyed by dataset_id/split_id/view_id/method_name, warnings).

    ``max_workers`` (default ``None`` -> serial, unchanged) enables a thread
    pool over the successful runs. Each run reads its own ``clustopt_results.csv``
    (I/O-bound, releases the GIL). ``ThreadPoolExecutor.map`` preserves input
    order, so the table -- and the warning tallies -- match the serial path.
    """
    warnings: Dict[str, int] = {}
    key_cols = ["dataset_id", "split_id", "view_id", "method_name"]
    if raw.empty:
        return pd.DataFrame(columns=key_cols + TRACE_RAW_COLUMNS), warnings

    success = raw[raw["status"] == "success"]
    success_rows = [r for _, r in success.iterrows()]
    n = len(success_rows)
    rows: List[Dict] = []

    if max_workers and max_workers > 1 and n:
        done = 0
        with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
            for row, warn in pool.map(_trace_best_for_row, success_rows):
                rows.append(row)
                if warn:
                    warnings[warn] = warnings.get(warn, 0) + 1
                done += 1
                if verbose and done % 500 == 0:
                    print(f"[trace-best] scanned {done}/{n} successful runs "
                          f"({int(max_workers)} workers)...", flush=True)
    else:
        for i, r in enumerate(success_rows, start=1):
            row, warn = _trace_best_for_row(r)
            rows.append(row)
            if warn:
                warnings[warn] = warnings.get(warn, 0) + 1
            if verbose and i % 500 == 0:
                print(f"[trace-best] scanned {i}/{n} successful runs...", flush=True)

    table = pd.DataFrame(rows, columns=key_cols + TRACE_RAW_COLUMNS)
    if verbose:
        print(f"[trace-best] {len(table)} runs; warnings={warnings or 'none'}", flush=True)
    return table, warnings


def merge_into_raw(raw: pd.DataFrame, trace_table: pd.DataFrame) -> pd.DataFrame:
    """Left-merge per-run trace fields onto raw_results."""
    if raw.empty or trace_table.empty:
        out = raw.copy()
        for c in TRACE_RAW_COLUMNS:
            if c not in out.columns:
                out[c] = np.nan
        return out
    key_cols = ["dataset_id", "split_id", "view_id", "method_name"]
    return raw.merge(trace_table, on=key_cols, how="left")


# --------------------------------------------------------------------------- #
# Grouped summaries
# --------------------------------------------------------------------------- #
def _gap_block(g: pd.DataFrame) -> dict:
    sub = g[g["status"] == "success"] if "status" in g.columns else g
    gap = pd.to_numeric(sub.get("ari_gap_to_trace_best"), errors="coerce")
    tba = pd.to_numeric(sub.get("trace_best_ari"), errors="coerce")
    sel = pd.to_numeric(sub.get("selected_ari"), errors="coerce")
    rel = pd.to_numeric(sub.get("relative_ari_gap"), errors="coerce")
    is_best = pd.to_numeric(sub.get("selected_is_trace_best"), errors="coerce")
    tb_rank = pd.to_numeric(sub.get("trace_best_rank_by_internal_objective"), errors="coerce")
    sel_rank = pd.to_numeric(sub.get("selected_rank_by_ari"), errors="coerce")
    n_gap = int(gap.notna().sum())

    def f(x, fn):
        return float(fn(x.dropna())) if int(x.notna().sum()) else float("nan")

    return {
        "count": int(len(g)),
        "n_success": int(len(sub)),
        "n_with_trace": n_gap,
        "mean_trace_best_ari": f(tba, np.mean),
        "median_trace_best_ari": f(tba, np.median),
        "mean_selected_ari": f(sel, np.mean),
        "median_selected_ari": f(sel, np.median),
        "mean_ari_gap_to_trace_best": f(gap, np.mean),
        "median_ari_gap_to_trace_best": f(gap, np.median),
        "std_ari_gap_to_trace_best": f(gap, np.std),
        "q25_gap": float(gap.dropna().quantile(0.25)) if n_gap else float("nan"),
        "q75_gap": float(gap.dropna().quantile(0.75)) if n_gap else float("nan"),
        "min_gap": f(gap, np.min),
        "max_gap": f(gap, np.max),
        "mean_relative_ari_gap": f(rel, np.mean),
        "median_relative_ari_gap": f(rel, np.median),
        "selected_is_trace_best_rate": f(is_best, np.mean),
        "mean_trace_best_rank_by_internal_objective": f(tb_rank, np.mean),
        "median_trace_best_rank_by_internal_objective": f(tb_rank, np.median),
        "mean_selected_rank_by_ari": f(sel_rank, np.mean),
        "median_selected_rank_by_ari": f(sel_rank, np.median),
    }


def trace_best_summary_by(raw_with_trace: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in group_cols if c in raw_with_trace.columns]
    if raw_with_trace.empty:
        return pd.DataFrame()
    if not cols:
        return pd.DataFrame([_gap_block(raw_with_trace)])
    out = []
    for key, g in raw_with_trace.groupby(cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {c: v for c, v in zip(cols, key_tuple)}
        row.update(_gap_block(g))
        out.append(row)
    df = pd.DataFrame(out)
    sort_cols = [c for c in cols if c != "method_name"] + (
        ["method_name"] if "method_name" in cols else [])
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


def build_dataset_level_trace_best(
    dataset_level: pd.DataFrame, trace_table: pd.DataFrame
) -> pd.DataFrame:
    """One row per (dataset, method): the best-view row's trace-best gap.

    Pairs the best-view row chosen in ``dataset_level_results`` with the trace
    fields of that same (dataset, view, method) run.
    """
    cols = [
        "dataset_id", "method_name", "family", "subfamily", "difficulty",
        "cluster_count", "best_view_id", "best_ari", "trace_best_ari",
        "ari_gap_to_trace_best", "relative_ari_gap", "selected_is_trace_best",
        "selected_k", "true_k", "selected_algorithm", "runtime_sec",
    ]
    if dataset_level.empty:
        return pd.DataFrame(columns=cols)
    tt = trace_table.rename(columns={"view_id": "best_view_id"})
    keep = ["dataset_id", "best_view_id", "method_name", "trace_best_ari",
            "ari_gap_to_trace_best", "relative_ari_gap", "selected_is_trace_best"]
    tt = tt[[c for c in keep if c in tt.columns]]
    merged = dataset_level.merge(
        tt, on=["dataset_id", "best_view_id", "method_name"], how="left"
    )
    merged = merged.rename(columns={
        "best_selected_algorithm": "selected_algorithm",
        "best_selected_k": "selected_k",
        "best_runtime_sec": "runtime_sec",
    })
    for c in cols:
        if c not in merged.columns:
            merged[c] = np.nan
    return merged[cols].reset_index(drop=True)
