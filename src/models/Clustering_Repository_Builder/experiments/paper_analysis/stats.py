"""Paired statistics with explicit confirmatory / exploratory labelling.

Every test is paired by ``dataset_id`` on Best-View ARI.

Why the labelling matters
-------------------------
Representatives are chosen **retrospectively on Split 1** and then tested against
the *same* Split-1 outcomes. Those p-values are optimistic by construction: the
winner was picked because it looked good on exactly this data. Any test whose
either arm is a retrospectively selected representative is therefore stamped
``post_selection_exploratory`` and must never be called confirmatory evidence.

Tests between two **fixed** evaluated variants are not affected by selection on
their own, but they are still only confirmatory when they are on the predeclared
primary-hypothesis list; everything else is ``exploratory``. Holm correction is
applied *within each hypothesis family*, never across the whole pile.

:func:`selection_aware_bootstrap` is the honest way to compare a retrospective
representative: it re-runs the selection rule inside every bootstrap resample, so
the interval reflects uncertainty from **both** dataset sampling and
representative selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:                                        # pragma: no cover - env dependent
    from scipy import stats as _scipy_stats
    _SCIPY = True
except Exception:                           # pragma: no cover
    _scipy_stats = None
    _SCIPY = False

from .method_registry import short_label

CLAIM_CONFIRMATORY = "confirmatory"
CLAIM_EXPLORATORY = "exploratory"
CLAIM_POST_SELECTION = "post_selection_exploratory"

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 20260725


def scipy_available() -> bool:
    return _SCIPY


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
def paired_values(dataset_level: pd.DataFrame, method_a: str, method_b: str,
                  *, value_col: str = "best_view_ari",
                  group: Optional[Tuple[str, Any]] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligned ``(a, b, dataset_ids)`` arrays for two methods.

    Pairing is a strict inner join on ``dataset_id``: a dataset counts only when
    both arms produced a finite value there.
    """
    frame = dataset_level
    if group is not None:
        col, key = group
        frame = frame[frame[col] == key]
    a = frame.loc[frame["method_name"] == method_a, ["dataset_id", value_col]]
    b = frame.loc[frame["method_name"] == method_b, ["dataset_id", value_col]]
    merged = a.merge(b, on="dataset_id", suffixes=("_a", "_b"))
    va = pd.to_numeric(merged[f"{value_col}_a"], errors="coerce").to_numpy(float)
    vb = pd.to_numeric(merged[f"{value_col}_b"], errors="coerce").to_numpy(float)
    ids = merged["dataset_id"].to_numpy()
    mask = np.isfinite(va) & np.isfinite(vb)
    return va[mask], vb[mask], ids[mask]


# --------------------------------------------------------------------------- #
# Effect sizes and intervals
# --------------------------------------------------------------------------- #
def cohens_dz(delta: np.ndarray) -> float:
    if delta.size < 2:
        return float("nan")
    sd = float(np.std(delta, ddof=1))
    return float(np.mean(delta) / sd) if sd > 0 else float("nan")


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from signed ranks.

    ``r = (W+ - W-) / (W+ + W-)``: +1 means every non-tied pair favours ``a``.
    Ties are excluded from the ranking, matching the Wilcoxon convention.
    """
    delta = a - b
    nz = delta[delta != 0]
    if nz.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    w_pos = float(ranks[nz > 0].sum())
    w_neg = float(ranks[nz < 0].sum())
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total > 0 else 0.0


def bootstrap_ci(delta: np.ndarray, *, statistic: str = "mean",
                 n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
                 confidence: float = DEFAULT_CONFIDENCE,
                 seed: int = DEFAULT_SEED) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the paired mean/median difference."""
    if delta.size == 0:
        return float("nan"), float("nan"), float("nan")
    fn = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, delta.size, size=(int(n_samples), delta.size))
    stats_ = fn(delta[idx], axis=1)
    lo, hi = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    return (float(fn(delta)), float(np.quantile(stats_, lo)),
            float(np.quantile(stats_, hi)))


# --------------------------------------------------------------------------- #
# One paired test
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    """One predeclared or exploratory paired comparison."""

    hypothesis_id: str
    family: str                       # Holm-correction family
    method_a: str
    method_b: str
    claim_level: str = CLAIM_EXPLORATORY
    direction: str = "two-sided"      # 'two-sided' | 'a_greater' | 'b_greater'
    rationale: str = ""
    is_primary: bool = False
    a_is_representative: bool = False
    b_is_representative: bool = False

    @property
    def effective_claim_level(self) -> str:
        """Post-selection whenever either arm is a retrospective representative."""
        if self.a_is_representative or self.b_is_representative:
            return CLAIM_POST_SELECTION
        return self.claim_level


def paired_test(dataset_level: pd.DataFrame, hyp: Hypothesis, *,
                value_col: str = "best_view_ari",
                group: Optional[Tuple[str, Any]] = None,
                n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
                confidence: float = DEFAULT_CONFIDENCE,
                seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Full paired test row for one hypothesis."""
    a, b, _ids = paired_values(dataset_level, hyp.method_a, hyp.method_b,
                               value_col=value_col, group=group)
    delta = a - b
    n = int(delta.size)
    row: Dict[str, Any] = {
        "hypothesis_id": hyp.hypothesis_id,
        "hypothesis_family": hyp.family,
        "method_a": hyp.method_a,
        "method_a_short_label": short_label(hyp.method_a),
        "method_b": hyp.method_b,
        "method_b_short_label": short_label(hyp.method_b),
        "value_col": value_col,
        "direction": hyp.direction,
        "is_primary": hyp.is_primary,
        "claim_level": hyp.effective_claim_level,
        "a_is_representative": hyp.a_is_representative,
        "b_is_representative": hyp.b_is_representative,
        "rationale": hyp.rationale,
        "n_paired": n,
        "mean_a": float(np.mean(a)) if n else float("nan"),
        "mean_b": float(np.mean(b)) if n else float("nan"),
        "mean_delta": float(np.mean(delta)) if n else float("nan"),
        "median_delta": float(np.median(delta)) if n else float("nan"),
        "std_delta": float(np.std(delta, ddof=1)) if n > 1 else float("nan"),
        "share_a_better": float((delta > 1e-12).mean()) if n else float("nan"),
        "share_tied": float(np.isclose(delta, 0.0, atol=1e-12).mean()) if n else float("nan"),
        "share_b_better": float((delta < -1e-12).mean()) if n else float("nan"),
        "cohens_dz": cohens_dz(delta) if n else float("nan"),
        "rank_biserial": rank_biserial(a, b) if n else float("nan"),
        "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
        "ttest_t": float("nan"), "ttest_p": float("nan"),
        "scipy_available": _SCIPY,
    }
    if n:
        for stat_name, prefix in (("mean", "mean"), ("median", "median")):
            point, lo, hi = bootstrap_ci(delta, statistic=stat_name,
                                         n_samples=n_bootstrap,
                                         confidence=confidence, seed=seed)
            row[f"{prefix}_delta_boot"] = point
            row[f"{prefix}_delta_ci_low"] = lo
            row[f"{prefix}_delta_ci_high"] = hi
        row["bootstrap_samples"] = int(n_bootstrap)
        row["confidence"] = float(confidence)
    if _SCIPY and n >= 2:
        alt = {"two-sided": "two-sided", "a_greater": "greater",
               "b_greater": "less"}[hyp.direction]
        try:
            if np.any(delta != 0):
                res = _scipy_stats.wilcoxon(a, b, alternative=alt)
                row["wilcoxon_stat"] = float(res.statistic)
                row["wilcoxon_p"] = float(res.pvalue)
        except Exception:                                     # pragma: no cover
            pass
        try:
            res = _scipy_stats.ttest_rel(a, b, alternative=alt)
            row["ttest_t"] = float(res.statistic)
            row["ttest_p"] = float(res.pvalue)
        except Exception:                                     # pragma: no cover
            pass
    return row


# --------------------------------------------------------------------------- #
# Holm correction, per family
# --------------------------------------------------------------------------- #
def holm_correct(frame: pd.DataFrame, *, p_col: str = "wilcoxon_p",
                 family_col: str = "hypothesis_family",
                 alpha: float = 0.05) -> pd.DataFrame:
    """Holm-Bonferroni within each hypothesis family.

    Correcting *within* families keeps a suite's own primary hypotheses from being
    penalised by unrelated exploratory tests elsewhere in the run.
    """
    if frame.empty or p_col not in frame.columns:
        return frame
    out = frame.copy()
    out["holm_p"] = np.nan
    out["holm_reject"] = pd.NA
    out["holm_family_size"] = np.nan
    out["holm_alpha"] = float(alpha)
    for family, idx in out.groupby(family_col, dropna=False).groups.items():
        block = out.loc[idx]
        p = pd.to_numeric(block[p_col], errors="coerce")
        valid = p.dropna().sort_values()
        m = int(len(valid))
        out.loc[idx, "holm_family_size"] = m
        if m == 0:
            continue
        adjusted: Dict[Any, float] = {}
        running = 0.0
        for rank, (label, value) in enumerate(valid.items()):
            candidate = float((m - rank) * value)
            running = max(running, min(1.0, candidate))
            adjusted[label] = running
        for label, value in adjusted.items():
            out.loc[label, "holm_p"] = value
            out.loc[label, "holm_reject"] = bool(value <= alpha)
    return out


def run_hypotheses(dataset_level: pd.DataFrame, hypotheses: Sequence[Hypothesis],
                   *, value_col: str = "best_view_ari",
                   group_col: Optional[str] = None,
                   n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
                   alpha: float = 0.05,
                   seed: int = DEFAULT_SEED,
                   verbose: bool = True) -> pd.DataFrame:
    """Run every hypothesis (optionally per group) and Holm-correct per family."""
    rows: List[Dict[str, Any]] = []
    groups: List[Optional[Tuple[str, Any]]] = [None]
    if group_col and group_col in dataset_level.columns:
        groups = [(group_col, key)
                  for key in sorted(dataset_level[group_col].dropna().unique())]
    for grp in groups:
        for hyp in hypotheses:
            row = paired_test(dataset_level, hyp, value_col=value_col, group=grp,
                              n_bootstrap=n_bootstrap, seed=seed)
            if grp is not None:
                row[grp[0]] = grp[1]
                row["hypothesis_family"] = f"{hyp.family}::{grp[1]}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = holm_correct(frame, alpha=alpha)
    if verbose:
        n_post = int((frame["claim_level"] == CLAIM_POST_SELECTION).sum())
        print(f"[stats] {len(frame)} paired tests "
              f"({n_post} post-selection exploratory); scipy={_SCIPY}", flush=True)
    return frame


# --------------------------------------------------------------------------- #
# Selection-aware bootstrap
# --------------------------------------------------------------------------- #
def selection_aware_bootstrap(
    best_view: pd.DataFrame,
    *,
    candidate_methods: Sequence[str],
    competitor_method: str,
    n_samples: int = 1000,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    label: str = "",
    value_col: str = "best_view_ari",
) -> Dict[str, Any]:
    """Bootstrap that **re-selects the representative inside every resample**.

    Procedure per resample:

    1. resample datasets with replacement;
    2. re-pick the candidate with the highest mean ``value_col`` in that resample
       (same primary rule as the selection engine; ties -> lexicographically
       smaller name, matching the engine's last tie-break);
    3. compute the paired delta against ``competitor_method`` on that resample;
    4. store the delta and which candidate won.

    The resulting interval is wider than a naive bootstrap because it prices in
    the selection step that the point estimate hides.
    """
    methods = list(candidate_methods)
    matrix = best_view[best_view["method_name"].isin(methods + [competitor_method])]
    pivot = matrix.pivot_table(index="dataset_id", columns="method_name",
                               values=value_col, aggfunc="max")
    cols = [m for m in methods if m in pivot.columns]
    if not cols or competitor_method not in pivot.columns:
        return {"label": label, "n_samples": 0, "error":
                "candidate pool or competitor missing from the frame"}
    pivot = pivot.dropna(subset=cols + [competitor_method])
    if pivot.empty:
        return {"label": label, "n_samples": 0, "error": "no complete rows"}

    values = pivot[cols].to_numpy(float)
    competitor = pivot[competitor_method].to_numpy(float)
    n_ds = values.shape[0]
    rng = np.random.default_rng(int(seed))

    deltas = np.empty(int(n_samples), dtype=float)
    winners: List[str] = []
    for i in range(int(n_samples)):
        idx = rng.integers(0, n_ds, size=n_ds)
        means = values[idx].mean(axis=0)
        best = int(np.argmax(means))          # argmax -> first (lowest-index) tie
        winners.append(cols[best])
        deltas[i] = float(values[idx, best].mean() - competitor[idx].mean())

    # Point estimate: the representative selected on the full sample.
    full_means = values.mean(axis=0)
    point_winner = cols[int(np.argmax(full_means))]
    point_delta = float(values[:, int(np.argmax(full_means))].mean()
                        - competitor.mean())
    lo, hi = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    counts = pd.Series(winners).value_counts()
    return {
        "label": label or f"{point_winner}_vs_{competitor_method}",
        "candidate_pool_size": len(cols),
        "candidate_methods": ";".join(cols),
        "competitor_method": competitor_method,
        "competitor_short_label": short_label(competitor_method),
        "n_datasets": int(n_ds),
        "n_samples": int(n_samples),
        "confidence": float(confidence),
        "point_selected_method": point_winner,
        "point_selected_short_label": short_label(point_winner),
        "point_delta": point_delta,
        "bootstrap_mean_delta": float(deltas.mean()),
        "bootstrap_median_delta": float(np.median(deltas)),
        "ci_low": float(np.quantile(deltas, lo)),
        "ci_high": float(np.quantile(deltas, hi)),
        "share_delta_positive": float((deltas > 0).mean()),
        "selection_stability": float(counts.iloc[0] / len(winners)),
        "most_frequent_winner": str(counts.index[0]),
        "most_frequent_winner_short_label": short_label(str(counts.index[0])),
        "n_distinct_winners": int(counts.size),
        "winner_distribution": ";".join(
            f"{k}:{v}" for k, v in counts.items()),
        "claim_level": CLAIM_POST_SELECTION,
        "notes": "interval prices in BOTH dataset resampling and representative "
                 "re-selection; wider than a naive paired bootstrap by design",
    }


def selection_aware_bootstrap_frame(records: Sequence[Dict[str, Any]]
                                    ) -> pd.DataFrame:
    return pd.DataFrame([r for r in records if r]) if records else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Win rates
# --------------------------------------------------------------------------- #
def winrates(dataset_level: pd.DataFrame, *, methods: Sequence[str],
             references: Sequence[str], value_col: str = "best_view_ari",
             group_col: Optional[str] = None,
             tie_tolerance: float = 1e-12) -> pd.DataFrame:
    """Paired win/tie/loss rates for every ``method x reference`` pair."""
    rows: List[Dict[str, Any]] = []
    groups: List[Optional[Tuple[str, Any]]] = [None]
    if group_col and group_col in dataset_level.columns:
        groups = [(group_col, key)
                  for key in sorted(dataset_level[group_col].dropna().unique())]
    for grp in groups:
        for method in methods:
            for ref in references:
                if method == ref:
                    continue
                a, b, _ = paired_values(dataset_level, method, ref,
                                        value_col=value_col, group=grp)
                delta = a - b
                n = int(delta.size)
                row: Dict[str, Any] = {
                    "method_name": method,
                    "method_short_label": short_label(method),
                    "reference_name": ref,
                    "reference_short_label": short_label(ref),
                    "n_paired": n,
                    "win_rate": float((delta > tie_tolerance).mean()) if n else float("nan"),
                    "tie_rate": (float((np.abs(delta) <= tie_tolerance).mean())
                                 if n else float("nan")),
                    "loss_rate": float((delta < -tie_tolerance).mean()) if n else float("nan"),
                    "mean_ari_gain": float(delta.mean()) if n else float("nan"),
                    "median_ari_gain": float(np.median(delta)) if n else float("nan"),
                }
                if grp is not None:
                    row[grp[0]] = grp[1]
                rows.append(row)
    return pd.DataFrame(rows)
