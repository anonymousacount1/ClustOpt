"""Selection stability: how fragile is a retrospective representative?

The preliminary study found selection stability around 0.57-0.62 for the
8-candidate MLP and KNN pools -- i.e. the "best observed" variant changes in
roughly 40% of bootstrap resamples. That is a scientifically important fact about
how much weight a single retrospective representative can carry, so it is
computed and reported rather than left implicit.

The bootstrap **repeats the selection step inside every resample** (amendment
§3.5). Bootstrapping the already-selected representative would answer a different
and much easier question -- "how variable is this method's mean?" -- and would
hide exactly the uncertainty that matters here.

Procedure, per pool:

1. resample datasets with replacement;
2. re-rank the candidates on that resample using the same primary metric;
3. record which candidate won;
4. repeat ``n_samples`` times with a fixed seed.

The resulting distribution gives the selection frequency of each candidate, the
entropy of that distribution (0 = one candidate always wins), and the probability
that the point winner remains the winner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .method_registry import CANDIDATE_POOL_BY_ID, METHOD_BY_NAME, short_label
from .selection import SELECTION_METRIC, Selection

DEFAULT_SAMPLES = 1000
DEFAULT_SEED = 20260725


@dataclass
class PoolStability:
    """Bootstrap selection distribution for one candidate pool."""

    representative_id: str
    candidate_pool_id: str
    candidate_methods: Tuple[str, ...]
    point_selected_method: str
    n_datasets: int
    n_samples: int
    seed: int
    frequencies: Dict[str, float]
    mean_metric_by_method: Dict[str, float]
    mean_delta_to_bootstrap_winner: float
    error: str = ""

    @property
    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self.frequencies.items(), key=lambda kv: (-kv[1], kv[0]))

    @property
    def bootstrap_winner(self) -> Optional[str]:
        return self.ranked[0][0] if self.ranked else None

    @property
    def bootstrap_runner_up(self) -> Optional[str]:
        """The most frequent *alternative* to the point winner.

        Deliberately not "whatever is second in the ranking": when the bootstrap
        majority prefers a different candidate, that candidate is the winner and
        the point selection lands second, so a naive second-place lookup would
        report the point winner as its own runner-up.
        """
        for method, _freq in self.ranked:
            if method != self.point_selected_method:
                return method
        return None

    @property
    def point_winner_frequency(self) -> float:
        return float(self.frequencies.get(self.point_selected_method, 0.0))

    @property
    def runner_up_frequency(self) -> float:
        runner = self.bootstrap_runner_up
        return float(self.frequencies.get(runner, 0.0)) if runner else 0.0

    @property
    def margin_over_runner_up(self) -> float:
        """Point-winner frequency minus its best alternative's. Negative means
        the bootstrap majority actually prefers the alternative."""
        return self.point_winner_frequency - self.runner_up_frequency

    @property
    def entropy(self) -> float:
        """Shannon entropy (bits) of the selected-method distribution.

        0 means one candidate always wins; log2(n_candidates) means the pool is
        indistinguishable under resampling.
        """
        p = np.array([v for v in self.frequencies.values() if v > 0], dtype=float)
        return float(-(p * np.log2(p)).sum()) if p.size else 0.0

    @property
    def max_entropy(self) -> float:
        n = len(self.candidate_methods)
        return float(np.log2(n)) if n > 1 else 0.0

    def n_distinct_bootstrap_winners_check(self) -> int:
        """How many distinct candidates ever won a resample."""
        return int(sum(1 for v in self.frequencies.values() if v > 0))

    def summary_record(self) -> Dict[str, Any]:
        return {
            "representative_id": self.representative_id,
            "candidate_pool_id": self.candidate_pool_id,
            "candidate_pool_size": len(self.candidate_methods),
            "selected_method": self.point_selected_method,
            "selected_short_label": short_label(self.point_selected_method),
            "selection_frequency": self.point_winner_frequency,
            "probability_point_winner_remains_winner": self.point_winner_frequency,
            "bootstrap_winner_method": self.bootstrap_winner,
            "bootstrap_winner_short_label": (short_label(self.bootstrap_winner)
                                             if self.bootstrap_winner else None),
            "point_winner_is_bootstrap_winner": (
                self.bootstrap_winner == self.point_selected_method),
            "runner_up_method": self.bootstrap_runner_up,
            "runner_up_short_label": (short_label(self.bootstrap_runner_up)
                                      if self.bootstrap_runner_up else None),
            "runner_up_frequency": self.runner_up_frequency,
            "margin_over_runner_up": self.margin_over_runner_up,
            "entropy_of_selected_method_distribution": self.entropy,
            "max_possible_entropy": self.max_entropy,
            "normalised_entropy": (self.entropy / self.max_entropy
                                   if self.max_entropy > 0 else 0.0),
            "n_distinct_bootstrap_winners": int(
                sum(1 for v in self.frequencies.values() if v > 0)),
            "mean_delta_to_bootstrap_winner": self.mean_delta_to_bootstrap_winner,
            "selection_stability": self.point_winner_frequency,
            "n_datasets": self.n_datasets,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "selection_metric": SELECTION_METRIC,
            "selection_repeated_inside_each_resample": True,
            "claim_level": "post_selection_exploratory",
            "error": self.error,
        }

    def frequency_records(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for method, freq in self.ranked:
            spec = METHOD_BY_NAME.get(method)
            rows.append({
                "representative_id": self.representative_id,
                "candidate_pool_id": self.candidate_pool_id,
                "method_name": method,
                "short_label": short_label(method),
                "utility_source": spec.utility_source if spec else "",
                "weighting_mode": spec.weighting_mode if spec else "",
                "k_policy": spec.k_policy if spec else "",
                "selection_frequency": freq,
                "is_point_winner": method == self.point_selected_method,
                "mean_best_view_ari_full_sample":
                    self.mean_metric_by_method.get(method, float("nan")),
                "n_samples": self.n_samples,
                "seed": self.seed,
                "claim_level": "post_selection_exploratory",
            })
        return rows


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_pool_stability(
    best_view: pd.DataFrame,
    *,
    representative_id: str,
    candidate_pool_id: str,
    candidate_methods: Optional[Sequence[str]] = None,
    point_selected_method: Optional[str] = None,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    value_col: str = "best_view_ari",
) -> PoolStability:
    """Re-select the representative inside each bootstrap resample."""
    if candidate_methods is None:
        pool = CANDIDATE_POOL_BY_ID.get(candidate_pool_id)
        if pool is None:                                       # pragma: no cover
            raise KeyError(f"unknown candidate pool {candidate_pool_id!r}")
        candidate_methods = pool.methods
    methods = tuple(candidate_methods)

    sub = best_view[best_view["method_name"].isin(methods)]
    pivot = sub.pivot_table(index="dataset_id", columns="method_name",
                            values=value_col, aggfunc="max")
    cols = [m for m in methods if m in pivot.columns]
    pivot = pivot[cols].dropna() if cols else pivot
    if not cols or pivot.empty:
        return PoolStability(
            representative_id=representative_id,
            candidate_pool_id=candidate_pool_id, candidate_methods=methods,
            point_selected_method=point_selected_method or "",
            n_datasets=0, n_samples=0, seed=int(seed), frequencies={},
            mean_metric_by_method={}, mean_delta_to_bootstrap_winner=float("nan"),
            error="no complete rows for this candidate pool")

    values = pivot.to_numpy(float)
    n_ds = values.shape[0]
    full_means = values.mean(axis=0)
    point = point_selected_method or cols[int(np.argmax(full_means))]

    rng = np.random.default_rng(int(seed))
    counts = {m: 0 for m in cols}
    deltas = np.empty(int(n_samples), dtype=float)
    point_idx = cols.index(point) if point in cols else int(np.argmax(full_means))
    for i in range(int(n_samples)):
        idx = rng.integers(0, n_ds, size=n_ds)
        means = values[idx].mean(axis=0)
        # argmax takes the first (lowest-index) maximum, matching the selection
        # engine's lexicographic last tie-break over an ordered candidate list.
        best = int(np.argmax(means))
        counts[cols[best]] += 1
        deltas[i] = float(means[best] - means[point_idx])

    total = float(sum(counts.values())) or 1.0
    return PoolStability(
        representative_id=representative_id,
        candidate_pool_id=candidate_pool_id, candidate_methods=tuple(cols),
        point_selected_method=point, n_datasets=int(n_ds),
        n_samples=int(n_samples), seed=int(seed),
        frequencies={m: c / total for m, c in counts.items()},
        mean_metric_by_method={m: float(full_means[j]) for j, m in enumerate(cols)},
        mean_delta_to_bootstrap_winner=float(deltas.mean()))


def stability_for_selections(
    best_view: pd.DataFrame, selections: Sequence[Selection], *,
    n_samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED,
    min_pool_size: int = 2, verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, PoolStability]]:
    """Run the bootstrap for every global selection with >= 2 candidates.

    Returns ``(summary frame, per-method frequency frame, results by
    representative_id)``.
    """
    results: Dict[str, PoolStability] = {}
    summaries: List[Dict[str, Any]] = []
    frequencies: List[Dict[str, Any]] = []
    for sel in selections:
        if sel.selection_scope != "global":
            continue
        if len(sel.candidate_methods) < int(min_pool_size):
            continue
        result = bootstrap_pool_stability(
            best_view, representative_id=sel.representative_id,
            candidate_pool_id=sel.candidate_pool_id,
            candidate_methods=sel.candidate_methods,
            point_selected_method=sel.selected_method,
            n_samples=n_samples, seed=seed)
        results[sel.representative_id] = result
        summaries.append(result.summary_record())
        frequencies.extend(result.frequency_records())
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values("selection_stability").reset_index(drop=True)
    freq = pd.DataFrame(frequencies)
    if verbose and not summary.empty:
        worst = summary.iloc[0]
        print(f"[stability] {len(summary)} pools bootstrapped "
              f"({n_samples} resamples, seed {seed}); least stable: "
              f"{worst['representative_id']} at "
              f"{worst['selection_stability']:.3f}", flush=True)
    return summary, freq, results


def attach_stability(selections: Sequence[Selection],
                     results: Dict[str, PoolStability]) -> None:
    """Write the bootstrap stability fields back onto the selection records."""
    for sel in selections:
        result = results.get(sel.representative_id)
        if result is None or result.error:
            continue
        sel.bootstrap_selection_frequency = result.point_winner_frequency
        sel.bootstrap_runner_up_method = result.bootstrap_runner_up
        sel.bootstrap_runner_up_frequency = result.runner_up_frequency
        sel.selection_stability = result.point_winner_frequency


def validate_stability(summary: pd.DataFrame) -> List[str]:
    """Sanity checks on the stability output (empty list == valid)."""
    problems: List[str] = []
    if summary.empty:
        problems.append("no stability rows produced")
        return problems
    if not summary["selection_repeated_inside_each_resample"].all():
        problems.append("a stability row does not repeat selection inside the "
                        "resample -- that would measure the wrong thing")
    bad = summary[(summary["selection_frequency"] < 0)
                  | (summary["selection_frequency"] > 1)]
    if len(bad):
        problems.append(f"{len(bad)} selection_frequency values outside [0, 1]")
    bad = summary[summary["entropy_of_selected_method_distribution"]
                  > summary["max_possible_entropy"] + 1e-9]
    if len(bad):
        problems.append(f"{len(bad)} entropy values above the pool maximum")
    if not (summary["claim_level"] == "post_selection_exploratory").all():
        problems.append("stability rows must be labelled "
                        "post_selection_exploratory")
    return problems
