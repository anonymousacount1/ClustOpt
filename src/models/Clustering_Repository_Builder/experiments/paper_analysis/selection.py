"""Retrospective representative-selection engine.

What this does
--------------
Given a **frozen candidate pool** of fixed evaluated variants and the Split-1
best-view frame, pick exactly one candidate -- the one with the highest global
(or per-family) Mean Best-View ARI -- and record *everything* about how it was
picked.

What this is NOT
----------------
Because every downstream clustering experiment was run on Split 1 only, there is
no independent split on which a policy could have been chosen. A representative
selected here is therefore:

* a **retrospective global representative** -- "the best observed single variant
  on Split 1 within the evaluated candidate pool"; or
* a **retrospective family representative** -- a family-level policy oracle.

It is never a "preselected deployable winner", an "independently selected
deployment policy" or a "held-out selected configuration". Every emitted record
carries ``is_independently_preselected = False`` and
``paper_claim_level = retrospective_summary`` so no downstream artifact can lose
that qualification. See ``terminology.md``.

Tie-break chain (§10.2, deterministic, fully logged)
---------------------------------------------------
1. higher ``best_view_mean_ari``
2. higher ``best_view_median_ari``
3. lower ``best_view_std_ari``
4. higher ``best_view_k_accuracy``
5. lower ``total_runtime_all_views_median_sec``
6. lexicographically smaller canonical method name

Steps 2-6 only ever run when the preceding key is equal within
``ari_tie_tolerance`` (default ``1e-6``). ``practical_ari_tolerance`` is a
*separate, optional* report-only rule: it records which candidates are
practically equivalent to the winner and (when enabled) which candidate the
practical rule would prefer -- it never overrides the raw ARI winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .frames import summarize_methods
from .method_registry import CANDIDATE_POOL_BY_ID, METHOD_BY_NAME, short_label

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SELECTION_METRIC = "best_view_mean_ari"

TIE_BREAK_RULES: Tuple[Tuple[str, bool], ...] = (
    ("best_view_mean_ari", False),                    # (column, ascending)
    ("best_view_median_ari", False),
    ("best_view_std_ari", True),
    ("best_view_k_accuracy", False),
    ("total_runtime_all_views_median_sec", True),
    ("method_name", True),
)
TIE_BREAK_DESCRIPTION = " > ".join(
    f"{col}{'^' if not asc else 'v'}" for col, asc in TIE_BREAK_RULES)

DEFAULT_ARI_TIE_TOLERANCE = 1e-6
DEFAULT_PRACTICAL_ARI_TOLERANCE = 0.001


@dataclass
class SelectionPolicy:
    """Selection rule parameters, frozen into every selection record."""

    selection_metric: str = SELECTION_METRIC
    ari_tie_tolerance: float = DEFAULT_ARI_TIE_TOLERANCE
    practical_ari_tolerance: float = DEFAULT_PRACTICAL_ARI_TOLERANCE
    report_practical_equivalence: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_metric": self.selection_metric,
            "ari_tie_tolerance": self.ari_tie_tolerance,
            "practical_ari_tolerance": self.practical_ari_tolerance,
            "report_practical_equivalence": self.report_practical_equivalence,
            "tie_break_rules": TIE_BREAK_DESCRIPTION,
        }


@dataclass
class Selection:
    """One frozen retrospective selection."""

    comparison_id: str
    representative_id: str
    candidate_pool_id: str
    candidate_methods: Tuple[str, ...]
    selection_scope: str                     # 'global' | 'family'
    scope_key: Optional[str]                  # family name for family scope
    selection_split: int
    selected_method: str
    stats: Dict[str, Any] = field(default_factory=dict)
    tie_breaks_invoked: Tuple[str, ...] = ()
    n_tied_on_primary: int = 1
    practical_equivalents: Tuple[str, ...] = ()
    practical_equivalence_representative: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    # Runner-up: how close the decision actually was. A representative that beat
    # its runner-up by 0.0004 is a very different claim from one that beat it by
    # 0.04, and a table showing only the winner hides that entirely.
    runner_up_method: Optional[str] = None
    runner_up_stats: Dict[str, Any] = field(default_factory=dict)
    # Filled in by :func:`attach_stability` from the selection-aware bootstrap.
    bootstrap_selection_frequency: Optional[float] = None
    bootstrap_runner_up_method: Optional[str] = None
    bootstrap_runner_up_frequency: Optional[float] = None
    selection_stability: Optional[float] = None

    @property
    def selection_type(self) -> str:
        return ("retrospective_global" if self.selection_scope == "global"
                else "retrospective_family")

    @property
    def tie_break_reason(self) -> str:
        if self.n_tied_on_primary <= 1:
            return "unique winner on the primary metric"
        return ("tie on best_view_mean_ari resolved by "
                + " then ".join(self.tie_breaks_invoked))

    @property
    def delta_to_runner_up(self) -> float:
        own = self.stats.get("best_view_mean_ari")
        other = self.runner_up_stats.get("best_view_mean_ari")
        if own is None or other is None:
            return float("nan")
        try:
            return float(own) - float(other)
        except (TypeError, ValueError):                        # pragma: no cover
            return float("nan")

    def to_record(self) -> Dict[str, Any]:
        spec = METHOD_BY_NAME.get(self.selected_method)
        runner_spec = METHOD_BY_NAME.get(self.runner_up_method or "")
        return {
            "comparison_id": self.comparison_id,
            "representative_id": self.representative_id,
            "candidate_pool_id": self.candidate_pool_id,
            "candidate_methods": ";".join(self.candidate_methods),
            "candidate_short_labels": ";".join(
                short_label(m) for m in self.candidate_methods),
            "candidate_pool_size": len(self.candidate_methods),
            "selection_scope": self.selection_scope,
            "family": self.scope_key,
            "selection_type": self.selection_type,
            "selection_split": self.selection_split,
            "selection_metric": self.policy.get("selection_metric", SELECTION_METRIC),
            "tie_break_rules": TIE_BREAK_DESCRIPTION,
            "tie_breaks_invoked": ";".join(self.tie_breaks_invoked),
            "n_tied_on_primary": self.n_tied_on_primary,
            "selected_method": self.selected_method,
            # Alias: the amendment's reporting contract names this field
            # selected_method_name, and the concrete method must always be
            # visible beside the representative id.
            "selected_method_name": self.selected_method,
            "selected_method_folder": spec.on_disk_name if spec else self.selected_method,
            "display_name": spec.display_name if spec else self.selected_method,
            "short_label": spec.short_label if spec else self.selected_method,
            "selected_short_label": spec.short_label if spec else self.selected_method,
            "selected_paper_label": spec.paper_label if spec else "",
            "selected_utility_source": spec.utility_source if spec else "",
            "selected_k_policy": spec.k_policy if spec else "",
            "selected_weighting_mode": spec.weighting_mode if spec else "",
            "selected_oracle_level": spec.oracle_level if spec else "",
            "selected_best_view_mean_ari": self.stats.get("best_view_mean_ari"),
            "selected_best_view_median_ari": self.stats.get("best_view_median_ari"),
            "selected_best_view_std_ari": self.stats.get("best_view_std_ari"),
            "selected_best_view_q25_ari": self.stats.get("best_view_q25_ari"),
            "selected_best_view_q75_ari": self.stats.get("best_view_q75_ari"),
            "selected_k_accuracy": self.stats.get("best_view_k_accuracy"),
            "selected_total_runtime_all_views_mean_sec":
                self.stats.get("total_runtime_all_views_mean_sec"),
            "selected_total_runtime_all_views_median_sec":
                self.stats.get("total_runtime_all_views_median_sec"),
            "selected_median_total_all_views_runtime_sec":
                self.stats.get("total_runtime_all_views_median_sec"),
            # ---- runner-up: how close the decision was --------------------- #
            "runner_up_method": self.runner_up_method,
            "runner_up_short_label": (runner_spec.short_label if runner_spec
                                      else self.runner_up_method),
            "runner_up_mean_best_view_ari":
                self.runner_up_stats.get("best_view_mean_ari"),
            "delta_to_runner_up": self.delta_to_runner_up,
            "tie_break_reason": self.tie_break_reason,
            # ---- selection stability (from the selection-aware bootstrap) --- #
            "bootstrap_selection_frequency": self.bootstrap_selection_frequency,
            "bootstrap_runner_up_method": self.bootstrap_runner_up_method,
            "bootstrap_runner_up_frequency": self.bootstrap_runner_up_frequency,
            "selection_stability": self.selection_stability,
            "selected_n_datasets": self.stats.get("n_datasets"),
            "selected_n_success": self.stats.get("n_success"),
            "selected_success_rate": self.stats.get("success_rate"),
            "raw_metric_winner": self.selected_method,
            "practical_equivalence_representative":
                self.practical_equivalence_representative,
            "n_practical_equivalents": len(self.practical_equivalents),
            "practical_equivalents": ";".join(self.practical_equivalents),
            "uses_view_oracle": True,
            "uses_variant_oracle": True,
            "is_operational_fixed_method": True,
            "is_independently_preselected": False,
            "paper_claim_level": "retrospective_summary",
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Core selection
# --------------------------------------------------------------------------- #
def _candidate_summary(best_view: pd.DataFrame, methods: Sequence[str],
                       scope_col: Optional[str] = None,
                       scope_key: Optional[str] = None) -> pd.DataFrame:
    """Primary-metric summary for one candidate pool, optionally scoped."""
    sub = best_view[best_view["method_name"].isin(list(methods))]
    if scope_col is not None:
        sub = sub[sub[scope_col] == scope_key]
    if sub.empty:
        return pd.DataFrame()
    return summarize_methods(sub, ("method_name",))


def _rank_candidates(summary: pd.DataFrame, policy: SelectionPolicy
                     ) -> Tuple[pd.DataFrame, List[str], int]:
    """Order candidates by the tie-break chain.

    Returns ``(ordered summary, tie-break columns actually invoked, n tied on the
    primary metric)``. A later key is only consulted when every earlier key is
    equal within the tolerance, and only those keys are reported as invoked.
    """
    work = summary.copy()
    for col, _asc in TIE_BREAK_RULES:
        if col not in work.columns:
            work[col] = np.nan
    # NaNs must never win a comparison: push them to the losing end of each key.
    fill = {
        "best_view_mean_ari": -np.inf, "best_view_median_ari": -np.inf,
        "best_view_std_ari": np.inf, "best_view_k_accuracy": -np.inf,
        "total_runtime_all_views_median_sec": np.inf,
    }
    for col, sentinel in fill.items():
        work[f"_{col}"] = pd.to_numeric(work[col], errors="coerce").fillna(sentinel)
    # Quantise the primary metric so equality within tolerance really ties.
    tol = float(policy.ari_tie_tolerance)
    work["_primary_bucket"] = np.round(work["_best_view_mean_ari"] / tol) if tol > 0 \
        else work["_best_view_mean_ari"]
    work["_median_bucket"] = np.round(work["_best_view_median_ari"] / tol) if tol > 0 \
        else work["_best_view_median_ari"]

    sort_cols = ["_primary_bucket", "_median_bucket", "_best_view_std_ari",
                 "_best_view_k_accuracy", "_total_runtime_all_views_median_sec",
                 "method_name"]
    ascending = [False, False, True, False, True, True]
    ordered = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    top = ordered.iloc[0]
    n_tied = int((ordered["_primary_bucket"] == top["_primary_bucket"]).sum())
    invoked: List[str] = []
    if n_tied > 1:
        tied = ordered[ordered["_primary_bucket"] == top["_primary_bucket"]]
        for col, key in (("best_view_median_ari", "_median_bucket"),
                         ("best_view_std_ari", "_best_view_std_ari"),
                         ("best_view_k_accuracy", "_best_view_k_accuracy"),
                         ("total_runtime_all_views_median_sec",
                          "_total_runtime_all_views_median_sec")):
            invoked.append(col)
            if tied[key].nunique() > 1:
                break
            tied = tied[tied[key] == tied.iloc[0][key]]
        else:
            invoked.append("method_name")
        if len(tied) > 1 and invoked[-1] != "method_name":
            invoked.append("method_name")
    return ordered.drop(columns=[c for c in ordered.columns if c.startswith("_")]), \
        invoked, n_tied


def select_representative(
    best_view: pd.DataFrame,
    *,
    comparison_id: str,
    representative_id: str,
    candidate_pool_id: str,
    candidate_methods: Optional[Sequence[str]] = None,
    split_id: int,
    scope: str = "global",
    scope_key: Optional[str] = None,
    scope_column: str = "family",
    policy: Optional[SelectionPolicy] = None,
    notes: str = "",
) -> Tuple[Optional[Selection], pd.DataFrame]:
    """Select one representative from a candidate pool.

    Returns ``(selection, candidate_table)``. ``selection`` is ``None`` when the
    pool has no usable rows in scope -- callers must handle that rather than
    receive a silently-fabricated winner. ``candidate_table`` is the full ranked
    candidate summary, always returned so the report can show the runners-up.
    """
    policy = policy or SelectionPolicy()
    if candidate_methods is None:
        pool = CANDIDATE_POOL_BY_ID.get(candidate_pool_id)
        if pool is None:                                       # pragma: no cover
            raise KeyError(f"unknown candidate pool {candidate_pool_id!r}")
        candidate_methods = pool.methods
    methods = tuple(candidate_methods)

    summary = _candidate_summary(
        best_view, methods,
        scope_col=(None if scope == "global" else scope_column),
        scope_key=(None if scope == "global" else scope_key))
    if summary.empty:
        return None, summary

    ordered, invoked, n_tied = _rank_candidates(summary, policy)
    ordered = ordered.reset_index(drop=True)
    ordered.insert(0, "selection_rank", range(1, len(ordered) + 1))
    winner = ordered.iloc[0]
    winner_ari = float(pd.to_numeric(pd.Series([winner[SELECTION_METRIC]]),
                                     errors="coerce").iloc[0])

    # Practical-equivalence bookkeeping: report-only, never overriding.
    practical: Tuple[str, ...] = ()
    practical_rep: Optional[str] = None
    if policy.report_practical_equivalence and np.isfinite(winner_ari):
        ari = pd.to_numeric(ordered[SELECTION_METRIC], errors="coerce")
        close = ordered[(winner_ari - ari) <= float(policy.practical_ari_tolerance)]
        practical = tuple(str(m) for m in close["method_name"])
        if len(close) > 1:
            # Within the practically-equivalent set prefer the cheapest run;
            # recorded separately as `practical_equivalence_representative`.
            rt = pd.to_numeric(close["total_runtime_all_views_median_sec"],
                               errors="coerce").fillna(np.inf)
            practical_rep = str(close.assign(_rt=rt).sort_values(
                ["_rt", "method_name"], kind="mergesort").iloc[0]["method_name"])
        else:
            practical_rep = str(winner["method_name"])

    _STAT_KEYS = (
        "best_view_mean_ari", "best_view_median_ari", "best_view_std_ari",
        "best_view_q25_ari", "best_view_q75_ari", "best_view_k_accuracy",
        "total_runtime_all_views_mean_sec", "total_runtime_all_views_median_sec",
        "n_datasets", "n_success", "success_rate")
    runner = ordered.iloc[1] if len(ordered) > 1 else None

    selection = Selection(
        comparison_id=comparison_id,
        representative_id=representative_id,
        candidate_pool_id=candidate_pool_id,
        candidate_methods=methods,
        selection_scope=scope,
        scope_key=(None if scope == "global" else scope_key),
        selection_split=int(split_id),
        selected_method=str(winner["method_name"]),
        runner_up_method=(str(runner["method_name"]) if runner is not None else None),
        runner_up_stats=({k: runner.get(k) for k in _STAT_KEYS}
                         if runner is not None else {}),
        stats={k: winner.get(k) for k in (
            "best_view_mean_ari", "best_view_median_ari", "best_view_std_ari",
            "best_view_q25_ari", "best_view_q75_ari", "best_view_k_accuracy",
            "total_runtime_all_views_mean_sec",
            "total_runtime_all_views_median_sec",
            "n_datasets", "n_success", "success_rate")},
        tie_breaks_invoked=tuple(invoked),
        n_tied_on_primary=n_tied,
        practical_equivalents=practical,
        practical_equivalence_representative=practical_rep,
        policy=policy.to_dict(),
        notes=notes,
    )
    ordered.insert(0, "representative_id", representative_id)
    ordered.insert(0, "comparison_id", comparison_id)
    ordered["candidate_pool_id"] = candidate_pool_id
    ordered["selection_scope"] = scope
    ordered["scope_key"] = scope_key
    ordered["is_selected"] = ordered["method_name"] == selection.selected_method
    return selection, ordered


def select_family_representatives(
    best_view: pd.DataFrame,
    *,
    comparison_id: str,
    representative_id: str,
    candidate_pool_id: str,
    candidate_methods: Optional[Sequence[str]] = None,
    split_id: int,
    scope_column: str = "family",
    policy: Optional[SelectionPolicy] = None,
    notes: str = "",
) -> Tuple[List[Selection], pd.DataFrame]:
    """Run :func:`select_representative` once inside every family.

    The result is a **family-level retrospective policy oracle**, not a global
    operational policy: it is allowed to pick a different variant per family and
    is reported only as family-level headroom.
    """
    families = sorted(best_view[scope_column].dropna().unique())
    selections: List[Selection] = []
    tables: List[pd.DataFrame] = []
    for fam in families:
        sel, table = select_representative(
            best_view, comparison_id=comparison_id,
            representative_id=f"{representative_id}__{_slug(fam)}",
            candidate_pool_id=candidate_pool_id,
            candidate_methods=candidate_methods, split_id=split_id,
            scope="family", scope_key=fam, scope_column=scope_column,
            policy=policy,
            notes=notes or "family-level retrospective policy oracle")
        if sel is not None:
            sel = Selection(**{**sel.__dict__, "representative_id": representative_id})
            selections.append(sel)
        if not table.empty:
            table = table.copy()
            table["representative_id"] = representative_id
            table[scope_column] = fam
            tables.append(table)
    combined = (pd.concat(tables, ignore_index=True) if tables
                else pd.DataFrame())
    return selections, combined


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in str(text)]
    return "".join(keep).strip("_").replace("__", "_")


# --------------------------------------------------------------------------- #
# Manifest assembly
# --------------------------------------------------------------------------- #
def selection_manifest(selections: Sequence[Selection]) -> pd.DataFrame:
    """The frozen ``selection_manifest`` table every later suite must read."""
    if not selections:
        return pd.DataFrame()
    return pd.DataFrame([s.to_record() for s in selections])


def observed_representative_map(selections: Sequence[Selection]) -> pd.DataFrame:
    """Compact ``representative_id -> concrete method`` side table.

    Exists so a reader can always answer "which concrete variant is behind this
    label?" without re-deriving anything.
    """
    if not selections:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for s in selections:
        spec = METHOD_BY_NAME.get(s.selected_method)
        rows.append({
            "representative_id": s.representative_id,
            "selection_scope": s.selection_scope,
            "family": s.scope_key or "",
            "selection_type": s.selection_type,
            "candidate_pool_id": s.candidate_pool_id,
            "candidate_pool_size": len(s.candidate_methods),
            "selected_method": s.selected_method,
            "selected_short_label": spec.short_label if spec else s.selected_method,
            "selected_full_name": spec.display_name if spec else "",
            "selected_paper_label": spec.paper_label if spec else "",
            "selection_metric": s.policy.get("selection_metric", SELECTION_METRIC),
            "selection_split": s.selection_split,
            "selected_best_view_mean_ari": s.stats.get("best_view_mean_ari"),
            "n_tied_on_primary": s.n_tied_on_primary,
            "tie_breaks_invoked": ";".join(s.tie_breaks_invoked),
            "is_independently_preselected": False,
            "paper_claim_level": "retrospective_summary",
        })
    return pd.DataFrame(rows)


def family_divergence(global_selections: Sequence[Selection],
                      family_selections: Sequence[Selection]) -> pd.DataFrame:
    """How often the family-level choice differs from the global one.

    One row per ``(representative_id, family)``, plus the per-representative
    divergence rate -- the headroom that a hypothetical per-family policy would
    unlock, and therefore *not* evidence about any single deployable method.
    """
    global_by_id = {s.representative_id: s for s in global_selections}
    rows: List[Dict[str, Any]] = []
    for s in family_selections:
        g = global_by_id.get(s.representative_id)
        rows.append({
            "representative_id": s.representative_id,
            "family": s.scope_key,
            "family_selected_method": s.selected_method,
            "family_selected_short_label": short_label(s.selected_method),
            "family_best_view_mean_ari": s.stats.get("best_view_mean_ari"),
            "global_selected_method": g.selected_method if g else None,
            "global_selected_short_label": short_label(g.selected_method) if g else None,
            "differs_from_global": (bool(g is not None
                                         and s.selected_method != g.selected_method)),
            "n_tied_on_primary": s.n_tied_on_primary,
            "tie_breaks_invoked": ";".join(s.tie_breaks_invoked),
            "selection_type": "retrospective_family",
            "paper_claim_level": "diagnostic_oracle",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    rates = (out.groupby("representative_id")["differs_from_global"]
             .agg(n_families="size", n_differing="sum")
             .assign(divergence_rate=lambda d: d["n_differing"] / d["n_families"])
             .reset_index())
    return out.merge(rates, on="representative_id", how="left")


def apply_global_representatives_by_family(
    best_view: pd.DataFrame,
    global_selections: Sequence[Selection],
    *,
    scope_column: str = "family",
) -> pd.DataFrame:
    """Per-family performance of the **globally fixed** representatives.

    This is the primary family-level view (§13.2 RQ2-C): one fixed method per
    representative, reported per family. Contrast with
    :func:`select_family_representatives`, which re-picks per family and is
    headroom only.
    """
    rows: List[pd.DataFrame] = []
    for s in global_selections:
        sub = best_view[best_view["method_name"] == s.selected_method]
        if sub.empty:
            continue
        summ = summarize_methods(sub, (scope_column,))
        summ["representative_id"] = s.representative_id
        summ["selected_method"] = s.selected_method
        summ["selected_short_label"] = short_label(s.selected_method)
        summ["selection_type"] = "retrospective_global"
        summ["selection_scope"] = "global"
        summ["paper_claim_level"] = "retrospective_summary"
        rows.append(summ)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())


# --------------------------------------------------------------------------- #
# Determinism check
# --------------------------------------------------------------------------- #
def selection_is_deterministic(
    best_view: pd.DataFrame, *, comparison_id: str, representative_id: str,
    candidate_pool_id: str, split_id: int, n_repeats: int = 3,
    candidate_methods: Optional[Sequence[str]] = None,
    policy: Optional[SelectionPolicy] = None,
) -> bool:
    """Re-run the same selection on shuffled input; the winner must not change."""
    winners: set = set()
    rng = np.random.default_rng(20260725)
    for _ in range(int(n_repeats)):
        shuffled = best_view.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        sel, _ = select_representative(
            shuffled, comparison_id=comparison_id,
            representative_id=representative_id,
            candidate_pool_id=candidate_pool_id,
            candidate_methods=candidate_methods,
            split_id=split_id, policy=policy)
        winners.add(sel.selected_method if sel else None)
    return len(winners) == 1
