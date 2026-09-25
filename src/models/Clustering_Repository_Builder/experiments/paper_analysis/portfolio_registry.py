"""Frozen Virtual-Best-Solver (VBS) portfolio definitions.

A **portfolio** is an explicit, frozen set of fixed evaluated variants. Its VBS
picks, *per dataset*, whichever member achieved the highest Best-View ARI --
using the dataset's true external ARI. It is therefore a **policy-selection
oracle** and a non-deployable upper bound (see ``oracle_levels.md``).

Naming contract (amendment §3.1)
--------------------------------
A portfolio name must not imply that every member is deployable. The 12-variant
MLP and KNN sets include the ``*_dynamic_realK_*`` variants, which take the
metric **count** from the real utility vector and therefore use information that
is unavailable at test time. Those sets are named ``*_all_variants_vbs``; the
10-variant subsets that exclude them are named ``*_strict_deployable_vbs``.

============================================  ====  ==================================
portfolio                                        n  operational status
============================================  ====  ==================================
``mlp_all_variants_vbs``                        12  oracle-assisted (Dynamic Real-K)
``mlp_strict_deployable_vbs``                   10  fully operational
``knn_all_variants_vbs``                        12  oracle-assisted (Dynamic Real-K)
``knn_strict_deployable_vbs``                   10  fully operational
``learned_source_all_variants_vbs``             24  oracle-assisted (Dynamic Real-K)
``strict_deployable_clustopt_vbs``              20  fully operational
``real_utility_vbs``                            10  utility oracle
``all_variant_clustopt_vbs``                    34  utility oracle + Dynamic Real-K
============================================  ====  ==================================

``strict_deployable_clustopt_vbs`` is the **only** learned-source ClustOpt
portfolio that may be described as fully operational.

Every portfolio declares ``contains_real_utility``, ``contains_oracle_dynamic_k``,
``fully_operational`` and ``oracle_contamination_reason``, so no report can
present an oracle-contaminated envelope as a deployable one.

Deprecated ids
--------------
``mlp_vbs``, ``knn_vbs`` and ``deployable_source_clustopt_vbs`` resolve to their
renamed equivalents through :data:`DEPRECATED_PORTFOLIO_IDS`. They were
misleading precisely because they suggested deployability; the alias exists only
so an older config or manifest still resolves, and :func:`portfolio` records the
substitution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .method_registry import METHOD_BY_NAME, METHODS, names_where, short_label

_TOP_KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class Portfolio:
    portfolio_id: str
    label: str
    short_label: str
    methods: Tuple[str, ...]
    notes: str = ""

    @property
    def size(self) -> int:
        return len(self.methods)

    @property
    def contains_real_utility(self) -> bool:
        return any(METHOD_BY_NAME[m].uses_real_utility for m in self.methods)

    @property
    def contains_oracle_dynamic_k(self) -> bool:
        """True when any member takes its metric COUNT from the real utility."""
        return any(METHOD_BY_NAME[m].is_oracle_assisted for m in self.methods)

    # Retained spelling used by the earlier manifests and tests.
    @property
    def contains_oracle_assisted_dynamic_k(self) -> bool:
        return self.contains_oracle_dynamic_k

    @property
    def fully_operational(self) -> bool:
        """True only when every member is runnable without test-time ground truth."""
        return not (self.contains_real_utility or self.contains_oracle_dynamic_k)

    @property
    def oracle_contamination_reason(self) -> str:
        reasons: List[str] = []
        if self.contains_real_utility:
            reasons.append("real_utility")
        if self.contains_oracle_dynamic_k:
            reasons.append("dynamic_realK")
        return ";".join(reasons)

    @property
    def contaminated_members(self) -> Tuple[str, ...]:
        return tuple(m for m in self.methods
                     if METHOD_BY_NAME[m].uses_real_utility
                     or METHOD_BY_NAME[m].is_oracle_assisted)

    @property
    def frameworks(self) -> Tuple[str, ...]:
        return tuple(sorted({METHOD_BY_NAME[m].framework for m in self.methods}))

    def to_record(self) -> Dict[str, object]:
        return {
            "portfolio_id": self.portfolio_id,
            "portfolio_label": self.label,
            "portfolio_short_label": self.short_label,
            "candidate_methods": ";".join(self.methods),
            "candidate_short_labels": ";".join(short_label(m) for m in self.methods),
            "portfolio_size": self.size,
            "selection_type": "dataset_vbs",
            "selection_metric": "best_view_ari",
            "uses_view_oracle": True,
            "uses_variant_oracle": True,
            "contains_real_utility": self.contains_real_utility,
            "contains_oracle_dynamic_k": self.contains_oracle_dynamic_k,
            # Kept for continuity with the pre-amendment manifests.
            "contains_oracle_assisted_dynamic_k": self.contains_oracle_dynamic_k,
            "fully_operational": self.fully_operational,
            "oracle_contamination_reason": self.oracle_contamination_reason,
            "n_contaminated_members": len(self.contaminated_members),
            "contaminated_members": ";".join(self.contaminated_members),
            "frameworks": ";".join(self.frameworks),
            "paper_claim_level": "upper_bound",
            "notes": self.notes,
        }


def _p(pid: str, label: str, short: str, methods: Sequence[str], notes: str = "") -> Portfolio:
    return Portfolio(pid, label, short, tuple(methods), notes)


# --------------------------------------------------------------------------- #
# Membership helpers -- all derived from the registry, never hand-listed.
# --------------------------------------------------------------------------- #
_MLP_ALL = names_where(method_family="clustopt_mlp")
_KNN_ALL = names_where(method_family="clustopt_knn")
_REAL_ALL = names_where(method_family="clustopt_real")
_FIXED_ALL = names_where(method_family="clustopt_fixed")

_MLP_STRICT = tuple(n for n in _MLP_ALL if not METHOD_BY_NAME[n].is_oracle_assisted)
_KNN_STRICT = tuple(n for n in _KNN_ALL if not METHOD_BY_NAME[n].is_oracle_assisted)


def _weighting_portfolio(weighting: str, *, deployable_only: bool) -> Tuple[str, ...]:
    """Matched raw/softmax membership in a fixed, documented order.

    Order: MLP fixed Top-K (ascending), MLP Dynamic Predict-K, MLP Dynamic
    Real-K, then the same for KNN, then Real fixed Top-K and Real Dynamic.
    ``deployable_only`` drops the real-utility block and the Dynamic Real-K
    (oracle-assisted) members, so the two matched portfolios stay symmetric.
    """
    out: List[str] = []
    for pfx in ("regressor", "knn"):
        out += [f"{pfx}_top{k}_{weighting}" for k in _TOP_KS]
        out.append(f"{pfx}_top_dynamic_predictK_{weighting}")
        if not deployable_only:
            out.append(f"{pfx}_top_dynamic_realK_{weighting}")
    if not deployable_only:
        out += [f"real_top{k}_{weighting}" for k in _TOP_KS]
        out.append(f"real_top_dynamic_{weighting}")
    return tuple(out)


_RAW = "raw"
_SM = "softmax_t05"

RAW_ONLY = _weighting_portfolio(_RAW, deployable_only=False)
SOFTMAX_ONLY = _weighting_portfolio(_SM, deployable_only=False)
DEPLOYABLE_RAW = _weighting_portfolio(_RAW, deployable_only=True)
DEPLOYABLE_SOFTMAX = _weighting_portfolio(_SM, deployable_only=True)

_ORACLE_DYN_NOTE = ("Includes the 2 Dynamic Real-K variants, which take the "
                    "metric COUNT from the real utility vector. NOT fully "
                    "operational -- see the strict counterpart.")


# --------------------------------------------------------------------------- #
# The frozen portfolios
# --------------------------------------------------------------------------- #
PORTFOLIOS: Tuple[Portfolio, ...] = (
    # ---- Comparison 5: framework envelopes ------------------------------- #
    _p("knn_all_variants_vbs", "KNN VBS (all 12 KNN variants)", "KNN All VBS",
       _KNN_ALL, _ORACLE_DYN_NOTE),
    _p("knn_strict_deployable_vbs",
       "KNN VBS (10 fully operational KNN variants)", "KNN Strict VBS",
       _KNN_STRICT,
       "8 fixed Top-K + 2 Dynamic Predict-K. No real utility, no Dynamic "
       "Real-K: every member is runnable without test-time ground truth."),
    _p("mlp_all_variants_vbs", "MLP VBS (all 12 MLP variants)", "MLP All VBS",
       _MLP_ALL, _ORACLE_DYN_NOTE),
    _p("mlp_strict_deployable_vbs",
       "MLP VBS (10 fully operational MLP variants)", "MLP Strict VBS",
       _MLP_STRICT,
       "8 fixed Top-K + 2 Dynamic Predict-K. No real utility, no Dynamic "
       "Real-K: every member is runnable without test-time ground truth."),
    _p("real_utility_vbs", "Real-Utility VBS (all 10 real variants)", "Real VBS",
       _REAL_ALL, "Utility-oracle envelope: not deployable at any portfolio size."),
    _p("learned_source_all_variants_vbs",
       "Learned-Source VBS, all variants (12 MLP + 12 KNN)", "Learned All VBS",
       _MLP_ALL + _KNN_ALL,
       "No real-utility members, but includes the 4 Dynamic Real-K variants. "
       "Deliberately NOT named deployable; use "
       "strict_deployable_clustopt_vbs for the operational envelope."),
    _p("strict_deployable_clustopt_vbs",
       "Strict Deployable ClustOpt VBS (20 fully operational variants)",
       "Strict VBS", _MLP_STRICT + _KNN_STRICT,
       "8 MLP fixed + 2 MLP Dynamic Predict-K + 8 KNN fixed + 2 KNN Dynamic "
       "Predict-K. The ONLY learned-source ClustOpt portfolio that may be "
       "described as fully operational."),
    _p("all_variant_clustopt_vbs",
       "All-Variant ClustOpt VBS (12 MLP + 12 KNN + 10 real)", "All VBS",
       _MLP_ALL + _KNN_ALL + _REAL_ALL,
       "Central policy-selection headroom portfolio (34 variants). Contains "
       "both real utility and Dynamic Real-K."),
    _p("ml2dac_vbs", "ML2DAC VBS (Original + Extended, same search space)",
       "ML2DAC VBS",
       ("ML2DAC_Original_InDomain_same_search_space",
        "ML2DAC_Extended_InDomain_same_search_space"),
       "Matched search space and 50-evaluation budget."),
    _p("autoclust_vbs", "AutoClust VBS (Original + Extended, same search space)",
       "AC VBS",
       ("AutoClust_Original_InDomain_same_search_space",
        "AutoClust_Extended_InDomain_same_search_space"),
       "Matched search space and 50-evaluation budget."),

    # ---- Comparison 3: matched weighting envelopes ----------------------- #
    _p("raw_only_vbs", "Raw-Only VBS (17 raw variants)", "Raw VBS", RAW_ONLY,
       "Primary Comparison-3 portfolio: every raw-weighted variant across all "
       "three utility sources and all five metric-count policies. Contains "
       "real utility and Dynamic Real-K."),
    _p("softmax_only_vbs", "Softmax-Only VBS (17 softmax variants)", "Softmax VBS",
       SOFTMAX_ONLY,
       "Exact softmax counterpart of raw_only_vbs (one-to-one matched members)."),
    _p("raw_softmax_union_vbs", "Raw+Softmax Union VBS (34 variants)", "Union VBS",
       RAW_ONLY + SOFTMAX_ONLY,
       "Union of the two matched weighting portfolios; equals the All-Variant "
       "ClustOpt VBS membership set."),
    _p("deployable_raw_vbs", "Strict Deployable Raw VBS (10 raw variants)",
       "Strict Raw", DEPLOYABLE_RAW,
       "Matched weighting portfolio with real utility and Dynamic Real-K "
       "removed: fully operational."),
    _p("deployable_softmax_vbs",
       "Strict Deployable Softmax VBS (10 softmax variants)", "Strict Sm",
       DEPLOYABLE_SOFTMAX,
       "Exact softmax counterpart of deployable_raw_vbs: fully operational."),
    _p("deployable_raw_softmax_union_vbs",
       "Strict Deployable Raw+Softmax Union VBS (20 variants)", "Strict Union",
       DEPLOYABLE_RAW + DEPLOYABLE_SOFTMAX,
       "Union of the two matched strict-deployable weighting portfolios. Equals "
       "strict_deployable_clustopt_vbs membership."),

    # ---- Supplementary --------------------------------------------------- #
    _p("fixed_baseline_vbs", "Fixed/Uniform Baseline VBS (6 baselines)",
       "Baseline VBS", _FIXED_ALL,
       "Envelope of the non-learned objectives; Comparison-1 reference."),
)

PORTFOLIO_BY_ID: Dict[str, Portfolio] = {p.portfolio_id: p for p in PORTFOLIOS}
PORTFOLIO_IDS: Tuple[str, ...] = tuple(p.portfolio_id for p in PORTFOLIOS)

# Pre-amendment ids kept resolvable. They were renamed because each implied that
# every member is deployable while all three include Dynamic Real-K.
DEPRECATED_PORTFOLIO_IDS: Dict[str, str] = {
    "mlp_vbs": "mlp_all_variants_vbs",
    "knn_vbs": "knn_all_variants_vbs",
    "deployable_source_clustopt_vbs": "learned_source_all_variants_vbs",
}

# Portfolios reported by Comparison Suite 5 (§12), in presentation order.
COMPARISON_05_PORTFOLIOS: Tuple[str, ...] = (
    "knn_all_variants_vbs", "knn_strict_deployable_vbs",
    "mlp_all_variants_vbs", "mlp_strict_deployable_vbs",
    "real_utility_vbs", "learned_source_all_variants_vbs",
    "strict_deployable_clustopt_vbs", "all_variant_clustopt_vbs",
    "ml2dac_vbs", "autoclust_vbs", "fixed_baseline_vbs",
)
# Portfolios reported by Comparison Suite 3 (§10), in presentation order.
COMPARISON_03_PORTFOLIOS: Tuple[str, ...] = (
    "raw_only_vbs", "softmax_only_vbs", "raw_softmax_union_vbs",
    "deployable_raw_vbs", "deployable_softmax_vbs",
    "deployable_raw_softmax_union_vbs",
)

# Portfolios that may be described as fully operational in prose.
FULLY_OPERATIONAL_PORTFOLIOS: Tuple[str, ...] = tuple(
    p.portfolio_id for p in PORTFOLIOS if p.fully_operational)


def portfolio(portfolio_id: str) -> Portfolio:
    """Registry lookup; resolves deprecated ids and raises on unknown ones."""
    pid = str(portfolio_id)
    if pid in PORTFOLIO_BY_ID:
        return PORTFOLIO_BY_ID[pid]
    if pid in DEPRECATED_PORTFOLIO_IDS:
        return PORTFOLIO_BY_ID[DEPRECATED_PORTFOLIO_IDS[pid]]
    raise KeyError(f"unknown portfolio {portfolio_id!r}; registered: "
                   f"{sorted(PORTFOLIO_BY_ID)}; deprecated aliases: "
                   f"{sorted(DEPRECATED_PORTFOLIO_IDS)}")


def canonical_portfolio_id(portfolio_id: str) -> str:
    return DEPRECATED_PORTFOLIO_IDS.get(str(portfolio_id), str(portfolio_id))


def manifest_records() -> List[Dict[str, object]]:
    records = [p.to_record() for p in PORTFOLIOS]
    for old, new in sorted(DEPRECATED_PORTFOLIO_IDS.items()):
        target = PORTFOLIO_BY_ID[new]
        records.append({
            **target.to_record(),
            "portfolio_id": old,
            "portfolio_label": f"DEPRECATED alias of {new}",
            "portfolio_short_label": target.short_label,
            "notes": (f"Deprecated id retained so older configs resolve. It was "
                      f"renamed to {new} because the old name implied every "
                      f"member is deployable, while it includes Dynamic Real-K."),
        })
    return records


EXPECTED_SIZES: Dict[str, int] = {
    "knn_all_variants_vbs": 12,
    "knn_strict_deployable_vbs": 10,
    "mlp_all_variants_vbs": 12,
    "mlp_strict_deployable_vbs": 10,
    "real_utility_vbs": 10,
    "learned_source_all_variants_vbs": 24,
    "strict_deployable_clustopt_vbs": 20,
    "all_variant_clustopt_vbs": 34,
    "ml2dac_vbs": 2,
    "autoclust_vbs": 2,
    "raw_only_vbs": 17,
    "softmax_only_vbs": 17,
    "raw_softmax_union_vbs": 34,
    "deployable_raw_vbs": 10,
    "deployable_softmax_vbs": 10,
    "deployable_raw_softmax_union_vbs": 20,
    "fixed_baseline_vbs": 6,
}

# Portfolios asserted to be fully operational: no real utility, no Dynamic Real-K.
STRICT_PORTFOLIOS: Tuple[str, ...] = (
    "mlp_strict_deployable_vbs", "knn_strict_deployable_vbs",
    "strict_deployable_clustopt_vbs", "deployable_raw_vbs",
    "deployable_softmax_vbs", "deployable_raw_softmax_union_vbs",
    "ml2dac_vbs", "autoclust_vbs", "fixed_baseline_vbs",
)
# Portfolios asserted to declare Dynamic Real-K contamination.
ORACLE_DYNAMIC_K_PORTFOLIOS: Tuple[str, ...] = (
    "mlp_all_variants_vbs", "knn_all_variants_vbs",
    "learned_source_all_variants_vbs", "all_variant_clustopt_vbs",
    "raw_only_vbs", "softmax_only_vbs", "raw_softmax_union_vbs",
)
# Portfolios asserted to contain real-utility (utility-oracle) members.
REAL_UTILITY_PORTFOLIOS: Tuple[str, ...] = (
    "real_utility_vbs", "all_variant_clustopt_vbs", "raw_only_vbs",
    "softmax_only_vbs", "raw_softmax_union_vbs",
)

SUBSET_INVARIANTS: Tuple[Tuple[str, str], ...] = (
    ("mlp_strict_deployable_vbs", "mlp_all_variants_vbs"),
    ("knn_strict_deployable_vbs", "knn_all_variants_vbs"),
    ("mlp_all_variants_vbs", "learned_source_all_variants_vbs"),
    ("knn_all_variants_vbs", "learned_source_all_variants_vbs"),
    ("strict_deployable_clustopt_vbs", "learned_source_all_variants_vbs"),
    ("mlp_strict_deployable_vbs", "strict_deployable_clustopt_vbs"),
    ("knn_strict_deployable_vbs", "strict_deployable_clustopt_vbs"),
    ("learned_source_all_variants_vbs", "all_variant_clustopt_vbs"),
    ("real_utility_vbs", "all_variant_clustopt_vbs"),
    ("raw_only_vbs", "raw_softmax_union_vbs"),
    ("softmax_only_vbs", "raw_softmax_union_vbs"),
    ("deployable_raw_vbs", "raw_only_vbs"),
    ("deployable_softmax_vbs", "softmax_only_vbs"),
    ("deployable_raw_softmax_union_vbs", "raw_softmax_union_vbs"),
)


def validate_portfolios() -> List[str]:
    """Return a list of portfolio problems (empty == valid)."""
    problems: List[str] = []
    ids = [p.portfolio_id for p in PORTFOLIOS]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate portfolio_id: {dupes}")
    labels = [p.short_label for p in PORTFOLIOS]
    dupes = sorted({l for l in labels if labels.count(l) > 1})
    if dupes:
        problems.append(f"duplicate portfolio short_label: {dupes}")

    for p in PORTFOLIOS:
        unknown = [m for m in p.methods if m not in METHOD_BY_NAME]
        if unknown:
            problems.append(f"{p.portfolio_id}: unknown methods {unknown}")
        if len(set(p.methods)) != len(p.methods):
            dup = sorted({m for m in p.methods if list(p.methods).count(m) > 1})
            problems.append(f"{p.portfolio_id}: duplicate members {dup}")
        if p.size < 2:
            problems.append(f"{p.portfolio_id}: needs >= 2 members (has {p.size})")
        want = EXPECTED_SIZES.get(p.portfolio_id)
        if want is not None and p.size != want:
            problems.append(f"{p.portfolio_id}: expected {want} members, "
                            f"found {p.size}")
        if len(p.short_label) > 16:
            problems.append(f"{p.portfolio_id}: short_label too long "
                            f"({p.short_label!r})")
        if "VBS" not in p.short_label and "Strict" not in p.short_label:
            problems.append(f"{p.portfolio_id}: short_label {p.short_label!r} "
                            f"does not identify itself as a VBS")

    # A name must never imply deployability that the membership does not have.
    for pid in STRICT_PORTFOLIOS:
        pf = PORTFOLIO_BY_ID[pid]
        if not pf.fully_operational:
            problems.append(f"{pid}: named strict/deployable but contains "
                            f"{pf.oracle_contamination_reason} members "
                            f"({list(pf.contaminated_members)})")
    for pid in ORACLE_DYNAMIC_K_PORTFOLIOS:
        pf = PORTFOLIO_BY_ID[pid]
        if not pf.contains_oracle_dynamic_k:
            problems.append(f"{pid}: expected to contain Dynamic Real-K members")
        if pf.fully_operational:
            problems.append(f"{pid}: declares fully_operational despite "
                            f"Dynamic Real-K membership")
    for pid in REAL_UTILITY_PORTFOLIOS:
        if not PORTFOLIO_BY_ID[pid].contains_real_utility:
            problems.append(f"{pid}: expected to contain real-utility members")
    for pid, pf in PORTFOLIO_BY_ID.items():
        if pf.fully_operational and pf.oracle_contamination_reason:
            problems.append(f"{pid}: fully_operational but reports "
                            f"contamination {pf.oracle_contamination_reason!r}")
        if not pf.fully_operational and not pf.oracle_contamination_reason:
            problems.append(f"{pid}: not fully_operational but reports no "
                            f"contamination reason")
        if "deployable" in pid and not pf.fully_operational:
            problems.append(f"{pid}: id contains 'deployable' but the portfolio "
                            f"is not fully operational")

    # Matched weighting portfolios must be one-to-one.
    for raw_id, sm_id in (("raw_only_vbs", "softmax_only_vbs"),
                          ("deployable_raw_vbs", "deployable_softmax_vbs")):
        raw_p, sm_p = PORTFOLIO_BY_ID[raw_id], PORTFOLIO_BY_ID[sm_id]
        if raw_p.size != sm_p.size:
            problems.append(f"{raw_id}/{sm_id}: sizes differ "
                            f"({raw_p.size} vs {sm_p.size})")
        for a, b in zip(raw_p.methods, sm_p.methods):
            sa, sb = METHOD_BY_NAME[a], METHOD_BY_NAME[b]
            if (sa.utility_source, sa.k_policy, sa.fixed_metric_count,
                    sa.dynamic_k_source) != (sb.utility_source, sb.k_policy,
                                             sb.fixed_metric_count, sb.dynamic_k_source):
                problems.append(f"{raw_id}/{sm_id}: unmatched pair {a} <-> {b}")
            if sa.weighting_mode != "raw" or sb.weighting_mode != "softmax_t05":
                problems.append(f"{raw_id}/{sm_id}: wrong weighting for {a}/{b}")

    # Membership equalities the reports rely on.
    for a, b in (("raw_softmax_union_vbs", "all_variant_clustopt_vbs"),
                 ("deployable_raw_softmax_union_vbs",
                  "strict_deployable_clustopt_vbs")):
        if set(PORTFOLIO_BY_ID[a].methods) != set(PORTFOLIO_BY_ID[b].methods):
            problems.append(f"{a} != {b} membership set")

    for sub, sup in SUBSET_INVARIANTS:
        if not set(PORTFOLIO_BY_ID[sub].methods) <= set(PORTFOLIO_BY_ID[sup].methods):
            problems.append(f"{sub} is not a subset of {sup}")

    for old, new in DEPRECATED_PORTFOLIO_IDS.items():
        if old in PORTFOLIO_BY_ID:
            problems.append(f"deprecated id {old} must not also be a live "
                            f"portfolio id")
        if new not in PORTFOLIO_BY_ID:
            problems.append(f"deprecated id {old} points at unknown portfolio "
                            f"{new}")
    return problems
