"""Rich declarative method registry for the paper-oriented analysis.

Every method that appears in any comparison suite is described here **once**, as
data. Nothing downstream (plotting, reporting, selection, packaging) may
re-derive a method's family, oracle level, weighting mode or label by parsing its
name -- it reads these fields instead.

On-disk layout (verified against split 1, 2026-07)
--------------------------------------------------
All 51 method folders live under a *single* split directory::

    <dataset_dir>/experiments/split_01/<on_disk_name>/<record_dir>/

with ``record_dir`` in ``x_only|y_only|xy_2d`` for the ClustOpt schema and
``1d_x|1d_y|2d`` for the external schema. The Phase-D and Phase-E2 merges moved
everything into ``split_01``, so no method needs a split-dir suffix any more
(``search_suffix`` is kept for future isolated runs).

Method inventory
----------------
==================================  ====  ====================================
group                                  n  note
==================================  ====  ====================================
MLP / regressor (``method_family``)   12  4 fixed Top-K x {raw, softmax} +
                                          dynamic {predictK, realK} x {raw, sm}
KNN                                   12  identical structure, KNN utility
Real utility                          10  4 fixed Top-K x {raw, softmax} +
                                          dynamic x {raw, softmax}
Fixed / uniform baselines              6  4 single CVIs + 2 uniform ensembles
External, same search space            4  {AutoClust, ML2DAC} x {Orig, Ext}
External, own search space (suppl.)    4  same four, unconstrained grid
External, original authors' code       3  AutoClust / AutoML4Clust / ML2DAC
==================================  ====  ====================================

The first five rows (34 ClustOpt variants + 6 baselines + 4 externals) are the
paper methods. The last two rows are registered so the frames stay complete and
auditable, but they are **not** members of any primary candidate pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Canonical view ids and on-disk record folders
# --------------------------------------------------------------------------- #
VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")

CLUSTOPT_VIEW_DIRS: Dict[str, str] = {v: v for v in VIEW_IDS}
EXTERNAL_VIEW_DIRS: Dict[str, str] = {
    "x_only": "1d_x", "y_only": "1d_y", "xy_2d": "2d",
}

# --------------------------------------------------------------------------- #
# Controlled vocabularies. Anything not in these tuples is a registry bug and is
# rejected by ``validate_registry`` / the registry unit tests.
# --------------------------------------------------------------------------- #
FRAMEWORKS: Tuple[str, ...] = ("clustopt", "autoclust", "ml2dac", "automl4clust")

UTILITY_SOURCES: Tuple[str, ...] = (
    "mlp", "knn", "real", "uniform", "single_cvi", "external_internal",
)

K_POLICIES: Tuple[str, ...] = (
    "fixed", "dynamic_predict", "dynamic_real", "not_applicable",
)

WEIGHTING_MODES: Tuple[str, ...] = (
    "raw", "softmax_t05", "uniform", "single", "external_internal",
)

METRIC_INVENTORIES: Tuple[str, ...] = (
    "all_metrics", "classic_only", "single_metric", "external_internal",
)

ORACLE_LEVELS: Tuple[str, ...] = (
    "none", "utility_oracle", "dynamic_k_oracle_assisted", "external_ari_vbs",
)

METHOD_FAMILIES: Tuple[str, ...] = (
    "clustopt_mlp", "clustopt_knn", "clustopt_real", "clustopt_fixed",
    "external_same_search_space", "external_own_search_space",
    "external_original_code", "external_metric_decomposition",
)

SCHEMAS: Tuple[str, ...] = ("clustopt", "external")

# The full metric inventory size (see metric_registry.METRIC_COUNT) -- recorded
# on the spec so reports can state the inventory without importing the metrics.
ALL_METRICS_COUNT = 60
CLASSIC_CVI_BASELINE_METRICS: Tuple[str, ...] = (
    "silhouette", "calinski_harabasz", "davies_bouldin", "dbcv",
)

DEFAULT_SEARCH_BUDGET = 50          # Optuna trials (ClustOpt) / evaluations (in-domain externals)


# --------------------------------------------------------------------------- #
# MethodSpec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MethodSpec:
    """Everything the analysis knows about one evaluated method."""

    # ---- identity ------------------------------------------------------- #
    method_name: str                 # canonical analysis id (== on_disk_name here)
    on_disk_name: str                # experiments/<split>/<on_disk_name>/
    display_name: str                # long human-readable name
    short_label: str                 # <= ~14 chars, for plot axes; unique
    paper_label: str                 # prose label for tables/captions
    schema: str                      # 'clustopt' | 'external'

    # ---- taxonomy ------------------------------------------------------- #
    framework: str
    method_family: str
    utility_source: str
    metric_inventory: str
    k_policy: str
    weighting_mode: str

    # ---- numeric / optional descriptors --------------------------------- #
    fixed_metric_count: Optional[int] = None
    dynamic_k_source: str = "none"          # 'none' | 'predicted' | 'real'
    softmax_temperature: Optional[float] = None
    metric_inventory_size: Optional[int] = None

    # ---- boolean character flags ---------------------------------------- #
    is_fixed_single_cvi: bool = False
    is_uniform_ensemble: bool = False
    is_meta_learning: bool = False
    uses_real_utility: bool = False
    uses_real_dynamic_k: bool = False
    is_oracle_assisted: bool = False
    is_utility_oracle: bool = False
    is_external: bool = False
    same_search_space: bool = True
    is_paper_method: bool = True            # member of the primary comparison set

    # ---- budget / disk --------------------------------------------------- #
    search_budget: Optional[int] = DEFAULT_SEARCH_BUDGET
    search_suffix: str = ""                 # split-dir suffix, '' == split_XX

    # ---- capability declarations (verified against disk by capability.py) - #
    trace_best_capable: bool = True
    metric_selection_capable: bool = True
    runtime_components_capable: bool = True

    # ---- grouping -------------------------------------------------------- #
    candidate_groups: Tuple[str, ...] = ()
    portfolio_groups: Tuple[str, ...] = ()
    notes: str = ""

    # ------------------------------------------------------------------ API
    @property
    def view_dirs(self) -> Dict[str, str]:
        return EXTERNAL_VIEW_DIRS if self.schema == "external" else CLUSTOPT_VIEW_DIRS

    @property
    def oracle_level(self) -> str:
        """Single canonical oracle label (see ``00_protocol/oracle_levels.md``).

        Note this covers only the *method's own* oracle usage. Best-View
        selection (a view oracle) applies to every method uniformly and is
        carried by the frames as ``uses_view_oracle``; dataset-level VBS
        (``external_ari_vbs``) is a property of a portfolio, not a method.
        """
        if self.is_utility_oracle:
            return "utility_oracle"
        if self.is_oracle_assisted:
            return "dynamic_k_oracle_assisted"
        return "none"

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["oracle_level"] = self.oracle_level
        d["candidate_groups"] = list(self.candidate_groups)
        d["portfolio_groups"] = list(self.portfolio_groups)
        return d


# --------------------------------------------------------------------------- #
# Builders. One factory per structural block keeps the registry declarative and
# makes the "12 / 12 / 10 / 6 / 4" counts obvious and testable.
# --------------------------------------------------------------------------- #
_WEIGHT_SUFFIX = {"raw": "raw", "softmax_t05": "softmax_t05"}
_WEIGHT_SHORT = {"raw": "R", "softmax_t05": "S"}
_WEIGHT_PROSE = {"raw": "raw normalized weights", "softmax_t05": "softmax(T=0.5) weights"}

_SOURCE_PREFIX = {"mlp": "regressor", "knn": "knn", "real": "real"}
_SOURCE_SHORT = {"mlp": "MLP", "knn": "KNN", "real": "Real"}
_SOURCE_PROSE = {
    "mlp": "learned MLP utility",
    "knn": "learned KNN utility",
    "real": "real (oracle) utility",
}
_SOURCE_FAMILY = {
    "mlp": "clustopt_mlp", "knn": "clustopt_knn", "real": "clustopt_real",
}


def _learned_fixed(source: str, top_k: int, weighting: str) -> MethodSpec:
    """One fixed-Top-K utility-driven ClustOpt variant."""
    name = f"{_SOURCE_PREFIX[source]}_top{top_k}_{_WEIGHT_SUFFIX[weighting]}"
    is_real = source == "real"
    return MethodSpec(
        method_name=name, on_disk_name=name,
        display_name=f"{_SOURCE_SHORT[source]} utility, Top-{top_k}, "
                     f"{'raw' if weighting == 'raw' else 'softmax T=0.5'}",
        short_label=f"{_SOURCE_SHORT[source]} T{top_k}-{_WEIGHT_SHORT[weighting]}",
        paper_label=f"{_SOURCE_PROSE[source]}, Top-{top_k}, {_WEIGHT_PROSE[weighting]}",
        schema="clustopt", framework="clustopt",
        method_family=_SOURCE_FAMILY[source], utility_source=source,
        metric_inventory="all_metrics", k_policy="fixed", weighting_mode=weighting,
        fixed_metric_count=top_k, dynamic_k_source="none",
        softmax_temperature=0.5 if weighting == "softmax_t05" else None,
        metric_inventory_size=ALL_METRICS_COUNT,
        is_meta_learning=not is_real, uses_real_utility=is_real,
        is_utility_oracle=is_real,
        candidate_groups=(f"{source}_fixed", f"{source}_all", "clustopt_all"),
        portfolio_groups=tuple(
            [f"{source}_vbs", "all_variant_clustopt_vbs",
             f"{'raw' if weighting == 'raw' else 'softmax'}_only_vbs"]
            + ([] if is_real else ["deployable_source_clustopt_vbs",
                                   f"deployable_{'raw' if weighting == 'raw' else 'softmax'}_vbs"])
        ),
    )


def _learned_dynamic(source: str, k_src: str, weighting: str) -> MethodSpec:
    """One dynamic-metric-count utility-driven ClustOpt variant.

    ``k_src`` is ``'predicted'`` (the metric count is derived from the *predicted*
    utility vector -- fully operational) or ``'real'`` (the count is derived from
    the *real* utility vector while the ranking stays predicted -- oracle
    assisted, diagnostic only). Real-utility methods have a single dynamic form
    whose count comes from the real vector they already use.
    """
    is_real = source == "real"
    if is_real:
        name = f"real_top_dynamic_{_WEIGHT_SUFFIX[weighting]}"
        short_k, prose_k, policy = "Dyn", "dynamic metric count", "dynamic_real"
        oracle_assisted = False
    else:
        tag = "predictK" if k_src == "predicted" else "realK"
        name = f"{_SOURCE_PREFIX[source]}_top_dynamic_{tag}_{_WEIGHT_SUFFIX[weighting]}"
        short_k = "Dyn" if k_src == "predicted" else "DynRK"
        prose_k = ("dynamic metric count from the predicted utility vector"
                   if k_src == "predicted" else
                   "dynamic metric count from the REAL utility vector "
                   "(oracle-assisted)")
        policy = "dynamic_predict" if k_src == "predicted" else "dynamic_real"
        oracle_assisted = k_src == "real"

    weight_group = "raw" if weighting == "raw" else "softmax"
    return MethodSpec(
        method_name=name, on_disk_name=name,
        display_name=f"{_SOURCE_SHORT[source]} utility, Dynamic "
                     f"{'Predict-K' if k_src == 'predicted' else 'Real-K'}, "
                     f"{'raw' if weighting == 'raw' else 'softmax T=0.5'}",
        short_label=f"{_SOURCE_SHORT[source]} {short_k}-{_WEIGHT_SHORT[weighting]}",
        paper_label=f"{_SOURCE_PROSE[source]}, {prose_k}, {_WEIGHT_PROSE[weighting]}",
        schema="clustopt", framework="clustopt",
        method_family=_SOURCE_FAMILY[source], utility_source=source,
        metric_inventory="all_metrics", k_policy=policy, weighting_mode=weighting,
        fixed_metric_count=None, dynamic_k_source=k_src,
        softmax_temperature=0.5 if weighting == "softmax_t05" else None,
        metric_inventory_size=ALL_METRICS_COUNT,
        is_meta_learning=not is_real,
        uses_real_utility=is_real, uses_real_dynamic_k=(is_real or oracle_assisted),
        is_oracle_assisted=oracle_assisted, is_utility_oracle=is_real,
        candidate_groups=tuple(
            [f"{source}_all", "clustopt_all"]
            + ([f"{source}_dynamic_predict"] if k_src == "predicted" and not is_real else [])
            + ([f"{source}_dynamic_real"] if k_src == "real" and not is_real else [])
            + (["real_dynamic"] if is_real else [])
        ),
        portfolio_groups=tuple(
            [f"{source}_vbs", "all_variant_clustopt_vbs", f"{weight_group}_only_vbs"]
            + ([] if is_real else ["deployable_source_clustopt_vbs"])
            # dynamic_realK is oracle-assisted -> excluded from the matched
            # *deployable* weighting portfolios but kept in the all-variant ones.
            + ([f"deployable_{weight_group}_vbs"]
               if (not is_real and k_src == "predicted") else [])
        ),
        notes=("oracle-assisted: metric count taken from the real utility vector"
               if oracle_assisted else ""),
    )


def _single_cvi(name: str, short: str, display: str) -> MethodSpec:
    return MethodSpec(
        method_name=name, on_disk_name=name,
        display_name=f"{display} (single CVI objective)",
        short_label=short, paper_label=f"single fixed CVI: {display}",
        schema="clustopt", framework="clustopt", method_family="clustopt_fixed",
        utility_source="single_cvi", metric_inventory="single_metric",
        k_policy="not_applicable", weighting_mode="single",
        fixed_metric_count=1, metric_inventory_size=1,
        is_fixed_single_cvi=True,
        candidate_groups=("fixed_baselines", "single_cvi_baselines"),
        portfolio_groups=("fixed_baseline_vbs",),
    )


def _uniform(name: str, short: str, display: str, inventory: str,
             size: int, note: str) -> MethodSpec:
    return MethodSpec(
        method_name=name, on_disk_name=name, display_name=display,
        short_label=short, paper_label=display,
        schema="clustopt", framework="clustopt", method_family="clustopt_fixed",
        utility_source="uniform", metric_inventory=inventory,
        k_policy="not_applicable", weighting_mode="uniform",
        metric_inventory_size=size, is_uniform_ensemble=True,
        candidate_groups=("fixed_baselines", "uniform_baselines"),
        portfolio_groups=("fixed_baseline_vbs",), notes=note,
    )


def _external(on_disk: str, framework: str, short: str, display: str,
              paper: str, *, family: str, same_ss: bool, paper_method: bool,
              budget: Optional[int], portfolios: Sequence[str] = (),
              note: str = "") -> MethodSpec:
    return MethodSpec(
        method_name=on_disk, on_disk_name=on_disk, display_name=display,
        short_label=short, paper_label=paper, schema="external",
        framework=framework, method_family=family,
        utility_source="external_internal", metric_inventory="external_internal",
        k_policy="not_applicable", weighting_mode="external_internal",
        is_external=True, same_search_space=same_ss,
        is_paper_method=paper_method, search_budget=budget,
        # Externals expose no ClustOpt candidate trace and no metric-selection
        # record; the capability preflight re-verifies this against disk.
        trace_best_capable=False, metric_selection_capable=False,
        runtime_components_capable=False,
        candidate_groups=("external_same_search_space",) if same_ss and paper_method else (),
        portfolio_groups=tuple(portfolios), notes=note,
    )


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
_TOP_KS: Tuple[int, ...] = (1, 3, 5, 10)
_WEIGHTINGS: Tuple[str, ...] = ("raw", "softmax_t05")

_MLP_METHODS: Tuple[MethodSpec, ...] = (
    *(_learned_fixed("mlp", k, w) for k in _TOP_KS for w in _WEIGHTINGS),
    *(_learned_dynamic("mlp", ks, w) for ks in ("predicted", "real") for w in _WEIGHTINGS),
)

_KNN_METHODS: Tuple[MethodSpec, ...] = (
    *(_learned_fixed("knn", k, w) for k in _TOP_KS for w in _WEIGHTINGS),
    *(_learned_dynamic("knn", ks, w) for ks in ("predicted", "real") for w in _WEIGHTINGS),
)

_REAL_METHODS: Tuple[MethodSpec, ...] = (
    *(_learned_fixed("real", k, w) for k in _TOP_KS for w in _WEIGHTINGS),
    *(_learned_dynamic("real", "real", w) for w in _WEIGHTINGS),
)

_FIXED_METHODS: Tuple[MethodSpec, ...] = (
    _uniform("all_metrics_uniform", "All Uniform",
             "Uniform ensemble over the full metric inventory",
             "all_metrics", ALL_METRICS_COUNT,
             f"uniform weights over all {ALL_METRICS_COUNT} utility metrics"),
    _uniform("uniform_classic_cvi", "Classic Uniform",
             "Uniform ensemble over the classic CVIs",
             "classic_only", len(CLASSIC_CVI_BASELINE_METRICS),
             "uniform weights over "
             + ", ".join(CLASSIC_CVI_BASELINE_METRICS)),
    _single_cvi("silhouette_single", "Silhouette", "Silhouette"),
    _single_cvi("dbcv_single", "DBCV", "DBCV"),
    _single_cvi("davies_bouldin_single", "DB", "Davies-Bouldin"),
    _single_cvi("calinski_harabasz_single", "CH", "Calinski-Harabasz"),
)

# Paper externals: same clustering search space + same 50-evaluation budget as
# the ClustOpt variants, meta-learning/CVI machinery unchanged.
_EXTERNAL_SAME_SS: Tuple[MethodSpec, ...] = (
    _external("AutoClust_Original_InDomain_same_search_space", "autoclust",
              "AC-Orig", "AutoClust (Original MKR) - same search space",
              "AutoClust, original metric repository, ClustOpt search space",
              family="external_same_search_space", same_ss=True, paper_method=True,
              budget=DEFAULT_SEARCH_BUDGET,
              portfolios=("autoclust_vbs",)),
    _external("AutoClust_Extended_InDomain_same_search_space", "autoclust",
              "AC-Ext", "AutoClust (Extended MKR) - same search space",
              "AutoClust, extended metric repository, ClustOpt search space",
              family="external_same_search_space", same_ss=True, paper_method=True,
              budget=DEFAULT_SEARCH_BUDGET,
              portfolios=("autoclust_vbs",)),
    _external("ML2DAC_Original_InDomain_same_search_space", "ml2dac",
              "ML2DAC-Orig", "ML2DAC (Original MKR) - same search space",
              "ML2DAC, original metric repository, ClustOpt search space",
              family="external_same_search_space", same_ss=True, paper_method=True,
              budget=DEFAULT_SEARCH_BUDGET,
              portfolios=("ml2dac_vbs",)),
    _external("ML2DAC_Extended_InDomain_same_search_space", "ml2dac",
              "ML2DAC-Ext", "ML2DAC (Extended MKR) - same search space",
              "ML2DAC, extended metric repository, ClustOpt search space",
              family="external_same_search_space", same_ss=True, paper_method=True,
              budget=DEFAULT_SEARCH_BUDGET,
              portfolios=("ml2dac_vbs",)),
)

# Supplementary externals, registered for completeness / auditability only.
# ``is_paper_method=False`` keeps them out of every primary candidate pool.
_EXTERNAL_SUPPLEMENTARY: Tuple[MethodSpec, ...] = (
    _external("AutoClust_Original_InDomain", "autoclust", "AC-Orig~",
              "AutoClust (Original MKR) - own search space",
              "AutoClust, original repository, unconstrained search space",
              family="external_own_search_space", same_ss=False, paper_method=False,
              budget=DEFAULT_SEARCH_BUDGET,
              note="supplementary: search space NOT matched to ClustOpt"),
    _external("AutoClust_Extended_InDomain", "autoclust", "AC-Ext~",
              "AutoClust (Extended MKR) - own search space",
              "AutoClust, extended repository, unconstrained search space",
              family="external_own_search_space", same_ss=False, paper_method=False,
              budget=DEFAULT_SEARCH_BUDGET,
              note="supplementary: search space NOT matched to ClustOpt"),
    _external("ML2DAC_Original_InDomain", "ml2dac", "ML2DAC-Orig~",
              "ML2DAC (Original MKR) - own search space",
              "ML2DAC, original repository, unconstrained search space",
              family="external_own_search_space", same_ss=False, paper_method=False,
              budget=DEFAULT_SEARCH_BUDGET,
              note="supplementary: search space NOT matched to ClustOpt"),
    _external("ML2DAC_Extended_InDomain", "ml2dac", "ML2DAC-Ext~",
              "ML2DAC (Extended MKR) - own search space",
              "ML2DAC, extended repository, unconstrained search space",
              family="external_own_search_space", same_ss=False, paper_method=False,
              budget=DEFAULT_SEARCH_BUDGET,
              note="supplementary: search space NOT matched to ClustOpt"),
    _external("AutoClust_phase_B", "autoclust", "AC-Paper",
              "AutoClust (original authors' code, Phase B)",
              "AutoClust as published (authors' implementation)",
              family="external_original_code", same_ss=False, paper_method=False,
              budget=None,
              note="supplementary: original authors' code; budget/search space "
                   "not matched; retained from the Phase-B comparison"),
    _external("ML2DAC_phase_B", "ml2dac", "ML2DAC-Paper",
              "ML2DAC (original authors' code, Phase B)",
              "ML2DAC as published (authors' implementation)",
              family="external_original_code", same_ss=False, paper_method=False,
              budget=None,
              note="supplementary: original authors' code; budget/search space "
                   "not matched; retained from the Phase-B comparison"),
    _external("AutoML4Clust_phase_B", "automl4clust", "AML4C",
              "AutoML4Clust (original authors' code, Phase B)",
              "AutoML4Clust as published (authors' implementation)",
              family="external_original_code", same_ss=False, paper_method=False,
              budget=None,
              note="supplementary: original authors' code; budget/search space "
                   "not matched; retained from the Phase-B comparison"),
)

# Stage-3A1 metric-inventory decomposition arms. Same InDomain implementation,
# same ClustOpt matched search space, same 50-evaluation budget and same seed as
# the four ``_EXTERNAL_SAME_SS`` arms -- the ONLY difference is which CVIs the
# derived repository was built from.
#
# They carry their own ``method_family`` and ``is_paper_method=False`` on
# purpose. Putting them in ``external_same_search_space`` would enlarge the
# paper's primary external comparator family (and break its ``EXPECTED_COUNTS``
# contract), and adding them to ``autoclust_vbs`` / ``ml2dac_vbs`` would silently
# change those portfolios' membership -- i.e. it would alter already-frozen
# results. ``same_search_space=True`` still records the scientific fact, and the
# collector/frames/reports pick them up exactly like every other registered
# method, so the standard aggregation covers them with no bespoke pipeline.
_EXTERNAL_METRIC_DECOMPOSITION: Tuple[MethodSpec, ...] = (
    _external("AutoClust_OriginalPlusEstablished_InDomain_same_search_space",
              "autoclust", "AC-Orig+Est",
              "AutoClust (Original + Established MKR) - same search space",
              "AutoClust, original plus established metric additions, "
              "ClustOpt search space",
              family="external_metric_decomposition", same_ss=True,
              paper_method=False, budget=DEFAULT_SEARCH_BUDGET,
              note="Stage-3A1 decomposition arm A1: 17 requested / 17 live CVIs"),
    _external("AutoClust_OriginalPlusNew46_InDomain_same_search_space",
              "autoclust", "AC-Orig+New46",
              "AutoClust (Original + New46 MKR) - same search space",
              "AutoClust, original plus the 46 newly designed metrics, "
              "ClustOpt search space",
              family="external_metric_decomposition", same_ss=True,
              paper_method=False, budget=DEFAULT_SEARCH_BUDGET,
              note="Stage-3A1 decomposition arm A2: 53 requested / 51 live CVIs "
                   "(2 of the 46 new metrics are all-NaN on the training "
                   "repository and are deterministically excluded)"),
    _external("ML2DAC_OriginalPlusEstablished_InDomain_same_search_space",
              "ml2dac", "ML2DAC-Orig+Est",
              "ML2DAC (Original + Established MKR) - same search space",
              "ML2DAC, original plus established metric additions, "
              "ClustOpt search space",
              family="external_metric_decomposition", same_ss=True,
              paper_method=False, budget=DEFAULT_SEARCH_BUDGET,
              note="Stage-3A1 decomposition arm A1: 17 requested / 17 live CVIs"),
    _external("ML2DAC_OriginalPlusNew46_InDomain_same_search_space",
              "ml2dac", "ML2DAC-O+New46",
              "ML2DAC (Original + New46 MKR) - same search space",
              "ML2DAC, original plus the 46 newly designed metrics, "
              "ClustOpt search space",
              family="external_metric_decomposition", same_ss=True,
              paper_method=False, budget=DEFAULT_SEARCH_BUDGET,
              note="Stage-3A1 decomposition arm A2: 53 requested / 51 live CVIs "
                   "(2 of the 46 new metrics are all-NaN on the training "
                   "repository and are deterministically excluded)"),
)

METHODS: Tuple[MethodSpec, ...] = (
    *_MLP_METHODS, *_KNN_METHODS, *_REAL_METHODS, *_FIXED_METHODS,
    *_EXTERNAL_SAME_SS, *_EXTERNAL_SUPPLEMENTARY,
    *_EXTERNAL_METRIC_DECOMPOSITION,
)

METHOD_BY_NAME: Dict[str, MethodSpec] = {m.method_name: m for m in METHODS}
METHOD_NAMES: Tuple[str, ...] = tuple(m.method_name for m in METHODS)
PAPER_METHOD_NAMES: Tuple[str, ...] = tuple(
    m.method_name for m in METHODS if m.is_paper_method)


def spec(method_name: str) -> MethodSpec:
    """Registry lookup; raises ``KeyError`` with a helpful message."""
    try:
        return METHOD_BY_NAME[method_name]
    except KeyError as exc:                                   # pragma: no cover
        raise KeyError(
            f"unknown method {method_name!r}; registered: "
            f"{sorted(METHOD_BY_NAME)}") from exc


def select_methods(names: Optional[Sequence[str]] = None) -> Tuple[MethodSpec, ...]:
    """Registry specs for ``names`` (registry order preserved); ``None`` = all."""
    if names is None:
        return METHODS
    wanted = set(names)
    unknown = wanted - set(METHOD_BY_NAME)
    if unknown:                                               # pragma: no cover
        raise KeyError(f"unknown method names: {sorted(unknown)}")
    return tuple(m for m in METHODS if m.method_name in wanted)


def names_where(**criteria) -> Tuple[str, ...]:
    """Method names whose spec matches every ``field=value`` in ``criteria``.

    Values may be a scalar or a collection (membership test), e.g.::

        names_where(utility_source="mlp", k_policy=("fixed",))
    """
    out: List[str] = []
    for m in METHODS:
        ok = True
        for key, want in criteria.items():
            got = getattr(m, key)
            if isinstance(want, (list, tuple, set, frozenset)):
                ok = got in want
            else:
                ok = got == want
            if not ok:
                break
        if ok:
            out.append(m.method_name)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Structural family groups (the 12 evaluated ClustOpt "structural families" of
# §11.2: utility source x metric-count policy). Each group holds the matched
# raw/softmax pair from which Comparison 2 picks a representative.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VariantFamily:
    """One (utility source x metric-count policy) cell with a raw/softmax pair."""

    family_id: str
    utility_source: str
    policy_id: str            # 'top1' | 'top3' | 'top5' | 'top10' | 'dynamic'
    policy_label: str
    raw_method: str
    softmax_method: str
    is_primary: bool          # part of the primary 15-cell meta-learning grid
    oracle_assisted: bool = False

    @property
    def members(self) -> Tuple[str, str]:
        return (self.raw_method, self.softmax_method)


def _variant_families() -> Tuple[VariantFamily, ...]:
    out: List[VariantFamily] = []
    for source in ("real", "mlp", "knn"):
        pfx = _SOURCE_PREFIX[source]
        for k in _TOP_KS:
            out.append(VariantFamily(
                family_id=f"{source}_top{k}", utility_source=source,
                policy_id=f"top{k}", policy_label=f"Top-{k}",
                raw_method=f"{pfx}_top{k}_raw",
                softmax_method=f"{pfx}_top{k}_softmax_t05",
                is_primary=True))
        if source == "real":
            out.append(VariantFamily(
                family_id="real_dynamic", utility_source="real",
                policy_id="dynamic", policy_label="Dynamic",
                raw_method="real_top_dynamic_raw",
                softmax_method="real_top_dynamic_softmax_t05",
                is_primary=True))
        else:
            out.append(VariantFamily(
                family_id=f"{source}_dynamic_predictK", utility_source=source,
                policy_id="dynamic", policy_label="Dynamic Predict-K",
                raw_method=f"{pfx}_top_dynamic_predictK_raw",
                softmax_method=f"{pfx}_top_dynamic_predictK_softmax_t05",
                is_primary=True))
            out.append(VariantFamily(
                family_id=f"{source}_dynamic_realK", utility_source=source,
                policy_id="dynamic_realK", policy_label="Dynamic Real-K",
                raw_method=f"{pfx}_top_dynamic_realK_raw",
                softmax_method=f"{pfx}_top_dynamic_realK_softmax_t05",
                is_primary=False, oracle_assisted=True))
    return tuple(out)


VARIANT_FAMILIES: Tuple[VariantFamily, ...] = _variant_families()
VARIANT_FAMILY_BY_ID: Dict[str, VariantFamily] = {
    f.family_id: f for f in VARIANT_FAMILIES}
PRIMARY_VARIANT_FAMILIES: Tuple[VariantFamily, ...] = tuple(
    f for f in VARIANT_FAMILIES if f.is_primary)
DIAGNOSTIC_VARIANT_FAMILIES: Tuple[VariantFamily, ...] = tuple(
    f for f in VARIANT_FAMILIES if not f.is_primary)


# --------------------------------------------------------------------------- #
# Candidate pools (frozen inputs to the retrospective selection engine)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandidatePool:
    pool_id: str
    label: str
    methods: Tuple[str, ...]
    description: str = ""

    @property
    def size(self) -> int:
        return len(self.methods)


def _pool(pool_id: str, label: str, methods: Sequence[str], desc: str = "") -> CandidatePool:
    return CandidatePool(pool_id, label, tuple(methods), desc)


_MLP_FIXED = names_where(utility_source="mlp", k_policy="fixed")
_KNN_FIXED = names_where(utility_source="knn", k_policy="fixed")
_REAL_FIXED_RAW = tuple(f"real_top{k}_raw" for k in _TOP_KS)

CANDIDATE_POOLS: Tuple[CandidatePool, ...] = (
    _pool("real_fixed_topk_raw", "Real-utility fixed Top-K (raw)", _REAL_FIXED_RAW,
          "Comparison 1 helper: best observed fixed Real Top-K raw variant. "
          "real_top_dynamic_raw is deliberately NOT a member."),
    _pool("mlp_fixed_all", "ClustOpt-MLP fixed Top-K (raw + softmax)", _MLP_FIXED,
          "Comparison 4 primary: best observed FIXED MLP variant (8 candidates)."),
    _pool("knn_fixed_all", "ClustOpt-KNN fixed Top-K (raw + softmax)", _KNN_FIXED,
          "Comparison 4 primary: best observed FIXED KNN variant (8 candidates)."),
    _pool("mlp_fixed_plus_dynamic_predict",
          "ClustOpt-MLP fixed Top-K + Dynamic Predict-K",
          _MLP_FIXED + names_where(utility_source="mlp", k_policy="dynamic_predict"),
          "Comparison 4 supplemental: adds the operational Dynamic Predict-K "
          "variants (10 candidates); does not replace mlp_fixed_all."),
    _pool("knn_fixed_plus_dynamic_predict",
          "ClustOpt-KNN fixed Top-K + Dynamic Predict-K",
          _KNN_FIXED + names_where(utility_source="knn", k_policy="dynamic_predict"),
          "Comparison 4 supplemental: adds the operational Dynamic Predict-K "
          "variants (10 candidates); does not replace knn_fixed_all."),
    *(_pool(f"variant_family__{f.family_id}",
            f"{_SOURCE_SHORT[f.utility_source]} x {f.policy_label} (raw vs softmax)",
            f.members,
            "Comparison 2 cell: pick raw or softmax by global Mean Best-View ARI."
            + (" Oracle-assisted diagnostic cell." if f.oracle_assisted else ""))
      for f in VARIANT_FAMILIES),
)

CANDIDATE_POOL_BY_ID: Dict[str, CandidatePool] = {
    p.pool_id: p for p in CANDIDATE_POOLS}


# --------------------------------------------------------------------------- #
# Label maps
# --------------------------------------------------------------------------- #
def label_map_records() -> List[Dict[str, object]]:
    """Rows for ``method_label_map.csv`` / ``.json``."""
    return [
        {
            "short_label": m.short_label,
            "paper_label": m.paper_label,
            "method_full_name": m.display_name,
            "method_name": m.method_name,
            "on_disk_name": m.on_disk_name,
            "framework": m.framework,
            "method_family": m.method_family,
            "utility_source": m.utility_source,
            "k_policy": m.k_policy,
            "weighting_mode": m.weighting_mode,
            "metric_inventory": m.metric_inventory,
            "oracle_level": m.oracle_level,
            "same_search_space": m.same_search_space,
            "search_budget": m.search_budget,
            "is_paper_method": m.is_paper_method,
            "notes": m.notes,
        }
        for m in METHODS
    ]


def short_label(method_name: str) -> str:
    m = METHOD_BY_NAME.get(method_name)
    return m.short_label if m else str(method_name)


def short_labels(method_names: Sequence[str]) -> List[str]:
    return [short_label(n) for n in method_names]


def registry_snapshot() -> Dict[str, object]:
    """Serialisable snapshot written to ``method_registry_snapshot.json``."""
    return {
        "n_methods": len(METHODS),
        "n_paper_methods": len(PAPER_METHOD_NAMES),
        "view_ids": list(VIEW_IDS),
        "controlled_vocabularies": {
            "frameworks": list(FRAMEWORKS),
            "utility_sources": list(UTILITY_SOURCES),
            "k_policies": list(K_POLICIES),
            "weighting_modes": list(WEIGHTING_MODES),
            "metric_inventories": list(METRIC_INVENTORIES),
            "oracle_levels": list(ORACLE_LEVELS),
            "method_families": list(METHOD_FAMILIES),
        },
        "methods": [m.to_dict() for m in METHODS],
        "variant_families": [
            {
                "family_id": f.family_id, "utility_source": f.utility_source,
                "policy_id": f.policy_id, "policy_label": f.policy_label,
                "raw_method": f.raw_method, "softmax_method": f.softmax_method,
                "is_primary": f.is_primary, "oracle_assisted": f.oracle_assisted,
            }
            for f in VARIANT_FAMILIES
        ],
        "candidate_pools": [
            {"pool_id": p.pool_id, "label": p.label, "size": p.size,
             "methods": list(p.methods), "description": p.description}
            for p in CANDIDATE_POOLS
        ],
    }


# --------------------------------------------------------------------------- #
# Self-validation
# --------------------------------------------------------------------------- #
EXPECTED_COUNTS: Dict[str, int] = {
    "clustopt_mlp": 12,
    "clustopt_knn": 12,
    "clustopt_real": 10,
    "clustopt_fixed": 6,
    "external_same_search_space": 4,
    "external_own_search_space": 4,
    "external_original_code": 3,
    "external_metric_decomposition": 4,
}
EXPECTED_CLUSTOPT_VARIANTS = 34        # 12 MLP + 12 KNN + 10 real


def validate_registry() -> List[str]:
    """Return a list of registry problems (empty == valid)."""
    problems: List[str] = []

    names = [m.method_name for m in METHODS]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(f"duplicate method_name: {dupes}")

    disk = [m.on_disk_name for m in METHODS]
    dupes = sorted({n for n in disk if disk.count(n) > 1})
    if dupes:
        problems.append(f"duplicate on_disk_name: {dupes}")

    labels = [m.short_label for m in METHODS]
    dupes = sorted({n for n in labels if labels.count(n) > 1})
    if dupes:
        problems.append(f"duplicate short_label: {dupes}")

    for m in METHODS:
        for fname, allowed in (
            ("schema", SCHEMAS), ("framework", FRAMEWORKS),
            ("method_family", METHOD_FAMILIES),
            ("utility_source", UTILITY_SOURCES), ("k_policy", K_POLICIES),
            ("weighting_mode", WEIGHTING_MODES),
            ("metric_inventory", METRIC_INVENTORIES),
        ):
            val = getattr(m, fname)
            if val not in allowed:
                problems.append(f"{m.method_name}: {fname}={val!r} not in {allowed}")
        if m.oracle_level not in ORACLE_LEVELS:
            problems.append(f"{m.method_name}: oracle_level={m.oracle_level!r}")
        if m.is_utility_oracle and m.is_meta_learning:
            problems.append(f"{m.method_name}: both utility oracle and meta-learning")
        if m.k_policy == "fixed" and m.fixed_metric_count is None:
            problems.append(f"{m.method_name}: fixed k_policy without fixed_metric_count")
        if m.weighting_mode == "softmax_t05" and m.softmax_temperature != 0.5:
            problems.append(f"{m.method_name}: softmax mode without T=0.5")
        if len(m.short_label) > 16:
            problems.append(f"{m.method_name}: short_label too long ({m.short_label!r})")

    counts: Dict[str, int] = {}
    for m in METHODS:
        counts[m.method_family] = counts.get(m.method_family, 0) + 1
    for fam, want in EXPECTED_COUNTS.items():
        got = counts.get(fam, 0)
        if got != want:
            problems.append(f"family {fam}: expected {want} methods, found {got}")

    n_variants = sum(1 for m in METHODS if m.method_family in
                     ("clustopt_mlp", "clustopt_knn", "clustopt_real"))
    if n_variants != EXPECTED_CLUSTOPT_VARIANTS:
        problems.append(f"ClustOpt variants: expected "
                        f"{EXPECTED_CLUSTOPT_VARIANTS}, found {n_variants}")

    if len(PRIMARY_VARIANT_FAMILIES) != 15:
        problems.append(f"primary variant families: expected 15, found "
                        f"{len(PRIMARY_VARIANT_FAMILIES)}")
    # 17 = 5 real (4 fixed + dynamic) + 6 MLP (4 fixed + predictK + realK)
    #      + 6 KNN, of which 15 are primary and 2 are the oracle-assisted
    #      Dynamic Real-K diagnostic cells.
    if len(VARIANT_FAMILIES) != 17:
        problems.append(f"variant families: expected 17 (15 primary + 2 "
                        f"oracle-assisted), found {len(VARIANT_FAMILIES)}")
    if len(DIAGNOSTIC_VARIANT_FAMILIES) != 2:
        problems.append(f"diagnostic (Dynamic Real-K) variant families: expected "
                        f"2, found {len(DIAGNOSTIC_VARIANT_FAMILIES)}")

    covered = {n for f in VARIANT_FAMILIES for n in f.members}
    variants = {m.method_name for m in METHODS if m.method_family in
                ("clustopt_mlp", "clustopt_knn", "clustopt_real")}
    if covered != variants:
        problems.append(
            f"variant families do not partition the 34 ClustOpt variants; "
            f"missing={sorted(variants - covered)} extra={sorted(covered - variants)}")

    for p in CANDIDATE_POOLS:
        unknown = [n for n in p.methods if n not in METHOD_BY_NAME]
        if unknown:
            problems.append(f"pool {p.pool_id}: unknown methods {unknown}")
        if p.size < 2:
            problems.append(f"pool {p.pool_id}: needs >= 2 candidates (has {p.size})")

    return problems
