"""Canonical metric taxonomy for the utility inventory.

Category assignment is **derived, not guessed**. Each metric's top-level
category comes from the implementation package its class lives in, read off the
live ClustOpt metric registry::

    metrics/core_cvi/       -> core_cvi
    metrics/structure_base/ -> structural
    metrics/pattern_base/   -> pattern
    metrics/image_base/     -> image

The finer ``metric_subcategory`` comes from the MLP utility model's head groups
(``models.metric_utility_mlp.head_groups.METRIC_HEAD_GROUPS``), which is the
grouping the predictor itself was trained with.

Two "classic" notions are kept **separate and explicitly named**, because they
answer different questions and conflating them would misstate RQ1:

``is_classic_baseline_metric``
    Member of the 4-metric objective actually used by the ``uniform_classic_cvi``
    baseline (silhouette, calinski_harabasz, davies_bouldin, dbcv). This is the
    inventory that RQ1-B compares against.

``is_classic_head_group``
    Member of the MLP head group named ``classic_cvi`` (14 metrics: the 6
    core CVIs plus the 8 structural indices). This is the predictor's own notion
    of "classic".

``is_new`` is defined as ``not is_classic_head_group`` -- i.e. the 46 pattern /
image metrics designed for this work -- and is the flag used by the
new-metric-usage analysis. Any metric whose two classic notions disagree is
listed in the audit rather than silently resolved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Category vocabulary
# --------------------------------------------------------------------------- #
METRIC_CATEGORIES: Tuple[str, ...] = ("core_cvi", "structural", "pattern", "image")

_PACKAGE_TO_CATEGORY: Dict[str, str] = {
    "core_cvi": "core_cvi",
    "structure_base": "structural",
    "pattern_base": "pattern",
    "image_base": "image",
}

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "core_cvi": "Classical internal cluster-validity indices (silhouette, "
                "Calinski-Harabasz, Davies-Bouldin, DBCV, S_Dbw, "
                "noise-aware silhouette).",
    "structural": "Point-cloud structural indices: compactness, dispersion, "
                  "size balance, neighbourhood purity, inter-cluster distance.",
    "pattern": "Geometric / pattern-shape indices computed on the point cloud "
               "(arcs, ladders, bands, parabolas, periodicity, curvature, "
               "symmetry, hollowness, ...).",
    "image": "Indices computed on a rasterised image of the cluster (Hough, "
             "morphology, texture/frequency, skeleton, Euler characteristic, "
             "shape fitting).",
}

# The classic-CVI baseline objective (verified against
# configs/experiments/phaseD/uniform_classic_cvi/*.json).
CLASSIC_BASELINE_METRICS: Tuple[str, ...] = (
    "silhouette", "calinski_harabasz", "davies_bouldin", "dbcv",
)

# Head group treated as "classic" by the utility predictor.
CLASSIC_HEAD_GROUP = "classic_cvi"

# Known alias pairs between the head-group naming and the live metric registry.
# Verified by inspecting both sources; recorded (not silently applied) so the
# audit can show them.
#
# These aliases are NOT cosmetic: the per-run ``selected_metrics.json`` files
# name metrics with the *utility-vector* (head-group) spelling, e.g.
# ``hough_circle_arc_strength``, while the ClustOpt implementation registry uses
# ``hough_arc_circle_strength``. Every lookup here therefore normalises through
# :func:`canonical_metric_name` so both spellings resolve to one taxonomy row and
# no selected metric is silently counted as ``unknown``.
HEAD_GROUP_ALIASES: Dict[str, str] = {
    "convexity_solidity_image": "convexity_ratio_image_based",
    "hough_circle_arc_strength": "hough_arc_circle_strength",
}


@dataclass(frozen=True)
class MetricSpec:
    metric_name: str
    metric_category: str
    metric_subcategory: str
    is_classic_baseline_metric: bool
    is_classic_head_group: bool
    is_new: bool
    source_module: str
    implementation_class: str
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Build from the live sources
# --------------------------------------------------------------------------- #
def _load_sources() -> Tuple[Dict[str, object], Dict[str, str], List[str]]:
    """Return (metric registry, metric -> head group, load warnings)."""
    warnings: List[str] = []
    try:
        from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
            METRIC_REGISTRY,
        )
    except Exception as exc:                                   # pragma: no cover
        warnings.append(f"could not import ClustOpt METRIC_REGISTRY: "
                        f"{type(exc).__name__}: {exc}")
        METRIC_REGISTRY = {}
    try:
        from models.metric_utility_mlp.head_groups import METRIC_HEAD_GROUPS
        head = {m: g for g, ms in METRIC_HEAD_GROUPS.items() for m in ms}
    except Exception as exc:                                   # pragma: no cover
        warnings.append(f"could not import METRIC_HEAD_GROUPS: "
                        f"{type(exc).__name__}: {exc}")
        head = {}
    return METRIC_REGISTRY, head, warnings


def _category_from_module(module: str) -> Optional[str]:
    for part in module.split("."):
        if part in _PACKAGE_TO_CATEGORY:
            return _PACKAGE_TO_CATEGORY[part]
    return None


def _first_doc_line(obj: object) -> str:
    doc = (type(obj).__doc__ or "").strip()
    return doc.splitlines()[0].strip() if doc else ""


def build_metric_registry() -> Tuple[Tuple[MetricSpec, ...], Dict[str, object]]:
    """Build the metric registry plus an audit payload.

    The audit payload records every fact a reader needs to trust the taxonomy:
    counts per category, the classic/new split, alias resolutions, metrics whose
    category could not be derived, and metrics present in one source but not the
    other. Nothing is silently defaulted.
    """
    registry, head, warnings = _load_sources()
    head_resolved: Dict[str, str] = {}
    alias_applied: List[Dict[str, str]] = []
    for name, group in head.items():
        target = HEAD_GROUP_ALIASES.get(name, name)
        if target != name:
            alias_applied.append({"head_group_name": name,
                                  "registry_name": target, "head_group": group})
        head_resolved[target] = group

    specs: List[MetricSpec] = []
    unmapped_category: List[str] = []
    unmapped_subcategory: List[str] = []
    ambiguous_classic: List[Dict[str, object]] = []

    for name in sorted(registry):
        obj = registry[name]
        module = type(obj).__module__
        cat = _category_from_module(module)
        if cat is None:
            unmapped_category.append(name)
            continue
        sub = head_resolved.get(name)
        if sub is None:
            unmapped_subcategory.append(name)
            sub = "unmapped"
        is_baseline = name in CLASSIC_BASELINE_METRICS
        is_head_classic = sub == CLASSIC_HEAD_GROUP
        if is_baseline and not is_head_classic:
            ambiguous_classic.append({
                "metric_name": name, "reason":
                "in the classic-CVI baseline objective but not in the "
                "classic_cvi head group"})
        specs.append(MetricSpec(
            metric_name=name, metric_category=cat, metric_subcategory=sub,
            is_classic_baseline_metric=is_baseline,
            is_classic_head_group=is_head_classic,
            is_new=not is_head_classic,
            source_module=module, implementation_class=type(obj).__name__,
            description=_first_doc_line(obj),
        ))

    by_cat: Dict[str, int] = {}
    by_sub: Dict[str, int] = {}
    for s in specs:
        by_cat[s.metric_category] = by_cat.get(s.metric_category, 0) + 1
        by_sub[s.metric_subcategory] = by_sub.get(s.metric_subcategory, 0) + 1

    audit: Dict[str, object] = {
        "n_metrics": len(specs),
        "n_registry_entries": len(registry),
        "n_head_group_entries": len(head),
        "counts_by_category": dict(sorted(by_cat.items())),
        "counts_by_subcategory": dict(sorted(by_sub.items())),
        "n_classic_baseline_metrics": sum(s.is_classic_baseline_metric for s in specs),
        "n_classic_head_group": sum(s.is_classic_head_group for s in specs),
        "n_new": sum(s.is_new for s in specs),
        "classic_baseline_metrics": list(CLASSIC_BASELINE_METRICS),
        "aliases_applied": alias_applied,
        "unmapped_category": unmapped_category,
        "unmapped_subcategory": unmapped_subcategory,
        "ambiguous_classic": ambiguous_classic,
        "in_registry_not_in_head_groups": sorted(
            set(registry) - set(head_resolved)),
        "in_head_groups_not_in_registry": sorted(
            set(head_resolved) - set(registry)),
        "load_warnings": warnings,
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "classic_definitions": {
            "is_classic_baseline_metric":
                "member of the 4-metric uniform_classic_cvi baseline objective",
            "is_classic_head_group":
                f"member of the '{CLASSIC_HEAD_GROUP}' MLP head group",
            "is_new": "NOT a member of the classic_cvi head group",
        },
    }
    return tuple(specs), audit


METRICS, METRIC_AUDIT = build_metric_registry()
METRIC_BY_NAME: Dict[str, MetricSpec] = {m.metric_name: m for m in METRICS}
METRIC_NAMES: Tuple[str, ...] = tuple(m.metric_name for m in METRICS)
METRIC_COUNT: int = len(METRICS)

CATEGORY_OF: Dict[str, str] = {m.metric_name: m.metric_category for m in METRICS}
SUBCATEGORY_OF: Dict[str, str] = {m.metric_name: m.metric_subcategory for m in METRICS}
IS_NEW: Dict[str, bool] = {m.metric_name: m.is_new for m in METRICS}

# Deterministic display order: category, then subcategory, then name.
_CAT_ORDER = {c: i for i, c in enumerate(METRIC_CATEGORIES)}
METRIC_DISPLAY_ORDER: Tuple[str, ...] = tuple(
    m.metric_name for m in sorted(
        METRICS, key=lambda s: (_CAT_ORDER.get(s.metric_category, 99),
                                s.metric_subcategory, s.metric_name)))


def canonical_metric_name(metric_name: str) -> str:
    """Resolve a utility-vector metric name to its canonical registry name.

    Applies :data:`HEAD_GROUP_ALIASES`; unknown names pass through unchanged so
    the caller (and the audit) can see exactly what was not recognised.
    """
    name = str(metric_name)
    return HEAD_GROUP_ALIASES.get(name, name)


def category_of(metric_name: str) -> str:
    """Category for ``metric_name``; ``'unknown'`` for anything unregistered.

    Callers must surface ``'unknown'`` counts rather than dropping them -- the
    metric-usage audit fails the run if any selected metric lands here.
    """
    return CATEGORY_OF.get(canonical_metric_name(metric_name), "unknown")


def subcategory_of(metric_name: str) -> str:
    return SUBCATEGORY_OF.get(canonical_metric_name(metric_name), "unknown")


def is_new_metric(metric_name: str) -> bool:
    return bool(IS_NEW.get(canonical_metric_name(metric_name), False))


def is_known_metric(metric_name: str) -> bool:
    return canonical_metric_name(metric_name) in METRIC_BY_NAME


def registry_records() -> List[Dict[str, object]]:
    return [m.to_dict() for m in METRICS]


EXPECTED_METRIC_COUNT = 60


def validate_metric_registry(observed_metrics: Optional[Sequence[str]] = None
                             ) -> List[str]:
    """Return taxonomy problems (empty == valid).

    ``observed_metrics`` -- when given, every metric actually selected by any run
    must be registered and categorised; unknown ones are reported by name.
    """
    problems: List[str] = []
    if METRIC_COUNT != EXPECTED_METRIC_COUNT:
        problems.append(f"expected {EXPECTED_METRIC_COUNT} metrics, "
                        f"found {METRIC_COUNT}")
    if METRIC_AUDIT["unmapped_category"]:
        problems.append(f"metrics without a derivable category: "
                        f"{METRIC_AUDIT['unmapped_category']}")
    if METRIC_AUDIT["load_warnings"]:
        problems.append(f"source load warnings: {METRIC_AUDIT['load_warnings']}")

    names = [m.metric_name for m in METRICS]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(f"duplicate metric_name: {dupes}")

    for m in METRICS:
        if m.metric_category not in METRIC_CATEGORIES:
            problems.append(f"{m.metric_name}: bad category {m.metric_category!r}")
        if m.is_classic_head_group == m.is_new:
            problems.append(f"{m.metric_name}: is_new must be the negation of "
                            f"is_classic_head_group")

    missing_baseline = [n for n in CLASSIC_BASELINE_METRICS if n not in METRIC_BY_NAME]
    if missing_baseline:
        problems.append(f"classic-baseline metrics absent from the registry: "
                        f"{missing_baseline}")

    if observed_metrics is not None:
        unknown = sorted({str(n) for n in observed_metrics
                          if str(n) and not is_known_metric(n)})
        if unknown:
            problems.append(f"selected metrics missing from the taxonomy: {unknown}")
    return problems


def audit_markdown() -> str:
    """Human-readable ``metric_registry_audit.md`` body."""
    a = METRIC_AUDIT
    lines: List[str] = [
        "# Metric Registry Audit",
        "",
        "Canonical taxonomy of the ClustOpt utility metric inventory. Categories "
        "are **derived** from each metric's implementation package (never from "
        "its name); subcategories are the MLP utility model's own head groups.",
        "",
        "## Sources",
        "",
        "| source | role | entries |",
        "|---|---|---|",
        f"| `models.ClustOpt.cluster_validity_indices.metrics.metrics_registry."
        f"METRIC_REGISTRY` | metric universe + implementation module -> category "
        f"| {a['n_registry_entries']} |",
        f"| `models.metric_utility_mlp.head_groups.METRIC_HEAD_GROUPS` | "
        f"subcategory + classic/new flag | {a['n_head_group_entries']} |",
        f"| `configs/experiments/phaseD/uniform_classic_cvi/*.json` | the "
        f"4-metric classic baseline objective | "
        f"{len(CLASSIC_BASELINE_METRICS)} |",
        "",
        f"**Registered metrics: {a['n_metrics']}**",
        "",
        "## Category derivation",
        "",
        "| implementation package | category |",
        "|---|---|",
    ]
    for pkg, cat in _PACKAGE_TO_CATEGORY.items():
        lines.append(f"| `metrics/{pkg}/` | `{cat}` |")
    lines += [
        "",
        "| category | n | description |",
        "|---|---|---|",
    ]
    for cat in METRIC_CATEGORIES:
        lines.append(f"| `{cat}` | {a['counts_by_category'].get(cat, 0)} | "
                     f"{CATEGORY_DESCRIPTIONS[cat]} |")
    lines += [
        "",
        "## Subcategories (MLP head groups)",
        "",
        "| subcategory | n |",
        "|---|---|",
    ]
    for sub, n in a["counts_by_subcategory"].items():
        lines.append(f"| `{sub}` | {n} |")

    lines += [
        "",
        "## Classic vs new",
        "",
        "Two distinct notions are tracked; they are **not** interchangeable.",
        "",
        "| flag | definition | n |",
        "|---|---|---|",
        f"| `is_classic_baseline_metric` | "
        f"{a['classic_definitions']['is_classic_baseline_metric']} | "
        f"{a['n_classic_baseline_metrics']} |",
        f"| `is_classic_head_group` | "
        f"{a['classic_definitions']['is_classic_head_group']} | "
        f"{a['n_classic_head_group']} |",
        f"| `is_new` | {a['classic_definitions']['is_new']} | {a['n_new']} |",
        "",
        f"Classic baseline objective: "
        + ", ".join(f"`{m}`" for m in CLASSIC_BASELINE_METRICS) + ".",
        "",
        "The new-metric-usage analysis uses `is_new`. RQ1-B (inventory effect) "
        "compares the classic **baseline objective** against the full inventory, "
        "so it uses `is_classic_baseline_metric`.",
        "",
        "## Audit checks",
        "",
        "| check | result |",
        "|---|---|",
        f"| metrics with a derivable category | "
        f"{a['n_metrics']}/{a['n_registry_entries']} |",
        f"| metrics without a derivable category | "
        f"{len(a['unmapped_category'])} |",
        f"| metrics without a head-group subcategory | "
        f"{len(a['unmapped_subcategory'])} |",
        f"| duplicate category assignments | 0 (category is a single derived "
        f"field per metric) |",
        f"| ambiguous classic assignments | {len(a['ambiguous_classic'])} |",
        "",
    ]
    if a["aliases_applied"]:
        lines += [
            "### Name aliases resolved",
            "",
            "The head-group table and the live metric registry use different "
            "spellings for these metrics. The registry spelling wins; the "
            "mapping is explicit, not fuzzy-matched.",
            "",
            "| head-group name | registry name | head group |",
            "|---|---|---|",
        ]
        for al in a["aliases_applied"]:
            lines.append(f"| `{al['head_group_name']}` | `{al['registry_name']}` "
                         f"| `{al['head_group']}` |")
        lines.append("")
    if a["ambiguous_classic"]:
        lines += ["### Metrics needing clarification", ""]
        for amb in a["ambiguous_classic"]:
            lines.append(f"- `{amb['metric_name']}`: {amb['reason']}")
        lines.append("")
    if a["unmapped_subcategory"]:
        lines += [
            "### Metrics with no head-group subcategory",
            "",
            "These are categorised (top level) but the utility model's head-group "
            "table does not name them. They are marked `unmapped` rather than "
            "being assigned a guessed group.",
            "",
        ] + [f"- `{n}`" for n in a["unmapped_subcategory"]] + [""]

    lines += [
        "## Full registry",
        "",
        "| metric | category | subcategory | classic (baseline) | classic (head) "
        "| new | implementation |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in METRIC_DISPLAY_ORDER:
        m = METRIC_BY_NAME[name]
        lines.append(
            f"| `{m.metric_name}` | {m.metric_category} | {m.metric_subcategory} "
            f"| {'yes' if m.is_classic_baseline_metric else ''} "
            f"| {'yes' if m.is_classic_head_group else ''} "
            f"| {'yes' if m.is_new else ''} | `{m.implementation_class}` |")
    lines.append("")
    return "\n".join(lines)
