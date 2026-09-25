"""Capability preflight: what each method can actually support, verified on disk.

The registry *declares* capabilities; this module *verifies* them against the
collected run frame and against a sample of run directories, then produces the
capability matrix every comparison suite consults before executing a
sub-analysis.

The contract for an unsupported analysis is fixed (§28):

1. warn **before** running the comparison;
2. name the exact methods and the exact missing field;
3. skip only that sub-analysis;
4. continue the rest of the comparison;
5. record the limitation in the comparison report and in
   ``comparison_XX_capability_warnings.csv``.

Nothing is ever back-filled with a substitute field, and no capability column is
silently null-filled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from . import paths
from .collector import STATUS_SUCCESS
from .method_registry import METHOD_BY_NAME, MethodSpec, select_methods

# --------------------------------------------------------------------------- #
# Capability vocabulary
# --------------------------------------------------------------------------- #
CAPABILITIES: Tuple[str, ...] = (
    "raw_result",
    "three_views",
    "external_ari",
    "selected_k",
    "true_k",
    "runtime",
    "all_view_runtime",
    "selected_metric_list",
    "selected_metric_weights",
    "top_metric",
    "metric_entropy",
    "per_trial_trace",
    "internal_objective",
    "trace_best_gap_inputs",
    "runtime_components",
    "end_to_end_runtime",
)

CAPABILITY_DESCRIPTIONS: Dict[str, str] = {
    "raw_result": "a readable per-run outcome file exists",
    "three_views": "all three canonical views were attempted",
    "external_ari": "external ARI against ground truth is recorded",
    "selected_k": "the selected cluster count is recorded",
    "true_k": "the ground-truth cluster count is recorded",
    "runtime": "per-view runtime is recorded",
    "all_view_runtime": "runtime is recorded for every one of the three views "
                        "(required for total_runtime_all_views_sec)",
    "selected_metric_list": "the list of selected utility metrics is recorded",
    "selected_metric_weights": "per-metric objective weights are recorded",
    "top_metric": "the single highest-weight metric is recorded",
    "metric_entropy": "the weight-distribution entropy is recorded",
    "per_trial_trace": "a per-candidate search trace (clustopt_results.csv) exists",
    "internal_objective": "the internal objective score of the selected candidate "
                          "is recorded",
    "trace_best_gap_inputs": "the trace contains both a per-candidate external ARI "
                             "and an internal objective, so the search-selection "
                             "gap can be computed",
    "runtime_components": "runtime is decomposed (fit/predict vs CVI evaluation)",
    "end_to_end_runtime": "meta-feature extraction / utility prediction / search "
                          "runtime are recorded separately",
}

# Analyses and the capabilities they require. Comparison configs reference these
# keys in ``required_capabilities`` / ``optional_capabilities``.
ANALYSIS_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "primary_ari": ("raw_result", "external_ari"),
    "best_view": ("raw_result", "external_ari", "three_views"),
    "k_analysis": ("selected_k", "true_k"),
    "runtime_best_view": ("runtime",),
    "runtime_total_all_views": ("all_view_runtime",),
    "runtime_components": ("runtime_components",),
    "end_to_end_runtime": ("end_to_end_runtime",),
    "metric_selection": ("selected_metric_list", "selected_metric_weights"),
    "metric_category_usage": ("selected_metric_list", "selected_metric_weights"),
    "new_metric_usage": ("selected_metric_list", "selected_metric_weights"),
    "metric_entropy": ("metric_entropy",),
    "trace_best_gap": ("per_trial_trace", "trace_best_gap_inputs"),
    "search_selection_gap": ("per_trial_trace", "trace_best_gap_inputs"),
    "vbs": ("raw_result", "external_ari"),
}

# Coverage threshold above which a capability counts as available for a method.
# Below it the capability is False and the shortfall is reported.
AVAILABILITY_THRESHOLD = 0.99


@dataclass
class CapabilityWarning:
    """One skipped sub-analysis, with the exact reason."""

    comparison_id: str
    analysis: str
    method_name: str
    method_short_label: str
    missing_capabilities: Tuple[str, ...]
    reason: str
    action: str = "skipped"

    def to_record(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "analysis": self.analysis,
            "method_name": self.method_name,
            "method_short_label": self.method_short_label,
            "missing_capabilities": ";".join(self.missing_capabilities),
            "reason": self.reason,
            "action": self.action,
        }


# --------------------------------------------------------------------------- #
# Building the matrix
# --------------------------------------------------------------------------- #
def _share(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def _probe_trace(sample_rows: pd.DataFrame, spec: MethodSpec, n_probe: int
                 ) -> Tuple[bool, bool, str]:
    """Sample real trace files to verify the trace capabilities.

    Returns ``(per_trial_trace, trace_best_gap_inputs, note)``. Declared-incapable
    methods are not probed (the external result schema has no trace file at all).
    """
    if not spec.trace_best_capable:
        return False, False, "no per-candidate search trace in this method's schema"
    probe = sample_rows.head(int(n_probe))
    if probe.empty:
        return False, False, "no successful runs to probe"
    n_found = 0
    n_usable = 0
    seen_note = ""
    for trace_path in probe["trace_path"].dropna():
        if not paths.exists(trace_path):
            continue
        n_found += 1
        try:
            head = pd.read_csv(paths.ext(trace_path), nrows=5)
        except Exception as exc:                              # pragma: no cover
            seen_note = f"unreadable trace: {type(exc).__name__}"
            continue
        cols = {c.lower() for c in head.columns}
        has_ari = bool(cols & {"ari", "external_ari", "adjusted_rand_index"})
        has_obj = bool(cols & {"aggregate_score", "score", "objective",
                               "cvi_score", "aggregate_objective"})
        if has_ari and has_obj:
            n_usable += 1
        elif not seen_note:
            seen_note = ("trace present but missing "
                         + ("external ARI" if not has_ari else "internal objective")
                         + " column")
    n = int(len(probe))
    return (n_found == n and n > 0), (n_usable == n and n > 0), seen_note


def build_capability_matrix(
    run_frame: pd.DataFrame,
    *,
    method_names: Optional[Sequence[str]] = None,
    trace_probe_per_method: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-method capability matrix verified against the collected run frame."""
    specs = select_methods(method_names)
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        grp = run_frame[run_frame["method_name"] == spec.method_name]
        ok = grp[grp["status"] == STATUS_SUCCESS]
        n_runs, n_ok = int(len(grp)), int(len(ok))
        if n_runs == 0:
            rows.append({
                "method_name": spec.method_name,
                "method_short_label": spec.short_label,
                "method_family": spec.method_family,
                "framework": spec.framework,
                "schema": spec.schema,
                "n_runs": 0, "n_success": 0,
                **{c: False for c in CAPABILITIES},
                "notes": "no rows collected for this method",
            })
            continue

        per_ds_views = grp.groupby("dataset_id")["view_id"].nunique()
        runtime_ok = pd.to_numeric(grp["runtime_sec"], errors="coerce").notna()
        runtime_by_ds = (grp.assign(_ok=runtime_ok)
                         .groupby("dataset_id")["_ok"].all())

        sel_list = ok["selected_metrics"].fillna("").astype(str)
        has_list = sel_list.str.len() > 2                     # not '' / '[]'
        sel_w = ok["selected_metric_weights"].fillna("").astype(str)
        has_weights = sel_w.str.len() > 2                     # not '' / '{}'

        per_trial, gap_inputs, note = _probe_trace(ok, spec, trace_probe_per_method)

        caps = {
            "raw_result": _share(grp["status"] != "missing") >= AVAILABILITY_THRESHOLD,
            "three_views": _share(per_ds_views >= 3) >= AVAILABILITY_THRESHOLD,
            "external_ari": (_share(pd.to_numeric(ok["ari"], errors="coerce").notna())
                             >= AVAILABILITY_THRESHOLD if n_ok else False),
            "selected_k": (_share(pd.to_numeric(ok["selected_k"], errors="coerce").notna())
                           >= AVAILABILITY_THRESHOLD if n_ok else False),
            "true_k": (_share(pd.to_numeric(ok["true_k"], errors="coerce").notna())
                       >= AVAILABILITY_THRESHOLD if n_ok else False),
            "runtime": _share(runtime_ok) >= AVAILABILITY_THRESHOLD,
            "all_view_runtime": _share(runtime_by_ds) >= AVAILABILITY_THRESHOLD,
            "selected_metric_list": (_share(has_list) >= AVAILABILITY_THRESHOLD
                                     if n_ok else False),
            "selected_metric_weights": (_share(has_weights) >= AVAILABILITY_THRESHOLD
                                        if n_ok else False),
            "top_metric": (_share(ok["top_metric"].notna()) >= AVAILABILITY_THRESHOLD
                           if n_ok else False),
            "metric_entropy": (
                _share(pd.to_numeric(ok["metric_weight_entropy"], errors="coerce").notna())
                >= AVAILABILITY_THRESHOLD if n_ok else False),
            "per_trial_trace": per_trial,
            "internal_objective": (
                _share(pd.to_numeric(ok["best_objective_score"], errors="coerce").notna())
                >= AVAILABILITY_THRESHOLD if n_ok else False),
            "trace_best_gap_inputs": gap_inputs,
            "runtime_components": (
                _share(pd.to_numeric(ok["fit_predict_sec"], errors="coerce").notna())
                >= AVAILABILITY_THRESHOLD if n_ok else False),
            # No method in this experiment set records meta-feature extraction,
            # utility-prediction or KNN-query time per run. Declared False
            # everywhere rather than approximated from ``runtime_sec``.
            "end_to_end_runtime": False,
        }
        notes: List[str] = []
        if note:
            notes.append(note)
        if spec.is_external:
            notes.append("external schema: no metric selection and no candidate "
                         "trace are written by the adapter")
        if spec.notes:
            notes.append(spec.notes)
        for cap in ("metric_selection_capable", "trace_best_capable",
                    "runtime_components_capable"):
            declared = getattr(spec, cap)
            key = {"metric_selection_capable": "selected_metric_weights",
                   "trace_best_capable": "per_trial_trace",
                   "runtime_components_capable": "runtime_components"}[cap]
            if bool(declared) != bool(caps[key]):
                notes.append(f"registry declares {cap}={declared} but disk shows "
                             f"{key}={caps[key]}")

        rows.append({
            "method_name": spec.method_name,
            "method_short_label": spec.short_label,
            "method_family": spec.method_family,
            "framework": spec.framework,
            "schema": spec.schema,
            "n_runs": n_runs, "n_success": n_ok,
            **caps,
            "notes": "; ".join(notes),
        })

    matrix = pd.DataFrame(rows)
    if verbose and not matrix.empty:
        n_full = int(matrix[list(CAPABILITIES)].all(axis=1).sum())
        print(f"[capability] {len(matrix)} methods; {n_full} support every "
              f"capability; end_to_end_runtime unavailable for all "
              f"(not recorded by any adapter)", flush=True)
    return matrix


def supports(matrix: pd.DataFrame, method_name: str, analysis: str) -> bool:
    """Whether ``method_name`` supports every capability ``analysis`` requires."""
    needed = ANALYSIS_REQUIREMENTS.get(analysis)
    if needed is None:                                        # pragma: no cover
        raise KeyError(f"unknown analysis {analysis!r}; known: "
                       f"{sorted(ANALYSIS_REQUIREMENTS)}")
    row = matrix[matrix["method_name"] == method_name]
    if row.empty:
        return False
    return all(bool(row.iloc[0].get(cap, False)) for cap in needed)


def missing_for(matrix: pd.DataFrame, method_name: str, analysis: str
                ) -> Tuple[str, ...]:
    needed = ANALYSIS_REQUIREMENTS.get(analysis, ())
    row = matrix[matrix["method_name"] == method_name]
    if row.empty:
        return tuple(needed)
    return tuple(cap for cap in needed if not bool(row.iloc[0].get(cap, False)))


def preflight(
    matrix: pd.DataFrame,
    *,
    comparison_id: str,
    methods: Sequence[str],
    analyses: Sequence[str],
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], List[CapabilityWarning]]:
    """Resolve which methods support which analyses, before anything runs.

    Returns ``(supported, warnings)`` where ``supported[analysis]`` is the subset
    of ``methods`` that can take part. An analysis that no method supports maps to
    an empty list -- callers skip it and keep going.
    """
    supported: Dict[str, List[str]] = {}
    warnings: List[CapabilityWarning] = []
    for analysis in analyses:
        keep: List[str] = []
        for name in methods:
            miss = missing_for(matrix, name, analysis)
            if miss:
                spec = METHOD_BY_NAME.get(name)
                warnings.append(CapabilityWarning(
                    comparison_id=comparison_id, analysis=analysis,
                    method_name=name,
                    method_short_label=spec.short_label if spec else name,
                    missing_capabilities=miss,
                    reason="; ".join(CAPABILITY_DESCRIPTIONS.get(c, c) + " is unavailable"
                                     for c in miss),
                ))
            else:
                keep.append(name)
        supported[analysis] = keep
        if verbose and len(keep) != len(methods):
            dropped = [n for n in methods if n not in keep]
            print(f"[preflight] {comparison_id}/{analysis}: skipping "
                  f"{len(dropped)} method(s): {dropped}", flush=True)
    return supported, warnings


def warnings_frame(warnings: Iterable[CapabilityWarning]) -> pd.DataFrame:
    records = [w.to_record() for w in warnings]
    if not records:
        return pd.DataFrame(columns=[
            "comparison_id", "analysis", "method_name", "method_short_label",
            "missing_capabilities", "reason", "action"])
    return pd.DataFrame(records)


def matrix_markdown(matrix: pd.DataFrame) -> str:
    """Human-readable ``method_capability_matrix.md``."""
    lines: List[str] = [
        "# Method Capability Matrix",
        "",
        "Which analyses each evaluated method can actually support, **verified "
        "against the collected run frame and against sampled run directories** "
        "-- not assumed from the registry.",
        "",
        "A capability is `yes` only when at least "
        f"{AVAILABILITY_THRESHOLD:.0%} of that method's runs carry the required "
        "field. Anything below that is `no`, and the shortfall is named in the "
        "notes column rather than silently null-filled.",
        "",
        "## Capabilities",
        "",
        "| capability | meaning |",
        "|---|---|",
    ]
    for cap in CAPABILITIES:
        lines.append(f"| `{cap}` | {CAPABILITY_DESCRIPTIONS[cap]} |")

    lines += [
        "",
        "## Analyses and their requirements",
        "",
        "| analysis | requires |",
        "|---|---|",
    ]
    for analysis, needed in ANALYSIS_REQUIREMENTS.items():
        lines.append(f"| `{analysis}` | "
                     + ", ".join(f"`{c}`" for c in needed) + " |")

    if matrix.empty:
        lines += ["", "_No capability rows (no methods collected)._", ""]
        return "\n".join(lines)

    lines += ["", "## Matrix", "",
              "| method | label | family | runs | ok | "
              + " | ".join(c.replace("_", " ") for c in CAPABILITIES) + " |",
              "|---" * (5 + len(CAPABILITIES)) + "|"]
    for _, r in matrix.iterrows():
        cells = ["yes" if bool(r.get(c)) else "no" for c in CAPABILITIES]
        lines.append(f"| `{r['method_name']}` | {r['method_short_label']} | "
                     f"{r['method_family']} | {int(r['n_runs'])} | "
                     f"{int(r['n_success'])} | " + " | ".join(cells) + " |")

    lines += ["", "## Known unavailable analyses", ""]
    unavailable: Dict[str, List[str]] = {}
    for analysis in ANALYSIS_REQUIREMENTS:
        bad = [r["method_short_label"] for _, r in matrix.iterrows()
               if missing_for(matrix, r["method_name"], analysis)]
        if bad:
            unavailable[analysis] = bad
    if not unavailable:
        lines.append("_Every collected method supports every analysis._")
    else:
        for analysis, bad in unavailable.items():
            lines.append(f"- **`{analysis}`** is unavailable for "
                         f"{len(bad)} method(s): " + ", ".join(bad) + ".")
    notes = matrix[matrix["notes"].astype(str).str.len() > 0]
    if len(notes):
        lines += ["", "## Notes", "",
                  "| method | note |", "|---|---|"]
        for _, r in notes.iterrows():
            lines.append(f"| `{r['method_name']}` | {r['notes']} |")
    lines.append("")
    return "\n".join(lines)
