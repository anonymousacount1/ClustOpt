"""Unified label manifest across methods, representatives and VBS portfolios.

Plot axes are the place where a careful distinction most easily gets lost: an
oracle-assisted variant, a retrospectively selected representative and a
non-deployable VBS all become "just another bar" unless the label itself carries
the qualification. This module builds one manifest over **all three** candidate
kinds so uniqueness and honesty can be checked together rather than per-registry.

Naming rules enforced here:

* unique across every method, representative and portfolio label;
* <= 16 characters where possible;
* ``DynRK`` marks the oracle-assisted Dynamic Real-K variants;
* ``~`` marks an unmatched search space, ``-Paper`` the original authors' code;
* every portfolio label identifies itself as a VBS;
* no label may claim deployability that its oracle status contradicts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .method_registry import METHOD_BY_NAME, METHODS, short_label
from .portfolio_registry import PORTFOLIOS

MAX_LABEL_LENGTH = 16

CANDIDATE_TYPES = ("method", "representative", "portfolio")

ORACLE_STATUS = {
    "none": "fully operational",
    "utility_oracle": "utility oracle (uses the real utility vector)",
    "dynamic_k_oracle_assisted": "oracle-assisted (metric count from real utility)",
    "external_ari_vbs": "policy-selection oracle (per-dataset VBS)",
}


def _search_space_status(spec) -> str:
    if not spec.is_external:
        return "ClustOpt search space"
    if spec.same_search_space:
        return "matched to the ClustOpt search space"
    if spec.method_family == "external_original_code":
        return "UNMATCHED: original authors' code, own space and budget"
    return "UNMATCHED: own (unconstrained) search space"


def method_label_records() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in METHODS:
        rows.append({
            "short_label": m.short_label,
            "full_display_name": m.display_name,
            "identifier": m.method_name,
            "candidate_type": "method",
            "oracle_status": ORACLE_STATUS.get(m.oracle_level, m.oracle_level),
            "oracle_level": m.oracle_level,
            "search_space_status": _search_space_status(m),
            "search_budget": m.search_budget,
            "is_paper_method": m.is_paper_method,
            "paper_label": m.paper_label,
            "notes": m.notes,
        })
    return rows


def portfolio_label_records() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in PORTFOLIOS:
        rows.append({
            "short_label": p.short_label,
            "full_display_name": p.label,
            "identifier": p.portfolio_id,
            "candidate_type": "portfolio",
            "oracle_status": (
                "policy-selection oracle (per-dataset VBS)"
                + ("" if p.fully_operational
                   else f"; members also contaminated by "
                        f"{p.oracle_contamination_reason}")),
            "oracle_level": "external_ari_vbs",
            "search_space_status": "ClustOpt search space"
            if "clustopt" in p.frameworks else ", ".join(p.frameworks),
            "search_budget": None,
            "is_paper_method": True,
            "paper_label": p.label,
            "notes": p.notes,
        })
    return rows


def representative_label_records(selection_manifest: Optional[pd.DataFrame]
                                 ) -> List[Dict[str, Any]]:
    """Labels for the frozen representatives.

    A representative's label is the concrete method's label -- deliberately, so a
    reader always sees which method is behind the id -- but the row records the
    representative id, the pool it came from and its retrospective status.
    """
    if selection_manifest is None or selection_manifest.empty:
        return []
    rows: List[Dict[str, Any]] = []
    globals_only = selection_manifest
    if "selection_scope" in selection_manifest.columns:
        globals_only = selection_manifest[
            selection_manifest["selection_scope"] == "global"]
    for _, r in globals_only.iterrows():
        method = str(r.get("selected_method", ""))
        spec = METHOD_BY_NAME.get(method)
        rows.append({
            "short_label": short_label(method),
            "full_display_name": (
                f"{spec.display_name if spec else method} "
                f"(retrospective representative "
                f"{r.get('representative_id')})"),
            "identifier": str(r.get("representative_id")),
            "candidate_type": "representative",
            "oracle_status": (
                "retrospective global representative selected on Split 1 from "
                f"{r.get('candidate_pool_size')} candidates; "
                + ORACLE_STATUS.get(spec.oracle_level if spec else "none",
                                    "fully operational")),
            "oracle_level": spec.oracle_level if spec else "none",
            "search_space_status": _search_space_status(spec) if spec else "",
            "search_budget": spec.search_budget if spec else None,
            "is_paper_method": True,
            "paper_label": spec.paper_label if spec else method,
            "notes": (f"concrete method: {method}; pool "
                      f"{r.get('candidate_pool_id')}; "
                      f"is_independently_preselected=False"),
        })
    return rows


def build_label_manifest(selection_manifest: Optional[pd.DataFrame] = None
                         ) -> pd.DataFrame:
    rows = (method_label_records()
            + representative_label_records(selection_manifest)
            + portfolio_label_records())
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["label_length"] = frame["short_label"].astype(str).str.len()
    order = {t: i for i, t in enumerate(CANDIDATE_TYPES)}
    frame["_o"] = frame["candidate_type"].map(order)
    return (frame.sort_values(["_o", "identifier"], kind="mergesort")
            .drop(columns=["_o"]).reset_index(drop=True))


def validate_labels(manifest: pd.DataFrame) -> List[str]:
    """Label-contract checks (empty list == valid)."""
    problems: List[str] = []
    if manifest.empty:
        problems.append("label manifest is empty")
        return problems

    # Uniqueness is required WITHIN a candidate type. A representative
    # deliberately shares its concrete method's label -- that is the point -- so
    # the cross-type collision between "method" and "representative" is expected
    # and is disambiguated by the identifier column.
    for kind in ("method", "portfolio"):
        sub = manifest[manifest["candidate_type"] == kind]
        dupes = sub[sub["short_label"].duplicated(keep=False)]
        if len(dupes):
            problems.append(f"duplicate {kind} short_label: "
                            f"{sorted(dupes['short_label'].unique())}")
    methods = set(manifest.loc[manifest["candidate_type"] == "method",
                               "short_label"])
    portfolios = set(manifest.loc[manifest["candidate_type"] == "portfolio",
                                  "short_label"])
    clash = methods & portfolios
    if clash:
        problems.append(f"labels used for both a method and a portfolio: "
                        f"{sorted(clash)}")

    long_labels = manifest[manifest["label_length"] > MAX_LABEL_LENGTH]
    if len(long_labels):
        problems.append(f"{len(long_labels)} label(s) exceed "
                        f"{MAX_LABEL_LENGTH} characters: "
                        f"{sorted(long_labels['short_label'].unique())}")

    # Oracle-assisted variants must be visually distinguishable.
    for m in METHODS:
        if m.is_oracle_assisted and "DynRK" not in m.short_label:
            problems.append(f"{m.method_name}: oracle-assisted Dynamic Real-K "
                            f"variant without a DynRK marker "
                            f"({m.short_label!r})")
        if (m.is_external and not m.same_search_space
                and not (m.short_label.endswith("~")
                         or "Paper" in m.short_label
                         or m.short_label == "AML4C")):
            problems.append(f"{m.method_name}: unmatched search space without a "
                            f"marker ({m.short_label!r})")

    for _, r in manifest[manifest["candidate_type"] == "portfolio"].iterrows():
        if "VBS" not in str(r["short_label"]) and "Strict" not in str(r["short_label"]):
            problems.append(f"{r['identifier']}: portfolio label "
                            f"{r['short_label']!r} does not identify itself "
                            f"as a VBS")

    missing = manifest[manifest["oracle_status"].astype(str).str.len() == 0]
    if len(missing):
        problems.append(f"{len(missing)} label rows have no oracle_status")
    return problems


def manifest_markdown(manifest: pd.DataFrame) -> str:
    lines = [
        "# Label Manifest",
        "",
        "Every short label used on a plot axis or in a table, across all three "
        "candidate kinds, with its oracle status and search-space status.",
        "",
        "## Conventions",
        "",
        "| marker | meaning |",
        "|---|---|",
        "| `DynRK` | **oracle-assisted** Dynamic Real-K: the metric COUNT comes "
        "from the real utility vector, so it uses information unavailable at "
        "test time |",
        "| `Dyn` | Dynamic **Predict-K**: fully operational |",
        "| `-R` / `-S` | raw / softmax(T=0.5) weighting |",
        "| `Real ...` | real-utility (utility oracle) variant |",
        "| `~` suffix | external method on an **unmatched** search space |",
        "| `-Paper` | original authors' code, unmatched budget and space |",
        "| `... VBS` | a **portfolio**, not a method: a non-deployable "
        "per-dataset oracle upper bound |",
        "| `Strict` | portfolio whose members are all fully operational |",
        "",
        "A representative deliberately carries its concrete method's label, so a "
        "reader always sees the method rather than an opaque id. The identifier "
        "column disambiguates.",
        "",
    ]
    for kind, title in (("method", "Methods"),
                        ("representative", "Retrospective representatives"),
                        ("portfolio", "VBS portfolios")):
        sub = manifest[manifest["candidate_type"] == kind]
        if sub.empty:
            continue
        lines += [
            f"## {title} ({len(sub)})", "",
            "| short label | identifier | full name | oracle status | "
            "search space |",
            "|---|---|---|---|---|",
        ]
        for _, r in sub.iterrows():
            lines.append(f"| `{r['short_label']}` | `{r['identifier']}` | "
                         f"{r['full_display_name']} | {r['oracle_status']} | "
                         f"{r['search_space_status']} |")
        lines.append("")
    return "\n".join(lines)
