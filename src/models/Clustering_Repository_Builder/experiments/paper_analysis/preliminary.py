"""The preliminary retrospective representative-selection stage.

This is the only analysis stage executed before the five comparison suites are
approved. It:

1. collects the canonical run frame for Split 1 and derives the best-view and
   dataset-level frames;
2. writes the protocol documents and the method label map;
3. audits the method registry, the metric taxonomy and the per-method capabilities;
4. runs every global and family-level retrospective selection the config declares;
5. freezes the selection and portfolio manifests that suites 1-5 will read;
6. computes the selection-aware bootstrap intervals;
7. writes the preliminary workbook and reports;
8. registers every artifact in the analysis manifest.

It performs **no** comparison between methods beyond what selection requires.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import (
    capability, collector, excel, frames, gaps, labels, metric_analysis,
    metric_registry, paths, portfolio_registry, reports, selection, stability, vbs,
)
from .artifact_manifest import ArtifactManifest
from .config import RunConfig, load_config
from .method_registry import (
    CANDIDATE_POOL_BY_ID, METHOD_BY_NAME, METHOD_NAMES, VARIANT_FAMILIES,
    label_map_records, registry_snapshot, short_label, validate_registry,
)

SECTION_PROTOCOL = "00_protocol_and_definitions"
SECTION_PRELIM = "01_preliminary_selection"
SECTION_RAW = "13_raw_and_reproducibility"


@dataclass
class PreliminaryOutput:
    """Everything the stage produced, returned so the CLI can report on it."""

    run_frame: pd.DataFrame
    best_view: pd.DataFrame
    dataset_level: pd.DataFrame
    capability_matrix: pd.DataFrame
    coverage_audit: pd.DataFrame
    method_summary: pd.DataFrame
    selection_manifest: pd.DataFrame
    global_selections: List[selection.Selection] = field(default_factory=list)
    family_selections: List[selection.Selection] = field(default_factory=list)
    portfolio_manifest: pd.DataFrame = field(default_factory=pd.DataFrame)
    capability_warnings: pd.DataFrame = field(default_factory=pd.DataFrame)
    trace_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, Optional[pd.DataFrame]] = field(default_factory=dict)


def _frame(tables: Dict[str, Optional[pd.DataFrame]], key: str) -> pd.DataFrame:
    """``tables[key]`` as a DataFrame, never ``None``.

    The obvious shorthand is a bug on DataFrames -- pandas raises on truthiness --
    so the fallback is written out explicitly.
    """
    value = tables.get(key)
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def _cached_run_frame(run: RunConfig):
    """The canonical run frame from a previous run, or ``None``.

    Only reused when it matches the requested split and method subset; otherwise
    the stage re-collects rather than silently analysing the wrong data.
    """
    for suffix in (".parquet", ".csv"):
        path = run.output_root / SECTION_RAW / f"canonical_run_frame{suffix}"
        if not paths.exists(path):
            continue
        frame = paths.read_table(path)
        if frame.empty:
            return None
        if int(frame["split_id"].iloc[0]) != int(run.split_id):
            return None
        if run.methods and set(run.methods) != set(frame["method_name"].unique()):
            return None
        if run.limit_datasets and frame["dataset_id"].nunique() != run.limit_datasets:
            return None
        return frame
    return None


def _cached_trace_table(run: RunConfig):
    """A previously computed search-selection gap table, or ``None``."""
    for suffix in (".parquet", ".csv"):
        path = run.output_root / SECTION_RAW / f"search_selection_gap_table{suffix}"
        if paths.exists(path):
            frame = paths.read_table(path)
            return frame if not frame.empty else None
    return None


def run_preliminary(run: RunConfig, manifest: ArtifactManifest
                    ) -> PreliminaryOutput:
    config = load_config("preliminary-selection")
    policy = _policy_from_config(config)
    run.ensure_dirs()

    # ---------------------------------------------------------------- collect
    cached = _cached_run_frame(run) if run.reuse_frames else None
    if cached is not None:
        run_frame = cached
        cstats = collector.CollectionStats(
            n_datasets=int(run_frame["dataset_id"].nunique()),
            n_rows=int(len(run_frame)),
            n_success=int((run_frame["status"] == collector.STATUS_SUCCESS).sum()),
            n_missing=int((run_frame["status"] == collector.STATUS_MISSING).sum()))
        cstats.n_failed = cstats.n_rows - cstats.n_success - cstats.n_missing
        cstats.n_provenance_mismatch = int(
            (run_frame["provenance_mismatch"].astype(str).str.len() > 0).sum())
        if run.verbose:
            print(f"[collect] REUSED cached run frame: {cstats.n_rows} rows, "
                  f"{cstats.n_datasets} datasets (no per-run files re-read)",
                  flush=True)
    else:
        assignments = paths.read_csv(run.split_assignments)
        run_frame, cstats = collector.collect_run_frame(
            assignments=assignments, split_id=run.split_id,
            method_names=run.methods, max_workers=run.collect_workers,
            limit_datasets=run.limit_datasets, verbose=run.verbose)
    best_view = frames.build_best_view_frame(run_frame, verbose=run.verbose)
    dataset_level = frames.build_dataset_level_frame(best_view)
    coverage = collector.coverage_audit(run_frame)
    method_summary = frames.summarize_methods(best_view, ("method_name",))

    # ------------------------------------------------------------ validation
    validation: Dict[str, List[str]] = {
        "method_registry": validate_registry(),
        "portfolio_registry": portfolio_registry.validate_portfolios(),
        "frames": frames.validate_frames(run_frame, best_view),
        "metric_registry": metric_registry.validate_metric_registry(
            collector.observed_selected_metrics(run_frame)),
    }

    # ------------------------------------------------- search-selection gap
    # Reads every clustopt_results.csv (~126k files), so it is opt-in. Computed
    # once here and cached, because all five comparison suites consume it and
    # none of them should pay this cost separately.
    trace_table = pd.DataFrame()
    trace_warnings: Dict[str, int] = {}
    if run.trace_gaps:
        cached_trace = _cached_trace_table(run) if run.reuse_frames else None
        if cached_trace is not None:
            trace_table = cached_trace
            if run.verbose:
                print(f"[gap:search] REUSED cached trace table: "
                      f"{len(trace_table)} runs", flush=True)
        else:
            trace_table, trace_warnings = gaps.compute_search_selection_gaps(
                run_frame, max_workers=run.collect_workers, verbose=run.verbose)
    elif run.verbose:
        print("[gap:search] skipped (pass --trace-gaps to compute the "
              "search-selection gap)", flush=True)

    # ------------------------------------------------------------ capability
    matrix = capability.build_capability_matrix(run_frame, verbose=run.verbose)
    _, warns = capability.preflight(
        matrix, comparison_id="preliminary", methods=list(METHOD_NAMES),
        analyses=list(capability.ANALYSIS_REQUIREMENTS), verbose=False)
    warnings_frame = capability.warnings_frame(warns)

    # ------------------------------------------------------------- selection
    global_selections: List[selection.Selection] = []
    family_selections: List[selection.Selection] = []
    ranking_parts: List[pd.DataFrame] = []
    family_ranking_parts: List[pd.DataFrame] = []

    def do_global(rep_id: str, pool_id: str, notes: str = "") -> None:
        sel, table = selection.select_representative(
            best_view, comparison_id="preliminary", representative_id=rep_id,
            candidate_pool_id=pool_id, split_id=run.split_id, policy=policy,
            notes=notes)
        if sel is None:
            validation.setdefault("selection", []).append(
                f"{rep_id}: candidate pool {pool_id} produced no usable rows")
            return
        global_selections.append(sel)
        if not table.empty:
            ranking_parts.append(table)

    def do_family(rep_id: str, pool_id: str, notes: str = "") -> None:
        sels, table = selection.select_family_representatives(
            best_view, comparison_id="preliminary", representative_id=rep_id,
            candidate_pool_id=pool_id, split_id=run.split_id, policy=policy,
            notes=notes)
        family_selections.extend(sels)
        if not table.empty:
            family_ranking_parts.append(table)

    # 11.1 Comparison 1 helper
    c1 = config.get("comparison_01_helper") or {}
    c1_rep = str(c1.get("representative_id", "c1_best_observed_real_fixed_raw"))
    c1_pool = str(c1.get("candidate_pool_id", "real_fixed_topk_raw"))
    do_global(c1_rep, c1_pool,
              "best observed FIXED Real-utility Top-K raw variant; "
              "real_top_dynamic_raw is recorded separately, not pooled")
    if "family" in (c1.get("scopes") or []):
        do_family(c1_rep, c1_pool)

    # 11.2 Comparison 2 raw-vs-softmax cells (one per variant family)
    c2 = config.get("comparison_02_representatives") or {}
    prefix = str(c2.get("representative_id_prefix", "c2_rep"))
    for fam in VARIANT_FAMILIES:
        rep_id = f"{prefix}__{fam.family_id}"
        pool_id = f"variant_family__{fam.family_id}"
        note = ("oracle-assisted diagnostic cell (metric count from the real "
                "utility vector); excluded from the primary 15-cell grid"
                if fam.oracle_assisted else
                "primary meta-learning cell: raw vs softmax at a matched "
                "utility source and metric-count policy")
        do_global(rep_id, pool_id, note)
        if "family" in (c2.get("scopes") or []):
            do_family(rep_id, pool_id, note)

    # 11.3 Comparison 4 fixed representatives
    c4 = config.get("comparison_04_representatives") or {}
    for item in (c4.get("primary") or []):
        do_global(str(item["representative_id"]), str(item["candidate_pool_id"]),
                  "best observed FIXED variant; retrospective global "
                  "representative, NOT an independently preselected policy")
    for item in (c4.get("supplemental") or []):
        do_global(str(item["representative_id"]), str(item["candidate_pool_id"]),
                  "supplemental pool including the operational Dynamic Predict-K "
                  "variants; does not replace the fixed representative")

    # ------------------------------------------------- selection stability
    # Run BEFORE the manifests are built: the bootstrap re-selects inside every
    # resample, and its result is written back onto each selection so the frozen
    # manifest carries the stability of the choice it records.
    stability_summary = pd.DataFrame()
    stability_frequencies = pd.DataFrame()
    if run.make_stats:
        sab = config.get("selection_aware_bootstrap") or {}
        stability_summary, stability_frequencies, stability_results = (
            stability.stability_for_selections(
                best_view, global_selections,
                n_samples=int(sab.get("n_samples",
                                      run.selection_bootstrap_samples)),
                seed=int(sab.get("seed", run.seed)), verbose=run.verbose))
        stability.attach_stability(global_selections, stability_results)
        problems = stability.validate_stability(stability_summary)
        if problems:
            validation.setdefault("selection_stability", []).extend(problems)

    selection_manifest = selection.selection_manifest(global_selections)
    family_manifest = selection.selection_manifest(family_selections)
    rep_map = selection.observed_representative_map(
        global_selections + family_selections)
    divergence = selection.family_divergence(global_selections, family_selections)
    candidate_rankings = (pd.concat(ranking_parts, ignore_index=True)
                          if ranking_parts else pd.DataFrame())
    family_rankings = (pd.concat(family_ranking_parts, ignore_index=True)
                       if family_ranking_parts else pd.DataFrame())

    # ------------------------------------------------------------ portfolios
    portfolio_manifest = pd.DataFrame(portfolio_registry.manifest_records())

    # -------------------------------------------------- selection uncertainty
    boot_frame = pd.DataFrame()
    sab = config.get("selection_aware_bootstrap") or {}
    if run.make_stats and sab.get("enabled", True):
        from . import stats as stats_mod
        records: List[Dict[str, Any]] = []
        for item in (sab.get("comparisons") or []):
            pool = CANDIDATE_POOL_BY_ID.get(str(item.get("candidate_pool_id")))
            competitor = str(item.get("competitor", ""))
            if pool is None or competitor not in METHOD_BY_NAME:
                continue
            records.append(stats_mod.selection_aware_bootstrap(
                best_view, candidate_methods=pool.methods,
                competitor_method=competitor,
                n_samples=int(sab.get("n_samples", run.selection_bootstrap_samples)),
                confidence=float(sab.get("confidence", 0.95)),
                seed=int(sab.get("seed", run.seed)),
                label=str(item.get("label", ""))))
        boot_frame = stats_mod.selection_aware_bootstrap_frame(records)

    # -------------------------------------------------------- derived subsets
    tables = _derived_tables(
        selection_manifest=selection_manifest, family_manifest=family_manifest,
        c1_rep=c1_rep, c2_prefix=prefix, c4_config=c4, best_view=best_view)
    tables.update({
        "global_representatives": selection_manifest,
        "family_representatives": family_manifest,
        "observed_representative_map": rep_map,
        "candidate_rankings": candidate_rankings,
        "family_candidate_rankings": family_rankings,
        "family_divergence": divergence,
        "portfolio_manifest": portfolio_manifest,
        "method_summary": method_summary,
        "coverage_audit": coverage,
        "capability_matrix": matrix,
        "metric_registry": pd.DataFrame(metric_registry.registry_records()),
        "selection_aware_bootstrap": boot_frame,
        "representative_stability_summary": stability_summary,
        "bootstrap_selection_frequency_by_method": stability_frequencies,
        "label_manifest": labels.build_label_manifest(selection_manifest),
    })

    output = PreliminaryOutput(
        run_frame=run_frame, best_view=best_view, dataset_level=dataset_level,
        capability_matrix=matrix, coverage_audit=coverage,
        method_summary=method_summary, selection_manifest=selection_manifest,
        global_selections=global_selections, family_selections=family_selections,
        portfolio_manifest=portfolio_manifest, capability_warnings=warnings_frame,
        trace_table=trace_table, validation=validation, tables=tables,
        stats={
            "collection": cstats.to_dict(),
            "n_datasets": int(run_frame["dataset_id"].nunique()),
            "n_families": int(run_frame["family"].nunique()),
            "n_subfamilies": int(run_frame["subfamily"].nunique()),
            "n_methods": int(run_frame["method_name"].nunique()),
            "n_methods_expected": len(METHOD_NAMES),
            "n_views": int(run_frame["view_id"].nunique()),
            "n_runs": int(len(run_frame)),
            "n_success": cstats.n_success,
            "n_failed": cstats.n_failed,
            "n_missing": cstats.n_missing,
            "n_provenance_mismatch": cstats.n_provenance_mismatch,
            "n_best_view_rows": int(len(best_view)),
            "n_best_view_with_success": int(best_view["has_success"].sum()),
            "n_global_selections": len(global_selections),
            "n_family_selections": len(family_selections),
            "n_portfolios": len(portfolio_manifest),
            "n_runs_with_search_trace": int(len(trace_table)),
            "trace_warnings": dict(sorted(trace_warnings.items())),
            "split_id": run.split_id,
        })

    if run.dry_run:
        if run.verbose:
            print("[preliminary] dry run: no files written", flush=True)
        return output

    _write_protocol(run, manifest)
    _write_raw(run, manifest, output)
    _write_preliminary(run, manifest, output, config)
    return output


# --------------------------------------------------------------------------- #
# Config -> policy
# --------------------------------------------------------------------------- #
def _policy_from_config(config: Dict[str, Any]) -> selection.SelectionPolicy:
    block = config.get("selection_policy") or {}
    return selection.SelectionPolicy(
        selection_metric=str(block.get("selection_metric",
                                       selection.SELECTION_METRIC)),
        ari_tie_tolerance=float(block.get("ari_tie_tolerance",
                                          selection.DEFAULT_ARI_TIE_TOLERANCE)),
        practical_ari_tolerance=float(block.get(
            "practical_ari_tolerance", selection.DEFAULT_PRACTICAL_ARI_TOLERANCE)),
        report_practical_equivalence=bool(block.get(
            "report_practical_equivalence", True)))


# --------------------------------------------------------------------------- #
# Derived per-comparison selection subsets
# --------------------------------------------------------------------------- #
def _derived_tables(*, selection_manifest: pd.DataFrame,
                    family_manifest: pd.DataFrame, c1_rep: str, c2_prefix: str,
                    c4_config: Dict[str, Any], best_view: pd.DataFrame
                    ) -> Dict[str, Optional[pd.DataFrame]]:
    out: Dict[str, Optional[pd.DataFrame]] = {}
    if selection_manifest.empty:
        return out
    rid = selection_manifest["representative_id"]

    out["comparison_01_helper"] = pd.concat([
        selection_manifest[rid == c1_rep],
        family_manifest[family_manifest["representative_id"] == c1_rep]
        if not family_manifest.empty else pd.DataFrame(),
    ], ignore_index=True)

    c2_mask = rid.str.startswith(f"{c2_prefix}__")
    c2_global = selection_manifest[c2_mask].copy()
    if not c2_global.empty:
        fam_by_rep = {f"{c2_prefix}__{f.family_id}": f for f in VARIANT_FAMILIES}
        c2_global["variant_family_id"] = c2_global["representative_id"].map(
            lambda r: fam_by_rep[r].family_id if r in fam_by_rep else "")
        c2_global["utility_source_cell"] = c2_global["representative_id"].map(
            lambda r: fam_by_rep[r].utility_source if r in fam_by_rep else "")
        c2_global["policy_cell"] = c2_global["representative_id"].map(
            lambda r: fam_by_rep[r].policy_label if r in fam_by_rep else "")
        c2_global["is_primary_cell"] = c2_global["representative_id"].map(
            lambda r: bool(fam_by_rep[r].is_primary) if r in fam_by_rep else False)
        c2_global["is_oracle_assisted_cell"] = c2_global["representative_id"].map(
            lambda r: bool(fam_by_rep[r].oracle_assisted) if r in fam_by_rep else False)
    out["comparison_02_global"] = c2_global
    out["comparison_02_family"] = (
        family_manifest[family_manifest["representative_id"].str.startswith(
            f"{c2_prefix}__")] if not family_manifest.empty else pd.DataFrame())

    c4_ids = [str(i["representative_id"])
              for i in (c4_config.get("primary") or [])
              + (c4_config.get("supplemental") or [])]
    c4 = selection_manifest[rid.isin(c4_ids)].copy()
    if not c4.empty:
        primary_ids = {str(i["representative_id"])
                       for i in (c4_config.get("primary") or [])}
        c4["is_primary_representative"] = c4["representative_id"].isin(primary_ids)
        c4["role"] = np.where(c4["is_primary_representative"], "primary",
                              "supplemental")
    out["comparison_04_fixed"] = c4

    # A compact one-row-per-representative summary for the workbook / report.
    keep = ["representative_id", "candidate_pool_id", "candidate_pool_size",
            "selected_method", "selected_short_label",
            "selected_best_view_mean_ari", "selected_best_view_median_ari",
            "selected_best_view_std_ari", "selected_k_accuracy",
            "selected_total_runtime_all_views_median_sec", "n_tied_on_primary",
            "tie_breaks_invoked", "selection_type", "selection_split",
            "is_independently_preselected", "paper_claim_level"]
    out["preliminary_selection_summary"] = selection_manifest[
        [c for c in keep if c in selection_manifest.columns]]
    return out


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _write_protocol(run: RunConfig, manifest: ArtifactManifest) -> None:
    d = run.output_root / SECTION_PROTOCOL
    docs = {
        "terminology.md": (reports.terminology_md(),
                           "The terminology contract: what each kind of winner in "
                           "this analysis does and does not mean.", "paper_main"),
        "oracle_levels.md": (reports.oracle_levels_md(),
                             "The four distinct oracles and why they stay "
                             "separate.", "paper_main"),
        "runtime_definitions.md": (reports.runtime_definitions_md(),
                                   "Runtime field semantics; why the all-views sum "
                                   "is the primary cost.", "paper_main"),
        "metric_definitions.md": (reports.metric_definitions_md(),
                                  "Primary and supplementary metric definitions.",
                                  "paper_main"),
    }
    for name, (text, desc, importance) in docs.items():
        paths.write_text(d / name, text)
        manifest.add_report(d / name, comparison_id="protocol",
                            section=SECTION_PROTOCOL, description=desc,
                            paper_importance=importance)
    label_frame = pd.DataFrame(label_map_records())
    manifest.add_table(
        paths.write_csv(d / "method_label_map.csv", label_frame, index=False),
        comparison_id="protocol", section=SECTION_PROTOCOL,
        description="Short plot label -> concrete method name and full taxonomy "
                    "for every registered method.",
        paper_importance="paper_main")
    paths.write_json(d / "method_label_map.json", label_map_records())
    manifest.add_metadata(d / "method_label_map.json", comparison_id="protocol",
                          section=SECTION_PROTOCOL,
                          description="Machine-readable method label map.",
                          paper_importance="appendix")


def _write_raw(run: RunConfig, manifest: ArtifactManifest,
               out: PreliminaryOutput) -> None:
    d = run.output_root / SECTION_RAW
    for name, frame, desc in (
        ("canonical_run_frame", out.run_frame,
         "One row per dataset x method x view with full registry provenance."),
        ("canonical_best_view_frame", out.best_view,
         "One row per dataset x method: the highest-ARI successful view (a view "
         "oracle), with all-views runtime and capability flags."),
        ("canonical_dataset_level_frame", out.dataset_level,
         "Best-view frame plus per-dataset cross-method rank and delta context."),
    ):
        written = paths.write_parquet(d / f"{name}.parquet", frame)
        manifest.add_frame(written, comparison_id="raw", section=SECTION_RAW,
                           description=desc, paper_importance="appendix",
                           appendix_only=True)
    if out.trace_table is not None and not out.trace_table.empty:
        written = paths.write_parquet(
            d / "search_selection_gap_table.parquet", out.trace_table)
        manifest.add_frame(
            written, comparison_id="raw", section=SECTION_RAW,
            description="Per-run search-selection gap: the best ARI in the "
                        "ClustOpt candidate trace vs the ARI the internal "
                        "objective selected. ClustOpt methods only -- the "
                        "external adapters expose no compatible trace.",
            paper_importance="appendix", appendix_only=True)

    # Compact CSV summaries beside the large Parquet frames, for readability.
    manifest.add_table(
        paths.write_csv(d / "method_summary.csv", out.method_summary, index=False),
        comparison_id="raw", section=SECTION_RAW,
        description="Primary-metric block per method (compact CSV view of the "
                    "best-view frame).",
        paper_importance="paper_main")
    manifest.add_table(
        paths.write_csv(d / "coverage_audit.csv", out.coverage_audit, index=False),
        comparison_id="raw", section=SECTION_RAW,
        description="Per-method dataset/view/success coverage on this split.",
        paper_importance="paper_supplement")
    manifest.add_table(
        paths.write_csv(d / "method_capability_matrix.csv", out.capability_matrix,
                        index=False),
        comparison_id="raw", section=SECTION_RAW,
        description="Verified per-method capability matrix.",
        paper_importance="paper_supplement")
    paths.write_text(d / "method_capability_matrix.md",
                     capability.matrix_markdown(out.capability_matrix))
    manifest.add_report(d / "method_capability_matrix.md", comparison_id="raw",
                        section=SECTION_RAW,
                        description="Human-readable capability matrix.",
                        paper_importance="paper_supplement")
    manifest.add_table(
        paths.write_csv(d / "preflight_warnings.csv", out.capability_warnings,
                        index=False),
        comparison_id="raw", section=SECTION_RAW,
        description="Every analysis/method pair that cannot be run, with the "
                    "exact missing capability.",
        paper_importance="paper_supplement")
    manifest.add_table(
        paths.write_csv(d / "metric_registry.csv",
                        pd.DataFrame(metric_registry.registry_records()),
                        index=False),
        comparison_id="raw", section=SECTION_RAW,
        description="Canonical metric taxonomy.", paper_importance="paper_main")
    paths.write_json(d / "metric_registry.json", {
        "audit": metric_registry.METRIC_AUDIT,
        "metrics": metric_registry.registry_records()})
    manifest.add_metadata(d / "metric_registry.json", comparison_id="raw",
                          section=SECTION_RAW,
                          description="Machine-readable metric taxonomy + audit.",
                          paper_importance="appendix")
    paths.write_json(d / "method_registry_snapshot.json", registry_snapshot())
    manifest.add_metadata(d / "method_registry_snapshot.json", comparison_id="raw",
                          section=SECTION_RAW,
                          description="Full method registry snapshot (all specs, "
                                      "variant families, candidate pools).",
                          paper_importance="appendix")
    paths.write_json(d / "run_config.json", {
        **run.to_dict(), "collection_stats": out.stats, "validation": out.validation})
    manifest.add_metadata(d / "run_config.json", comparison_id="raw",
                          section=SECTION_RAW,
                          description="Resolved run configuration, collection "
                                      "statistics and validation results.",
                          paper_importance="appendix")
    paths.write_json(d / "code_version.json", _code_version(run.repo_root))
    manifest.add_metadata(d / "code_version.json", comparison_id="raw",
                          section=SECTION_RAW,
                          description="Git commit, branch and dirty-state of the "
                                      "code that produced this run.",
                          paper_importance="appendix")

    # Audits.
    paths.write_text(d / "METHOD_REGISTRY_AUDIT.md",
                     reports.method_registry_audit_md(
                         out.coverage_audit, out.validation.get("method_registry", [])))
    manifest.add_report(d / "METHOD_REGISTRY_AUDIT.md", comparison_id="raw",
                        section=SECTION_RAW,
                        description="Method registry audit: expected vs registered "
                                    "counts, taxonomy validation, on-disk coverage.",
                        paper_importance="paper_supplement")
    paths.write_text(d / "METRIC_REGISTRY_AUDIT.md", metric_registry.audit_markdown())
    manifest.add_report(d / "METRIC_REGISTRY_AUDIT.md", comparison_id="raw",
                        section=SECTION_RAW,
                        description="Metric taxonomy audit: derivation rules, "
                                    "coverage, aliases, classic/new split.",
                        paper_importance="paper_supplement")
    paths.write_text(d / "CAPABILITY_AUDIT.md",
                     reports.capability_audit_md(
                         capability.matrix_markdown(out.capability_matrix),
                         out.capability_warnings))
    manifest.add_report(d / "CAPABILITY_AUDIT.md", comparison_id="raw",
                        section=SECTION_RAW,
                        description="Capability audit and every preflight warning.",
                        paper_importance="paper_supplement")


def _write_preliminary(run: RunConfig, manifest: ArtifactManifest,
                       out: PreliminaryOutput, config: Dict[str, Any]) -> None:
    d = run.output_root / SECTION_PRELIM
    file_map: Dict[str, Tuple[str, str]] = {
        "global_representatives": (
            "global_representatives.csv",
            "One retrospective global representative per candidate pool."),
        "family_representatives": (
            "family_representatives.csv",
            "Family-level retrospective representatives (policy oracle; "
            "headroom only)."),
        "observed_representative_map": (
            "observed_representative_map.csv",
            "representative_id -> concrete method name, for global and family "
            "selections."),
        "candidate_rankings": (
            "candidate_rankings.csv",
            "Full ranked candidate table behind every global selection."),
        "family_candidate_rankings": (
            "family_candidate_rankings.csv",
            "Full ranked candidate tables behind every family selection."),
        "family_divergence": (
            "family_divergence.csv",
            "How often the family-level choice differs from the global one."),
        "comparison_01_helper": (
            "comparison_01_helper_selections.csv",
            "Comparison 1: best observed fixed Real Top-K raw variant "
            "(global + per family)."),
        "comparison_02_global": (
            "comparison_02_global_representatives.csv",
            "Comparison 2: raw-vs-softmax representative per (utility source x "
            "metric-count policy) cell."),
        "comparison_02_family": (
            "comparison_02_family_representatives.csv",
            "Comparison 2: family-level raw-vs-softmax reselection (headroom "
            "only)."),
        "comparison_04_fixed": (
            "comparison_04_fixed_representatives.csv",
            "Comparison 4: best observed fixed MLP/KNN representatives plus the "
            "supplemental pools."),
        "portfolio_manifest": (
            "portfolio_manifest.csv",
            "Frozen VBS portfolio membership with oracle-contamination flags."),
        "preliminary_selection_summary": (
            "preliminary_selection_summary.csv",
            "Compact one-row-per-representative selection summary."),
        "selection_aware_bootstrap": (
            "selection_aware_bootstrap_summary.csv",
            "Bootstrap that re-selects the representative inside every resample."),
        "representative_stability_summary": (
            "representative_stability_summary.csv",
            "How often each retrospective representative survives re-selection "
            "inside a bootstrap resample: selection frequency, runner-up "
            "frequency, entropy of the selected-method distribution."),
        "bootstrap_selection_frequency_by_method": (
            "bootstrap_selection_frequency_by_method.csv",
            "Per-candidate probability of being selected across bootstrap "
            "resamples, for every pool with >= 2 candidates."),
        "label_manifest": (
            "label_manifest.csv",
            "Every short label across methods, representatives and portfolios, "
            "with its oracle status and search-space status."),
        "method_summary": (
            "method_summary.csv",
            "Primary-metric block for every collected method (no selection "
            "applied)."),
    }
    for key, (filename, desc) in file_map.items():
        frame = out.tables.get(key)
        if frame is None or frame.empty:
            manifest.add_table(None, comparison_id="preliminary",
                               section=SECTION_PRELIM, description=desc,
                               created=False,
                               notes="not created: no rows produced")
            continue
        manifest.add_table(
            paths.write_csv(d / filename, frame, index=False),
            comparison_id="preliminary", section=SECTION_PRELIM,
            description=desc,
            paper_importance=("paper_main" if key in (
                "global_representatives", "observed_representative_map",
                "comparison_01_helper", "comparison_02_global",
                "comparison_04_fixed", "portfolio_manifest",
                "preliminary_selection_summary") else "paper_supplement"),
            selection_type=("retrospective_family" if "family" in key
                            else "retrospective_global"))

    # selection_manifest is the canonical frozen artifact -- also written under
    # its own name so consumers never have to guess.
    manifest.add_table(
        paths.write_csv(d / "selection_manifest.csv", out.selection_manifest,
                        index=False),
        comparison_id="preliminary", section=SECTION_PRELIM,
        description="FROZEN selection manifest. Every comparison suite reads its "
                    "representatives from here and must never reselect.",
        paper_importance="paper_main", selection_type="retrospective_global")
    paths.write_json(d / "selection_manifest.json", {
        "selection_metric": selection.SELECTION_METRIC,
        "tie_break_rules": selection.TIE_BREAK_DESCRIPTION,
        "selection_split": run.split_id,
        "is_independently_preselected": False,
        "global": paths.as_records(out.selection_manifest),
        "family": paths.as_records(_frame(out.tables, "family_representatives")),
    })
    manifest.add_metadata(d / "selection_manifest.json",
                          comparison_id="preliminary", section=SECTION_PRELIM,
                          description="Machine-readable frozen selection manifest.",
                          paper_importance="appendix")
    paths.write_text(d / "selection_manifest.md",
                     _selection_manifest_md(out, run.split_id))
    manifest.add_report(d / "selection_manifest.md", comparison_id="preliminary",
                        section=SECTION_PRELIM,
                        description="Human-readable frozen selection manifest.",
                        paper_importance="paper_main")
    paths.write_json(d / "observed_representative_map.json",
                     paths.as_records(_frame(out.tables,
                                             "observed_representative_map")))
    manifest.add_metadata(d / "observed_representative_map.json",
                          comparison_id="preliminary", section=SECTION_PRELIM,
                          description="Machine-readable representative map.",
                          paper_importance="appendix")
    paths.write_json(d / "portfolio_manifest.json",
                     portfolio_registry.manifest_records())
    manifest.add_metadata(d / "portfolio_manifest.json",
                          comparison_id="preliminary", section=SECTION_PRELIM,
                          description="Machine-readable frozen portfolio manifest.",
                          paper_importance="appendix")
    paths.write_text(d / "portfolio_manifest.md", _portfolio_manifest_md())
    manifest.add_report(d / "portfolio_manifest.md", comparison_id="preliminary",
                        section=SECTION_PRELIM,
                        description="Human-readable frozen portfolio manifest.",
                        paper_importance="paper_main")
    label_manifest = _frame(out.tables, "label_manifest")
    if not label_manifest.empty:
        paths.write_text(run.output_root / SECTION_PROTOCOL / "label_manifest.md",
                         labels.manifest_markdown(label_manifest))
        manifest.add_report(
            run.output_root / SECTION_PROTOCOL / "label_manifest.md",
            comparison_id="protocol", section=SECTION_PROTOCOL,
            description="Every short label across methods, representatives and "
                        "portfolios, with the marker conventions that keep "
                        "oracle and unmatched-search-space candidates visible.",
            paper_importance="paper_main")
        manifest.add_table(
            paths.write_csv(run.output_root / SECTION_PROTOCOL
                            / "label_manifest.csv", label_manifest, index=False),
            comparison_id="protocol", section=SECTION_PROTOCOL,
            description="Machine-readable label manifest.",
            paper_importance="paper_main")

    # ---- report --------------------------------------------------------- #
    real_dyn = None
    ms = out.method_summary
    if ms is not None and not ms.empty:
        row = ms[ms["method_name"] == "real_top_dynamic_raw"]
        if not row.empty:
            real_dyn = row.iloc[0].to_dict()
    report = reports.preliminary_selection_report_md(
        split_id=run.split_id, coverage_summary=out.stats,
        global_reps=out.selection_manifest,
        candidate_rankings=_frame(out.tables, "candidate_rankings"),
        comparison_01=_frame(out.tables, "comparison_01_helper"),
        comparison_02_global=_frame(out.tables, "comparison_02_global"),
        comparison_02_family=_frame(out.tables, "comparison_02_family"),
        family_divergence=_frame(out.tables, "family_divergence"),
        comparison_04=_frame(out.tables, "comparison_04_fixed"),
        portfolio_manifest=out.portfolio_manifest,
        method_summary=out.method_summary,
        capability_warnings=out.capability_warnings,
        selection_bootstrap=out.tables.get("selection_aware_bootstrap"),
        real_dynamic_row=real_dyn)
    paths.write_text(d / "PRELIMINARY_SELECTION_REPORT.md", report)
    manifest.add_report(d / "PRELIMINARY_SELECTION_REPORT.md",
                        comparison_id="preliminary", section=SECTION_PRELIM,
                        description="The preliminary selection study: every "
                                    "candidate pool, every selection, tie-breaks, "
                                    "limitations.",
                        paper_importance="paper_main")

    # ---- workbook -------------------------------------------------------- #
    if run.make_excel:
        spec = excel.preliminary_workbook(out.tables)
        path = excel.write_workbook(
            spec, d, selection_manifest=out.selection_manifest,
            capability_warnings=(out.capability_warnings
                                 if not out.capability_warnings.empty else None),
            extra_notes=[
                "Every representative in this workbook was selected "
                "RETROSPECTIVELY on Split 1 from an explicit candidate pool.",
                "No representative is an independently preselected deployment "
                "policy: Split 1 is the only split with downstream clustering "
                "experiments.",
                "Read the Definitions sheet before quoting any number.",
            ],
            dry_run=run.dry_run)
        manifest.add_workbook(path, comparison_id="preliminary",
                              section=SECTION_PRELIM,
                              description="Formatted preliminary selection "
                                          "workbook.",
                              paper_importance="paper_supplement")


def _selection_manifest_md(out: PreliminaryOutput, split_id: int) -> str:
    from .reports import _table as md_table
    lines = [
        "# Frozen Selection Manifest",
        "",
        f"Split {split_id}. Selection metric: `{selection.SELECTION_METRIC}`. "
        f"Tie-break chain: `{selection.TIE_BREAK_DESCRIPTION}`.",
        "",
        "Every comparison suite reads its representatives from this manifest and "
        "must never reselect a winner in plotting, reporting or packaging code.",
        "",
        "**Every row has `is_independently_preselected = False`** and "
        "`paper_claim_level = retrospective_summary`.",
        "",
        "## Global representatives",
        "",
    ]
    lines += md_table(out.selection_manifest, [
        "representative_id", "candidate_pool_id", "candidate_pool_size",
        "selected_short_label", "selected_method",
        "selected_best_view_mean_ari", "selected_best_view_median_ari",
        "selected_best_view_std_ari", "selected_k_accuracy",
        "selected_total_runtime_all_views_median_sec", "n_tied_on_primary",
        "tie_breaks_invoked", "n_practical_equivalents"], limit=100)
    fam = out.tables.get("family_representatives")
    if fam is not None and not fam.empty:
        lines += ["## Family representatives (policy oracle; headroom only)", ""]
        lines += md_table(fam, [
            "representative_id", "family", "selected_short_label",
            "selected_method", "selected_best_view_mean_ari",
            "n_tied_on_primary", "tie_breaks_invoked"], limit=400)
    return "\n".join(lines)


def _portfolio_manifest_md() -> str:
    lines = [
        "# Frozen Portfolio Manifest",
        "",
        "Each portfolio's VBS picks a different member per dataset using that "
        "dataset's true external ARI. **Every portfolio here is a non-deployable "
        "oracle upper bound** (`selection_type = dataset_vbs`, "
        "`paper_claim_level = upper_bound`).",
        "",
        "The contamination flags matter: a portfolio containing `real_*` members "
        "inherits a utility oracle, and one containing `*_dynamic_realK_*` members "
        "inherits dynamic-K oracle assistance. "
        "`strict_deployable_clustopt_vbs` is the only portfolio whose members are "
        "all fully operational.",
        "",
        "| portfolio | label | size | real utility | oracle dyn-K | frameworks |",
        "|---|---|---|---|---|---|",
    ]
    for p in portfolio_registry.PORTFOLIOS:
        lines.append(
            f"| `{p.portfolio_id}` | {p.short_label} | {p.size} "
            f"| {'yes' if p.contains_real_utility else 'no'} "
            f"| {'yes' if p.contains_oracle_assisted_dynamic_k else 'no'} "
            f"| {', '.join(p.frameworks)} |")
    lines += ["", "## Membership", ""]
    for p in portfolio_registry.PORTFOLIOS:
        lines += [f"### `{p.portfolio_id}` ({p.size} members)", "",
                  p.label, "", p.notes or "", "",
                  ", ".join(f"`{short_label(m)}`" for m in p.methods), ""]
    return "\n".join(lines)


def _code_version(repo_root: Path) -> Dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=str(repo_root),
                                  capture_output=True, text=True, timeout=30
                                  ).stdout.strip()
        except Exception:                                      # pragma: no cover
            return ""
    return {
        "commit": git("rev-parse", "HEAD"),
        "short_commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
