"""Build the Master Unified Repository by merging all subfamily outputs, then
run the Master Readiness Audit.

This does NOT derive any model-specific MKRs (ML2DAC / AutoClust) and does NOT
move or duplicate the subfamily outputs or their label files. It produces one
canonical merged layer plus readiness reports.

Example (Windows cmd)::

    python experiments/external_baselines/Unified_MKR/runners/build_master_unified_repository.py ^
      --outputs-root "...\\Unified_MKR\\outputs" ^
      --master-root  "...\\Unified_MKR\\master" ^
      --overwrite
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --- import wiring ---------------------------------------------------------
_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1]  # .../Unified_MKR
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from unified_mkr import io_utils  # noqa: E402
from unified_mkr import master_audit, master_merge  # noqa: E402
from unified_mkr.master_storage import MasterStorage  # noqa: E402


# Known renamed/dead ClustOpt image metrics: registry carries the new name but
# the computed values are stored under the old name, so the new-named column is
# fully NaN by design. Not used by ML2DAC/AutoClust derivations.
_KNOWN_DEAD = {
    "convexity_ratio_image_based": "renamed/dead (values under convexity_solidity_image)",
    "hough_arc_circle_strength": "renamed/dead (values under hough_circle_arc_strength)",
}


def _log(msg: str) -> None:
    print(f"[master-mkr] {msg}", flush=True)


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _clean(obj):
    """Recursively convert numpy/pandas scalars to JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, float) and (pd.isna(obj)):
        return None
    return obj


# ===========================================================================
# Markdown renderers
# ===========================================================================

def _md_table(headers: List[str], rows: List[List[object]]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _yn(flag) -> str:
    return "✅ PASS" if flag else "❌ FAIL"


def render_schema_audit(merge: master_merge.MergeResult) -> str:
    sr = merge.schema_report
    lines = ["# Master Schema Audit", "", f"_Generated {_now()}_", ""]
    lines += ["## Schema version", "",
              f"- Versions found: {sr['schema_versions']}",
              f"- Consistent: {_yn(sr['schema_version_consistent'])}", ""]

    lines += ["## Stacked-table column consistency", ""]
    rows = []
    for tbl, info in sr["tables"].items():
        rows.append([tbl, info["canonical_n_columns"], _yn(info["consistent"]),
                     len(info["mismatches"])])
    lines.append(_md_table(["table", "canonical #cols", "consistent", "#mismatched subfamilies"], rows))
    for tbl, info in sr["tables"].items():
        if info["mismatches"]:
            lines += ["", f"### {tbl} mismatches", ""]
            for m in info["mismatches"][:50]:
                lines.append(f"- `{m['subfamily']}` missing={m['missing_columns']} extra={m['extra_columns']}")
    lines.append("")

    cfg = merge.configuration_report
    reg = merge.registry_report
    lines += ["## Configuration catalogue consistency", "",
              f"- Identical across subfamilies: {_yn(cfg['identical_across_subfamilies'])}",
              f"- Distinct fingerprints: {cfg['n_distinct_fingerprints']}",
              f"- Canonical rows: {cfg['n_rows_canonical']} (cols={cfg['n_cols_canonical']})",
              f"- Canonical fingerprint: `{cfg['canonical_fingerprint'][:16]}…`", ""]
    if not cfg["identical_across_subfamilies"]:
        lines += ["### Configuration divergences", ""]
        for d in cfg.get("divergences", []):
            lines.append(f"- {d['subfamilies']}: missing={d['missing_columns']} extra={d['extra_columns']}")
        lines.append("")

    lines += ["## Metric registry consistency", "",
              f"- Identical across subfamilies: {_yn(reg['identical_across_subfamilies'])}",
              f"- Distinct fingerprints: {reg['n_distinct_fingerprints']}",
              f"- Canonical rows: {reg['n_rows_canonical']} (cols={reg['n_cols_canonical']})",
              f"- Canonical fingerprint: `{reg['canonical_fingerprint'][:16]}…`", ""]
    if not reg["identical_across_subfamilies"]:
        lines += ["### Registry divergences", ""]
        for d in reg.get("divergences", []):
            lines.append(f"- {d['subfamilies']}: missing={d['missing_columns']} extra={d['extra_columns']}")
        lines.append("")
    return "\n".join(lines)


def render_coverage(audit: dict) -> str:
    fam = audit["family_coverage"]
    c = audit["counts"]
    lines = ["# Master Coverage Report", "", f"_Generated {_now()}_", "",
             "## Global counts (actual vs expected)", ""]
    lines.append(_md_table(
        ["metric", "actual", "expected", "match"],
        [["datasets", c["datasets"], c["expected"]["datasets"], _yn(c["datasets_match_expected"])],
         ["records", c["records"], c["expected"]["records"], _yn(c["records_match_expected"])],
         ["evaluations", c["evaluations"], c["expected"]["evaluations"], _yn(c["evaluations_match_expected"])]]))
    lines += ["", f"- records == datasets × 3: {_yn(c['records_eq_datasets_x3'])}",
              f"- evaluations == records × 32: {_yn(c['evaluations_eq_records_x32'])}", ""]

    lines += ["## By family", ""]
    rows = [[f["family"], f["subfamilies"], f["datasets"], f["records"],
             f["evaluations"], f["real_failures"]] for f in fam["by_family"]]
    t = fam["totals"]
    rows.append(["**TOTAL**", fam["n_subfamilies"], t["datasets"], t["records"],
                 t["evaluations"], t["real_failures"]])
    lines.append(_md_table(["family", "subfamilies", "datasets", "records", "evaluations", "real_failures"], rows))

    lines += ["", "## By subfamily", ""]
    rows = [[s["family"], s["subfamily"], s["datasets"], s["records"],
             s["evaluations"], s["real_failures"]] for s in fam["by_subfamily"]]
    lines.append(_md_table(["family", "subfamily", "datasets", "records", "evaluations", "real_failures"], rows))
    return "\n".join(lines)


def render_split(audit: dict) -> str:
    sp = audit["split_distribution"]
    lines = ["# Master Split Distribution", "", f"_Generated {_now()}_", "",
             f"- Splits: {sp['n_splits']}",
             f"- Records with null split_id: {sp['records_with_null_split']}",
             f"- All records have split_id: {_yn(sp['all_records_have_split'])}", "",
             "## By split", ""]
    rows = [[r["split_id"], r["datasets"], r["records"], r["evaluations"], r.get("failures", 0)]
            for r in sp["by_split"]]
    lines.append(_md_table(["split_id", "datasets", "records", "evaluations", "failures"], rows))
    return "\n".join(lines)


def render_failures(audit: dict) -> str:
    fa = audit["failure_analysis"]
    lines = ["# Master Failure Analysis", "", f"_Generated {_now()}_", "",
             f"- Total failure rows: **{fa['total_failure_rows']}**",
             f"- Real failures (clustering stage): **{fa['real_failures']}**",
             f"- Meta-feature-stage bookkeeping rows: {fa['metafeature_stage_rows']} "
             f"(deferred ml2dac meta-features, populated 100% by the pymfe pass; not real failures)",
             f"- Records losing all 32 configs: {fa['records_losing_all_32']}",
             f"- Affects derived-MKR readiness: {_yn(not fa['affects_readiness'])} "
             f"({'blocks' if fa['affects_readiness'] else 'no impact'})", ""]
    for title, key in [("Real failures by family", "real_by_family"),
                       ("Real failures by subfamily", "real_by_subfamily"),
                       ("Real failures by algorithm", "real_by_algorithm"),
                       ("Real failures by view type", "real_by_view_type"),
                       ("Real failures by reason", "real_by_reason")]:
        d = fa.get(key, {})
        if d:
            lines += [f"## {title}", ""]
            lines.append(_md_table(["key", "count"], [[k, v] for k, v in d.items()]))
            lines.append("")
    return "\n".join(lines)


def render_readiness(audit: dict, merge: master_merge.MergeResult) -> str:
    rd = audit["readiness"]
    c = audit["counts"]
    mf_ml, mf_ac = audit["ml2dac_features"], audit["autoclust_features"]
    ext, ml_cvi, ac_cvi = audit["external_metrics"], audit["ml2dac_cvi"], audit["autoclust_cvi"]
    clustopt, im = audit["clustopt_metrics"], audit["internal_metrics"]
    lab, st = audit["label_audit"], audit["storage"]

    lines = ["# Master Readiness Report", "", f"_Generated {_now()}_", "",
             f"## ✅ READY: {rd['ready']}" if rd["ready"] else "## ❌ NOT READY", ""]
    if rd["blockers"]:
        lines += ["**Blockers:**", ""] + [f"- {b}" for b in rd["blockers"]] + [""]

    lines += ["## Consistency (pre-merge)", ""]
    sr = merge.schema_report
    lines.append(_md_table(["check", "result"], [
        ["schema version consistent", _yn(sr["schema_version_consistent"])],
        ["stacked schemas consistent", _yn(all(t["consistent"] for t in sr["tables"].values()))],
        ["registry identical", _yn(merge.registry_report["identical_across_subfamilies"])],
        ["configurations identical", _yn(merge.configuration_report["identical_across_subfamilies"])],
    ]))

    lines += ["", "## Counts", ""]
    lines.append(_md_table(["metric", "actual", "expected", "match"], [
        ["datasets", c["datasets"], c["expected"]["datasets"], _yn(c["datasets_match_expected"])],
        ["records", c["records"], c["expected"]["records"], _yn(c["records_match_expected"])],
        ["evaluations", c["evaluations"], c["expected"]["evaluations"], _yn(c["evaluations_match_expected"])],
        ["real failures", c["real_failures_clustering_stage"], c["expected"]["real_failures"],
         _yn(c["real_failures_match_expected"])]]))
    lines += ["", f"- records == datasets × 3: {_yn(c['records_eq_datasets_x3'])}",
              f"- evaluations == records × 32: {_yn(c['evaluations_eq_records_x32'])}"]

    lines += ["", "## Feature coverage", ""]
    lines.append(_md_table(["feature set", "#cols", "all populated", "partial", "none", "status all success", "readiness"], [
        ["ML2DAC", mf_ml["n_feature_columns"], mf_ml["records_all_features_populated"],
         mf_ml["records_partial_features"], mf_ml["records_no_features"],
         _yn(mf_ml["all_status_success"]), rd["ml2dac_feature_readiness"]],
        ["AutoClust", mf_ac["n_feature_columns"], mf_ac["records_all_features_populated"],
         mf_ac["records_partial_features"], mf_ac["records_no_features"],
         _yn(mf_ac["all_status_success"]), rd["autoclust_feature_readiness"]]]))
    if mf_ml["top20_by_missingness"]:
        lines += ["", "### ML2DAC features by missingness (top 20)", ""]
        lines.append(_md_table(["feature", "null_ratio", "null_count"],
            [[r["feature"], round(r["null_ratio"], 4), r["null_count"]] for r in mf_ml["top20_by_missingness"]]))

    lines += ["", "## Metric coverage", ""]
    lines.append(_md_table(["metric set", "result", "detail"], [
        ["external metrics", _yn(ext["all_present"]), f"{len(ext['expected'])} expected, all present={ext['all_present']}"],
        ["ML2DAC CVIs (7)", _yn(ml_cvi["pass"]), rd["ml2dac_cvi_readiness"]],
        ["AutoClust CVIs (7)", _yn(ac_cvi["pass"]), rd["autoclust_cvi_readiness"]],
        ["ClustOpt metrics", _yn(clustopt["all_present"] and clustopt["count_matches_expected"]),
         f"{clustopt['present_count']}/{clustopt['expected_count']} present, missing={clustopt['missing_clustopt_metrics']}"],
        ["internal registry metrics", _yn(im["n_present"] == im["n_registered"]),
         f"{im['n_present']}/{im['n_registered']} present"]]))

    lines += ["", "### ML2DAC CVI detail", "",
              f"_Internal CVIs are undefined for single-cluster partitions; "
              f"{ml_cvi['n_degenerate_single_cluster']:,} of {ml_cvi['n_valid_evaluations']:,} valid "
              f"evaluations are degenerate (predicted_k=1). Coverage is judged over the "
              f"{ml_cvi['n_defined_partitions']:,} defined (predicted_k≥2) partitions._", ""]
    lines.append(_md_table(["CVI", "column", "present", "null_ratio(valid)", "null_ratio(defined)", "ok"],
        [[name, d["column"], d["present"], d["null_ratio_valid"], d["null_ratio_defined"], _yn(d["ok"])]
         for name, d in ml_cvi["cvis"].items()]))

    if im["high_missingness_over_10pct_defined"]:
        lines += ["", "### Internal metrics with >10% missingness over DEFINED (predicted_k≥2) partitions", ""]
        lines.append(_md_table(["metric_id", "null_ratio(defined)", "null_ratio(valid)", "note"],
            [[r["metric_id"], r["null_ratio_defined"], r["null_ratio_valid"], _KNOWN_DEAD.get(r["metric_id"], "")]
             for r in im["high_missingness_over_10pct_defined"]]))
        if any(r["metric_id"] in _KNOWN_DEAD for r in im["high_missingness_over_10pct_defined"]):
            lines += ["",
                      "> The fully-NaN metrics above are **known renamed/dead ClustOpt image metrics**: "
                      "the registry carries the new names while the computed values are stored under the "
                      "old names (`convexity_solidity_image`, `hough_circle_arc_strength`). This is a "
                      "pre-existing condition in the ClustOpt repository (not introduced by the merge) and "
                      "is not used by the ML2DAC/AutoClust derivations, so it does not block readiness."]
    else:
        lines += ["", "_No internal metric exceeds 10% missingness over defined partitions._"]

    lines += ["", "## Label audit", "",
              f"- Existence check: **{lab['existence_check']}** over {lab['n_valid_evaluations']:,} valid evaluations",
              f"- Full-existence missing files: **{lab['full_existence_missing_files']}**",
              f"- Valid rows missing labels_path: {lab['valid_missing_labels_path']}",
              f"- Load+hash mode: **{lab['load_hash_mode']}** (checked {lab['load_hash_checked']:,})",
              f"- Load failures / length mismatch / hash mismatch: "
              f"{lab['load_failures']} / {lab['length_mismatch']} / {lab['hash_mismatch']}",
              f"- Result: {_yn(lab['pass'])}", ""]

    lines += ["## Storage", ""]
    lines.append(_md_table(["table", "MB"], [[k, v] for k, v in st["table_mb"].items()]))
    lines += ["", f"- Total parquet size: **{st['total_parquet_mb']} MB**",
              f"- Labels duplicated in master: {st['labels_duplicated_in_master']} "
              f"({st['labels_note']})", ""]

    lines += ["## Readiness decision", "",
              f"**Is the Master Unified Repository ready for deriving ML2DAC and AutoClust MKRs? "
              f"{'YES' if rd['ready'] else 'NO'}**", ""]
    if not rd["ready"]:
        lines += ["Blockers:", ""] + [f"- {b}" for b in rd["blockers"]]
    return "\n".join(lines)


# ===========================================================================
# Orchestration
# ===========================================================================

def write_reports(storage: MasterStorage, audit: dict, merge: master_merge.MergeResult) -> None:
    # validation JSONs
    storage.write_validation("schema_consistency_report.json", _clean(merge.schema_report))
    storage.write_validation("registry_consistency_report.json", _clean(merge.registry_report))
    storage.write_validation("configuration_consistency_report.json", _clean(merge.configuration_report))
    storage.write_validation("integrity_report.json", _clean({
        "counts": audit["counts"],
        "primary_keys": audit["primary_keys"],
        "config_coverage": audit["config_coverage"],
        "split_distribution": audit["split_distribution"],
    }))
    storage.write_validation("feature_coverage_report.json", _clean({
        "ml2dac_features": audit["ml2dac_features"],
        "autoclust_features": audit["autoclust_features"],
    }))
    storage.write_validation("metric_coverage_report.json", _clean({
        "external_metrics": audit["external_metrics"],
        "internal_metrics": audit["internal_metrics"],
        "ml2dac_cvi": audit["ml2dac_cvi"],
        "autoclust_cvi": audit["autoclust_cvi"],
        "clustopt_metrics": audit["clustopt_metrics"],
    }))
    storage.write_validation("failure_report.json", _clean(audit["failure_analysis"]))

    # markdown reports
    storage.write_report("MASTER_SCHEMA_AUDIT.md", render_schema_audit(merge))
    storage.write_report("MASTER_COVERAGE_REPORT.md", render_coverage(audit))
    storage.write_report("MASTER_SPLIT_DISTRIBUTION.md", render_split(audit))
    storage.write_report("MASTER_FAILURE_ANALYSIS.md", render_failures(audit))
    storage.write_report("MASTER_READINESS_REPORT.md", render_readiness(audit, merge))
    storage.write_report("MASTER_READINESS_REPORT.json",
                         __import__("json").dumps(_clean({**audit, "merge_counts": merge.merged_counts}), indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build Master Unified Repository + Readiness Audit")
    ap.add_argument("--outputs-root", required=True)
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="discover + consistency checks only; no writes")
    ap.add_argument("--validate-only", action="store_true", help="audit an already-built master; no merge")
    ap.add_argument("--strict", action="store_true", help="abort merge on any schema/catalogue mismatch")
    ap.add_argument("--label-mode", choices=["full", "sampled"], default="sampled")
    ap.add_argument("--label-sample", type=int, default=3000)
    args = ap.parse_args(argv)

    repo_root = io_utils.find_repo_root()
    io_utils.ensure_repo_on_path(repo_root)
    io_utils.ensure_ml2dac_on_path(repo_root)

    outputs_root = Path(args.outputs_root)
    master_root = Path(args.master_root)
    storage = MasterStorage(master_root)

    # ---- validate-only: audit existing master ---------------------------
    if args.validate_only:
        _log("validate-only: auditing existing master")
        reg = storage.read_table("metric_registry")
        cfg = storage.read_table("configurations")
        if reg is None or cfg is None:
            _log("ERROR: master tables not found; run a merge first")
            return 2
        audit = master_audit.run_audit(
            master_root=master_root, outputs_root=outputs_root,
            metric_registry=reg, configurations=cfg,
            label_mode=args.label_mode, label_sample=args.label_sample, log=_log)
        # rebuild a thin merge result for report headers
        subs = master_merge.discover_subfamilies(outputs_root)
        merge = master_merge.MergeResult(subfamilies=subs)
        merge.schema_report = io_utils.read_json(storage.validation_dir / "schema_consistency_report.json") or {
            "schema_versions": [], "schema_version_consistent": True, "tables": {}}
        merge.registry_report = io_utils.read_json(storage.validation_dir / "registry_consistency_report.json") or {
            "identical_across_subfamilies": True, "n_distinct_fingerprints": 1,
            "n_rows_canonical": len(reg), "n_cols_canonical": reg.shape[1], "canonical_fingerprint": ""}
        merge.configuration_report = io_utils.read_json(storage.validation_dir / "configuration_consistency_report.json") or {
            "identical_across_subfamilies": True, "n_distinct_fingerprints": 1,
            "n_rows_canonical": len(cfg), "n_cols_canonical": cfg.shape[1], "canonical_fingerprint": ""}
        write_reports(storage, audit, merge)
        _print_summary(audit, merge)
        return 0 if audit["readiness"]["ready"] else 1

    # ---- discovery -------------------------------------------------------
    subs = master_merge.discover_subfamilies(outputs_root)
    _log(f"discovered {len(subs)} subfamilies under {outputs_root}")
    missing_any = []
    for sf in subs:
        miss = master_merge.check_required_files(sf)
        if miss:
            missing_any.append((sf.key, miss))
    if missing_any:
        _log(f"WARNING: {len(missing_any)} subfamilies missing required files:")
        for key, miss in missing_any[:20]:
            _log(f"   {key}: {miss}")
        if args.strict:
            _log("STRICT: aborting due to missing files")
            return 2

    # ---- overwrite guard -------------------------------------------------
    if not args.dry_run:
        existing = storage.table_path("evaluations")
        if existing.exists() and not args.overwrite:
            _log(f"ERROR: master already exists at {master_root}; pass --overwrite")
            return 2
        storage.ensure_dirs()

    # ---- merge -----------------------------------------------------------
    def _writer(table, df):
        return storage.write_table(table, df)

    merge = master_merge.run_merge(
        subfamilies=subs, write_table=_writer, outputs_root=outputs_root,
        strict=args.strict, dry_run=args.dry_run, log=_log)

    if merge.aborted:
        _log(f"MERGE ABORTED (strict): {merge.abort_reason}")
        # still write a schema audit so the user can see why
        if not args.dry_run:
            storage.ensure_dirs()
            storage.write_validation("schema_consistency_report.json", _clean(merge.schema_report))
            storage.write_validation("registry_consistency_report.json", _clean(merge.registry_report))
            storage.write_validation("configuration_consistency_report.json", _clean(merge.configuration_report))
            storage.write_report("MASTER_SCHEMA_AUDIT.md", render_schema_audit(merge))
        return 3

    if args.dry_run:
        _log("dry-run complete (no writes). Consistency:")
        _log(f"  schema_version_consistent={merge.schema_report['schema_version_consistent']}")
        _log(f"  registry_identical={merge.registry_report['identical_across_subfamilies']}")
        _log(f"  configurations_identical={merge.configuration_report['identical_across_subfamilies']}")
        return 0

    # ---- master metadata -------------------------------------------------
    storage.write_json("schema_version.json", {"schema_version": merge.schema_version})
    storage.write_json("master_build_config.json", _clean({
        "created_at": _now(),
        "outputs_root": str(outputs_root),
        "master_root": str(master_root),
        "labels_root": str(outputs_root),
        "labels_path_convention": "<family>/<subfamily>/labels/<record_id>/<config_id>.npy relative to labels_root",
        "n_subfamilies_discovered": len(subs),
        "n_subfamilies_merged": len(subs),
        "schema_version": merge.schema_version,
        "merged_counts": merge.merged_counts,
        "subfamilies": [
            {"family": sf.family, "subfamily": sf.subfamily,
             "n_datasets": (io_utils.read_json(sf.dir / "build_config.json") or {}).get("n_datasets")}
            for sf in subs
        ],
    }))

    # ---- audit -----------------------------------------------------------
    audit = master_audit.run_audit(
        master_root=master_root, outputs_root=outputs_root,
        metric_registry=merge.canonical_metric_registry,
        configurations=merge.canonical_configurations,
        label_mode=args.label_mode, label_sample=args.label_sample, log=_log)

    write_reports(storage, audit, merge)
    _print_summary(audit, merge)
    return 0 if audit["readiness"]["ready"] else 1


def _print_summary(audit: dict, merge: master_merge.MergeResult) -> None:
    c = audit["counts"]
    rd = audit["readiness"]
    _log("================ MASTER SUMMARY ================")
    _log(f"subfamilies merged   : {len(merge.subfamilies)}")
    _log(f"merged counts        : {merge.merged_counts}")
    _log(f"datasets/records/eval: {c['datasets']} / {c['records']} / {c['evaluations']}")
    _log(f"counts match expected: ds={c['datasets_match_expected']} "
         f"rec={c['records_match_expected']} eval={c['evaluations_match_expected']}")
    _log(f"schema consistent    : {merge.schema_report['schema_version_consistent']} / "
         f"registry={merge.registry_report['identical_across_subfamilies']} / "
         f"config={merge.configuration_report['identical_across_subfamilies']}")
    _log(f"ML2DAC feat readiness: {rd['ml2dac_feature_readiness']}")
    _log(f"AutoClust feat ready : {rd['autoclust_feature_readiness']}")
    _log(f"external/ml2dac/autoclust/clustopt CVI: "
         f"{rd['external_metrics_readiness']}/{rd['ml2dac_cvi_readiness']}/"
         f"{rd['autoclust_cvi_readiness']}/{rd['clustopt_metric_readiness']}")
    lab = audit["label_audit"]
    _log(f"label audit          : existence=full missing={lab['full_existence_missing_files']} "
         f"load_hash={lab['load_hash_mode']} pass={lab['pass']}")
    _log(f"real failures        : {c['real_failures_clustering_stage']} (expected {c['expected']['real_failures']}) "
         f"+ {c['metafeature_stage_rows']} cosmetic metafeature-stage rows")
    _log(f"READY FOR DERIVED MKRs: {'YES' if rd['ready'] else 'NO'}")
    if rd["blockers"]:
        _log(f"BLOCKERS: {rd['blockers']}")
    _log("===============================================")


if __name__ == "__main__":
    raise SystemExit(main())
