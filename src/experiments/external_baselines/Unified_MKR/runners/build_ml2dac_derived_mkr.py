"""Build the derived ML2DAC MKRs (original-CVI + extended-CVI) from the master.

Split policy (corrected): train on splits 2..16, hold out split 1 for later
testing, exclude nothing. Split 1 is never used in any training, CVI* labelling,
scaler/imputer fit, NN index, or warmstart construction.

Example::

    python .../runners/build_ml2dac_derived_mkr.py \
      --master-root ".../Unified_MKR/master" \
      --output-root ".../Unified_MKR/derived/ml2dac_split1" \
      --heldout-split 1 --train-splits 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --overwrite
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from unified_mkr import io_utils  # noqa: E402
from unified_mkr import ml2dac_derived as mld  # noqa: E402


def _log(msg: str) -> None:
    print(f"[ml2dac-derived] {msg}", flush=True)


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# --- candidate sets --------------------------------------------------------

def _original_candidates(registry: pd.DataFrame) -> List[str]:
    return list(registry[registry["used_by_ml2dac"] == True]["metric_id"])  # noqa: E712


def _extended_pool(registry: pd.DataFrame) -> List[str]:
    # union of ML2DAC CVIs + ClustOpt metrics, in evaluations column order
    ml2dac = set(registry[registry["used_by_ml2dac"] == True]["metric_id"])  # noqa: E712
    pool: List[str] = []
    for mid in registry["metric_id"]:
        if mid in ml2dac or registry.set_index("metric_id").loc[mid, "used_by_clustopt"]:
            if mid not in pool:
                pool.append(mid)
    return pool


# --- Stage-3A1: explicit frozen metric-subset support (additive) -------------
def _load_variant_spec(path: str, registry: pd.DataFrame) -> Dict[str, Any]:
    """Load and validate a frozen Stage-3A1 variant spec.

    The spec is the scientific authority for WHICH metrics the variant requests;
    the builder remains the authority for canonical resolution, direction
    resolution, all-NaN filtering, split filtering, CVI*/classifier fitting and
    leakage validation.
    """
    import json as _json
    from pathlib import Path as _Path
    from runners.freeze_variant_specs import spec_hash

    spec = _json.loads(_Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != "stage3a1.variant_spec.v1":
        raise SystemExit("unsupported variant-spec schema: %s"
                         % spec.get("schema_version"))
    if spec_hash(spec) != spec.get("manifest_hash"):
        raise SystemExit("variant spec %s fails its own manifest hash" % path)
    req = list(spec["requested_metric_ids"])
    unknown = [m for m in req if m not in set(registry["metric_id"])]
    if unknown:
        raise SystemExit("variant spec requests unknown metric ids: %s" % unknown)
    if len(set(req)) != len(req) or len(req) != int(spec["requested_metric_count"]):
        raise SystemExit("variant spec metric list is not a %d-element set"
                         % int(spec["requested_metric_count"]))
    return spec


def _requested_in_registry_order(spec: Dict[str, Any],
                                 registry: pd.DataFrame) -> List[str]:
    """Requested ids in master-registry column order (see the AutoClust twin)."""
    want = set(spec["requested_metric_ids"])
    return [m for m in registry["metric_id"] if m in want]


def _exclude_all_nan_on_train(pool: List[str], train_eval: pd.DataFrame) -> Dict[str, str]:
    valid = train_eval[train_eval["valid_result"] == True]  # noqa: E712
    excluded: Dict[str, str] = {}
    for mid in pool:
        if mid not in train_eval.columns or valid[mid].isna().all():
            excluded[mid] = "100% NaN on train (dead/renamed or unavailable metric)"
    return excluded


# --- reports ---------------------------------------------------------------

def _dist(cvi_star: pd.DataFrame) -> Dict[str, int]:
    res = cvi_star[cvi_star["resolved"]]
    return {str(k): int(v) for k, v in res["best_cvi"].value_counts().items()}


def _write_comparison(out_dir: Path, orig: dict, ext: dict, cvi_ids: set) -> None:
    od, ed = _dist(orig["cvi_star"]), _dist(ext["cvi_star"])
    ext_clustopt = {k: v for k, v in ed.items() if k not in cvi_ids}
    lines = ["# ML2DAC Original vs Extended Comparison", "", f"_Generated {_now()}_", "",
             "| aspect | original | extended |", "| --- | --- | --- |",
             f"| candidate metrics | {int(orig['candidates']['included'].sum())} | {int(ext['candidates']['included'].sum())} |",
             f"| excluded metrics | {int((~orig['candidates']['included']).sum())} | {int((~ext['candidates']['included']).sum())} |",
             f"| CVI* resolved records | {int(orig['cvi_star']['resolved'].sum())} | {int(ext['cvi_star']['resolved'].sum())} |",
             f"| CVI* unresolved records | {int((~orig['cvi_star']['resolved']).sum())} | {int((~ext['cvi_star']['resolved']).sum())} |",
             f"| classifier CV accuracy | {orig['classifier_meta']['cv_accuracy']} | {ext['classifier_meta']['cv_accuracy']} |",
             f"| classifier CV balanced acc | {orig['classifier_meta']['cv_balanced_accuracy']} | {ext['classifier_meta']['cv_balanced_accuracy']} |",
             f"| classifier #classes | {orig['classifier_meta']['n_classes']} | {ext['classifier_meta']['n_classes']} |",
             "", "## Original CVI* distribution", ""]
    lines += [f"- `{k}`: {v}" for k, v in sorted(od.items(), key=lambda kv: -kv[1])]
    lines += ["", "## Extended CVI* distribution", ""]
    lines += [f"- `{k}`: {v}" for k, v in sorted(ed.items(), key=lambda kv: -kv[1])]
    n_ext_resolved = int(ext["cvi_star"]["resolved"].sum())
    n_clustopt_selected = sum(ext_clustopt.values())
    lines += ["", "## Did extended select ClustOpt structural metrics?", "",
              f"- Records whose CVI* is a ClustOpt structural metric (not one of the 7 ML2DAC CVIs): "
              f"**{n_clustopt_selected}** / {n_ext_resolved} resolved "
              f"({100.0 * n_clustopt_selected / max(1, n_ext_resolved):.1f}%)", ""]
    if ext_clustopt:
        lines += ["Top ClustOpt structural metrics selected as CVI*:", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(ext_clustopt.items(), key=lambda kv: -kv[1])[:20]]
    io_utils.write_text_atomic(out_dir / "ML2DAC_ORIGINAL_VS_EXTENDED_COMPARISON.md", "\n".join(lines))


def _write_leakage_audit(out_dir: Path, orig: dict, ext: dict, policy: dict) -> None:
    lines = ["# ML2DAC Derived MKR — Leakage Audit", "", f"_Generated {_now()}_", "",
             f"- Heldout test split: **{policy['heldout_test_split']}** (never used in training)",
             f"- Train splits: {policy['train_splits']}",
             f"- Excluded splits: {policy['excluded_splits'] or 'none'}", ""]
    for name, v in [("original_cvis", orig), ("extended_cvis", ext)]:
        lk = v["validation"]["leakage"]
        lines += [f"## {name}", "",
                  f"- train records contain split 1: {lk['train_records_contain_heldout']}",
                  f"- train evaluations contain split 1: {lk['train_evaluations_contain_heldout']}",
                  f"- CVI* contains split 1: {lk['cvi_star_contains_heldout']}",
                  f"- warmstarts contain split 1: {lk['warmstarts_contain_heldout']}",
                  f"- NN index contains split 1: {lk['nn_index_contains_heldout']}",
                  f"- classifier trained on split 1: {lk['classifier_trained_on_heldout']}",
                  f"- **no leakage: {lk['no_leakage']}**", ""]
    io_utils.write_text_atomic(out_dir / "ML2DAC_LEAKAGE_AUDIT.md", "\n".join(lines))


def _write_summary(out_dir: Path, orig: dict, ext: dict, policy: dict, ready: bool, blockers: List[str]) -> None:
    def block(v):
        c = v["validation"]["counts"]
        cm = v["classifier_meta"]
        return [f"- train records: {c['n_train_records']}",
                f"- train evaluations: {c['n_train_evaluations']}",
                f"- candidate metrics: {c['n_cvi_candidates']}",
                f"- CVI* resolved / unresolved: {c['n_cvi_star_resolved']} / {c['n_cvi_star_unresolved']}",
                f"- classifier: {cm['classifier_type']} ({cm['n_classes']} classes), "
                f"CV acc={cm['cv_accuracy']}, bal_acc={cm['cv_balanced_accuracy']}",
                f"- NN index: {v['nn_meta']['n_train_records']} records",
                f"- warmstart rows: {v['n_warmstart_rows']}",
                f"- validation all pass: {v['validation']['all_checks_pass']}"]
    lines = ["# ML2DAC Derived MKR Summary", "", f"_Generated {_now()}_", "",
             f"## Readiness: {'YES' if ready else 'NO'}", ""]
    if blockers:
        lines += ["Blockers:", ""] + [f"- {b}" for b in blockers] + [""]
    lines += ["## Split policy", "",
              f"- heldout test split: {policy['heldout_test_split']}",
              f"- train splits: {policy['train_splits']}",
              f"- excluded splits: {policy['excluded_splits'] or 'none'}", "",
              "## Original-CVI MKR", ""] + block(orig)
    lines += ["", "## Extended-CVI MKR", ""] + block(ext)
    io_utils.write_text_atomic(out_dir / "ML2DAC_DERIVED_MKR_SUMMARY.md", "\n".join(lines))


# --- main ------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build derived ML2DAC MKRs from master")
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--heldout-split", type=int, default=1)
    ap.add_argument("--train-splits", type=int, nargs="+",
                    default=list(range(2, 17)))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--random-seed", type=int, default=1234)
    # Stage-3A1 (additive): build ONE variant from a frozen metric-subset
    # spec. Omitting both flags preserves the historical
    # original+extended behaviour exactly.
    ap.add_argument("--candidate-metrics-file", default=None,
                    help="frozen Stage-3A1 variant spec JSON; builds only "
                         "that variant into <output-root>/<variant-name>/")
    ap.add_argument("--variant-name", default=None,
                    help="output subdirectory / variant id (defaults to "
                         "the spec's variant_name)")
    args = ap.parse_args(argv)
    if args.variant_name and not args.candidate_metrics_file:
        raise SystemExit("--variant-name requires --candidate-metrics-file")

    repo_root = io_utils.find_repo_root()
    io_utils.ensure_repo_on_path(repo_root)
    io_utils.ensure_ml2dac_on_path(repo_root)

    master = Path(args.master_root)
    out_root = Path(args.output_root)

    # ---- load master ----
    records = io_utils.read_parquet(master / "records.parquet")
    metafeatures = io_utils.read_parquet(master / "metafeatures.parquet")
    evaluations = io_utils.read_parquet(master / "evaluations.parquet")
    registry = io_utils.read_parquet(master / "metric_registry.parquet")
    if any(x is None for x in (records, metafeatures, evaluations, registry)):
        _log("ERROR: could not read master tables")
        return 2

    # ---- split policy ----
    heldout = args.heldout_split
    train_splits = sorted(set(args.train_splits))
    all_splits = sorted(int(s) for s in records["split_id"].dropna().unique())
    excluded = [s for s in all_splits if s not in train_splits and s != heldout]
    policy = {"heldout_test_split": heldout, "train_splits": train_splits,
              "excluded_splits": excluded, "all_splits_present": all_splits,
              "random_seed": args.random_seed}

    # ---- HARD leakage assertions (corrected protocol: train 2..16, heldout 1) ----
    assert heldout not in train_splits, "heldout split must not be in train_splits"
    assert 1 not in train_splits, "split 1 must never be a training split"
    assert 16 in train_splits, "split 16 must be included in training (corrected protocol)"

    # ---- filter train ----
    train_records = records[records["split_id"].isin(train_splits)].copy()
    train_ids = set(train_records["record_id"])
    train_eval = evaluations[evaluations["split_id"].isin(train_splits)].copy()
    train_mf = metafeatures[metafeatures["record_id"].isin(train_ids)].copy()

    assert (train_records["split_id"] != heldout).all()
    assert (train_eval["split_id"] != heldout).all()
    assert (train_mf["split_id"] != heldout).all()
    assert set(train_eval["record_id"]) <= train_ids
    _log(f"train: {len(train_records):,} records, {len(train_eval):,} evaluations "
         f"(splits {train_splits}, heldout {heldout}, excluded {excluded})")

    if args.validate_only:
        _log("validate-only: not implemented to rebuild; re-run without it to (re)build")
        return 0

    spec = None
    variant_name = None
    if args.candidate_metrics_file:
        spec = _load_variant_spec(args.candidate_metrics_file, registry)
        variant_name = args.variant_name or spec["variant_name"]
        guard_glob = f"{variant_name}/cvi_star.parquet"
    else:
        guard_glob = "*/cvi_star.parquet"
    if out_root.exists() and any(out_root.glob(guard_glob)) and not args.overwrite:
        _log(f"ERROR: output exists at {out_root}/{guard_glob}; pass --overwrite")
        return 2

    io_utils.ensure_dir(out_root)
    io_utils.ensure_dir(out_root / "reports")
    _bc = _clean({
        "created_at": _now(), "master_root": str(master), "output_root": str(out_root),
        "random_seed": args.random_seed, "split_policy": policy,
        "ml2dac_source": "external/ml2dac/src (LearningPhase/ApplicationPhase logic mirrored)",
    })
    if spec is None:
        io_utils.write_json_atomic(out_root / "split_policy.json", _clean(policy))
        io_utils.write_json_atomic(out_root / "build_config.json", _bc)
    else:
        # Subset path: never clobber the root provenance of the historical
        # variants; the new variant records its own inside its directory.
        _existing = io_utils.read_json(out_root / "split_policy.json")
        if _existing and _existing.get("train_splits") != policy["train_splits"]:
            _log("ERROR: existing split_policy.json disagrees with this build")
            return 2

    # ---- candidate sets ----
    if spec is not None:
        req_ids = _requested_in_registry_order(spec, registry)
        sub_excluded = _exclude_all_nan_on_train(req_ids, train_eval)
        sub_ids = [m for m in req_ids if m not in sub_excluded]
        _log(f"variant '{variant_name}': requested {len(req_ids)}, live "
             f"{len(sub_ids)} (excluded {len(sub_excluded)}: {list(sub_excluded)})")
        if len(sub_ids) != int(spec["expected_live_metric_count"]):
            _log(f"ERROR: live count {len(sub_ids)} != frozen expectation "
                 f"{spec['expected_live_metric_count']}")
            return 2
        if sorted(sub_excluded) != sorted(spec["expected_dead_metric_ids"]):
            _log(f"ERROR: dead set {sorted(sub_excluded)} != frozen expectation "
                 f"{sorted(spec['expected_dead_metric_ids'])}")
            return 2
        v = mld.build_variant(
            variant=variant_name, out_dir=out_root / variant_name,
            train_records=train_records, train_evaluations=train_eval,
            train_metafeatures=train_mf, metric_registry=registry,
            candidate_ids=sub_ids, excluded=sub_excluded, split_policy=policy,
            seed=args.random_seed, variant_spec=spec, log=_log)
        io_utils.write_json_atomic(
            out_root / variant_name / "build_config.json",
            {**_bc, "variant_name": variant_name,
             "variant_spec_path": str(args.candidate_metrics_file),
             "variant_spec_hash": spec["manifest_hash"]})
        ok = bool(v["validation"]["all_checks_pass"]
                  and v["validation"]["leakage"]["no_leakage"])
        c, cm = v["validation"]["counts"], v["classifier_meta"]
        _log("================ DERIVED ML2DAC VARIANT ================")
        _log(f"{variant_name}: candidates={c['n_cvi_candidates']} "
             f"live={int(v['candidates']['included'].sum())} "
             f"resolved={c['n_cvi_star_resolved']}/"
             f"{c['n_cvi_star_resolved'] + c['n_cvi_star_unresolved']} "
             f"clf_classes={cm['n_classes']} cv_acc={cm['cv_accuracy']} "
             f"nn={v['nn_meta']['n_train_records']} warm={v['n_warmstart_rows']} "
             f"leak_free={v['validation']['leakage']['no_leakage']} "
             f"pass={v['validation']['all_checks_pass']}")
        _log(f"READY FOR SPLIT-1 TEST: {'YES' if ok else 'NO'}")
        _log("=======================================================")
        return 0 if ok else 1

    orig_ids = _original_candidates(registry)
    ext_pool = _extended_pool(registry)
    ext_excluded = _exclude_all_nan_on_train(ext_pool, train_eval)
    ext_ids = [m for m in ext_pool if m not in ext_excluded]
    cvi_id_set = set(orig_ids)
    _log(f"original candidates: {len(orig_ids)}; extended candidates: {len(ext_ids)} "
         f"(excluded {len(ext_excluded)}: {list(ext_excluded)})")

    # ---- build variants ----
    orig = mld.build_variant(
        variant="original_cvis", out_dir=out_root / "original_cvis",
        train_records=train_records, train_evaluations=train_eval,
        train_metafeatures=train_mf, metric_registry=registry,
        candidate_ids=orig_ids, excluded={}, split_policy=policy,
        seed=args.random_seed, log=_log)
    ext = mld.build_variant(
        variant="extended_cvis", out_dir=out_root / "extended_cvis",
        train_records=train_records, train_evaluations=train_eval,
        train_metafeatures=train_mf, metric_registry=registry,
        candidate_ids=ext_ids, excluded=ext_excluded, split_policy=policy,
        seed=args.random_seed, log=_log)

    # ---- reports ----
    _write_comparison(out_root / "reports", orig, ext, cvi_id_set)
    _write_leakage_audit(out_root / "reports", orig, ext, policy)

    blockers = []
    for v in (orig, ext):
        if not v["validation"]["all_checks_pass"]:
            blockers.append(f"{v['variant']}: validation checks failed")
        if not v["validation"]["leakage"]["no_leakage"]:
            blockers.append(f"{v['variant']}: leakage detected")
    ready = len(blockers) == 0
    _write_summary(out_root / "reports", orig, ext, policy, ready, blockers)

    _log("================ DERIVED ML2DAC SUMMARY ================")
    for v in (orig, ext):
        c = v["validation"]["counts"]
        cm = v["classifier_meta"]
        _log(f"{v['variant']}: candidates={c['n_cvi_candidates']} "
             f"resolved={c['n_cvi_star_resolved']}/{c['n_cvi_star_resolved']+c['n_cvi_star_unresolved']} "
             f"clf_classes={cm['n_classes']} cv_acc={cm['cv_accuracy']} "
             f"nn={v['nn_meta']['n_train_records']} warm={v['n_warmstart_rows']} "
             f"leak_free={v['validation']['leakage']['no_leakage']} pass={v['validation']['all_checks_pass']}")
    _log(f"READY FOR SPLIT-1 TEST: {'YES' if ready else 'NO'}")
    if blockers:
        _log(f"BLOCKERS: {blockers}")
    _log("=======================================================")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
