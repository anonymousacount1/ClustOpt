"""Build the derived AutoClust repositories (original-CVI + extended-CVI).

Split policy (corrected): train splits 2..16, hold out split 1, exclude nothing.
Split 1 is never used for algorithm labels, the algorithm selector, the ARI
predictor, or any preprocessing fit. Mirrors AutoClust
(Experiments/RelatedWork/AutoClust.py): KDTree-NN algorithm selector over
MeanShift landmarking meta-features + an MLPRegressor(60,30,10) CVI->ARI surrogate.
"""
from __future__ import annotations

import argparse
import datetime as _dt
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
from unified_mkr import autoclust_derived as acd  # noqa: E402
from unified_mkr import autoclust_models as acm  # noqa: E402
from unified_mkr import autoclust_training_data as actd  # noqa: E402

ML2DAC_CVIS = {"calinski_harabasz", "davies_bouldin", "silhouette", "dbcv",
               "dunn_index", "coggins_jain_index", "cop"}


def _log(msg: str) -> None:
    print(f"[autoclust-derived] {msg}", flush=True)


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _original_candidates(reg: pd.DataFrame) -> List[str]:
    return list(reg[reg["used_by_ml2dac"] == True]["metric_id"])  # noqa: E712


def _extended_pool(reg: pd.DataFrame) -> List[str]:
    by_id = reg.set_index("metric_id")
    ml2dac = set(reg[reg["used_by_ml2dac"] == True]["metric_id"])  # noqa: E712
    pool = []
    for mid in reg["metric_id"]:
        if mid in ml2dac or bool(by_id.loc[mid, "used_by_clustopt"]):
            if mid not in pool:
                pool.append(mid)
    return pool


# --- Stage-3A1: explicit frozen metric-subset support (additive) -------------
def _load_variant_spec(path: str, registry: pd.DataFrame) -> Dict[str, Any]:
    """Load and validate a frozen Stage-3A1 variant spec.

    The spec is the scientific authority for WHICH metrics the variant requests;
    the builder remains the authority for canonical resolution, alias handling,
    all-NaN filtering, direction validation, split filtering, model fitting and
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
    """Requested ids in master-registry column order.

    The spec stores the requested ids as a sorted *set*; the builder fixes the
    order so a subset variant lays out its CVI columns exactly the way the
    historical extended build did.
    """
    want = set(spec["requested_metric_ids"])
    return [m for m in registry["metric_id"] if m in want]


def _exclude_all_nan(pool, train_eval) -> Dict[str, str]:
    valid = train_eval[train_eval["valid_result"] == True]  # noqa: E712
    out = {}
    for mid in pool:
        if mid not in train_eval.columns or valid[mid].isna().all():
            out[mid] = "100% NaN on train (dead/renamed or unavailable metric)"
    return out


def _md_table(h, rows):
    return "\n".join(["| " + " | ".join(map(str, h)) + " |",
                      "| " + " | ".join("---" for _ in h) + " |"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def _fmt(x, n=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{n}f}" if isinstance(x, float) else str(x)


def _write_reports(reports: Path, orig, ext, policy, ready, blockers):
    # comparison
    ov, ev = orig["predictor_meta"]["validation_scores"], ext["predictor_meta"]["validation_scores"]
    sm = orig["selector_meta"]
    L = ["# AutoClust Original vs Extended Comparison", "", f"_Generated {_now()}_", "",
         "## Candidate metrics", "",
         _md_table(["variant", "included", "excluded"], [
             ["original", int(orig["candidates"]["included"].sum()), int((~orig["candidates"]["included"]).sum())],
             ["extended", int(ext["candidates"]["included"].sum()), int((~ext["candidates"]["included"]).sum())]]),
         "", "## ARI predictor validation (grouped 20% holdout)", "",
         _md_table(["metric", "original", "extended"], [
             ["MAE", _fmt(ov["mae"]), _fmt(ev["mae"])],
             ["RMSE", _fmt(ov["rmse"]), _fmt(ev["rmse"])],
             ["R2", _fmt(ov["r2"]), _fmt(ev["r2"])],
             ["Spearman(pred,true)", _fmt(ov["spearman_pred_vs_true"]), _fmt(ev["spearman_pred_vs_true"])]]),
         "", f"- Extended improves Spearman: **{ev['spearman_pred_vs_true'] > ov['spearman_pred_vs_true']}** "
         f"(Δ {ev['spearman_pred_vs_true'] - ov['spearman_pred_vs_true']:+.4f})",
         f"- Extended improves R2: **{ev['r2'] > ov['r2']}** (Δ {ev['r2'] - ov['r2']:+.4f})", "",
         "## Algorithm selector (shared across variants)", "",
         f"- Model: {sm['model_type']}",
         f"- CV accuracy: {sm['cv_accuracy']} · balanced: {sm['cv_balanced_accuracy']}",
         f"- best_algorithm distribution: {sm['target_distribution']}"]
    io_utils.write_text_atomic(reports / "AUTOCLUST_ORIGINAL_VS_EXTENDED_COMPARISON.md", "\n".join(L))

    # regression audit
    R = ["# AutoClust ARI Regression Audit", "", f"_Generated {_now()}_", ""]
    for nm, v in [("original_cvis", orig), ("extended_cvis", ext)]:
        pm = v["predictor_meta"]; vs = pm["validation_scores"]; ri = pm["regression_dataset_info"]
        R += [f"## {nm}", "",
              f"- Candidates (CVI inputs): {int(v['candidates']['included'].sum())}",
              f"- Regression rows: {pm['train_row_count']} (before {ri['n_rows_before']}, "
              f"dropped invalid/NaN-ARI {ri['n_rows_dropped_invalid_or_nan_ari']})",
              f"- MLP: {pm['architecture']['hidden_layer_sizes']} {pm['architecture']['activation']}; "
              f"final training loss {pm['training_loss_final_iter']:.5f}",
              f"- Validation: MAE={_fmt(vs['mae'])} RMSE={_fmt(vs['rmse'])} R2={_fmt(vs['r2'])} "
              f"Spearman={_fmt(vs['spearman_pred_vs_true'])}",
              f"- Preprocessing: imputer={pm['preprocessing']['imputer']}; scaler={pm['preprocessing']['scaler']}", ""]
    io_utils.write_text_atomic(reports / "AUTOCLUST_REGRESSION_AUDIT.md", "\n".join(R))

    # leakage audit
    LA = ["# AutoClust Derived Repository — Leakage Audit", "", f"_Generated {_now()}_", "",
          f"- Heldout test split: **{policy['heldout_test_split']}**",
          f"- Train splits: {policy['train_splits']}",
          f"- Excluded splits: {policy['excluded_splits'] or 'none'}", ""]
    for nm, v in [("original_cvis", orig), ("extended_cvis", ext)]:
        lk = v["validation"]["leakage"]
        LA += [f"## {nm}", "",
               f"- train records contain split 1: {lk['train_records_contain_heldout']}",
               f"- train evaluations contain split 1: {lk['train_evaluations_contain_heldout']}",
               f"- algorithm labels contain split 1: {lk['algorithm_labels_contain_heldout']}",
               f"- regression rows contain split 1: {lk['regression_rows_contain_heldout']}",
               f"- selector trained on split 1: {lk['selector_trained_on_heldout']}",
               f"- train contains excluded split 16: {lk['train_contains_excluded_split16']}",
               f"- **no leakage: {lk['no_leakage']}**", ""]
    io_utils.write_text_atomic(reports / "AUTOCLUST_LEAKAGE_AUDIT.md", "\n".join(LA))

    # summary
    def blk(v):
        c = v["validation"]["counts"]; pm = v["predictor_meta"]["validation_scores"]
        return [f"- train records: {c['n_train_records']}",
                f"- train evaluations: {c['n_train_evaluations']}",
                f"- candidate CVIs: {c['n_cvi_candidates']}",
                f"- algorithm labels: {c['n_algorithm_labels']}",
                f"- regression rows: {c['n_regression_rows']}",
                f"- ARI predictor: MLP(60,30,10) MAE={_fmt(pm['mae'])} R2={_fmt(pm['r2'])} "
                f"Spearman={_fmt(pm['spearman_pred_vs_true'])}",
                f"- validation all pass: {v['validation']['all_checks_pass']}"]
    S = ["# AutoClust Derived Repository Summary", "", f"_Generated {_now()}_", "",
         f"## Readiness: {'YES' if ready else 'NO'}", ""]
    if blockers:
        S += ["Blockers:", ""] + [f"- {b}" for b in blockers] + [""]
    S += ["## Split policy", "",
          f"- heldout: {policy['heldout_test_split']} · train: {policy['train_splits']} · excluded: {policy['excluded_splits'] or 'none'}",
          "", "## Algorithm selector (shared)", "",
          f"- {orig['selector_meta']['model_type']}; CV acc={orig['selector_meta']['cv_accuracy']}, "
          f"bal={orig['selector_meta']['cv_balanced_accuracy']}", "",
          "## Original-CVI repository", ""] + blk(orig)
    S += ["", "## Extended-CVI repository", ""] + blk(ext)
    io_utils.write_text_atomic(reports / "AUTOCLUST_DERIVED_REPOSITORY_SUMMARY.md", "\n".join(S))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build derived AutoClust repositories")
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--heldout-split", type=int, default=1)
    ap.add_argument("--train-splits", type=int, nargs="+", default=list(range(2, 17)))
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

    records = io_utils.read_parquet(master / "records.parquet")
    metafeatures = io_utils.read_parquet(master / "metafeatures.parquet")
    evaluations = io_utils.read_parquet(master / "evaluations.parquet")
    registry = io_utils.read_parquet(master / "metric_registry.parquet")
    if any(x is None for x in (records, metafeatures, evaluations, registry)):
        _log("ERROR: could not read master tables")
        return 2

    heldout = args.heldout_split
    train_splits = sorted(set(args.train_splits))
    all_splits = sorted(int(s) for s in records["split_id"].dropna().unique())
    excluded = [s for s in all_splits if s not in train_splits and s != heldout]
    policy = {"heldout_test_split": heldout, "train_splits": train_splits,
              "excluded_splits": excluded, "all_splits_present": all_splits,
              "random_seed": args.random_seed}

    # ---- HARD assertions (corrected protocol: train 2..16, heldout 1) ----
    assert 1 not in train_splits, "split 1 must never be a training split"
    assert heldout not in train_splits
    assert 16 in train_splits, "split 16 must be included in training (corrected protocol)"

    train_records = records[records["split_id"].isin(train_splits)].copy()
    train_ids = set(train_records["record_id"])
    train_eval = evaluations[evaluations["split_id"].isin(train_splits)].copy()
    train_mf = metafeatures[metafeatures["record_id"].isin(train_ids)].copy()

    assert (train_records["split_id"] != heldout).all()
    assert (train_eval["split_id"] != heldout).all()
    assert (train_mf["split_id"] != heldout).all()
    _log(f"train: {len(train_records):,} records / {len(train_eval):,} evaluations "
         f"(splits {train_splits}, heldout {heldout}, excluded {excluded})")

    if args.validate_only:
        _log("validate-only: re-run without it to (re)build")
        return 0
    spec = None
    variant_name = None
    if args.candidate_metrics_file:
        spec = _load_variant_spec(args.candidate_metrics_file, registry)
        variant_name = args.variant_name or spec["variant_name"]
        guard_glob = f"{variant_name}/ari_predictor.pkl"
    else:
        guard_glob = "*/ari_predictor.pkl"
    if out_root.exists() and any(out_root.glob(guard_glob)) and not args.overwrite:
        _log(f"ERROR: output exists at {out_root}/{guard_glob}; pass --overwrite")
        return 2

    io_utils.ensure_dir(out_root)
    io_utils.ensure_dir(out_root / "reports")
    _bc = _clean({
        "created_at": _now(), "master_root": str(master), "output_root": str(out_root),
        "random_seed": args.random_seed, "split_policy": policy,
        "autoclust_source": "external/ml2dac/src/Experiments/RelatedWork/AutoClust.py (logic mirrored)",
    })
    if spec is None:
        io_utils.write_json_atomic(out_root / "split_policy.json", _clean(policy))
        io_utils.write_json_atomic(out_root / "build_config.json", _bc)
    else:
        # Subset path: the root already records the shared master + split policy
        # for the historical variants; never clobber it. Provenance for the new
        # variant is written inside the variant's own directory.
        _existing = io_utils.read_json(out_root / "split_policy.json")
        if _existing and _existing.get("train_splits") != policy["train_splits"]:
            _log("ERROR: existing split_policy.json disagrees with this build")
            return 2

    # ---- shared: AutoClust meta-features, algorithm labels + selector ----
    ac_cols = actd.autoclust_feature_columns(train_mf)
    _log(f"autoclust landmarking features: {ac_cols}")
    algo_labels = actd.build_algorithm_labels(train_eval, train_records)
    _log(f"building shared algorithm selector (k=1 NN) on {len(algo_labels)} labels")
    selector_pipe, selector_meta = acm.train_algorithm_selector(
        train_mf, algo_labels, ac_cols, args.random_seed)

    # ---- candidate sets ----
    common = dict(train_records=train_records, train_evaluations=train_eval,
                  train_metafeatures=train_mf, metric_registry=registry,
                  autoclust_feature_cols=ac_cols, algorithm_labels=algo_labels,
                  selector_pipe=selector_pipe, selector_meta=selector_meta,
                  split_policy=policy, seed=args.random_seed, log=_log)

    if spec is not None:
        req_ids = _requested_in_registry_order(spec, registry)
        sub_excluded = _exclude_all_nan(req_ids, train_eval)
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
        v = acd.build_variant(variant=variant_name, out_dir=out_root / variant_name,
                              candidate_ids=sub_ids, excluded=sub_excluded,
                              variant_spec=spec, **common)
        io_utils.write_json_atomic(
            out_root / variant_name / "build_config.json",
            {**_bc, "variant_name": variant_name,
             "variant_spec_path": str(args.candidate_metrics_file),
             "variant_spec_hash": spec["manifest_hash"]})
        ok = bool(v["validation"]["all_checks_pass"]
                  and v["validation"]["leakage"]["no_leakage"])
        pm = v["predictor_meta"]["validation_scores"]
        _log("================ DERIVED AUTOCLUST VARIANT ================")
        _log(f"{variant_name}: cvis={int(v['candidates']['included'].sum())} "
             f"reg_rows={v['n_regression_rows']} MAE={pm['mae']:.4f} "
             f"R2={pm['r2']:.4f} Spearman={pm['spearman_pred_vs_true']:.4f} "
             f"leak_free={v['validation']['leakage']['no_leakage']} "
             f"pass={v['validation']['all_checks_pass']}")
        _log(f"READY FOR SPLIT-1 TEST: {'YES' if ok else 'NO'}")
        _log("==========================================================")
        return 0 if ok else 1

    orig_ids = _original_candidates(registry)
    ext_pool = _extended_pool(registry)
    ext_excluded = _exclude_all_nan(ext_pool, train_eval)
    ext_ids = [m for m in ext_pool if m not in ext_excluded]
    _log(f"original candidates: {len(orig_ids)}; extended: {len(ext_ids)} "
         f"(excluded {len(ext_excluded)}: {list(ext_excluded)})")

    orig = acd.build_variant(variant="original_cvis", out_dir=out_root / "original_cvis",
                             candidate_ids=orig_ids, excluded={}, **common)
    ext = acd.build_variant(variant="extended_cvis", out_dir=out_root / "extended_cvis",
                            candidate_ids=ext_ids, excluded=ext_excluded, **common)

    blockers = []
    for v in (orig, ext):
        if not v["validation"]["all_checks_pass"]:
            blockers.append(f"{v['variant']}: validation failed")
        if not v["validation"]["leakage"]["no_leakage"]:
            blockers.append(f"{v['variant']}: leakage detected")
    ready = len(blockers) == 0
    _write_reports(out_root / "reports", orig, ext, policy, ready, blockers)

    _log("================ DERIVED AUTOCLUST SUMMARY ================")
    _log(f"algorithm selector: {selector_meta['model_type']} "
         f"cv_acc={selector_meta['cv_accuracy']} bal={selector_meta['cv_balanced_accuracy']}")
    for v in (orig, ext):
        pm = v["predictor_meta"]["validation_scores"]
        _log(f"{v['variant']}: cvis={int(v['candidates']['included'].sum())} "
             f"reg_rows={v['n_regression_rows']} MAE={pm['mae']:.4f} R2={pm['r2']:.4f} "
             f"Spearman={pm['spearman_pred_vs_true']:.4f} "
             f"leak_free={v['validation']['leakage']['no_leakage']} pass={v['validation']['all_checks_pass']}")
    _log(f"READY FOR SPLIT-1 TEST: {'YES' if ready else 'NO'}")
    if blockers:
        _log(f"BLOCKERS: {blockers}")
    _log("==========================================================")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
