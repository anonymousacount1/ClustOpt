"""
run_phase_e1.py
===============

End-to-end Phase-E1 orchestrator (plan §28).  Stages, each resumable:

    load + audit -> LOSO 96-config search -> rank + one-SE select
    -> build frozen 3-view model -> reload/verify
    -> OOF predictions (selected) -> locked split-1 eval -> MLP comparison
    -> tables + plots + reports + validation + git-friendly artefacts

Run:
    python -u -m models.metric_utility_knn.run_phase_e1
Options:
    --skip-plots        skip figure generation
    --limit-rows N      debug on a subset (not for the real run)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from .configuration_search import run_configuration_search
from .configuration_selection import (aggregate_composite, apply_one_se_rule,
                                       compute_fold_composite,
                                       selected_knn_config)
from .cross_validation import assert_split1_absent, build_loso_folds
from .data_loader import load_unified_data, UnifiedData
from .final_model_builder import (build_final_model, oof_predictions_for_config,
                                   reload_and_verify)
from .holdout_evaluation import (evaluate_split1, load_mlp_reference,
                                 mlp_comparison_table)
from .metrics import (compute_full_suite, dynamic_k_confusion,
                      dynamic_k_metrics, per_metric_diagnostics)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def grouped_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                    group_name: str, metric_names: List[str],
                    min_records: int = 1) -> pd.DataFrame:
    rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        if int(m.sum()) < min_records:
            continue
        suite = compute_full_suite(y_true[m], y_pred[m], metric_names)
        rows.append({group_name: g, "n_records": int(m.sum()), **suite})
    return pd.DataFrame(rows)


def _validate_predictions(y_true, y_pred, tag: str) -> Dict:
    finite = bool(np.all(np.isfinite(y_pred)))
    in_range = bool(np.all((y_pred >= -1e-9) & (y_pred <= 1 + 1e-9)))
    shape_ok = bool(y_pred.shape == y_true.shape)
    return {
        f"{tag}_finite": finite, f"{tag}_in_range": in_range,
        f"{tag}_shape_ok": shape_ok, f"{tag}_n_records": int(y_pred.shape[0]),
        f"{tag}_n_metrics": int(y_pred.shape[1]),
    }


def _neighbor_geometry(model, data: UnifiedData) -> Dict:
    """Neighbour distances / effective counts on split-1 queries (clean, held out)."""
    dists, neffs = [], []
    for view in C.VIEWS:
        m = (data.view == view) & (data.split_id == C.LOCKED_HOLDOUT_SPLIT)
        _, dbg = model.predict_utility(data.X[m], view, return_neighbor_debug=True)
        dists.append(dbg.neighbor_distances)
        neffs.append(dbg.effective_neighbor_count)
    return {
        "neighbor_distances": np.concatenate([d.ravel() for d in dists]),
        "effective_counts": np.concatenate(neffs),
        "summary": {
            "mean_neighbor_distance": float(np.mean(np.concatenate(
                [d.ravel() for d in dists]))),
            "median_neighbor_distance": float(np.median(np.concatenate(
                [d.ravel() for d in dists]))),
            "p90_neighbor_distance": float(np.percentile(np.concatenate(
                [d.ravel() for d in dists]), 90)),
            "mean_effective_neighbor_count": float(np.mean(
                np.concatenate(neffs))),
        },
    }


def _readiness(ctx: Dict) -> Dict:
    checks = {
        "record_id_unique": ctx["data_report"]["record_id_unique"],
        "three_views": len(ctx["data_report"]["views"]) == 3,
        "features_250": ctx["data_report"]["n_features"] == C.N_FEATURE_COLUMNS,
        "targets_60": ctx["data_report"]["n_targets"] == C.N_TARGET_COLUMNS,
        "split1_absent_dev": ctx["dev_audit"]["split1_absent"],
        "all_folds_clean": all(
            a["validation_split_absent_from_train"] and a["dataset_ids_disjoint"]
            and a["split1_absent_from_fold"] for a in ctx["fold_audits"]),
        "final_split1_excluded": not ctx["final_metadata"]["split1_in_training"],
        "reload_identical": ctx["reload_validation"]["reload_identical"],
        "oof_pred_valid": ctx["prediction_validation"]["oof_finite"]
        and ctx["prediction_validation"]["oof_in_range"],
        "split1_pred_valid": ctx["prediction_validation"]["split1_finite"]
        and ctx["prediction_validation"]["split1_in_range"],
        "dynamic_k_in_bounds": ctx["dynamic_k_in_bounds"],
    }
    blockers = [k for k, v in checks.items() if not v]
    return {"ready": len(blockers) == 0, "checks": checks, "blockers": blockers}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase E1 KNN utility predictor.")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--results-root", type=str, default=None,
                    help="Override results root (smoke tests only).")
    args = ap.parse_args(argv)

    dirs = C.result_dirs(Path(args.results_root) if args.results_root else None)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("[e1] loading data ...")
    data = load_unified_data(limit_rows=args.limit_rows)
    print(f"[e1] {data.n_records} records, {data.report['n_features']} features, "
          f"{data.report['n_targets']} metrics")

    # development subset audit (split 1 excluded)
    dev_mask = data.split_mask(C.DEV_SPLITS)
    dev_view = UnifiedData(
        feature_columns=data.feature_columns, target_columns=data.target_columns,
        metric_names=data.metric_names, X=data.X[dev_mask], Y=data.Y[dev_mask],
        dataset_id=data.dataset_id[dev_mask], record_id=data.record_id[dev_mask],
        view=data.view[dev_mask], split_id=data.split_id[dev_mask],
        family=data.family[dev_mask], subfamily=data.subfamily[dev_mask],
        difficulty=data.difficulty[dev_mask], report=data.report,
    )
    dev_audit = assert_split1_absent(dev_view)
    folds = build_loso_folds(data)
    fold_audits = [f.audit for f in folds]
    print(f"[e1] {len(folds)} LOSO folds, leakage audits OK")

    # ---- STAGE: search ---- #
    print("[e1] running/ resuming 96-config LOSO search ...")
    fold_metrics = run_configuration_search(data, dirs["cache"], verbose=True)
    fold_metrics.to_parquet(dirs["cache"] / "all_fold_metrics.parquet", index=False)
    fold_all = fold_metrics[fold_metrics["level"] == "ALL"].reset_index(drop=True)

    # ---- STAGE: rank + select ---- #
    print("[e1] ranking configurations + one-SE selection ...")
    fold_composite = compute_fold_composite(fold_all)
    ranking = aggregate_composite(fold_composite, fold_all)
    selection = apply_one_se_rule(ranking)
    sel_cfg = selected_knn_config(selection)
    print(f"[e1] raw best={selection['raw_best_configuration']['config_id']} | "
          f"one-SE selected={sel_cfg.config_id}")

    # ---- STAGE: final model + reload verify ---- #
    print("[e1] building frozen final model on splits 2-16 ...")
    final_metadata = build_final_model(data, sel_cfg, dirs["models"])
    reload_validation = reload_and_verify(data, sel_cfg, dirs["models"])
    print(f"[e1] reload identical: {reload_validation['reload_identical']}")

    from .knn_predictor import KnnUtilityPredictor
    model = KnnUtilityPredictor.load(dirs["models"], data.feature_columns,
                                     data.metric_names, sel_cfg)

    # ---- STAGE: OOF predictions (selected config) ---- #
    print("[e1] computing out-of-fold predictions for selected config ...")
    oof_pred, oof_true, oof_meta = oof_predictions_for_config(data, sel_cfg)
    oof_suite = compute_full_suite(oof_true, oof_pred, data.metric_names)
    oof_per_metric = per_metric_diagnostics(oof_true, oof_pred, data.metric_names)
    oof_dk, oof_kt, oof_kp = dynamic_k_metrics(oof_true, oof_pred)
    oof_dk_conf = dynamic_k_confusion(oof_kt, oof_kp)

    view_metrics = grouped_metrics(oof_true, oof_pred, oof_meta["view"].to_numpy(),
                                   "view", data.metric_names)
    family_metrics = grouped_metrics(oof_true, oof_pred,
                                     oof_meta["family"].to_numpy(), "family",
                                     data.metric_names, min_records=30)
    subfamily_metrics = grouped_metrics(oof_true, oof_pred,
                                        oof_meta["subfamily"].to_numpy(),
                                        "subfamily", data.metric_names,
                                        min_records=30)

    # per-metric-by-k plot needs per-metric MAE tracked per n_neighbors, which the
    # scalar search does not persist; left as None (plot is skipped gracefully).
    per_metric_by_k = None

    # ---- STAGE: locked split-1 eval ---- #
    print("[e1] locked split-1 evaluation ...")
    s1 = evaluate_split1(data, model)
    s1_suite = s1["suite"]
    s1_dk_conf = dynamic_k_confusion(s1["k_true"], s1["k_pred"])
    mlp_ref = load_mlp_reference()
    mlp_comparison = mlp_comparison_table(s1_suite, mlp_ref)

    # ---- neighbour geometry ---- #
    geom = _neighbor_geometry(model, data)

    # ---- validations ---- #
    pred_validation = {}
    pred_validation.update(_validate_predictions(oof_true, oof_pred, "oof"))
    pred_validation.update(_validate_predictions(s1["y_true"], s1["y_pred"], "split1"))
    dynamic_k_in_bounds = bool(
        oof_kt.min() >= 1 and oof_kt.max() <= 10 and oof_kp.min() >= 1
        and oof_kp.max() <= 10 and s1["k_true"].min() >= 1
        and s1["k_pred"].max() <= 10)

    # ---- selected-config fold summary ---- #
    sel_fold = fold_all[fold_all["config_id"] == sel_cfg.config_id]
    summ_cols = ["mae", "rmse", "r2", "spearman_mean", "kendall_mean",
                 "pairwise_ranking_accuracy", "ndcg@all", "top1_accuracy",
                 "top5_overlap", "top10_overlap", "cosine_similarity_mean",
                 "dynamic_k_exact_accuracy", "dynamic_k_within_1_accuracy"]
    summ_cols = [c for c in summ_cols if c in sel_fold.columns]
    selected_fold_summary = pd.DataFrame({
        "metric": summ_cols,
        "mean": [sel_fold[c].mean() for c in summ_cols],
        "std": [sel_fold[c].std() for c in summ_cols],
    })

    runtime_summary = {
        "total_fit_time_sec": float(fold_all["fit_time_sec"].sum()),
        "total_query_time_sec": float(fold_all["query_time_sec"].sum()),
        "total_prediction_time_sec": float(fold_all["prediction_time_sec"].sum()),
        "mean_records_per_second": float(fold_all["records_per_second"].mean()),
    }

    dynamic_k_rows = [
        {"scope": "development_oof", **oof_dk},
        {"scope": "split1_locked", **{k: v for k, v in s1_suite.items()
                                      if k.startswith("dynamic_k")}},
    ]

    # ---- assemble context ---- #
    ctx = {
        "data_report": data.report,
        "feature_columns": data.feature_columns,
        "metric_names": data.metric_names,
        "dev_audit": dev_audit, "fold_audits": fold_audits,
        "fold_metrics_all": fold_all,
        "fold_metrics_all_levels": fold_metrics,
        "fold_composite": fold_composite,
        "ranking": ranking, "selection": selection,
        "selected_id": sel_cfg.config_id,
        "final_metadata": final_metadata, "reload_validation": reload_validation,
        "oof_pred": oof_pred, "oof_true": oof_true, "oof_meta": oof_meta,
        "oof_suite": oof_suite, "per_metric": oof_per_metric,
        "oof_dynamic_k_confusion": oof_dk_conf,
        "oof_k_true": oof_kt, "oof_k_pred": oof_kp,
        "view_metrics": view_metrics, "family_metrics": family_metrics,
        "subfamily_metrics": subfamily_metrics,
        "per_metric_by_k": per_metric_by_k,
        "split1_suite": s1_suite, "split1_per_view": s1["per_view"],
        "split1_dynamic_k_confusion": s1_dk_conf,
        "mlp_ref": mlp_ref, "mlp_comparison": mlp_comparison,
        "neighbor_distances": geom["neighbor_distances"],
        "effective_counts": geom["effective_counts"],
        "neighbor_summary": geom["summary"],
        "prediction_validation": pred_validation,
        "dynamic_k_in_bounds": dynamic_k_in_bounds,
        "selected_fold_summary": selected_fold_summary,
        "dynamic_k_rows": dynamic_k_rows,
        "runtime_summary": runtime_summary,
    }
    ctx["readiness"] = _readiness(ctx)
    print(f"[e1] readiness: {'YES' if ctx['readiness']['ready'] else 'NO'} "
          f"blockers={ctx['readiness']['blockers']}")

    # ---- write predictions parquet ---- #
    print("[e1] writing predictions ...")
    pd.DataFrame(oof_pred, columns=data.target_columns).to_parquet(
        dirs["predictions"] / "out_of_fold_predictions.parquet", index=False)
    pd.DataFrame(oof_true, columns=data.target_columns).to_parquet(
        dirs["predictions"] / "out_of_fold_targets.parquet", index=False)
    oof_meta.to_parquet(
        dirs["predictions"] / "out_of_fold_record_metadata.parquet", index=False)
    pd.DataFrame(s1["y_pred"], columns=data.target_columns).to_parquet(
        dirs["predictions"] / "split1_locked_predictions.parquet", index=False)
    pd.DataFrame(s1["y_true"], columns=data.target_columns).to_parquet(
        dirs["predictions"] / "split1_locked_targets.parquet", index=False)
    s1["meta"].to_parquet(
        dirs["predictions"] / "split1_locked_record_metadata.parquet", index=False)

    # ---- tables / reports / validation / metadata ---- #
    from . import reporting
    print("[e1] writing tables + reports + validation ...")
    reporting.write_tables(ctx, dirs["tables"])
    reporting.write_validation(ctx, dirs["validation"])
    reporting.write_run_metadata(ctx, dirs["root"])
    reporting.write_reports(ctx, dirs["reports"])
    (dirs["validation"] / "neighbor_geometry.json").write_text(
        json.dumps(geom["summary"], indent=2))

    # ---- plots ---- #
    if not args.skip_plots:
        from . import plotting
        print("[e1] generating plots ...")
        produced = plotting.generate_all_plots(ctx, dirs["plots"])
        print(f"[e1] {len(produced)} plots written")

    _write_root_readme(ctx, dirs["root"])
    print("[e1] DONE. readiness =", "YES" if ctx["readiness"]["ready"] else "NO")
    return 0


def _write_root_readme(ctx: Dict, root: Path) -> None:
    sel = ctx["selection"]["one_se_selected_configuration"]
    (root / "README.md").write_text(f"""# Phase E1 — KNN Metric-Utility Predictor (results)

Leakage-free KNN predictor of metric-utility vectors from dataset-view
meta-features, selected by 15-fold LOSO CV over splits 2-16 and evaluated once on
the locked split-1 holdout.

## Selected configuration
- **{sel['config_id']}** — n_neighbors={sel['n_neighbors']}, scaler={sel['scaler']},
  distance={sel['distance']}, weighting={sel['neighbor_weighting']}
- Composite (LOSO): {sel['mean_composite_score']:.4f} +/- {sel['standard_error']:.4f} SE

## Readiness for Phase E2: **{'YES' if ctx['readiness']['ready'] else 'NO'}**

## Layout
- `reports/`  — 8 markdown reports (start with PHASE_E1_FINAL_REPORT.md)
- `tables/`   — all CSV tables
- `predictions/` — OOF + locked split-1 predictions/targets/metadata (parquet)
- `models/`   — frozen 3-view model + schemas + manifest + hashes
- `plots/`    — all figures
- `validation/` — leakage/coverage/alignment/prediction/final-model JSONs
- `cache/`    — per-fold search metrics (reproducible; git-ignored)

Reproduce: `python -u -m models.metric_utility_knn.run_phase_e1`
""")


if __name__ == "__main__":
    sys.exit(main())
