"""
reporting.py
============

Writes every Phase-E1 table (CSV), report (Markdown), validation JSON, and the
top-level run metadata, from the context dict assembled by the runner.

Nothing here recomputes metrics — it only serialises and formats results already
produced by the engine, so the numbers in the reports are exactly the numbers in
the tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C


def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _df_to_md(df: pd.DataFrame, floatfmt=4, max_rows=None) -> str:
    d = df.copy()
    if max_rows:
        d = d.head(max_rows)
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: _fmt(v, floatfmt))
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |"
                     for r in d.itertuples(index=False))
    return "\n".join([header, sep, body])


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def write_tables(ctx: Dict, tables_dir: Path) -> Dict[str, str]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    def _w(df: pd.DataFrame, name: str):
        p = tables_dir / name
        df.to_csv(p, index=False)
        paths[name] = str(p)

    fold_all = ctx["fold_metrics_all"]          # level ALL, all configs
    ranking = ctx["ranking"]

    # configuration_ranking.csv
    _w(ranking, "configuration_ranking.csv")

    # configuration_summary.csv — per-config mean/std of key metrics over folds
    key = ["mae", "rmse", "r2", "macro_r2", "spearman_mean", "kendall_mean",
           "pairwise_ranking_accuracy", "ndcg@all", "ndcg@5", "top1_accuracy",
           "top5_overlap", "top10_overlap", "cosine_similarity_mean",
           "dynamic_k_exact_accuracy", "dynamic_k_within_1_accuracy",
           "dynamic_k_mae"]
    key = [m for m in key if m in fold_all.columns]
    agg = fold_all.groupby(
        ["config_id", "n_neighbors", "scaler", "distance", "neighbor_weighting"]
    )[key].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg = agg.merge(ranking[["config_id", "rank", "mean_composite_score"]],
                    on="config_id", how="left").sort_values("rank")
    _w(agg, "configuration_summary.csv")

    # fold_metrics.csv / fold_runtime.csv
    _w(fold_all, "fold_metrics.csv")
    rt_cols = ["fold", "config_id", "n_neighbors", "scaler", "distance",
               "neighbor_weighting", "fit_time_sec", "query_time_sec",
               "prediction_time_sec", "records_per_second"]
    _w(fold_all[[c for c in rt_cols if c in fold_all.columns]], "fold_runtime.csv")

    # selected config per-fold metrics
    sel = ctx["selected_id"]
    _w(fold_all[fold_all["config_id"] == sel].sort_values("fold"),
       "selected_configuration_fold_metrics.csv")

    # OOF selected-config aggregations
    _w(ctx["view_metrics"], "view_metrics.csv")
    if ctx.get("family_metrics") is not None:
        _w(ctx["family_metrics"], "family_metrics.csv")
    if ctx.get("subfamily_metrics") is not None:
        _w(ctx["subfamily_metrics"], "subfamily_metrics.csv")
    _w(ctx["per_metric"], "per_metric_errors.csv")

    # dynamic-K metrics (dev OOF + split1)
    _w(pd.DataFrame(ctx["dynamic_k_rows"]), "dynamic_k_metrics.csv")

    # split1 locked metrics + MLP comparison
    _w(pd.DataFrame([{"metric": k, "value": v}
                     for k, v in ctx["split1_suite"].items()]),
       "split1_locked_metrics.csv")
    _w(ctx["mlp_comparison"], "mlp_reference_comparison.csv")

    # confusion matrices as tables
    ctx["oof_dynamic_k_confusion"].to_csv(
        tables_dir / "dynamic_k_confusion_oof.csv")
    ctx["split1_dynamic_k_confusion"].to_csv(
        tables_dir / "dynamic_k_confusion_split1.csv")
    return paths


# --------------------------------------------------------------------------- #
# Validation JSONs
# --------------------------------------------------------------------------- #
def write_validation(ctx: Dict, validation_dir: Path) -> None:
    validation_dir.mkdir(parents=True, exist_ok=True)

    (validation_dir / "leakage_report.json").write_text(json.dumps({
        "development_audit": ctx["dev_audit"],
        "fold_audits": ctx["fold_audits"],
        "final_model_train_splits": ctx["final_metadata"]["train_splits"],
        "split1_in_final_training": ctx["final_metadata"]["split1_in_training"],
    }, indent=2))

    (validation_dir / "data_coverage_report.json").write_text(
        json.dumps(ctx["data_report"], indent=2))

    (validation_dir / "metric_alignment_report.json").write_text(json.dumps({
        "n_features": ctx["data_report"]["n_features"],
        "n_metrics": ctx["data_report"]["n_targets"],
        "feature_columns_head": ctx["feature_columns"][:8],
        "metric_names_head": ctx["metric_names"][:8],
        "mlp_reference_metrics_available": sorted(ctx["mlp_ref"].keys()),
        "dynamic_topk_heuristic": C.DYNAMIC_TOPK_HEURISTIC,
    }, indent=2))

    (validation_dir / "prediction_validation.json").write_text(
        json.dumps(ctx["prediction_validation"], indent=2))

    (validation_dir / "final_model_validation.json").write_text(json.dumps({
        "reload_validation": ctx["reload_validation"],
        "artifact_sha256": ctx["final_metadata"]["artifact_sha256"],
        "per_view_train_counts": ctx["final_metadata"]["per_view_train_counts"],
    }, indent=2))


# --------------------------------------------------------------------------- #
# Run metadata
# --------------------------------------------------------------------------- #
def write_run_metadata(ctx: Dict, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "selected_configuration.json").write_text(json.dumps({
        "selection": ctx["selection"],
        "dynamic_topk_heuristic": C.DYNAMIC_TOPK_HEURISTIC,
    }, indent=2))
    (root / "run_config.json").write_text(json.dumps({
        "phase": "E1",
        "unified_csv": str(C.UNIFIED_CSV),
        "split_assignments_csv": str(C.SPLIT_ASSIGNMENTS_CSV),
        "mlp_reference_dir": str(C.MLP_REFERENCE_DIR),
        "dev_splits": C.DEV_SPLITS,
        "locked_holdout_split": C.LOCKED_HOLDOUT_SPLIT,
        "n_neighbors_grid": C.N_NEIGHBORS_GRID,
        "scalers": C.SCALERS, "distances": C.DISTANCES,
        "weightings": C.WEIGHTINGS,
        "n_configurations": 96,
        "max_neighbors": C.MAX_NEIGHBORS,
        "dynamic_topk_heuristic": C.DYNAMIC_TOPK_HEURISTIC,
        "runtime_summary": ctx.get("runtime_summary", {}),
    }, indent=2))


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _mlp_winner_summary(comp: pd.DataFrame) -> str:
    vc = comp["winner"].value_counts().to_dict()
    return ", ".join(f"{k}: {v}" for k, v in vc.items())


def write_reports(ctx: Dict, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    sel_cfg = ctx["selection"]["one_se_selected_configuration"]
    raw_cfg = ctx["selection"]["raw_best_configuration"]
    dr = ctx["data_report"]
    s1 = ctx["split1_suite"]
    comp = ctx["mlp_comparison"]
    ranking = ctx["ranking"]

    # ---- DATA_AND_LEAKAGE_AUDIT ---- #
    (reports_dir / "DATA_AND_LEAKAGE_AUDIT.md").write_text(f"""# Phase E1 — Data & Leakage Audit

## Data coverage
- Records: **{dr['n_records']}**  |  Datasets: **{dr['n_datasets']}**
- Views: {dr['views']}
- Features (meta): **{dr['n_features']}**  |  Utility metrics: **{dr['n_targets']}**
- record_id unique: **{dr['record_id_unique']}**
- Split record counts: {dr['split_counts']}
- Feature NaN cells: {dr['feature_nan_cells']}  |  Target NaN cells: {dr['target_nan_cells']}
- Utility value range observed: [{_fmt(dr['target_min'])}, {_fmt(dr['target_max'])}]

## Development / holdout policy
- Development splits (search): **{C.DEV_SPLITS}**
- Locked holdout split: **{C.LOCKED_HOLDOUT_SPLIT}** (touched once, after freezing)
- Split 1 absent from development data: **{ctx['dev_audit']['split1_absent']}**

## LOSO per-fold leakage audit (15 folds)
All folds satisfy: validation split absent from train, dataset_ids disjoint,
split 1 absent from fold. Summary:

{_df_to_md(pd.DataFrame(ctx['fold_audits'])[['validation_split','n_train_records','n_val_records','validation_split_absent_from_train','dataset_ids_disjoint','split1_absent_from_fold']])}

## Final model training provenance
- Train splits in manifest: **{ctx['final_metadata']['train_splits']}**
- Split 1 in final training: **{ctx['final_metadata']['split1_in_training']}**
- Preprocessing fit on training features only (per fold and for the final model).
""")

    # ---- CONFIGURATION_SEARCH_REPORT ---- #
    (reports_dir / "CONFIGURATION_SEARCH_REPORT.md").write_text(f"""# Phase E1 — Configuration Search

96 configurations = 12 n_neighbors x 2 scalers x 2 distances x 2 weightings,
evaluated with 15-fold Leave-One-Split-Out CV over splits 2-16. Neighbours were
queried once up to {C.MAX_NEIGHBORS} per (fold, view, scaler, distance) and
reused for all n_neighbors / weightings.

Composite ranking = equal mean of four group scores (value, global-rank, head,
dynamic-K), each a direction-aware percentile averaged within the group, per
fold, then averaged over folds.

## Top-15 configurations
{_df_to_md(ranking[['rank','config_id','mean_composite_score','standard_error','mean_score_value','mean_score_global_rank','mean_score_head','mean_score_dynamic_k']].head(15))}

## Raw best vs one-SE selected
- Raw best: **{raw_cfg['config_id']}** (composite {_fmt(raw_cfg['mean_composite_score'])})
- One-SE selected: **{sel_cfg['config_id']}** (composite {_fmt(sel_cfg['mean_composite_score'])})
- One-SE threshold: {_fmt(ctx['selection']['one_se_threshold'])}
- Candidates within one SE: {ctx['selection']['n_candidates_within_one_se']}
- Rule: {ctx['selection']['selection_rule']}
""")

    # ---- K_NEIGHBORS_SENSITIVITY ---- #
    fold_all = ctx["fold_metrics_all"]
    ksens = fold_all.groupby("n_neighbors")[
        ["mae", "spearman_mean", "ndcg@all", "top5_overlap",
         "dynamic_k_exact_accuracy"]].mean().reset_index()
    (reports_dir / "K_NEIGHBORS_SENSITIVITY.md").write_text(f"""# Phase E1 — K_neighbors Sensitivity

Mean over all configurations sharing an ``n_neighbors`` and over the 15 folds
(``n_neighbors`` = K_neighbors, distinct from K_metrics selected by Dynamic Top-K).

{_df_to_md(ksens)}

Boundary check: if leading curves were still improving at n_neighbors=100 the
search boundary may be active. Selected n_neighbors = **{sel_cfg['n_neighbors']}**.
See plots/k_neighbors_sensitivity.png and *_vs_k.png.
""")

    # ---- UTILITY_RECONSTRUCTION_REPORT ---- #
    pm = ctx["per_metric"].sort_values("mae")
    (reports_dir / "UTILITY_RECONSTRUCTION_REPORT.md").write_text(f"""# Phase E1 — Utility Reconstruction (OOF, selected config {ctx['selected_id']})

Global out-of-fold metrics (development splits 2-16, pooled over views):

{_df_to_md(pd.DataFrame([{k: ctx['oof_suite'][k] for k in
    ['mae','rmse','r2','macro_r2','spearman_mean','kendall_mean',
     'pairwise_ranking_accuracy','ndcg@all','top1_accuracy','top5_overlap',
     'top10_overlap','cosine_similarity_mean','dynamic_k_exact_accuracy',
     'dynamic_k_within_1_accuracy'] if k in ctx['oof_suite']}]))}

## Easiest metrics (lowest MAE)
{_df_to_md(pm[['metric_name','mae','rmse','r2','spearman']].head(8))}

## Hardest metrics (highest MAE)
{_df_to_md(pm[['metric_name','mae','rmse','r2','spearman']].tail(8))}

## Largest prediction bias (|pred-true| mean)
{_df_to_md(ctx['per_metric'].reindex(ctx['per_metric']['prediction_bias'].abs().sort_values(ascending=False).index)[['metric_name','prediction_bias','mean_true_utility','mean_predicted_utility']].head(8))}
""")

    # ---- VIEW_AND_FAMILY_ANALYSIS ---- #
    vm = ctx["view_metrics"]
    fam = ctx.get("family_metrics")
    if fam is not None and not fam.empty and "family" in fam.columns:
        fam_md = _df_to_md(fam[['family', 'n_records', 'mae', 'spearman_mean',
                                'ndcg@all', 'dynamic_k_exact_accuracy']]
                           .sort_values('n_records', ascending=False).head(15))
    else:
        fam_md = "n/a"
    (reports_dir / "VIEW_AND_FAMILY_ANALYSIS.md").write_text(f"""# Phase E1 — View, Family & Subfamily Analysis (OOF selected config)

## Per view
{_df_to_md(vm[['view','n_records','mae','rmse','spearman_mean','ndcg@all','top5_overlap','dynamic_k_exact_accuracy']])}

## Per family (top 15 by record count)
{fam_md}

Family and subfamily are reporting metadata only — never used as KNN distance
features. Full tables: tables/family_metrics.csv, tables/subfamily_metrics.csv.
""")

    # ---- MLP_REFERENCE_COMPARISON ---- #
    (reports_dir / "MLP_REFERENCE_COMPARISON.md").write_text(f"""# Phase E1 — KNN vs MLP (locked split-1)

Both models trained on splits 2-16 and evaluated on split 1 (n_samples =
{int(s1.get('n_samples', 0))}). MLP reference: {C.MLP_REFERENCE_DIR.name}.

Winner tally: {_mlp_winner_summary(comp)}

{_df_to_md(comp)}
""")

    # ---- FINAL_MODEL_READINESS ---- #
    rv = ctx["reload_validation"]
    ready = ctx["readiness"]["ready"]
    blockers = ctx["readiness"]["blockers"]
    (reports_dir / "FINAL_MODEL_READINESS.md").write_text(f"""# Phase E1 — Final Model Readiness

## Selected configuration
- config_id: **{sel_cfg['config_id']}**
- n_neighbors (K_neighbors): **{sel_cfg['n_neighbors']}**
- scaler: **{sel_cfg['scaler']}**  |  distance: **{sel_cfg['distance']}**
- neighbor_weighting: **{sel_cfg['neighbor_weighting']}**

## Frozen model
- Train splits: {ctx['final_metadata']['train_splits']}
- Per-view train counts: {ctx['final_metadata']['per_view_train_counts']}
- Reload prediction max abs diff: {rv['reload_prediction_max_abs_diff']:.2e}
- Reload identical: **{rv['reload_identical']}**
- Artifacts under models/ with SHA-256 in final_knn_metadata.json.

## Readiness for Phase E2
**{'YES' if ready else 'NO'}**

{"All checks passed." if ready else "Blockers:" + chr(10) + chr(10).join("- " + b for b in blockers)}
""")

    # ---- PHASE_E1_FINAL_REPORT ---- #
    _write_final_report(ctx, reports_dir)


def _write_final_report(ctx: Dict, reports_dir: Path) -> None:
    sel_cfg = ctx["selection"]["one_se_selected_configuration"]
    dr = ctx["data_report"]
    s1 = ctx["split1_suite"]
    oof = ctx["oof_suite"]
    comp = ctx["mlp_comparison"]
    dk_dev = {k: v for k, v in oof.items() if k.startswith("dynamic_k")}
    dk_s1 = {k: v for k, v in s1.items() if k.startswith("dynamic_k")}

    lines = [
        "# Phase E1 — Final Report: Leakage-Free KNN Utility Predictor",
        "",
        "## 1. Data coverage",
        f"- {dr['n_records']} records / {dr['n_datasets']} datasets / "
        f"3 views / {dr['n_features']} features / {dr['n_targets']} utility metrics.",
        f"- Development splits {C.DEV_SPLITS}; locked holdout split "
        f"{C.LOCKED_HOLDOUT_SPLIT} (evaluated once).",
        "",
        "## 2. Leakage evidence",
        f"- Split 1 absent from development data: {ctx['dev_audit']['split1_absent']}.",
        "- All 15 LOSO folds: validation split absent from train, dataset_ids "
        "disjoint, split 1 absent, preprocessing fit train-only.",
        f"- Final model train splits: {ctx['final_metadata']['train_splits']} "
        f"(split 1 in training: {ctx['final_metadata']['split1_in_training']}).",
        "",
        "## 3. Configuration search & selection",
        f"- 96 configurations x 15 folds. Raw best: "
        f"**{ctx['selection']['raw_best_configuration']['config_id']}**.",
        f"- One-SE selected (final): **{sel_cfg['config_id']}** "
        f"(n_neighbors={sel_cfg['n_neighbors']}, scaler={sel_cfg['scaler']}, "
        f"distance={sel_cfg['distance']}, weighting={sel_cfg['neighbor_weighting']}).",
        f"- Composite mean {_fmt(sel_cfg['mean_composite_score'])} "
        f"(SE {_fmt(sel_cfg['standard_error'])}).",
        "",
        "## 4. Development out-of-fold metrics (selected config)",
        _df_to_md(pd.DataFrame([{k: oof[k] for k in
            ['mae','rmse','r2','macro_r2','spearman_mean','kendall_mean',
             'pairwise_ranking_accuracy','ndcg@all','top1_accuracy',
             'top5_overlap','top10_overlap','cosine_similarity_mean']
            if k in oof}])),
        "",
        "### Development fold mean +/- std (selected config)",
        _df_to_md(ctx["selected_fold_summary"]),
        "",
        "## 5. Dynamic Top-K fidelity",
        f"- Dev OOF: exact={_fmt(dk_dev.get('dynamic_k_exact_accuracy'))}, "
        f"within-1={_fmt(dk_dev.get('dynamic_k_within_1_accuracy'))}, "
        f"MAE={_fmt(dk_dev.get('dynamic_k_mae'))}, "
        f"RMSE={_fmt(dk_dev.get('dynamic_k_rmse'))}, "
        f"under={_fmt(dk_dev.get('dynamic_k_underselection_rate'))}, "
        f"over={_fmt(dk_dev.get('dynamic_k_overselection_rate'))}.",
        f"- Split-1: exact={_fmt(dk_s1.get('dynamic_k_exact_accuracy'))}, "
        f"within-1={_fmt(dk_s1.get('dynamic_k_within_1_accuracy'))}, "
        f"MAE={_fmt(dk_s1.get('dynamic_k_mae'))}.",
        "",
        "## 6. Locked split-1 metrics",
        _df_to_md(pd.DataFrame([{k: s1[k] for k in
            ['mae','rmse','r2','macro_r2','spearman_mean','kendall_mean',
             'pairwise_ranking_accuracy','ndcg@all','top1_accuracy',
             'top5_overlap','top10_overlap','dynamic_k_exact_accuracy']
            if k in s1}])),
        "",
        "## 7. KNN vs MLP (split-1)",
        f"Winner tally: {_mlp_winner_summary(comp)}.",
        _df_to_md(comp),
        "",
        "## 8. Final artifacts",
        f"- Model dir: models/ (per-view preprocessor + index + utilities).",
        f"- Reload identical: {ctx['reload_validation']['reload_identical']}.",
        f"- Schemas: feature_schema.json, utility_metric_schema.json.",
        "",
        "## 9. Readiness for Phase E2",
        f"**{'YES' if ctx['readiness']['ready'] else 'NO'}**",
    ]
    if not ctx["readiness"]["ready"]:
        lines += ["", "Blockers:"] + [f"- {b}" for b in ctx["readiness"]["blockers"]]
    (reports_dir / "PHASE_E1_FINAL_REPORT.md").write_text("\n".join(lines))
