# metric_utility_knn — Phase E1

A leakage-free **K-Nearest-Neighbors utility-vector predictor** that maps a
dataset-view meta-feature vector to a predicted metric-utility vector, as a
drop-in alternative to the existing ClustOpt MLP predictor. Phase E1 selects,
validates, and freezes the predictor; Phase E2 (future) integrates it.

```
dataset-view meta-features  ->  predicted metric-utility vector
```

## Terminology
- **`n_neighbors`** (`K_neighbors`) — neighbours used by the KNN predictor.
- **`selected_metric_count`** (`K_metrics`) — metrics chosen by the Dynamic
  Top-K heuristic (`dynamic_topk_relsoft_a092_k5_ceil`, imported from ClustOpt).

These are always kept distinct; a bare `k` is never used ambiguously.

## What it reuses (no re-definition)
- Meta-feature / utility columns, order, ids, views, and split assignments from
  the unified training dataset (same schema as the MLP).
- The MLP metric functions (`models.metric_utility_mlp.metrics.compute_all`).
- The frozen Dynamic Top-K heuristic from ClustOpt.

## Modules
| file | role |
| --- | --- |
| `config.py` | paths, split policy, 96-config grid, ranking groups |
| `data_loader.py` | load unified CSV + attach split_id + schema alignment |
| `preprocessing.py` | median imputer + standard/robust scaler (train-only) |
| `neighbor_search.py` | single max-neighbour query, reused for all `n_neighbors` |
| `knn_predictor.py` | prediction kernel + frozen 3-view `KnnUtilityPredictor` |
| `metrics.py` | MLP core suite + extra/dynamic-K/per-metric diagnostics |
| `cross_validation.py` | 15-fold LOSO over splits 2-16 + leakage audits |
| `configuration_search.py` | efficient, resumable 96-config LOSO search |
| `configuration_selection.py` | 4-group composite ranking + one-SE rule |
| `final_model_builder.py` | freeze model, hashes, reload verify, OOF predictions |
| `holdout_evaluation.py` | locked split-1 evaluation + MLP comparison |
| `plotting.py`, `reporting.py` | figures, tables, reports, validation JSONs |
| `run_phase_e1.py` | end-to-end orchestrator (resumable) |

## Run
```
python -u -m models.metric_utility_knn.run_phase_e1
```
Options: `--skip-plots`, `--limit-rows N` (debug), `--results-root PATH` (smoke).

Resume: per-fold search metrics are cached under
`results_analysis/metric_utility_knn/phase_e1/cache/`; completed folds are
skipped on re-run.

## Tests
```
python -m pytest models/metric_utility_knn/tests/ -q
```

## Results
`results_analysis/metric_utility_knn/phase_e1/` (canonical, timestamp-free).
Start with `reports/PHASE_E1_FINAL_REPORT.md`.
