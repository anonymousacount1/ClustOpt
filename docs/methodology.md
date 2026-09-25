# Methodology (implementation map)

This page maps each component of the method to the code that implements it. The paper is the authoritative description.

## Data

**Controlled repository**
- 17,068 two-dimensional datasets in the unit square.
- 12 structural families and 86 subfamilies, with 2–5 clusters and four difficulty levels.
- Produced by a structural causal generator: `src/models/HYBRID_SCM` (`family_generation/`, `src/generator.py`).
- The family and subfamily parameterizations are in `data/family_configs/`. Every dataset's `family_seed` and `dataset_seed` are in `data/synthetic_generator/dataset_catalogue.csv`.

**Views**
- Every dataset is studied in three views: `x_only`, `y_only` and `xy_2d`.
- The partition is fitted on the view.
- The validity indices always evaluate the full two-dimensional data.

**Splits**
- There are 16 dataset-level splits, stratified by family, subfamily, difficulty and cluster count (`data/splits/`).
- All learned components are fitted on splits 2–16 and evaluated on the held-out split 1 (1,055 datasets).

## Utility targets

Candidate partitions are enumerated for each dataset view. The utility of index *j* measures how well *j*'s ranking of the candidates agrees with their ground-truth ARI ranking. It is a clipped blend of Spearman, Kendall, pairwise accuracy, NDCG and top/bottom-k overlap, computed by `src/models/ClustOpt/external_evaluation/compute_metric_utility.py`.

## Meta-features

250 label-free descriptors per view (`src/models/Clustering_Repository_Builder/features_extraction`), written to `<dataset>/features/features_records.csv`.

## Utility-profile predictors

| Predictor | Model | Code |
|---|---|---|
| Neural | multi-head MLP, 60 outputs | `src/models/metric_utility_mlp` |
| k-NN | one reference set per view | `src/models/metric_utility_knn` |

Released models are in `models/utility_predictor/`.

## Objective and search

- `regressor_dynamic` / `knn_dynamic` objectives (`src/models/ClustOpt/cluster_validity_indices/cvi_implementations/cvi_dynamic.py`) keep the top-k indices of the predicted profile, with raw weights.
- **CLUSTOPT as reported (arm C4)** uses the k-NN profile with k = 10 and the `KNN_TOP10` reranker (`configs/clustopt/knn_profile_top10/`; public entry point `clustopt.pipeline`). The neural profile with k = 5 is the variant evaluated as arms C0/C3.
- The objective is the weighted mean of the normalised index values.
- Search is Optuna TPE with 50 evaluations per view (`src/models/ClustOpt/search_algorithm`), over the search space in the configuration (`src/models/ClustOpt/search_space`).
- The random-search control uses Optuna's `RandomSampler` (`"sampler": "random"`).

## Candidate reranking

A learned reranker per selection policy (`src/models/ClustOpt_Candidate_Reranker`) chooses the final partition among the candidates the search visited. Released models are in `models/candidate_reranker/`.

## Policy selector (secondary variant)

A learned selector choosing among selection policies (`src/models/ClustOpt_Policy_Predictor`). Its models are not released, so the auxiliary arms C2 and C5 cannot be re-run; their per-dataset outputs are released. CLUSTOPT as reported (C4) does not use it.

## Validity indices

- **CLUSTOPT's inventory:** 60 indices (`clustopt.validity_indices`).
- **The auxiliary external index analysis:** 67 indices, i.e. those 60, three further classical indices, and four recent comparators (CDbw, CVDD, CVNN, DCSI; `src/models/Independent_Domain_Benchmark/metric_validation`).
- **CVNN:** computed through the external R package fpc. See `analysis/index_analysis/`.

## External benchmark arms

| Arm | System |
|---|---|
| C0 / C1 | CLUSTOPT, neural / k-NN profile, no reranking |
| C3 / C4 | CLUSTOPT, neural / k-NN profile, with reranking |
| C2 / C5 | policy selector v1 / v2 (C5 with reranking) |
| A0–A3 | AutoClust: original, + established, + new, extended index inventory |
| M0–M3 | ML2DAC: same four inventories |

The primary measure is best-view ARI over the three views, reported together with the three-view runtime. Configurations: `configs/external_benchmark/`.
