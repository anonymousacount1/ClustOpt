# Original location -> public release path

Every file in this release was copied from the development tree (or written for the release).
The left column is the original repository-relative location; the right column is where it is in
this release. Configuration files and implementation files are byte-identical except for the
release-side edits listed at the end (paths, notices, packaging, anonymity, and the CVNN exclusion).

## Implementation

| original | public |
|---|---|
| `models/**` | `src/models/**` |
| `experiments/**` | `src/experiments/**` |
| `scripts/stage1/**`, `scripts/stage2/**` | `src/scripts/...` |

Development-only checks (verification gates, audits, monitors, dry runs) were not copied.

## `analysis/paper`

| original | public |
|---|---|
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/scripts/build_v12_evidence.py` | `analysis/paper/build_evidence.py` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/scripts/build_v12_figures.py` | `analysis/paper/build_figures.py` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/scripts/build_v12_tables.py` | `analysis/paper/build_tables.py` |

## `analysis/statistics`

| original | public |
|---|---|
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_controlled_analysis.py` | `analysis/statistics/controlled_paired_statistics.py` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_criterion_baselines.py` | `analysis/statistics/criterion_baselines.py` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e13_external_analysis.py` | `analysis/statistics/external_paired_statistics.py` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_predictor_and_alignment.py` | `analysis/statistics/predictor_alignment.py` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/_stats_common.py` | `analysis/statistics/stats_common.py` |

## `configs/baselines`

| original | public |
|---|---|
| `experiments/external_baselines/AutoClust/configs/autoclust_fallback_no_artifacts.json` | `configs/baselines/autoclust/autoclust_fallback_no_artifacts.json` |
| `experiments/external_baselines/AutoClust/configs/autoclust_full_ml2dac_reimplementation.json` | `configs/baselines/autoclust/autoclust_full_ml2dac_reimplementation.json` |
| `experiments/external_baselines/AutoClust/configs/autoclust_original_paper_40_trials.json` | `configs/baselines/autoclust/autoclust_original_paper_40_trials.json` |
| `experiments/external_baselines/AutoClust/configs/autoclust_smoke_test_config.json` | `configs/baselines/autoclust/autoclust_smoke_test_config.json` |
| `experiments/external_baselines/AutoML4Clust/configs/automl4clust_smac_1d_x.json` | `configs/baselines/automl4clust/automl4clust_smac_1d_x.json` |
| `experiments/external_baselines/AutoML4Clust/configs/automl4clust_smac_1d_y.json` | `configs/baselines/automl4clust/automl4clust_smac_1d_y.json` |
| `experiments/external_baselines/AutoML4Clust/configs/automl4clust_smac_2d.json` | `configs/baselines/automl4clust/automl4clust_smac_2d.json` |
| `experiments/external_baselines/AutoML4Clust/configs/automl4clust_smac_all_records.json` | `configs/baselines/automl4clust/automl4clust_smac_all_records.json` |
| `experiments/external_baselines/AutoML4Clust/configs/smoke_test_config.json` | `configs/baselines/automl4clust/smoke_test_config.json` |
| `experiments/external_baselines/ML2DAC/configs/ml2dac_fallback_no_mkr_random_or_smac.json` | `configs/baselines/ml2dac/ml2dac_fallback_no_mkr_random_or_smac.json` |
| `experiments/external_baselines/ML2DAC/configs/ml2dac_full_meta_learning.json` | `configs/baselines/ml2dac/ml2dac_full_meta_learning.json` |
| `experiments/external_baselines/ML2DAC/configs/ml2dac_smoke_test_config.json` | `configs/baselines/ml2dac/ml2dac_smoke_test_config.json` |

## `configs/clustopt`

| original | public |
|---|---|
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top10_softmax_t05/x_only.json` | `configs/clustopt/ablations/knn_top10_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top10_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/knn_top10_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top10_softmax_t05/y_only.json` | `configs/clustopt/ablations/knn_top10_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top1_raw/x_only.json` | `configs/clustopt/ablations/knn_top1_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top1_raw/xy_2d.json` | `configs/clustopt/ablations/knn_top1_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top1_raw/y_only.json` | `configs/clustopt/ablations/knn_top1_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top1_softmax_t05/x_only.json` | `configs/clustopt/ablations/knn_top1_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top1_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/knn_top1_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top1_softmax_t05/y_only.json` | `configs/clustopt/ablations/knn_top1_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top3_raw/x_only.json` | `configs/clustopt/ablations/knn_top3_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top3_raw/xy_2d.json` | `configs/clustopt/ablations/knn_top3_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top3_raw/y_only.json` | `configs/clustopt/ablations/knn_top3_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top3_softmax_t05/x_only.json` | `configs/clustopt/ablations/knn_top3_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top3_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/knn_top3_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top3_softmax_t05/y_only.json` | `configs/clustopt/ablations/knn_top3_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top5_raw/x_only.json` | `configs/clustopt/ablations/knn_top5_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top5_raw/xy_2d.json` | `configs/clustopt/ablations/knn_top5_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top5_raw/y_only.json` | `configs/clustopt/ablations/knn_top5_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top5_softmax_t05/x_only.json` | `configs/clustopt/ablations/knn_top5_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top5_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/knn_top5_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top5_softmax_t05/y_only.json` | `configs/clustopt/ablations/knn_top5_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_predictK_raw/x_only.json` | `configs/clustopt/ablations/knn_top_dynamic_predictK_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_predictK_raw/xy_2d.json` | `configs/clustopt/ablations/knn_top_dynamic_predictK_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_predictK_raw/y_only.json` | `configs/clustopt/ablations/knn_top_dynamic_predictK_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_predictK_softmax_t05/x_only.json` | `configs/clustopt/ablations/knn_top_dynamic_predictK_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_predictK_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/knn_top_dynamic_predictK_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_predictK_softmax_t05/y_only.json` | `configs/clustopt/ablations/knn_top_dynamic_predictK_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_realK_raw/x_only.json` | `configs/clustopt/ablations/knn_top_dynamic_realK_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_realK_raw/xy_2d.json` | `configs/clustopt/ablations/knn_top_dynamic_realK_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_realK_raw/y_only.json` | `configs/clustopt/ablations/knn_top_dynamic_realK_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_realK_softmax_t05/x_only.json` | `configs/clustopt/ablations/knn_top_dynamic_realK_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_realK_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/knn_top_dynamic_realK_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top_dynamic_realK_softmax_t05/y_only.json` | `configs/clustopt/ablations/knn_top_dynamic_realK_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top10_raw/x_only.json` | `configs/clustopt/ablations/neural_top10_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top10_raw/xy_2d.json` | `configs/clustopt/ablations/neural_top10_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top10_raw/y_only.json` | `configs/clustopt/ablations/neural_top10_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top10_softmax_t05/x_only.json` | `configs/clustopt/ablations/neural_top10_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top10_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/neural_top10_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top10_softmax_t05/y_only.json` | `configs/clustopt/ablations/neural_top10_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top1_raw/x_only.json` | `configs/clustopt/ablations/neural_top1_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top1_raw/xy_2d.json` | `configs/clustopt/ablations/neural_top1_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top1_raw/y_only.json` | `configs/clustopt/ablations/neural_top1_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top1_softmax_t05/x_only.json` | `configs/clustopt/ablations/neural_top1_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top1_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/neural_top1_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top1_softmax_t05/y_only.json` | `configs/clustopt/ablations/neural_top1_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top3_raw/x_only.json` | `configs/clustopt/ablations/neural_top3_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top3_raw/xy_2d.json` | `configs/clustopt/ablations/neural_top3_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top3_raw/y_only.json` | `configs/clustopt/ablations/neural_top3_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top3_softmax_t05/x_only.json` | `configs/clustopt/ablations/neural_top3_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top3_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/neural_top3_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top3_softmax_t05/y_only.json` | `configs/clustopt/ablations/neural_top3_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top5_raw/x_only.json` | `configs/clustopt/ablations/neural_top5_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top5_raw/xy_2d.json` | `configs/clustopt/ablations/neural_top5_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top5_raw/y_only.json` | `configs/clustopt/ablations/neural_top5_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top5_softmax_t05/x_only.json` | `configs/clustopt/ablations/neural_top5_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top5_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/neural_top5_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top5_softmax_t05/y_only.json` | `configs/clustopt/ablations/neural_top5_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_predictK_raw/x_only.json` | `configs/clustopt/ablations/neural_top_dynamic_predictK_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_predictK_raw/xy_2d.json` | `configs/clustopt/ablations/neural_top_dynamic_predictK_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_predictK_raw/y_only.json` | `configs/clustopt/ablations/neural_top_dynamic_predictK_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_predictK_softmax_t05/x_only.json` | `configs/clustopt/ablations/neural_top_dynamic_predictK_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_predictK_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/neural_top_dynamic_predictK_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_predictK_softmax_t05/y_only.json` | `configs/clustopt/ablations/neural_top_dynamic_predictK_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_realK_raw/x_only.json` | `configs/clustopt/ablations/neural_top_dynamic_realK_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_realK_raw/xy_2d.json` | `configs/clustopt/ablations/neural_top_dynamic_realK_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_realK_raw/y_only.json` | `configs/clustopt/ablations/neural_top_dynamic_realK_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_realK_softmax_t05/x_only.json` | `configs/clustopt/ablations/neural_top_dynamic_realK_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_realK_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/neural_top_dynamic_realK_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/regressor_top_dynamic_realK_softmax_t05/y_only.json` | `configs/clustopt/ablations/neural_top_dynamic_realK_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top10_raw/x_only.json` | `configs/clustopt/ablations/oracle_top10_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top10_raw/xy_2d.json` | `configs/clustopt/ablations/oracle_top10_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top10_raw/y_only.json` | `configs/clustopt/ablations/oracle_top10_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top10_softmax_t05/x_only.json` | `configs/clustopt/ablations/oracle_top10_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top10_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/oracle_top10_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top10_softmax_t05/y_only.json` | `configs/clustopt/ablations/oracle_top10_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top1_raw/x_only.json` | `configs/clustopt/ablations/oracle_top1_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top1_raw/xy_2d.json` | `configs/clustopt/ablations/oracle_top1_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top1_raw/y_only.json` | `configs/clustopt/ablations/oracle_top1_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top1_softmax_t05/x_only.json` | `configs/clustopt/ablations/oracle_top1_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top1_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/oracle_top1_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top1_softmax_t05/y_only.json` | `configs/clustopt/ablations/oracle_top1_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top3_raw/x_only.json` | `configs/clustopt/ablations/oracle_top3_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top3_raw/xy_2d.json` | `configs/clustopt/ablations/oracle_top3_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top3_raw/y_only.json` | `configs/clustopt/ablations/oracle_top3_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top3_softmax_t05/x_only.json` | `configs/clustopt/ablations/oracle_top3_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top3_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/oracle_top3_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top3_softmax_t05/y_only.json` | `configs/clustopt/ablations/oracle_top3_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top5_raw/x_only.json` | `configs/clustopt/ablations/oracle_top5_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top5_raw/xy_2d.json` | `configs/clustopt/ablations/oracle_top5_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top5_raw/y_only.json` | `configs/clustopt/ablations/oracle_top5_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top5_softmax_t05/x_only.json` | `configs/clustopt/ablations/oracle_top5_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top5_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/oracle_top5_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top5_softmax_t05/y_only.json` | `configs/clustopt/ablations/oracle_top5_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top_dynamic_raw/x_only.json` | `configs/clustopt/ablations/oracle_top_dynamic_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top_dynamic_raw/xy_2d.json` | `configs/clustopt/ablations/oracle_top_dynamic_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top_dynamic_raw/y_only.json` | `configs/clustopt/ablations/oracle_top_dynamic_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top_dynamic_softmax_t05/x_only.json` | `configs/clustopt/ablations/oracle_top_dynamic_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top_dynamic_softmax_t05/xy_2d.json` | `configs/clustopt/ablations/oracle_top_dynamic_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/real_top_dynamic_softmax_t05/y_only.json` | `configs/clustopt/ablations/oracle_top_dynamic_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/utilities_maps/utility_map_x_only_32.json` | `configs/clustopt/candidate_bank_32/x_only.json` |
| `models/ClustOpt/configs/utilities_maps/utility_map_xy_2d_32.json` | `configs/clustopt/candidate_bank_32/xy_2d.json` |
| `models/ClustOpt/configs/utilities_maps/utility_map_y_only_32.json` | `configs/clustopt/candidate_bank_32/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top10_raw/x_only.json` | `configs/clustopt/knn_profile_top10/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top10_raw/xy_2d.json` | `configs/clustopt/knn_profile_top10/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseE2_knn/knn_top10_raw/y_only.json` | `configs/clustopt/knn_profile_top10/y_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_mlp_top5_raw_fixed50/x_only.json` | `configs/clustopt/neural_profile_top5/x_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_mlp_top5_raw_fixed50/xy_2d.json` | `configs/clustopt/neural_profile_top5/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_mlp_top5_raw_fixed50/y_only.json` | `configs/clustopt/neural_profile_top5/y_only.json` |

## `configs/controlled_benchmark`

| original | public |
|---|---|
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_oracle_top5_raw_fixed50/x_only.json` | `configs/controlled_benchmark/inventory_factorial/full60_oracle_top5/x_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_oracle_top5_raw_fixed50/xy_2d.json` | `configs/controlled_benchmark/inventory_factorial/full60_oracle_top5/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_oracle_top5_raw_fixed50/y_only.json` | `configs/controlled_benchmark/inventory_factorial/full60_oracle_top5/y_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_uniform_fixed50/x_only.json` | `configs/controlled_benchmark/inventory_factorial/full60_uniform/x_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_uniform_fixed50/xy_2d.json` | `configs/controlled_benchmark/inventory_factorial/full60_uniform/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_full60_uniform_fixed50/y_only.json` | `configs/controlled_benchmark/inventory_factorial/full60_uniform/y_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_oracle_top5_raw_fixed50/x_only.json` | `configs/controlled_benchmark/inventory_factorial/head14_oracle_top5/x_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_oracle_top5_raw_fixed50/xy_2d.json` | `configs/controlled_benchmark/inventory_factorial/head14_oracle_top5/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_oracle_top5_raw_fixed50/y_only.json` | `configs/controlled_benchmark/inventory_factorial/head14_oracle_top5/y_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_mlp_top5_raw_fixed50/x_only.json` | `configs/controlled_benchmark/inventory_factorial/head14_predicted_top5/x_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_mlp_top5_raw_fixed50/xy_2d.json` | `configs/controlled_benchmark/inventory_factorial/head14_predicted_top5/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_mlp_top5_raw_fixed50/y_only.json` | `configs/controlled_benchmark/inventory_factorial/head14_predicted_top5/y_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_uniform_fixed50/x_only.json` | `configs/controlled_benchmark/inventory_factorial/head14_uniform/x_only.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_uniform_fixed50/xy_2d.json` | `configs/controlled_benchmark/inventory_factorial/head14_uniform/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage1_2x3/stage1_head14_uniform_fixed50/y_only.json` | `configs/controlled_benchmark/inventory_factorial/head14_uniform/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_dynamic_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_dynamic_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_dynamic_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_dynamic_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_dynamic_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_dynamic_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_dynamic_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_dynamic_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_dynamic_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_dynamic_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_dynamic_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_dynamic_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top10_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top10_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top10_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_top10_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top10_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top10_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top10_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top10_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top10_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_top10_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top10_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top10_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top1_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top1_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top1_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_top1_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top1_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top1_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top1_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top1_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top1_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_top1_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top1_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top1_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top3_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top3_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top3_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_top3_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top3_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top3_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top3_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top3_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top3_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_top3_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top3_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top3_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top5_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top5_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top5_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_top5_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top5_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top5_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top5_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top5_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top5_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/knn_top5_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_knn_top5_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/knn_top5_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_dynamic_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_dynamic_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_dynamic_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_dynamic_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_dynamic_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_dynamic_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_dynamic_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_dynamic_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_dynamic_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_dynamic_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_dynamic_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_dynamic_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top10_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top10_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top10_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top10_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top10_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top10_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top10_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top10_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top10_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top10_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top10_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top10_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top1_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top1_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top1_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top1_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top1_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top1_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top1_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top1_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top1_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top1_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top1_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top1_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top3_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top3_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top3_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top3_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top3_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top3_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top3_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top3_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top3_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top3_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top3_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top3_softmax_t05/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top5_raw/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top5_raw/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top5_raw/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top5_raw/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top5_raw/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top5_raw/y_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top5_softmax_t05/x_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top5_softmax_t05/x_only.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top5_softmax_t05/xy_2d.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top5_softmax_t05/xy_2d.json` |
| `models/ClustOpt/configs/experiments/stage2c4/stage2c4_mlp_top5_softmax_t05/y_only.json` | `configs/controlled_benchmark/weighting_ablation/mlp_top5_softmax_t05/y_only.json` |

## `configs/criterion_baselines`

| original | public |
|---|---|
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_best1/x_only.json` | `configs/criterion_baselines/best_single_index/x_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_best1/xy_2d.json` | `configs/criterion_baselines/best_single_index/xy_2d.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_best1/y_only.json` | `configs/criterion_baselines/best_single_index/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/calinski_harabasz_single/x_only.json` | `configs/criterion_baselines/calinski_harabasz_single/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/calinski_harabasz_single/xy_2d.json` | `configs/criterion_baselines/calinski_harabasz_single/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/calinski_harabasz_single/y_only.json` | `configs/criterion_baselines/calinski_harabasz_single/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/davies_bouldin_single/x_only.json` | `configs/criterion_baselines/davies_bouldin_single/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/davies_bouldin_single/xy_2d.json` | `configs/criterion_baselines/davies_bouldin_single/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/davies_bouldin_single/y_only.json` | `configs/criterion_baselines/davies_bouldin_single/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/dbcv_single/x_only.json` | `configs/criterion_baselines/dbcv_single/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/dbcv_single/xy_2d.json` | `configs/criterion_baselines/dbcv_single/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/dbcv_single/y_only.json` | `configs/criterion_baselines/dbcv_single/y_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_c4_seed/x_only.json` | `configs/criterion_baselines/knn_profile_matched_seed/x_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_c4_seed/xy_2d.json` | `configs/criterion_baselines/knn_profile_matched_seed/xy_2d.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_c4_seed/y_only.json` | `configs/criterion_baselines/knn_profile_matched_seed/y_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_meanprof/x_only.json` | `configs/criterion_baselines/mean_utility_profile/x_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_meanprof/xy_2d.json` | `configs/criterion_baselines/mean_utility_profile/xy_2d.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_meanprof/y_only.json` | `configs/criterion_baselines/mean_utility_profile/y_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_random/x_only.json` | `configs/criterion_baselines/random_search/x_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_random/xy_2d.json` | `configs/criterion_baselines/random_search/xy_2d.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_random/y_only.json` | `configs/criterion_baselines/random_search/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/silhouette_single/x_only.json` | `configs/criterion_baselines/silhouette_single/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/silhouette_single/xy_2d.json` | `configs/criterion_baselines/silhouette_single/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/silhouette_single/y_only.json` | `configs/criterion_baselines/silhouette_single/y_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_core3/x_only.json` | `configs/criterion_baselines/static_generalist_core3/x_only.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_core3/xy_2d.json` | `configs/criterion_baselines/static_generalist_core3/xy_2d.json` |
| `models/ClustOpt/configs/experiments/e1_conditioning/e1_core3/y_only.json` | `configs/criterion_baselines/static_generalist_core3/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/all_metrics_uniform/x_only.json` | `configs/criterion_baselines/uniform_all60/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/all_metrics_uniform/xy_2d.json` | `configs/criterion_baselines/uniform_all60/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/all_metrics_uniform/y_only.json` | `configs/criterion_baselines/uniform_all60/y_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/uniform_classic_cvi/x_only.json` | `configs/criterion_baselines/uniform_classic_cvi/x_only.json` |
| `models/ClustOpt/configs/experiments/phaseD/uniform_classic_cvi/xy_2d.json` | `configs/criterion_baselines/uniform_classic_cvi/xy_2d.json` |
| `models/ClustOpt/configs/experiments/phaseD/uniform_classic_cvi/y_only.json` | `configs/criterion_baselines/uniform_classic_cvi/y_only.json` |

## `configs/external_benchmark`

| original | public |
|---|---|
| `models/Independent_Domain_Benchmark/configs/benchmark_protocol.json` | `configs/external_benchmark/benchmark_protocol.json` |
| `models/Independent_Domain_Benchmark/configs/domain_shift_protocol.json` | `configs/external_benchmark/domain_shift_protocol.json` |
| `models/Independent_Domain_Benchmark/configs/method_registry.json` | `configs/external_benchmark/method_registry.json` |
| `models/Independent_Domain_Benchmark/configs/metric_analysis_registry.json` | `configs/external_benchmark/metric_analysis_registry.json` |
| `models/Independent_Domain_Benchmark/configs/metric_registry.json` | `configs/external_benchmark/metric_registry.json` |
| `models/Independent_Domain_Benchmark/configs/ml2dac_cvi_applicability_policy.json` | `configs/external_benchmark/ml2dac_cvi_applicability_policy.json` |
| `models/Independent_Domain_Benchmark/configs/ml2dac_direction_registry.json` | `configs/external_benchmark/ml2dac_direction_registry.json` |
| `models/Independent_Domain_Benchmark/configs/modern_cvi_registry.json` | `configs/external_benchmark/modern_cvi_registry.json` |
| `models/Independent_Domain_Benchmark/configs/numerical_reproducibility_policy.json` | `configs/external_benchmark/numerical_reproducibility_policy.json` |
| `models/Independent_Domain_Benchmark/configs/runtime_protocol.json` | `configs/external_benchmark/runtime_protocol.json` |

## `data/external_benchmark`

| original | public |
|---|---|
| `models/Independent_Domain_Benchmark/configs/dataset_manifest.csv` | `data/external_benchmark/dataset_manifest.csv` |
| `models/Independent_Domain_Benchmark/configs/dataset_manifest.json` | `data/external_benchmark/dataset_manifest.json` |
| `results_analysis/final_scientific_handoff/clustopt_final_scientific_analysis_package/04_stage4_external_benchmark/corpus_manifest/stage4_external_corpus_publication_manifest.json` | `data/external_benchmark/prepared_representation_manifest.json` |
| `results_analysis/final_scientific_handoff/clustopt_final_scientific_analysis_package/04_stage4_external_benchmark/corpus_manifest/manifest_hash.txt` | `data/external_benchmark/prepared_representation_manifest.sha256_16.txt` |

## `data/family_configs`

| original | public |
|---|---|
| `models/HYBRID_SCM/configs/repository_generation/families/arcs_circles_rings.json` | `data/family_configs/arcs_circles_rings.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/deinterleaving_pri_toa_amp_like.json` | `data/family_configs/deinterleaving_pri_toa_amp_like.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/hollow_topological_components.json` | `data/family_configs/hollow_topological_components.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/hybrid_mixed_geometry.json` | `data/family_configs/hybrid_mixed_geometry.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/ladder_grid_lattice.json` | `data/family_configs/ladder_grid_lattice.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/linear_bands_parallel_stripes.json` | `data/family_configs/linear_bands_parallel_stripes.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/parabolic_curved_chains.json` | `data/family_configs/parabolic_curved_chains.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/piecewise_polylines_corners.json` | `data/family_configs/piecewise_polylines_corners.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/polygons_rectangles_frames.json` | `data/family_configs/polygons_rectangles_frames.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/radial_spiral_meander.json` | `data/family_configs/radial_spiral_meander.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/standard_clustering_blobs.json` | `data/family_configs/standard_clustering_blobs.json` |
| `models/HYBRID_SCM/configs/repository_generation/families/texture_density_fields.json` | `data/family_configs/texture_density_fields.json` |

## `data/splits`

| original | public |
|---|---|
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/dataset_split_assignments.csv` | `data/splits/dataset_split_assignments.csv` |
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/dataset_split_assignments.json` | `data/splits/dataset_split_assignments.json` |
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/split_distribution_by_cluster_count.csv` | `data/splits/split_distribution_by_cluster_count.csv` |
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/split_distribution_by_difficulty.csv` | `data/splits/split_distribution_by_difficulty.csv` |
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/split_distribution_by_family.csv` | `data/splits/split_distribution_by_family.csv` |
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/split_distribution_by_subfamily.csv` | `data/splits/split_distribution_by_subfamily.csv` |
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/split_distribution_full_strata.csv` | `data/splits/split_distribution_full_strata.csv` |
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/split_summary.csv` | `data/splits/split_summary.csv` |
| `results_analysis/clustering_repository/analyzed_data/experiment_splits/split_summary.md` | `data/splits/split_summary.md` |

## `data/synthetic_generator`

| original | public |
|---|---|
| `derived: seeds from data/family_configs (all 17,068 ids reproduced) + frozen point counts` | `data/synthetic_generator/dataset_catalogue.csv` |
| `results_analysis/clustering_repository/analyzed_data/unified_training_dataset/build_metadata.json` | `data/synthetic_generator/training_table_build_metadata.json` |

## `experiments/controlled_benchmark`

| original | public |
|---|---|
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/aggregate_e1_final.py` | `experiments/controlled_benchmark/criterion_baselines/aggregate.py` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/build_e1_configs.py` | `experiments/controlled_benchmark/criterion_baselines/build_configs.py` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/build_e1_manifests.py` | `experiments/controlled_benchmark/criterion_baselines/build_manifests.py` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/derive_development_baselines.py` | `experiments/controlled_benchmark/criterion_baselines/derive_development_baselines.py` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/run_e1_all_subfamilies.py` | `experiments/controlled_benchmark/criterion_baselines/run_all_subfamilies.py` |

## `models/candidate_reranker`

| original | public |
|---|---|
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_DYNAMIC/model.joblib` | `models/candidate_reranker/KNN_DYNAMIC/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_DYNAMIC/model_manifest.json` | `models/candidate_reranker/KNN_DYNAMIC/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP1/model.joblib` | `models/candidate_reranker/KNN_TOP1/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP1/model_manifest.json` | `models/candidate_reranker/KNN_TOP1/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP10/model.joblib` | `models/candidate_reranker/KNN_TOP10/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP10/model_manifest.json` | `models/candidate_reranker/KNN_TOP10/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP3/model.joblib` | `models/candidate_reranker/KNN_TOP3/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP3/model_manifest.json` | `models/candidate_reranker/KNN_TOP3/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP5/model.joblib` | `models/candidate_reranker/KNN_TOP5/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP5/model_manifest.json` | `models/candidate_reranker/KNN_TOP5/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_DYNAMIC/model.joblib` | `models/candidate_reranker/MLP_DYNAMIC/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_DYNAMIC/model_manifest.json` | `models/candidate_reranker/MLP_DYNAMIC/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP1/model.joblib` | `models/candidate_reranker/MLP_TOP1/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP1/model_manifest.json` | `models/candidate_reranker/MLP_TOP1/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP10/model.joblib` | `models/candidate_reranker/MLP_TOP10/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP10/model_manifest.json` | `models/candidate_reranker/MLP_TOP10/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP3/model.joblib` | `models/candidate_reranker/MLP_TOP3/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP3/model_manifest.json` | `models/candidate_reranker/MLP_TOP3/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP5/model.joblib` | `models/candidate_reranker/MLP_TOP5/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP5/model_manifest.json` | `models/candidate_reranker/MLP_TOP5/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_model_training_summary.csv` | `models/candidate_reranker/model_training_summary.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/policy_to_reranker_mapping.json` | `models/candidate_reranker/policy_to_reranker_mapping.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_reranker_registry.json` | `models/candidate_reranker/reranker_registry.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_training_population.csv` | `models/candidate_reranker/training_population.csv` |

## `models/utility_predictor`

| original | public |
|---|---|
| `results_analysis/metric_utility_knn/phase_e1/models/feature_schema.json` | `models/utility_predictor/knn/feature_schema.json` |
| `results_analysis/metric_utility_knn/phase_e1/models/final_knn_metadata.json` | `models/utility_predictor/knn/final_knn_metadata.json` |
| `results_analysis/metric_utility_knn/phase_e1/run_config.json` | `models/utility_predictor/knn/run_config.json` |
| `results_analysis/metric_utility_knn/phase_e1/selected_configuration.json` | `models/utility_predictor/knn/selected_configuration.json` |
| `results_analysis/metric_utility_knn/phase_e1/models/training_manifest.parquet` | `models/utility_predictor/knn/training_manifest.parquet` |
| `results_analysis/metric_utility_knn/phase_e1/models/utility_metric_schema.json` | `models/utility_predictor/knn/utility_metric_schema.json` |
| `results_analysis/metric_utility_knn/phase_e1/models/x_only/preprocessor.pkl` | `models/utility_predictor/knn/x_only/preprocessor.pkl` |
| `results_analysis/metric_utility_knn/phase_e1/models/x_only/training_features.npy` | `models/utility_predictor/knn/x_only/training_features.npy` |
| `results_analysis/metric_utility_knn/phase_e1/models/x_only/training_record_ids.npy` | `models/utility_predictor/knn/x_only/training_record_ids.npy` |
| `results_analysis/metric_utility_knn/phase_e1/models/x_only/training_utilities.npy` | `models/utility_predictor/knn/x_only/training_utilities.npy` |
| `results_analysis/metric_utility_knn/phase_e1/models/xy_2d/preprocessor.pkl` | `models/utility_predictor/knn/xy_2d/preprocessor.pkl` |
| `results_analysis/metric_utility_knn/phase_e1/models/xy_2d/training_features.npy` | `models/utility_predictor/knn/xy_2d/training_features.npy` |
| `results_analysis/metric_utility_knn/phase_e1/models/xy_2d/training_record_ids.npy` | `models/utility_predictor/knn/xy_2d/training_record_ids.npy` |
| `results_analysis/metric_utility_knn/phase_e1/models/xy_2d/training_utilities.npy` | `models/utility_predictor/knn/xy_2d/training_utilities.npy` |
| `results_analysis/metric_utility_knn/phase_e1/models/y_only/preprocessor.pkl` | `models/utility_predictor/knn/y_only/preprocessor.pkl` |
| `results_analysis/metric_utility_knn/phase_e1/models/y_only/training_features.npy` | `models/utility_predictor/knn/y_only/training_features.npy` |
| `results_analysis/metric_utility_knn/phase_e1/models/y_only/training_record_ids.npy` | `models/utility_predictor/knn/y_only/training_record_ids.npy` |
| `results_analysis/metric_utility_knn/phase_e1/models/y_only/training_utilities.npy` | `models/utility_predictor/knn/y_only/training_utilities.npy` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/config.json` | `models/utility_predictor/neural_full60/config.json` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/folds/fold_0/best_model.pt` | `models/utility_predictor/neural_full60/folds/fold_0/best_model.pt` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/folds/fold_0/feature_columns.json` | `models/utility_predictor/neural_full60/folds/fold_0/feature_columns.json` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/folds/fold_0/head_mapping.json` | `models/utility_predictor/neural_full60/folds/fold_0/head_mapping.json` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/folds/fold_0/split_summary.json` | `models/utility_predictor/neural_full60/folds/fold_0/split_summary.json` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/folds/fold_0/target_columns.json` | `models/utility_predictor/neural_full60/folds/fold_0/target_columns.json` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/folds/fold_0/test_metrics.json` | `models/utility_predictor/neural_full60/folds/fold_0/test_metrics.json` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/folds/fold_0/x_scaler.pkl` | `models/utility_predictor/neural_full60/folds/fold_0/x_scaler.pkl` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/config.json` | `models/utility_predictor/neural_head14/config.json` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/folds/fold_0/best_model.pt` | `models/utility_predictor/neural_head14/folds/fold_0/best_model.pt` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/folds/fold_0/feature_columns.json` | `models/utility_predictor/neural_head14/folds/fold_0/feature_columns.json` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/folds/fold_0/head_mapping.json` | `models/utility_predictor/neural_head14/folds/fold_0/head_mapping.json` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/folds/fold_0/split_summary.json` | `models/utility_predictor/neural_head14/folds/fold_0/split_summary.json` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/folds/fold_0/target_columns.json` | `models/utility_predictor/neural_head14/folds/fold_0/target_columns.json` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/folds/fold_0/test_metrics.json` | `models/utility_predictor/neural_head14/folds/fold_0/test_metrics.json` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/folds/fold_0/x_scaler.pkl` | `models/utility_predictor/neural_head14/folds/fold_0/x_scaler.pkl` |

## `results/controlled`

| original | public |
|---|---|
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/coverage_audit.csv` | `results/controlled/baseline_inventory_decomposition/autoclust/coverage_audit.csv` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/experiment_manifest.json` | `results/controlled/baseline_inventory_decomposition/autoclust/experiment_manifest.json` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/factorial_effects.json` | `results/controlled/baseline_inventory_decomposition/autoclust/factorial_effects.json` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/factorial_uncertainty.csv` | `results/controlled/baseline_inventory_decomposition/autoclust/factorial_uncertainty.csv` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/four_arm_dataset_results.csv.gz` | `results/controlled/baseline_inventory_decomposition/autoclust/four_arm_dataset_results.csv.gz` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/four_arm_family_summary.csv` | `results/controlled/baseline_inventory_decomposition/autoclust/four_arm_family_summary.csv` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/four_arm_global_summary.csv` | `results/controlled/baseline_inventory_decomposition/autoclust/four_arm_global_summary.csv` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/four_arm_subfamily_summary.csv` | `results/controlled/baseline_inventory_decomposition/autoclust/four_arm_subfamily_summary.csv` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/paired_comparisons.csv` | `results/controlled/baseline_inventory_decomposition/autoclust/paired_comparisons.csv` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/per_dataset_factorial_terms.csv.gz` | `results/controlled/baseline_inventory_decomposition/autoclust/per_dataset_factorial_terms.csv.gz` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/provenance.json` | `results/controlled/baseline_inventory_decomposition/autoclust/provenance.json` |
| `results_analysis/clustering_repository/external_baselines/AutoClust_metric_decomposition_split1/runtime_comparison.csv` | `results/controlled/baseline_inventory_decomposition/autoclust/runtime_comparison.csv` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/coverage_audit.csv` | `results/controlled/baseline_inventory_decomposition/ml2dac/coverage_audit.csv` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/experiment_manifest.json` | `results/controlled/baseline_inventory_decomposition/ml2dac/experiment_manifest.json` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/factorial_effects.json` | `results/controlled/baseline_inventory_decomposition/ml2dac/factorial_effects.json` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/factorial_uncertainty.csv` | `results/controlled/baseline_inventory_decomposition/ml2dac/factorial_uncertainty.csv` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/four_arm_dataset_results.csv.gz` | `results/controlled/baseline_inventory_decomposition/ml2dac/four_arm_dataset_results.csv.gz` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/four_arm_family_summary.csv` | `results/controlled/baseline_inventory_decomposition/ml2dac/four_arm_family_summary.csv` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/four_arm_global_summary.csv` | `results/controlled/baseline_inventory_decomposition/ml2dac/four_arm_global_summary.csv` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/four_arm_subfamily_summary.csv` | `results/controlled/baseline_inventory_decomposition/ml2dac/four_arm_subfamily_summary.csv` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/paired_comparisons.csv` | `results/controlled/baseline_inventory_decomposition/ml2dac/paired_comparisons.csv` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/per_dataset_factorial_terms.csv.gz` | `results/controlled/baseline_inventory_decomposition/ml2dac/per_dataset_factorial_terms.csv.gz` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/provenance.json` | `results/controlled/baseline_inventory_decomposition/ml2dac/provenance.json` |
| `results_analysis/clustering_repository/external_baselines/ML2DAC_metric_decomposition_split1/runtime_comparison.csv` | `results/controlled/baseline_inventory_decomposition/ml2dac/runtime_comparison.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_controlled_criterion_baselines.csv` | `results/controlled/criterion_baselines/canonical_frame/controlled_criterion_baselines.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_controlled_family_stats.csv` | `results/controlled/criterion_baselines/canonical_frame/controlled_family_stats.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_controlled_headline_stats.csv` | `results/controlled/criterion_baselines/canonical_frame/controlled_headline_stats.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_controlled_provenance.json` | `results/controlled/criterion_baselines/canonical_frame/controlled_provenance.json` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_criterion_baseline_notes.json` | `results/controlled/criterion_baselines/canonical_frame/criterion_baseline_notes.json` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_final_arm_summary.csv` | `results/controlled/criterion_baselines/seeded_sweep/arm_summary.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_config_build_manifest.json` | `results/controlled/criterion_baselines/seeded_sweep/config_build_manifest.json` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_final_dataset_results.csv` | `results/controlled/criterion_baselines/seeded_sweep/dataset_results.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/development_best_single_index.csv` | `results/controlled/criterion_baselines/seeded_sweep/development_best_single_index.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/development_core3.csv` | `results/controlled/criterion_baselines/seeded_sweep/development_core3.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/development_derivation_manifest.json` | `results/controlled/criterion_baselines/seeded_sweep/development_derivation_manifest.json` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/development_mean_profile.csv` | `results/controlled/criterion_baselines/seeded_sweep/development_mean_profile.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_final_family_contrasts.csv` | `results/controlled/criterion_baselines/seeded_sweep/family_contrasts.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_final_hypotheses.json` | `results/controlled/criterion_baselines/seeded_sweep/hypotheses.json` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_final_paired_contrasts.csv` | `results/controlled/criterion_baselines/seeded_sweep/paired_contrasts.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_run_level_results.csv` | `results/controlled/criterion_baselines/seeded_sweep/run_level_results.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_run_manifest.json` | `results/controlled/criterion_baselines/seeded_sweep/run_manifest.json` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_final_sample_vs_full.csv` | `results/controlled/criterion_baselines/seeded_sweep/sample_vs_full.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/e1_conditioning_decision/e1_seed_manifest.json` | `results/controlled/criterion_baselines/seeded_sweep/seed_manifest.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/autoclust_family_group_reliance.csv` | `results/controlled/family_analysis/autoclust_family_group_reliance.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/autoclust_group_reliance.csv` | `results/controlled/family_analysis/autoclust_group_reliance.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/autoclust_metric_reliance.csv` | `results/controlled/family_analysis/autoclust_metric_reliance.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/autoclust_reliance_provenance.json` | `results/controlled/family_analysis/autoclust_reliance_provenance.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/stage1_bundle_context.json` | `results/controlled/family_analysis/bundle_context.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/canonical_metric_atlas.csv` | `results/controlled/family_analysis/canonical_metric_atlas.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/clustopt_family_metric_usage.csv` | `results/controlled/family_analysis/clustopt_family_metric_usage.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/clustopt_metric_prediction_quality.csv` | `results/controlled/family_analysis/clustopt_metric_prediction_quality.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/clustopt_metric_usage.csv` | `results/controlled/family_analysis/clustopt_metric_usage.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/comparator_coverage.json` | `results/controlled/family_analysis/comparator_coverage.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/dead_metric_registry.csv` | `results/controlled/family_analysis/dead_metric_registry.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/experiment_manifest.json` | `results/controlled/family_analysis/experiment_manifest.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/external_metric_validation_plan.json` | `results/controlled/family_analysis/external_metric_validation_plan.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_publication_table.csv` | `results/controlled/family_analysis/family_publication_table.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/group_oracle_coverage.csv` | `results/controlled/family_analysis/group_oracle_coverage.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_family_utility.csv.gz` | `results/controlled/family_analysis/metric_family_utility.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_global_utility.csv` | `results/controlled/family_analysis/metric_global_utility.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_group_contrasts.csv` | `results/controlled/family_analysis/metric_group_contrasts.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_group_summary.csv` | `results/controlled/family_analysis/metric_group_summary.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_group_utility_summary.csv` | `results/controlled/family_analysis/metric_group_utility_summary.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_publication_table.csv` | `results/controlled/family_analysis/metric_publication_table.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_subfamily_utility.csv.gz` | `results/controlled/family_analysis/metric_subfamily_utility.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_topk_coverage.csv` | `results/controlled/family_analysis/metric_topk_coverage.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_utility_definition.json` | `results/controlled/family_analysis/metric_utility_definition.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/metric_view_level_utility_descriptive.csv` | `results/controlled/family_analysis/metric_view_level_utility_descriptive.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/ml2dac_cvi_star_diagnostics.csv` | `results/controlled/family_analysis/ml2dac_cvi_star_diagnostics.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/ml2dac_family_metric_selection.csv` | `results/controlled/family_analysis/ml2dac_family_metric_selection.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/ml2dac_split1_cvi_star.csv.gz` | `results/controlled/family_analysis/ml2dac_heldout_cvi_star.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/ml2dac_metric_selection.csv` | `results/controlled/family_analysis/ml2dac_metric_selection.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/ml2dac_per_cvi_recall.csv` | `results/controlled/family_analysis/ml2dac_per_cvi_recall.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/ml2dac_selection_records.csv.gz` | `results/controlled/family_analysis/ml2dac_selection_records.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/multiple_testing_summary.json` | `results/controlled/family_analysis/multiple_testing_summary.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/new46_complementarity.csv` | `results/controlled/family_analysis/new46_complementarity.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/new46_unique_win_coverage.csv` | `results/controlled/family_analysis/new46_unique_win_coverage.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/part1_manifest.json` | `results/controlled/family_analysis/part1_manifest.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/part2_manifest.json` | `results/controlled/family_analysis/part2_manifest.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/provenance.json` | `results/controlled/family_analysis/provenance.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/publication_path_assessment.json` | `results/controlled/family_analysis/publication_path_assessment.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/autoclust_family_group_reliance.csv` | `results/controlled/family_analysis/specialization/autoclust_family_group_reliance.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/clustopt_family_downstream_gain.csv` | `results/controlled/family_analysis/specialization/clustopt_family_downstream_gain.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/clustopt_subfamily_downstream_gain.csv` | `results/controlled/family_analysis/specialization/clustopt_subfamily_downstream_gain.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/core_candidates.csv` | `results/controlled/family_analysis/specialization/core_candidates.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/core_definition.json` | `results/controlled/family_analysis/specialization/core_definition.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/cross_method_family_mechanism.csv` | `results/controlled/family_analysis/specialization/cross_method_family_mechanism.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/cross_method_subfamily_mechanism.csv.gz` | `results/controlled/family_analysis/specialization/cross_method_subfamily_mechanism.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/downstream_source.json` | `results/controlled/family_analysis/specialization/downstream_source.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/experiment_manifest.json` | `results/controlled/family_analysis/specialization/experiment_manifest.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/exploited_vs_missed_specialists.csv` | `results/controlled/family_analysis/specialization/exploited_vs_missed_specialists.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_core_breadth.csv` | `results/controlled/family_analysis/specialization/family_core_breadth.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_dataset_counts.csv` | `results/controlled/family_analysis/specialization/family_dataset_counts.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_metric_full_matrix.csv.gz` | `results/controlled/family_analysis/specialization/family_metric_full_matrix.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_metric_selection.csv.gz` | `results/controlled/family_analysis/specialization/family_metric_selection.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_metric_structure_permutation.csv` | `results/controlled/family_analysis/specialization/family_metric_structure_permutation.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_metric_utility.csv.gz` | `results/controlled/family_analysis/specialization/family_metric_utility.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_predicted_vs_true.csv` | `results/controlled/family_analysis/specialization/family_predicted_vs_true.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_signature.csv` | `results/controlled/family_analysis/specialization/family_signature.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_signature.json` | `results/controlled/family_analysis/specialization/family_signature.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_specialist_jaccard.csv` | `results/controlled/family_analysis/specialization/family_specialist_jaccard.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_specialists_tierA.csv` | `results/controlled/family_analysis/specialization/family_specialists_tierA.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_specialists_tierB.csv` | `results/controlled/family_analysis/specialization/family_specialists_tierB.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_top5_selection_jaccard.csv` | `results/controlled/family_analysis/specialization/family_top5_selection_jaccard.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_top5_utility_jaccard.csv` | `results/controlled/family_analysis/specialization/family_top5_utility_jaccard.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_true_topk_recall.csv` | `results/controlled/family_analysis/specialization/family_true_topk_recall.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/family_utility_selection_alignment.csv` | `results/controlled/family_analysis/specialization/family_utility_selection_alignment.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/figures/fig1_family_utility_rank_heatmap.png` | `results/controlled/family_analysis/specialization/figures/fig1_family_utility_rank_heatmap.png` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/figures/fig2_family_selection_enrichment.png` | `results/controlled/family_analysis/specialization/figures/fig2_family_selection_enrichment.png` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/figures/fig3_new46_specialization_heatmap.png` | `results/controlled/family_analysis/specialization/figures/fig3_new46_specialization_heatmap.png` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/figures/fig4_core_vs_specialist_map.png` | `results/controlled/family_analysis/specialization/figures/fig4_core_vs_specialist_map.png` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/figures/fig5_utility_selection_alignment.png` | `results/controlled/family_analysis/specialization/figures/fig5_utility_selection_alignment.png` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/figures/fig6_family_signature.png` | `results/controlled/family_analysis/specialization/figures/fig6_family_signature.png` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/figures/fig7_specialization_vs_downstream.png` | `results/controlled/family_analysis/specialization/figures/fig7_specialization_vs_downstream.png` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/figures/fig8_subfamily_specialization.png` | `results/controlled/family_analysis/specialization/figures/fig8_subfamily_specialization.png` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/hypothesis_verdict.json` | `results/controlled/family_analysis/specialization/hypothesis_verdict.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/ml2dac_family_specialist_capture.csv` | `results/controlled/family_analysis/specialization/ml2dac_family_specialist_capture.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/ml2dac_subfamily_specialist_capture.csv.gz` | `results/controlled/family_analysis/specialization/ml2dac_subfamily_specialist_capture.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/pattern_vs_image_family.csv` | `results/controlled/family_analysis/specialization/pattern_vs_image_family.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/pattern_vs_image_subfamily.csv.gz` | `results/controlled/family_analysis/specialization/pattern_vs_image_subfamily.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/provenance.json` | `results/controlled/family_analysis/specialization/provenance.json` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/specialization_downstream_association.csv` | `results/controlled/family_analysis/specialization/specialization_downstream_association.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_dataset_counts.csv` | `results/controlled/family_analysis/specialization/subfamily_dataset_counts.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_downstream_association_hierarchical.csv` | `results/controlled/family_analysis/specialization/subfamily_downstream_association_hierarchical.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_metric_full_matrix.csv.gz` | `results/controlled/family_analysis/specialization/subfamily_metric_full_matrix.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_metric_selection.csv.gz` | `results/controlled/family_analysis/specialization/subfamily_metric_selection.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_metric_structure_permutation.csv` | `results/controlled/family_analysis/specialization/subfamily_metric_structure_permutation.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_metric_utility.csv.gz` | `results/controlled/family_analysis/specialization/subfamily_metric_utility.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_predicted_vs_true.csv` | `results/controlled/family_analysis/specialization/subfamily_predicted_vs_true.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_signature.csv.gz` | `results/controlled/family_analysis/specialization/subfamily_signature.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_signature.json.gz` | `results/controlled/family_analysis/specialization/subfamily_signature.json.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_specialists_tierA.csv` | `results/controlled/family_analysis/specialization/subfamily_specialists_tierA.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_specialists_tierB.csv` | `results/controlled/family_analysis/specialization/subfamily_specialists_tierB.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_true_topk_recall.csv` | `results/controlled/family_analysis/specialization/subfamily_true_topk_recall.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/subfamily_utility_selection_alignment.csv` | `results/controlled/family_analysis/specialization/subfamily_utility_selection_alignment.csv` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/family_subfamily_specialization/view_specific_specialization.csv.gz` | `results/controlled/family_analysis/specialization/view_specific_specialization.csv.gz` |
| `results_analysis/clustering_repository/metric_analysis_existing_domain/subfamily_publication_table.csv.gz` | `results/controlled/family_analysis/subfamily_publication_table.csv.gz` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/experiment_manifest.json` | `results/controlled/inventory_factorial/fixed_budget_32/experiment_manifest.json` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_family_metric_attribution.csv` | `results/controlled/inventory_factorial/fixed_budget_32/family_metric_attribution.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_dataset_results.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_dataset_results.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_family_summary.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_family_summary.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_family_table.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_family_table.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_overall_summary.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_overall_summary.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_paired_contrasts.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_paired_contrasts.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_per_view_summary.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_per_view_summary.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_record_results.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_record_results.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_subfamily_summary.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_subfamily_summary.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_tie_analysis.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_tie_analysis.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/offline_2x3_winner_counts.csv` | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_winner_counts.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_group_prediction_quality.csv` | `results/controlled/inventory_factorial/fixed_budget_32/group_prediction_quality.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_metric_attribution.csv` | `results/controlled/inventory_factorial/fixed_budget_32/metric_attribution.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_metric_set_overlap.csv` | `results/controlled/inventory_factorial/fixed_budget_32/metric_set_overlap.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_new46_oracle_misses.csv` | `results/controlled/inventory_factorial/fixed_budget_32/new46_oracle_misses.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_summary.json` | `results/controlled/inventory_factorial/fixed_budget_32/summary.json` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_tie_plateau_analysis.csv` | `results/controlled/inventory_factorial/fixed_budget_32/tie_plateau_analysis.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_topk_composition.csv` | `results/controlled/inventory_factorial/fixed_budget_32/topk_composition.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32/stage1d_view_metric_attribution.csv` | `results/controlled/inventory_factorial/fixed_budget_32/view_metric_attribution.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/experiment_manifest.json` | `results/controlled/inventory_factorial/online_search_50/experiment_manifest.json` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_best_view_choice.csv` | `results/controlled/inventory_factorial/online_search_50/online_best_view_choice.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_best_visited_contrasts.csv` | `results/controlled/inventory_factorial/online_search_50/online_best_visited_contrasts.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_dataset_results.csv` | `results/controlled/inventory_factorial/online_search_50/online_dataset_results.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_family_summary.csv` | `results/controlled/inventory_factorial/online_search_50/online_family_summary.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_family_table.csv` | `results/controlled/inventory_factorial/online_search_50/online_family_table.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_overall_summary.csv` | `results/controlled/inventory_factorial/online_search_50/online_overall_summary.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_paired_contrasts.csv` | `results/controlled/inventory_factorial/online_search_50/online_paired_contrasts.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_per_view_summary.csv` | `results/controlled/inventory_factorial/online_search_50/online_per_view_summary.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_record_results.csv` | `results/controlled/inventory_factorial/online_search_50/online_record_results.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_reranker_evidence.csv` | `results/controlled/inventory_factorial/online_search_50/online_reranker_evidence.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_search_selection_decomposition.csv` | `results/controlled/inventory_factorial/online_search_50/online_search_selection_decomposition.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_subfamily_summary.csv` | `results/controlled/inventory_factorial/online_search_50/online_subfamily_summary.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_trial_accounting.csv` | `results/controlled/inventory_factorial/online_search_50/online_trial_accounting.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_vs_offline_comparison.csv` | `results/controlled/inventory_factorial/online_search_50/online_vs_offline_comparison.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/online_2x3_winner_counts.csv` | `results/controlled/inventory_factorial/online_search_50/online_winner_counts.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1e_subfamily_progress.csv` | `results/controlled/inventory_factorial/online_search_50/subfamily_progress.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a3_a4_bestview_upper_bounds.csv` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/bestview_upper_bounds.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a12_fp_vs_fo_ratios.json` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/fp_vs_fo_ratios.json` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a6_stage2_mechanism_comparison.csv` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/mechanism_comparison.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a5_policy_vbs.csv` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/policy_vbs.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a9_pool_ceiling_comparison.json` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/pool_ceiling_comparison.json` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a7_prespecified_contrasts.csv` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/prespecified_contrasts.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a2_regret_aggregation_audit.csv` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/regret_aggregation_audit.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a11_search_vs_selection_ratio.csv` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/search_vs_selection_ratio.csv` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/stage1f_summary.json` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/summary.json` |
| `results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50/stage1f/a8_winner_counts.csv` | `results/controlled/inventory_factorial/online_search_50/upper_bounds/winner_counts.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/aggregation_variants.csv` | `results/controlled/matched_autoclust/aggregation_variants.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/all_policies_vs_primary_comparator.csv` | `results/controlled/matched_autoclust/all_policies_vs_primary_comparator.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/audit_summary.json` | `results/controlled/matched_autoclust/audit_summary.json` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/stage2b5d_autoclust_arms_bestview.csv.gz` | `results/controlled/matched_autoclust/autoclust_arms_bestview.csv.gz` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/autoclust_split1_raw_rows.csv.gz` | `results/controlled/matched_autoclust/autoclust_heldout_raw_rows.csv.gz` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/comparison_layer_recomputation.json` | `results/controlled/matched_autoclust/comparison_layer_recomputation.json` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/configuration_comparison_table.csv` | `results/controlled/matched_autoclust/configuration_comparison_table.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/configuration_identity_audit.json` | `results/controlled/matched_autoclust/configuration_identity_audit.json` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/configuration_identity_summary.json` | `results/controlled/matched_autoclust/configuration_identity_summary.json` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/dataset_population_comparison.csv` | `results/controlled/matched_autoclust/dataset_population_comparison.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/extended_pair_per_dataset_bestview.csv` | `results/controlled/matched_autoclust/extended_pair_per_dataset_bestview.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/stage2b5d_final_comparison_table.csv` | `results/controlled/matched_autoclust/final_comparison_table.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/historical_value_matches.json` | `results/controlled/matched_autoclust/historical_value_matches.json` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/per_dataset_bestview__AutoClust_Extended_InDomain_same_search_space.csv` | `results/controlled/matched_autoclust/per_dataset_bestview_autoclust_extended_matched.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/per_dataset_bestview__AutoClust_Extended_InDomain.csv` | `results/controlled/matched_autoclust/per_dataset_bestview_autoclust_extended_native.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/recomputed_autoclust_comparison.csv` | `results/controlled/matched_autoclust/recomputed_autoclust_comparison.csv` |
| `results_analysis/clustering_repository/autoclust_extended_result_provenance_audit/stage2b5d_resolution.json` | `results/controlled/matched_autoclust/resolution.json` |
| `results_analysis/clustering_repository/paper_analysis_v2/13_raw_and_reproducibility/canonical_best_view_frame.parquet (path columns dropped)` | `results/controlled/per_dataset/controlled_best_view_results.parquet` |
| `results_analysis/clustering_repository/paper_analysis_v2/13_raw_and_reproducibility/canonical_run_frame.parquet (path columns dropped)` | `results/controlled/per_dataset/controlled_run_results.parquet` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/autoclust_matched_comparison.csv` | `results/controlled/policy_selection/autoclust_matched_comparison.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/autoclust_native_comparison.csv` | `results/controlled/policy_selection/autoclust_native_comparison.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/development_selected_fixed_policy.json` | `results/controlled/policy_selection/development_selected_fixed_policy.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/experiment_manifest.json` | `results/controlled/policy_selection/experiment_manifest.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/family_analysis.csv` | `results/controlled/policy_selection/family_analysis.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/stage2c3_final_summary.json` | `results/controlled/policy_selection/final_summary.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/final_v1_model_manifest.json` | `results/controlled/policy_selection/final_v1_model_manifest.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/final_v2_model_manifest.json` | `results/controlled/policy_selection/final_v2_model_manifest.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/four_level_online_table.csv` | `results/controlled/policy_selection/four_level_online_table.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/split1_feature_manifest.json` | `results/controlled/policy_selection/heldout_feature_manifest.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/split1_v1_inference_runtime.json` | `results/controlled/policy_selection/heldout_v1_inference_runtime.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/split1_v1_policy_selections.csv.gz` | `results/controlled/policy_selection/heldout_v1_policy_selections.csv.gz` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/split1_v2_inference_runtime.json` | `results/controlled/policy_selection/heldout_v2_inference_runtime.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/split1_v2_policy_selections.csv.gz` | `results/controlled/policy_selection/heldout_v2_policy_selections.csv.gz` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/hindsight_fixed_diagnostic.json` | `results/controlled/policy_selection/hindsight_fixed_diagnostic.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/online_headroom_recovery.json` | `results/controlled/policy_selection/online_headroom_recovery.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/online_near_optimality.csv` | `results/controlled/policy_selection/online_near_optimality.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/per_dataset_online_results.csv.gz` | `results/controlled/policy_selection/per_dataset_online_results.csv.gz` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/per_view_online_results.csv.gz` | `results/controlled/policy_selection/per_view_online_results.csv.gz` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/policy_selection_frequencies.csv` | `results/controlled/policy_selection/policy_selection_frequencies.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/pre_outcome_selection_manifest.json` | `results/controlled/policy_selection/pre_outcome_selection_manifest.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/raw_softmax_online_diagnostic.csv` | `results/controlled/policy_selection/raw_softmax_online_diagnostic.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/raw_softmax_online_summary.json` | `results/controlled/policy_selection/raw_softmax_online_summary.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/runtime_summary.json` | `results/controlled/policy_selection/runtime_summary.json` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/subfamily_analysis.csv` | `results/controlled/policy_selection/subfamily_analysis.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/v2_vs_fixed.csv` | `results/controlled/policy_selection/v2_vs_fixed.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/v2_vs_v1.csv` | `results/controlled/policy_selection/v2_vs_v1.csv` |
| `results_analysis/clustopt_policy_predictor/stage2c_final_v2_split1_online_replay/view_analysis.csv` | `results/controlled/policy_selection/view_analysis.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_predictor_alignment_provenance.json` | `results/controlled/predictor_alignment/predictor_alignment_provenance.json` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_predictor_metrics.csv` | `results/controlled/predictor_alignment/predictor_metrics.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_selection_alignment.csv` | `results/controlled/predictor_alignment/selection_alignment.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e8_selection_alignment_per_family.csv` | `results/controlled/predictor_alignment/selection_alignment_per_family.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/configuration_ranking.csv` | `results/controlled/predictor_evaluation/knn/configuration_ranking.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/configuration_summary.csv` | `results/controlled/predictor_evaluation/knn/configuration_summary.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/dynamic_k_confusion_split1.csv` | `results/controlled/predictor_evaluation/knn/dynamic_k_confusion_heldout.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/dynamic_k_confusion_oof.csv` | `results/controlled/predictor_evaluation/knn/dynamic_k_confusion_oof.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/dynamic_k_metrics.csv` | `results/controlled/predictor_evaluation/knn/dynamic_k_metrics.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/family_metrics.csv` | `results/controlled/predictor_evaluation/knn/family_metrics.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/fold_metrics.csv` | `results/controlled/predictor_evaluation/knn/fold_metrics.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/fold_runtime.csv` | `results/controlled/predictor_evaluation/knn/fold_runtime.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/split1_locked_metrics.csv` | `results/controlled/predictor_evaluation/knn/heldout_locked_metrics.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/mlp_reference_comparison.csv` | `results/controlled/predictor_evaluation/knn/mlp_reference_comparison.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/per_metric_errors.csv` | `results/controlled/predictor_evaluation/knn/per_metric_errors.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/selected_configuration_fold_metrics.csv` | `results/controlled/predictor_evaluation/knn/selected_configuration_fold_metrics.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/subfamily_metrics.csv` | `results/controlled/predictor_evaluation/knn/subfamily_metrics.csv` |
| `results_analysis/metric_utility_knn/phase_e1/tables/view_metrics.csv` | `results/controlled/predictor_evaluation/knn/view_metrics.csv` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/artifacts/grouped_mean_by_view_id.csv` | `results/controlled/predictor_evaluation/neural_full60_grouped_mean_by_view_id.csv` |
| `results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout/overall_results.csv` | `results/controlled/predictor_evaluation/neural_full60_overall_results.csv` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/artifacts/grouped_mean_by_view_id.csv` | `results/controlled/predictor_evaluation/neural_head14_grouped_mean_by_view_id.csv` |
| `results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout/overall_results.csv` | `results/controlled/predictor_evaluation/neural_head14_overall_results.csv` |
| (internal evaluation report table, identical copy) | `results/controlled/predictor_evaluation/neural_head14_vs_full60.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/algorithm_domain_shift.csv` | `results/controlled/reranking/algorithm_domain_shift.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/stage2b5c_authoritative_values.json` | `results/controlled/reranking/authoritative_values.json` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/autoclust_bestview.csv.gz` | `results/controlled/reranking/autoclust_bestview.csv.gz` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/autoclust_performance_comparison.csv` | `results/controlled/reranking/autoclust_performance_comparison.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/best_fixed_original_policy.json` | `results/controlled/reranking/best_fixed_original_policy.json` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/best_fixed_reranked_policy.json` | `results/controlled/reranking/best_fixed_reranked_policy.json` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/domain_shift_strata.csv` | `results/controlled/reranking/domain_shift_strata.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/experiment_manifest.json` | `results/controlled/reranking/experiment_manifest.json` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/stage2b5c_final_audit.csv` | `results/controlled/reranking/final_audit.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/stage2b5_final_summary.json` | `results/controlled/reranking/final_summary.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/autoclust_runtime_comparability.json` | `results/controlled/reranking/heldout_preparation/autoclust_runtime_comparability.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/development_reproduction_checks.csv` | `results/controlled/reranking/heldout_preparation/development_reproduction_checks.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/experiment_manifest.json` | `results/controlled/reranking/heldout_preparation/experiment_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_DYNAMIC/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/KNN_DYNAMIC/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_DYNAMIC/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/KNN_DYNAMIC/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP1/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/KNN_TOP1/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP1/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/KNN_TOP1/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP10/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/KNN_TOP10/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP10/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/KNN_TOP10/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP3/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/KNN_TOP3/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP3/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/KNN_TOP3/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP5/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/KNN_TOP5/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/KNN_TOP5/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/KNN_TOP5/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_DYNAMIC/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/MLP_DYNAMIC/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_DYNAMIC/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/MLP_DYNAMIC/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP1/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/MLP_TOP1/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP1/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/MLP_TOP1/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP10/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/MLP_TOP10/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP10/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/MLP_TOP10/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP3/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/MLP_TOP3/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP3/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/MLP_TOP3/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP5/model.joblib` | `results/controlled/reranking/heldout_preparation/final_models/MLP_TOP5/model.joblib` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models/MLP_TOP5/model_manifest.json` | `results/controlled/reranking/heldout_preparation/final_models/MLP_TOP5/model_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_coverage_by_policy.csv` | `results/controlled/reranking/heldout_preparation/heldout_coverage_by_policy.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_coverage_by_subfamily.csv` | `results/controlled/reranking/heldout_preparation/heldout_coverage_by_subfamily.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_coverage_by_view.csv` | `results/controlled/reranking/heldout_preparation/heldout_coverage_by_view.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_eligible_bucket_distribution.csv` | `results/controlled/reranking/heldout_preparation/heldout_eligible_bucket_distribution.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_feature_availability.csv` | `results/controlled/reranking/heldout_preparation/heldout_feature_availability.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_final_evaluation_protocol.json` | `results/controlled/reranking/heldout_preparation/heldout_final_evaluation_protocol.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_knn_utility_context.csv.gz` | `results/controlled/reranking/heldout_preparation/heldout_knn_utility_context.csv.gz` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_knn_utility_context_manifest.json` | `results/controlled/reranking/heldout_preparation/heldout_knn_utility_context_manifest.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_per_policy_trial_counts.csv` | `results/controlled/reranking/heldout_preparation/heldout_per_policy_trial_counts.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_per_view_trial_counts.csv` | `results/controlled/reranking/heldout_preparation/heldout_per_view_trial_counts.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_result_schema.json` | `results/controlled/reranking/heldout_preparation/heldout_result_schema.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_run_coverage.csv` | `results/controlled/reranking/heldout_preparation/heldout_run_coverage.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_run_to_reranker_manifest.csv.gz` | `results/controlled/reranking/heldout_preparation/heldout_run_to_reranker_manifest.csv.gz` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_timeout_summary.csv` | `results/controlled/reranking/heldout_preparation/heldout_timeout_summary.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_trace_coverage.csv.gz` | `results/controlled/reranking/heldout_preparation/heldout_trace_coverage.csv.gz` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_trial_count_distribution.csv` | `results/controlled/reranking/heldout_preparation/heldout_trial_count_distribution.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_utility_context_manifest.csv.gz` | `results/controlled/reranking/heldout_preparation/heldout_utility_context_manifest.csv.gz` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/split1_utility_context_readiness.csv` | `results/controlled/reranking/heldout_preparation/heldout_utility_context_readiness.csv` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/online_trace_timeout_semantics.json` | `results/controlled/reranking/heldout_preparation/online_trace_timeout_semantics.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/original_runtime_semantics.json` | `results/controlled/reranking/heldout_preparation/original_runtime_semantics.json` |
| `results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/stage2b5a_summary.json` | `results/controlled/reranking/heldout_preparation/summary.json` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/hybrid_visited_oracle.json` | `results/controlled/reranking/hybrid_visited_oracle.json` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/mlp_vs_knn.csv` | `results/controlled/reranking/mlp_vs_knn.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/original_vs_reranked.csv` | `results/controlled/reranking/original_vs_reranked.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/per_dataset_policy_results.csv.gz` | `results/controlled/reranking/per_dataset_policy_results.csv.gz` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/per_family_results.csv` | `results/controlled/reranking/per_family_results.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/per_policy_results.csv` | `results/controlled/reranking/per_policy_results.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/per_run_reranker_results.csv.gz` | `results/controlled/reranking/per_run_reranker_results.csv.gz` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/per_subfamily_results.csv` | `results/controlled/reranking/per_subfamily_results.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/raw_vs_softmax.csv` | `results/controlled/reranking/raw_vs_softmax.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/reranker_assisted_vbs.json` | `results/controlled/reranking/reranker_assisted_vbs.json` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/reranker_recovery.csv` | `results/controlled/reranking/reranker_recovery.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/runtime_candidate_count_strata.csv` | `results/controlled/reranking/runtime_candidate_count_strata.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/runtime_per_family.csv` | `results/controlled/reranking/runtime_per_family.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/runtime_per_policy.csv` | `results/controlled/reranking/runtime_per_policy.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/runtime_per_subfamily.csv` | `results/controlled/reranking/runtime_per_subfamily.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/runtime_per_view.csv` | `results/controlled/reranking/runtime_per_view.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/runtime_summary.json` | `results/controlled/reranking/runtime_summary.json` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/seen_family_oracle_gap.csv` | `results/controlled/reranking/seen_family_oracle_gap.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/statistical_comparisons.csv` | `results/controlled/reranking/statistical_comparisons.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/timeout_slate_size_analysis.csv` | `results/controlled/reranking/timeout_slate_size_analysis.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/timeout_vs_full_budget.csv` | `results/controlled/reranking/timeout_vs_full_budget.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/top5_vs_top10.csv` | `results/controlled/reranking/top5_vs_top10.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/unseen_family_selection_rates.csv` | `results/controlled/reranking/unseen_family_selection_rates.csv` |
| `results_analysis/clustering_repository/candidate_reranker_split1_online_final/visited_oracle_summary.csv` | `results/controlled/reranking/visited_oracle_summary.csv` |

## `results/external`

| original | public |
|---|---|
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/generalist_core_review.csv` | `results/external/family_analysis/generalist_core_review.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/new46_specialist_table.csv` | `results/external/family_analysis/new46_specialist_table.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/new46_vs_reference_groups.json` | `results/external/family_analysis/new46_vs_reference_groups.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_bestview_ari_vs_three_view_runtime.csv` | `results/external/figure_data/bestview_ari_vs_three_view_runtime.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/external_figure4_c4_pairwise.csv` | `results/external/figure_data/knn_rerank_pairwise.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_method_bestview_distribution.csv` | `results/external/figure_data/method_bestview_distribution.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_metric_rank_per_unit_heatmap.csv.gz` | `results/external/figure_data/metric_rank_per_unit_heatmap.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_metric_utility_vs_coverage.csv` | `results/external/figure_data/metric_utility_vs_coverage.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_paired_c5_vs_baselines.csv` | `results/external/figure_data/paired_c5_vs_baselines.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_reranker_ari_delta_vs_runtime_delta.csv` | `results/external/figure_data/reranker_ari_delta_vs_runtime_delta.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_reranker_delta_distribution.csv` | `results/external/figure_data/reranker_delta_distribution.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_runtime_vs_ari.csv` | `results/external/figure_data/runtime_vs_ari.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_source_method_comparison.csv` | `results/external/figure_data/source_method_comparison.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/figure_data/fig_xy2d_ari_vs_runtime.csv` | `results/external/figure_data/xy2d_ari_vs_runtime.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c/fixed32_candidate_ari.csv.gz` | `results/external/index_analysis/authoritative/candidate_ari.csv.gz` |
| `(this release validation)` | `results/external/index_analysis/authoritative/cvnn_fpc_utility_reproduction.csv` |
| `(this release validation)` | `results/external/index_analysis/authoritative/cvnn_frozen_values.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/generalist_core_external.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/generalist_core_external.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/joint_pipeline_per_unit.csv.gz` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/joint_pipeline_per_unit.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/joint_pipeline_summary.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/joint_pipeline_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/metric_aggregate_summary.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/metric_aggregate_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/metric_group_summary.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/metric_group_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/metric_rank_table.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/metric_rank_table.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/metric_rank_table_full_coverage_only.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/metric_rank_table_full_coverage_only.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/metric_source_summary.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/metric_source_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/metric_specialisation_spread.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/metric_specialisation_spread.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/metric_top5_frequency.csv` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/metric_top5_frequency.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/metric_utility_per_dataset_view.csv.gz` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/metric_utility_per_dataset_view.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/modern_comparator_analysis.json` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/modern_comparator_analysis.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_canonical_utility/package_manifest.json` | `results/external/index_analysis/authoritative/generalist_core_and_pipeline/package_manifest.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/joint_pipeline_c5_units.csv` | `results/external/index_analysis/authoritative/joint_pipeline_clustopt_ppv2_rerank_units.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/joint_pipeline_quadrants.csv` | `results/external/index_analysis/authoritative/joint_pipeline_quadrants.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/joint_pipeline_summary.csv` | `results/external/index_analysis/authoritative/joint_pipeline_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/metric_master_table.csv` | `results/external/index_analysis/authoritative/metric_master_table.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/metric_slice_all.csv` | `results/external/index_analysis/authoritative/metric_slice_all.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/metric_slice_coverage_100.csv` | `results/external/index_analysis/authoritative/metric_slice_coverage_100.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/metric_slice_coverage_ge_090.csv` | `results/external/index_analysis/authoritative/metric_slice_coverage_ge_090.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/metric_slice_coverage_ge_095.csv` | `results/external/index_analysis/authoritative/metric_slice_coverage_ge_095.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/modern_comparator_review.json` | `results/external/index_analysis/authoritative/modern_comparator_review.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c_canonical/canonical_utility_summary.json` | `results/external/index_analysis/authoritative/utility_canonical_basis/canonical_utility_summary.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c_canonical/impact_vs_previous.csv.gz` | `results/external/index_analysis/authoritative/utility_canonical_basis/impact_vs_previous.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c_canonical/metric_utility_canonical.csv.gz` | `results/external/index_analysis/authoritative/utility_canonical_basis/metric_utility_canonical.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/generalist_core_external.csv` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/generalist_core_external.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/joint_pipeline_per_unit.csv.gz` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/joint_pipeline_per_unit.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/joint_pipeline_summary.csv` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/joint_pipeline_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/metric_aggregate_summary.csv` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/metric_aggregate_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/metric_group_summary.csv` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/metric_group_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/metric_specialisation_spread.csv` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/metric_specialisation_spread.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/metric_top5_frequency.csv` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/metric_top5_frequency.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/metric_utility_per_dataset_view.csv.gz` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/metric_utility_per_dataset_view.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/modern_comparator_analysis.json` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/modern_comparator_analysis.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d_corrected/phase_d_corrected_summary.json` | `results/external/index_analysis/superseded/generalist_core_corrected_dispatch_pre_canonical_basis/phase_d_corrected_summary.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/fixed32_candidate_ari.csv.gz` | `results/external/index_analysis/superseded/generalist_core_initial/fixed32_candidate_ari.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/fixed32_candidate_validity.csv` | `results/external/index_analysis/superseded/generalist_core_initial/fixed32_candidate_validity.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/generalist_core_external.csv` | `results/external/index_analysis/superseded/generalist_core_initial/generalist_core_external.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/joint_pipeline_per_unit.csv.gz` | `results/external/index_analysis/superseded/generalist_core_initial/joint_pipeline_per_unit.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/joint_pipeline_summary.csv` | `results/external/index_analysis/superseded/generalist_core_initial/joint_pipeline_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/method_aggregate_summary.csv` | `results/external/index_analysis/superseded/generalist_core_initial/method_aggregate_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/method_bestview_per_dataset.csv.gz` | `results/external/index_analysis/superseded/generalist_core_initial/method_bestview_per_dataset.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/method_failure_summary.csv` | `results/external/index_analysis/superseded/generalist_core_initial/method_failure_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/method_per_dataset_view.csv.gz` | `results/external/index_analysis/superseded/generalist_core_initial/method_per_dataset_view.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/method_runtime_summary.csv` | `results/external/index_analysis/superseded/generalist_core_initial/method_runtime_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/method_xy2d_per_dataset.csv.gz` | `results/external/index_analysis/superseded/generalist_core_initial/method_xy2d_per_dataset.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/metric_aggregate_summary.csv` | `results/external/index_analysis/superseded/generalist_core_initial/metric_aggregate_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/metric_group_summary.csv` | `results/external/index_analysis/superseded/generalist_core_initial/metric_group_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/metric_specialisation_spread.csv` | `results/external/index_analysis/superseded/generalist_core_initial/metric_specialisation_spread.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/metric_top5_frequency.csv` | `results/external/index_analysis/superseded/generalist_core_initial/metric_top5_frequency.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/metric_utility_per_dataset_view.csv.gz` | `results/external/index_analysis/superseded/generalist_core_initial/metric_utility_per_dataset_view.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/modern_comparator_analysis.json` | `results/external/index_analysis/superseded/generalist_core_initial/modern_comparator_analysis.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_d/phase_d_summary.json` | `results/external/index_analysis/superseded/generalist_core_initial/phase_d_summary.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c_v2/metric_utility_v2.csv.gz` | `results/external/index_analysis/superseded/utility_corrected_dispatch_pre_canonical_basis/metric_utility_v2.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c_v2/phase_c_v2_summary.json` | `results/external/index_analysis/superseded/utility_corrected_dispatch_pre_canonical_basis/phase_c_v2_summary.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c/metric_utility.csv.gz` | `results/external/index_analysis/superseded/utility_initial/metric_utility.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c/phase_c_summary.json` | `results/external/index_analysis/superseded/utility_initial/summary.json` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e13_external_ami_per_view.csv` | `results/external/paired_statistics/external_ami_per_view.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e13_external_headline_stats.csv` | `results/external/paired_statistics/external_headline_stats.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e13_leave_circles_moons_out.csv` | `results/external/paired_statistics/leave_circles_moons_out.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/paired_comparisons.csv` | `results/external/paired_statistics/paired_comparisons.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e13_provenance.json` | `results/external/paired_statistics/provenance.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/reranker_deltas.csv` | `results/external/paired_statistics/reranker_deltas.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/reranker_material_harm.csv` | `results/external/paired_statistics/reranker_material_harm.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/reranker_summary.csv` | `results/external/paired_statistics/reranker_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific/*.json (2,100 records, selected fields)` | `results/external/per_dataset/final_partitions_by_unit.jsonl.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/phase_c/method_ari.csv.gz` | `results/external/per_dataset/method_ari_by_view.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/method_bestview_per_dataset.csv` | `results/external/per_dataset/method_bestview_per_dataset.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/per_dataset_method_matrix.csv` | `results/external/per_dataset/per_dataset_method_matrix.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/top10_c5_advantages.csv` | `results/external/per_dataset/top10_clustopt_ppv2_rerank_advantages.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/top10_c5_disadvantages.csv` | `results/external/per_dataset/top10_clustopt_ppv2_rerank_disadvantages.csv` |
| `results_analysis/clustering_repository/final_iclr_sprint_20260922/p0_e8_e13_factual_audit/e13_external_source_stats.csv` | `results/external/per_source/external_source_stats.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/source_method_summary.csv` | `results/external/per_source/source_method_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/source_metric_group_summary.csv` | `results/external/per_source/source_metric_group_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/consolidation_report.json` | `results/external/provenance/consolidation_report.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/metric_repair_summary.json` | `results/external/provenance/metric_repair_summary.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/recovery_report.json` | `results/external/provenance/recovery_report.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/targeted_rerun_report.json` | `results/external/provenance/targeted_rerun_report.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/targeted_rerun_set.json` | `results/external/provenance/targeted_rerun_set.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_by_source.csv` | `results/external/runtime/runtime_by_source.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_by_view.csv` | `results/external/runtime/runtime_by_view.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_cost_decomposition.csv` | `results/external/runtime/runtime_cost_decomposition.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_method_summary.csv` | `results/external/runtime/runtime_method_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_per_arm_records.csv.gz` | `results/external/runtime/runtime_per_arm_records.csv.gz` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_performance_tradeoff.csv` | `results/external/runtime/runtime_performance_tradeoff.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_physical_sharing.csv` | `results/external/runtime/runtime_physical_sharing.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_physical_sharing_summary.csv` | `results/external/runtime/runtime_physical_sharing_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_provenance.json` | `results/external/runtime/runtime_provenance.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_reranker_ablation.csv` | `results/external/runtime/runtime_reranker_ablation.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_reranker_summary.csv` | `results/external/runtime/runtime_reranker_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_three_view_bestview_cost.csv` | `results/external/runtime/runtime_three_view_bestview_cost.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_three_view_per_dataset.csv` | `results/external/runtime/runtime_three_view_per_dataset.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_tradeoff.csv` | `results/external/runtime/runtime_tradeoff.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_xy2d_deployment.csv` | `results/external/runtime/runtime_xy2d_deployment.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/runtime_xy2d_tradeoff.csv` | `results/external/runtime/runtime_xy2d_tradeoff.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/baseline_arm_selection.json` | `results/external/summary/baseline_arm_selection.json` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/method_master_table.csv` | `results/external/summary/method_master_table.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/policy_selection_summary.csv` | `results/external/summary/policy_selection_summary.csv` |
| `results_analysis/independent_domain_benchmark/stage4c_v2_323194df26/scientific_review/provenance.json` | `results/external/summary/provenance.json` |

## `results/paper_evidence`

| original | public |
|---|---|
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/arm_catalogue_stage4.csv` | `results/paper_evidence/arm_catalogue_stage4.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/controlled_family_deltas.csv` | `results/paper_evidence/controlled_family_deltas.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/controlled_policy_prepost.csv` | `results/paper_evidence/controlled_policy_prepost.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/controlled_progression.csv` | `results/paper_evidence/controlled_progression.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/evidence_summary.json` | `results/paper_evidence/evidence_summary.json` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/external_by_source.csv` | `results/paper_evidence/external_by_source.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/external_corpus_metadata.csv` | `results/paper_evidence/external_corpus_metadata.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/external_per_dataset.csv` | `results/paper_evidence/external_per_dataset.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/family_metric_selection_matrix.csv` | `results/paper_evidence/family_metric_selection_matrix.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/family_metric_utility_matrix.csv` | `results/paper_evidence/family_metric_utility_matrix.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/fixed32_composition.json` | `results/paper_evidence/fixed32_composition.json` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/gallery_points.json` | `results/paper_evidence/gallery_points.json` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/metric_group_summary_internal.csv` | `results/paper_evidence/metric_group_summary_internal.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/metric_master.csv` | `results/paper_evidence/metric_master.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/new46_inventory.json` | `results/paper_evidence/new46_inventory.json` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/new46_metadata.json` | `results/paper_evidence/new46_metadata.json` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/policy_set.csv` | `results/paper_evidence/policy_set.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/predictor_by_view.csv` | `results/paper_evidence/predictor_by_view.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/predictor_head14_vs_full60.csv` | `results/paper_evidence/predictor_head14_vs_full60.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/repo_family_difficulty.csv` | `results/paper_evidence/repo_family_difficulty.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/repo_family_summary.csv` | `results/paper_evidence/repo_family_summary.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/repo_subfamily_summary.csv` | `results/paper_evidence/repo_subfamily_summary.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/search_space.json` | `results/paper_evidence/search_space.json` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/stage3_alignment.csv` | `results/paper_evidence/stage3_alignment.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/stage3_core.csv` | `results/paper_evidence/stage3_core.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/stage3_family_recall.csv` | `results/paper_evidence/stage3_family_recall.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/stage3_family_tierA.csv` | `results/paper_evidence/stage3_family_tierA.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/stage3_pattern_vs_image.csv` | `results/paper_evidence/stage3_pattern_vs_image.csv` |
| `results_analysis/clustering_repository/clustopt_paper_manuscript_v16_iclr_revision/generated/evidence/stage3_subfamily_tierA.csv` | `results/paper_evidence/stage3_subfamily_tierA.csv` |

## Release-side edits

| file | change |
|---|---|
| `src/models/Independent_Domain_Benchmark/metric_validation/external_cvis.py` | CVNN implementation removed; calls routed to the external fpc adapter |
| `src/models/Independent_Domain_Benchmark/metric_validation/external_cvis.py` | CVNN docstring |
| `src/models/Independent_Domain_Benchmark/metric_validation/external_cvis.py` | CDbw notice (already in committed source) |
| `src/models/Independent_Domain_Benchmark/metric_validation/external_cvis.py` | CVDD notice (already in committed source) |
| `src/models/Independent_Domain_Benchmark/metric_validation/external_cvis.py` | CDbw notice (already in committed source) |
| `src/models/Independent_Domain_Benchmark/metric_validation/external_cvis.py` | CVDD notice (already in committed source) |
| `src/models/Independent_Domain_Benchmark/datasets/fcps.py` | FCPS licence string (already in committed source) |
| `results/external/candidate_banks/*.npz` | valid candidate partitions of the 150 external units (no data values) |
| `analysis/index_analysis/reproduce_cvnn_with_fpc.py` | output folder |
| `analysis/paper/paper_paths.py` | roots resolved to released results; outputs to analysis/output/paper |
| `analysis/paper/build_evidence.py` | import the release path module |
| `analysis/paper/build_tables.py` | import the release path module |
| `analysis/paper/build_figures.py` | import the release path module |
| `analysis/paper/build_evidence.py` | dataset catalogue replaces training table |
| `analysis/paper/build_evidence.py` | repository summary from the released dataset catalogue |
| `analysis/statistics/criterion_baselines.py` | inputs resolved to released results; outputs to analysis/output/statistics |
| `analysis/statistics/controlled_paired_statistics.py` | inputs resolved to released results; outputs to analysis/output/statistics |
| `analysis/statistics/predictor_alignment.py` | inputs resolved to released results; outputs to analysis/output/statistics |
| `analysis/statistics/external_paired_statistics.py` | inputs resolved to released results; outputs to analysis/output/statistics |
| `requirements.txt / requirements-lock.txt` | pinned runtime + cdbw + pyarrow + openpyxl |
| `src/models/Clustering_Repository_Builder/utility_generation/run_subfamily_utility_generation.py` | anonymity: private checkout name in --repo-root help text |
| `src/models/metric_utility_mlp/run_train_cv.py` | anonymity: private checkout name in docstring example |
| `src/models/metric_utility_mlp/README_metric_utility_mlp.md` | anonymity: Colab checkout path in README example (--config, --csv-path) |
| `src/models/metric_utility_mlp/README_metric_utility_mlp.md` | anonymity: Colab checkout path in README example (--repo-root) |
| `models/utility_predictor/neural_full60/artifacts/feature_columns.json` | packaging: byte-identical copy of folds/fold_0/feature_columns.json at the path the C4 context step reads |
| `src/models/Independent_Domain_Benchmark/methods/clustopt_adapter.py` | C4 runtime: policy-selector models no longer resolved for every arm |
| `src/models/Independent_Domain_Benchmark/methods/clustopt_adapter.py` | C4 runtime: policy-selector model resolved on first use |
| `src/models/Independent_Domain_Benchmark/methods/clustopt_adapter.py` | C4 runtime: policy-selector metadata resolved on first use |
| `src/models/Independent_Domain_Benchmark/methods/clustopt_adapter.py` | C4 runtime: C2 / C5 still fail fast at initialisation when their model is absent |
| `analysis/statistics/criterion_baselines.py` | module docstring (text only) |
| `analysis/statistics/controlled_paired_statistics.py` | module docstring (text only) |
| `analysis/statistics/predictor_alignment.py` | module docstring (text only) |
| `analysis/statistics/external_paired_statistics.py` | module docstring (text only) |
| `analysis/statistics/stats_common.py` | module docstring (text only) |
| `analysis/paper/build_evidence.py` | module docstring (text only) |
| `analysis/paper/build_tables.py` | module docstring (text only) |
| `analysis/paper/build_figures.py` | module docstring (text only) |
| `experiments/controlled_benchmark/criterion_baselines/aggregate.py` | docstring lead (internal label E1 explained) |
| `experiments/controlled_benchmark/criterion_baselines/build_configs.py` | docstring lead (internal label E1 explained) |
| `experiments/controlled_benchmark/criterion_baselines/build_manifests.py` | docstring lead (internal label E1 explained) |
| `experiments/controlled_benchmark/criterion_baselines/derive_development_baselines.py` | docstring lead (internal label E1 explained) |
| `experiments/controlled_benchmark/criterion_baselines/run_all_subfamilies.py` | docstring lead (internal label E1 explained) |
| `experiments/controlled_benchmark/criterion_baselines/run_all_subfamilies.py` | usage example: internal path replaced by the public path |
| `analysis/paper/build_evidence.py` | console message |
| `analysis/paper/build_tables.py` | console message |
| `analysis/paper/build_figures.py` | console message |
| `analysis/paper/build_figures.py` | helper docstring |
| `src/experiments/external_baselines/AutoClust/README_AutoClust_baseline.md` | banner: describes material not included in the release |
| `src/experiments/external_baselines/AutoClust/setup/setup_wsl.md` | banner: describes material not included in the release |
| `src/experiments/external_baselines/AutoClust/setup/setup_windows.md` | banner: describes material not included in the release |
| `src/models/Clustering_Repository_Builder/experiments/paper_analysis/README_paper_analysis.md` | banner: describes material not included in the release |
| `src/**`, `results/**` (text files) | absolute local paths made repository-relative; internal report paths and process wording neutralised; files covered by recorded configuration hashes left byte-identical |
