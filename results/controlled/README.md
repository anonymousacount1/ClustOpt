# Controlled benchmark results

All results are on the held-out split: 1,055 generated datasets × 3 views, excluded from all learned-component fitting. Which file backs which table is listed in `docs/paper_results_index.md`.

## Protocols (kept separate on purpose)

| Folder | Protocol |
|---|---|
| `criterion_baselines/seeded_sweep/` | **Seeded sweep**: all criteria compared with the same seeded online search (50 evaluations per view); every arm returns its own criterion's maximiser. Paper Tables 1 (rows 1–6), 16, 17 and Fig. 2 |
| `criterion_baselines/canonical_frame/` | **Replay**: the earlier replay of the same split, with each system on its own search trajectory. Reranked CLUSTOPT and the matched AutoClust / ML2DAC arms (Table 1, last 3 rows), per-family statistics and the AutoML4Clust supplementary arm |

## Folders

| Folder | Content | Key files |
|---|---|---|
| `per_dataset/` | every arm × dataset (best view) and × view: ARI, AMI, selected configuration, selected indices | `controlled_best_view_results.parquet`, `controlled_run_results.parquet` |
| `criterion_baselines/seeded_sweep/` | arm summaries, paired contrasts, family contrasts, per-dataset and per-run results, development-set baselines, seed manifest | `arm_summary.csv`, `paired_contrasts.csv`, `dataset_results.csv` |
| `criterion_baselines/canonical_frame/` | replay contrasts and level means with bootstrap intervals, per-family statistics | `controlled_headline_stats.csv`, `controlled_criterion_baselines.csv`, `controlled_family_stats.csv` |
| `inventory_factorial/` | 60 vs 14 indices × uniform / predicted / oracle weighting, at a fixed budget of 32 candidates and in online search (Tables 10–11) | `*_overall_summary.csv`, `*_paired_contrasts.csv` |
| `reranking/` | the 20 policies before and after reranking, recovery of the slate oracle, runtime | `per_policy_results.csv`, `statistical_comparisons.csv` |
| `policy_selection/` | the policy selector (versions 1 and 2) on the replay | `four_level_online_table.csv`, `family_analysis.csv` |
| `matched_autoclust/` | AutoClust under CLUSTOPT's search space, per dataset | `per_dataset_bestview_autoclust_extended_matched.csv` |
| `baseline_inventory_decomposition/` | AutoClust / ML2DAC under four index inventories (Table 18) | `*/four_arm_global_summary.csv` |
| `family_analysis/` | index attribution, family / subfamily specialisation (Tables 20–23) | `specialization/family_specialists_tierA.csv` |
| `predictor_evaluation/`, `predictor_alignment/` | predictor quality (Tables 8–9); utility–selection alignment (Table 22) | |

## Arm legend (internal identifiers in the result files)

The result files keep the identifiers under which the arms were run. `scripts/print_paper_table.py` prints the paper's labels; this table maps one to the other.

**Seeded sweep** (`criterion_baselines/seeded_sweep/`; Tables 1 rows 1–6, 16, 17; Fig. 2)

| `arm` | `label` in the file | Paper row | Configuration |
|---|---|---|---|
| `e1_c4_seed` | E1-C4-SEED (conditioned: KNN predicted profile) | Predicted profile, top-10 (CLUSTOPT) — the search of C4 before reranking | `configs/criterion_baselines/knn_profile_matched_seed/` (search space identical to `configs/clustopt/knn_profile_top10/`) |
| `e1_meanprof` | E1-MEANPROF (dev mean profile) | Dataset-independent mean profile, top-10 | `configs/criterion_baselines/mean_utility_profile/` |
| `e1_core3` | E1-CORE3 (static generalist core) | Static three-index core | `configs/criterion_baselines/static_generalist_core3/` |
| `e1_best1` | E1-BEST1 (dev best single index) | Best single index (development, per view) | `configs/criterion_baselines/best_single_index/` |
| `e1_random` | E1-RANDOM (no criterion) | Random search (no criterion) | `configs/criterion_baselines/random_search/` |
| `stage1_full60_uniform_fixed50` | E1-UNIF60 (uniform over FULL60 …) | Uniform over FULL60 | `configs/controlled_benchmark/inventory_factorial/full60_uniform/` (reused from the factorial's online arm: same seeded protocol) |
| `stage1_full60_mlp_top5_raw_fixed50` | Stage-1 MLP top-5 (conditioned, reused) | Table 16: neural predicted profile, top-5 (conditioned reference) | `configs/clustopt/neural_profile_top5/` |
| `stage1_full60_oracle_top5_raw_fixed50` | Stage-1 ORACLE top-5 (ceiling, reused) | Table 16: true (oracle) profile, top-5 (ceiling reference) | `configs/controlled_benchmark/inventory_factorial/full60_oracle_top5/` |

"E1" and "Stage-1" are internal names of the sweep and of the inventory × weighting factorial.

**Replay** (`criterion_baselines/canonical_frame/`, `per_dataset/`, `reranking/`; Table 1 last three rows, §5.3, Table 15, Table 19, Fig. 8)

| Identifier | Meaning | Mean best-view ARI |
|---|---|---|
| `fixed_knn_top10_raw_reranked` (policy `knn_top10_raw` + `KNN_TOP10`) | **CLUSTOPT, reranked** (C4 on the controlled benchmark) | 0.616 |
| `fixed_knn_top10_raw_unreranked` | the same search without reranking (Table 15 row 1) | 0.558 |
| `fixed_knn_top10_raw_slate_oracle` | the best candidate the search visited | 0.653 |
| `AutoClust_Extended_InDomain_same_search_space` | **AutoClust, extended inventory, matched search space**: the reported comparator (Table 1, §5.3) | 0.614 |
| `ML2DAC_OriginalPlusEstablished_InDomain_same_search_space` | ML2DAC, corrected, +established (Table 1) | 0.574 |
| `ML2DAC_Extended_InDomain_same_search_space` | ML2DAC, extended inventory, matched search space | 0.566 |
| `AutoClust_Extended_InDomain` | AutoClust, extended inventory, **its own unconstrained search space**: an auxiliary row, **not** a comparator reported in the paper (see below) | 0.617 |
| `silhouette_single`, `dbcv_single`, `davies_bouldin_single`, `calinski_harabasz_single`, `uniform_classic_cvi`, `all_metrics_uniform`, `AutoML4Clust_phase_B` | replay criterion arms (Appendix; `configs/criterion_baselines/{silhouette_single,…,uniform_all60}/`); `AutoML4Clust_phase_B` is the supplementary AutoML4Clust arm | |

**External benchmark codes** (`results/external/`): `IDB_ClustOpt_C4_KNN_TOP10_RAW_R` = **C4, CLUSTOPT as reported**; `…_C1_KNN_TOP10_RAW_noR` = C1 (C4 without reranking); `…_C0_MLP_TOP5_RAW_noR` / `…_C3_MLP_TOP5_RAW_R` = neural-profile variant without / with reranking; `…_C2_PPv1_noR` / `…_C5_PPv2_R` = policy selector v1 / v2 (auxiliary; not re-runnable, models not distributed); A0–A3 = AutoClust original / +established / +new / extended; M0–M3 = ML2DAC with the same inventories. The `short` column of `results/external/summary/method_master_table.csv` holds the code.

**Notes on specific files**
- **The auxiliary AutoClust row.** `criterion_baselines/canonical_frame/controlled_headline_stats.csv` contains, besides the reported comparison with AutoClust under the matched search space (0.6145), a row for AutoClust in its own unconstrained search space (`AutoClust_Extended_InDomain`, "NATIVE domain", level mean 0.6171). That row is auxiliary. It is not the comparator reported in the paper, which matches every system to CLUSTOPT's search space; it is kept unchanged because the file is a frozen result.
- **Reranker copies.** `reranking/heldout_preparation/final_models/` holds byte-identical copies of the released rerankers in `models/candidate_reranker/`, kept at the paths recorded in the reranking manifests.

**Column conventions:**
- `*bestview*` is best-view ARI; `*xy*` / `single` is the `xy_2d` view.
- `mean_delta` is always "CLUSTOPT minus the other arm".
- `wins_eps` / `ties_eps` / `losses_eps` use the tie band |Δ| < 0.001.
