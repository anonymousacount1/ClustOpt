# Paper results index

For every table and figure of the paper, this page lists:
- the released file(s) that contain its numbers;
- the command that prints or rebuilds them;
- the protocol behind them;
- whether a full re-run is needed to regenerate them from scratch.

Numbering follows the submitted manuscript. Every path is relative to the repository root.

**Glossary**

| Term | Meaning |
|---|---|
| **Best view** | the best of the three views per dataset (an oracle over views) |
| **Single** | the `xy_2d` view |
| **Slate oracle** | the best candidate a search visited |
| **FULL60 / HEAD14** | the 60-index vocabulary / its 14 established indices |
| **U / P / O** | uniform / predicted / oracle (true-utility) index weighting |
| **Arm codes (external benchmark)** | **C4** = CLUSTOPT as reported (deployed): k-NN utility profile, top-10, `KNN_TOP10` reranker (run it with `clustopt.pipeline`); C1 = C4 without reranking; C0/C3 = neural profile without/with reranking; C2/C5 = policy selector v1/v2 (auxiliary; cannot be re-run, models not distributed); A0–A3 = AutoClust with original / +established / +new / extended index inventory; M0–M3 = ML2DAC with the same inventories. Internal identifiers in result files: `results/controlled/README.md` (arm legend) |

**Protocols**

| Protocol | Definition |
|---|---|
| **seeded sweep** | controlled benchmark, held-out split (1,055 datasets × 3 views), online search with 50 evaluations per view, one seeded protocol; every arm returns its own criterion's maximiser. Files: `results/controlled/criterion_baselines/seeded_sweep/` |
| **replay** | the earlier controlled replay of the same held-out split (same datasets and search domain; each system follows its own search trajectory, without shared seeds). Files: `results/controlled/criterion_baselines/canonical_frame/`, `results/controlled/reranking/`, `results/controlled/policy_selection/`, `results/controlled/per_dataset/` |
| **fixed-32** | a fixed bank of 32 candidate configurations per view (no search) |
| **online-50** | search with 50 evaluations per view |
| **external** | 50 external datasets × 3 views, 14 arms. Files: `results/external/` |

**Verified** marks values checked number-by-number against the manuscript in the release audit.

---

## Main paper

| Item | Scientific purpose | Authoritative released input(s) | Command | Output | Protocol | Full re-run needed? |
|---|---|---|---|---|---|---|
| Fig. 1 | problem illustration; method schematic | drawn in the manuscript; panel A shows two generated datasets (see the [Figure map](#figure-map)) | — (provenance only) | — | — | — |
| **Table 1** | search criteria compared at equal search | `results/controlled/criterion_baselines/seeded_sweep/arm_summary.csv` (rows 1–6); `…/canonical_frame/controlled_headline_stats.csv` (`LEVEL MEAN` rows; reranked CLUSTOPT, AutoClust, ML2DAC); `results/controlled/reranking/per_policy_results.csv` (`knn_top10_raw`, slate oracle) | `python scripts/print_paper_table.py table1` | stdout / `--csv` | seeded sweep (rows 1–6); replay (last 3 rows) | no |
| Fig. 2 | the criterion decides what is returned, not what is available | `results/controlled/criterion_baselines/seeded_sweep/arm_summary.csv` (six arms: `mean_endpoint_bestview`, `ci_lo`, `ci_hi`, `mean_slate_oracle`); drawn in the manuscript | `python scripts/print_paper_table.py table1` (the plotted values) | stdout | seeded sweep | no |
| Fig. 3A | generalist indices transfer in ordering, not level | `results/external/family_analysis/generalist_core_review.csv`; `results/external/index_analysis/authoritative/metric_master_table.csv` (`utility_fcps`, `utility_sklearn_shapes`, `utility_real_pca`) | `python scripts/reproduce_paper_artifacts.py` (`fig3_metric`, shown cropped in the paper) | `analysis/output/paper/figures/` | external (fixed-32) | no |
| Fig. 3B | 14 family-level specialist relations | `results/controlled/family_analysis/specialization/family_specialists_tierA.csv` | same file, panel B | same | fixed-32, controlled | no |
| **Table 2** | primary external comparison | `results/external/summary/method_master_table.csv` (ARI columns); `results/external/runtime/runtime_three_view_bestview_cost.csv` (`three_view_median`) | `python scripts/print_paper_table.py table2` | stdout | external | no |
| Table 3 | criterion adaptation across systems (literature) | — (descriptive) | — | — | — | — |
| Table 4 | index inventories used | `configs/external_benchmark/metric_analysis_registry.json`, `metric_registry.json`; `clustopt.validity_indices.list_indices()` | direct read | — | — | no |
| Fig. 4 | where the external differences lie (per dataset) | `results/external/figure_data/knn_rerank_pairwise.csv` (`C4_minus_A0`, `C4_minus_M1`, `source`, `circle_or_moon_group`); drawn in the manuscript | direct read | — | external | no |
| §5 claim "+0.227 over uniform" | profile vs uniform weighting | `seeded_sweep/paired_contrasts.csv` (Table 16 row "Uniform, FULL60", unit `bestview_endpoint`) | `python scripts/print_paper_table.py table16` | stdout | seeded sweep | no |
| §5.3 "0.616 against 0.614" | reranked CLUSTOPT vs matched AutoClust (extended) | `canonical_frame/controlled_headline_stats.csv` (`LEVEL MEAN`) | `python scripts/print_paper_table.py section5_3` | stdout | replay | no |
| §5 claim "+0.058 from reranking" (controlled) | exact reranker ablation | `canonical_frame/controlled_headline_stats.csv` (reranked vs no-rerank) | direct read | — | replay | no |
| §7 claim "+0.083 from reranking" (external) | exact reranker ablation | `results/external/paired_statistics/reranker_summary.csv`, `paired_comparisons.csv` | direct read | — | external | no |

## Appendices

| Item | Scientific purpose | Authoritative released input(s) | Command | Protocol | Full re-run needed? |
|---|---|---|---|---|---|
| Tables 5–6 | controlled repository by family / subfamily | `data/synthetic_generator/dataset_catalogue.csv`; `results/paper_evidence/repo_family_summary.csv`, `repo_subfamily_summary.csv` | `python scripts/reproduce_paper_artifacts.py` | — | no |
| Fig. 5 | one dataset per family | `results/paper_evidence/gallery_points.json` (frozen, deterministic selection) | same (`figA_family_gallery`) | — | no (the selection itself: yes, it needs the regenerated corpus) |
| Table 7 | search space per view | `configs/clustopt/neural_profile_top5/*.json`; `results/paper_evidence/search_space.json` | same | — | no |
| Tables 8–9 | utility-prediction quality | `results/controlled/predictor_evaluation/neural_full60_overall_results.csv`, `neural_full60_grouped_mean_by_view_id.csv`, `neural_head14_vs_full60.csv`, `knn/` | direct read | held-out split | no |
| Tables 10–11, Fig. 6 | inventory × weighting factorial | `results/controlled/inventory_factorial/fixed_budget_32/fixed_budget_{overall_summary,paired_contrasts}.csv`, `online_search_50/online_{overall_summary,paired_contrasts}.csv` | direct read; Fig. 6 via `scripts/reproduce_paper_artifacts.py` (`figH_inventory_effect`) | fixed-32 / online-50 | no |
| Fig. 7 | arm lineage (schematic) | `configs/external_benchmark/method_registry.json`; `models/candidate_reranker/policy_to_reranker_mapping.json` | `scripts/reproduce_paper_artifacts.py` (`figD_arm_lineage`) | — | no |
| Table 12 | the 14 external arms | `configs/external_benchmark/method_registry.json`; the arm catalogue `results/paper_evidence/arm_catalogue_stage4.csv` | direct read | external | no |
| Tables 13–14 | the 20 policies before/after reranking | `results/controlled/reranking/per_policy_results.csv`; `models/candidate_reranker/policy_to_reranker_mapping.json` | direct read | replay | no |
| Table 15 | four-level progression | rows 1–2 (fixed policy without / with reranking, the exact reranking isolation): `results/controlled/criterion_baselines/canonical_frame/controlled_headline_stats.csv` (`LEVEL MEAN`, same bootstrap as Table 1); rows 3–5 (policy selector v1, v2, policy oracle): `results/controlled/policy_selection/four_level_online_table.csv` (its level A row holds an independent bootstrap of the same mean) | `python scripts/print_paper_table.py table15` | replay | no |
| **Table 16** | criterion baselines with paired statistics | `seeded_sweep/arm_summary.csv`, `seeded_sweep/paired_contrasts.csv` | `python scripts/print_paper_table.py table16` | seeded sweep | no |
| **Table 17** | slate ceilings vs returned endpoints | same | `python scripts/print_paper_table.py table17` | seeded sweep | no |
| Table 18 | baseline index-inventory arms (controlled) | `results/controlled/baseline_inventory_decomposition/{autoclust,ml2dac}/four_arm_global_summary.csv` | direct read | replay | no |
| Table 19, Figs 8–9 | per-family results vs matched AutoClust | `results/controlled/criterion_baselines/canonical_frame/controlled_family_stats.csv` (Fig. 8: `delta_fixed_minus_autoclust`, `ci_lo`, `ci_hi`; drawn in the manuscript); `results/paper_evidence/controlled_family_deltas.csv`; `results/controlled/per_dataset/controlled_best_view_results.parquet` | direct read; `python analysis/statistics/controlled_paired_statistics.py` rebuilds the family table; Fig. 9 via `scripts/reproduce_paper_artifacts.py` (`figB_family_absolute`) | replay | no |
| Tables 20–21 | family / subfamily specialisation | `results/controlled/family_analysis/specialization/{family,subfamily}_specialists_tierA.csv` | direct read | fixed-32 | no |
| Table 22 | utility–selection alignment | `…/specialization/family_utility_selection_alignment.csv`; `results/controlled/predictor_alignment/selection_alignment_per_family.csv`, `selection_alignment.csv` | direct read (frozen outputs). `analysis/statistics/predictor_alignment.py` re-runs only its predictor-metrics part: the alignment part needs training-split predictions that are not redistributed (`docs/reproduction.md`) | replay | no |
| Table 23 | pattern vs image indices | `…/specialization/pattern_vs_image_family.csv` | direct read | fixed-32 | no |
| Fig. 10 | index selection by family | `results/paper_evidence/family_metric_selection_matrix.csv`, `family_metric_utility_matrix.csv` | `scripts/reproduce_paper_artifacts.py` (`figE_selection_heatmap`) | replay | no |
| Table 24 | what the policy selector selects, by family | `results/controlled/policy_selection/family_analysis.csv`, `policy_selection_frequencies.csv` | direct read | replay | no |
| Table 25 | external corpus metadata | `data/external_benchmark/dataset_manifest.csv`, `prepared_representation_manifest.json` | `python scripts/reconstruct_external_datasets.py --verify-only` (checksums) | external | no (the datasets themselves are rebuilt from public sources) |
| Table 26 | per-dataset external results | `results/external/per_dataset/per_dataset_method_matrix.csv`; recomputable from `final_partitions_by_unit.jsonl.gz` | `python analysis/statistics/verify_external_from_partitions.py` (needs reconstructed data) | external | no |
| Table 27, Figs 11–12 | all 14 arms; accuracy vs cost | `results/external/summary/method_master_table.csv`; `results/external/runtime/runtime_performance_tradeoff.csv`, `runtime_xy2d_tradeoff.csv` | direct read; Figs. 11–12 via `scripts/reproduce_paper_artifacts.py` (`fig4_external`, `figC_single_view_pareto`) | external | no |
| Table 28 | pre-specified paired contrasts | `results/external/paired_statistics/paired_comparisons.csv` | direct read | external | no |
| Tables 29–30 | index utility on the external corpus (67 indices) | `results/external/index_analysis/authoritative/metric_master_table.csv`; CVNN: `…/authoritative/cvnn_frozen_values.csv` | direct read; CVNN optionally `python analysis/index_analysis/reproduce_cvnn_with_fpc.py` (R + fpc) | external, fixed-32, canonical candidate basis | no |
| Table 31 | the 46 new indices individually | `…/authoritative/metric_master_table.csv` (group `New46`); `results/external/family_analysis/new46_specialist_table.csv` | direct read | external | no |
| Table 32 | runtime by scope | `results/external/runtime/runtime_method_summary.csv`, `runtime_three_view_bestview_cost.csv` | direct read | external | no |
| Appendix T | corrections to the external index analysis | the superseded layers in `results/external/index_analysis/superseded/`, for comparison with `authoritative/` | direct read | external | no |

## Figure map

How each figure of the paper was produced, verified against the submitted PDF. Three modes occur:
- **generated figure file.** The paper includes a file that `python scripts/reproduce_paper_artifacts.py` builds here (in `analysis/output/paper/figures/`). For all eight such figures, the rebuilt file is identical to the one the paper includes.
- **drawn in the manuscript.** The figure is drawn with LaTeX (TikZ/PGFPlots) from values taken from a released file. The plotted values agree with that file at the manuscript's four-decimal precision.
- **provenance only.** The figure cannot be regenerated from this release.

| Fig. | Content | Mode | Released source data | File built here | Status |
|---|---|---|---|---|---|
| 1 | the problem and the method | drawn in the manuscript (TikZ/PGFPlots); panel A plots two generated datasets with their silhouette-best and DBCV-best partitions | not released: the point tables were produced by a script that was not preserved | — (`fig1_problem_and_architecture` is an earlier rendering that the paper does not use) | provenance only; illustrative, no reported number depends on it |
| 2 | the criterion decides what is returned | drawn in the manuscript (PGFPlots) | `results/controlled/criterion_baselines/seeded_sweep/arm_summary.csv` (six arms; endpoint, 95% interval, slate oracle) | — | traceable exactly to released data |
| 3 | generalist core and family specialists (A, B) | generated figure file, cropped in the manuscript (the paper shows panels A and B) | `results/external/family_analysis/generalist_core_review.csv`; `results/controlled/family_analysis/specialization/family_specialists_tierA.csv`, `family_utility_selection_alignment.csv` | `fig3_metric` | directly regenerable |
| 4 | where the external differences lie | drawn in the manuscript (PGFPlots) | `results/external/figure_data/knn_rerank_pairwise.csv` (written by `analysis/statistics/external_paired_statistics.py`) | — | traceable exactly to released data |
| 5 | one dataset per family | generated figure file | `results/paper_evidence/gallery_points.json` | `figA_family_gallery` | directly regenerable |
| 6 | FULL60 − HEAD14 paired effect | generated figure file | `results/controlled/inventory_factorial/*/..._paired_contrasts.csv` | `figH_inventory_effect` | directly regenerable |
| 7 | arm lineage | generated figure file | `configs/external_benchmark/method_registry.json`; `models/candidate_reranker/policy_to_reranker_mapping.json` | `figD_arm_lineage` | directly regenerable |
| 8 | per-family difference from matched AutoClust | drawn in the manuscript (PGFPlots) | `results/controlled/criterion_baselines/canonical_frame/controlled_family_stats.csv` (rebuilt by `analysis/statistics/controlled_paired_statistics.py`) | — (`fig3_family_delta` is an earlier version that the paper does not use) | traceable exactly to released data |
| 9 | absolute family means | generated figure file | `results/paper_evidence/controlled_family_deltas.csv` | `figB_family_absolute` | directly regenerable |
| 10 | index selection by family | generated figure file | `results/controlled/family_analysis/specialization/family_metric_selection.csv.gz` | `figE_selection_heatmap` | directly regenerable |
| 11 | accuracy vs cost, three views | generated figure file | `results/external/runtime/runtime_performance_tradeoff.csv` | `fig4_external` | directly regenerable |
| 12 | accuracy vs cost, single view | generated figure file | `results/external/runtime/runtime_xy2d_tradeoff.csv` | `figC_single_view_pareto` | directly regenerable |

The builder's file names follow an earlier numbering. `fig2_controlled_mechanism` and `figG_paired_delta` are also built but are not figures of the paper.

**Regenerating everything from scratch** (corpus generation, training, searches, the external benchmark) is described in `docs/reproduction.md`.
