# External benchmark results

50 datasets (7 FCPS, 15 scikit-learn generator, 28 real) × 3 views = 150 units, and 14 arms. Arm codes are in the glossary of `docs/paper_results_index.md`. No external data values are stored here; datasets are rebuilt from their public sources (`docs/external_data.md`).

## Folders

| Folder | Content | Key files |
|---|---|---|
| `summary/` | one row per arm: mean / CI of best-view and `xy_2d` ARI, by source, runtime (Tables 2, 27) | `method_master_table.csv` |
| `per_dataset/` | per-dataset best-view ARI of all arms (Table 26); per view; the returned partitions | `per_dataset_method_matrix.csv`, `method_bestview_per_dataset.csv`, `method_ari_by_view.csv.gz`, `final_partitions_by_unit.jsonl.gz` |
| `per_source/` | results by source | `source_method_summary.csv` |
| `paired_statistics/` | pre-specified paired contrasts: bootstrap intervals, Wilcoxon, rank-biserial, W/T/L (Table 28); reranker ablations | `paired_comparisons.csv`, `reranker_summary.csv` |
| `family_analysis/` | generalist core and new-index specialists (Fig. 3A) | `generalist_core_review.csv` |
| `index_analysis/authoritative/` | **the index analysis reported in the paper** (Tables 29–31): index utility on the 32-candidate bank under the canonical candidate basis, including CVNN | `metric_master_table.csv`, `utility_canonical_basis/`, `generalist_core_and_pipeline/`, `candidate_ari.csv.gz`, `cvnn_frozen_values.csv` |
| `index_analysis/superseded/` | earlier versions of the same analysis, kept only to document the corrections described in the paper's appendix. **Do not use for reported values.** | |
| `candidate_banks/` | the 32 candidate partitions of each unit (cluster assignments only) | `<dataset>__<view>.npz` |
| `runtime/` | runtime per arm, view, source; cost decomposition (Table 32, Figs 11–12) | `runtime_three_view_bestview_cost.csv` |
| `figure_data/` | data behind the external figures (Fig. 4 and appendix figures) | |
| `provenance/` | run consolidation, recovery and repair reports | |

## File formats

**`final_partitions_by_unit.jsonl.gz`.** One JSON record per line (dataset × view × arm):
- `dataset_id`, `view_id`, `method_id`, `status`;
- `selected_algorithm`, `selected_configuration`;
- `details` (selected indices, reranker choice);
- `n_evaluations`, `final_labels` (cluster id per point, noise = −1);
- `final_labels_hash`, `runtime_seconds`.

**`candidate_banks/*.npz`.** `candidate_index` (int) and `labels` (n_points × n_candidates). Only candidates that produced a valid partition are included.
