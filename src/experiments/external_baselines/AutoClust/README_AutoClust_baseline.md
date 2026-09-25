> **Not used by the reported results.** This page describes AutoClust's native adapter, which is not included in this release; the runner modules it names are absent. The reported AutoClust arms are described in `baselines/README.md`.

# AutoClust external baseline

Integration of **AutoClust** as an external AutoClustering baseline, evaluated on
the same datasets/splits as ClustOpt, AutoML4Clust, and ML2DAC. Mirrors the
AutoML4Clust / ML2DAC baseline structure.

## What AutoClust is
AutoClust (Poulakis et al., IEEE ICDM 2020) is a meta-learning AutoClustering
framework: it extracts **landmarking** meta-features (via MeanShift), selects a
clustering algorithm from meta-knowledge (KD-tree similarity over past datasets),
and tunes that algorithm's hyperparameters with **Bayesian optimization (SMAC)**,
using a **learned MLP objective** that maps internal CVI values to a predicted
external clustering quality (ARI). The MLP is the distinctive idea — a learned,
data-dependent objective instead of a single fixed CVI.

## Implementation: the ML2DAC re-implementation
AutoClust has **no official public source from its original authors**. The ML2DAC
authors re-implemented it for their related-work comparison; that implementation
lives inside the existing ML2DAC submodule at
`external/ml2dac/src/Experiments/RelatedWork/AutoClust.py`. **This baseline uses
that re-implementation** (not ML2DAC itself). `external/ml2dac` is never modified —
the adapter wraps the building blocks. See `AUTOCLUST_REPOSITORY_AUDIT.md`.

## How it differs from the other baselines
| | objective | algorithm choice | search |
|---|---|---|---|
| **AutoML4Clust** | one fixed internal CVI | searches the space | SMAC/Random/… |
| **ML2DAC** | one *predicted* CVI | meta-learned reduction + warmstart | SMAC |
| **ClustOpt** | metric-utility–weighted multi-CVI | searches the space | Optuna/brute |
| **AutoClust** | **learned MLP(CVIs→ARI)** | meta-learned single algorithm (landmarking) | **SMAC** |

AutoClust is conceptually closest to ClustOpt in that both optimize a **learned,
data-dependent objective** rather than a single off-the-shelf CVI.

## Artifacts — all ship (no offline phase / training needed)
Under `external/ml2dac/src/Experiments/RelatedWork/related_work/`: the **trained
MLP** (`rw_mlp.pkl`), the landmarking **KD-tree** + **meta-features**
(`meanshift_*`), and the offline **ARI rankings** (`related_work_offline_opt.csv`).
The online phase runs immediately. (Meta-knowledge was built on ML2DAC's 78
synthetic datasets — **no leakage** to ours; our labels are never input.)

## Environment
Reuses the existing **`ml2dac` conda env** (Python 3.9, SMAC 1.4, ConfigSpace,
scikit-learn 0.24.2, …) — no new environment. See `setup/setup_wsl.md`. Run real
experiments under **WSL/Linux** (SMAC/pynisher spawn subprocesses, and the runner's
NoDaemon fork pool needs a fork start-method).

## How to run
```bash
cd .
PY=~/miniconda3/envs/ml2dac/bin/python
# readiness check
$PY experiments/external_baselines/AutoClust/setup/check_autoclust_install.py --repo-root .
# smoke test (tiny synthetic)
$PY -m experiments.external_baselines.AutoClust.runners.smoke_test_autoclust --repo-root .
# one dataset
$PY -m experiments.external_baselines.AutoClust.runners.run_autoclust_dataset \
  --dataset-folder "<dataset_folder>" --split-id 1 \
  --config experiments/external_baselines/AutoClust/configs/autoclust_full_ml2dac_reimplementation.json \
  --record-types 1d_x 1d_y 2d --repo-root . --overwrite
# runtime feasibility audit (pilot; do NOT run the full benchmark yet)
$PY -m experiments.external_baselines.AutoClust.runners.audit_autoclust_runtime \
  --subfamily-root "results_analysis/clustering_repository/analyzed_data/arcs_circles_rings/subfamilies/clean_arcs" \
  --split-assignment-csv "results_analysis/clustering_repository/analyzed_data/experiment_splits/dataset_split_assignments.csv" \
  --split-id 1 --config experiments/external_baselines/AutoClust/configs/autoclust_full_ml2dac_reimplementation.json \
  --record-types 1d_x 1d_y 2d --max-datasets 3 --repo-root .
# one subfamily + split (FINAL form; do NOT run the full thing yet)
$PY -m experiments.external_baselines.AutoClust.runners.run_autoclust_subfamily_split \
  --subfamily-root "<subfamily_folder>" \
  --split-assignment-csv "results_analysis/clustering_repository/analyzed_data/experiment_splits/dataset_split_assignments.csv" \
  --split-id 1 --config experiments/external_baselines/AutoClust/configs/autoclust_full_ml2dac_reimplementation.json \
  --record-types 1d_x 1d_y 2d --workers 2 --repo-root .
```

## What gets saved
Per dataset, under `<dataset>/experiments/split_XX/AutoClust/`:
`{1d_x,1d_y,2d}/result.json|labels.npy|summary_metrics.json|raw_result.json|run_log.txt|autoclust_artifact_report.json`,
plus `dataset_summary.json` (**best record by ARI** + selected algorithm, optimizer
evals, MLP-objective status, artifact status, runtimes, statuses) and
`README_AUTOCLUST_RUN.md`. Run-level rollup under
`results_analysis/clustering_repository/external_baselines/AutoClust/<ts>_split_XX_<subfamily>/`
(`run_summary.csv, failed_datasets.csv, skipped_datasets.csv, run_config.json,
README_RUN_SUMMARY.md`). Dataset-level metric is **Best View by ARI** (comparable to
ClustOpt Best-View Mean ARI).

## Metrics
Identical to the other baselines (re-exported from `AutoML4Clust/utils/metrics.py`):
ARI, NMI, AMI, v_measure, homogeneity, completeness, fowlkes_mallows, purity,
predicted/true noise ratios + noise P/R/F1, selected_k, true_k, k_error/abs_k_error/
exact_k_match (+ ClustOpt aliases), n_predicted_clusters_excluding_noise,
n_noise_points, runtime_sec, status.

## Recommended final config
`configs/autoclust_full_ml2dac_reimplementation.json` — full online phase (MeanShift
landmarking + KD-tree selection + SMAC HPO with the MLP objective),
**`optimizer_evaluations=100`** (ML2DAC's value; the AutoClust paper uses 40),
`random_state=42`. **No wall-clock timeout** — AutoClust runs its native
evaluation-count budget to completion (`smac_wallclock_limit_sec=null` → SMAC's
native 7200 s ceiling + 300 s per-config cutoff apply).

## Fairness rules (enforced)
AutoClust receives **only** the partition space. **No** true labels, true K,
family/subfamily/difficulty metadata, or ClustOpt utilities. `true_labels` is **not**
passed to SMAC (the application-phase MLP objective never uses labels). True labels
are used only for external evaluation after prediction. `external/ml2dac` is not
modified (the adapter installs nothing, only reads).

## Known runtime issues / budget
AutoClust is the heaviest of the three: **MeanShift landmarking** (O(n²), once per
record, before SMAC) **+ 7 internal CVIs incl. DBCV** (O(n²)) on every SMAC eval.
At n≈1,400–3,400 (our data) a record's 100 evaluations typically take ~150–300 s.

**AutoClust's budget is a fixed evaluation count, NOT a wall-clock timeout** — this
is true of both the original paper (40 trials) and the ML2DAC implementation (100
loops). We therefore **do not impose any hard timeout**: AutoClust runs its full
evaluation budget to natural completion (the earlier 150 s hard timeout was removed
after a verification audit). SMAC's own native ceilings (7200 s wallclock, 300 s
per-config cutoff) are the only time bounds. **Run the runtime audit first** to size
the full benchmark.

## How to cite / report
- **Original AutoClust:** Poulakis, Doulkeridis, Kyriazis, *AutoClust: A Framework
  for Automated Clustering based on Cluster Validity Indices*, IEEE ICDM 2020.
- **Implementation used:** the ML2DAC authors' re-implementation
  (Treder-Tschechlov et al., *ML2DAC*, SIGMOD/PACMMOD 2023), bundled in their repo.
  Report AutoClust results as "AutoClust (ML2DAC re-implementation)".

## What remains for later (not in this task)
Aggregation across AutoClust results; comparison tables vs ClustOpt /
AutoML4Clust / ML2DAC (Best-View Mean ARI); statistical tests vs ClustOpt; a deeper
runtime audit if the pilot is borderline.
