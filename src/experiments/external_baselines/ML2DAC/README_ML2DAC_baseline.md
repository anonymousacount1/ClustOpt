# ML2DAC — external meta-learning AutoClustering baseline

This package integrates **ML2DAC** (https://github.com/tschechlovdev/ml2dac) as
an external AutoClustering baseline for a **fair** comparison against ClustOpt
(and AutoML4Clust) on the *same* datasets, splits, and record types. It mirrors
the AutoML4Clust baseline structure and reuses its metrics/IO so results are
directly comparable.

## What ML2DAC is

ML2DAC ("Meta-learning to Democratize AutoML for Clustering") is a
**meta-learning** AutoClustering approach. For a new dataset it:

1. **Selects a CVI** — a trained classifier predicts a suitable internal Cluster
   Validity Index from the dataset's meta-features (`cvi="predict"`).
2. **Warmstarts the optimizer** — it retrieves the most *similar* prior dataset
   (via a meta-feature KD-tree) and seeds SMAC with that dataset's best evaluated
   configurations.
3. **Reduces the search space** — it limits the clustering algorithms to those
   appearing in the warmstart configs (`limit_cs=True`).

It then runs **SMAC Bayesian optimization** over that reduced space, optimising
the predicted CVI.

## Why it is a stronger / "more dangerous" baseline than AutoML4Clust

AutoML4Clust optimises a *single fixed* internal CVI (Silhouette) over a fixed
algorithm set with no prior knowledge. ML2DAC is **data-dependent**: it adapts
the objective (CVI), the starting configurations, and the search space to the
dataset using knowledge transferred from prior datasets — much closer to
ClustOpt's philosophy of a data-dependent objective/selection. So ML2DAC is the
more serious yardstick.

## How it differs from ClustOpt

ClustOpt learns a metric-utility–weighted objective from *our* repository and
includes geometry/image-aware CVIs; ML2DAC transfers knowledge from *its own*
prior (synthetic + UCI) datasets and uses 7 classical internal CVIs. Both are
data-dependent, but over different knowledge bases and CVI families.

## Where the external repository is stored

Git submodule at `external/ml2dac` (commit `c946b9e`). **Never modified**; wrapped
only through `adapters/`.

## Trained artifacts (MetaKnowledgeRepository) — YES, they ship

The full MKR is included in the submodule (verified real, not LFS pointers):

```
external/ml2dac/src/MetaKnowledgeRepository/
  evaluated_configs.csv        # 134 MB — warmstart config source (Git-LFS)
  optimal_cvi.csv              # 20 KB
  meta_features/<set>_kdtree.pkl + <set>_metafeatures.csv   # similar-dataset search
  models/RandomForestClassifier/<mf_set>/{<dataset>|None}   # 311 MB — trained CVI classifiers
```

So **the full ML2DAC application phase runs immediately — no learning phase
required.** See `ML2DAC_REPOSITORY_AUDIT.md` for the full audit. (If the LFS CSVs
ever appear as stubs, run `git lfs pull` inside `external/ml2dac`.)

## Fairness contract (what is NOT allowed)

ML2DAC receives **only the partition space**. It never receives:
❌ true labels, ❌ true K, ❌ ClustOpt metric utilities, ❌ family/subfamily/difficulty
metadata. (`true_labels=None` is passed through internally.) Ground truth is used
**only for external evaluation** after prediction.

## Record types and the partition / evaluation split

| record_type | partition_space (clustered) | evaluation_space |
| --- | --- | --- |
| `1d_x` | `X[:, [0]]` | `X[:, [0, 1]]` |
| `1d_y` | `X[:, [1]]` | `X[:, [0, 1]]` |
| `2d`   | `X[:, [0, 1]]` | `X[:, [0, 1]]` |

Loaded via the same ClustOpt/AutoML4Clust loader, so the spaces are identical.
ML2DAC StandardScales the input internally; the adapter passes the **raw**
partition space (no double scaling).

## Recommended configuration (paper-level)

`configs/ml2dac_full_meta_learning.json` — the strongest faithful setup
(`cvi="predict"`, warmstarting on, algorithm reduction on, SMAC), meta-feature
set `statistical+info-theory+general` (index 4):

```jsonc
{ "use_meta_cvi_selection": true, "use_warmstarting": true,
  "use_algorithm_reduction": true, "optimizer": "SMAC",
  "time_budget_sec": 120, "n_optimizer_loops": 100, "n_warmstarts": 25,
  "mf_set_index": 4, "mkr_path": null,
  "artifact_policy": "require_pretrained_or_fail", "fail_on_missing_mkr": true }
```

### Budget
ML2DAC's SMAC scenario enforces **both** `runcount-limit = n_optimizer_loops`
(100, which includes the 25 warmstarts) **and** `wallclock-limit =
time_budget_sec` (120 s). So unlike AutoML4Clust, `time_budget_sec` is a **real
wall-clock limit** — directly matching ClustOpt's 120 s philosophy. `random_state`
is fixed to 1234 internally (deterministic).

### Fallback / ablation
`configs/ml2dac_fallback_no_mkr_random_or_smac.json` disables the three
meta-learning components (plain SMAC). **Debug only — not for the final
comparison.** A true "no-MKR" mode is impossible: ML2DAC's application phase
cannot run without the MKR. If the MKR were ever missing, the adapter writes
`status="failed"` with `failure_kind="artifacts_missing"` — it never silently
runs a degraded variant and calls it ML2DAC.

## Environment

Needs a **dedicated, isolated** env: Python **3.9** + `smac==1.4.0`,
`ConfigSpace==0.6.0`, `hdbscan==0.8.32`, `pymfe==0.4.1`, `scikit-learn==0.24.2`,
`pandas==1.1.5`, `numpy==1.24.2`, `scipy==1.8.0`. It will **not** run in the main
env or the AutoML4Clust env. **WSL/conda is recommended** — see
`setup/setup_wsl.md` (`create_env_wsl_conda.sh`). Windows native is discouraged
(`setup/setup_windows.md`).

## How to run

### 1. Smoke test (infra + artifacts; tiny run if env ready)
```
python -m experiments.external_baselines.ML2DAC.runners.smoke_test_ml2dac
```
Passes in "infrastructure OK" mode even before the env is built (it reports
artifacts present + that the env/import is not yet ready). Once the env is set up
it runs a tiny synthetic dataset end-to-end. Saved under
`results_analysis/clustering_repository/external_baselines_smoke_tests/ML2DAC/`.

### 2. One dataset
```
python -m experiments.external_baselines.ML2DAC.runners.run_ml2dac_dataset ^
  --dataset-folder "<dataset_folder>" --split-id 1 ^
  --config "experiments/external_baselines/ML2DAC/configs/ml2dac_full_meta_learning.json" ^
  --record-types 1d_x 1d_y 2d ^
  --repo-root "." --overwrite
```

### 3. One subfamily + split (final runner)
```
python -m experiments.external_baselines.ML2DAC.runners.run_ml2dac_subfamily_split ^
  --subfamily-root "<path_to_subfamily_folder>" ^
  --split-assignment-csv "results_analysis/clustering_repository/analyzed_data/experiment_splits/dataset_split_assignments.csv" ^
  --split-id 1 ^
  --config "experiments/external_baselines/ML2DAC/configs/ml2dac_full_meta_learning.json" ^
  --record-types 1d_x 1d_y 2d --workers 8 ^
  --repo-root "."
```
(`--dry-run` previews datasets. From WSL use `/mnt/c/...` paths.)

## What gets saved

### Per dataset (primary results)
```
<dataset>/experiments/split_01/ML2DAC/
  1d_x/ 1d_y/ 2d/
    result.json            status, selected_k, selected_cvi, selected_algorithm,
                           best_configuration, all metrics, artifact_status, provenance
    labels.npy             predicted labels (success only)
    summary_metrics.json   comparable metric subset (+ selected_cvi/algorithm)
    raw_result.json        ML2DAC meta-learning info + optimizer history summary
    run_log.txt            human-readable per-record log
    ml2dac_artifact_report.json   MKR availability report
  dataset_summary.json     best record by ARI (+ CVI/algorithm/warmstart/artifact status)
  README_ML2DAC_RUN.md
```
The dataset-level headline uses **Best View by ARI** (same as AutoML4Clust /
ClustOpt Best-View Mean ARI).

### Per subfamily/split run (rollup)
```
results_analysis/clustering_repository/external_baselines/ML2DAC/<ts>_split_XX_<subfamily>/
  run_summary.csv                # dataset_id, family, subfamily, difficulty, split_id,
                                 # status_1d_x/1d_y/2d, best_record_type, best_ari/nmi/ami,
                                 # selected_k, true_k, k_correct, selected_algorithm,
                                 # selected_cvi, artifact_status, total_runtime_sec, error_message
  failed_datasets.csv  skipped_datasets.csv  missing_artifacts_datasets.csv
  run_config.json      README_RUN_SUMMARY.md
```

### Metrics (identical to AutoML4Clust / ClustOpt)
ARI, NMI, AMI, v_measure, homogeneity, completeness, fowlkes_mallows, purity,
predicted/true noise ratios + noise precision/recall/f1, selected_k, true_k,
k_error, abs_k_error, exact_k_match (+ ClustOpt aliases k_signed_error /
k_abs_error / k_correct), n_predicted_clusters_excluding_noise, n_noise_points,
runtime_sec, status. Noise label = -1, excluded from selected_k.

## Robustness

Every failure mode is captured into `result.json` (`status="failed"` +
`error_type`/`error_message`/`traceback`/`stage`, and `failure_kind` for
artifacts/repo issues) and the pipeline continues: missing repo, missing
PYTHONPATH, missing MKR/artifacts, import errors, ApplicationPhase errors, SMAC
crashes, invalid/degenerate labels, missing files, serialization failures,
multiprocessing exceptions. `--overwrite` is required to recompute a completed
record; otherwise completed records are skipped.

## Known dependency / environment issues

* `smac==1.4.0` (pyrfr) + `hdbscan==0.8.32` need SWIG + a C++ compiler — use the
  conda-forge prebuilt binaries (the WSL conda recipe does this).
* `pandas==1.1.5` is required (ApplicationPhase imports
  `pandas.core.common.SettingWithCopyWarning`).
* `evaluated_configs.csv` is Git-LFS; if it is a stub, `git lfs pull` in the
  submodule.
* ML2DAC reads the 134 MB EC + extracts pymfe meta-features per record →
  heavier than AutoML4Clust; consider fewer workers.

## Still to do later (out of scope here)
* Aggregation across ML2DAC results (all splits/subfamilies → one table).
* Comparison tables ML2DAC vs. ClustOpt (and vs. AutoML4Clust).
* Statistical tests (e.g. Wilcoxon signed-rank) vs. ClustOpt.
* MKR (re)building is **not** needed (the trained MKR ships with the repo).
