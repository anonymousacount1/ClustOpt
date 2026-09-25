# AutoML4Clust — external AutoClustering baseline

This package integrates **AutoML4Clust**
(https://github.com/tschechlovdev/Automl4Clust) as an external, black-box
AutoClustering baseline, for a **fair** comparison against the ClustOpt
experiments on the *same* datasets, splits, and record types.

It is intentionally **model-specific and isolated**, so sibling baselines
(`ML2DAC`, `cSmartML`) can be added later without entanglement.

---

## What AutoML4Clust is

AutoML4Clust treats clustering as an AutoML problem: it searches over a
configuration space of *k-center clustering algorithms* (k-Means, MiniBatch
k-Means, GMM, k-Medoids) and the number of clusters *k*, using a state-of-the-art
optimizer (we use **SMAC**, Bayesian optimization with random forests) to
minimise an **internal** clustering metric (default here: Silhouette). It returns
the best configuration `{k, algorithm}` it found.

## Why it is included

It is a strong, widely-cited *automated* clustering baseline that — like ClustOpt
— selects both an algorithm and *k* without being told the true number of
clusters. That makes it a fair external yardstick for ClustOpt's metric-driven
optimisation.

## How it differs from ClustOpt

| | ClustOpt | AutoML4Clust |
| --- | --- | --- |
| Objective | Metric-utility–weighted CVI(s), incl. geometry/image metrics | A single internal CVI (Silhouette / CH / DBI) |
| Search | ClustOpt search algorithms (Optuna / brute force) | SMAC / Random / Hyperband / BOHB |
| Algorithms | Broad (KMeans, DBSCAN, HDBSCAN, GMM, Spectral, …) | 4 k-center algorithms |
| Budget | Time/trial budget per record | **Iteration budget** (`n_loops`) |
| Metric inputs | Rich CVI feature space | Raw partition space only |

## Where the external repository lives

Added as a **git submodule**:

```
external/Automl4Clust/        # never modified; wrapped only via adapters/
```

Initialise it (if a fresh clone) with:

```
git submodule update --init --recursive
```

---

## Fairness contract (what is NOT allowed)

AutoML4Clust receives **only the partition space**. It never receives:

* ❌ true labels,
* ❌ true K,
* ❌ ClustOpt metric utilities / utility vectors,
* ❌ family / subfamily / difficulty metadata.

True labels and true K are used **only for external evaluation** (ARI/NMI/AMI,
K-accuracy) *after* AutoML4Clust has predicted. This is enforced structurally:
the adapter's `fit_predict` is handed only the partition array.

---

## Record types and the partition / evaluation split

The three record types map to ClustOpt's views and define the **partition space**
(what AutoML4Clust clusters on); the **evaluation space** is always the full 2-D
space:

| record_type | ClustOpt view | partition_space | evaluation_space |
| --- | --- | --- | --- |
| `1d_x` | `x_only` | `X[:, [0]]` | `X[:, [0, 1]]` |
| `1d_y` | `y_only` | `X[:, [1]]` | `X[:, [0, 1]]` |
| `2d`   | `xy_2d`  | `X[:, [0, 1]]` | `X[:, [0, 1]]` |

The dataset loader reuses the canonical ClustOpt loader
(`utility_generation/record_loader.py` + `view_builder.py`) so the spaces are
*identical* to what ClustOpt evaluated.

---

## The recommended configuration (paper-level comparison)

`configs/automl4clust_smac_2d.json` (and the `1d_x` / `1d_y` variants):

```jsonc
{
  "optimizer": "SMAC",          // primary optimizer
  "metric": "silhouette",       // internal CVI to optimise
  "n_loops": 100,               // ENFORCED budget: optimizer iterations
  "time_budget_sec": 120,       // provenance only (see "Budget" below)
  "random_state": 42,
  "use_warmstarting": false,
  "fail_on_invalid_clustering": false,
  "normalize_input": true,      // StandardScaler on the partition space
  "k_range": null,              // null = AutoML4Clust default [2, n/10], clamped
  "algorithms": null            // null = the 4 default k-center algorithms
}
```

### Budget — important

AutoML4Clust's **native budget is an iteration count** (`n_loops` → SMAC's
`runcount-limit`), **not** wall-clock time. ClustOpt's final experiments use a
120 s/record philosophy; AutoML4Clust cannot enforce an exact wall-clock budget
without modifying its source, so we:

* enforce a fixed **iteration budget** `n_loops` (default **100**) — the closest
  deterministic, comparable equivalent; and
* record `time_budget_sec` (120) for provenance / fairness alignment.

An optional best-effort wall-clock guard exists
(`enforce_wallclock_timeout: true`) but is **off by default** because the
underlying optimizers cannot be interrupted cleanly. Tune `n_loops` if you want
AutoML4Clust's per-record runtime to land near ClustOpt's 120 s in your hardware.

### SMAC vs. fallback

SMAC is the **preferred** optimizer. If SMAC cannot be installed in your
environment, set `"optimizer": "Random"` as a documented fallback — but note the
upstream `RandomOptimizer` has a known bug (it references an undefined global),
so prefer fixing SMAC (most reliably via **WSL**, see `setup/`).

---

## Environment

AutoML4Clust needs a **dedicated, isolated** environment (legacy stack: `smac`,
`hpbandster`, `ConfigSpace`, `scikit-learn==0.21.3`, `scikit-learn-extra`,
`pyclustering`). It will **not** run in the main project `.venv` (Python 3.12,
scikit-learn 1.6).

* Windows: `setup/setup_windows.md` (+ `create_env.ps1`,
  `install_automl4clust_requirements.ps1`)
* WSL/Ubuntu (recommended): `setup/setup_wsl.md` (+ `create_env.sh`,
  `install_automl4clust_requirements.sh`)

The adapter imports AutoML4Clust **lazily** and wires `external/Automl4Clust/src`
onto `sys.path` automatically, so you only need the dependency stack installed in
the active env.

---

## How to run

### 1. Smoke test (verify the environment)

```
python -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust
```

Reports `Import OK → Fit/predict OK → Labels extracted OK → Metrics computed OK`
on success; on failure prints the exact error, Python version, `sys.path`, and
whether the submodule exists. Saves to
`results_analysis/clustering_repository/external_baselines_smoke_tests/AutoML4Clust/`.

### 2. One dataset

```
python -m experiments.external_baselines.AutoML4Clust.runners.run_automl4clust_dataset ^
  --dataset-folder "<dataset_folder>" ^
  --split-id 1 ^
  --config "experiments/external_baselines/AutoML4Clust/configs/automl4clust_smac_2d.json" ^
  --record-types 1d_x 1d_y 2d ^
  --repo-root "." ^
  --overwrite
```

### 3. One subfamily + split (the final runner)

```
python -m experiments.external_baselines.AutoML4Clust.runners.run_automl4clust_subfamily_split ^
  --subfamily-root "<path_to_subfamily_folder>" ^
  --split-assignment-csv "results_analysis/clustering_repository/analyzed_data/experiment_splits/dataset_split_assignments.csv" ^
  --split-id 1 ^
  --config "experiments/external_baselines/AutoML4Clust/configs/automl4clust_smac_2d.json" ^
  --record-types 1d_x 1d_y 2d ^
  --workers 8 ^
  --repo-root "."
```

Add `--dry-run` to see which datasets would be processed without running.

---

## What gets saved

### Per dataset (the primary results)

```
<dataset_folder>/experiments/split_XX/AutoML4Clust/
  1d_x/  1d_y/  2d/
    result.json          full result: status, selected_k, best_configuration,
                         all metrics, provenance, and (on failure) error+traceback
    labels.npy           predicted labels (success only)
    summary_metrics.json comparable metric subset (ARI/NMI/AMI/K-metrics/...)
    raw_result.json      AutoML4Clust optimizer history (k/score/iteration ...)
    run_log.txt          human-readable per-record log
  dataset_summary.json   best record by ARI, per-record statuses, total runtime
  README_AUTOML4CLUST_RUN.md
```

### Per subfamily/split run (rollup only)

```
results_analysis/clustering_repository/external_baselines/AutoML4Clust/<ts>_split_XX_<subfamily>/
  run_summary.csv        one row per dataset (statuses + best ARI)
  failed_datasets.csv    datasets with a fatal error
  skipped_datasets.csv   datasets where every record was skipped
  run_config.json        exact run configuration
  README_RUN_SUMMARY.md
```

### Metrics computed (match ClustOpt exactly)

ARI, NMI, AMI, V-measure, homogeneity, completeness, Fowlkes–Mallows, purity;
`selected_k`, `true_k`, `k_error`, `abs_k_error`, `exact_k_match`,
`n_predicted_clusters_excluding_noise`, `n_noise_points`,
predicted/true noise ratios; `runtime_sec`; `status`. Noise convention: label
`-1`, excluded from `selected_k` (same as ClustOpt's `k_metrics`).

---

## Robustness

Every failure mode is captured into `result.json` with `status="failed"`,
`error_type`, `error_message`, `traceback`, `stage`, and `runtime_sec`, and the
pipeline continues: import errors, AutoML4Clust crashes, invalid labels,
single-cluster / all-noise results, missing files, serialization failures, and
worker/multiprocessing exceptions. Results are written atomically and are
long-path-safe on Windows. `--overwrite` is required to recompute a completed
record; otherwise completed records are skipped (`[SKIP] ...`).

---

## Verified environment

The baseline has been validated end-to-end (smoke test green for `1d_x` /
`1d_y` / `2d`) in an isolated **conda env on WSL**: Ubuntu 24.04, Miniconda,
Python 3.7, with conda-forge binaries for `pyrfr` / `swig` / `scikit-learn
0.21.3` / `scikit-learn-extra` and pip for `smac 0.12` / `hpbandster` /
`ConfigSpace`. Reproduce with `setup/create_env_wsl_conda.sh` (see
`setup/setup_wsl.md`, "Proven recipe").

## Known issues with the old dependencies (and the fixes applied)

* `smac==0.12.0` / `pyrfr` need SWIG + a C++ compiler — use the **conda-forge
  prebuilt** `pyrfr`/`swig` (the conda recipe does this) to avoid building.
* `scikit-learn==0.21.3` lacks wheels for Python ≥ 3.9 — use Python 3.6/3.7/3.8.
* **`pynisher`**: `smac 0.12` calls `pynisher.enforce_limits`, removed in
  `pynisher 1.x`; pip installs 1.x and SMAC crashes on the first run. Fix:
  pin **`pynisher==0.5.0`** (done by the setup scripts).
* **`scikit-learn-extra==0.0.3`** (the upstream pin) is **yanked from PyPI**;
  install conda-forge's **`0.1.0b2`** (compatible with scikit-learn 0.21.3).
  Needed because AutoML4Clust imports `KMedoids` at module load.
* **Upstream bug (handled by an adapter shim, no source edit):**
  `Metrics/MetricHandler.py::score_metric` references
  `MetricCollection.SILHOUETTE_SAMPLE_10`, which is never defined on the class,
  so *every* metric evaluation raises `AttributeError` and SMAC aborts. The
  adapter defines that attribute at runtime before optimizing
  (`adapters/automl4clust_adapter.py::_apply_upstream_shims`). On Linux,
  pynisher's forked evaluation subprocesses inherit the patch.
* `hpbandster` needs `Pyro4` + `serpent`.
* The upstream `requirements.txt` contains Ubuntu-only packages
  (`systemd-python`, `python-apt`, …) — never install it verbatim; use the
  curated `install_automl4clust_requirements.*` scripts or the conda recipe.

---

## Still to be implemented later (out of scope for this milestone)

* Aggregation across AutoML4Clust results (all splits/subfamilies → one table).
* Comparison tables AutoML4Clust vs. ClustOpt (per record type / difficulty /
  family).
* Statistical tests (e.g. Wilcoxon signed-rank) AutoML4Clust vs. ClustOpt.
