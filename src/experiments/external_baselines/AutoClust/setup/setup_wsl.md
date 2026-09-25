> **Partly historical.** The environment set-up on this page (reuse of the ML2DAC environment, `check_autoclust_install.py`) also serves the reported AutoClust arms. The smoke-test and run commands (`smoke_test_autoclust`, `run_autoclust_dataset`) belong to AutoClust's native adapter, which is not included. The reported arms are run as described in `baselines/README.md`.

# AutoClust setup (WSL / Linux) — reuse the ML2DAC env

AutoClust is the ML2DAC authors' re-implementation living inside the ML2DAC
submodule, so it imports the **same ML2DAC building blocks** (SMAC, ConfigSpace,
the CVI/ClusteringCS/MetaFeatureExtractor modules). It therefore needs **no new
environment** — reuse the existing `ml2dac` conda env.

## 0. Prerequisites
- The `ml2dac` conda env already exists (see
  `experiments/external_baselines/ML2DAC/setup/`). It has Python 3.9, smac==1.4.0,
  ConfigSpace==0.6.0, hdbscan, pymfe, scikit-learn==0.24.2, pandas==1.1.5, etc.
- The ML2DAC submodule is checked out at `external/ml2dac`
  (`git submodule update --init --recursive`), including the AutoClust artifacts
  under `external/ml2dac/src/Experiments/RelatedWork/related_work/`.

## 1. Verify (no install needed)
```bash
cd .
~/miniconda3/envs/ml2dac/bin/python \
  experiments/external_baselines/AutoClust/setup/check_autoclust_install.py --repo-root .
```
Expect `VERDICT: AutoClust READY (env + artifacts ok)`.

## 2. Smoke test (tiny synthetic)
```bash
~/miniconda3/envs/ml2dac/bin/python \
  -m experiments.external_baselines.AutoClust.runners.smoke_test_autoclust --repo-root .
```
Output under
`results_analysis/clustering_repository/external_baselines_smoke_tests/AutoClust/`.

## 3. One real dataset (short budget)
```bash
~/miniconda3/envs/ml2dac/bin/python \
  -m experiments.external_baselines.AutoClust.runners.run_autoclust_dataset \
  --dataset-folder "<dataset_folder>" --split-id 1 \
  --config experiments/external_baselines/AutoClust/configs/autoclust_smoke_test_config.json \
  --record-types 2d --repo-root . --overwrite
```

## PYTHONPATH / imports
Runners are invoked as `python -m experiments.external_baselines.AutoClust...` from
the repo root (repo root on `sys.path`). The adapter additionally prepends
`external/ml2dac/src` to `sys.path` at runtime. **No chdir is needed** — the adapter
passes absolute artifact paths and does not import `AutoClust.py` (which has heavy
import-time side effects).

## Notes / gotchas
- Run real experiments under **WSL/Linux**: the hard per-record timeout uses
  fork/setsid/killpg, and SMAC spawns pynisher subprocesses.
- Heavy per record: MeanShift landmarking (O(n²)) + 7 internal CVIs incl. DBCV per
  SMAC eval. Use small `--workers` (1–2). See the runtime audit before the full run.
