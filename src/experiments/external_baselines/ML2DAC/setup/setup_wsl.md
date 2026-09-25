# ML2DAC environment setup — WSL / Ubuntu (recommended)

ML2DAC requires **Python 3.9** and a specific legacy stack (`smac==1.4.0`,
`ConfigSpace==0.6.0`, `hdbscan==0.8.32`, `pymfe==0.4.1`, `scikit-learn==0.24.2`,
`pandas==1.1.5`, `numpy==1.24.2`, `scipy==1.8.0`). `smac`/`pyrfr` and `hdbscan`
need a C++ compiler + SWIG, which build far more reliably on Linux. This env is
fully isolated from your main project env **and** from the AutoML4Clust env
(which is Python 3.7).

> Path mapping: `.`
> maps to `.` in WSL.

---

## ✅ Proven recipe (conda) — no sudo

The same Miniconda used for AutoML4Clust works here. From the repo root inside
WSL:

```bash
cd .
bash experiments/external_baselines/ML2DAC/setup/create_env_wsl_conda.sh
```

This builds an isolated conda env `ml2dac` (Python 3.9): conda-forge prebuilt
`swig` / `pyrfr` / `hdbscan` (so nothing has to compile), then pip for
`smac==1.4.0`, `ConfigSpace==0.6.0`, `pymfe==0.4.1`, `scikit-learn==0.24.2`,
`pandas==1.1.5`. It then runs `check_ml2dac_install.py` automatically.

Verify end-to-end:

```bash
REPO=.
~/miniconda3/envs/ml2dac/bin/python \
  -m experiments.external_baselines.ML2DAC.runners.smoke_test_ml2dac --repo-root "$REPO"
```

A healthy install prints:

```
[ML2DAC Smoke Test] External repo exists
[ML2DAC Smoke Test] Artifact check ... has_pretrained=True missing=[]
[ML2DAC Smoke Test] Import check ... OK
[ML2DAC Smoke Test] Full ML2DAC runnable: yes
[ML2DAC Smoke Test] Fit/predict OK (2d, selected_k=..., cvi=..., algo=...)
[ML2DAC Smoke Test] Metrics computed OK (2d): ari=... nmi=... ami=...
```

---

## Manual venv path (if you prefer pip + deadsnakes)

```bash
sudo apt update
sudo apt install -y software-properties-common build-essential swig
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.9 python3.9-venv python3.9-dev

cd .
bash experiments/external_baselines/ML2DAC/setup/create_env.sh python3.9
source .venv_ml2dac_wsl/bin/activate
bash experiments/external_baselines/ML2DAC/setup/install_ml2dac_requirements.sh
```

---

## MetaKnowledgeRepository (MKR)

The trained MKR **ships with the submodule** at
`external/ml2dac/src/MetaKnowledgeRepository` (evaluated_configs.csv ~134 MB,
meta-feature kdtrees, trained CVI classifiers). No learning phase is needed.

* `*.csv` in the submodule is Git-LFS-tracked. If `evaluated_configs.csv` looks
  like a tiny `version https://git-lfs...` stub, fetch the objects:
  ```bash
  cd external/ml2dac && git lfs install && git lfs pull && cd -
  ```
* The artifact locator (`check_ml2dac_install.py`, the smoke test, and every
  runner) reports `has_pretrained_artifacts` and exactly what is missing.

---

## PYTHONPATH

The runners add `external/ml2dac/src` to `sys.path` automatically. For a manual
REPL:

```bash
export PYTHONPATH="external/ml2dac/src:."
```

---

## Running the experiment from WSL (outputs land in the Windows project)

Use Linux-style `/mnt/c/...` paths and `--repo-root /mnt/c/...`. Results are
written inside each Windows dataset folder
(`experiments/split_XX/ML2DAC/`), visible from both Windows and WSL.
Multiprocessing works the same. **Note:** ML2DAC reads the 134 MB
`evaluated_configs.csv` + extracts pymfe meta-features per record, so it is
heavier than AutoML4Clust — consider `--workers 4`–`6` if RAM is constrained.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `pyrfr` / `smac` build error | use the conda recipe (prebuilt `pyrfr`/`swig`) or `sudo apt install -y swig build-essential`. |
| `hdbscan` build error | `conda install -c conda-forge hdbscan` (prebuilt) or install a compiler. |
| `evaluated_configs.csv` is a tiny stub | `cd external/ml2dac && git lfs pull`. |
| `ImportError: cannot import name 'SettingWithCopyWarning'` | you have a too-new pandas; pin `pandas==1.1.5`. |
| `ConfigSpace`/`smac` version errors | pin `ConfigSpace==0.6.0` + `smac==1.4.0`. |
| sklearn/numpy ABI warning | the ML2DAC pins (`scikit-learn==0.24.2`, `numpy==1.24.2`) are what the authors used; keep them. |
