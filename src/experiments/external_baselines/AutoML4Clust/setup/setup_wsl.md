# AutoML4Clust environment setup — WSL / Ubuntu (recommended)

The legacy AutoML4Clust stack (`smac==0.12.0` → `pyrfr`, `hpbandster`,
`ConfigSpace==0.4.12`, `scikit-learn==0.21.3`, `scikit-learn-extra`,
`pyclustering`) was written for **Ubuntu + Python 3.6** and installs far more
reliably on Linux than on native Windows. On Windows you already have WSL
available; the repository is reachable from WSL under `/mnt/c/...`, so you can run
the **exact same** runners and write to the **exact same** dataset folders.

> This environment is fully isolated from your main project `.venv`. Nothing
> here touches your Windows Python.

---

## ✅ Proven recipe (conda) — verified working on this machine

The fastest, **verified** path (no sudo, no apt, no SWIG install) uses the
existing Miniconda in WSL and prebuilt conda-forge binaries. This is the recipe
the smoke test was actually validated with (WSL Ubuntu 24.04 + Miniconda;
Python 3.7; smoke test green for `1d_x` / `1d_y` / `2d`):

```bash
# from the repo root, inside WSL
bash experiments/external_baselines/AutoML4Clust/setup/create_env_wsl_conda.sh

# verify end-to-end
REPO=.
~/miniconda3/envs/automl4clust/bin/python \
  -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust \
  --repo-root "$REPO" --record-types 1d_x 1d_y 2d
```

The script bakes in two non-obvious fixes discovered during setup:

1. **`scikit-learn-extra`** — the upstream pin `==0.0.3` was **yanked from
   PyPI**; conda-forge's `0.1.0b2` is used instead (compatible with
   scikit-learn 0.21.3). Required because AutoML4Clust imports `KMedoids` at
   module load.
2. **`pynisher`** — pip pulls `1.x` (which removed `enforce_limits`), and
   `smac==0.12` then crashes on the first run. Pinned to `pynisher==0.5.0`.

There is also one **adapter-side runtime shim** (no external-source edit) for a
genuine upstream bug: `Metrics/MetricHandler.py` references
`MetricCollection.SILHOUETTE_SAMPLE_10`, which is never defined, so every metric
evaluation would crash. The adapter defines that attribute at runtime before
optimizing (see `adapters/automl4clust_adapter.py::_apply_upstream_shims`).

If you prefer a pip/venv install or need Python from source, the manual steps
below still apply — but the conda recipe above is recommended.

---

## 1. Install WSL + Ubuntu (one-time)

In an elevated PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then open the **Ubuntu** terminal and create your UNIX user.

---

## 2. Get a Python 3.6/3.7 interpreter

Ubuntu 18.04 ships Python 3.6 (ideal). On newer Ubuntu, use the deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.6 python3.6-venv python3.6-dev
# (python3.7 also works and is easier to find on newer Ubuntu)
```

Install build tools needed by SMAC/pyrfr and scikit-learn-extra:

```bash
sudo apt install -y build-essential swig
```

---

## 3. Create the dedicated environment

From the repo root **inside WSL** (note the `/mnt/c/...` path):

```bash
cd .

# Helper script: creates .venv_automl4clust_wsl and upgrades pip tooling
bash experiments/external_baselines/AutoML4Clust/setup/create_env.sh python3.6

source .venv_automl4clust_wsl/bin/activate
```

---

## 4. Install AutoML4Clust requirements

```bash
bash experiments/external_baselines/AutoML4Clust/setup/install_automl4clust_requirements.sh
```

This installs a **curated** subset of the upstream `requirements.txt`. The
upstream file pins many Ubuntu-system packages (`systemd-python`, `python-apt`,
`ufw`, `cloud-init`, …) that are irrelevant to AutoML4Clust and would fail; we
install only what the SMAC clustering path actually needs.

If you prefer to try the upstream file verbatim (not recommended), only the
clustering/optimizer lines matter:

```
scikit-learn==0.21.3
scikit-learn-extra
pymfe==0.0.3
scikit-optimize==0.5.2
smac==0.12.0
hpbandster==0.7.4
ConfigSpace==0.4.12
pyclustering
```

---

## 5. PYTHONPATH

The runners add `external/Automl4Clust/src` to `sys.path` automatically. For a
manual REPL:

```bash
export PYTHONPATH="external/Automl4Clust/src:."
```

---

## 6. Verify

```bash
cd .
python -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust \
  --repo-root .
```

Expected healthy output:

```
[AutoML4Clust Smoke Test] Starting
[AutoML4Clust Smoke Test] Import OK
[AutoML4Clust Smoke Test] Fit/predict OK (2d)
[AutoML4Clust Smoke Test] Labels extracted OK (2d, selected_k=...)
[AutoML4Clust Smoke Test] Metrics computed OK (2d): ari=... nmi=... ami=...
[AutoML4Clust Smoke Test] Saved results to ...
```

---

## 7. Notes on running the real experiment from WSL

* Paths: pass `--repo-root .`
  and Linux-style `--subfamily-root` / `--split-assignment-csv` paths under
  `/mnt/c/...`.
* Long paths / `\\?\` prefixing are Windows-only and become no-ops on Linux —
  the output writer handles this transparently.
* Multiprocessing (`--workers`) works the same; one dataset per worker.
* Output still lands inside each Windows dataset folder
  (`experiments/split_XX/AutoML4Clust/`), visible from both Windows and WSL.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `pyrfr` build error | `sudo apt install -y swig build-essential`, then reinstall `smac==0.12.0`. |
| `scikit-learn==0.21.3` build fails | Use Python 3.6/3.7 (deadsnakes); newer Python lacks compatible wheels. |
| `hpbandster` Pyro4/serpent errors | `pip install Pyro4 serpent`. |
| `pymfe` conflicts | Optional (meta-learning/warmstart only); skip it — the SMAC baseline does not use warmstarting. |
| SMAC very slow | Reduce `n_loops` in the config; the budget is iteration-based. |
