# AutoML4Clust environment setup — Windows (PyCharm)

AutoML4Clust is an older research prototype. Its `requirements.txt` pins
**Python 3.6** and a heavy, legacy dependency stack (`smac==0.12.0`,
`hpbandster==0.7.4`, `ConfigSpace==0.4.12`, `scikit-learn==0.21.3`,
`scikit-learn-extra`, `scikit-optimize==0.5.2`, `pyclustering`). These versions
do **not** install on the main project environment (Python 3.12, scikit-learn
1.6). You must therefore create a **dedicated, isolated environment** for
AutoML4Clust so it never breaks your main project.

> **Important:** do not `pip install` any of this into your main `.venv`.
> Everything below goes into a separate environment.

There are two supported paths on Windows:

1. **WSL / Ubuntu (recommended, most reliable)** — see `setup_wsl.md`.
   The legacy `smac`/`pyrfr`/`hpbandster` stack was built for Linux and installs
   far more smoothly there.
2. **Native Windows dedicated environment** — documented here. Workable, but the
   SMAC build (`pyrfr`, needs SWIG + a C++ compiler) is the hard part.

---

## Why a dedicated environment

The adapter (`adapters/automl4clust_adapter.py`) imports AutoML4Clust **lazily**
and only ever needs it on `sys.path`. So the dedicated environment only has to:

* have the AutoML4Clust dependency stack installed, and
* be able to import `external/Automl4Clust/src` (we wire this automatically via
  `PYTHONPATH` / `sys.path` — see below).

Everything else in this repository (the loader, metrics, runners, output writing)
runs on plain `numpy` / `pandas` / `scikit-learn`, which the dedicated env also
has.

---

## Option A — native Windows dedicated venv

### 1. Pick a Python interpreter

Try, in order:

```powershell
py -3.7 --version      # 3.7 is the most realistic Windows target
py -3.8 --version
py -3.6 --version      # original target; hard to find prebuilt wheels now
```

Python **3.7 or 3.8** is the most realistic on Windows because more prebuilt
wheels exist for the old SMAC stack. If you only have 3.10+, prefer WSL.

### 2. Create + activate the dedicated venv

From the **repo root**:

```powershell
# Creates .venv_automl4clust at the repo root (already git-ignored)
powershell -ExecutionPolicy Bypass -File experiments\external_baselines\AutoML4Clust\setup\create_env.ps1 -PythonLauncher "py -3.7"

# Activate it
.\.venv_automl4clust\Scripts\Activate.ps1
```

### 3. Install build tools (needed for SMAC / pyrfr)

`smac==0.12.0` depends on `pyrfr`, a SWIG-wrapped C++ extension:

* Install **Microsoft C++ Build Tools** (Visual Studio Build Tools →
  "Desktop development with C++").
* Install **SWIG** and put it on `PATH`
  (`choco install swig`, or download from https://www.swig.org/download.html).

Verify:

```powershell
swig -version
cl            # should print the MSVC compiler banner (from a VS dev shell)
```

### 4. Install the AutoML4Clust dependency stack

```powershell
powershell -ExecutionPolicy Bypass -File experiments\external_baselines\AutoML4Clust\setup\install_automl4clust_requirements.ps1
```

This installs a **curated** subset (the upstream `requirements.txt` contains
Ubuntu-only packages such as `systemd-python`, `python-apt`, `ufw` that cannot
install on Windows — we never use the full file directly).

### 5. Set `PYTHONPATH` so the external repo imports

The runners add `external/Automl4Clust/src` to `sys.path` automatically. If you
import AutoML4Clust manually from a REPL, set:

```powershell
$env:PYTHONPATH = "external\Automl4Clust\src;."
```

### 6. Verify the installation

```powershell
# From the repo root, with .venv_automl4clust activated:
python -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust
```

A healthy install prints:

```
[AutoML4Clust Smoke Test] Starting
[AutoML4Clust Smoke Test] Import OK
[AutoML4Clust Smoke Test] Fit/predict OK (2d)
[AutoML4Clust Smoke Test] Labels extracted OK (2d, selected_k=...)
[AutoML4Clust Smoke Test] Metrics computed OK (2d): ari=... nmi=... ami=...
[AutoML4Clust Smoke Test] Saved results to ...
```

If import fails, the smoke test prints the **exact** missing module, the Python
version, `sys.path`, and whether `external/Automl4Clust` exists.

---

## Using the dedicated env from PyCharm

* **Settings → Project → Python Interpreter → Add Interpreter → Existing** →
  point at `.venv_automl4clust\Scripts\python.exe`.
* Or, simpler: open the PyCharm **Terminal**, activate the env
  (`.\.venv_automl4clust\Scripts\Activate.ps1`), and run the `python -m ...`
  commands there. The main project keeps using its own `.venv`.

---

## Troubleshooting old dependencies

| Symptom | Fix |
| --- | --- |
| `No module named 'ConfigSpace'` | `pip install ConfigSpace==0.4.12` (needs Cython + a compiler). |
| `pyrfr` / `smac` build fails | Install SWIG + MSVC build tools (step 3). If it still fails, use **WSL** (`setup_wsl.md`) — by far the most reliable for SMAC. |
| `hpbandster` import errors (`Pyro4`, `serpent`) | `pip install Pyro4 serpent ConfigSpace==0.4.12`; hpbandster needs these. |
| `scikit-learn-extra` build fails | Needs a compiler + a compatible numpy; pin `scikit-learn==0.21.3` first, then `pip install scikit-learn-extra==0.0.3`. |
| `pyclustering` install issues | `pip install pyclustering` (pure-Python wheel usually works). |
| `scikit-learn==0.21.3` won't build on 3.10+ | Use Python 3.7/3.8, or WSL with 3.6. Newer Python lacks prebuilt 0.21.3 wheels. |
| SMAC unstable / cannot build at all | Set `"optimizer": "Random"` in the config as a **fallback** (note: upstream `RandomOptimizer` has a known bug; SMAC remains the preferred baseline). Prefer fixing SMAC via WSL. |

If SMAC simply cannot be made to work in any Windows environment, run the final
experiment from **WSL** (`setup_wsl.md`) — the adapter, runners, dataset folders,
and output paths are all identical there (the repo is reachable under
`/mnt/c/...`).
