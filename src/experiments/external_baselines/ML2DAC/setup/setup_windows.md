# ML2DAC environment setup — Windows (PyCharm)

**Native Windows is discouraged for ML2DAC.** ML2DAC pins `smac==1.4.0` (pulls
`pyrfr`, a SWIG/C++ extension) and `hdbscan==0.8.32` (C++), plus `pymfe`,
`ConfigSpace==0.6.0`, `pandas==1.1.5` on **Python 3.9**. Building `pyrfr` and
`hdbscan` on Windows needs SWIG + the MSVC C++ Build Tools, and several pins lack
Windows wheels. This is the same class of legacy stack as AutoML4Clust, which we
ultimately ran from **WSL** — do the same here.

> **Recommended:** follow `setup_wsl.md` (conda recipe). It is the path we verified.

This env must **not** be installed into your main project `.venv` (Python 3.12)
or the AutoML4Clust env (Python 3.7). Keep it isolated.

---

## If you still want to try native Windows

1. Install **Python 3.9** (`py -3.9 --version` to confirm) and **SWIG** +
   **MSVC C++ Build Tools** ("Desktop development with C++").
2. From the repo root:
   ```powershell
   powershell -ExecutionPolicy Bypass -File experiments\external_baselines\ML2DAC\setup\create_env.ps1 -PythonLauncher "py -3.9"
   .\.venv_ml2dac\Scripts\Activate.ps1
   powershell -ExecutionPolicy Bypass -File experiments\external_baselines\ML2DAC\setup\install_ml2dac_requirements.ps1
   ```
3. Verify:
   ```powershell
   python experiments\external_baselines\ML2DAC\setup\check_ml2dac_install.py
   python -m experiments.external_baselines.ML2DAC.runners.smoke_test_ml2dac
   ```

If `smac`/`pyrfr` or `hdbscan` fail to build (likely), switch to WSL.

---

## MetaKnowledgeRepository (MKR)

The trained MKR ships with the submodule
(`external\ml2dac\src\MetaKnowledgeRepository`). The `*.csv` files are Git-LFS;
if `evaluated_configs.csv` is a tiny pointer stub, run inside the submodule:
```powershell
cd external\ml2dac; git lfs install; git lfs pull; cd ..\..
```
The artifact locator / smoke test / runners report `has_pretrained_artifacts` and
exactly what is missing — the infrastructure works (writes structured "artifacts
missing" / "import failed" results) even before the env is ready.

---

## Using the dedicated env from PyCharm

Easiest: open the PyCharm **Terminal**, type `wsl`, then follow `setup_wsl.md`.
Results still write into the Windows project tree
(`<dataset>\experiments\split_XX\ML2DAC\`), visible from both Windows and WSL.
