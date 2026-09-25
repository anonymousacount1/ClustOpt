> **Partly historical.** The environment set-up on this page (reuse of the ML2DAC environment, `check_autoclust_install.py`) also serves the reported AutoClust arms. The smoke-test and run commands (`smoke_test_autoclust`, `run_autoclust_dataset`) belong to AutoClust's native adapter, which is not included. The reported arms are run as described in `baselines/README.md`.

# AutoClust setup (native Windows) — limited; prefer WSL

AutoClust reuses the **ML2DAC** stack (SMAC, ConfigSpace, …) and the ML2DAC
submodule artifacts. Native Windows is **not recommended** for real runs:

- SMAC (pyrfr) is hard to install natively, and the hard per-record timeout uses
  `fork`/`setsid`/`killpg` (Linux-only). The subfamily runner needs WSL/Linux.
- Use WSL with the existing `ml2dac` conda env (see `setup_wsl.md`).

For an import/smoke check only, use whatever env has the ML2DAC stack:
```powershell
cd .
conda run -n ml2dac python `
  experiments\external_baselines\AutoClust\setup\check_autoclust_install.py --repo-root .
```
Run the final subfamily experiments under **WSL** (`setup_wsl.md`).
