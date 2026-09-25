#!/usr/bin/env bash
# PROVEN working setup for AutoML4Clust on WSL using conda (no sudo required).
#
# This is the recipe that was actually verified end-to-end on this machine
# (WSL Ubuntu 24.04, Miniconda) -- the smoke test passes with it. It builds an
# isolated conda env `automl4clust` with Python 3.7 and the legacy AutoML4Clust
# stack, using prebuilt conda-forge binaries for the pieces that will not build
# from pip (pyrfr, swig, scikit-learn 0.21.3, scikit-learn-extra).
#
# Two non-obvious fixes are baked in (see comments):
#   * scikit-learn-extra: pin ==0.0.3 is YANKED from PyPI; use conda-forge's
#     0.1.0b2, which is compatible with scikit-learn 0.21.3.
#   * pynisher: pip pulls 1.x (no `enforce_limits`), which breaks smac 0.12;
#     pin pynisher==0.5.0 (the upstream pin).
#
# Usage (from anywhere):
#   bash create_env_wsl_conda.sh
# Optional: set CONDA=/path/to/conda and ENV=name to override.
set -uo pipefail

CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV="${ENV:-automl4clust}"
PY="$("$CONDA" info --base)/envs/$ENV/bin/python"
PIP="$("$CONDA" info --base)/envs/$ENV/bin/pip"

if [ ! -x "$CONDA" ]; then
  echo "[error] conda not found at $CONDA. Install Miniconda or set CONDA=..." >&2
  exit 1
fi

step() { echo ""; echo "########## $* ##########"; }

step "1. create env: Python 3.7 (conda-forge only)"
"$CONDA" create -y -n "$ENV" -c conda-forge --override-channels python=3.7 \
  || { echo "CREATE_FAILED"; exit 11; }

step "2. conda-forge binaries (build tools + hard-to-build pkgs)"
# pyrfr (needs SWIG/C++), legacy scikit-learn, numpy/scipy ABI-matched.
"$CONDA" install -y -n "$ENV" -c conda-forge --override-channels \
  swig pyrfr 'scikit-learn=0.21.3' 'numpy=1.18*' 'scipy=1.4*' pandas cython \
  || { echo "CONDA_INSTALL_FAILED"; exit 12; }

step "3. scikit-learn-extra (k-Medoids) -- conda-forge (pip pin ==0.0.3 is yanked)"
"$CONDA" install -y -n "$ENV" -c conda-forge --override-channels scikit-learn-extra \
  || echo "WARN: scikit-learn-extra (conda-forge) failed"

step "4. pip: ConfigSpace + optimizers"
"$PIP" install --no-input "ConfigSpace==0.4.12" || { echo "PIP_CONFIGSPACE_FAILED"; exit 13; }
"$PIP" install --no-input "scikit-optimize==0.5.2" || echo "WARN: skopt failed"
"$PIP" install --no-input "Pyro4" "serpent" "netifaces" || echo "WARN: hpbandster deps failed"
"$PIP" install --no-input "hpbandster==0.7.4" || echo "WARN: hpbandster failed"

step "5. pip: smac (primary optimizer) + the pynisher 0.5.0 fix"
"$PIP" install --no-input "smac==0.12.0" || { echo "PIP_SMAC_FAILED"; exit 14; }
# smac 0.12 uses pynisher.enforce_limits (removed in pynisher 1.x); pin 0.5.0.
"$PIP" install --no-input "pynisher==0.5.0" || { echo "PIP_PYNISHER_FAILED"; exit 15; }

step "6. pip: clustering extras"
"$PIP" install --no-input "pyclustering" || echo "WARN: pyclustering failed"

step "7. import check"
"$PY" - <<'PYEOF'
import importlib
for m in ["ConfigSpace","smac","hpbandster","skopt","sklearn","sklearn_extra",
          "pyclustering","pyrfr","pynisher","numpy","scipy","pandas"]:
    try:
        importlib.import_module(m); print(f"  import {m}: OK")
    except Exception as e:
        print(f"  import {m}: FAIL -> {type(e).__name__}: {e}")
import sklearn, pynisher
print("sklearn", sklearn.__version__, "| pynisher.enforce_limits:", hasattr(pynisher,"enforce_limits"))
PYEOF

echo ""
echo "DONE. Verify end-to-end with:"
echo "  cd <repo_root> && $PY -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust --repo-root <repo_root>"
