#!/usr/bin/env bash
# Proven WSL/conda setup for ML2DAC (no sudo). Mirrors the AutoML4Clust recipe
# but targets Python 3.9 + the ML2DAC stack (smac 1.4.0, ConfigSpace 0.6.0,
# hdbscan 0.8.32, pymfe 0.4.1, scikit-learn 0.24.2, pandas 1.1.5, ...).
#
# Strategy: conda-forge prebuilt binaries for the hard-to-build pieces
# (swig, pyrfr, hdbscan), pip for the rest. Builds an isolated env `ml2dac`.
#
# Usage (from repo root, inside WSL):
#   bash experiments/external_baselines/ML2DAC/setup/create_env_wsl_conda.sh
set -uo pipefail

CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV="${ENV:-ml2dac}"
PY="$("$CONDA" info --base)/envs/$ENV/bin/python"
PIP="$("$CONDA" info --base)/envs/$ENV/bin/pip"

if [ ! -x "$CONDA" ]; then
  echo "[error] conda not found at $CONDA. Install Miniconda or set CONDA=..." >&2
  exit 1
fi
step() { echo ""; echo "########## $* ##########"; }

step "1. create env: Python 3.9 (conda-forge)"
"$CONDA" create -y -n "$ENV" -c conda-forge --override-channels python=3.9 \
  || { echo "CREATE_FAILED"; exit 11; }

step "2. conda-forge binaries: swig + pyrfr (for smac) + hdbscan"
"$CONDA" install -y -n "$ENV" -c conda-forge --override-channels \
  swig pyrfr hdbscan numpy=1.24.2 scipy=1.8.0 \
  || echo "WARN: some conda-forge installs failed (will retry via pip)"

step "3. pip: ML2DAC stack (ConfigSpace, smac, pymfe, sklearn, pandas, ...)"
"$PIP" install --no-input "ConfigSpace==0.6.0" || { echo "PIP_CONFIGSPACE_FAILED"; exit 13; }
"$PIP" install --no-input "smac==1.4.0" || { echo "PIP_SMAC_FAILED"; exit 14; }
"$PIP" install --no-input "scikit-learn==0.24.2" "joblib==1.0.0" \
  || echo "WARN: sklearn pin failed"
"$PIP" install --no-input "pymfe==0.4.1" || echo "WARN: pymfe failed"
"$PIP" install --no-input "Definitions==0.2.0" || echo "WARN: Definitions failed (optional)"
# IMPORTANT: pin pandas LAST -- pymfe pulls pandas 2.x, which removes
# pandas.core.common.SettingWithCopyWarning that ML2DAC's ApplicationPhase
# imports. Re-pin 1.1.5 after pymfe so the import works.
"$PIP" install --no-input "pandas==1.1.5" || { echo "PIP_PANDAS_FAILED"; exit 15; }

step "4. import + artifact check"
REPO="${REPO:-.}"
"$PY" "$REPO/experiments/external_baselines/ML2DAC/setup/check_ml2dac_install.py" --repo-root "$REPO" || true

echo ""
echo "DONE. Verify end-to-end with:"
echo "  cd $REPO && $PY -m experiments.external_baselines.ML2DAC.runners.smoke_test_ml2dac --repo-root $REPO"
