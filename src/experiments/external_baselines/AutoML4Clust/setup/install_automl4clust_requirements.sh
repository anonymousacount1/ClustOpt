#!/usr/bin/env bash
# Install a curated AutoML4Clust dependency stack into the ACTIVE environment
# (WSL/Linux). Installs only what the SMAC clustering path actually needs.
#
# The upstream requirements.txt pins many Ubuntu-system packages
# (systemd-python, python-apt, ufw, cloud-init, ...) that are unused by
# AutoML4Clust; we install the curated subset instead.
#
# Run AFTER activating the dedicated env created by create_env.sh:
#   source .venv_automl4clust_wsl/bin/activate
#
# Build prerequisites (once): sudo apt install -y build-essential swig
#
# Options (env vars):
#   SKIP_SMAC=1       install everything except smac
#   INCLUDE_PYMFE=1   also install pymfe (meta-learning/warmstart; not needed)
set -euo pipefail

PY="$(command -v python)"
echo "Using python: ${PY}"
"${PY}" --version

pip_install() {
  echo "[pip] $*"
  "${PY}" -m pip install "$@"
}

# 1) Core numerical stack compatible with legacy scikit-learn.
pip_install "Cython==0.29.17"
pip_install "numpy==1.18.4" "scipy==1.4.1" "pandas==1.0.3"

# 2) Legacy scikit-learn + k-Medoids + clustering library.
pip_install "scikit-learn==0.21.3"
pip_install "scikit-learn-extra==0.0.3"
pip_install "pyclustering"

# 3) Configuration space + optimizers.
pip_install "ConfigSpace==0.4.12"
pip_install "scikit-optimize==0.5.2"
pip_install "Pyro4" "serpent"          # hpbandster runtime deps
pip_install "hpbandster==0.7.4"

# 4) SMAC (Bayesian optimizer) -- primary optimizer for the baseline.
if [ "${SKIP_SMAC:-0}" = "1" ]; then
  echo "[info] Skipping smac (SKIP_SMAC=1)."
else
  pip_install "smac==0.12.0"
  # smac 0.12 calls pynisher.enforce_limits, removed in pynisher 1.x. pip pulls
  # 1.x by default and SMAC then crashes on the first run -- pin the 0.5.0 API.
  pip_install "pynisher==0.5.0"
fi

# 5) Optional meta-learning (NOT used by the SMAC baseline; warmstarting is off).
if [ "${INCLUDE_PYMFE:-0}" = "1" ]; then
  pip_install "pymfe==0.0.3"
fi

echo ""
echo "[done] Curated AutoML4Clust stack installed."
echo "Verify: python -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust --repo-root \$(pwd)"
