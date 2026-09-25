#!/usr/bin/env bash
# Install the ML2DAC dependency stack into the ACTIVE environment (WSL/Linux).
# Prefer create_env_wsl_conda.sh (conda-forge prebuilt swig/pyrfr/hdbscan). This
# pip-only path requires build tools (sudo apt install -y build-essential swig
# python3.9-dev) for hdbscan + pyrfr to compile.
set -uo pipefail
PY="$(command -v python)"
"${PY}" --version
pin() { echo "[pip] $*"; "${PY}" -m pip install "$@"; }

pin "numpy==1.24.2" "scipy==1.8.0"
pin "ConfigSpace==0.6.0"
pin "smac==1.4.0"             # pulls pyrfr (needs swig + compiler)
pin "hdbscan==0.8.32"         # needs a C++ compiler
pin "scikit-learn==0.24.2" "joblib==1.0.0"
pin "pymfe==0.4.1"
pin "Definitions==0.2.0" || echo "WARN: Definitions optional"
# Pin pandas LAST: pymfe pulls pandas 2.x, which removes
# pandas.core.common.SettingWithCopyWarning (imported by ML2DAC ApplicationPhase).
pin "pandas==1.1.5"

echo "[done] Verify: python experiments/external_baselines/ML2DAC/setup/check_ml2dac_install.py --repo-root \$(pwd)"
