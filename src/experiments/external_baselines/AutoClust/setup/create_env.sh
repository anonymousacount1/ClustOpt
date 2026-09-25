#!/usr/bin/env bash
# AutoClust reuses the existing `ml2dac` conda env -- there is NO separate env to
# create. This script just verifies the ml2dac env exists and that AutoClust is
# ready (imports + artifacts).
set -uo pipefail
ENV_NAME="${ENV_NAME:-ml2dac}"
PY="$HOME/miniconda3/envs/${ENV_NAME}/bin/python"

if [ ! -x "$PY" ]; then
  echo "[error] ml2dac env not found at $PY"
  echo "        Create it first: experiments/external_baselines/ML2DAC/setup/create_env_wsl_conda.sh"
  exit 1
fi
echo "[ok] reusing env: $PY"
"$PY" experiments/external_baselines/AutoClust/setup/check_autoclust_install.py --repo-root "$(pwd)"
