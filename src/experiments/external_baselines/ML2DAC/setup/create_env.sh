#!/usr/bin/env bash
# Create a dedicated, isolated venv for ML2DAC (WSL/Linux).
# ML2DAC needs Python 3.9 + a legacy stack that must NOT touch the main env or
# the AutoML4Clust env (Python 3.7). This creates `.venv_ml2dac_wsl` and upgrades
# pip tooling; run install_ml2dac_requirements.sh afterwards. For the most
# reliable path (prebuilt swig/pyrfr/hdbscan) use create_env_wsl_conda.sh instead.
#
# Usage: bash create_env.sh [python3.9]
set -euo pipefail
PYTHON_BIN="${1:-python3.9}"
ENV_NAME=".venv_ml2dac_wsl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ENV_PATH="${REPO_ROOT}/${ENV_NAME}"

echo "Repo root : ${REPO_ROOT}"
echo "Env path  : ${ENV_PATH}"
echo "Python    : ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] '${PYTHON_BIN}' not found. Install python3.9 (deadsnakes PPA) or use create_env_wsl_conda.sh." >&2
  exit 1
fi
[ -d "${ENV_PATH}" ] || "${PYTHON_BIN}" -m venv "${ENV_PATH}"
"${ENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_PATH}/bin/python" --version
echo "[done] ${ENV_NAME} ready. Next: bash install_ml2dac_requirements.sh (with it activated)."
