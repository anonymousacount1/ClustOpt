#!/usr/bin/env bash
# Create a dedicated, isolated virtual environment for AutoML4Clust (WSL/Linux).
#
# AutoML4Clust needs a legacy dependency stack that must NOT be installed into
# the main project environment. This creates a separate venv
# `.venv_automl4clust_wsl` at the repo root and upgrades pip tooling. It does
# NOT install the AutoML4Clust requirements -- run
# install_automl4clust_requirements.sh afterwards.
#
# Usage:
#   bash create_env.sh [python_interpreter]   # default: python3.6
set -euo pipefail

PYTHON_BIN="${1:-python3.6}"
ENV_NAME=".venv_automl4clust_wsl"

# Repo root: this script lives at
#   <repo>/experiments/external_baselines/AutoML4Clust/setup/create_env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ENV_PATH="${REPO_ROOT}/${ENV_NAME}"

echo "Repo root : ${REPO_ROOT}"
echo "Env path  : ${ENV_PATH}"
echo "Python    : ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] '${PYTHON_BIN}' not found. Install it (e.g. deadsnakes PPA) or pass another interpreter." >&2
  exit 1
fi

if [ -d "${ENV_PATH}" ]; then
  echo "[info] ${ENV_NAME} already exists. Reusing it (delete it to recreate)."
else
  echo "[step] Creating virtual environment..."
  "${PYTHON_BIN}" -m venv "${ENV_PATH}"
fi

VENV_PY="${ENV_PATH}/bin/python"
echo "[step] Upgrading pip / setuptools / wheel..."
"${VENV_PY}" -m pip install --upgrade "pip<24" "setuptools<60" "wheel"

"${VENV_PY}" --version
echo ""
echo "[done] Dedicated AutoML4Clust env ready at ${ENV_PATH}"
echo "Next:"
echo "  1) Activate : source ${ENV_NAME}/bin/activate"
echo "  2) Install  : bash experiments/external_baselines/AutoML4Clust/setup/install_automl4clust_requirements.sh"
echo "  3) Verify   : python -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust --repo-root ${REPO_ROOT}"
