#!/usr/bin/env bash
# Convenience launcher: run ML2DAC (full meta-learning config) on ONE subfamily
# for a given split, using the dedicated WSL conda env `ml2dac`. Resumable (no
# --overwrite). Console output should be redirected to a log by the caller.
#
# Usage:
#   bash run_subfamily_split.sh <family_id> <subfamily> [split_id] [workers]
# Example:
#   bash run_subfamily_split.sh arcs_circles_rings annuli_variable_thickness 1 8
set -uo pipefail

FAMILY="${1:?family_id required}"
SUB="${2:?subfamily required}"
SPLIT="${3:-1}"
WORKERS="${4:-8}"

REPO="${REPO:-.}"
PY="${PY:-$HOME/miniconda3/envs/ml2dac/bin/python}"
CONFIG="$REPO/experiments/external_baselines/ML2DAC/configs/ml2dac_full_meta_learning.json"
SUBROOT="$REPO/results_analysis/clustering_repository/analyzed_data/$FAMILY/subfamilies/$SUB"
SPLITCSV="$REPO/results_analysis/clustering_repository/analyzed_data/experiment_splits/dataset_split_assignments.csv"

echo "=== ML2DAC subfamily run ==="
echo "family=$FAMILY subfamily=$SUB split=$SPLIT workers=$WORKERS"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$REPO"
"$PY" -m experiments.external_baselines.ML2DAC.runners.run_ml2dac_subfamily_split \
  --subfamily-root "$SUBROOT" \
  --split-assignment-csv "$SPLITCSV" \
  --split-id "$SPLIT" \
  --config "$CONFIG" \
  --record-types 1d_x 1d_y 2d \
  --workers "$WORKERS" \
  --repo-root "$REPO"
RC=$?
echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=$RC"
exit $RC
