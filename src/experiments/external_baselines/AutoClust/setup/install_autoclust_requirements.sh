#!/usr/bin/env bash
# AutoClust needs NO additional packages beyond the ML2DAC stack (it imports the
# same SMAC / ConfigSpace / CVI / ClusteringCS / MetaFeatureExtractor modules).
# Install the ML2DAC requirements instead (if not already done):
echo "AutoClust reuses the ml2dac env -- no extra requirements."
echo "If the ml2dac env is missing, run:"
echo "  bash experiments/external_baselines/ML2DAC/setup/install_ml2dac_requirements.sh"
echo "Then verify:"
echo "  ~/miniconda3/envs/ml2dac/bin/python experiments/external_baselines/AutoClust/setup/check_autoclust_install.py --repo-root \$(pwd)"
