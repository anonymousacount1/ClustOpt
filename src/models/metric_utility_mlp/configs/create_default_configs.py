"""
create_default_configs.py
=========================

Regenerate the loss-weight experiment configs from the :class:`RunConfig`
dataclasses, so the JSON files always match the current config schema.

Each experiment changes ONLY the loss weights (and ``run_name``); every other
field stays at the current :class:`RunConfig` defaults.

Usage
-----
    python -m models.metric_utility_mlp.configs.create_default_configs

This script only writes JSON files - it does NOT train anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..config import RunConfig

# Default Windows paths (the user overrides these from the CLI in Colab/Linux).
DEFAULT_CSV_PATH = (
    r"results_analysis"
    r"\clustering_repository\analyzed_data\unified_training_dataset"
    r"\unified_training_dataset.csv"
)
DEFAULT_REPO_ROOT = r"."

# Experiment name -> (run_name, loss-weight overrides).
EXPERIMENTS: Dict[str, Dict] = {
    "mlp_baseline_combined": {
        "run_name": "metric_utility_mlp_baseline_combined",
        "loss": {
            "mse_weight": 0.60,
            "pairwise_rank_weight": 0.25,
            "topk_mse_weight": 0.15,
            "topk": 10,
        },
    },
    "mlp_topk_balanced": {
        "run_name": "metric_utility_mlp_topk_balanced",
        "loss": {
            "mse_weight": 0.40,
            "pairwise_rank_weight": 0.30,
            "topk_mse_weight": 0.30,
            "topk": 10,
        },
    },
    "mlp_topk_aggressive": {
        "run_name": "metric_utility_mlp_topk_aggressive",
        "loss": {
            "mse_weight": 0.30,
            "pairwise_rank_weight": 0.30,
            "topk_mse_weight": 0.40,
            "topk": 10,
        },
    },
}


def build_run_config(run_name: str, loss_overrides: Dict) -> RunConfig:
    cfg = RunConfig()
    cfg.run_name = run_name
    cfg.repo_root = DEFAULT_REPO_ROOT
    cfg.data.csv_path = DEFAULT_CSV_PATH
    for key, value in loss_overrides.items():
        setattr(cfg.loss, key, value)
    return cfg


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for filename, spec in EXPERIMENTS.items():
        cfg = build_run_config(spec["run_name"], spec["loss"])
        path = out_dir / f"{filename}.json"
        cfg.to_json(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
