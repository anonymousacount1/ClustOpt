"""
run_train_cv.py
===============

Command-line entry point for the metric-utility MLP cross-validation pipeline.

Example
-------
    python -m models.metric_utility_mlp.run_train_cv \\
      --csv-path "...\\unified_training_dataset.csv" \\
      --repo-root "<repo-root>" \\
      --run-name "metric_utility_mlp_v1" \\
      --n-splits 5 --batch-size 512 --epochs 300 \\
      --lr 0.001 --weight-decay 0.0001 --patience 25 --device auto

Debug helpers
-------------
    --debug              shorter run (few epochs, row cap) for smoke testing
    --limit-rows N       only read the first N CSV rows
    --folds-to-run 0 1   only train the listed fold indices
    --model-type ...     multi_head_mlp | single_head_mlp
    --loss-mode ...      combined | mse_only

This script never auto-launches a full run beyond the parsed arguments; the
caller controls epochs / folds explicitly.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import RunConfig
from .cross_validation import run_cross_validation


def build_config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig()
    if args.config:
        cfg = RunConfig.from_json(args.config)

    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.repo_root is not None:
        cfg.repo_root = args.repo_root
    if args.csv_path is not None:
        cfg.data.csv_path = args.csv_path
    if args.limit_rows is not None:
        cfg.data.limit_rows = args.limit_rows
    if args.no_strict_columns:
        cfg.data.strict_column_counts = False

    if args.n_splits is not None:
        cfg.training.n_splits = args.n_splits
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    if args.patience is not None:
        cfg.training.early_stopping_patience = args.patience
    if args.device is not None:
        cfg.training.device = args.device
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.folds_to_run is not None:
        cfg.training.folds_to_run = args.folds_to_run

    if args.model_type is not None:
        cfg.model.type = args.model_type
    if args.loss_mode is not None:
        cfg.loss.mode = args.loss_mode

    # Loss-weight overrides (only applied when explicitly provided).
    if args.mse_weight is not None:
        cfg.loss.mse_weight = args.mse_weight
    if args.pairwise_rank_weight is not None:
        cfg.loss.pairwise_rank_weight = args.pairwise_rank_weight
    if args.topk_mse_weight is not None:
        cfg.loss.topk_mse_weight = args.topk_mse_weight
    if args.topk is not None:
        cfg.loss.topk = args.topk
    if args.pairwise_margin is not None:
        cfg.loss.pairwise_margin = args.pairwise_margin
    if args.max_pairs_per_sample is not None:
        cfg.loss.max_pairs_per_sample = args.max_pairs_per_sample
    if args.tie_epsilon is not None:
        cfg.loss.tie_epsilon = args.tie_epsilon
    if args.topk_extra_weight is not None:
        cfg.loss.topk_extra_weight = args.topk_extra_weight

    if args.debug:
        cfg.debug = True
        # Lightweight smoke-test defaults (only override what the user didn't set).
        if args.epochs is None:
            cfg.training.epochs = 3
        if args.patience is None:
            cfg.training.early_stopping_patience = 3
        if args.limit_rows is None:
            cfg.data.limit_rows = 3000
        if args.folds_to_run is None:
            cfg.training.folds_to_run = [0]
        if args.run_name is None:
            cfg.run_name = cfg.run_name + "_debug"

    return cfg


def register_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every shared CLI flag on ``p``.

    Shared by both entry points (``run_train_cv`` and the predefined-split
    holdout runner) so they accept an identical set of training / model / loss
    options and can both be consumed by :func:`build_config_from_args`.
    """
    p.add_argument("--config", type=str, default=None, help="Optional JSON RunConfig to load.")
    p.add_argument("--csv-path", dest="csv_path", type=str, default=None)
    p.add_argument("--repo-root", dest="repo_root", type=str, default=None)
    p.add_argument("--run-name", dest="run_name", type=str, default=None)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--folds-to-run", dest="folds_to_run", type=int, nargs="+", default=None)
    p.add_argument("--model-type", dest="model_type", type=str, default=None,
                   choices=["multi_head_mlp", "single_head_mlp"])
    p.add_argument("--loss-mode", dest="loss_mode", type=str, default=None,
                   choices=["combined", "mse_only"])

    # Loss-weight overrides (None -> keep config/default value).
    loss_group = p.add_argument_group("loss weights (override config/defaults)")
    loss_group.add_argument("--mse-weight", dest="mse_weight", type=float, default=None)
    loss_group.add_argument("--pairwise-rank-weight", dest="pairwise_rank_weight", type=float, default=None)
    loss_group.add_argument("--topk-mse-weight", dest="topk_mse_weight", type=float, default=None)
    loss_group.add_argument("--topk", dest="topk", type=int, default=None)
    loss_group.add_argument("--pairwise-margin", dest="pairwise_margin", type=float, default=None)
    loss_group.add_argument("--max-pairs-per-sample", dest="max_pairs_per_sample", type=int, default=None)
    loss_group.add_argument("--tie-epsilon", dest="tie_epsilon", type=float, default=None)
    loss_group.add_argument("--topk-extra-weight", dest="topk_extra_weight", type=float, default=None)

    p.add_argument("--limit-rows", dest="limit_rows", type=int, default=None)
    p.add_argument("--no-strict-columns", dest="no_strict_columns", action="store_true")
    p.add_argument("--debug", action="store_true", help="Short smoke-test run.")
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metric Utility MLP — 5-fold CV trainer.")
    register_common_args(p)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config_from_args(args)
    print("[cli] effective configuration:")
    print(f"      run_name={cfg.run_name} model={cfg.model.type} loss={cfg.loss.mode}")
    print(f"      n_splits={cfg.training.n_splits} epochs={cfg.training.epochs} "
          f"batch_size={cfg.training.batch_size} device={cfg.training.device} "
          f"folds_to_run={cfg.training.folds_to_run} limit_rows={cfg.data.limit_rows}")
    print(f"      loss weights: mse={cfg.loss.mse_weight} "
          f"pairwise_rank={cfg.loss.pairwise_rank_weight} "
          f"topk_mse={cfg.loss.topk_mse_weight} topk={cfg.loss.topk}")
    print(f"      loss params: pairwise_margin={cfg.loss.pairwise_margin} "
          f"max_pairs_per_sample={cfg.loss.max_pairs_per_sample} "
          f"tie_epsilon={cfg.loss.tie_epsilon} topk_extra_weight={cfg.loss.topk_extra_weight}")
    run_cross_validation(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
