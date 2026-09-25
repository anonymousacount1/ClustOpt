"""
run_train_split_holdout.py
==========================

Command-line entry point that trains the metric-utility MLP on the project's
*predefined experiment splits* instead of the internal ``GroupKFold`` folds.

The outer test fold is fixed by the experiment split assignments
(``dataset_split_assignments.csv``): all datasets in ``--test-split-id`` become
the held-out test set, and the datasets in ``--train-split-ids`` (default: every
other split) form the train+validation pool.  The inner train/validation
division is made **exactly as in the standard runner** — the same group-safe
``GroupShuffleSplit`` with the same ``inner_val_ratio`` (0.15) and ``seed`` (42).
Everything downstream (preprocessing, model, loss, training loop, evaluation,
reports, plots, artifacts) is the unchanged pipeline from ``run_cross_validation``.

The default configuration reproduces the request: **train on splits 2–16, test
on split 1.**

Example (Windows PowerShell)
----------------------------
    python -u -m models.metric_utility_mlp.run_train_split_holdout `
      --test-split-id 1 `
      --run-name "metric_utility_mlp_split1_holdout" `
      --device auto

Every flag accepted by ``run_train_cv`` is also accepted here (``--config``,
``--epochs``, ``--batch-size``, ``--lr``, loss-weight overrides, ``--debug`` ...).
Split-specific flags:

    --split-assignments-csv PATH   dataset_id -> split_id mapping CSV
    --test-split-id N              split held out for testing (default 1)
    --train-split-ids N [N ...]    splits used for train+val (default: all others)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import RunConfig
from .cross_validation import run_cross_validation
from .run_train_cv import build_config_from_args, register_common_args
from .head_groups import resolve_target_group
from .splitters import build_holdout_split_folds

# Default location of the experiment split assignments (dataset_id -> split_id).
DEFAULT_SPLIT_ASSIGNMENTS_CSV = (
    r"results_analysis"
    r"\clustering_repository\analyzed_data\experiment_splits"
    r"\dataset_split_assignments.csv"
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Metric Utility MLP — predefined-split holdout trainer "
        "(train on splits 2-16, test on split 1 by default)."
    )
    register_common_args(p)
    split_group = p.add_argument_group("predefined split holdout")
    split_group.add_argument(
        "--split-assignments-csv", dest="split_assignments_csv", type=str,
        default=DEFAULT_SPLIT_ASSIGNMENTS_CSV,
        help="CSV mapping dataset_id -> split_id (default: experiment_splits file).",
    )
    split_group.add_argument(
        "--test-split-id", dest="test_split_id", type=int, default=1,
        help="Split id held out for testing (default: 1).",
    )
    split_group.add_argument(
        "--train-split-ids", dest="train_split_ids", type=int, nargs="+", default=None,
        help="Split ids used for train+val (default: every split except the test split).",
    )
    target_group = p.add_argument_group("target subset")
    target_group.add_argument(
        "--target-include-group", dest="target_include_group", type=str, default=None,
        help="Restrict the model's outputs to a named target group resolved via "
             "head_groups.resolve_target_group (e.g. 'head14' -> the 14 classic_cvi "
             "metrics). Default: all 60 utility targets.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg: RunConfig = build_config_from_args(args)

    # Optional target-subset restriction (e.g. the Head-14 ablation arm).  The
    # resolved metric list -- not the alias -- is written into the config, so the
    # saved artifact is self-describing and replayable without this CLI.
    target_group_label = None
    if args.target_include_group is not None:
        target_group_label = args.target_include_group
        metrics = resolve_target_group(args.target_include_group)
        cfg.data.target_include = list(metrics)
        cfg.data.n_target_columns = len(metrics)
        cfg.model.output_dim = len(metrics)

    # Sensible default run name when the user did not provide one.
    if args.run_name is None:
        suffix = f"_{target_group_label}" if target_group_label else ""
        cfg.run_name = (
            f"metric_utility_mlp{suffix}_split{args.test_split_id}_holdout"
        )
        if cfg.debug:
            cfg.run_name += "_debug"

    # Build the predefined-split holdout fold; the inner train/val division uses
    # the SAME settings as the standard runner (cfg.training.inner_val_ratio/seed).
    def fold_builder(df, group_column, group_summary_columns):
        return build_holdout_split_folds(
            df=df,
            group_column=group_column,
            split_assignments_csv=args.split_assignments_csv,
            test_split_id=args.test_split_id,
            train_split_ids=args.train_split_ids,
            inner_val_ratio=cfg.training.inner_val_ratio,
            seed=cfg.training.seed,
            group_summary_columns=group_summary_columns,
        )

    print("[cli] effective configuration (predefined-split holdout):")
    print(f"      run_name={cfg.run_name} model={cfg.model.type} loss={cfg.loss.mode}")
    print(f"      test_split_id={args.test_split_id} "
          f"train_split_ids={args.train_split_ids if args.train_split_ids is not None else 'ALL-OTHERS'}")
    print(f"      split_assignments_csv={args.split_assignments_csv}")
    if cfg.data.target_include is not None:
        print(f"      target_group={target_group_label} "
              f"n_targets={len(cfg.data.target_include)} "
              f"targets={cfg.data.target_include}")
    print(f"      inner_val_ratio={cfg.training.inner_val_ratio} seed={cfg.training.seed} "
          f"epochs={cfg.training.epochs} batch_size={cfg.training.batch_size} "
          f"device={cfg.training.device} limit_rows={cfg.data.limit_rows}")
    print(f"      loss weights: mse={cfg.loss.mse_weight} "
          f"pairwise_rank={cfg.loss.pairwise_rank_weight} "
          f"topk_mse={cfg.loss.topk_mse_weight} topk={cfg.loss.topk}")

    run_cross_validation(cfg, fold_builder=fold_builder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
