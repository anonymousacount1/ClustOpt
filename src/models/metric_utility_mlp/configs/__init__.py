"""
configs
=======

Ready-to-use ``RunConfig`` JSON files for the metric-utility MLP loss-weight
experiments:

  * ``mlp_baseline_combined.json``  - reproduces the previous baseline run.
  * ``mlp_topk_balanced.json``      - more ranking / Top-K emphasis.
  * ``mlp_topk_aggressive.json``    - strongest Top-K optimisation.

Regenerate them with ``python -m models.metric_utility_mlp.configs.create_default_configs``.
Load one via ``--config <path>`` on ``run_train_cv`` (CLI flags still override).
"""

from __future__ import annotations

__all__ = ["CONFIG_FILES"]

CONFIG_FILES = [
    "mlp_baseline_combined.json",
    "mlp_topk_balanced.json",
    "mlp_topk_aggressive.json",
]
