"""
config.py
=========

Dataclass-based configuration for the metric-utility MLP pipeline.

All configuration is grouped into small dataclasses that can be serialised to
and from plain ``dict`` / JSON.  ``RunConfig`` is the top-level object that the
CLI builds and that every other module consumes.

The defaults encode the values from the implementation plan, so a bare
``RunConfig()`` is already a valid, sensible configuration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# The 19 identifier / metadata columns of the unified training dataset.
# These are *not* used as model inputs.  Detection falls back to "first
# n_id_columns" if any of these names is missing (see data_loader.py).
DEFAULT_ID_COLUMNS: List[str] = [
    "dataset_id",
    "record_id",
    "view_mode",
    "view_dim",
    "source_dataset_dir",
    "family_id",
    "family_name",
    "subfamily_id",
    "difficulty",
    "cluster_count_from_generator",
    "dataset_seed",
    "family_seed",
    "n_points",
    "feature_names_original",
    "selected_features",
    "has_labels",
    "has_rendering",
    "has_structural_ground_truth",
    "extraction_status",
]

# Logical grouping name -> actual column name in the CSV.  Used for grouped
# (per-family / per-view / ...) evaluation.  Columns that are absent are simply
# skipped at evaluation time.
DEFAULT_GROUP_COLUMNS: Dict[str, str] = {
    "family": "family_name",
    "subfamily": "subfamily_id",
    "view_id": "view_mode",
    "view_dim": "view_dim",
    "difficulty": "difficulty",
    "cluster_count": "cluster_count_from_generator",
}


@dataclass
class DataConfig:
    csv_path: str = (
        r"results_analysis"
        r"\clustering_repository\analyzed_data\unified_training_dataset"
        r"\unified_training_dataset.csv"
    )
    n_id_columns: int = 19
    n_feature_columns: int = 250
    n_target_columns: int = 60
    target_prefix: str = "utility__"
    # Optional explicit target subset, given as bare metric names (no prefix) in
    # the exact order the model should emit them.  ``None`` keeps the historical
    # behaviour: every ``target_prefix`` column, in CSV order.  When set, only
    # these targets are used and ``n_target_columns`` is validated against the
    # length of the list (see data_loader.detect_columns).
    target_include: Optional[List[str]] = None
    split_group_column: str = "dataset_id"
    id_columns: List[str] = field(default_factory=lambda: list(DEFAULT_ID_COLUMNS))
    group_columns: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_GROUP_COLUMNS)
    )
    # When True, mismatching column counts raise; when False they only warn.
    strict_column_counts: bool = True
    limit_rows: Optional[int] = None


@dataclass
class ModelConfig:
    type: str = "multi_head_mlp"  # "multi_head_mlp" | "single_head_mlp"
    input_dim: int = 250
    output_dim: int = 60
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256, 256])
    dropout: List[float] = field(default_factory=lambda: [0.15, 0.15, 0.10, 0.10])
    activation: str = "gelu"
    batch_norm: bool = True
    output_activation: str = "sigmoid"
    # Per-head architecture (multi-head model only).
    head_hidden_dim: int = 128
    head_dropout: float = 0.05


@dataclass
class LossConfig:
    mode: str = "combined"  # "combined" | "mse_only"
    mse_weight: float = 0.60
    pairwise_rank_weight: float = 0.25
    topk_mse_weight: float = 0.15
    topk: int = 10
    pairwise_margin: float = 0.05
    max_pairs_per_sample: int = 256
    tie_epsilon: float = 1e-4
    topk_extra_weight: float = 2.0  # weight = 1 + topk_extra_weight * is_true_topk


@dataclass
class TrainingConfig:
    n_splits: int = 5
    inner_val_ratio: float = 0.15
    batch_size: int = 512
    epochs: int = 300
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    scheduler_min_lr: float = 1e-6
    early_stopping_patience: int = 25
    gradient_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    num_workers: int = 0
    folds_to_run: Optional[List[int]] = None  # None -> all folds


@dataclass
class EvaluationConfig:
    topk_list: List[int] = field(default_factory=lambda: [3, 5, 10])
    ndcg_list: List[int] = field(default_factory=lambda: [3, 5, 10])
    ndcg_all: bool = True


@dataclass
class RunConfig:
    run_name: str = "metric_utility_mlp_v1"
    repo_root: str = r"."
    output_subdir: str = "results_analysis/mlp"
    debug: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ---- serialisation helpers -------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        data = dict(data)
        sub = {
            "data": DataConfig,
            "model": ModelConfig,
            "loss": LossConfig,
            "training": TrainingConfig,
            "evaluation": EvaluationConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, klass in sub.items():
            if key in data and data[key] is not None:
                kwargs[key] = klass(**data.pop(key))
        kwargs.update(data)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
