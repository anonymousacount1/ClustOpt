"""Configuration dataclasses, the method registry, and constants."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Views, in fixed execution order.
VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")


@dataclass(frozen=True)
class MethodSpec:
    """One experiment method: maps to a config folder + group."""

    name: str
    group: str          # 'regressor' | 'baseline'
    config_subdir: str  # relative to models/ClustOpt/configs/experiments/


# The 10 methods. ``config_subdir`` is relative to the experiments config root.
METHODS: Tuple[MethodSpec, ...] = (
    MethodSpec("regressor_top1_softmax_t05", "regressor",
               "regressor_metric_selection/regressor_top1_softmax_t05"),
    MethodSpec("regressor_top3_softmax_t05", "regressor",
               "regressor_metric_selection/regressor_top3_softmax_t05"),
    MethodSpec("regressor_top5_softmax_t05", "regressor",
               "regressor_metric_selection/regressor_top5_softmax_t05"),
    MethodSpec("regressor_top10_softmax_t05", "regressor",
               "regressor_metric_selection/regressor_top10_softmax_t05"),
    MethodSpec("silhouette_single", "baseline", "baselines/silhouette_single"),
    MethodSpec("calinski_harabasz_single", "baseline", "baselines/calinski_harabasz_single"),
    MethodSpec("davies_bouldin_single", "baseline", "baselines/davies_bouldin_single"),
    MethodSpec("dbcv_single", "baseline", "baselines/dbcv_single"),
    MethodSpec("uniform_classic_cvi", "baseline", "baselines/uniform_classic_cvi"),
    MethodSpec("all_metrics_uniform", "baseline", "baselines/all_metrics_uniform"),
)

METHOD_NAMES: Tuple[str, ...] = tuple(m.name for m in METHODS)


def _phased_spec(name: str, group: str) -> MethodSpec:
    """Phase-D methods all resolve to ``configs/experiments/phaseD/<name>/``."""
    return MethodSpec(name, group, f"phaseD/{name}")


# Phase-D ablation: 28 methods (additive; the original 10 ``METHODS`` are
# untouched). 12 regressor-utility, 10 real/oracle-utility, 6 fixed baselines.
# Names + groups mirror generate_phaseD_configs.py (validated in the smoke test).
_PHASED_REGRESSOR = (
    "regressor_top1_raw", "regressor_top3_raw", "regressor_top5_raw",
    "regressor_top10_raw", "regressor_top_dynamic_realK_raw",
    "regressor_top_dynamic_predictK_raw",
    "regressor_top1_softmax_t05", "regressor_top3_softmax_t05",
    "regressor_top5_softmax_t05", "regressor_top10_softmax_t05",
    "regressor_top_dynamic_realK_softmax_t05",
    "regressor_top_dynamic_predictK_softmax_t05",
)
_PHASED_REAL = (
    "real_top1_raw", "real_top3_raw", "real_top5_raw", "real_top10_raw",
    "real_top_dynamic_raw",
    "real_top1_softmax_t05", "real_top3_softmax_t05", "real_top5_softmax_t05",
    "real_top10_softmax_t05", "real_top_dynamic_softmax_t05",
)
_PHASED_FIXED = (
    "all_metrics_uniform", "calinski_harabasz_single", "davies_bouldin_single",
    "dbcv_single", "silhouette_single", "uniform_classic_cvi",
)

METHODS_PHASED: Tuple[MethodSpec, ...] = (
    *(_phased_spec(n, "regressor") for n in _PHASED_REGRESSOR),
    *(_phased_spec(n, "oracle") for n in _PHASED_REAL),
    *(_phased_spec(n, "baseline") for n in _PHASED_FIXED),
)
METHOD_NAMES_PHASED: Tuple[str, ...] = tuple(m.name for m in METHODS_PHASED)


def _phaseE2_knn_spec(name: str) -> MethodSpec:
    """Phase-E2 KNN methods resolve to ``configs/experiments/phaseE2_knn/<name>/``."""
    return MethodSpec(name, "knn", f"phaseE2_knn/{name}")


# Phase-E2 KNN ablation: exactly 12 methods (additive; original 10 + Phase-D 28
# are untouched). Same downstream ClustOpt as the regressor methods; only the
# utility source is the frozen Phase-E1 KNN. Names mirror
# generate_phaseE2_knn_configs.py (validated in the smoke test).
_PHASE_E2_KNN = (
    "knn_top1_raw", "knn_top3_raw", "knn_top5_raw", "knn_top10_raw",
    "knn_top1_softmax_t05", "knn_top3_softmax_t05", "knn_top5_softmax_t05",
    "knn_top10_softmax_t05",
    "knn_top_dynamic_realK_raw", "knn_top_dynamic_predictK_raw",
    "knn_top_dynamic_realK_softmax_t05", "knn_top_dynamic_predictK_softmax_t05",
)
METHODS_PHASE_E2_KNN: Tuple[MethodSpec, ...] = tuple(
    _phaseE2_knn_spec(n) for n in _PHASE_E2_KNN
)
METHOD_NAMES_PHASE_E2_KNN: Tuple[str, ...] = tuple(m.name for m in METHODS_PHASE_E2_KNN)

def _stage1_spec(name: str, group: str) -> MethodSpec:
    """Stage-1 methods resolve to ``configs/experiments/stage1_2x3/<name>/``."""
    return MethodSpec(name, group, f"stage1_2x3/{name}")


# Stage-1 controlled 2x3 ablation: exactly 6 methods (additive; the original 10,
# Phase-D 28 and Phase-E2 12 are untouched). Rows = metric inventory
# {Head-14, Full-60}; columns = utility strategy {Uniform, Predicted, Oracle}.
# Unlike every historical method these run with NO global Optuna timeout, so the
# binding stop criterion is exactly n_trials=50, and with a deterministic
# per-(dataset, view) sampler seed shared by all six arms.
# Names mirror generate_stage1_2x3_configs.py.
#
# The two oracle arms use group 'oracle' -- the same value Phase-D uses for
# real_* -- because they consume the ground-truth utility vector and are
# therefore non-deployable.
_STAGE1_2X3 = (
    ("stage1_head14_uniform_fixed50", "baseline"),
    ("stage1_head14_mlp_top5_raw_fixed50", "regressor"),
    ("stage1_head14_oracle_top5_raw_fixed50", "oracle"),
    ("stage1_full60_uniform_fixed50", "baseline"),
    ("stage1_full60_mlp_top5_raw_fixed50", "regressor"),
    ("stage1_full60_oracle_top5_raw_fixed50", "oracle"),
)
METHODS_STAGE1_2X3: Tuple[MethodSpec, ...] = tuple(
    _stage1_spec(n, g) for n, g in _STAGE1_2X3
)
METHOD_NAMES_STAGE1_2X3: Tuple[str, ...] = tuple(m.name for m in METHODS_STAGE1_2X3)

# Selectable method sets. ``default`` preserves the original 10-method behaviour.
METHOD_SETS: Dict[str, Tuple[MethodSpec, ...]] = {
    "default": METHODS,
    "phaseD": METHODS_PHASED,
    "phaseE2_knn": METHODS_PHASE_E2_KNN,
    "stage1_2x3": METHODS_STAGE1_2X3,
}

# Per method/view output files (see plan §22).
PER_RUN_OUTPUT_FILES: Tuple[str, ...] = (
    "clustopt_results.csv",
    "best_result.json",
    "external_metrics.json",
    "selected_metrics.json",
    "timing_summary.json",
    "run_log.json",
    "status.json",
    "config_snapshot.json",
)


def _default_max_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(cpu - 1, 8))


@dataclass
class ExperimentExecutionConfig:
    """Resolved CLI options for one subfamily/split experiment run."""

    repo_root: Path
    subfamily_dir: Path
    split_assignments: Path
    split_id: int

    metric_mode: str = "fast"
    fast_metric_sample_size: int = 3000
    fast_metric_random_state: int = 42

    max_workers: int = field(default_factory=_default_max_workers)
    resume: bool = True
    overwrite: bool = False
    dry_run: bool = False

    limit_datasets: int | None = None
    # ``method_set`` selects which registry ``methods`` are drawn from. Defaults
    # to the original 10-method set so old runs are unaffected. When set to
    # ``phaseD`` and ``methods`` is left at the default, all 28 Phase-D methods run.
    method_set: str = "default"
    methods: Tuple[str, ...] = METHOD_NAMES
    views: Tuple[str, ...] = VIEW_IDS

    # Parallelism granularity. ``dataset`` (default, legacy) schedules one worker
    # per dataset (each runs its views x methods serially). ``record`` schedules
    # each individual (dataset, view, method) run as its own pool task so free
    # workers never idle waiting on a busy dataset; workers cache the loaded
    # dataset + built views per process to avoid redundant reloads.
    parallelism: str = "dataset"

    # Optional suffix appended to the per-dataset output split directory. When
    # set, outputs go to ``experiments/split_XX_<suffix>/`` (and the subfamily
    # summaries to ``experiment_execution_summaries/split_XX_<suffix>/``) instead
    # of the canonical ``split_XX``. Dataset *selection* still uses the numeric
    # ``split_id``; only the output folder name changes. This lets a re-run write
    # a fresh, isolated split folder without ever touching the original one.
    split_dir_suffix: str = ""

    progress_every: int = 1

    # ------------------------------------------------------------------ helpers
    def experiments_config_root(self) -> Path:
        return self.repo_root / "models" / "ClustOpt" / "configs" / "experiments"

    def split_dir_name(self) -> str:
        base = f"split_{int(self.split_id):02d}"
        return f"{base}_{self.split_dir_suffix}" if self.split_dir_suffix else base

    def runtime_block(self) -> Dict[str, object]:
        block: Dict[str, object] = {"metric_mode": self.metric_mode}
        if self.metric_mode == "fast":
            block["fast_metric_sample_size"] = int(self.fast_metric_sample_size)
            block["fast_metric_random_state"] = int(self.fast_metric_random_state)
        return block

    def active_registry(self) -> Tuple[MethodSpec, ...]:
        return METHOD_SETS.get(self.method_set, METHODS)

    def selected_methods(self) -> List[MethodSpec]:
        wanted = set(self.methods)
        return [m for m in self.active_registry() if m.name in wanted]

    def to_summary_dict(self) -> Dict[str, object]:
        return {
            "repo_root": str(self.repo_root),
            "subfamily_dir": str(self.subfamily_dir),
            "split_assignments": str(self.split_assignments),
            "split_id": int(self.split_id),
            "split_dir_suffix": self.split_dir_suffix,
            "split_dir_name": self.split_dir_name(),
            "metric_mode": self.metric_mode,
            "fast_metric_sample_size": int(self.fast_metric_sample_size),
            "fast_metric_random_state": int(self.fast_metric_random_state),
            "max_workers": int(self.max_workers),
            "resume": bool(self.resume),
            "overwrite": bool(self.overwrite),
            "dry_run": bool(self.dry_run),
            "limit_datasets": self.limit_datasets,
            "method_set": self.method_set,
            "methods": list(self.methods),
            "views": list(self.views),
            "parallelism": self.parallelism,
        }
