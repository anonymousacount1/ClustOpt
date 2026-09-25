"""Aggregation-side registry of the (phase-tagged) experiment methods.

This is intentionally **decoupled** from
``experiment_execution.config.METHODS`` (which drives *execution* and still maps
to the original ClustOpt config sub-dirs). After the phase-tag rename migration,
the on-disk method folders carry ``_phase_A`` / ``_phase_AB`` / ``_phase_B``
suffixes, and the comparison now also includes the **external** AutoClustering
baselines (AutoClust / AutoML4Clust / ML2DAC) which use a *different* on-disk
record-folder layout (``1d_x/1d_y/2d``) and a *different* output schema
(``result.json`` / ``summary_metrics.json`` rather than the ClustOpt per-run
files). This registry captures, per method, everything the collector needs:

  * ``name``   — the on-disk folder name (phase-tagged),
  * ``group``  — ``regressor`` | ``baseline`` | ``external``,
  * ``schema`` — ``clustopt`` | ``external`` (which output files to read),
  * the mapping canonical view id -> on-disk record-folder name.

Canonical view ids stay ``x_only/y_only/xy_2d`` everywhere downstream so the
best-view selection and all view-grouped analyses are uniform across methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Canonical view ids (analysis space) -- unchanged across all methods.
VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")

# On-disk record-folder names, keyed by canonical view id.
CLUSTOPT_VIEW_DIRS: Dict[str, str] = {"x_only": "x_only", "y_only": "y_only", "xy_2d": "xy_2d"}
EXTERNAL_VIEW_DIRS: Dict[str, str] = {"x_only": "1d_x", "y_only": "1d_y", "xy_2d": "2d"}


@dataclass(frozen=True)
class AggMethod:
    name: str       # on-disk folder name (phase-tagged)
    group: str      # 'regressor' | 'baseline' | 'external'
    schema: str     # 'clustopt' | 'external'

    @property
    def view_dirs(self) -> Dict[str, str]:
        return EXTERNAL_VIEW_DIRS if self.schema == "external" else CLUSTOPT_VIEW_DIRS


METHODS: Tuple[AggMethod, ...] = (
    # Phase A+B: learned dynamic metric-selection (regressor) methods.
    AggMethod("regressor_top1_softmax_t05_phase_AB", "regressor", "clustopt"),
    AggMethod("regressor_top3_softmax_t05_phase_AB", "regressor", "clustopt"),
    AggMethod("regressor_top5_softmax_t05_phase_AB", "regressor", "clustopt"),
    AggMethod("regressor_top10_softmax_t05_phase_AB", "regressor", "clustopt"),
    # Phase A: fixed-CVI / uniform baselines (ClustOpt search, fixed objective).
    AggMethod("silhouette_single_phase_A", "baseline", "clustopt"),
    AggMethod("calinski_harabasz_single_phase_A", "baseline", "clustopt"),
    AggMethod("davies_bouldin_single_phase_A", "baseline", "clustopt"),
    AggMethod("dbcv_single_phase_A", "baseline", "clustopt"),
    AggMethod("uniform_classic_cvi_phase_A", "baseline", "clustopt"),
    AggMethod("all_metrics_uniform_phase_A", "baseline", "clustopt"),
    # Phase B: external AutoClustering baselines.
    AggMethod("AutoClust_phase_B", "external", "external"),
    AggMethod("AutoML4Clust_phase_B", "external", "external"),
    AggMethod("ML2DAC_phase_B", "external", "external"),
    # Phase C: re-run regressors (untagged, split-1 holdout model) vs the
    # in-domain external baselines (AutoClust / ML2DAC, Original + Extended).
    # These on-disk folder names are globally unique, so they intentionally have
    # NO DISPLAY_NAME entry (display == on-disk name); this keeps Phase A/B's
    # ``regressor_top{n}`` display->on-disk resolution untouched and reproducible.
    AggMethod("regressor_top1_softmax_t05", "regressor", "clustopt"),
    AggMethod("regressor_top3_softmax_t05", "regressor", "clustopt"),
    AggMethod("regressor_top5_softmax_t05", "regressor", "clustopt"),
    AggMethod("regressor_top10_softmax_t05", "regressor", "clustopt"),
    AggMethod("AutoClust_Original_InDomain", "external", "external"),
    AggMethod("AutoClust_Extended_InDomain", "external", "external"),
    AggMethod("ML2DAC_Original_InDomain", "external", "external"),
    AggMethod("ML2DAC_Extended_InDomain", "external", "external"),
    # ------------------------------------------------------------------- #
    # Phase D: additive ClustOpt ablation (28 methods, 50 Optuna trials, no
    # early stopping), all merged into split_01 with globally-unique untagged
    # folder names. Like Phase C, these carry NO DISPLAY_NAME entry (display ==
    # on-disk name), so Phase A/B/C reproduce byte-identically. The 4 untagged
    # ``regressor_top{1,3,5,10}_softmax_t05`` are ALREADY registered above
    # (Phase C) -- their on-disk content is now the Phase-D run after the merge,
    # so they are intentionally not re-listed here.
    #   * regressor-utility (learned MLP): group 'regressor'
    #   * real/oracle-utility + fixed-CVI: group 'baseline' (non-regressor)
    # Regressor-utility: raw (normalized-positive) variants of the Top-K sweep.
    AggMethod("regressor_top1_raw", "regressor", "clustopt"),
    AggMethod("regressor_top3_raw", "regressor", "clustopt"),
    AggMethod("regressor_top5_raw", "regressor", "clustopt"),
    AggMethod("regressor_top10_raw", "regressor", "clustopt"),
    # Regressor-utility: dynamic Top-K (predictK sizes K from predicted utility;
    # realK ranks by predicted but sizes K from real utility), raw + softmax.
    AggMethod("regressor_top_dynamic_predictK_raw", "regressor", "clustopt"),
    AggMethod("regressor_top_dynamic_predictK_softmax_t05", "regressor", "clustopt"),
    AggMethod("regressor_top_dynamic_realK_raw", "regressor", "clustopt"),
    AggMethod("regressor_top_dynamic_realK_softmax_t05", "regressor", "clustopt"),
    # Real/oracle-utility: fixed Top-K sweep, raw + softmax.
    AggMethod("real_top1_raw", "baseline", "clustopt"),
    AggMethod("real_top1_softmax_t05", "baseline", "clustopt"),
    AggMethod("real_top3_raw", "baseline", "clustopt"),
    AggMethod("real_top3_softmax_t05", "baseline", "clustopt"),
    AggMethod("real_top5_raw", "baseline", "clustopt"),
    AggMethod("real_top5_softmax_t05", "baseline", "clustopt"),
    AggMethod("real_top10_raw", "baseline", "clustopt"),
    AggMethod("real_top10_softmax_t05", "baseline", "clustopt"),
    # Real/oracle-utility: dynamic Top-K, raw + softmax.
    AggMethod("real_top_dynamic_raw", "baseline", "clustopt"),
    AggMethod("real_top_dynamic_softmax_t05", "baseline", "clustopt"),
    # Fixed-CVI / uniform baselines (untagged Phase-D re-run, 50 trials).
    AggMethod("silhouette_single", "baseline", "clustopt"),
    AggMethod("calinski_harabasz_single", "baseline", "clustopt"),
    AggMethod("davies_bouldin_single", "baseline", "clustopt"),
    AggMethod("dbcv_single", "baseline", "clustopt"),
    AggMethod("uniform_classic_cvi", "baseline", "clustopt"),
    AggMethod("all_metrics_uniform", "baseline", "clustopt"),
    # ------------------------------------------------------------------- #
    # Stage 1: controlled 2x3 ablation (metric inventory x utility strategy),
    # additive and globally unique, so every phase above reproduces unchanged.
    # These are the only methods run under the fixed-50 protocol (no global
    # Optuna timeout) with a deterministic per-(dataset, view) sampler seed;
    # historical methods were wall-clock limited and unseeded, so Stage-1 arms
    # must never be pooled with them as matched observations.
    #   * uniform / predicted arms: deployable
    #   * oracle arms: NON-OPERATIONAL (consume the ground-truth utility vector)
    AggMethod("stage1_head14_uniform_fixed50", "baseline", "clustopt"),
    AggMethod("stage1_head14_mlp_top5_raw_fixed50", "regressor", "clustopt"),
    AggMethod("stage1_head14_oracle_top5_raw_fixed50", "baseline", "clustopt"),
    AggMethod("stage1_full60_uniform_fixed50", "baseline", "clustopt"),
    AggMethod("stage1_full60_mlp_top5_raw_fixed50", "regressor", "clustopt"),
    AggMethod("stage1_full60_oracle_top5_raw_fixed50", "baseline", "clustopt"),
)

# Stage-1 arm metadata for the 2x3 analysis: (metric_inventory, utility_strategy,
# deployable). Kept beside the registry so the aggregate layer never has to
# parse semantics out of method names.
STAGE1_2X3_ARMS: Dict[str, Tuple[str, str, bool]] = {
    "stage1_head14_uniform_fixed50": ("head14", "uniform", True),
    "stage1_head14_mlp_top5_raw_fixed50": ("head14", "predicted", True),
    "stage1_head14_oracle_top5_raw_fixed50": ("head14", "oracle", False),
    "stage1_full60_uniform_fixed50": ("full60", "uniform", True),
    "stage1_full60_mlp_top5_raw_fixed50": ("full60", "predicted", True),
    "stage1_full60_oracle_top5_raw_fixed50": ("full60", "oracle", False),
}

METHOD_BY_NAME: Dict[str, AggMethod] = {m.name: m for m in METHODS}
METHOD_NAMES: Tuple[str, ...] = tuple(m.name for m in METHODS)

REGRESSOR_METHODS: Tuple[str, ...] = tuple(m.name for m in METHODS if m.group == "regressor")
BASELINE_METHODS: Tuple[str, ...] = tuple(m.name for m in METHODS if m.group == "baseline")
EXTERNAL_METHODS: Tuple[str, ...] = tuple(m.name for m in METHODS if m.group == "external")

# --------------------------------------------------------------------------- #
# Display names. The on-disk folders stay phase-tagged, but the aggregation /
# packaging *outputs* show trimmed names: drop the ``_phase_*`` suffix from
# every method, and collapse the regressors to just ``regressor_top{n}``.
# --------------------------------------------------------------------------- #
DISPLAY_NAME: Dict[str, str] = {
    "regressor_top1_softmax_t05_phase_AB": "regressor_top1",
    "regressor_top3_softmax_t05_phase_AB": "regressor_top3",
    "regressor_top5_softmax_t05_phase_AB": "regressor_top5",
    "regressor_top10_softmax_t05_phase_AB": "regressor_top10",
    "silhouette_single_phase_A": "silhouette_single",
    "calinski_harabasz_single_phase_A": "calinski_harabasz_single",
    "davies_bouldin_single_phase_A": "davies_bouldin_single",
    "dbcv_single_phase_A": "dbcv_single",
    "uniform_classic_cvi_phase_A": "uniform_classic_cvi",
    "all_metrics_uniform_phase_A": "all_metrics_uniform",
    "AutoClust_phase_B": "AutoClust",
    "AutoML4Clust_phase_B": "AutoML4Clust",
    "ML2DAC_phase_B": "ML2DAC",
}
NAME_BY_DISPLAY: Dict[str, str] = {v: k for k, v in DISPLAY_NAME.items()}


def display_name(on_disk_name: str) -> str:
    """On-disk (phase-tagged) folder name -> trimmed display name."""
    return DISPLAY_NAME.get(on_disk_name, on_disk_name)


def ondisk_name(display: str) -> str:
    """Trimmed display name -> on-disk folder name (for path resolution)."""
    return NAME_BY_DISPLAY.get(display, display)


# Top-K ordering for the regressor sweep comparison (DISPLAY names, since every
# analysis frame carries the trimmed method_name).
TOPK_ORDER: Dict[str, int] = {
    "regressor_top1": 1,
    "regressor_top3": 3,
    "regressor_top5": 5,
    "regressor_top10": 10,
    # Phase C regressors carry their full (untrimmed) display name.
    "regressor_top1_softmax_t05": 1,
    "regressor_top3_softmax_t05": 3,
    "regressor_top5_softmax_t05": 5,
    "regressor_top10_softmax_t05": 10,
}


def is_regressor(method_name: str) -> bool:
    """True for the learned dynamic metric-selection methods (phase-tag safe)."""
    return str(method_name).startswith("regressor")


def select_methods(names) -> Tuple[AggMethod, ...]:
    """Return the registry methods whose name is in ``names`` (preserving order).

    ``names`` of ``None`` selects all registered methods. Unknown names are
    ignored (the collector simply finds no folders for them).
    """
    if names is None:
        return METHODS
    wanted = list(names)
    return tuple(METHOD_BY_NAME[n] for n in wanted if n in METHOD_BY_NAME)
