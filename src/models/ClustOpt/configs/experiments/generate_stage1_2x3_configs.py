"""Generate the Stage-1 controlled 2x3 ablation configs (6 arms x 3 views = 18).

The factorial is ``{Head-14, Full-60} x {Uniform, Predicted, Oracle}``.

This is an *additive* experiment set: it does not touch the existing
``regressor_metric_selection`` / ``baselines`` / ``phaseD`` / ``phaseE2_knn``
configs. All Stage-1 configs live under
``configs/experiments/stage1_2x3/<method_name>/<view>.json``.

Three controlled differences from every historical config:

* **no global Optuna timeout** -- the ``optuna`` block omits ``timeout``, so the
  binding stop criterion is exactly ``n_trials = 50``. Stage-0 showed the
  historical 150 s wall-clock cap truncated expensive objectives (60-CVI runs
  delivered ~24 trials vs ~50 for cheap ones), confounding any inventory
  comparison;
* **early stopping disabled** -- no ``patience`` key;
* **deterministic search seed** -- ``"seed": "auto"`` + ``"base_seed"`` derives a
  per-(dataset, view) seed via SHA-256 so all six arms start from identical
  sampler randomness for a given record. The arm id is deliberately not part of
  the derivation.

The clustering search space is reused verbatim from
``generate_experiment_configs.build_search_space``, so all six arms share an
identical algorithm/hyper-parameter space; only the objective differs.

Run from the repo root::

    python -m models.ClustOpt.configs.experiments.generate_stage1_2x3_configs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from models.metric_utility_mlp.head_groups import (
    METRIC_HEAD_GROUPS,
    resolve_target_group,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.metric_registry import (
    HEAD_GROUP_ALIASES,
)
from .generate_experiment_configs import build_search_space

HERE = Path(__file__).resolve().parent
STAGE1_ROOT = HERE / "stage1_2x3"
VIEWS = ("x_only", "y_only", "xy_2d")

PHASE_NAME = "ClustOpt_Stage1_2x3_Ablation"

# The canonical Full-60 predictor (immutable; reused unchanged for FP).
FULL60_REGRESSOR_PATH = (
    "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout"
)
# The Head-14 predictor does not exist until it is trained. Configs carry a
# stable *indirection* token instead of a fake timestamped path; the training
# step injects the real run directory (see inject_head14_regressor_path).
HEAD14_REGRESSOR_PLACEHOLDER = "<HEAD14_MLP_RUN_DIR>"

REAL_UTILITY_SOURCE = "utility/<view_id>/utility_vector.json"
WEIGHTING_RAW = "normalized_positive"      # RAW == clip-negatives + sum-to-1
TOP_K = 5
BASE_SEED = 42
GENERIC_PARAMS = {"remove_noise": True, "noise_label": -1, "min_clusters": 2}


def optuna_block_fixed50() -> Dict[str, Any]:
    """Exactly 50 trials: no timeout, no early-stop patience, seeded sampler."""
    return {
        "type": "optuna",
        "params": {
            "n_trials": 50,
            "seed": "auto",
            "base_seed": BASE_SEED,
        },
    }


# ---------------------------------------------------------------- inventories
def head14_metric_names() -> List[str]:
    """The 14 established CVIs, from the single canonical source."""
    return resolve_target_group("head14")


def full60_metric_names() -> List[str]:
    """All 60 CVIs, in canonical head-group order."""
    out: List[str] = []
    for metrics in METRIC_HEAD_GROUPS.values():
        out.extend(metrics)
    return out


def to_registry_keys(metric_names: List[str]) -> List[str]:
    """Map data-facing metric ``.name`` values to ClustOpt registry keys.

    ``head_groups`` and the utility vectors use each metric's ``.name``; the
    ClustOpt ``cvi.metrics`` block is keyed by *registry key*. Exactly two
    metrics differ between the namespaces, so the mapping must go through the
    repository's canonical alias table rather than assuming they coincide.

    ``HEAD_GROUP_ALIASES`` is used in preference to
    ``naming.build_name_to_key()`` because the latter imports the live
    ``METRIC_REGISTRY``, which pulls in the optional image/geometry stack
    (shapely, skimage, cv2). Config generation must work without those.
    Equivalence of the two sources is asserted by the Stage-1 tests whenever the
    heavy dependencies are importable.
    """
    return [HEAD_GROUP_ALIASES.get(m, m) for m in metric_names]


def uniform_metrics_block(metric_names: List[str]) -> Dict[str, float]:
    """Normalised uniform weights: every metric gets 1/N and the weights sum to 1.

    Deliberately NOT the historical convention where each metric receives 1.0:
    normalising keeps J on a comparable scale across HU (1/14) and FU (1/60) and
    against the Top-5 arms, whose ``normalized_positive`` weights also sum to 1.
    Because a constant rescaling is monotone, this cannot change the argmax.
    """
    keys = to_registry_keys(metric_names)
    w = 1.0 / len(keys)
    return {k: w for k in keys}


# ------------------------------------------------------------------ arm specs
def uniform_arm(method: str, inventory: str, metric_names: List[str]) -> Dict[str, Any]:
    return {
        "method": method,
        "inventory": inventory,
        "utility_strategy": "uniform",
        "group": "baseline",
        "deployable": True,
        "cvi": {
            "type": "generic",
            "metrics": uniform_metrics_block(metric_names),
            "params": dict(GENERIC_PARAMS),
        },
    }


def predicted_arm(method: str, inventory: str, model_run_dir: str) -> Dict[str, Any]:
    return {
        "method": method,
        "inventory": inventory,
        "utility_strategy": "predicted",
        "group": "regressor",
        "deployable": True,
        "cvi": {
            "type": "regressor_dynamic",
            "params": {
                "model_run_dir": model_run_dir,
                "checkpoint_policy": "fold_ensemble",
                "top_k": TOP_K,
                "weighting": WEIGHTING_RAW,
                "metric_name_prefix": "utility__",
                "feature_source": "features/features_records.csv",
                "view_aware": True,
                "fallback_on_error": True,
                "fallback_metrics": {"silhouette": 1.0},
            },
            "params_generic": dict(GENERIC_PARAMS),
        },
    }


def oracle_arm(method: str, inventory: str, metric_names: List[str] | None) -> Dict[str, Any]:
    """Oracle arm. ``metric_names=None`` => unrestricted Top-5 over all 60."""
    params: Dict[str, Any] = {
        "top_k": TOP_K,
        "weighting": WEIGHTING_RAW,
        "metric_name_prefix": "utility__",
        "utility_source": REAL_UTILITY_SOURCE,
        "fallback_on_error": True,
        "fallback_metrics": {"silhouette": 1.0},
    }
    if metric_names is not None:
        # Restrict the true-utility candidate pool BEFORE Top-K selection.
        # Uses metric .name spelling: the utility vector is keyed by .name.
        params["candidate_metrics"] = list(metric_names)
    return {
        "method": method,
        "inventory": inventory,
        "utility_strategy": "oracle",
        "group": "oracle",
        "deployable": False,
        "cvi": {
            "type": "oracle_dynamic",
            "params": params,
            "params_generic": dict(GENERIC_PARAMS),
        },
    }


def build_arm_specs() -> List[Dict[str, Any]]:
    head14 = head14_metric_names()
    full60 = full60_metric_names()
    if len(head14) != 14:
        raise ValueError(f"Head-14 resolved to {len(head14)} metrics, expected 14.")
    if len(full60) != 60:
        raise ValueError(f"Full-60 resolved to {len(full60)} metrics, expected 60.")
    return [
        uniform_arm("stage1_head14_uniform_fixed50", "head14", head14),
        predicted_arm("stage1_head14_mlp_top5_raw_fixed50", "head14",
                      HEAD14_REGRESSOR_PLACEHOLDER),
        oracle_arm("stage1_head14_oracle_top5_raw_fixed50", "head14", head14),
        uniform_arm("stage1_full60_uniform_fixed50", "full60", full60),
        predicted_arm("stage1_full60_mlp_top5_raw_fixed50", "full60",
                      FULL60_REGRESSOR_PATH),
        oracle_arm("stage1_full60_oracle_top5_raw_fixed50", "full60", None),
    ]


def build_config(spec: Dict[str, Any], view: str) -> Dict[str, Any]:
    inventory_metrics = (
        head14_metric_names() if spec["inventory"] == "head14" else full60_metric_names()
    )
    return {
        "search_space": build_search_space(view),
        "search_algorithm": optuna_block_fixed50(),
        "cvi": spec["cvi"],
        # Descriptor block; the ClustOpt builder ignores unknown top-level keys.
        "stage1": {
            "phase": PHASE_NAME,
            "method_name": spec["method"],
            "view_id": view,
            "metric_inventory": spec["inventory"],
            "metric_inventory_size": len(inventory_metrics),
            "metric_inventory_metrics": inventory_metrics,
            "utility_strategy": spec["utility_strategy"],
            "group": spec["group"],
            "deployable": spec["deployable"],
            "top_k": None if spec["utility_strategy"] == "uniform" else TOP_K,
            "weighting_mode": (
                "uniform_normalized" if spec["utility_strategy"] == "uniform"
                else WEIGHTING_RAW
            ),
            "requested_trials": 50,
            "timeout_disabled": True,
            "early_stopping_enabled": False,
            "search_seed_policy": {"seed": "auto", "base_seed": BASE_SEED},
        },
    }


def write_configs(root: Path = STAGE1_ROOT) -> List[Path]:
    written: List[Path] = []
    for spec in build_arm_specs():
        out_dir = root / spec["method"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for view in VIEWS:
            path = out_dir / f"{view}.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(build_config(spec, view), fh, indent=2)
                fh.write("\n")
            written.append(path)
    return written


def inject_head14_regressor_path(run_dir: str, root: Path = STAGE1_ROOT) -> List[Path]:
    """Late-bind the HP arm to the ACTUAL trained Head-14 run directory.

    The Head-14 predictor's timestamped path is only known after training
    (Stage 1B-2), so the generated configs carry
    :data:`HEAD14_REGRESSOR_PLACEHOLDER` instead of a fabricated timestamp.
    This rewrites only ``cvi.params.model_run_dir`` for the three HP configs and
    touches nothing else -- in particular the FP arm keeps pointing at the
    immutable historical Full-60 artifact.

    Idempotent: re-binding an already-bound config to the same directory is a
    no-op, and re-binding to a different directory simply replaces the value.
    """
    run_dir = str(run_dir).replace("\\", "/").rstrip("/")
    if not (Path(HERE.parents[3]) / run_dir).is_dir():
        raise FileNotFoundError(
            f"Head-14 run directory does not exist: {run_dir}. "
            "Train the model before binding the HP configs.")

    method = "stage1_head14_mlp_top5_raw_fixed50"
    touched: List[Path] = []
    for view in VIEWS:
        path = root / method / f"{view}.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        params = cfg["cvi"]["params"]
        previous = params.get("model_run_dir")
        params["model_run_dir"] = run_dir
        cfg.setdefault("stage1", {})["head14_regressor_bound_from"] = previous
        with path.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
        touched.append(path)
    return touched


def main() -> int:
    written = write_configs()
    print(f"Wrote {len(written)} Stage-1 2x3 configs "
          f"({len(build_arm_specs())} arms x {len(VIEWS)} views):")
    for p in written:
        print(f"  {p.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
