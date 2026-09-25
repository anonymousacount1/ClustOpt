"""Seeded criterion-baseline sweep (internal label E1; paper Tables 1, 16 and 17). Included for transparency; it expects the regenerated controlled repository.

E1 / Step 3-4 -- generate the five E1 arm configs.

Every arm inherits the search_space VERBATIM from the deployed C4 controlled
config (phaseE2_knn/knn_top10_raw/<view>.json), so the eligible configuration
domain, the per-view algorithm set and every hyper-parameter range are
identical by construction rather than by transcription.

Differences from the historical C4 config, applied IDENTICALLY to all five arms:

  * ``seed: "auto"`` + ``base_seed: 42`` -- the established Stage-1 derivation
    (seeding.derive_seed: SHA-256 over "base_seed|dataset_id|view_id", arm id
    deliberately NOT in the payload), so every arm receives the same sampler
    seed for the same (dataset, view).
  * the 150 s per-view ``timeout`` is REMOVED, so the binding stop criterion is
    exactly n_trials = 50 for every arm. The historical end-to-end runs kept the
    timeout and therefore did NOT deliver equal budgets: measured mean delivered
    trials were 48.4 (knn_top10_raw) but only 24.2 (all_metrics_uniform). A
    conditioning contrast run under that timeout would confound criterion with
    budget.

STEP-8 GATE (enforced here, not assumed):
  GenericCVIEvaluator selects metrics by REGISTRY KEY but looks weights up by
  each metric's ``.name`` (cvi_generic.py: ``self.weights.get(m.name, 1.0)``).
  Two registry keys disagree with their ``.name``. This script therefore
  refuses to emit a ``generic`` arm that would place a NON-UNIFORM weight on an
  aliased metric, and refuses to emit any metric name the live METRIC_REGISTRY
  does not contain. It also asserts that C4's FULL60 target vocabulary is the
  full 60-metric registry.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

C4_SRC = REPO / "models" / "ClustOpt" / "configs" / "experiments" / "phaseE2_knn" / "knn_top10_raw"
DEST = REPO / "models" / "ClustOpt" / "configs" / "experiments" / "e1_conditioning"
VIEWS = ("x_only", "y_only", "xy_2d")
BASE_SEED = 42
N_TRIALS = 50
CORE3 = ["silhouette", "noise_aware_silhouette", "gridness_fft_acf"]

GENERIC_PARAMS = {"remove_noise": True, "noise_label": -1, "min_clusters": 2}

ARMS = ("E1_C4_SEED", "E1_MEANPROF", "E1_CORE3", "E1_BEST1", "E1_RANDOM")


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")


def main() -> int:
    from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
        METRIC_REGISTRY,
    )
    name_of_key = {k: getattr(m, "name", k) for k, m in METRIC_REGISTRY.items()}
    aliased = {k for k, n in name_of_key.items() if n != k}

    prof = pd.read_csv(HERE / "development_mean_profile.csv")
    best = pd.read_csv(HERE / "development_best_single_index.csv")
    best_pv = {r["view"]: r["registry_key"]
               for _, r in best[best.scope == "per_view"].iterrows()}

    # ---- Step-8 completeness: C4's FULL60 == the live 60-metric registry ----
    full60 = sorted(prof[prof.view == VIEWS[0]]["registry_key"])
    assert len(full60) == 60, "expected 60 FULL60 metrics, got %d" % len(full60)
    assert set(full60) == set(METRIC_REGISTRY), (
        "C4 FULL60 vocabulary != live METRIC_REGISTRY; refusing to build")
    assert aliased <= set(full60), "aliased metrics missing from FULL60"

    def guard(weights: dict, arm: str, view: str):
        """Refuse any config the generic CVI path would mis-weight."""
        unknown = [m for m in weights if m not in METRIC_REGISTRY]
        if unknown:
            raise SystemExit("STOP [%s/%s]: metrics not in METRIC_REGISTRY: %s"
                             % (arm, view, unknown))
        uniform = len({round(float(w), 12) for w in weights.values()}) == 1
        bad = [m for m in weights if m in aliased]
        if bad and not uniform:
            raise SystemExit(
                "STOP [%s/%s]: aliased metric(s) %s carry non-uniform weights. "
                "GenericCVIEvaluator would silently substitute weight 1.0 "
                "(cvi_generic.py: weights.get(m.name, 1.0)). Refusing to run."
                % (arm, view, bad))
        return {"uniform_weights": uniform, "aliased_present": bad}

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    manifest = {
        "generated_for": "E1 conditioning decision",
        "search_space_inherited_verbatim_from": rel(C4_SRC),
        "search_algorithm": {"type": "optuna", "n_trials": N_TRIALS,
                             "seed": "auto", "base_seed": BASE_SEED,
                             "timeout": "REMOVED (historical C4 used 150 s)"},
        "seed_derivation": ("models/Clustering_Repository_Builder/experiments/"
                            "experiment_execution/seeding.py :: derive_seed -- "
                            "int.from_bytes(sha256('<base_seed>|<dataset_id>|"
                            "<view_id>')[:4],'big') %% 2**31; the arm id is NOT "
                            "part of the payload"),
        "design_statement": ("shared sampler initialisation / paired "
                            "dataset-view seed -- NOT a shared candidate slate: "
                            "TPE is adaptive, so a different objective produces "
                            "a different trajectory from the same seed"),
        "step8": {"full60_equals_live_registry": True,
                  "aliased_metrics": {k: name_of_key[k] for k in sorted(aliased)},
                  "broken_unified_mkr_lookup_used": False,
                  "arms": {}},
        "arms": {},
    }

    for view in VIEWS:
        c4 = json.loads((C4_SRC / f"{view}.json").read_text(encoding="utf-8"))
        space = c4["search_space"]
        sa = {"type": "optuna",
              "params": {"n_trials": N_TRIALS, "seed": "auto",
                         "base_seed": BASE_SEED}}

        pv = prof[(prof.view == view) & (prof.selected_top10)] \
            .sort_values("rank")
        mean_w = {r["registry_key"]: float(r["normalized_weight"])
                  for _, r in pv.iterrows()}

        specs = {
            "E1_C4_SEED": dict(
                cvi=dict(c4["cvi"]),                      # identical to C4
                note="dataset-specific KNN predicted profile, FULL60, top-10, "
                     "RAW normalized weighting -- the deployed C4 objective",
                weights=None),
            "E1_MEANPROF": dict(
                cvi={"type": "generic", "metrics": mean_w,
                     "params": dict(GENERIC_PARAMS)},
                note="development-only mean TRUE utility profile (splits 2-16), "
                     "FULL60, top-10, identical RAW normalized weighting",
                weights=mean_w),
            "E1_CORE3": dict(
                cvi={"type": "generic",
                     "metrics": {m: 1.0 for m in CORE3},
                     "params": dict(GENERIC_PARAMS)},
                note="static three-index generalist core, UNIFORM weights",
                weights={m: 1.0 for m in CORE3}),
            "E1_BEST1": dict(
                cvi={"type": "generic",
                     "metrics": {best_pv[view]: 1.0},
                     "params": dict(GENERIC_PARAMS)},
                note="development-selected best single index for this view",
                weights={best_pv[view]: 1.0}),
            "E1_RANDOM": dict(
                cvi={"type": "generic", "metrics": {"silhouette": 1.0},
                     "params": dict(GENERIC_PARAMS)},
                note="uniform RANDOM sampler over the same space. Optuna's "
                     "RandomSampler ignores objective values, so the trajectory "
                     "is objective-independent; the recorded aggregate_score is "
                     "inert and the arm's endpoint is drawn uniformly from the "
                     "valid visited candidates in post-processing, with a "
                     "deterministic RNG seeded by the same (dataset, view) seed.",
                weights={"silhouette": 1.0}),
        }
        specs["E1_RANDOM"]["sampler"] = "random"

        for arm, spec in specs.items():
            cfg = {"search_space": space,
                   "search_algorithm": json.loads(json.dumps(sa)),
                   "cvi": spec["cvi"]}
            if spec.get("sampler"):
                cfg["search_algorithm"]["params"]["sampler"] = spec["sampler"]
            if spec["weights"] is not None:
                g = guard(spec["weights"], arm, view)
                manifest["step8"]["arms"].setdefault(arm, {})[view] = g
            else:
                manifest["step8"]["arms"].setdefault(arm, {})[view] = {
                    "uniform_weights": None, "aliased_present": [],
                    "note": "dynamic knn CVI resolves weights per record through "
                            "cvi_dynamic._build_delegate, which maps .name -> "
                            "registry key correctly; not affected by the generic "
                            "weight-lookup hazard"}
            cfg["e1"] = {
                "experiment": "E1_conditioning_decision",
                "arm": arm, "view_id": view,
                "objective_note": spec["note"],
                "n_trials": N_TRIALS, "timeout": None,
                "seed_rule": "auto (per dataset|view, arm-independent)",
                "base_seed": BASE_SEED,
                "selected_metrics": (sorted(spec["weights"]) if spec["weights"]
                                     else "resolved per record by the KNN predictor"),
            }
            d = DEST / arm.lower()
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{view}.json").write_text(json.dumps(cfg, indent=2),
                                            encoding="utf-8")
            manifest["arms"].setdefault(arm, {})[view] = {
                "config": rel(d / f"{view}.json"),
                "cvi_type": spec["cvi"]["type"],
                "n_metrics": (len(spec["weights"]) if spec["weights"] else 60),
                "metrics": (sorted(spec["weights"]) if spec["weights"] else None),
                "sampler": spec.get("sampler", "tpe"),
            }

    (HERE / "e1_config_build_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("STEP-8 GATE: PASS (no arm mis-weights an aliased metric)")
    print("configs written under:", rel(DEST))
    for arm in ARMS:
        a = manifest["arms"][arm]["xy_2d"]
        print("  %-12s cvi=%-13s sampler=%-6s n_metrics=%s"
              % (arm, a["cvi_type"], a["sampler"], a["n_metrics"]))
        if a["metrics"]:
            print("      ", ", ".join(a["metrics"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
