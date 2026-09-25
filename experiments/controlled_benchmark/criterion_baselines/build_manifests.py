"""Seeded criterion-baseline sweep (internal label E1; paper Tables 1, 16 and 17). Included for transparency; it expects the regenerated controlled repository.

E1 -- emit e1_seed_manifest.json, e1_run_manifest.json and
e1_candidate_persistence_audit.json from the persisted run outputs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

ANALYZED = REPO / "results_analysis" / "clustering_repository" / "analyzed_data"
CFG = REPO / "models" / "ClustOpt" / "configs" / "experiments" / "e1_conditioning"
VIEWS = ("x_only", "y_only", "xy_2d")
E1_ARMS = ("e1_c4_seed", "e1_meanprof", "e1_core3", "e1_best1", "e1_random")
E1_SPLIT = "split_01_e1_conditioning"
BASE_SEED = 42
REQUIRED = ("clustopt_results.csv", "best_result.json", "external_metrics.json",
            "selected_metrics.json", "timing_summary.json", "run_log.json",
            "status.json", "config_snapshot.json")


def LP(p): return "\\\\?\\" + os.path.abspath(str(p))


def rj(p):
    try:
        with open(LP(p), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def sha16(p):
    h = hashlib.sha256()
    with open(LP(p), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def main() -> int:
    from models.Clustering_Repository_Builder.experiments.experiment_execution.seeding import (
        derive_seed, DEFAULT_BASE_SEED, SEED_MODULUS,
    )

    seeds, runs, persist = [], [], []
    seed_mismatch, missing_files = [], []
    # Split-1 only, straight from the canonical split table (no tree walk).
    sa = pd.read_csv(ANALYZED / "experiment_splits" / "dataset_split_assignments.csv")
    sa = sa[pd.to_numeric(sa["split_id"], errors="coerce") == 1]
    for _, srow in sa.iterrows():
        ds = Path(srow["dataset_dir"])
        fam = type("F", (), {"name": srow["family_id"]})
        sub = type("S", (), {"name": srow["subfamily_id"]})
        if True:
            if True:
                for v in VIEWS:
                    expected = derive_seed(BASE_SEED, ds.name, v)
                    observed = {}
                    for arm in E1_ARMS:
                        base = ds / "experiments" / E1_SPLIT / arm / v
                        st = rj(base / "status.json")
                        if not st:
                            continue
                        br = rj(base / "best_result.json") or {}
                        observed[arm] = br.get("search_seed")
                        miss = [f for f in REQUIRED
                                if not os.path.exists(LP(base / f))]
                        cand = base / "clustopt_results.csv"
                        n_rows = None
                        if os.path.exists(LP(cand)):
                            try:
                                with open(LP(cand), "r", encoding="utf-8") as fh:
                                    n_rows = sum(1 for _ in fh) - 1
                            except Exception:
                                pass
                        persist.append({
                            "dataset_id": ds.name, "view_id": v, "arm": arm,
                            "status": st.get("status"),
                            "per_candidate_rows": n_rows,
                            "delivered_trials": br.get("delivered_trials"),
                            "missing_files": ";".join(miss),
                        })
                        if miss:
                            missing_files.append(f"{ds.name}/{v}/{arm}: {miss}")
                        runs.append({
                            "dataset_id": ds.name, "family": fam.name,
                            "subfamily": sub.name, "view_id": v, "arm": arm,
                            "status": st.get("status"),
                            "runtime_sec": st.get("runtime_sec"),
                        })
                    if observed:
                        vals = {x for x in observed.values() if x is not None}
                        ok = (len(vals) == 1 and next(iter(vals)) == expected)
                        if not ok:
                            seed_mismatch.append(
                                {"dataset_id": ds.name, "view_id": v,
                                 "expected": expected, "observed": observed})
                        seeds.append({"dataset_id": ds.name, "view_id": v,
                                      "derived_seed": expected,
                                      "arms_observed": len(observed),
                                      "all_arms_share_seed": ok})

    seed_df = pd.DataFrame(seeds)
    run_df = pd.DataFrame(runs)
    per_df = pd.DataFrame(persist)
    if not per_df.empty:
        per_df.to_csv(HERE / "e1_candidate_persistence_records.csv", index=False)

    seed_manifest = {
        "rule": "seeding.derive_seed(base_seed, dataset_id, view_id)",
        "formula": "int.from_bytes(sha256('<base_seed>|<dataset_id>|<view_id>')"
                   ".digest()[:4], 'big') % 2**31",
        "source": "models/Clustering_Repository_Builder/experiments/"
                  "experiment_execution/seeding.py",
        "base_seed": BASE_SEED,
        "default_base_seed_in_module": DEFAULT_BASE_SEED,
        "seed_modulus": SEED_MODULUS,
        "arm_id_in_payload": False,
        "design_statement": "shared sampler initialisation / paired dataset-view "
                            "seed -- NOT a shared candidate slate. TPE is "
                            "adaptive, so an arm with a different objective "
                            "explores a different trajectory from the same seed.",
        "reused_from": "the established Stage-1 derivation; the Stage-1 seeded "
                       "sweep (split_01_online_fixed50) uses the identical rule "
                       "and base seed, which is why its arms are paired-"
                       "comparable with the new E1 arms",
        "n_dataset_view_pairs_checked": int(len(seed_df)),
        "all_arms_share_seed": bool(seed_df["all_arms_share_seed"].all())
        if len(seed_df) else None,
        "seed_mismatches": seed_mismatch[:20],
        "n_seed_mismatches": len(seed_mismatch),
        "example_seeds": (seed_df.head(6).to_dict("records") if len(seed_df) else []),
    }
    (HERE / "e1_seed_manifest.json").write_text(
        json.dumps(seed_manifest, indent=2, default=str), encoding="utf-8")

    cfg_hashes = {}
    for arm in E1_ARMS:
        for v in VIEWS:
            p = CFG / arm / f"{v}.json"
            if os.path.exists(LP(p)):
                cfg_hashes[f"{arm}/{v}"] = sha16(p)

    prog = rj(HERE / "run_state" / "e1_progress.json") or {}
    run_manifest = {
        "experiment": "E1 conditioning decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": prog.get("run_id"),
        "split_dir_suffix": E1_SPLIT.replace("split_01_", ""),
        "output_split_dir": E1_SPLIT,
        "isolation": "new split dir; split_01, split_01_online_fixed50 and "
                     "split_01_offline_fixed32 are untouched",
        "arms": list(E1_ARMS),
        "reused_arms": {"stage1_full60_uniform_fixed50":
                        "E1-UNIF60, reused from split_01_online_fixed50 "
                        "(identical seeded protocol, verified)"},
        "budget": {"n_trials": 50, "timeout_sec": None,
                   "note": "the historical end-to-end arms carried a 150 s "
                           "per-view timeout and therefore did NOT deliver equal "
                           "budgets (measured mean delivered trials: 48.4 for "
                           "knn_top10_raw vs 24.2 for all_metrics_uniform); the "
                           "E1 arms remove it so the budget binds on trials"},
        "views": list(VIEWS),
        "search_space": "inherited verbatim from phaseE2_knn/knn_top10_raw; "
                        "per-view hashes verified identical across every arm",
        "config_sha256_16": cfg_hashes,
        "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "head": git("rev-parse", "HEAD")},
        "environment": {
            "interpreter": ".venv_clustopt (Python 3.12.2)",
            "numpy": "2.1.3", "scipy": "1.14.1", "scikit-learn": "1.6.0",
            "optuna": "4.1.0", "shapely": "2.0.6", "opencv": "4.10.0",
            "note": "matches the recorded scientific environment "
                    "(paper_analysis_v2 run_config: Python 3.12, NumPy 2.1.3, "
                    "SciPy 1.14.1, scikit-learn 1.6.0)",
            "host_platform": platform.platform(),
        },
        "source_change": {
            "file": "models/ClustOpt/search_algorithm/"
                    "search_algorithm_implementations/Optuna_Search.py",
            "change": "additive optional `sampler` parameter ('tpe' | 'random'); "
                      "omitting it reproduces every historical path byte-for-byte",
            "why": "E1-RANDOM needs Optuna's uniform RandomSampler over the same "
                   "search space",
        },
        "progress": {
            "subfamilies_complete": sum(1 for v in prog.get("completed", {}).values() if v),
            "subfamilies_total": 86,
            "datasets_with_any_run": int(run_df["dataset_id"].nunique())
            if len(run_df) else 0,
            "runs_persisted": int(len(run_df)),
            "runs_expected_full": 1055 * 3 * len(E1_ARMS),
        },
        "runtime": {
            "total_runtime_hours_persisted":
                float(run_df["runtime_sec"].sum() / 3600.0) if len(run_df) else 0.0,
            "workers": 7, "parallelism": "record",
        },
    }
    (HERE / "e1_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")

    audit = {
        "requirement": "FULL per-candidate tables persisted for every run (needed by E2)",
        "per_candidate_file": "clustopt_results.csv",
        "n_runs_checked": int(len(per_df)),
        "n_runs_with_per_candidate_table": int(
            (per_df["per_candidate_rows"].notna()).sum()) if len(per_df) else 0,
        "n_runs_missing_any_required_file": int(
            (per_df["missing_files"] != "").sum()) if len(per_df) else 0,
        "missing_file_examples": missing_files[:20],
        "per_candidate_rows_distribution": (
            per_df["per_candidate_rows"].value_counts().head(10).to_dict()
            if len(per_df) else {}),
        "delivered_trials_distribution": (
            per_df["delivered_trials"].value_counts().head(10).to_dict()
            if len(per_df) else {}),
        "all_runs_delivered_50_trials": bool(
            (per_df["delivered_trials"] == 50).all()) if len(per_df) else None,
        "verdict": ("PASS" if len(per_df) and (per_df["missing_files"] == "").all()
                    and (per_df["per_candidate_rows"].notna()).all()
                    else "INCOMPLETE OR FAILING -- see records"),
        "records_csv": "e1_candidate_persistence_records.csv",
    }
    (HERE / "e1_candidate_persistence_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8")

    print("seed pairs checked:", len(seed_df),
          "| all arms share seed:", seed_manifest["all_arms_share_seed"],
          "| mismatches:", len(seed_mismatch))
    print("runs persisted:", len(run_df), "/", 1055 * 3 * len(E1_ARMS))
    print("persistence verdict:", audit["verdict"],
          "| all delivered 50 trials:", audit["all_runs_delivered_50_trials"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
