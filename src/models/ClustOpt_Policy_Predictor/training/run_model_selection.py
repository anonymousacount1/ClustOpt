"""Stage 2A-3B: staged, pre-registered Policy Predictor model selection.

Phase 1  3 feature sets x 3 targets, ExtraTrees + UNIFIED       (9 configs)
Phase 2  Phase-1 winner: UNIFIED vs SEPARATE                    (+1)
Phase 3  Phase-2 winner: Ridge vs ExtraTrees vs MLP             (+2)
                                                        approx. 12 unique configs

Split 1 is never read. All training data is the Stage-2A-3B0 nested-clean
outer-fold package; outer-test rows use the correct single-exclusion source.

Run::

    .venv_clustopt/Scripts/python.exe -m models.ClustOpt_Policy_Predictor\
.training.run_model_selection --repo-root . [--phase 1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

configure_process(thread_limit=4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_json_atomic,
)

from ..data import target_schema as TS  # noqa: E402
from . import evaluate_fold as EV  # noqa: E402
from . import feature_sets as FSET  # noqa: E402
from . import model_registry as MR  # noqa: E402
from . import target_transforms as TT  # noqa: E402
from . import train_fold as TF  # noqa: E402

NESTED_REL = ("results_analysis/clustopt_policy_predictor/"
              "stage2a3b0_nested_crossfit")
OUT_REL = ("results_analysis/clustopt_policy_predictor/"
           "stage2a3b_model_selection")
SPLITS = list(range(2, 17))


def cfg_dir(out: Path, cid: str) -> Path:
    return out / "outer_fold_predictions" / cid


def fold_done(out: Path, cid: str, chash: str, s: int) -> bool:
    """Completion barrier: manifest must match config hash, split and schema."""
    d = cfg_dir(out, cid)
    man = read_json(d / f"fold_{s:02d}_manifest.json")
    if not man or man.get("config_hash") != chash:
        return False
    if int(man.get("outer_split", -1)) != s or man.get("status") != "complete":
        return False
    return path_exists(d / f"fold_{s:02d}_predictions.csv.gz")


def evaluate_config(repo: Path, out: Path, feature_set: str, target: str,
                    view_arch: str, family: str, pids: List[str],
                    folds_cache: Dict[int, Any]) -> pd.DataFrame:
    """Run one configuration across all 15 outer folds (resumable)."""
    root = repo / NESTED_REL
    cid = TF.config_id(feature_set, target, view_arch, family)
    chash = TF.config_hash(feature_set, target, view_arch, family)
    d = ensure_dir(cfg_dir(out, cid))
    rows = []
    for s in SPLITS:
        mpath = d / f"fold_{s:02d}_manifest.json"
        if fold_done(out, cid, chash, s):
            rows.append(read_json(mpath)["metrics"])
            continue
        F = folds_cache.get(s)
        if F is None:
            F = FSET.load_fold(root, s)
            folds_cache.clear()          # keep at most one fold resident
            folds_cache[s] = F
        res = TF.run_fold(root, s, feature_set, target, view_arch, family, fold=F)
        base = EV.fold_baselines(F, pids)
        met = EV.fold_metrics(res["predictions"], base)
        met.update(EV.prediction_quality(res["predictions"],
                                         F["test_ari"].to_numpy(np.float64),
                                         target))
        met.update({"outer_split": s, "config_id": cid,
                    "feature_set": feature_set, "target": target,
                    "view_arch": view_arch, "model_family": family,
                    "runtime_sec": res["runtime_sec"],
                    "input_dim": res["input_dim"],
                    "global_best_single_policy": base["global_best_single_policy"],
                    "per_view_policies": json.dumps(base["per_view_policies"])})
        res["predictions"].to_csv(_ext(d / f"fold_{s:02d}_predictions.csv.gz"),
                                  index=False, compression="gzip")
        write_json_atomic(mpath, {
            "config_id": cid, "config_hash": chash, "outer_split": s,
            "status": "complete", "metrics": met, "fit_info": res["fit_info"],
            "n_train": res["n_train"], "n_test": res["n_test"],
            "nested_source": NESTED_REL, "seed": MR.GLOBAL_SEED,
            "policy_order_hash": TS.target_schema_hash(),
        })
        rows.append(met)
        print(f"      fold {s:02d}: BV={met['bestview_selector_ari']:.4f} "
              f"Rg={met['recovery_global']:+.3f} Rv={met['recovery_view']:+.3f} "
              f"({res['runtime_sec']:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def macro(df: pd.DataFrame) -> Dict[str, Any]:
    """Split-aware macro aggregation across the 15 outer folds."""
    bv = df["bestview_selector_ari"].to_numpy(float)
    rg = df["recovery_global"].to_numpy(float)
    rv = df["recovery_view"].to_numpy(float)
    ci = (float(np.mean(bv) - 1.96 * np.std(bv, ddof=1) / np.sqrt(len(bv))),
          float(np.mean(bv) + 1.96 * np.std(bv, ddof=1) / np.sqrt(len(bv))))
    return {
        "config_id": df["config_id"].iloc[0],
        "feature_set": df["feature_set"].iloc[0], "target": df["target"].iloc[0],
        "view_arch": df["view_arch"].iloc[0],
        "model_family": df["model_family"].iloc[0],
        "n_folds": int(len(df)),
        "macro_bestview": float(np.mean(bv)), "median_bestview": float(np.median(bv)),
        "std_bestview": float(np.std(bv, ddof=1)),
        "min_bestview": float(np.min(bv)), "max_bestview": float(np.max(bv)),
        "ci_lo": ci[0], "ci_hi": ci[1],
        "macro_b_global": float(df["b_global"].mean()),
        "macro_b_view": float(df["b_view"].mean()),
        "macro_oracle": float(df["oracle_bestview"].mean()),
        "macro_recovery_global": float(np.mean(rg)),
        "macro_recovery_view": float(np.mean(rv)),
        "recovery_global_min": float(np.min(rg)),
        "recovery_global_max": float(np.max(rg)),
        "recovery_view_min": float(np.min(rv)),
        "recovery_view_max": float(np.max(rv)),
        "folds_beating_b_global": int(df["beats_b_global"].sum()),
        "folds_beating_b_view": int(df["beats_b_view"].sum()),
        "mean_selected_in_winner_set": float(df["selected_in_winner_set"].mean()),
        "total_runtime_sec": float(df["runtime_sec"].sum()),
    }


def pick_winner(summaries: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Deterministic tie-break ladder, exactly as pre-registered."""
    order = {"UTILITIES": 0, "META": 1, "META_UTILITIES": 2}
    log = []
    ranked = sorted(summaries, key=lambda r: (
        -round(r["macro_bestview"], 6),
        -round(r["macro_recovery_view"], 6),
        -r["folds_beating_b_view"],
        -round(r["min_bestview"], 6),
        order.get(r["feature_set"], 9)))
    for i, r in enumerate(ranked):
        log.append(f"{i+1}. {r['config_id']}: BV={r['macro_bestview']:.5f} "
                   f"Rv={r['macro_recovery_view']:+.4f} "
                   f"beat_Bv={r['folds_beating_b_view']}/15 "
                   f"worst={r['min_bestview']:.4f}")
    return ranked[0], log


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--phase", type=int, default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    pids = TS.policy_order()
    folds_cache: Dict[int, Any] = {}
    t0 = time.time()
    all_folds: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []

    def run(fs: str, tg: str, va: str, fam: str) -> Dict[str, Any]:
        cid = TF.config_id(fs, tg, va, fam)
        print(f"\n  [{cid}]", flush=True)
        df = evaluate_config(repo, out, fs, tg, va, fam, pids, folds_cache)
        all_folds.append(df)
        m = macro(df)
        summaries.append(m)
        print(f"    macro BV={m['macro_bestview']:.4f}  "
              f"Rg={m['macro_recovery_global']:+.4f}  "
              f"Rv={m['macro_recovery_view']:+.4f}  "
              f"beat Bg {m['folds_beating_b_global']}/15  "
              f"beat Bv {m['folds_beating_b_view']}/15", flush=True)
        return m

    print("=" * 78); print("STAGE 2A-3B  --  POLICY PREDICTOR MODEL SELECTION")
    print("=" * 78)
    print(f"  nested source : {NESTED_REL}")
    print(f"  policy order  : {TS.target_schema_hash()[:16]} (20 policies)")
    print(f"  seed          : {MR.GLOBAL_SEED} | Split 1 never read")

    # ------------------------------------------------------------- PHASE 1 --
    if args.phase in (None, 1):
        print("\n" + "-" * 78)
        print("PHASE 1 -- 3 feature sets x 3 targets (ExtraTrees, UNIFIED)")
        print("-" * 78)
        p1 = []
        for fs in FSET.FEATURE_SETS:
            for tg in TT.TARGETS:
                p1.append(run(fs, tg, "UNIFIED", "EXTRATREES"))
        pd.DataFrame(p1).to_csv(_ext(out / "phase1_feature_target_results.csv"),
                                index=False)
        w1, log1 = pick_winner(p1)
        write_json_atomic(out / "phase1_winner.json",
                          {"winner": w1, "ranking": log1})
        print("\n" + "=" * 60)
        print("STAGE 2A-3B -- PHASE 1 COMPLETE\n")
        print("9 ExtraTrees unified configurations\n")
        print(f"Best feature/target pair: {w1['feature_set']} x {w1['target']}")
        print(f"Macro BestView ARI:       {w1['macro_bestview']:.4f}")
        print(f"Recovery_global:          {w1['macro_recovery_global']:+.4f}")
        print(f"Recovery_view:            {w1['macro_recovery_view']:+.4f}\n")
        print(f"15-fold range:            {w1['min_bestview']:.4f} - "
              f"{w1['max_bestview']:.4f}")
        print(f"Folds beating B_global:   {w1['folds_beating_b_global']}/15")
        print(f"Folds beating B_view:     {w1['folds_beating_b_view']}/15\n")
        print(f"Phase-1 winner: {w1['config_id']}")
        print("=" * 60, flush=True)

    if args.phase == 1:
        _persist(out, all_folds, summaries, t0, repo)
        return 0

    w1 = read_json(out / "phase1_winner.json")["winner"]
    fs1, tg1 = w1["feature_set"], w1["target"]

    # ------------------------------------------------------------- PHASE 2 --
    if args.phase in (None, 2):
        print("\n" + "-" * 78)
        print(f"PHASE 2 -- view architecture ({fs1} x {tg1}, ExtraTrees)")
        print("-" * 78)
        m_sep = run(fs1, tg1, "SEPARATE", "EXTRATREES")
        p2 = [w1, m_sep]
        pd.DataFrame(p2).to_csv(_ext(out / "phase2_view_architecture_results.csv"),
                                index=False)
        w2, log2 = pick_winner(p2)
        write_json_atomic(out / "phase2_winner.json",
                          {"winner": w2, "ranking": log2})
        print("\n" + "=" * 60)
        print("PHASE 2 -- VIEW ARCHITECTURE\n")
        print(f"Unified:  BV={w1['macro_bestview']:.4f} "
              f"Rv={w1['macro_recovery_view']:+.4f}")
        print(f"Separate: BV={m_sep['macro_bestview']:.4f} "
              f"Rv={m_sep['macro_recovery_view']:+.4f}\n")
        print(f"Winner: {w2['view_arch']}")
        print("=" * 60, flush=True)

    if args.phase == 2:
        _persist(out, all_folds, summaries, t0, repo)
        return 0

    w2 = read_json(out / "phase2_winner.json")["winner"]
    va2 = w2["view_arch"]

    # ------------------------------------------------------------- PHASE 3 --
    print("\n" + "-" * 78)
    print(f"PHASE 3 -- model family ({fs1} x {tg1} x {va2})")
    print("-" * 78)
    m_ridge = run(fs1, tg1, va2, "RIDGE")
    m_mlp = run(fs1, tg1, va2, "MLP")
    p3 = [m_ridge, w2, m_mlp]
    pd.DataFrame(p3).to_csv(_ext(out / "phase3_model_family_results.csv"),
                            index=False)
    w3, log3 = pick_winner(p3)
    write_json_atomic(out / "phase3_winner.json", {"winner": w3, "ranking": log3})
    print("\n" + "=" * 60)
    print("PHASE 3 -- MODEL FAMILY\n")
    for nm, m in (("Ridge", m_ridge), ("ExtraTrees", w2), ("MLP", m_mlp)):
        print(f"{nm:12} BV={m['macro_bestview']:.4f} "
              f"Rg={m['macro_recovery_global']:+.4f} "
              f"Rv={m['macro_recovery_view']:+.4f} "
              f"beatBv={m['folds_beating_b_view']}/15")
    print(f"\nFinal selected formulation: {w3['config_id']}")
    print("=" * 60, flush=True)

    write_json_atomic(out / "selected_configuration.json", {
        "config_id": w3["config_id"], "feature_set": w3["feature_set"],
        "target": w3["target"], "view_arch": w3["view_arch"],
        "model_family": w3["model_family"],
        "input_dim": FSET.expected_dim(w3["feature_set"], w3["view_arch"]),
        "output_dim": 20,
        "model_config": MR.config_of(w3["model_family"]),
        "selection_rule": ("argmax" if w3["target"] in ("ARI", "CENTERED")
                           else "argmin"),
        "policy_order_hash": TS.target_schema_hash(),
        "seed": MR.GLOBAL_SEED,
        "macro": w3,
        "strong_go": {
            "A_recovery_global_ge_0.50": bool(w3["macro_recovery_global"] >= 0.50),
            "B_beats_b_view_ge_12_of_15": bool(w3["folds_beating_b_view"] >= 12),
            "C_bestview_gt_b_view": bool(w3["macro_bestview"] > w3["macro_b_view"]),
            "PASS": bool(w3["macro_recovery_global"] >= 0.50
                         and w3["folds_beating_b_view"] >= 12
                         and w3["macro_bestview"] > w3["macro_b_view"]),
        },
    })
    _persist(out, all_folds, summaries, t0, repo)
    return 0


def _persist(out: Path, all_folds, summaries, t0, repo) -> None:
    if all_folds:
        pd.concat(all_folds, ignore_index=True).to_csv(
            _ext(out / "all_configuration_fold_metrics.csv"), index=False)
    if summaries:
        pd.DataFrame(summaries).to_csv(
            _ext(out / "all_configuration_summary.csv"), index=False)
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "Stage 2A-3B -- Policy Predictor model selection",
        "nested_source": NESTED_REL, "split1_read": False,
        "n_outer_folds": 15, "policy_order_hash": TS.target_schema_hash(),
        "seed": MR.GLOBAL_SEED,
        "extratrees_deviation": MR.EXTRATREES_CONFIG["deviation_reason"],
        "runtime_sec": time.time() - t0,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                        capture_output=True,
                                        text=True).stdout.strip(),
    })


if __name__ == "__main__":
    sys.exit(main())
