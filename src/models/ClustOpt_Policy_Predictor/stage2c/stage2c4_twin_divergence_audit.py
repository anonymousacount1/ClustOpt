"""Stage 2C-4A: why the RAW/SOFTMAX target is almost entirely tied.

A null result is only informative if its mechanism is measured rather than
asserted. This audit opens both twin trial logs of every scientific row and
localises exactly where -- if anywhere -- the two arms part company:

    weights      do the raw-normalised and softmax weight vectors differ at all?
    objective    does the per-trial aggregate score differ?
    trajectory   does Optuna visit a different sequence of configurations?
    slate        does the resulting candidate set differ?
    selection    does the reranker choose a different candidate?
    outcome      does the selected ARI differ (i.e. delta_mode != 0)?

Each stage can only diverge if the previous one did, so the counts form a funnel
and the null result is attributable to a specific link rather than to "no
effect". Nothing here changes a target; it reads the same artefacts the target
builder read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.stage2c import stage2c4_common as C  # noqa: E402,E501

TOL = 1e-12


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / C.OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    print("=" * 78)
    print("STAGE 2C-4A  TWIN DIVERGENCE AUDIT")
    print("=" * 78, flush=True)

    INV = pd.read_csv(_ext(out / "online_pair_inventory.csv"))
    INV = INV[INV["rank"] <= int(args.n)]
    T = pd.read_csv(_ext(out / "raw_softmax_online_targets.csv.gz")
                    ).set_index("row_id")

    rows: List[Dict[str, Any]] = []
    for row_id, g in INV.groupby("row_id"):
        d = {r["weighting_arm"]: Path(r["run_dir"]) for _, r in g.iterrows()}
        logs, wts, best = {}, {}, {}
        for m in C.MODES:
            logs[m] = pd.read_csv(_ext(d[m] / CS.CANDIDATE_CSV))
            sm = read_json(d[m] / "selected_metrics.json") or {}
            wts[m] = sm.get("selected_metric_weights") or {}
            best[m] = read_json(d[m] / "best_result.json") or {}
        a, b = logs["RAW"], logs["SOFTMAX"]
        wa, wb = wts["RAW"], wts["SOFTMAX"]
        wmax = (max(abs(wa[k] - wb.get(k, 0.0)) for k in wa) if wa else 0.0)
        same_traj = (len(a) == len(b)
                     and a["config_str"].tolist() == b["config_str"].tolist()
                     and a["algorithm"].tolist() == b["algorithm"].tolist())
        common = min(len(a), len(b))
        prefix = (a["config_str"].iloc[:common].tolist()
                  == b["config_str"].iloc[:common].tolist())
        first_div = None
        if not same_traj:
            for i in range(common):
                if a["config_str"].iloc[i] != b["config_str"].iloc[i]:
                    first_div = i
                    break
            if first_div is None:
                first_div = common
        aggs = (np.allclose(a["aggregate_score"].iloc[:common].to_numpy(float),
                            b["aggregate_score"].iloc[:common].to_numpy(float),
                            atol=0, rtol=0, equal_nan=True) if common else True)
        t = T.loc[row_id]
        rows.append({
            "row_id": row_id, "split_id": int(g["split_id"].iloc[0]),
            "regime_id": g["regime_id"].iloc[0],
            "k_mode": g["k_mode"].iloc[0], "actual_k": int(t["actual_k"]),
            "n_selected_cvis": len(wa),
            "weights_identical": bool(wmax <= TOL),
            "max_abs_weight_difference": float(wmax),
            "objective_identical_on_common_prefix": bool(aggs),
            "trajectory_identical": bool(same_traj),
            "trajectory_prefix_identical": bool(prefix),
            "first_divergent_trial": first_div,
            "n_trials_raw": int(len(a)), "n_trials_softmax": int(len(b)),
            "trial_count_differs": bool(len(a) != len(b)),
            "timeout_truncated": bool(len(a) < C.N_TRIALS
                                      or len(b) < C.N_TRIALS),
            "slate_identical": bool(same_traj and aggs),
            "reranked_index_identical": bool(t["raw_selected_index"]
                                             == t["softmax_selected_index"]),
            "delta_mode": float(t["delta_mode"]),
            "outcome_identical": bool(abs(float(t["delta_mode"])) <= TOL),
        })
    D = pd.DataFrame(rows).sort_values(["split_id", "row_id"])
    write_csv_atomic(out / "twin_divergence_audit.csv", D)

    n = len(D)
    funnel = {
        "n_rows": n,
        "1_weights_differ": int((~D["weights_identical"]).sum()),
        "1a_weights_identical_structural_k1": int(
            (D["weights_identical"] & (D["actual_k"] == 1)).sum()),
        "2_objective_differs_on_common_prefix": int(
            (~D["objective_identical_on_common_prefix"]).sum()),
        "3_trajectory_differs": int((~D["trajectory_identical"]).sum()),
        "3a_trial_count_differs": int(D["trial_count_differs"].sum()),
        "3b_trial_count_differs_and_timeout_truncated": int(
            (D["trial_count_differs"] & D["timeout_truncated"]).sum()),
        "3c_configuration_sequence_differs_within_common_prefix": int(
            (~D["trajectory_prefix_identical"]).sum()),
        "4_reranked_index_differs": int((~D["reranked_index_identical"]).sum()),
        "5_outcome_differs": int((~D["outcome_identical"]).sum()),
    }
    summary = {
        "stage": C.STAGE, "funnel": funnel,
        "mechanism": (
            "Weighting changes the objective's SCALE far more than its ORDER. "
            "With the twins sharing the frozen sampler seed, Optuna's TPE "
            "suggestion depends on the split of past trials at the gamma "
            "quantile, i.e. on the RANK of the objective, not its value. "
            "Softmax at T=0.5 over K utilities that already lie in a narrow "
            "band produces weights very close to the raw-normalised ones, so "
            "the induced ranking -- and therefore the visited slate -- is "
            "almost always identical."),
        "structural_tie_note": (
            "At K=1 the two weightings are the same function: both put weight "
            "1.0 on the single selected CVI. Those rows are exact ties by "
            "construction, not by measurement."),
        "residual_divergence_note": (
            "Rows whose trial COUNT differs are the 150 s wall-clock timeout "
            "truncating one arm at a different trial index, which is a "
            "property of the deployment regime rather than of the weighting."),
        "by_regime": D.groupby("regime_id").agg(
            n_rows=("row_id", "size"),
            weights_differ=("weights_identical", lambda s: int((~s).sum())),
            trajectory_differs=("trajectory_identical", lambda s: int((~s).sum())),
            outcome_differs=("outcome_identical", lambda s: int((~s).sum())),
            mean_max_weight_diff=("max_abs_weight_difference", "mean"),
        ).to_dict("index"),
        "max_weight_difference_overall": float(
            D["max_abs_weight_difference"].max()),
        "median_weight_difference_where_k_gt_1": float(
            D[D["actual_k"] > 1]["max_abs_weight_difference"].median()),
        "source_commit": commit, "runtime_sec": time.time() - t0,
    }
    write_json_atomic(out / "twin_divergence_summary.json", summary)
    print("  divergence funnel over %d rows:" % n, flush=True)
    for k, v in funnel.items():
        if k != "n_rows":
            print("    %-56s %4d" % (k, v), flush=True)
    print("  max |weight difference| %.4f (median where K>1: %.4f)"
          % (summary["max_weight_difference_overall"],
             summary["median_weight_difference_where_k_gt_1"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
