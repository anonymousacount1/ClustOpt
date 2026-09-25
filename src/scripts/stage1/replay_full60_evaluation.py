"""Stage-1B-2 gate: reproduce the historical Full-60 evaluation exactly.

Before the Head-14 model may be evaluated, we must prove that the *production*
evaluator, running in the reconstructed Stage-1 environment, reproduces the
metric values the historical Full-60 run recorded.

This script is a CALLER ONLY. Every metric is computed by
``models.metric_utility_mlp.metrics.compute_all`` -- the same function
``evaluator.evaluate_fold`` calls during training. No metric is redefined here.

Inputs (immutable, read-only):
    .../20260615_231715__metric_utility_mlp_split1_holdout/
        config.json                     -> the historical EvaluationConfig
        folds/fold_0/test_predictions.csv -> stored y_true / y_pred
        folds/fold_0/test_metrics.json    -> stored metric values

Output: docs/internal_reports/stage1/full60_evaluation_replay.csv

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/replay_full60_evaluation.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

RUN = os.path.join(REPO, "results_analysis", "mlp",
                   "20260615_231715__metric_utility_mlp_split1_holdout")
OUT = os.path.join(REPO, "docs", "project_lead_revision", "stage1",
                   "full60_evaluation_replay.csv")

# Metrics that are run bookkeeping rather than evaluation output.
NON_EVAL_KEYS = {"fold", "best_epoch", "best_val_loss"}
TOL = 1e-9


def main() -> int:
    print("=" * 78)
    print("HISTORICAL FULL-60 EVALUATION REPLAY GATE")
    print("=" * 78)
    print(f"  interpreter : {sys.executable}")

    from models.metric_utility_mlp.metrics import compute_all

    cfg = json.load(open(os.path.join(RUN, "config.json"), encoding="utf-8"))
    ev = cfg["evaluation"]
    target_cols = json.load(open(os.path.join(RUN, "artifacts", "target_columns.json"),
                                 encoding="utf-8"))
    prefix = cfg["data"]["target_prefix"]
    target_names = [c[len(prefix):] if c.startswith(prefix) else c for c in target_cols]
    print(f"  eval config : topk_list={ev['topk_list']} ndcg_list={ev['ndcg_list']} "
          f"ndcg_all={ev['ndcg_all']}")
    print(f"  targets     : {len(target_names)}")

    preds = pd.read_csv(os.path.join(RUN, "folds", "fold_0", "test_predictions.csv"))
    y_true = preds[[f"true_{m}" for m in target_names]].to_numpy(dtype=np.float64)
    y_pred = preds[[f"pred_{m}" for m in target_names]].to_numpy(dtype=np.float64)
    print(f"  predictions : {y_true.shape[0]:,} x {y_true.shape[1]}")

    stored = json.load(open(os.path.join(RUN, "folds", "fold_0", "test_metrics.json"),
                            encoding="utf-8"))

    replayed, _per_metric = compute_all(
        y_true, y_pred, target_names,
        ev["topk_list"], ev["ndcg_list"], ev["ndcg_all"],
    )

    rows, n_ok, n_bad, n_missing = [], 0, 0, 0
    max_disc, worst = 0.0, ""
    for key, sval in stored.items():
        if key in NON_EVAL_KEYS:
            continue
        if key not in replayed:
            n_missing += 1
            rows.append({"metric": key, "stored": sval, "replayed": "",
                         "abs_diff": "", "tolerance": TOL, "status": "MISSING"})
            continue
        rval = replayed[key]
        if isinstance(sval, (int, float)) and isinstance(rval, (int, float)):
            d = abs(float(sval) - float(rval))
            if math.isnan(float(sval)) and math.isnan(float(rval)):
                d = 0.0
            ok = d <= TOL
        else:
            d = 0.0 if sval == rval else float("nan")
            ok = sval == rval
        n_ok += ok
        n_bad += (not ok)
        if isinstance(d, float) and not math.isnan(d) and d > max_disc:
            max_disc, worst = d, key
        rows.append({"metric": key, "stored": sval, "replayed": rval,
                     "abs_diff": d, "tolerance": TOL,
                     "status": "PASS" if ok else "FAIL"})

    extra = [k for k in replayed if k not in stored and k not in NON_EVAL_KEYS]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  stored evaluation metrics : {len(rows)}")
    print(f"  reproduced (|d| <= {TOL})  : {n_ok}")
    print(f"  mismatched                : {n_bad}")
    print(f"  missing from replay       : {n_missing}")
    print(f"  extra in replay (not stored): {len(extra)} {extra[:6]}")
    print(f"  max discrepancy           : {max_disc:.3e}" +
          (f"  (on '{worst}')" if worst else ""))
    for r in rows:
        if r["status"] != "PASS":
            print(f"    {r['status']}: {r['metric']} stored={r['stored']} "
                  f"replayed={r['replayed']}")
    print(f"  detail -> {os.path.relpath(OUT, REPO)}")
    ok = (n_bad == 0 and n_missing == 0)
    print()
    print(f"  VERDICT: {'EVALUATION CONTRACT REPRODUCED' if ok else 'REPLAY FAILED'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
