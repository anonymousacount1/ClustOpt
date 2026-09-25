"""Stage-1B-2 post-training validation of the Head-14 MLP artifact.

CALLER ONLY: every scientific quantity is computed by the production modules
under ``models/metric_utility_mlp`` (metrics.compute_all) or by the production
ClustOpt predictor/resolver. Nothing scientific is redefined here.

Covers:
  17 artifact integrity + SHA256
  18/19 Head-14 evaluation via the production evaluator
  20 production load test (RegressorPredictor, 3,165 x 14)
  21 HP Top-5 policy selection check via the production resolver
  22 DIAGNOSTIC: Head-14 vs Full-60-restricted-to-Head-14 (NOT an arm)
  23 side-by-side Full-60 vs Head-14 evaluation table
  24 parity audit

Outputs (docs/internal_reports/stage1/):
  head14_mlp_validation.json
  head14_evaluation_metrics.csv
  head14_vs_full60_eval_table.csv
  head14_vs_restricted_full60_diagnostic.csv
  head14_parity_audit.csv

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/validate_head14_artifact.py \
        --run-dir results_analysis/mlp/<ts>__metric_utility_mlp_head14_split1_holdout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FULL60_REL = "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout"
FULL60 = os.path.join(REPO, FULL60_REL)
DOCS = os.path.join(REPO, "docs", "project_lead_revision", "stage1")
UNIFIED = os.path.join(REPO, "results_analysis", "clustering_repository", "analyzed_data",
                       "unified_training_dataset", "unified_training_dataset.csv")

RESULTS: list[tuple[str, str, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_rel = args.run_dir.replace("\\", "/").rstrip("/")
    RUN = os.path.join(REPO, run_rel)

    print("=" * 78)
    print("HEAD-14 ARTIFACT VALIDATION")
    print("=" * 78)
    print(f"  artifact : {run_rel}")
    print(f"  reference: {FULL60_REL}")

    from models.metric_utility_mlp.metrics import compute_all
    from models.metric_utility_mlp.head_groups import resolve_target_group
    head14 = resolve_target_group("head14")
    all_ok = True

    # ------------------------------------------------------------- 17 artifacts
    print("\n-- 17. ARTIFACT INTEGRITY")
    expect = ["config.json", "run_summary.md", "fold_results.csv", "overall_results.csv",
              "artifacts/feature_columns.json", "artifacts/target_columns.json",
              "artifacts/head_mapping.json", "artifacts/target_head_mapping.json",
              "artifacts/cv_split_summary.csv", "artifacts/model_architecture.txt",
              "folds/fold_0/best_model.pt", "folds/fold_0/x_scaler.pkl",
              "folds/fold_0/test_predictions.csv", "folds/fold_0/test_metrics.json",
              "folds/fold_0/split_summary.json", "folds/fold_0/train_log.csv",
              "folds/fold_0/val_log.csv", "folds/fold_0/per_metric_errors.csv"]
    missing = [p for p in expect if not os.path.exists(os.path.join(RUN, p))]
    all_ok &= rec("all expected production artifacts present", not missing,
                  f"{len(expect) - len(missing)}/{len(expect)}" +
                  (f"; MISSING {missing}" if missing else ""))

    hashes = {}
    for p in ["config.json", "artifacts/feature_columns.json", "artifacts/target_columns.json",
              "artifacts/head_mapping.json", "folds/fold_0/best_model.pt",
              "folds/fold_0/x_scaler.pkl", "folds/fold_0/test_predictions.csv"]:
        fp = os.path.join(RUN, p)
        if os.path.exists(fp):
            hashes[p] = {"sha256": sha256(fp), "size": os.path.getsize(fp)}
            print(f"       {p:42} {hashes[p]['sha256'][:16]}...  {hashes[p]['size']:,} B")

    tgt = json.load(open(os.path.join(RUN, "artifacts", "target_columns.json"), encoding="utf-8"))
    feat = json.load(open(os.path.join(RUN, "artifacts", "feature_columns.json"), encoding="utf-8"))
    hist_feat = json.load(open(os.path.join(FULL60, "artifacts", "feature_columns.json"),
                               encoding="utf-8"))
    all_ok &= rec("persisted targets == canonical 14 in order",
                  tgt == [f"utility__{m}" for m in head14], f"n={len(tgt)}")
    all_ok &= rec("persisted features == Full-60's 250 (names AND order)",
                  feat == hist_feat, f"n={len(feat)}")

    # ------------------------------------------------------- 18/19 evaluation
    print("\n-- 18/19. HEAD-14 EVALUATION (production evaluator)")
    cfg = json.load(open(os.path.join(RUN, "config.json"), encoding="utf-8"))
    ev = cfg["evaluation"]
    names14 = [c[len("utility__"):] for c in tgt]
    pf = pd.read_csv(os.path.join(RUN, "folds", "fold_0", "test_predictions.csv"))
    yt = pf[[f"true_{m}" for m in names14]].to_numpy(dtype=np.float64)
    yp = pf[[f"pred_{m}" for m in names14]].to_numpy(dtype=np.float64)
    h14_summary, h14_per_metric = compute_all(
        yt, yp, names14, ev["topk_list"], ev["ndcg_list"], ev["ndcg_all"])
    stored = json.load(open(os.path.join(RUN, "folds", "fold_0", "test_metrics.json"),
                            encoding="utf-8"))
    diffs = [k for k, v in h14_summary.items()
             if isinstance(v, (int, float)) and isinstance(stored.get(k), (int, float))
             and abs(float(v) - float(stored[k])) > 1e-9]
    all_ok &= rec("replaying the stored Head-14 predictions reproduces its metrics",
                  not diffs, f"{len(h14_summary)} metrics, {len(diffs)} mismatch")
    all_ok &= rec("same K values as Full-60 (topk/ndcg lists unchanged)",
                  ev["topk_list"] == [3, 5, 10] and ev["ndcg_list"] == [3, 5, 10],
                  f"topk={ev['topk_list']} ndcg={ev['ndcg_list']}")
    pd.DataFrame([{"metric": k, "head14": v} for k, v in h14_summary.items()]
                 ).to_csv(os.path.join(DOCS, "head14_evaluation_metrics.csv"), index=False)
    print(f"       MAE={h14_summary.get('mae'):.6f}  top1={h14_summary.get('top1_accuracy')}  "
          f"ndcg@10={h14_summary.get('ndcg@10')}")

    # ------------------------------------------------------------ 23 side-by-side
    print("\n-- 23. FULL-60 vs HEAD-14 EVALUATION TABLE")
    f60 = json.load(open(os.path.join(FULL60, "folds", "fold_0", "test_metrics.json"),
                         encoding="utf-8"))
    rows = []
    for k in sorted(set(f60) | set(h14_summary)):
        a, b = f60.get(k), h14_summary.get(k)
        d = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else ""
        rows.append({"metric": k, "full60_historical": a, "head14_new": b, "difference": d})
    pd.DataFrame(rows).to_csv(os.path.join(DOCS, "head14_vs_full60_eval_table.csv"), index=False)
    for k in ["mae", "rmse", "top1_accuracy", "top10_overlap", "ndcg@10", "spearman_mean"]:
        if k in f60 or k in h14_summary:
            print(f"       {k:24} full60={f60.get(k)}   head14={h14_summary.get(k)}")

    # --------------------------------------------------------- 20 production load
    print("\n-- 20. PRODUCTION LOAD TEST (RegressorPredictor)")
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.regressor_predictor import (
        get_regressor_predictor,
    )
    pred = get_regressor_predictor(run_rel)
    all_ok &= rec("predictor exposes exactly 14 metrics", len(pred.metric_names) == 14,
                  f"n={len(pred.metric_names)}")
    all_ok &= rec("predictor metric order == canonical 14",
                  list(pred.metric_names) == head14)
    all_ok &= rec("predictor expects 250 features", len(pred.feature_columns) == 250)

    key = "record_id"
    ids = pf[key].tolist()
    uni = pd.read_csv(UNIFIED, usecols=list(dict.fromkeys([key] + feat)))
    uni = uni.drop_duplicates(subset=[key]).set_index(key)
    all_ok &= rec("no missing records", all(r in uni.index for r in ids))
    all_ok &= rec("no duplicate records", len(set(ids)) == len(ids), f"{len(ids):,} unique")
    X = uni.loc[ids, feat].to_numpy(dtype=np.float64)
    P = np.empty((X.shape[0], 14))
    for i in range(X.shape[0]):
        o = pred.predict(X[i])
        P[i] = [o[m] for m in head14]
    all_ok &= rec("prediction matrix shape == (3165, 14)", P.shape == (3165, 14), str(P.shape))
    all_ok &= rec("all predictions finite", bool(np.isfinite(P).all()))
    all_ok &= rec("predictions within sigmoid range [0,1]",
                  bool((P >= 0).all() and (P <= 1).all()),
                  f"min={P.min():.6f} max={P.max():.6f}")
    all_ok &= rec("production inference matches stored test_predictions",
                  bool(np.allclose(P, yp, rtol=1e-4, atol=1e-5)),
                  f"max|d|={np.abs(P - yp).max():.3e}")

    # -------------------------------------------------------------- 21 HP policy
    print("\n-- 21. HP POLICY SELECTION (production resolver)")
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.weighting import (
        select_top_k, compute_weights,
    )
    head_set, new46_hits, bad = set(head14), 0, 0
    sums, sizes = [], []
    for i in range(P.shape[0]):
        util = {m: float(P[i, j]) for j, m in enumerate(head14)}
        sel = select_top_k(util, 5)
        w = compute_weights(sel, "normalized_positive", 0.5)
        sizes.append(len(w))
        sums.append(sum(w.values()))
        if not set(w) <= head_set:
            new46_hits += 1
        if not all(np.isfinite(v) and v >= 0 for v in w.values()):
            bad += 1
    all_ok &= rec("exactly 5 metrics selected per record", set(sizes) == {5},
                  f"sizes={sorted(set(sizes))}")
    all_ok &= rec("NO New-46 metric ever selected", new46_hits == 0,
                  f"{new46_hits} violations")
    all_ok &= rec("weights finite and non-negative", bad == 0, f"{bad} bad records")
    all_ok &= rec("RAW weights sum to 1",
                  bool(np.allclose(sums, 1.0, atol=1e-9)),
                  f"min={min(sums):.12f} max={max(sums):.12f}")

    # ------------------------------------------------------------ 22 diagnostic
    print("\n-- 22. DIAGNOSTIC (NOT AN EXPERIMENTAL ARM)")
    print("       Head-14 model vs Full-60 model restricted post-hoc to the same 14 outputs")
    f60pf = pd.read_csv(os.path.join(FULL60, "folds", "fold_0", "test_predictions.csv"))
    f60pf = f60pf.set_index(key).loc[ids]
    yp_r = f60pf[[f"pred_{m}" for m in head14]].to_numpy(dtype=np.float64)
    yt_r = f60pf[[f"true_{m}" for m in head14]].to_numpy(dtype=np.float64)
    all_ok &= rec("diagnostic uses identical true targets", bool(np.allclose(yt_r, yt)),
                  f"max|d|={np.abs(yt_r - yt).max():.3e}")
    r_summary, _ = compute_all(yt_r, yp_r, head14, ev["topk_list"], ev["ndcg_list"],
                               ev["ndcg_all"])
    drows = []
    for k in sorted(set(h14_summary) | set(r_summary)):
        a, b = r_summary.get(k), h14_summary.get(k)
        d = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else ""
        drows.append({"metric": k, "full60_restricted_to_head14": a,
                      "head14_native": b, "difference": d})
    # agreement BETWEEN predictors
    t1 = int(sum(np.argmax(P[i]) == np.argmax(yp_r[i]) for i in range(P.shape[0])))
    t5 = int(sum(set(np.argsort(-P[i])[:5]) == set(np.argsort(-yp_r[i])[:5])
                 for i in range(P.shape[0])))
    n = P.shape[0]
    drows.append({"metric": "top1_agreement_between_predictors",
                  "full60_restricted_to_head14": "", "head14_native": "",
                  "difference": f"{t1}/{n} = {100.0*t1/n:.2f}%"})
    drows.append({"metric": "top5_set_agreement_between_predictors",
                  "full60_restricted_to_head14": "", "head14_native": "",
                  "difference": f"{t5}/{n} = {100.0*t5/n:.2f}%"})
    pd.DataFrame(drows).to_csv(
        os.path.join(DOCS, "head14_vs_restricted_full60_diagnostic.csv"), index=False)
    for k in ["mae", "top1_accuracy", "top10_overlap", "ndcg@10", "spearman_mean"]:
        print(f"       {k:24} restricted-full60={r_summary.get(k)}   head14={h14_summary.get(k)}")
    print(f"       Top-1 agreement between predictors : {t1}/{n} = {100.0*t1/n:.2f}%")
    print(f"       Top-5 set agreement                : {t5}/{n} = {100.0*t5/n:.2f}%")

    # ------------------------------------------------------------- 24 parity audit
    print("\n-- 24. PARITY AUDIT")
    hist_cfg = json.load(open(os.path.join(FULL60, "config.json"), encoding="utf-8"))

    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            kk = f"{p}.{k}" if p else k
            o.update(flat(v, kk)) if isinstance(v, dict) else o.update({kk: v})
        return o

    fa, fb = flat(hist_cfg), flat(cfg)
    EXPECTED_DIFF = {"run_name", "data.target_include", "data.n_target_columns",
                     "model.output_dim"}
    prows, cs, ce, cu = [], 0, 0, 0
    for k in sorted(set(fa) | set(fb)):
        a, b = fa.get(k, "<absent>"), fb.get(k, "<absent>")
        if a == b:
            c = "SAME"; cs += 1
        elif k in EXPECTED_DIFF:
            c = "EXPECTED_DIFFERENCE"; ce += 1
        else:
            c = "UNEXPECTED_DIFFERENCE"; cu += 1
        prows.append({"category": "config", "field": k, "full60": str(a)[:200],
                      "head14": str(b)[:200], "classification": c})
    for label, a, b, c in [
        ("source_module_root", "models/metric_utility_mlp", "models/metric_utility_mlp", "SAME"),
        ("dataset_csv", hist_cfg["data"]["csv_path"], cfg["data"]["csv_path"],
         "SAME" if hist_cfg["data"]["csv_path"] == cfg["data"]["csv_path"] else "UNEXPECTED_DIFFERENCE"),
        ("n_features", 250, len(feat), "SAME" if len(feat) == 250 else "UNEXPECTED_DIFFERENCE"),
        ("feature_order", "full60", "identical" if feat == hist_feat else "DIFFERENT",
         "SAME" if feat == hist_feat else "UNEXPECTED_DIFFERENCE"),
        ("n_targets", 60, len(tgt), "EXPECTED_DIFFERENCE"),
        ("evaluator", "metrics.compute_all", "metrics.compute_all", "SAME"),
        ("eval_topk_list", hist_cfg["evaluation"]["topk_list"], ev["topk_list"],
         "SAME" if hist_cfg["evaluation"]["topk_list"] == ev["topk_list"] else "UNEXPECTED_DIFFERENCE"),
        ("loss_topk", hist_cfg["loss"]["topk"], cfg["loss"]["topk"],
         "SAME" if hist_cfg["loss"]["topk"] == cfg["loss"]["topk"] else "UNEXPECTED_DIFFERENCE"),
    ]:
        prows.append({"category": "contract", "field": label, "full60": str(a),
                      "head14": str(b), "classification": c})
        cs += (c == "SAME"); ce += (c == "EXPECTED_DIFFERENCE"); cu += (c == "UNEXPECTED_DIFFERENCE")
    pd.DataFrame(prows).to_csv(os.path.join(DOCS, "head14_parity_audit.csv"), index=False)
    print(f"       SAME={cs}  EXPECTED_DIFFERENCE={ce}  UNEXPECTED_DIFFERENCE={cu}")
    for r in prows:
        if r["classification"] == "UNEXPECTED_DIFFERENCE":
            print(f"       ! {r['field']}: full60={r['full60']} head14={r['head14']}")
    all_ok &= rec("no unexpected parity differences", cu == 0, f"{cu} unexpected")

    # ------------------------------------------------------------------ summary
    payload = {
        "artifact_path": run_rel,
        "full60_reference": FULL60_REL,
        "checks_total": len(RESULTS),
        "checks_passed": sum(1 for _, s, _ in RESULTS if s == "PASS"),
        "checks_failed": sum(1 for _, s, _ in RESULTS if s == "FAIL"),
        "sha256": hashes,
        "head14_targets": head14,
        "head14_metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in h14_summary.items()},
        "full60_metrics": f60,
        "diagnostic_restricted_full60_metrics": {
            k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r_summary.items()},
        "predictor_top1_agreement": f"{t1}/{n}",
        "predictor_top5_set_agreement": f"{t5}/{n}",
        "parity": {"same": cs, "expected_difference": ce, "unexpected_difference": cu},
        "checks": [{"name": a, "status": b, "detail": c} for a, b, c in RESULTS],
    }
    with open(os.path.join(DOCS, "head14_mlp_validation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print("=" * 78)
    print(f"  checks: {payload['checks_passed']}/{payload['checks_total']} PASS, "
          f"{payload['checks_failed']} FAIL")
    print(f"  VERDICT: {'HEAD-14 ARTIFACT VALIDATED' if all_ok else 'VALIDATION FAILED'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
