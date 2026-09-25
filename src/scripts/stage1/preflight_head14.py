"""Stage-1B-2 pre-training gates for the Head-14 MLP.

Runs every assertion that must hold BEFORE training starts. Nothing is trained
here. Every check that fails prints loudly and the script exits non-zero, so
training is never launched on a broken contract.

Gates:
  1 target definition   -- exactly the canonical METRIC_HEAD_GROUPS['classic_cvi']
  2 feature contract    -- identical 250 features (names + order) to Full-60,
                           and NO utility__ column leaks in as a feature
  3 split contract      -- Split-1 held out, splits 2-16 pooled, exact counts,
                           no ID overlap, 3 views grouped
  4 config parity       -- effective Head-14 config vs the historical Full-60
                           config, field by field

Outputs:
  docs/internal_reports/stage1/head14_config_parity.csv
  docs/internal_reports/stage1/head14_preflight.json

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/preflight_head14.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FULL60 = os.path.join(REPO, "results_analysis", "mlp",
                      "20260615_231715__metric_utility_mlp_split1_holdout")
DOCS = os.path.join(REPO, "docs", "project_lead_revision", "stage1")

EXPECTED_HEAD14 = [
    "silhouette", "noise_aware_silhouette", "calinski_harabasz", "davies_bouldin",
    "s_dbw", "dbcv", "avg_within_cluster_dispersion", "avg_pca_isotropy",
    "min_centroid_distance", "min_intercluster_distance",
    "cluster_size_balance_entropy", "cluster_size_imbalance_entropy",
    "neighborhood_purity", "neighborhood_purity_multi_k",
]

EXPECTED = {
    "total_datasets": 17068, "total_records": 51204,
    "test_datasets": 1055, "test_records": 3165,
    "pool_datasets": 16013, "pool_records": 48039,
    "train_datasets": 13611, "train_records": 40833,
    "val_datasets": 2402, "val_records": 7206,
}

RESULTS: list[tuple[str, str, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 78)
    print("HEAD-14 PRE-TRAINING GATES")
    print("=" * 78)

    from models.metric_utility_mlp.head_groups import (
        METRIC_HEAD_GROUPS, resolve_target_group,
    )
    from models.metric_utility_mlp.config import RunConfig
    from models.metric_utility_mlp.data_loader import load_unified_dataset
    from models.metric_utility_mlp.splitters import build_holdout_split_folds

    all_ok = True

    # ---------------------------------------------------------- 1 targets
    print("\n-- 1. TARGET DEFINITION")
    head14 = resolve_target_group("head14")
    all_ok &= rec("resolved from METRIC_HEAD_GROUPS['classic_cvi']",
                  head14 == list(METRIC_HEAD_GROUPS["classic_cvi"]),
                  "canonical registry, no duplicate list")
    all_ok &= rec("exactly 14 targets", len(head14) == 14, f"n={len(head14)}")
    all_ok &= rec("all unique", len(set(head14)) == 14)
    all_ok &= rec("exact expected order", head14 == EXPECTED_HEAD14,
                  "matches the project_lead-specified order" if head14 == EXPECTED_HEAD14
                  else f"got {head14}")

    # ---------------------------------------------------------- config
    print("\n-- 2. EFFECTIVE CONFIG CONSTRUCTION")
    cfg = RunConfig()
    cfg.data.target_include = list(head14)
    cfg.data.n_target_columns = len(head14)
    cfg.model.output_dim = len(head14)
    cfg.run_name = "metric_utility_mlp_head14_split1_holdout"
    hist = json.load(open(os.path.join(FULL60, "config.json"), encoding="utf-8"))
    new = json.loads(json.dumps(cfg.to_dict() if hasattr(cfg, "to_dict") else None)) \
        if hasattr(cfg, "to_dict") else None
    if new is None:                       # fall back to the production serialiser
        tmp = os.path.join(DOCS, "_head14_effective_config.json")
        os.makedirs(DOCS, exist_ok=True)
        cfg.to_json(tmp)
        new = json.load(open(tmp, encoding="utf-8"))
    rec("effective Head-14 config constructed", True,
        f"output_dim={new['model']['output_dim']} n_targets={new['data']['n_target_columns']}")

    # ---------------------------------------------------------- 4 parity
    print("\n-- 3. CONFIG PARITY vs HISTORICAL FULL-60")
    EXPECTED_DIFF = {
        "run_name", "data.target_include", "data.n_target_columns",
        "model.output_dim",
    }

    def flatten(d, p=""):
        out = {}
        for k, v in d.items():
            key = f"{p}.{k}" if p else k
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
        return out

    fh, fn = flatten(hist), flatten(new)
    rows, n_same, n_exp, n_unexp = [], 0, 0, 0
    for key in sorted(set(fh) | set(fn)):
        hv, nv = fh.get(key, "<absent>"), fn.get(key, "<absent>")
        if hv == nv:
            cls = "SAME"; n_same += 1
        elif key in EXPECTED_DIFF:
            cls = "EXPECTED_DIFFERENCE"; n_exp += 1
        else:
            cls = "UNEXPECTED_DIFFERENCE"; n_unexp += 1
        rows.append({"field": key, "full60": str(hv)[:200],
                     "head14": str(nv)[:200], "classification": cls})
    os.makedirs(DOCS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DOCS, "head14_config_parity.csv"), index=False)
    print(f"     SAME={n_same}  EXPECTED_DIFFERENCE={n_exp}  UNEXPECTED_DIFFERENCE={n_unexp}")
    for r in rows:
        if r["classification"] == "UNEXPECTED_DIFFERENCE":
            print(f"     ! {r['field']}: full60={r['full60']} head14={r['head14']}")
    all_ok &= rec("no unexpected config differences", n_unexp == 0,
                  f"{n_unexp} unexpected")

    # explicit scientific-hyperparameter equality
    crit = ["model.type", "model.input_dim", "model.hidden_dims", "model.dropout",
            "model.activation", "model.batch_norm", "model.output_activation",
            "model.head_hidden_dim", "model.head_dropout",
            "loss.mode", "loss.mse_weight", "loss.pairwise_rank_weight",
            "loss.topk_mse_weight", "loss.topk", "loss.pairwise_margin",
            "training.optimizer", "training.learning_rate", "training.weight_decay",
            "training.batch_size", "training.epochs", "training.scheduler",
            "training.scheduler_factor", "training.scheduler_patience",
            "training.early_stopping_patience", "training.gradient_clip_norm",
            "training.seed", "training.inner_val_ratio",
            "evaluation.topk_list", "evaluation.ndcg_list", "evaluation.ndcg_all",
            "data.csv_path", "data.n_feature_columns", "data.target_prefix",
            "data.split_group_column"]
    bad = [k for k in crit if fh.get(k) != fn.get(k)]
    all_ok &= rec("all scientific hyperparameters identical", not bad,
                  f"{len(crit)} fields checked" + (f"; DIFFER: {bad}" if bad else ""))
    all_ok &= rec("loss.topk kept at historical value (NOT changed to 5)",
                  fn.get("loss.topk") == fh.get("loss.topk") == 10,
                  f"topk={fn.get('loss.topk')}")

    # ---------------------------------------------------------- 2 features
    print("\n-- 4. FEATURE + TARGET CONTRACT (loads the dataset)")
    loaded = load_unified_dataset(cfg.data)
    feat, targ = loaded.feature_columns, loaded.target_columns
    hist_feat = json.load(open(os.path.join(FULL60, "artifacts", "feature_columns.json"),
                               encoding="utf-8"))
    all_ok &= rec("n_features == 250", len(feat) == 250, f"n={len(feat)}")
    all_ok &= rec("n_targets == 14", len(targ) == 14, f"n={len(targ)}")
    all_ok &= rec("feature list identical to Full-60 (names AND order)",
                  feat == hist_feat, f"{len(hist_feat)} historical features")
    leaked = [c for c in feat if c.startswith(cfg.data.target_prefix)]
    all_ok &= rec("NO utility__ column used as a feature", not leaked,
                  f"{len(leaked)} leaked" if leaked else "0 leaked")
    all_ok &= rec("target columns are the canonical 14 in order",
                  targ == [f"{cfg.data.target_prefix}{m}" for m in head14])
    all_ok &= rec("all 14 targets present in the unified table",
                  all(f"{cfg.data.target_prefix}{m}" in loaded.df.columns for m in head14))

    # ---------------------------------------------------------- 3 split
    print("\n-- 5. SPLIT CONTRACT")
    from models.metric_utility_mlp.run_train_split_holdout import DEFAULT_SPLIT_ASSIGNMENTS_CSV
    folds = build_holdout_split_folds(
        df=loaded.df, group_column=loaded.group_column,
        split_assignments_csv=DEFAULT_SPLIT_ASSIGNMENTS_CSV,
        test_split_id=1, train_split_ids=None,
        inner_val_ratio=cfg.training.inner_val_ratio, seed=cfg.training.seed,
        group_summary_columns=[],
    )
    all_ok &= rec("exactly one outer fold", len(folds) == 1)
    fs = folds[0]
    g = loaded.df[loaded.group_column].to_numpy()
    tr, va, te = set(g[fs.train_idx]), set(g[fs.val_idx]), set(g[fs.test_idx])
    counts = {
        "total_records": len(loaded.df), "total_datasets": len(set(g)),
        "train_records": len(fs.train_idx), "train_datasets": len(tr),
        "val_records": len(fs.val_idx), "val_datasets": len(va),
        "test_records": len(fs.test_idx), "test_datasets": len(te),
        "pool_records": len(fs.train_idx) + len(fs.val_idx),
        "pool_datasets": len(tr | va),
    }
    for k, v in EXPECTED.items():
        all_ok &= rec(f"{k} == {v:,}", counts[k] == v, f"got {counts[k]:,}")

    sh = fs.summary["split_holdout"]
    all_ok &= rec("test_split_id == 1", sh["test_split_id"] == 1)
    all_ok &= rec("train_split_ids == 2..16", sorted(sh["train_split_ids"]) == list(range(2, 17)))
    all_ok &= rec("Split 16 present in pool", 16 in sh["train_split_ids"])
    all_ok &= rec("Split 1 absent from pool", 1 not in sh["train_split_ids"])
    all_ok &= rec("train n val == empty", not (tr & va), f"{len(tr & va)} overlap")
    all_ok &= rec("train n test == empty", not (tr & te), f"{len(tr & te)} overlap")
    all_ok &= rec("val n test == empty", not (va & te), f"{len(va & te)} overlap")

    smap = pd.read_csv(DEFAULT_SPLIT_ASSIGNMENTS_CSV, usecols=["dataset_id", "split_id"])
    s1 = set(smap.loc[smap.split_id == 1, "dataset_id"])
    all_ok &= rec("test set == Split 1 exactly", te == s1,
                  f"{len(te)} vs {len(s1)}")
    all_ok &= rec("no Split-1 dataset in train or val", not ((tr | va) & s1),
                  f"{len((tr | va) & s1)} leaked")

    vm = loaded.df.groupby(loaded.group_column)["view_mode"].nunique()
    all_ok &= rec("every dataset has exactly 3 views", bool((vm == 3).all()),
                  f"{int((vm == 3).sum()):,}/{len(vm):,}")
    for name, idx in (("train", fs.train_idx), ("val", fs.val_idx), ("test", fs.test_idx)):
        sub = loaded.df.iloc[idx]
        per = sub.groupby(loaded.group_column)["view_mode"].nunique()
        all_ok &= rec(f"{name}: all datasets carry all 3 views", bool((per == 3).all()),
                      f"{int((per == 3).sum()):,}/{len(per):,}")

    # ---------------------------------------------------------- summary
    payload = {
        "gates_total": len(RESULTS),
        "gates_passed": sum(1 for _, s, _ in RESULTS if s == "PASS"),
        "gates_failed": sum(1 for _, s, _ in RESULTS if s == "FAIL"),
        "counts": counts,
        "head14_targets": head14,
        "config_parity": {"same": n_same, "expected_difference": n_exp,
                          "unexpected_difference": n_unexp},
        "n_features": len(feat), "n_targets": len(targ),
        "feature_list_matches_full60": feat == hist_feat,
    }
    with open(os.path.join(DOCS, "head14_preflight.json"), "w", encoding="utf-8") as fh_:
        json.dump(payload, fh_, indent=2)

    print()
    print("=" * 78)
    print(f"  gates: {payload['gates_passed']}/{payload['gates_total']} PASS, "
          f"{payload['gates_failed']} FAIL")
    print(f"  VERDICT: {'CLEARED FOR TRAINING' if all_ok else 'DO NOT TRAIN'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
