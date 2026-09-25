"""Stage 2A-3B result package, per-dataset traceability records, integrity gate.

Reads only artifacts produced by ``run_model_selection.py`` plus the read-only
Stage-2A-3B0 outer folds. Split 1 is never opened: the outer folds cover splits
2..16 only and every check below re-asserts that.

Outputs
  results_analysis/clustopt_policy_predictor/stage2a3b_model_selection/
  <dataset_dir>/experiments/split_XX_policy_predictor_nested_cv/result.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E402,E501
    stats as S,
)
from models.ClustOpt_Policy_Predictor.data import target_schema as TS  # noqa: E402
from models.ClustOpt_Policy_Predictor.training import feature_sets as FSET  # noqa: E402,E501

OUT_REL = "results_analysis/clustopt_policy_predictor/stage2a3b_model_selection"
NESTED_REL = "results_analysis/clustopt_policy_predictor/stage2a3b0_nested_crossfit"
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
DATASET_DIR_TEMPLATE = "split_{split:02d}_policy_predictor_nested_cv"
RESULT_SCHEMA_VERSION = "stage2a3b.result.v1"

OUTER_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
FORBIDDEN_SPLIT = 1
N_POLICIES = 20
TOL = 1e-9

# Feature-name leakage guard: explicit semantic sets, not substring matching.
# ``ari``/``regret`` as substrings hit legitimate geometric meta-features such as
# G_linearity_pca_ratio, so membership is tested on whole tokens only.
FORBIDDEN_TOKENS = {"ari", "adjusted_rand", "adjustedrand", "regret", "nmi",
                    "ami", "label", "labels", "y_true", "ground_truth",
                    "policy_ari", "policy_regret", "oracle", "best_policy"}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _tokens(name: str) -> set:
    out, cur = set(), ""
    for ch in name.lower():
        if ch.isalnum():
            cur += ch
        else:
            if cur:
                out.add(cur)
            cur = ""
    if cur:
        out.add(cur)
    return out


def read_predictions(out_root: Path, config_id: str, s: int) -> pd.DataFrame:
    p = out_root / "outer_fold_predictions" / config_id / f"fold_{s:02d}_predictions.csv.gz"
    return pd.read_csv(_ext(p))


def read_fold_test(nested: Path, s: int) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Test identifiers + ARI matrix + (meta, util) column names for fold ``s``."""
    d = nested / "outer_folds" / f"outer_{s:02d}"
    ident = pd.read_csv(_ext(d / "test_identifiers.csv.gz"))
    ari = pd.read_csv(_ext(d / "test_policy_ari_targets.csv.gz"))
    meta_cols = pd.read_csv(_ext(d / "test_meta_features.csv.gz"), nrows=0).columns
    util_cols = pd.read_csv(_ext(d / "test_single_oof_utility_features.csv.gz"),
                            nrows=0).columns
    return ident, ari.to_numpy(np.float64), list(meta_cols) + list(util_cols)


def dataset_bestview(dataset_ids: np.ndarray, vals: np.ndarray) -> pd.Series:
    return pd.DataFrame({"d": dataset_ids, "a": vals}).groupby("d")["a"].max()


# ---------------------------------------------------------------------------
# per-fold assembly for one configuration
# ---------------------------------------------------------------------------
def assemble_fold(out_root: Path, nested: Path, config_id: str, s: int,
                  fold_meta: pd.Series) -> Dict[str, Any]:
    """Row-aligned predictions, ARI matrix and the two baselines for one fold."""
    pred = read_predictions(out_root, config_id, s)
    ident, ari, feat_cols = read_fold_test(nested, s)

    if len(pred) != len(ident):
        raise AssertionError(f"row count mismatch fold {s}: {len(pred)} vs {len(ident)}")
    same = (pred["dataset_id"].to_numpy() == ident["dataset_id"].to_numpy()).all() and \
           (pred["view_id"].to_numpy() == ident["view_id"].to_numpy()).all()
    if not same:
        raise AssertionError(f"row alignment mismatch on fold {s}")

    ds = pred["dataset_id"].to_numpy()
    sel_idx = pred["selected_policy_index"].to_numpy(int)
    pids = TS.policy_order()

    b_idx = int(fold_meta["global_best_single_index"]) \
        if "global_best_single_index" in fold_meta else \
        pids.index(str(fold_meta["global_best_single_policy"]))
    b_global_row = ari[:, b_idx]

    pv = fold_meta["per_view_policies"]
    pv = json.loads(pv.replace("'", '"')) if isinstance(pv, str) else dict(pv)
    b_view_row = np.empty(len(pred), dtype=np.float64)
    for v, pid in pv.items():
        m = (pred["view_id"] == v).to_numpy()
        b_view_row[m] = ari[m][:, pids.index(pid)]

    return {"pred": pred, "ari": ari, "ds": ds, "sel_idx": sel_idx,
            "b_global_row": b_global_row, "b_view_row": b_view_row,
            "per_view_policies": pv, "global_policy": pids[b_idx],
            "global_index": b_idx, "feat_cols": feat_cols, "ident": ident}


def dataset_level_frame(F: Dict[str, Any], s: int) -> pd.DataFrame:
    """Dataset-level BestView values for selected / baselines / oracle."""
    ds = F["ds"]
    sel = F["pred"]["selected_ari"].to_numpy(np.float64)
    orc = F["ari"].max(axis=1)
    out = pd.DataFrame({
        "selected": dataset_bestview(ds, sel),
        "b_global": dataset_bestview(ds, F["b_global_row"]),
        "b_view": dataset_bestview(ds, F["b_view_row"]),
        "oracle": dataset_bestview(ds, orc),
    })
    out.index.name = "dataset_id"
    return out.reset_index().assign(outer_split=s)


# ---------------------------------------------------------------------------
# per-dataset traceability records
# ---------------------------------------------------------------------------
def write_dataset_records(F: Dict[str, Any], s: int, dirs: Dict[str, str],
                          selected_cfg: Dict[str, Any], commit: str,
                          overwrite: bool) -> Tuple[int, int]:
    pred, ari = F["pred"], F["ari"]
    pids = TS.policy_order()
    pcols = [f"pred_{j:02d}" for j in range(N_POLICIES)]
    P = pred[pcols].to_numpy(np.float64)
    written = skipped = 0

    for dsid, grp in pred.groupby("dataset_id", sort=False):
        ddir = dirs.get(dsid)
        if ddir is None:
            raise AssertionError(f"no dataset_dir for {dsid}")
        out_dir = Path(ddir) / "experiments" / DATASET_DIR_TEMPLATE.format(split=s)
        target = out_dir / "result.json"
        if not overwrite and path_exists(target):
            skipped += 1
            continue

        views: Dict[str, Any] = {}
        for i, r in zip(grp.index, grp.itertuples(index=False)):
            k = int(r.selected_policy_index)
            views[r.view_id] = {
                "selected_policy_index": k,
                "selected_policy_id": pids[k],
                "selected_ari": float(r.selected_ari),
                "view_oracle_ari": float(r.view_oracle_ari),
                "view_regret": float(r.view_regret),
                "predicted_scores": [float(x) for x in P[pred.index.get_loc(i)]],
                "policy_ari": [float(x) for x in ari[pred.index.get_loc(i)]],
            }
        bv = max(v["selected_ari"] for v in views.values())
        bo = max(v["view_oracle_ari"] for v in views.values())
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "identity": {"dataset_id": dsid, "split_id": int(s),
                         "stage": "stage2a3b_policy_predictor_model_selection",
                         "evaluation": "nested_cross_fitted_outer_fold"},
            "configuration": {
                "config_id": selected_cfg["config_id"],
                "feature_set": selected_cfg["feature_set"],
                "target": selected_cfg["target"],
                "view_arch": selected_cfg["view_arch"],
                "model_family": selected_cfg["model_family"],
                "input_dim": selected_cfg["input_dim"],
                "output_dim": selected_cfg["output_dim"],
                "selection_rule": selected_cfg["selection_rule"],
                "seed": selected_cfg["seed"],
            },
            "views": views,
            "dataset_level": {"best_view_selected_ari": bv,
                              "best_view_oracle_ari": bo,
                              "best_view_regret": bo - bv},
            "provenance": {
                "policy_order_hash": selected_cfg["policy_order_hash"],
                "utility_source_train": "stage2a3b0 pair-exclusion (S \\ {s,t})",
                "utility_source_test": "stage2a1 single-exclusion (S \\ {s})",
                "split_1_used": False,
                "source_commit": commit,
            },
        }
        ensure_dir(out_dir)
        write_json_atomic(target, payload)
        written += 1
    return written, skipped


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def paired_row(name: str, a: np.ndarray, b: np.ndarray, level: str,
               label_a: str, label_b: str) -> Dict[str, Any]:
    d = np.asarray(a, float) - np.asarray(b, float)
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    row = {"comparison": name, "level": level, "arm_a": label_a, "arm_b": label_b,
           "n": int(len(d)), "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
           "mean_delta": float(pt), "ci_lo": float(lo), "ci_hi": float(hi),
           "wins_a": int((d > TOL).sum()), "wins_b": int((d < -TOL).sum()),
           "ties": int((np.abs(d) <= TOL).sum()),
           "cohens_dz": float(S.cohens_dz(d))}
    try:
        from scipy import stats as sst
        nz = d[np.abs(d) > TOL]
        row["wilcoxon_p"] = float(sst.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    except Exception:                                               # noqa: BLE001
        row["wilcoxon_p"] = float("nan")
    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--skip-dataset-records", action="store_true")
    p.add_argument("--overwrite-records", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out_root = repo / OUT_REL
    nested = repo / NESTED_REL
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t_all = time.time()

    print("=" * 78)
    print("STAGE 2A-3B  --  RESULT PACKAGE + INTEGRITY GATE")
    print("=" * 78, flush=True)

    summary = pd.read_csv(_ext(out_root / "all_configuration_summary.csv"))
    folds = pd.read_csv(_ext(out_root / "all_configuration_fold_metrics.csv"))
    sel_cfg = json.loads((out_root / "selected_configuration.json").read_text("utf-8"))
    sel_id = sel_cfg["config_id"]
    assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
    dirs = dict(zip(assign["dataset_id"], assign["dataset_dir"]))

    checks: List[Dict[str, Any]] = []

    def chk(cid: str, desc: str, ok: bool, detail: Any = "") -> None:
        checks.append({"check_id": cid, "description": desc,
                       "status": "PASS" if ok else "FAIL", "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid:<5} {desc}"
              + (f"  ({detail})" if detail else ""), flush=True)

    # ---- assemble the selected configuration across all 15 outer folds -------
    print("\n-- assembling selected configuration across outer folds --", flush=True)
    ds_frames, per_fold, feat_cols_seen = [], [], None
    fold_store: Dict[int, Dict[str, Any]] = {}
    for s in OUTER_SPLITS:
        fm = folds[(folds["config_id"] == sel_id) & (folds["outer_split"] == s)]
        if len(fm) != 1:
            raise AssertionError(f"expected 1 fold row, got {len(fm)} for split {s}")
        F = assemble_fold(out_root, nested, sel_id, s, fm.iloc[0])
        fold_store[s] = F
        feat_cols_seen = F["feat_cols"]
        ds_frames.append(dataset_level_frame(F, s))
        per_fold.append({"outer_split": s, **fm.iloc[0].to_dict()})
        print(f"   fold {s:02d}: {len(F['pred'])} rows, "
              f"{F['pred']['dataset_id'].nunique()} datasets", flush=True)

    D = pd.concat(ds_frames, ignore_index=True)
    FM = pd.DataFrame(per_fold)

    # ---- §39 artifacts -------------------------------------------------------
    print("\n-- writing package artifacts --", flush=True)

    reg = summary[["config_id", "feature_set", "target", "view_arch",
                   "model_family", "n_folds"]].copy()
    phase_of = {}
    for _, r in summary.iterrows():
        cid = r["config_id"]
        if r["model_family"] == "EXTRATREES" and r["view_arch"] == "UNIFIED":
            phase_of[cid] = "phase1"
        elif r["view_arch"] == "SEPARATE":
            phase_of[cid] = "phase2"
        else:
            phase_of[cid] = "phase3"
    phase_of[sel_id] = "phase1+2+3 (carried)"
    reg["phase"] = reg["config_id"].map(phase_of)
    reg["input_dim"] = reg.apply(
        lambda r: FSET.expected_dim(r["feature_set"], r["view_arch"]), axis=1)
    reg["is_selected"] = reg["config_id"] == sel_id
    hashes = {}
    for cid in reg["config_id"]:
        mp = out_root / "outer_fold_predictions" / cid / "fold_02_manifest.json"
        hashes[cid] = json.loads(mp.read_text("utf-8")).get("config_hash", "")
    reg["config_hash"] = reg["config_id"].map(hashes)
    write_csv_atomic(out_root / "configuration_registry.csv", reg)

    base = FM[["outer_split", "b_global", "b_view", "global_best_single_policy",
               "per_view_policies", "n_test_datasets"]].copy()
    base["b_view_minus_b_global"] = base["b_view"] - base["b_global"]
    write_csv_atomic(out_root / "outer_fold_baselines.csv", base)

    orc = FM[["outer_split", "oracle_bestview", "b_global", "b_view",
              "bestview_selector_ari"]].copy()
    orc["headroom_over_b_global"] = orc["oracle_bestview"] - orc["b_global"]
    orc["headroom_over_b_view"] = orc["oracle_bestview"] - orc["b_view"]
    orc["headroom_remaining"] = orc["oracle_bestview"] - orc["bestview_selector_ari"]
    write_csv_atomic(out_root / "outer_fold_oracles.csv", orc)

    sel_flat = {k: v for k, v in sel_cfg.items()
                if not isinstance(v, (dict, list))}
    sel_flat.update({f"macro__{k}": v for k, v in sel_cfg["macro"].items()
                     if not isinstance(v, (dict, list))})
    sel_flat.update({f"strong_go__{k}": v for k, v in sel_cfg["strong_go"].items()})
    write_csv_atomic(out_root / "selected_configuration.csv",
                     pd.DataFrame([sel_flat]))

    for lab, col in (("global", "b_global"), ("view", "b_view")):
        cmp = FM[["outer_split", "bestview_selector_ari", col]].copy()
        cmp["delta"] = cmp["bestview_selector_ari"] - cmp[col]
        cmp["beats"] = cmp["delta"] > TOL
        dd = D.groupby("outer_split").apply(
            lambda g: pd.Series({
                "n_datasets": len(g),
                "datasets_selected_better": int((g["selected"] > g[col] + TOL).sum()),
                "datasets_baseline_better": int((g["selected"] < g[col] - TOL).sum()),
                "datasets_tied": int((g["selected"] - g[col]).abs().le(TOL).sum()),
            }), include_groups=False).reset_index()
        write_csv_atomic(out_root / f"selected_vs_{lab}_baseline.csv",
                         cmp.merge(dd, on="outer_split"))

    vrows = []
    for _, r in FM.iterrows():
        for v in FSET.VIEW_IDS:
            vrows.append({
                "outer_split": int(r["outer_split"]), "view_id": v,
                "mean_selected_ari": r[f"{v}__mean_selected_ari"],
                "mean_regret": r[f"{v}__mean_regret"],
                "within_0_001": r[f"{v}__within_0_001"],
                "within_0_005": r[f"{v}__within_0_005"],
                "within_0_01": r[f"{v}__within_0_01"],
            })
    VD = pd.DataFrame(vrows)
    write_csv_atomic(out_root / "view_level_diagnostics.csv", VD)

    rec = FM[["outer_split", "bestview_selector_ari", "b_global", "b_view",
              "oracle_bestview", "recovery_global", "recovery_view",
              "n_test_datasets"]].copy()
    macro = {"outer_split": "MACRO",
             **{c: rec[c].mean() for c in rec.columns if c != "outer_split"}}
    write_csv_atomic(out_root / "recovery_summary.csv",
                     pd.concat([rec, pd.DataFrame([macro])], ignore_index=True))

    # ---- statistical comparisons --------------------------------------------
    print("\n-- statistical comparisons --", flush=True)
    stat_rows: List[Dict[str, Any]] = []
    stat_rows.append(paired_row("selected_vs_b_global", D["selected"], D["b_global"],
                                "dataset", sel_id, "B_global"))
    stat_rows.append(paired_row("selected_vs_b_view", D["selected"], D["b_view"],
                                "dataset", sel_id, "B_view"))
    stat_rows.append(paired_row("oracle_vs_selected", D["oracle"], D["selected"],
                                "dataset", "Oracle", sel_id))
    stat_rows.append(paired_row("selected_vs_b_global", FM["bestview_selector_ari"],
                                FM["b_global"], "fold", sel_id, "B_global"))
    stat_rows.append(paired_row("selected_vs_b_view", FM["bestview_selector_ari"],
                                FM["b_view"], "fold", sel_id, "B_view"))

    def config_dataset_bv(cid: str) -> pd.Series:
        parts = []
        for s in OUTER_SPLITS:
            pr = read_predictions(out_root, cid, s)
            parts.append(dataset_bestview(pr["dataset_id"].to_numpy(),
                                          pr["selected_ari"].to_numpy(np.float64)))
        return pd.concat(parts)

    contrasts = [
        ("family__extratrees_vs_ridge", sel_id,
         "RIDGE__UNIFIED__META_UTILITIES__CENTERED"),
        ("family__extratrees_vs_mlp", sel_id,
         "MLP__UNIFIED__META_UTILITIES__CENTERED"),
        ("view_arch__unified_vs_separate", sel_id,
         "EXTRATREES__SEPARATE__META_UTILITIES__CENTERED"),
        ("features__meta_utilities_vs_meta", sel_id,
         "EXTRATREES__UNIFIED__META__CENTERED"),
        ("features__utilities_vs_meta", "EXTRATREES__UNIFIED__UTILITIES__CENTERED",
         "EXTRATREES__UNIFIED__META__CENTERED"),
        ("features__meta_utilities_vs_utilities", sel_id,
         "EXTRATREES__UNIFIED__UTILITIES__CENTERED"),
    ]
    bv_cache: Dict[str, pd.Series] = {}
    for name, ca, cb in contrasts:
        for c in (ca, cb):
            if c not in bv_cache:
                bv_cache[c] = config_dataset_bv(c)
        a, b = bv_cache[ca].sort_index(), bv_cache[cb].sort_index()
        if not a.index.equals(b.index):
            raise AssertionError(f"dataset index mismatch for {name}")
        stat_rows.append(paired_row(name, a.to_numpy(), b.to_numpy(),
                                    "dataset", ca, cb))
        fa = summary.set_index("config_id").loc[ca]
        fb = summary.set_index("config_id").loc[cb]
        ffa = folds[folds["config_id"] == ca].sort_values("outer_split")
        ffb = folds[folds["config_id"] == cb].sort_values("outer_split")
        stat_rows.append(paired_row(
            name, ffa["bestview_selector_ari"].to_numpy(),
            ffb["bestview_selector_ari"].to_numpy(), "fold", ca, cb))
        del fa, fb

    ST = pd.DataFrame(stat_rows)

    # Holm families are declared here rather than pooled: the two baseline
    # comparisons are this stage's primary hypotheses and must not be penalised
    # by the six exploratory design contrasts. Fold-level rows (n=15) duplicate
    # the dataset-level evidence and are reported descriptively only.
    def _family(r: pd.Series) -> str:
        if r["level"] == "fold":
            return "fold_level_descriptive"
        if r["comparison"] in ("selected_vs_b_global", "selected_vs_b_view"):
            return "primary_vs_baselines"
        if r["comparison"] == "oracle_vs_selected":
            return "headroom_descriptive"
        return "design_contrasts"

    ST["hypothesis_family"] = ST.apply(_family, axis=1)
    ST = S.holm_correct(ST, p_col="wilcoxon_p")
    write_csv_atomic(out_root / "statistical_comparisons.csv", ST)
    for _, r in ST[ST["level"] == "dataset"].iterrows():
        print(f"   {r['comparison']:<42} d={r['mean_delta']:+.5f} "
              f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}] p={r['wilcoxon_p']:.3g}",
              flush=True)

    # ---- per-dataset traceability records -----------------------------------
    n_written = n_skipped = 0
    if not args.skip_dataset_records:
        print("\n-- per-dataset result.json --", flush=True)
        for s in OUTER_SPLITS:
            t0 = time.time()
            w, k = write_dataset_records(fold_store[s], s, dirs, sel_cfg, commit,
                                         args.overwrite_records)
            n_written += w
            n_skipped += k
            print(f"   split {s:02d}: wrote {w}, skipped {k} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    # ---- §40 integrity gate --------------------------------------------------
    print("\n" + "=" * 78)
    print("INTEGRITY GATE")
    print("=" * 78, flush=True)

    all_ident_splits = set()
    for s in OUTER_SPLITS:
        all_ident_splits |= set(fold_store[s]["pred"]["split_id"].unique().tolist())
    chk("I01", "Split 1 absent from every prediction identifier",
        FORBIDDEN_SPLIT not in all_ident_splits, sorted(all_ident_splits))
    chk("I02", "outer folds are exactly splits 2..16",
        sorted(FM["outer_split"].tolist()) == list(OUTER_SPLITS))
    chk("I03", "15 outer folds", len(FM) == 15, len(FM))
    chk("I04", "12 configurations registered", len(reg) == 12, len(reg))
    chk("I05", "every configuration has 15 folds",
        bool((summary["n_folds"] == 15).all()))
    chk("I06", "180 configuration-fold evaluations", len(folds) == 180, len(folds))
    chk("I07", "selected configuration present in registry",
        bool(reg["is_selected"].sum() == 1))
    chk("I08", "config hashes unique",
        reg["config_hash"].nunique() == len(reg))
    chk("I09", "policy order hash matches frozen schema",
        sel_cfg["policy_order_hash"] == TS.target_schema_hash(),
        sel_cfg["policy_order_hash"][:16])
    chk("I10", "20 policies in schema", len(TS.policy_order()) == 20)

    okA = okB = okC = okD = okE = okF = True
    n_rows = 0
    for s in OUTER_SPLITS:
        F = fold_store[s]
        pr, ari = F["pred"], F["ari"]
        n_rows += len(pr)
        idx = np.arange(len(pr))
        sel = F["sel_idx"]
        okA &= bool(np.allclose(pr["selected_ari"].to_numpy(np.float64),
                                ari[idx, sel], atol=1e-12))
        okB &= bool(np.allclose(pr["view_oracle_ari"].to_numpy(np.float64),
                                ari.max(axis=1), atol=1e-12))
        okC &= bool(np.allclose(
            pr["view_regret"].to_numpy(np.float64),
            pr["view_oracle_ari"].to_numpy(np.float64)
            - pr["selected_ari"].to_numpy(np.float64), atol=1e-12))
        okD &= bool(((sel >= 0) & (sel < N_POLICIES)).all())
        okE &= bool((pr["view_regret"].to_numpy(np.float64) >= -1e-12).all())
        okF &= bool(pr[[f"pred_{j:02d}" for j in range(N_POLICIES)]]
                    .notna().all().all())
    chk("I11", "selected_ari == ARI[row, selected_index] on every row", okA)
    chk("I12", "view_oracle_ari == rowwise max over 20 policies", okB)
    chk("I13", "view_regret == oracle - selected", okC)
    chk("I14", "selected policy index within [0,20)", okD)
    chk("I15", "view_regret non-negative everywhere", okE)
    chk("I16", "no NaN in any predicted score", okF)

    exp_rows = int(FM["n_test_datasets"].sum() * 3)
    chk("I17", "prediction rows == 3 x test datasets", n_rows == exp_rows,
        f"{n_rows} vs {exp_rows}")
    views_per_ds = D.shape[0]
    chk("I18", "each dataset contributes exactly 3 view rows",
        n_rows == views_per_ds * 3, f"{n_rows} / {views_per_ds}")

    test_ids = {s: set(fold_store[s]["pred"]["dataset_id"].unique())
                for s in OUTER_SPLITS}
    overlap = 0
    ks = list(OUTER_SPLITS)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            overlap += len(test_ids[ks[i]] & test_ids[ks[j]])
    chk("I19", "no dataset appears in two outer test folds", overlap == 0, overlap)

    union = set().union(*test_ids.values())
    expected_union = set(assign.loc[assign["split_id"] != FORBIDDEN_SPLIT,
                                    "dataset_id"])
    chk("I20", "test union == all splits 2..16 datasets",
        union == expected_union, f"{len(union)} vs {len(expected_union)}")
    for s in OUTER_SPLITS:
        exp = set(assign.loc[assign["split_id"] == s, "dataset_id"])
        if test_ids[s] != exp:
            chk("I21", f"fold {s} test ids match split assignment", False,
                f"{len(test_ids[s])} vs {len(exp)}")
            break
    else:
        chk("I21", "every fold's test ids match its split assignment", True)

    split1_ids = set(assign.loc[assign["split_id"] == FORBIDDEN_SPLIT, "dataset_id"])
    chk("I22", "no Split-1 dataset id anywhere in the package",
        len(union & split1_ids) == 0, len(union & split1_ids))

    bad_tok = sorted({c for c in (feat_cols_seen or [])
                      if _tokens(c) & FORBIDDEN_TOKENS})
    chk("I23", "no target-derived token in any feature column name",
        not bad_tok, bad_tok[:5])
    chk("I24", "feature block width == 370 continuous",
        len(feat_cols_seen or []) == 370, len(feat_cols_seen or []))
    chk("I25", "selected input_dim == expected 373",
        sel_cfg["input_dim"] == FSET.expected_dim(sel_cfg["feature_set"],
                                                  sel_cfg["view_arch"]),
        sel_cfg["input_dim"])

    # baselines were chosen on TRAIN; if they had been fitted on TEST they would
    # coincide with the test-optimal single policy in all 15 folds.
    disagree = 0
    for s in OUTER_SPLITS:
        F = fold_store[s]
        te_best = int(np.argmax(
            pd.DataFrame({"d": F["ds"]}).assign(**{
                str(j): F["ari"][:, j] for j in range(N_POLICIES)})
            .groupby("d").max().mean().to_numpy()))
        if te_best != F["global_index"]:
            disagree += 1
    chk("I26", "train-chosen B_global differs from test-optimal in >=1 fold "
        "(proves it is not test-fitted)", disagree >= 1, f"{disagree}/15")

    rg = (FM["bestview_selector_ari"] - FM["b_global"]) / \
         (FM["oracle_bestview"] - FM["b_global"])
    rv = (FM["bestview_selector_ari"] - FM["b_view"]) / \
         (FM["oracle_bestview"] - FM["b_view"])
    chk("I27", "recovery_global recomputes from stored components",
        bool(np.allclose(rg.to_numpy(), FM["recovery_global"].to_numpy(), atol=1e-9)))
    chk("I28", "recovery_view recomputes from stored components",
        bool(np.allclose(rv.to_numpy(), FM["recovery_view"].to_numpy(), atol=1e-9)))
    chk("I29", "macro == unweighted mean over the 15 folds",
        bool(abs(FM["bestview_selector_ari"].mean()
                 - sel_cfg["macro"]["macro_bestview"]) < 1e-12))
    chk("I30", "oracle >= selected in every fold",
        bool((FM["oracle_bestview"] >= FM["bestview_selector_ari"] - TOL).all()))

    CH = pd.DataFrame(checks)
    write_csv_atomic(out_root / "integrity_checks.csv", CH)
    n_fail = int((CH["status"] == "FAIL").sum())

    # ---- summary -------------------------------------------------------------
    payload = {
        "stage": "stage2a3b_policy_predictor_model_selection",
        "selected_configuration": sel_cfg["config_id"],
        "selected_detail": {k: sel_cfg[k] for k in
                            ("feature_set", "target", "view_arch", "model_family",
                             "input_dim", "output_dim", "selection_rule", "seed")},
        "macro": sel_cfg["macro"],
        "strong_go": sel_cfg["strong_go"],
        "n_configurations": int(len(reg)),
        "n_outer_folds": 15,
        "n_configuration_fold_evaluations": int(len(folds)),
        "n_datasets_evaluated": int(len(D)),
        "n_view_rows": int(n_rows),
        "split_1_read": False,
        "dataset_records_written": n_written,
        "dataset_records_skipped": n_skipped,
        "dataset_record_dir_template": DATASET_DIR_TEMPLATE,
        "integrity": {"total": len(CH), "passed": int(len(CH) - n_fail),
                      "failed": n_fail},
        "source_commit": commit,
        "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out_root / "stage2a3b_summary.json", payload)

    print("\n" + "=" * 78)
    print(f"INTEGRITY {len(CH) - n_fail}/{len(CH)} PASS"
          + (f"  --  {n_fail} FAILED" if n_fail else ""))
    print(f"package written to {out_root}")
    print(f"total {time.time() - t_all:.1f}s")
    print("=" * 78, flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
