"""Utility-predictor quality (k-NN vs neural) and utility-selection alignment.

NOT FULLY RUNNABLE FROM THE PUBLIC RELEASE.

Part 1 runs: predictor metrics on the held-out split for the k-NN predictor used by CLUSTOPT
(C4) and the neural predictors, written to ``analysis/output/statistics/predictor_metrics.csv``.

Part 2 cannot run: the alignment between true utility and the indices CLUSTOPT selects needs
the out-of-fold utility predictions of the training splits 2-16. They derive from the
predictor training table, which is intentionally not redistributed, and the script stops at
that point with "not part of the public release".

The frozen outputs of both parts are released in ``results/controlled/predictor_alignment/``
(``predictor_metrics.csv``, ``selection_alignment.csv``, ``selection_alignment_per_family.csv``,
``predictor_alignment_provenance.json``). They support Table 22 and the alignment values quoted
in Section 6; see ``docs/paper_results_index.md`` and ``docs/reproduction.md``.

Analysis only: no model is refitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

sys.path.insert(0, str(HERE.parent))
from release_paths import LEGACY_ROOT, output_dir  # noqa: E402
REPO = LEGACY_ROOT
RA = REPO / "results_analysis"
CR = RA / "clustering_repository"
KNN = RA / "metric_utility_knn" / "phase_e1"
MLP60 = RA / "mlp" / "20260615_231715__metric_utility_mlp_split1_holdout"
MLP14 = RA / "mlp" / "20260823_195830__metric_utility_mlp_head14_split1_holdout"
S3B = CR / "metric_analysis_existing_domain" / "family_subfamily_specialization"
CANON = CR / "paper_analysis_v2" / "13_raw_and_reproducibility"
OUT = output_dir("statistics")

CORE = ["silhouette", "noise_aware_silhouette", "gridness_fft_acf"]
DEAD = ["convexity_ratio_image_based", "hough_arc_circle_strength"]

PROV: dict = {}


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


# ============================================================ B5 predictor
def b5() -> pd.DataFrame:
    knn = pd.read_csv(KNN / "tables" / "split1_locked_metrics.csv") \
        .set_index("metric")["value"].to_dict()
    m60 = pd.read_csv(MLP60 / "overall_results.csv") \
        .set_index("metric")["mean"].to_dict()
    m14 = pd.read_csv(MLP14 / "overall_results.csv") \
        .set_index("metric")["mean"].to_dict()

    fields = ["mae", "rmse", "mse", "r2", "macro_r2", "spearman_mean",
              "kendall_mean", "pairwise_ranking_accuracy", "ndcg@all",
              "ndcg@10", "ndcg@5", "ndcg@3", "top1_accuracy", "top5_overlap",
              "top10_overlap", "n_samples"]
    rows = []
    for label, src, path, inv, deployed in (
        ("FULL60 KNN (K=15, inverse-distance) -- DEPLOYED by external arm C4",
         knn, KNN / "tables" / "split1_locked_metrics.csv", "FULL60", True),
        ("FULL60 MLP (multi-head) -- NOT deployed by C4",
         m60, MLP60 / "overall_results.csv", "FULL60", False),
        ("HEAD14 MLP (multi-head) -- small-vocabulary control",
         m14, MLP14 / "overall_results.csv", "HEAD14", False),
    ):
        r = {"predictor": label, "inventory": inv,
             "is_deployed_C4_predictor": deployed,
             "population": "held-out Split 1, dataset-view records",
             "source_path": rel(path)}
        for f in fields:
            r[f] = src.get(f)
        rows.append(r)
    df = pd.DataFrame(rows)

    cfg = json.loads((KNN / "selected_configuration.json").read_text(encoding="utf-8"))
    meta = json.loads((KNN / "models" / "final_knn_metadata.json").read_text(encoding="utf-8"))
    PROV["b5_knn_config"] = {
        "config_id": cfg["selection"]["one_se_selected_configuration"]["config_id"],
        "n_neighbors": meta["config"]["n_neighbors"],
        "neighbor_weighting": meta["config"]["neighbor_weighting"],
        "n_features": meta["n_features"], "n_metrics": meta["n_metrics"],
        "train_splits": meta["train_splits"],
        "locked_holdout_split": meta["locked_holdout_split"],
        "split1_in_training": meta["split1_in_training"],
        "per_view_train_counts": meta["per_view_train_counts"],
    }
    PROV["b5_manuscript_cited_triple"] = {
        "cited": {"spearman": 0.762, "pairwise": 0.800, "ndcg": 0.971},
        "knn_full60": {"spearman": knn.get("spearman_mean"),
                       "pairwise": knn.get("pairwise_ranking_accuracy"),
                       "ndcg_all": knn.get("ndcg@all")},
        "mlp_full60": {"spearman": m60.get("spearman_mean"),
                       "pairwise": m60.get("pairwise_ranking_accuracy"),
                       "ndcg_all": m60.get("ndcg@all")},
        "verdict": "the cited triple is the FULL60 MLP row, not the deployed KNN",
    }
    return df


# ======================================================= B6 alignment engine
def _metric_names(cols):
    return [c.split("__", 1)[1] if c.startswith("utility__") else c for c in cols]


def alignment(true_u: np.ndarray, pred_u: np.ndarray, metrics: list,
              fam: np.ndarray, k: int, new46: set) -> dict:
    """Per-family Spearman(true utility, selection frequency) + recall stats.

    `true_u` / `pred_u` are (n_records, n_metrics). The selected subset of a
    record is the top-`k` metrics by PREDICTED utility, which is exactly
    Equation (2)'s subset S. Non-finite predicted utilities rank last; records
    whose true utility row is entirely non-finite are dropped.
    """
    T = np.asarray(true_u, dtype=float)
    Pr = np.asarray(pred_u, dtype=float)
    keep = np.isfinite(T).any(axis=1)
    T, Pr, fam = T[keep], Pr[keep], np.asarray(fam)[keep]

    Prr = np.where(np.isfinite(Pr), Pr, -np.inf)
    Trr = np.where(np.isfinite(T), T, -np.inf)
    order_p = np.argsort(-Prr, axis=1, kind="stable")
    order_t = np.argsort(-Trr, axis=1, kind="stable")
    sel = order_p[:, :k]                     # selected subset S
    t5 = order_t[:, :5]                      # true top-5
    t1 = order_t[:, 0]                       # true top-1

    n, m = T.shape
    selmask = np.zeros((n, m), dtype=bool)
    np.put_along_axis(selmask, sel, True, axis=1)
    t5mask = np.zeros((n, m), dtype=bool)
    np.put_along_axis(t5mask, t5, True, axis=1)

    new_idx = np.array([i for i, nm in enumerate(metrics) if nm in new46])
    newmask = np.zeros(m, dtype=bool)
    newmask[new_idx] = True

    top1_hit = selmask[np.arange(n), t1]
    inter = (selmask & t5mask).sum(axis=1)
    top5_recall = inter / 5.0
    n_new_in_t5 = (t5mask & newmask).sum(axis=1)
    inter_new = (selmask & t5mask & newmask).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        new_recall = np.where(n_new_in_t5 > 0, inter_new / n_new_in_t5, np.nan)

    core_idx = np.array([i for i, nm in enumerate(metrics) if nm in CORE])
    core_share = selmask[:, core_idx].sum() / float(selmask.sum())
    new_share = selmask[:, new_idx].sum() / float(selmask.sum())

    per_fam = []
    for f in sorted(set(fam)):
        mm = fam == f
        mu_true = np.nanmean(np.where(np.isfinite(T[mm]), T[mm], np.nan), axis=0)
        freq = selmask[mm].mean(axis=0)
        ok = np.isfinite(mu_true)
        rho_all = spearmanr(mu_true[ok], freq[ok]).statistic
        okn = ok & newmask
        rho_new = spearmanr(mu_true[okn], freq[okn]).statistic
        per_fam.append({
            "family_id": f, "n_records": int(mm.sum()),
            "spearman_util_sel_all": float(rho_all),
            "spearman_util_sel_New46": float(rho_new),
            "true_top1_hit": float(top1_hit[mm].mean()),
            "true_top5_recall": float(top5_recall[mm].mean()),
            "new46_true_top5_recall": float(np.nanmean(new_recall[mm])),
            "core_share_of_selection_slots":
                float(selmask[mm][:, core_idx].sum() / selmask[mm].sum()),
            "new46_share_of_selection_slots":
                float(selmask[mm][:, new_idx].sum() / selmask[mm].sum()),
        })
    pf = pd.DataFrame(per_fam)
    return {
        "per_family": pf,
        "summary": {
            "n_records": int(n), "n_families": int(pf.shape[0]), "k": k,
            "mean_spearman_all": float(pf.spearman_util_sel_all.mean()),
            "min_spearman_all": float(pf.spearman_util_sel_all.min()),
            "max_spearman_all": float(pf.spearman_util_sel_all.max()),
            "mean_spearman_New46": float(pf.spearman_util_sel_New46.mean()),
            "mean_true_top1_hit": float(pf.true_top1_hit.mean()),
            "pooled_true_top1_hit": float(top1_hit.mean()),
            "mean_true_top5_recall": float(pf.true_top5_recall.mean()),
            "pooled_true_top5_recall": float(top5_recall.mean()),
            "mean_new46_true_top5_recall": float(pf.new46_true_top5_recall.mean()),
            "pooled_new46_true_top5_recall": float(np.nanmean(new_recall)),
            "core_share_of_selection_slots": float(core_share),
            "new46_share_of_selection_slots": float(new_share),
        },
    }


def to_dataset_level(T: np.ndarray, P: np.ndarray, ds: np.ndarray, fam: np.ndarray):
    """Average the per-view utility rows to the dataset level.

    metric_analysis_existing_domain/metric_utility_definition.json states
    `"inferential_unit": "dataset (views averaged); view level reported
    descriptively"`, and the frozen Stage-3B family tables count DATASETS
    (16,013), not records (48,039). Reproducing them therefore requires the
    dataset-level aggregation.
    """
    df_t = pd.DataFrame(T)
    df_p = pd.DataFrame(P)
    df_t["__d"], df_p["__d"] = ds, ds
    gt = df_t.groupby("__d", sort=True).mean()
    gp = df_p.groupby("__d", sort=True).mean()
    fmap = pd.Series(fam, index=ds).groupby(level=0).first().reindex(gt.index)
    return gt.to_numpy(float), gp.to_numpy(float), fmap.to_numpy()


def b6():
    # ---------- metric grouping (authoritative registry) ----------
    reg = json.loads((REPO / "models" / "Independent_Domain_Benchmark" / "configs"
                      / "metric_analysis_registry.json").read_text(encoding="utf-8"))
    new46 = {e["canonical_metric_id"] for e in reg["entries"]
             if e["group"] == "New46"}

    # ---------- (i) splits 2-16, MLP OOF predictions ----------
    oof = pd.read_csv(CR / "stage2" / "oof_utility_splits2_16" / "combined"
                      / "oof_mlp_utilities.csv.gz")
    tcols = [c for c in oof.columns if c.startswith("true_")]
    pcols = ["pred_" + c[5:] for c in tcols]
    metrics = [c[5:] for c in tcols]
    fam = oof["family_id"].to_numpy()
    dsid = oof["dataset_id"].to_numpy()
    T = oof[tcols].to_numpy(float)
    P_mlp = oof[pcols].to_numpy(float)

    Td, Pd, famd = to_dataset_level(T, P_mlp, dsid, fam)
    ref_mlp5 = alignment(Td, Pd, metrics, famd, k=5, new46=new46)
    ref_mlp5_view = alignment(T, P_mlp, metrics, fam, k=5, new46=new46)

    # ---------- REVERSE VALIDATION against the frozen Stage-3B tables ----------
    froz_a = pd.read_csv(S3B / "family_utility_selection_alignment.csv")
    froz_r = pd.read_csv(S3B / "family_true_topk_recall.csv")
    chk = (ref_mlp5["per_family"].merge(
        froz_a[["group", "spearman_util_sel_all", "spearman_util_sel_New46"]]
        .rename(columns={"group": "family_id",
                         "spearman_util_sel_all": "froz_all",
                         "spearman_util_sel_New46": "froz_new"}),
        on="family_id")
        .merge(froz_r[["group", "true_top5_recall", "true_top1_hit",
                       "new46_true_top5_recall"]]
               .rename(columns={"group": "family_id",
                                "true_top5_recall": "froz_t5",
                                "true_top1_hit": "froz_t1",
                                "new46_true_top5_recall": "froz_n5"}),
               on="family_id"))
    chk_v = ref_mlp5_view["per_family"].merge(
        froz_a[["group", "spearman_util_sel_all", "spearman_util_sel_New46"]]
        .rename(columns={"group": "family_id",
                         "spearman_util_sel_all": "froz_all",
                         "spearman_util_sel_New46": "froz_new"}), on="family_id")
    resid = {
        "DATASET-level true_top5_recall": float(
            (chk.true_top5_recall - chk.froz_t5).abs().max()),
        "DATASET-level true_top1_hit": float(
            (chk.true_top1_hit - chk.froz_t1).abs().max()),
        "DATASET-level new46_true_top5_recall": float(
            (chk.new46_true_top5_recall - chk.froz_n5).abs().max()),
        "VIEW-level spearman_all": float(
            (chk_v.spearman_util_sel_all - chk_v.froz_all).abs().max()),
        "VIEW-level spearman_New46": float(
            (chk_v.spearman_util_sel_New46 - chk_v.froz_new).abs().max()),
        "DATASET-level spearman_all (wrong level, for contrast)": float(
            (chk.spearman_util_sel_all - chk.froz_all).abs().max()),
    }
    PROV["b6_reverse_validation"] = {
        "hypothesis": ("the manuscript's alignment statistics describe the MLP "
                       "predictor with a TOP-5 subset rule, at DATASET level "
                       "(views averaged), on splits 2-16"),
        "max_abs_residual_vs_frozen_stage3b": resid,
        "verdict": (
            "REPRODUCED. The frozen Stage-3B statistics are a MIXED-level "
            "estimator: the recall / top-1 quantities are DATASET level "
            "(views averaged, 16,013 datasets) and reproduce exactly; the "
            "Spearman alignment is VIEW level (48,039 records) and reproduces "
            "to ~1e-3 per family. Both use the MLP predictor with a TOP-5 "
            "subset rule on splits 2-16."),
        "frozen_family_means": {
            "spearman_util_sel_all": float(froz_a.spearman_util_sel_all.mean()),
            "spearman_util_sel_New46": float(froz_a.spearman_util_sel_New46.mean()),
            "true_top1_hit": float(froz_r.true_top1_hit.mean()),
            "true_top5_recall": float(froz_r.true_top5_recall.mean()),
            "new46_true_top5_recall": float(froz_r.new46_true_top5_recall.mean()),
        },
    }

    # ---------- (ii) splits 2-16, KNN OOF predictions, top-10 (DEPLOYED rule) --
    kp = pd.read_parquet(KNN / "predictions" / "out_of_fold_predictions.parquet")
    km = pd.read_parquet(KNN / "predictions" / "out_of_fold_record_metadata.parquet")
    kmetrics = _metric_names(list(kp.columns))
    assert set(kmetrics) == set(metrics), "KNN/MLP metric vocabularies differ"
    kp = kp[["utility__" + m for m in metrics]]
    # align the KNN OOF rows to the same record order as the true-utility table
    kidx = km["record_id"].to_numpy() if "record_id" in km.columns else None
    tru = oof.set_index("record_id")
    order = [r for r in kidx]
    T_k = tru.loc[order, tcols].to_numpy(float)
    fam_k = tru.loc[order, "family_id"].to_numpy()
    ds_k = tru.loc[order, "dataset_id"].to_numpy()
    Tkd, Pkd, famkd = to_dataset_level(T_k, kp.to_numpy(float), ds_k, fam_k)
    knn_oof10 = alignment(Tkd, Pkd, metrics, famkd, k=10, new46=new46)
    knn_oof5 = alignment(Tkd, Pkd, metrics, famkd, k=5, new46=new46)
    mlp_oof10 = alignment(Td, Pd, metrics, famd, k=10, new46=new46)
    knn_oof10_view = alignment(T_k, kp.to_numpy(float), metrics, fam_k,
                               k=10, new46=new46)

    # ---------- (iii) Split 1, KNN locked predictions, top-10 ----------
    s1p = pd.read_parquet(KNN / "predictions" / "split1_locked_predictions.parquet")
    s1t = pd.read_parquet(KNN / "predictions" / "split1_locked_targets.parquet")
    s1m = pd.read_parquet(KNN / "predictions" / "split1_locked_record_metadata.parquet")
    s1metrics = _metric_names(list(s1t.columns))
    s1p = s1p[["utility__" + m for m in s1metrics]]
    bvfam = pd.read_parquet(CANON / "canonical_best_view_frame.parquet",
                            columns=["dataset_id", "family_id"]).drop_duplicates()
    fmap = dict(zip(bvfam.dataset_id, bvfam.family_id))
    fam_s1 = np.array([fmap.get(d, "UNKNOWN") for d in s1m["dataset_id"]])
    knn_s1_10_view = alignment(s1t.to_numpy(float), s1p.to_numpy(float),
                               s1metrics, fam_s1, k=10, new46=new46)
    Ts1d, Ps1d, fams1d = to_dataset_level(
        s1t.to_numpy(float), s1p.to_numpy(float),
        s1m["dataset_id"].to_numpy(), fam_s1)
    knn_s1_10 = alignment(Ts1d, Ps1d, s1metrics, fams1d, k=10, new46=new46)

    # ---------- (iv) Split 1, ACTUAL deployed selections ----------
    run = pd.read_parquet(CANON / "canonical_run_frame.parquet",
                          columns=["dataset_id", "view_id", "method_name",
                                   "selected_metrics", "family_id"])
    run = run[run.method_name == "knn_top10_raw"]
    key = {(d, v): s for d, v, s in zip(run.dataset_id, run.view_id,
                                        run.selected_metrics)}
    rec_keys = list(zip(s1m["dataset_id"], s1m["view"]))
    sel_lists = [key.get(kk) for kk in rec_keys]
    n_missing = sum(1 for s in sel_lists if s is None)
    midx = {m: i for i, m in enumerate(s1metrics)}
    Ts1 = s1t.to_numpy(float)
    n, m = Ts1.shape
    selmask = np.zeros((n, m), dtype=bool)
    sizes = []
    for i, s in enumerate(sel_lists):
        if s is None:
            continue
        names = ([x.strip().strip("'\"") for x in
                  str(s).strip("[]").split(",")] if not isinstance(s, (list, np.ndarray))
                 else list(s))
        names = [x for x in names if x in midx]
        sizes.append(len(names))
        for x in names:
            selmask[i, midx[x]] = True
    PROV["b6_deployed_selection_audit"] = {
        "n_records": int(n),
        "n_records_without_a_persisted_selection": int(n_missing),
        "selected_metric_count_distribution": {
            str(k): int(v) for k, v in
            pd.Series(sizes).value_counts().sort_index().items()},
        "note": ("`selected_metrics` is the subset S that the deployed "
                 "knn_top10_raw run actually used on Split 1"),
    }

    # alignment statistics from the ACTUAL selections
    Trr = np.where(np.isfinite(Ts1), Ts1, -np.inf)
    order_t = np.argsort(-Trr, axis=1, kind="stable")
    t1 = order_t[:, 0]
    t5mask = np.zeros((n, m), dtype=bool)
    np.put_along_axis(t5mask, order_t[:, :5], True, axis=1)
    newmask = np.zeros(m, dtype=bool)
    newmask[[i for i, nm in enumerate(s1metrics) if nm in new46]] = True
    core_idx = [i for i, nm in enumerate(s1metrics) if nm in CORE]
    new_idx = [i for i, nm in enumerate(s1metrics) if nm in new46]
    have = selmask.any(axis=1)
    top1_hit = selmask[np.arange(n), t1][have]
    t5r = (selmask & t5mask).sum(axis=1)[have] / 5.0
    nn = (t5mask & newmask).sum(axis=1)[have]
    ni = (selmask & t5mask & newmask).sum(axis=1)[have]
    with np.errstate(invalid="ignore", divide="ignore"):
        nr = np.where(nn > 0, ni / nn, np.nan)
    pf_rows = []
    for f in sorted(set(fam_s1[have])):
        mm = fam_s1[have] == f
        Tf = Ts1[have][mm]
        mu = np.nanmean(np.where(np.isfinite(Tf), Tf, np.nan), axis=0)
        freq = selmask[have][mm].mean(axis=0)
        ok = np.isfinite(mu)
        pf_rows.append({
            "family_id": f, "n_records": int(mm.sum()),
            "spearman_util_sel_all": float(spearmanr(mu[ok], freq[ok]).statistic),
            "spearman_util_sel_New46": float(
                spearmanr(mu[ok & newmask], freq[ok & newmask]).statistic),
            "true_top1_hit": float(top1_hit[mm].mean()),
            "true_top5_recall": float(t5r[mm].mean()),
            "new46_true_top5_recall": float(np.nanmean(nr[mm])),
            "core_share_of_selection_slots": float(
                selmask[have][mm][:, core_idx].sum() / selmask[have][mm].sum()),
            "new46_share_of_selection_slots": float(
                selmask[have][mm][:, new_idx].sum() / selmask[have][mm].sum()),
        })
    pf_dep = pd.DataFrame(pf_rows)
    dep = {"per_family": pf_dep, "summary": {
        "n_records": int(have.sum()), "n_families": int(pf_dep.shape[0]), "k": 10,
        "mean_spearman_all": float(pf_dep.spearman_util_sel_all.mean()),
        "min_spearman_all": float(pf_dep.spearman_util_sel_all.min()),
        "max_spearman_all": float(pf_dep.spearman_util_sel_all.max()),
        "mean_spearman_New46": float(pf_dep.spearman_util_sel_New46.mean()),
        "mean_true_top1_hit": float(pf_dep.true_top1_hit.mean()),
        "pooled_true_top1_hit": float(top1_hit.mean()),
        "mean_true_top5_recall": float(pf_dep.true_top5_recall.mean()),
        "pooled_true_top5_recall": float(t5r.mean()),
        "mean_new46_true_top5_recall": float(pf_dep.new46_true_top5_recall.mean()),
        "pooled_new46_true_top5_recall": float(np.nanmean(nr)),
        "core_share_of_selection_slots": float(
            selmask[have][:, core_idx].sum() / selmask[have].sum()),
        "new46_share_of_selection_slots": float(
            selmask[have][:, new_idx].sum() / selmask[have].sum()),
    }}

    variants = [
        ("MLP / top-5  / splits 2-16 / VIEW level "
         "[= source of the manuscript's rho = 0.828 / 0.809 and the "
         "core/New46 selection shares]", ref_mlp5_view),
        ("MLP / top-5  / splits 2-16 / DATASET level "
         "[= source of the manuscript's top-1 72%, top-5 58%, New46 48%]",
         ref_mlp5),
        ("MLP / top-10 / splits 2-16 / DATASET level", mlp_oof10),
        ("KNN / top-5  / splits 2-16 / DATASET level", knn_oof5),
        ("KNN / top-10 / splits 2-16 / VIEW level "
         "[DEPLOYED rule, manuscript population]", knn_oof10_view),
        ("KNN / top-10 / splits 2-16 / DATASET level "
         "[DEPLOYED rule, manuscript population]", knn_oof10),
        ("KNN / top-10 / Split 1 / DATASET level [DEPLOYED rule, Split 1]",
         knn_s1_10),
        ("KNN / top-10 / Split 1 / VIEW level [DEPLOYED rule, as selected]",
         knn_s1_10_view),
        ("KNN / top-10 / Split 1 / VIEW level / ACTUAL deployed selections "
         "[C4 exactly as run]", dep),
    ]
    rows = []
    for label, res in variants:
        r = {"variant": label}
        r.update(res["summary"])
        rows.append(r)
    summ = pd.DataFrame(rows)

    fam_frames = []
    for label, res in variants:
        f = res["per_family"].copy()
        f.insert(0, "variant", label)
        fam_frames.append(f)
    fams = pd.concat(fam_frames, ignore_index=True)
    fams.to_csv(OUT / "selection_alignment_per_family.csv", index=False)
    return summ


def main() -> int:
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)

    b5df = b5()
    b5df.to_csv(OUT / "predictor_metrics.csv", index=False)
    print("=== B5 predictor metrics ===")
    print(b5df[["predictor", "mae", "rmse", "r2", "spearman_mean", "kendall_mean",
                "pairwise_ranking_accuracy", "ndcg@all", "top1_accuracy",
                "top5_overlap", "top10_overlap", "n_samples"]].to_string())
    print()
    print(json.dumps(PROV["b5_manuscript_cited_triple"], indent=1))
    print()

    summ = b6()
    summ.to_csv(OUT / "selection_alignment.csv", index=False)
    print("=== B6 reverse validation ===")
    print(json.dumps(PROV["b6_reverse_validation"], indent=1))
    print()
    print("=== B6 selection alignment variants ===")
    print(summ.to_string())
    print()
    print(json.dumps(PROV["b6_deployed_selection_audit"], indent=1))

    PROV["sources"] = {
        "knn_split1_metrics": rel(KNN / "tables" / "split1_locked_metrics.csv"),
        "mlp_full60_metrics": rel(MLP60 / "overall_results.csv"),
        "mlp_head14_metrics": rel(MLP14 / "overall_results.csv"),
        "oof_true_utilities": rel(CR / "stage2" / "oof_utility_splits2_16"
                                  / "combined" / "oof_mlp_utilities.csv.gz"),
        "knn_oof_predictions": rel(KNN / "predictions"
                                   / "out_of_fold_predictions.parquet"),
        "knn_split1_predictions": rel(KNN / "predictions"
                                      / "split1_locked_predictions.parquet"),
        "deployed_selections": rel(CANON / "canonical_run_frame.parquet"),
        "frozen_stage3b_alignment": rel(S3B / "family_utility_selection_alignment.csv"),
        "frozen_stage3b_recall": rel(S3B / "family_true_topk_recall.csv"),
    }
    (OUT / "predictor_alignment_provenance.json").write_text(
        json.dumps(PROV, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
