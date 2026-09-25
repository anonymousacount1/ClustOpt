"""External benchmark: paired statistics at full precision.

Per-dataset and per-view comparisons between CLUSTOPT (C4) and the baseline arms on the 50
external datasets: bootstrap intervals, Wilcoxon tests, rank-biserial effects, the per-source
breakdown, AMI, and the concentration of the advantage (for example leaving out the
circles-and-moons datasets). Also writes the per-dataset table behind Fig. 4
(released as ``results/external/figure_data/knn_rerank_pairwise.csv``).

Needs the reconstructed external datasets (``python scripts/reconstruct_external_datasets.py``);
see ``analysis/README.md``. Analysis only: no clustering, search or model fitting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stats_common import paired_block, group_bootstrap_ci  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from release_paths import LEGACY_ROOT, output_dir  # noqa: E402
REPO = LEGACY_ROOT
IDB_CFG = REPO / "models" / "Independent_Domain_Benchmark" / "configs"
RUN = (REPO / "results_analysis" / "independent_domain_benchmark"
       / "stage4c_v2_323194df26")
PUB = REPO / "results_analysis" / "independent_domain_benchmark" / "publication"
OUT = output_dir("statistics")

C4 = "IDB_ClustOpt_C4_KNN_TOP10_RAW_R"
C1 = "IDB_ClustOpt_C1_KNN_TOP10_RAW_noR"
C5 = "IDB_ClustOpt_C5_PPv2_R"
A0 = "IDB_AutoClust_A0_original_cvis"
A3 = "IDB_AutoClust_A3_extended_cvis"
M1 = "IDB_ML2DAC_M1_original_plus_established"

PROV: dict = {}


# --------------------------------------------------------------- load / align
def load_manifest():
    man = json.loads((IDB_CFG / "dataset_manifest.json").read_text(encoding="utf-8"))
    entries = {e["dataset_id"]: e for e in man["datasets"] if e.get("accepted", True)}
    # dependence groups: EXACTLY the frozen Stage-4 rule --
    # one sklearn generator (easy/medium/hard) = one group; everything else alone.
    gmap = {d: (("skgen_%s" % e["generator"]) if e.get("generator") else d)
            for d, e in entries.items()}
    return man, entries, gmap


def load_bestview():
    ari = pd.read_csv(RUN / "phase_c" / "method_ari.csv.gz")
    ari = ari[ari["status"] == "success"]
    bv = (ari.groupby(["dataset_id", "source", "method_id"], as_index=False)["ari"]
          .max().rename(columns={"ari": "bestview_ari"}))
    xy = (ari[ari["view_id"] == "xy_2d"]
          .groupby(["dataset_id", "source", "method_id"], as_index=False)["ari"]
          .max().rename(columns={"ari": "xy_ari"}))
    return bv, xy


def pivot(bv, col="bestview_ari"):
    p = bv.pivot(index="dataset_id", columns="method_id", values=col)
    return p


# --------------------------------------------------------------------- AMI
def compute_external_ami() -> pd.DataFrame:
    """Pure offline recomputation: adjusted_mutual_info_score(y, persisted labels).

    Uses EXACTLY the convention of the frozen evaluation layer
    (models/Independent_Domain_Benchmark/evaluation/clustering_metrics.py and
    runners/run_stage4c_phase_c.py): raw predicted labels, noise label -1 kept
    as its own group, nothing removed and nothing re-clustered.
    """
    from sklearn.metrics import adjusted_mutual_info_score
    rows = []
    sci = RUN / "scientific"
    gt_cache: dict = {}
    for f in sorted(sci.glob("*.json")):
        stem = f.stem
        ds, view, method = stem.split("__", 2)
        if method not in (C4, C1, C5, A0, A3, M1):
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        c = rec["content"]
        if c.get("status") != "success":
            continue
        lab = c.get("final_labels")
        if lab is None:
            continue
        if ds not in gt_cache:
            p = PUB / "corpus" / ds / "ground_truth.npy"
            gt_cache[ds] = np.load(p) if p.is_file() else None
        y = gt_cache[ds]
        if y is None or len(y) != len(lab):
            continue
        rows.append({"dataset_id": ds, "view_id": view, "method_id": method,
                     "ami": float(adjusted_mutual_info_score(
                         np.asarray(y), np.asarray(lab, dtype=np.int64)))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main
def main() -> int:
    man, entries, gmap = load_manifest()
    bv, xy = load_bestview()
    P = pivot(bv)
    PX = pivot(xy, "xy_ari")
    src = {d: e["source"] for d, e in entries.items()}

    datasets = list(P.index)
    groups = np.array([gmap[d] for d in datasets])
    PROV["n_datasets"] = len(datasets)
    PROV["n_dependence_groups"] = int(len(set(groups)))
    PROV["method_ari_source"] = str((RUN / "phase_c" / "method_ari.csv.gz")
                                    .relative_to(REPO)).replace("\\", "/")
    PROV["dataset_manifest"] = str((IDB_CFG / "dataset_manifest.json")
                                   .relative_to(REPO)).replace("\\", "/")
    PROV["dependence_group_rule"] = (
        "one sklearn generator (easy/medium/hard) = one group; FCPS and "
        "real-PCA datasets stand alone -- copied from "
        "runners/build_stage4_scientific_review.py")

    # ---------------------------------------------------- C1: headline C4 vs A0
    headline_rows = []

    def add(label, a_id, b_id, frame=P, unit="bestview_ari", subset=None):
        idx = list(frame.index) if subset is None else subset
        sub = frame.loc[idx]
        g = np.array([gmap[d] for d in idx])
        blk = paired_block(sub[a_id].to_numpy(), sub[b_id].to_numpy(), groups=g)
        blk.update({"contrast": label, "method_a": a_id, "method_b": b_id,
                    "unit": unit, "population": ("all_%d" % len(idx)
                                                 if subset is None
                                                 else "subset_%d" % len(idx))})
        headline_rows.append(blk)
        return blk

    add("C4 vs A0 (Best-View ARI)", C4, A0)
    add("C4 vs M1 (Best-View ARI)", C4, M1)
    add("C4 vs A3 (Best-View ARI)", C4, A3)
    add("C4 vs C1 (reranker ablation, Best-View ARI)", C4, C1)
    add("C5 vs A0 (Best-View ARI)", C5, A0)
    add("C4 vs A0 (single-view xy_2d ARI)", C4, A0, frame=PX, unit="xy_2d_ari")
    add("C4 vs M1 (single-view xy_2d ARI)", C4, M1, frame=PX, unit="xy_2d_ari")

    # ------------------------------------------------- C3: circles + moons out
    gen = {d: e.get("generator") for d, e in entries.items()}
    cm_groups = sorted({gmap[d] for d in datasets
                        if gen.get(d) in ("circles", "moons")})
    cm_datasets = sorted([d for d in datasets if gmap[d] in cm_groups])
    rest = [d for d in datasets if d not in set(cm_datasets)]
    PROV["circle_moon_groups"] = cm_groups
    PROV["circle_moon_datasets"] = cm_datasets
    PROV["n_remaining"] = len(rest)

    d_all = (P[C4] - P[A0]).to_numpy()
    d_cm = (P.loc[cm_datasets, C4] - P.loc[cm_datasets, A0]).to_numpy()
    contribution = float(np.sum(d_cm) / len(datasets))

    lco = []
    blk_rest = add("C4 vs A0 -- leave circles+moons out (44 datasets)",
                   C4, A0, subset=rest)
    blk_rest_c5 = add("C5 vs A0 -- leave circles+moons out (44 datasets)",
                      C5, A0, subset=rest)
    lco.append({
        "quantity": "overall_mean_delta_C4_minus_A0_50ds",
        "value": float(np.mean(d_all))})
    lco.append({
        "quantity": "circles_moons_contribution_to_50ds_mean",
        "value": contribution,
        "note": "sum(delta over the 6 circle/moon datasets) / 50"})
    lco.append({
        "quantity": "mean_delta_on_remaining_44",
        "value": blk_rest["mean_delta"]})
    lco.append({"quantity": "C4_mean_on_remaining_44",
                "value": blk_rest["mean_a"]})
    lco.append({"quantity": "A0_mean_on_remaining_44",
                "value": blk_rest["mean_b"]})
    lco.append({"quantity": "C5_minus_A0_mean_on_remaining_44",
                "value": blk_rest_c5["mean_delta"]})
    lco.append({"quantity": "n_circle_moon_datasets",
                "value": float(len(cm_datasets))})
    lco.append({"quantity": "n_circle_moon_dependence_groups",
                "value": float(len(cm_groups))})
    lco.append({"quantity": "decomposition_check_overall_equals_contrib_plus_rest",
                "value": float(np.mean(d_all)
                               - (contribution
                                  + blk_rest["mean_delta"] * len(rest) / len(datasets)))})
    pd.DataFrame(lco).to_csv(OUT / "leave_circles_moons_out.csv", index=False)

    # -------------------------------------------------------- C2: by source
    src_rows = []
    for s in sorted({src[d] for d in datasets}):
        idx = [d for d in datasets if src[d] == s]
        g = np.array([gmap[d] for d in idx])
        for a_id, b_id, lbl in ((C4, A0, "C4-A0"), (C4, M1, "C4-M1"),
                                (C5, A0, "C5-A0")):
            dd = (P.loc[idx, a_id] - P.loc[idx, b_id]).to_numpy()
            ci = group_bootstrap_ci(dd, g)
            src_rows.append({
                "source": s, "n_datasets": len(idx),
                "n_dependence_groups": int(len(set(g))),
                "contrast": lbl,
                "mean_a": float(P.loc[idx, a_id].mean()),
                "mean_b": float(P.loc[idx, b_id].mean()),
                "mean_delta": float(dd.mean()),
                "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
        # per-arm means for the table
        for m in (C4, C1, C5, A0, A3, M1):
            src_rows.append({
                "source": s, "n_datasets": len(idx),
                "n_dependence_groups": int(len(set(g))),
                "contrast": "ARM_MEAN:%s" % m,
                "mean_a": float(P.loc[idx, m].mean()),
                "mean_b": np.nan, "mean_delta": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan})
    pd.DataFrame(src_rows).to_csv(OUT / "external_source_stats.csv", index=False)

    # ------------------------------------------------------------- AMI (offline)
    ami = compute_external_ami()
    ami.to_csv(OUT / "external_ami_per_view.csv", index=False)
    if not ami.empty:
        # Best-view AMI is taken AT THE ARI-SELECTED VIEW, so that the AMI
        # contrast describes the same returned partitions as the ARI contrast.
        long = pd.read_csv(RUN / "phase_c" / "method_ari.csv.gz")
        long = long[long["status"] == "success"]
        pick = (long.sort_values("ari")
                .groupby(["dataset_id", "method_id"], as_index=False).last()
                [["dataset_id", "method_id", "view_id"]])
        amibv = pick.merge(ami, on=["dataset_id", "method_id", "view_id"], how="left")
        AP = amibv.pivot(index="dataset_id", columns="method_id", values="ami")
        AP = AP.loc[datasets]
        for a_id, b_id, lbl in ((C4, A0, "C4 vs A0 (Best-View AMI)"),
                                (C4, M1, "C4 vs M1 (Best-View AMI)"),
                                (C4, C1, "C4 vs C1 (reranker ablation, Best-View AMI)")):
            blk = paired_block(AP[a_id].to_numpy(), AP[b_id].to_numpy(), groups=groups)
            blk.update({"contrast": lbl, "method_a": a_id, "method_b": b_id,
                        "unit": "bestview_ami_at_ari_selected_view",
                        "population": "all_%d" % len(datasets)})
            headline_rows.append(blk)

    hdf = pd.DataFrame(headline_rows)
    front = ["contrast", "method_a", "method_b", "unit", "population", "n",
             "mean_a", "mean_b", "mean_delta", "median_delta", "ci_lo", "ci_hi",
             "n_dependence_groups", "bootstrap_unit", "wins", "ties", "losses",
             "wins_eps", "ties_eps", "losses_eps", "statistic", "p_value",
             "rank_biserial", "valid"]
    hdf = hdf[[c for c in front if c in hdf.columns]
              + [c for c in hdf.columns if c not in front]]
    hdf.to_csv(OUT / "external_headline_stats.csv", index=False)

    # ------------------------------------------------- C5: figure-4 source table
    fig = pd.DataFrame({"dataset_id": datasets})
    fig["source"] = [src[d] for d in datasets]
    fig["generator"] = [gen.get(d) for d in datasets]
    fig["dependence_group"] = [gmap[d] for d in datasets]
    fig["circle_or_moon_group"] = [gmap[d] in cm_groups for d in datasets]
    for code, mid in (("C4", C4), ("C1", C1), ("C5", C5),
                      ("A0", A0), ("A3", A3), ("M1", M1)):
        fig["%s_bestview_ari" % code] = P.loc[datasets, mid].to_numpy()
        fig["%s_xy_ari" % code] = PX.loc[datasets, mid].to_numpy()
    fig["C4_minus_A0"] = fig["C4_bestview_ari"] - fig["A0_bestview_ari"]
    fig["C4_minus_M1"] = fig["C4_bestview_ari"] - fig["M1_bestview_ari"]
    fig["C4_minus_C1"] = fig["C4_bestview_ari"] - fig["C1_bestview_ari"]
    fig["C5_minus_A0"] = fig["C5_bestview_ari"] - fig["A0_bestview_ari"]
    fig.to_csv(OUT / "knn_rerank_pairwise.csv", index=False)

    # --------------------------------------------------- D6 corpus composition
    comp = {"total": len(entries)}
    for s in sorted({e["source"] for e in entries.values()}):
        comp[s] = sum(1 for e in entries.values() if e["source"] == s)
    real = [e for e in entries.values() if e["source"] == "real_pca"]
    comp["real_pca_applied_true"] = sum(1 for e in real if e.get("pca_applied"))
    comp["real_pca_applied_false_native_2d"] = sum(
        1 for e in real if not e.get("pca_applied"))
    for s in ("fcps", "sklearn_shapes"):
        ss = [e for e in entries.values() if e["source"] == s]
        comp["%s_pca_applied_true" % s] = sum(1 for e in ss if e.get("pca_applied"))
        comp["%s_pca_applied_false" % s] = sum(1 for e in ss if not e.get("pca_applied"))
    comp["all_sources_pca_applied_true"] = sum(
        1 for e in entries.values() if e.get("pca_applied"))
    comp["manifest_sha256_16"] = man.get("manifest_sha256_16")
    comp["accepted_by_source"] = man.get("accepted_by_source")
    PROV["corpus_composition"] = comp

    # --------------------------------------------------- D8 runtime accounting
    rt = []
    for f in sorted((RUN / "scientific").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        r = (rec["content"].get("runtime") or {})
        rt.append({"method_id": rec["provenance"]["method_id"],
                   "metafeature_runtime": r.get("metafeature_runtime"),
                   "model_init_runtime": r.get("model_init_runtime"),
                   "persistence_runtime": r.get("persistence_runtime"),
                   "method_runtime_equivalent": r.get("method_runtime_equivalent")})
    rtd = pd.DataFrame(rt)
    PROV["runtime_audit"] = {
        "n_records": int(len(rtd)),
        "metafeature_runtime_all_zero": bool((rtd["metafeature_runtime"] == 0).all()),
        "metafeature_runtime_max": float(rtd["metafeature_runtime"].max()),
        "model_init_runtime_all_zero": bool((rtd["model_init_runtime"] == 0).all()),
        "method_runtime_equivalent_median_by_arm": {
            k: float(v) for k, v in
            rtd.groupby("method_id")["method_runtime_equivalent"].median().items()},
    }

    (OUT / "external_statistics_provenance.json").write_text(
        json.dumps(PROV, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"datasets": len(datasets),
                      "groups": PROV["n_dependence_groups"],
                      "circle_moon_groups": cm_groups,
                      "circle_moon_datasets": cm_datasets,
                      "corpus": comp,
                      "runtime": PROV["runtime_audit"]}, indent=1, default=str))
    print(hdf[["contrast", "n", "mean_a", "mean_b", "mean_delta", "ci_lo",
               "ci_hi", "wins", "ties", "losses", "p_value",
               "rank_biserial"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
