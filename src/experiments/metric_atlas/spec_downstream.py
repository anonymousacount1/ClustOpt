"""Stage 3C downstream and cross-method layer.

Reads the authoritative Stage-1 per-dataset online results and the Stage-3B
ML2DAC / AutoClust artefacts. Nothing is re-run.

A population note that matters for interpretation. Specialisation is measured on
**Splits 2-16** (the leakage-free OOF utility population, 16,013 datasets), while
the downstream Full60-vs-Head14 gain is measured on **Split 1** (1,055 datasets,
the Stage-1 evaluation split). The two are different dataset populations sharing
the same 12 families and 86 subfamilies. That is deliberate: it makes the
specialisation signal and the downstream signal independent by construction, so
their association cannot be an artefact of measuring both on the same rows. It
also means the association is a **family-level** statement, never a per-dataset one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import common as C

STAGE1_ONLINE = "stage1_2x3_aggregation/online_fixed50/online_2x3_dataset_results.csv"
STAGE1_OFFLINE = "stage1_2x3_aggregation/offline_fixed32/offline_2x3_dataset_results.csv"


def stage1_gain(repo: Path, log, arm_hi: str = "FP", arm_lo: str = "HP"
                ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Paired per-dataset Full60-predicted minus Head14-predicted BestView ARI.

    The ONLINE fixed-50 comparison is chosen as the single authoritative
    downstream contrast because it is the deployed regime (a real search with a
    real budget), it has complete per-dataset outputs for all six arms, and both
    arms are marked deployable. The offline fixed-32 table is never mixed in.
    """
    f = repo / C.STAGE1_REL.replace("results_analysis/clustering_repository/", "")
    src = repo / "results_analysis" / "clustering_repository" / STAGE1_ONLINE
    d = pd.read_csv(C.ext(src))
    hi = d[d.arm == arm_hi].set_index("dataset_id")
    lo = d[d.arm == arm_lo].set_index("dataset_id")
    common = sorted(set(hi.index) & set(lo.index))
    assert len(common) == len(hi) == len(lo), "unpaired Stage-1 datasets"
    g = pd.DataFrame({
        "dataset_id": common,
        "family": hi.loc[common, "family"].to_numpy(),
        "subfamily": hi.loc[common, "subfamily"].to_numpy(),
        "ari_full60": hi.loc[common, "best_view_ari"].to_numpy(float),
        "ari_head14": lo.loc[common, "best_view_ari"].to_numpy(float),
    })
    g["delta"] = g["ari_full60"] - g["ari_head14"]

    def agg(by: str) -> pd.DataFrame:
        rows = []
        for k, s in g.groupby(by):
            dv = s["delta"].to_numpy()
            lo_, hi_ = C.boot_ci_vector(dv, b=2000)
            rows.append({
                by: k, "n_datasets": int(len(s)),
                "mean_full60": float(s["ari_full60"].mean()),
                "mean_head14": float(s["ari_head14"].mean()),
                "mean_delta": float(dv.mean()), "median_delta": float(np.median(dv)),
                "ci_low": lo_, "ci_high": hi_,
                "wins": int((dv > 1e-12).sum()), "losses": int((dv < -1e-12).sum()),
                "ties": int((np.abs(dv) <= 1e-12).sum()),
                "wilcoxon_p": C.wilcoxon_p(dv)})
        r = pd.DataFrame(rows)
        r["wilcoxon_q_bh"] = C.bh_fdr(r["wilcoxon_p"].to_numpy())
        return r

    F, S = agg("family"), agg("subfamily")
    prov = {
        "source": str(src), "regime": "online fixed-50 (deployed regime)",
        "arms": {"high": arm_hi + " = stage1_full60_mlp_top5_raw_fixed50",
                 "low": arm_lo + " = stage1_head14_mlp_top5_raw_fixed50"},
        "why_online": ("both arms deployable, complete per-dataset outputs, and it "
                       "is the regime the deployed system actually runs; the "
                       "offline fixed-32 table is NOT mixed in"),
        "population": "Split 1, 1,055 datasets", "paired": True,
        "n_paired": int(len(g)),
        "overall_mean_delta": float(g["delta"].mean()),
        "stage1_rerun": False,
    }
    log("Stage-1 downstream FP-HP: n=%d, mean delta %+0.6f, %d/%d families positive"
        % (len(g), g["delta"].mean(), int((F["mean_delta"] > 0).sum()), len(F)))
    return F, S, prov


def association(FAMspec: pd.DataFrame, FAMgain: pd.DataFrame, log
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Family-level association between specialisation signals and Full60 gain."""
    j = FAMspec.merge(FAMgain.rename(columns={"family": "group"}), on="group",
                      how="inner", suffixes=("", "_gain"))
    sigs = ["new46_selection_share", "n_tier_a", "n_tier_b",
            "max_new46_utility_uplift", "mean_positive_new46_uplift",
            "new46_oracle_win_rate", "new46_true_top5_share"]
    rows = []
    for s in sigs:
        if s not in j.columns:
            continue
        v = j[[s, "mean_delta"]].dropna()
        rows.append({
            "signal": s, "n_families": int(len(v)),
            "spearman_vs_full60_gain": (float(v[s].corr(v["mean_delta"],
                                                        method="spearman"))
                                        if len(v) >= 4 else np.nan),
            "pearson_vs_full60_gain": (float(v[s].corr(v["mean_delta"]))
                                       if len(v) >= 4 else np.nan),
            "evidence_level": "2 (association; n=12 families, descriptive only)",
        })
    A = pd.DataFrame(rows)
    log("family association (n=12, descriptive): " + ", ".join(
        "%s rho=%.3f" % (r["signal"], r["spearman_vs_full60_gain"])
        for _, r in A.iterrows()))
    return A, j


def hier_association(SUBspec: pd.DataFrame, SUBgain: pd.DataFrame, log,
                     b: int = 2000, seed: int = 20260910) -> pd.DataFrame:
    """Subfamily association with a FAMILY-BLOCK bootstrap.

    Subfamilies are nested in families, so an iid bootstrap over 86 subfamilies
    would understate the uncertainty. Families are resampled as whole blocks.
    """
    j = SUBspec.merge(SUBgain.rename(columns={"subfamily": "group"}), on="group",
                      how="inner", suffixes=("", "_gain"))
    if "parent_family" not in j.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    fams = sorted(j["parent_family"].unique())
    idx_by_fam = {f: j.index[j["parent_family"] == f].to_numpy() for f in fams}
    sigs = [s for s in ("new46_selection_share", "n_tier_a", "n_tier_b",
                        "max_new46_utility_uplift", "new46_oracle_win_rate")
            if s in j.columns]
    rows = []
    for s in sigs:
        v = j[[s, "mean_delta"]].dropna()
        obs = float(v[s].corr(v["mean_delta"], method="spearman")) if len(v) >= 6 else np.nan
        reps = []
        for _ in range(b):
            pick = rng.choice(fams, len(fams), replace=True)
            ii = np.concatenate([idx_by_fam[f] for f in pick])
            vv = j.loc[ii, [s, "mean_delta"]].dropna()
            if len(vv) >= 6 and vv[s].std() > 0:
                reps.append(vv[s].corr(vv["mean_delta"], method="spearman"))
        rows.append({
            "signal": s, "n_subfamilies": int(len(v)), "n_family_blocks": len(fams),
            "spearman": obs,
            "ci_low": float(np.percentile(reps, 2.5)) if reps else np.nan,
            "ci_high": float(np.percentile(reps, 97.5)) if reps else np.nan,
            "resampling": "family-block bootstrap (families resampled whole)",
            "evidence_level": "2 (association only)"})
    R = pd.DataFrame(rows)
    log("subfamily hierarchical association: " + ", ".join(
        "%s rho=%.3f [%.3f, %.3f]" % (r["signal"], r["spearman"], r["ci_low"],
                                      r["ci_high"]) for _, r in R.iterrows()))
    return R


# ------------------------------------------------------------------- ML2DAC
def ml2dac_capture(repo: Path, TierA: pd.DataFrame, TierB: pd.DataFrame,
                   FAMspec: pd.DataFrame, log) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Do the New46 family specialists show up in ML2DAC's CVI* and selections?"""
    O = repo / C.OUT_REL
    sel = pd.read_csv(C.ext(O / "ml2dac_selection_records.csv.gz"))
    star = pd.read_csv(C.ext(O / "ml2dac_split1_cvi_star.csv.gz"))
    atlas = pd.read_csv(C.ext(O / "canonical_metric_atlas.csv"))
    grp = atlas.set_index("metric_id")["group"].to_dict()
    from .ml2dac import canonical_name_map
    alias = canonical_name_map()
    sel = sel.copy()
    sel["selected_cvi"] = sel["selected_cvi"].map(lambda x: alias.get(x, x))
    sel["sel_group"] = sel["selected_cvi"].map(grp).fillna("Unknown")
    star = star.copy()
    star["star_group"] = star["best_cvi"].map(grp).fillna("Unknown")

    a3s = sel[sel.arm == "A3"]
    a3t = star[(star.arm == "A3") & star.resolved]
    j = a3s.merge(a3t[["record_id", "best_cvi", "star_group"]], on="record_id",
                  how="left")
    tierA_by_fam = TierA.groupby("group")["metric_id"].apply(set).to_dict()
    tierB_by_fam = TierB.groupby("group")["metric_id"].apply(set).to_dict()

    rows = []
    for fam, g in j.groupby("family"):
        spec = tierA_by_fam.get(fam, set())
        cand = tierB_by_fam.get(fam, set())
        res = g[g["best_cvi"].notna()]
        rec = {
            "family": fam, "n_view_records": int(len(g)),
            "new46_cvi_star_share": float((g["star_group"] == "New46").mean()),
            "new46_selected_share": float((g["sel_group"] == "New46").mean()),
            "cvi_star_accuracy": float((res["selected_cvi"] == res["best_cvi"]).mean()),
            "n_tier_a_specialists": len(spec), "n_tier_b_specialists": len(cand),
        }
        for tag, ms in (("tier_a", spec), ("tier_b", cand)):
            if ms:
                rec["%s_cvi_star_share" % tag] = float(g["best_cvi"].isin(ms).mean())
                rec["%s_selected_share" % tag] = float(g["selected_cvi"].isin(ms).mean())
                sub = res[res["best_cvi"].isin(ms)]
                rec["%s_recall" % tag] = (float((sub["selected_cvi"] == sub["best_cvi"]).mean())
                                          if len(sub) else np.nan)
                rec["%s_n_records_where_specialist_is_cvi_star" % tag] = int(len(sub))
            else:
                for k in ("cvi_star_share", "selected_share", "recall",
                          "n_records_where_specialist_is_cvi_star"):
                    rec["%s_%s" % (tag, k)] = np.nan
        for gname in ("Original", "Established", "New46"):
            sub = res[res["star_group"] == gname]
            rec["recall_%s" % gname] = (float((sub["selected_cvi"] == sub["best_cvi"]).mean())
                                        if len(sub) else np.nan)
        rows.append(rec)
    F = pd.DataFrame(rows)

    srows = []
    for (fam, sub), g in j.groupby(["family", "subfamily"]):
        res = g[g["best_cvi"].notna()]
        srows.append({
            "family": fam, "subfamily": sub, "n_view_records": int(len(g)),
            "new46_cvi_star_share": float((g["star_group"] == "New46").mean()),
            "new46_selected_share": float((g["sel_group"] == "New46").mean()),
            "cvi_star_accuracy": (float((res["selected_cvi"] == res["best_cvi"]).mean())
                                  if len(res) else np.nan)})
    S = pd.DataFrame(srows)
    log("ML2DAC capture: A3 New46 CVI* share %.3f vs selected %.3f (12 families)"
        % (F["new46_cvi_star_share"].mean(), F["new46_selected_share"].mean()))
    return F, S


def autoclust_family(repo: Path, log) -> pd.DataFrame:
    """Reuse the Stage-3B family reliance table; do not recompute."""
    f = repo / C.OUT_REL / "autoclust_family_group_reliance.csv"
    d = pd.read_csv(C.ext(f))
    d["source"] = str(f)
    d["recomputed_in_stage3c"] = False
    d["evidence_level"] = "2 (predictive reliance, NOT causal ARI contribution)"
    log("AutoClust family reliance reused from Stage 3B (%d rows, no recompute)"
        % len(d))
    return d
