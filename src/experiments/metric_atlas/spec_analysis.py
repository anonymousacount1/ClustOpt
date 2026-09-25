"""Stage 3C analysis: core breadth, family/subfamily specialists, structure tests.

Every threshold used here was fixed in the Stage-3C brief BEFORE any value was
computed, and none is altered afterwards. The comparators are asymmetric on
purpose: a *family* specialist is judged against everything outside its family,
while a *subfamily* specialist is judged against its sibling subfamilies inside
the same parent family, so that broad family effects cannot masquerade as fine
subfamily specialisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import common as C
from .spec_data import SpecData, ranks_desc

BOOT_FAMILY = 1000
BOOT_SUBFAMILY = 500
PERM_FAMILY = 2000
PERM_SUBFAMILY = 1000
MIN_N_SUBFAMILY = 8
SEED = 20260910

#: operational CORE definition, frozen before computation (§15 of the brief)
CORE_TOPK = 5
CORE_MIN_FAMILIES = 6
CORE_SENSITIVITY = (5, 6, 7)

#: Tier thresholds, frozen before computation (§16)
TIER_A_UTIL_RANK = 5
TIER_A_SEL_RANK = 10
TIER_B_UTIL_RANK = 10
TIER_B_SEL_RANK = 15
TIER_B_ENRICH = 1.5


# --------------------------------------------------------------- uplift core
def _boot_diff(A: np.ndarray, B: np.ndarray, b: int, rng: np.random.Generator
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile CI of (colmean(A) - colmean(B)), resampling rows of each."""
    na, nb = A.shape[0], B.shape[0]
    reps = np.empty((b, A.shape[1]), np.float32)
    for i in range(b):
        with np.errstate(invalid="ignore"):
            reps[i] = (np.nanmean(A[rng.integers(0, na, na)], axis=0)
                       - np.nanmean(B[rng.integers(0, nb, nb)], axis=0))
    return (np.nanpercentile(reps, 2.5, axis=0),
            np.nanpercentile(reps, 97.5, axis=0))


def _wilcoxon_like_p(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Two-sided Mann-Whitney p per column (in-group vs out-group, unpaired)."""
    from scipy.stats import mannwhitneyu
    out = np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]):
        a, b = A[:, j], B[:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 3 or b.size < 3:
            continue
        try:
            out[j] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            pass
    return out


def level_table(D: SpecData, level: str, log) -> pd.DataFrame:
    """One row per (metric, group) with utility/selection uplift and inference.

    ``level`` is "family" (comparator = everything outside the family) or
    "subfamily" (comparator = sibling subfamilies of the same parent family).
    """
    rng = np.random.default_rng(SEED)
    labels = D.fam_of if level == "family" else D.sub_of
    groups = sorted(pd.unique(labels))
    b = BOOT_FAMILY if level == "family" else BOOT_SUBFAMILY
    gTRUE = np.nanmean(D.TRUE, axis=0)
    gSEL = D.SEL.mean(axis=0)
    grank_u = ranks_desc(gTRUE)
    grank_s = ranks_desc(gSEL)

    rows = []
    for g in groups:
        inm = labels == g
        if level == "family":
            outm = ~inm
            comparator = "outside_family"
        else:
            outm = (D.fam_of == D.parent[g]) & ~inm
            comparator = "sibling_subfamilies_same_family"
        n_in, n_out = int(inm.sum()), int(outm.sum())
        with np.errstate(invalid="ignore"):
            u_in = np.nanmean(D.TRUE[inm], axis=0)
            u_out = np.nanmean(D.TRUE[outm], axis=0) if n_out else np.full(len(D.metrics), np.nan)
            u_med = np.nanmedian(D.TRUE[inm], axis=0)
            p_in = np.nanmean(D.PRED[inm], axis=0)
        s_in = D.SEL[inm].mean(axis=0)
        s_out = D.SEL[outm].mean(axis=0) if n_out else np.full(len(D.metrics), np.nan)

        eligible = (n_in >= MIN_N_SUBFAMILY) if level == "subfamily" else True
        if eligible and n_out >= 3:
            ulo, uhi = _boot_diff(D.TRUE[inm], D.TRUE[outm], b, rng)
            slo, shi = _boot_diff(D.SEL[inm], D.SEL[outm], b, rng)
            pvals = _wilcoxon_like_p(D.TRUE[inm], D.TRUE[outm])
        else:
            nanv = np.full(len(D.metrics), np.nan)
            ulo = uhi = slo = shi = pvals = nanv

        r_u, r_s = ranks_desc(u_in), ranks_desc(s_in)
        with np.errstate(invalid="ignore", divide="ignore"):
            enrich = np.where(s_out > 0, s_in / s_out, np.inf)
            enrich = np.where((s_out == 0) & (s_in == 0), np.nan, enrich)
        for j, m in enumerate(D.metrics):
            rows.append({
                "level": level, "group": g,
                "parent_family": D.parent[g] if level == "subfamily" else g,
                "metric_id": m, "n_datasets": n_in, "n_comparator_datasets": n_out,
                "comparator": comparator,
                "eligible_for_inference": bool(eligible and n_out >= 3),
                "U_global": float(gTRUE[j]), "U_group": float(u_in[j]),
                "U_group_median": float(u_med[j]), "U_comparator": float(u_out[j]),
                "DeltaU": float(u_in[j] - u_out[j]),
                "DeltaU_ci_low": float(ulo[j]), "DeltaU_ci_high": float(uhi[j]),
                "DeltaU_p": float(pvals[j]),
                "P_group": float(p_in[j]),
                "pred_error_group": float(p_in[j] - u_in[j]),
                "S_global": float(gSEL[j]), "S_group": float(s_in[j]),
                "S_comparator": float(s_out[j]),
                "DeltaS": float(s_in[j] - s_out[j]),
                "DeltaS_ci_low": float(slo[j]), "DeltaS_ci_high": float(shi[j]),
                "enrichment": float(enrich[j]),
                "enrichment_undefined": bool(not np.isfinite(enrich[j])),
                "utility_rank_global": float(grank_u[j]),
                "utility_rank_group": float(r_u[j]),
                "rank_uplift_utility": float(grank_u[j] - r_u[j]),
                "selection_rank_global": float(grank_s[j]),
                "selection_rank_group": float(r_s[j]),
                "rank_uplift_selection": float(grank_s[j] - r_s[j]),
            })
    T = pd.DataFrame(rows)
    T = T.merge(D.atlas[["metric_id", "group", "subgroup"]]
                .rename(columns={"group": "metric_group"}), on="metric_id", how="left")
    # BH-FDR within the predeclared family of New46 uplift tests at this level
    m = (T["metric_group"] == "New46") & T["eligible_for_inference"]
    T["DeltaU_q_bh"] = np.nan
    T.loc[m, "DeltaU_q_bh"] = C.bh_fdr(T.loc[m, "DeltaU_p"].to_numpy())
    T["test_family"] = np.where(m, "New46_x_%s_utility_uplift" % level, "")
    log("%s table: %d rows (%d groups x %d metrics); %d New46 tests FDR-corrected"
        % (level, len(T), T["group"].nunique(), len(D.metrics), int(m.sum())))
    return T


# ------------------------------------------------------------- core breadth
def core_breadth(D: SpecData, FAM: pd.DataFrame, log
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rows = []
    for m in D.metrics:
        s = FAM[FAM.metric_id == m]
        d = {"metric_id": m}
        for k in (3, 5, 10):
            d["B_U_top%d" % k] = int((s["utility_rank_group"] <= k).sum())
            d["B_S_top%d" % k] = int((s["selection_rank_group"] <= k).sum())
        # threshold-free specialisation: normalised entropy of selection mass
        r = s.set_index("group")["S_group"].reindex(sorted(FAM["group"].unique())).to_numpy()
        tot = np.nansum(r)
        if tot > 0:
            p = r / tot
            p = p[p > 0]
            d["H_selection"] = float(-(p * np.log(p)).sum() / np.log(len(r)))
        else:
            d["H_selection"] = np.nan
        u = s.set_index("group")["U_group"].to_numpy()
        up = np.clip(u, 0, None)
        tu = np.nansum(up)
        if tu > 0:
            q = up / tu
            q = q[q > 0]
            d["H_utility"] = float(-(q * np.log(q)).sum() / np.log(len(u)))
        else:
            d["H_utility"] = np.nan
        d["selection_specificity"] = 1.0 - d["H_selection"] if np.isfinite(d["H_selection"]) else np.nan
        rows.append(d)
    B = pd.DataFrame(rows).merge(
        D.atlas[["metric_id", "group", "subgroup"]].rename(
            columns={"group": "metric_group"}), on="metric_id", how="left")
    B["global_selection_rate"] = [float(D.SEL[:, D.metrics.index(m)].mean())
                                  for m in B["metric_id"]]
    B["is_core_candidate"] = ((B["B_U_top5"] >= CORE_MIN_FAMILIES)
                              & (B["B_S_top5"] >= CORE_MIN_FAMILIES))
    sens = {}
    for t in CORE_SENSITIVITY:
        sel = B[(B["B_U_top5"] >= t) & (B["B_S_top5"] >= t)]
        sens["threshold_%d_of_12" % t] = {
            "n_core": int(len(sel)), "metrics": list(sel["metric_id"]),
            "by_group": sel["metric_group"].value_counts().to_dict()}
    core = list(B[B["is_core_candidate"]]["metric_id"])
    log("core candidates (Top-%d utility AND selection in >=%d/12 families): %d -> %s"
        % (CORE_TOPK, CORE_MIN_FAMILIES, len(core), core))
    Cdf = B[B["is_core_candidate"]].sort_values("B_U_top5", ascending=False)
    return B, Cdf, {"definition": {
        "top_k": CORE_TOPK, "min_families": CORE_MIN_FAMILIES,
        "requires": "Top-5 by TRUE utility AND Top-5 by selection frequency",
        "frozen_before_computation": True,
        "note": "operational descriptive label, not a natural-law threshold"},
        "core_candidates": core, "sensitivity": sens}


# --------------------------------------------------------------- specialists
def specialists(T: pd.DataFrame, level: str, core: Sequence[str], log
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = T[T.metric_group == "New46"].copy()
    a = n[(n["utility_rank_group"] <= TIER_A_UTIL_RANK)
          & (n["DeltaU"] > 0)
          & (n["DeltaU_q_bh"] < 0.05)
          & (n["selection_rank_group"] <= TIER_A_SEL_RANK)
          & (n["DeltaS"] > 0)
          & (n["DeltaS_ci_low"] > 0)].copy()
    a["tier"] = "A"
    a["is_core_candidate"] = a["metric_id"].isin(core)
    a["unique_specialist"] = ~a["is_core_candidate"]

    b = n[(n["utility_rank_group"] <= TIER_B_UTIL_RANK)
          & (n["DeltaU"] > 0)
          & (n["enrichment"] >= TIER_B_ENRICH)
          & (n["selection_rank_group"] <= TIER_B_SEL_RANK)].copy()
    b = b[~b.set_index(["metric_id", "group"]).index.isin(
        a.set_index(["metric_id", "group"]).index)]
    b["tier"] = "B"
    b["is_core_candidate"] = b["metric_id"].isin(core)
    log("%s specialists: Tier-A %d pairs (%d unique), Tier-B %d pairs"
        % (level, len(a), int(a["unique_specialist"].sum()), len(b)))
    return a.sort_values(["group", "utility_rank_group"]), b.sort_values(
        ["group", "utility_rank_group"])


# ------------------------------------------------- utility/selection alignment
def alignment(D: SpecData, T: pd.DataFrame, level: str) -> pd.DataFrame:
    grp = D.atlas.set_index("metric_id")["group"].to_dict()
    sub = D.atlas.set_index("metric_id")["subgroup"].to_dict()
    rows = []
    for g, s in T.groupby("group"):
        s = s.copy()
        s["mg"] = s["metric_id"].map(grp)
        s["sg"] = s["metric_id"].map(sub)
        rec = {"level": level, "group": g,
               "parent_family": s["parent_family"].iloc[0],
               "n_datasets": int(s["n_datasets"].iloc[0]),
               "n_metrics": int(len(s))}
        for key, m in (("all", s), ("Original", s[s.mg == "Original"]),
                       ("Established", s[s.mg == "Established"]),
                       ("New46", s[s.mg == "New46"]),
                       ("New46-Pattern", s[s.sg == "New46-Pattern"]),
                       ("New46-Image", s[s.sg == "New46-Image"])):
            if len(m) >= 4:
                rec["spearman_util_sel_%s" % key] = float(
                    m["U_group"].corr(m["S_group"], method="spearman"))
                rec["spearman_rank_%s" % key] = float(
                    m["utility_rank_group"].corr(m["selection_rank_group"],
                                                 method="spearman"))
            else:
                rec["spearman_util_sel_%s" % key] = np.nan
                rec["spearman_rank_%s" % key] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def topk_recall(D: SpecData, level: str) -> pd.DataFrame:
    """True Top-k recall of the predicted/selected set, per group."""
    labels = D.fam_of if level == "family" else D.sub_of
    grp = np.array([D.atlas.set_index("metric_id")["group"].get(m, "") for m in D.metrics])
    isnew = grp == "New46"
    rt = np.apply_along_axis(ranks_desc, 1, D.TRUE)
    rp = np.apply_along_axis(ranks_desc, 1, D.PRED)
    rows = []
    for g in sorted(pd.unique(labels)):
        m = labels == g
        t5, p5 = rt[m] <= 5, rp[m] <= 5
        inter = (t5 & p5).sum(axis=1)
        rec = {
            "level": level, "group": g,
            "parent_family": D.parent[g] if level == "subfamily" else g,
            "n_datasets": int(m.sum()),
            "true_top5_recall": float(np.mean(inter / 5.0)),
            "true_top1_hit": float(np.mean(((rt[m] <= 1) & p5).sum(axis=1) > 0)),
        }
        tn = t5 & isnew[None, :]
        pn = p5 & isnew[None, :]
        cnt_tn = tn.sum(axis=1)
        rec["new46_true_top5_share"] = float(np.mean(cnt_tn / 5.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            rec["new46_true_top5_recall"] = float(np.nanmean(
                np.where(cnt_tn > 0, (tn & pn).sum(axis=1) / np.maximum(cnt_tn, 1), np.nan)))
            rec["new46_selection_precision"] = float(np.nanmean(
                np.where(pn.sum(axis=1) > 0,
                         (tn & pn).sum(axis=1) / np.maximum(pn.sum(axis=1), 1), np.nan)))
            rec["missed_high_utility_new46_rate"] = float(np.nanmean(
                np.where(cnt_tn > 0, (tn & ~pn).sum(axis=1) / np.maximum(cnt_tn, 1), np.nan)))
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------- permutation testing
def _group_means(M: np.ndarray, cnt: np.ndarray, starts: np.ndarray) -> np.ndarray:
    s = np.add.reduceat(M, starts, axis=0)
    c = np.add.reduceat(cnt, starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def structure_permutation(D: SpecData, level: str, log) -> pd.DataFrame:
    """Is a metric's between-group variance larger than chance?

    Family test shuffles family labels across all datasets (sizes preserved).
    Subfamily test shuffles subfamily labels **within each parent family only**,
    so family structure cannot leak into the subfamily result.
    """
    rng = np.random.default_rng(SEED)
    labels = D.fam_of if level == "family" else D.sub_of
    nperm = PERM_FAMILY if level == "family" else PERM_SUBFAMILY
    Tf = np.nan_to_num(D.TRUE, nan=0.0)
    Tm = np.isfinite(D.TRUE).astype(np.float32)
    S = D.SEL.astype(np.float32)
    Sm = np.ones_like(S)

    if level == "family":
        order = np.argsort(labels, kind="stable")
        blocks = [order]
    else:
        order = np.lexsort((labels, D.fam_of))
        blocks = []
        fam_sorted = D.fam_of[order]
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and fam_sorted[j] == fam_sorted[i]:
                j += 1
            blocks.append(order[i:j])
            i = j
    lab_sorted = np.asarray(labels)[order]
    # labels are strings, so boundaries are found by inequality, not np.diff
    starts = np.r_[0, np.flatnonzero(lab_sorted[1:] != lab_sorted[:-1]) + 1]

    def var_stat(perm_order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu_t = _group_means(Tf[perm_order], Tm[perm_order], starts)
        mu_s = _group_means(S[perm_order], Sm[perm_order], starts)
        return np.nanvar(mu_t, axis=0), np.nanvar(mu_s, axis=0)

    obs_t, obs_s = var_stat(order)
    ge_t = np.zeros(len(D.metrics)); ge_s = np.zeros(len(D.metrics))
    for _ in range(nperm):
        # shuffle dataset rows inside each block; the block boundaries and the
        # group sizes are untouched, so only the label assignment changes.
        parts = []
        for blk in blocks:
            b = blk.copy()
            rng.shuffle(b)
            parts.append(b)
        shuffled = np.concatenate(parts)
        vt, vs = var_stat(shuffled)
        ge_t += (vt >= obs_t)
        ge_s += (vs >= obs_s)
    p_t = (ge_t + 1) / (nperm + 1)
    p_s = (ge_s + 1) / (nperm + 1)
    R = pd.DataFrame({
        "level": level, "metric_id": D.metrics,
        "obs_between_group_utility_var": obs_t,
        "obs_between_group_selection_var": obs_s,
        "utility_p": p_t, "selection_p": p_s,
        "n_permutations": nperm, "seed": SEED,
    })
    R["utility_q_bh"] = C.bh_fdr(R["utility_p"].to_numpy())
    R["selection_q_bh"] = C.bh_fdr(R["selection_p"].to_numpy())
    R = R.merge(D.atlas[["metric_id", "group", "subgroup"]]
                .rename(columns={"group": "metric_group"}), on="metric_id", how="left")
    log("%s structure permutation (%d perms): %d/%d metrics with utility q<0.05, "
        "%d with selection q<0.05"
        % (level, nperm, int((R.utility_q_bh < 0.05).sum()), len(R),
           int((R.selection_q_bh < 0.05).sum())))
    return R


# ------------------------------------------------------------------- jaccard
def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else np.nan


def top5_jaccard(T: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    tops = {g: set(s.nsmallest(5, col)["metric_id"])
            for g, s in T.groupby("group")}
    gs = sorted(tops)
    M = pd.DataFrame(index=gs, columns=gs, dtype=float)
    vals = []
    for i, a in enumerate(gs):
        for j, b in enumerate(gs):
            v = _jac(tops[a], tops[b])
            M.loc[a, b] = v
            if i < j:
                vals.append(v)
    stats = {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
             "min": float(np.min(vals)), "max": float(np.max(vals)),
             "n_pairs": len(vals)}
    out = M.reset_index().rename(columns={"index": "group"})
    return out, stats
