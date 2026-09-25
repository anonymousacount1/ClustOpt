"""Stage 3C runner: family- and subfamily-specific metric specialization.

ANALYSIS ONLY. Zero clustering searches, zero model fits, zero MKR rebuilds.
Stage-3B outputs are read, never overwritten: everything new is written under
``metric_analysis_existing_domain/family_subfamily_specialization/``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metric_atlas import common as C            # noqa: E402
from metric_atlas import spec_data as SD        # noqa: E402
from metric_atlas import spec_analysis as SA    # noqa: E402
from metric_atlas import spec_downstream as DS  # noqa: E402

SUB_REL = "family_subfamily_specialization"


def log(m: str) -> None:
    print("[3c] %s" % m, flush=True)


def _out(repo: Path) -> Path:
    d = repo / C.OUT_REL / SUB_REL
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else C.REPO
    out = _out(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 3C  FAMILY / SUBFAMILY METRIC SPECIALIZATION")
    print("=" * 78, flush=True)

    D = SD.load(repo, log)
    fcnt, scnt = SD.group_counts(D)
    C.write_csv(fcnt, out / "family_dataset_counts.csv")
    C.write_csv(scnt, out / "subfamily_dataset_counts.csv")
    log("subfamilies eligible for inference (n>=8): %d/%d"
        % (int(scnt.eligible_for_inference.sum()), len(scnt)))

    # ------------------------------------------------- family / subfamily tables
    FAM = SA.level_table(D, "family", log)
    SUB = SA.level_table(D, "subfamily", log)
    C.write_csv(FAM, out / "family_metric_full_matrix.csv.gz", gz=True)
    C.write_csv(SUB, out / "subfamily_metric_full_matrix.csv.gz", gz=True)
    ucols = ["level", "group", "parent_family", "metric_id", "metric_group",
             "subgroup", "n_datasets", "U_global", "U_group", "U_group_median",
             "U_comparator", "DeltaU", "DeltaU_ci_low", "DeltaU_ci_high",
             "DeltaU_p", "DeltaU_q_bh", "utility_rank_global",
             "utility_rank_group", "rank_uplift_utility", "eligible_for_inference"]
    scols = ["level", "group", "parent_family", "metric_id", "metric_group",
             "subgroup", "n_datasets", "S_global", "S_group", "S_comparator",
             "DeltaS", "DeltaS_ci_low", "DeltaS_ci_high", "enrichment",
             "enrichment_undefined", "selection_rank_global",
             "selection_rank_group", "rank_uplift_selection"]
    C.write_csv(FAM[ucols], out / "family_metric_utility.csv.gz", gz=True)
    C.write_csv(SUB[ucols], out / "subfamily_metric_utility.csv.gz", gz=True)
    C.write_csv(FAM[scols], out / "family_metric_selection.csv.gz", gz=True)
    C.write_csv(SUB[scols], out / "subfamily_metric_selection.csv.gz", gz=True)

    # ---------------------------------------------------------------- core
    B, Cdf, coredef = SA.core_breadth(D, FAM, log)
    C.write_csv(B, out / "family_core_breadth.csv")
    C.write_csv(Cdf, out / "core_candidates.csv")
    C.write_json(coredef, out / "core_definition.json")
    core = coredef["core_candidates"]

    # --------------------------------------------------------- specialists
    fa, fb = SA.specialists(FAM, "family", core, log)
    sa, sb = SA.specialists(SUB, "subfamily", core, log)
    fam_pairs = set(zip(fa["metric_id"], fa["parent_family"]))
    sa = sa.copy()
    sa["also_family_specialist"] = [
        (m, p) in fam_pairs for m, p in zip(sa["metric_id"], sa["parent_family"])]
    sa["subfamily_only"] = ~sa["also_family_specialist"]
    C.write_csv(fa, out / "family_specialists_tierA.csv")
    C.write_csv(fb, out / "family_specialists_tierB.csv")
    C.write_csv(sa, out / "subfamily_specialists_tierA.csv")
    C.write_csv(sb, out / "subfamily_specialists_tierB.csv")
    log("subfamily-only Tier-A specialists: %d of %d"
        % (int(sa["subfamily_only"].sum()), len(sa)))

    # ------------------------------------------------------------ alignment
    AF = SA.alignment(D, FAM, "family")
    AS = SA.alignment(D, SUB, "subfamily")
    C.write_csv(AF, out / "family_utility_selection_alignment.csv")
    C.write_csv(AS, out / "subfamily_utility_selection_alignment.csv")
    RF = SA.topk_recall(D, "family")
    RS = SA.topk_recall(D, "subfamily")
    C.write_csv(RF, out / "family_true_topk_recall.csv")
    C.write_csv(RS, out / "subfamily_true_topk_recall.csv")
    log("family utility-selection alignment: mean rho(all)=%.3f, New46=%.3f"
        % (AF["spearman_util_sel_all"].mean(), AF["spearman_util_sel_New46"].mean()))

    # -------------------------------------------------- exploited vs missed
    n46 = FAM[FAM.metric_group == "New46"].copy()
    hi_u = n46["utility_rank_group"] <= 10
    hi_s = (n46["enrichment"] >= 1.2) | (n46["selection_rank_group"] <= 10)
    n46["classification"] = np.select(
        [hi_u & hi_s, hi_u & ~hi_s, ~hi_u & hi_s],
        ["A_exploited_specialist", "B_missed_specialist",
         "C_frequently_selected_low_utility"], default="D_neither")
    n46.loc[n46["metric_id"].isin(core), "classification"] = "D_generalist_core"
    C.write_csv(n46, out / "exploited_vs_missed_specialists.csv")
    log("exploited/missed: %s" % n46["classification"].value_counts().to_dict())

    # ------------------------------------------------------- permutation
    PF = SA.structure_permutation(D, "family", log)
    PS = SA.structure_permutation(D, "subfamily", log)
    C.write_csv(PF, out / "family_metric_structure_permutation.csv")
    C.write_csv(PS, out / "subfamily_metric_structure_permutation.csv")

    # ---------------------------------------------------------- jaccard
    JU, ju = SA.top5_jaccard(FAM, "utility_rank_group")
    JS, js = SA.top5_jaccard(FAM, "selection_rank_group")
    C.write_csv(JU, out / "family_top5_utility_jaccard.csv")
    C.write_csv(JS, out / "family_top5_selection_jaccard.csv")
    specsets = {g: set(s["metric_id"]) for g, s in
                pd.concat([fa, fb]).groupby("group")}
    gs = sorted(FAM["group"].unique())
    M = pd.DataFrame(index=gs, columns=gs, dtype=float)
    vals = []
    for i, a in enumerate(gs):
        for k, b2 in enumerate(gs):
            A, Bs = specsets.get(a, set()), specsets.get(b2, set())
            v = (len(A & Bs) / len(A | Bs)) if (A | Bs) else np.nan
            M.loc[a, b2] = v
            if i < k:
                vals.append(v)
    C.write_csv(M.reset_index().rename(columns={"index": "group"}),
                out / "family_specialist_jaccard.csv")
    jspec = {"mean": float(np.nanmean(vals)), "median": float(np.nanmedian(vals)),
             "n_pairs": int(np.isfinite(vals).sum())}
    log("Jaccard: utility Top-5 %.3f, selection Top-5 %.3f, specialist sets %.3f"
        % (ju["mean"], js["mean"], jspec["mean"]))

    # ------------------------------------------------------ view specificity
    vrows = []
    allspec = pd.concat([fa.assign(tier="A"), fb.assign(tier="B")])
    for _, r in allspec.iterrows():
        j = D.metrics.index(r["metric_id"])
        fmask = D.fam_of == r["group"]
        rec = {"metric_id": r["metric_id"], "family": r["group"], "tier": r["tier"],
               "subgroup": r["subgroup"]}
        best, bestv = None, -np.inf
        for v in C.VIEWS:
            idx = D.view_index[v]
            inv = np.zeros(len(D.ds), bool); inv[idx] = True
            m_in, m_out = fmask & inv, (~fmask) & inv
            with np.errstate(invalid="ignore"):
                du = (np.nanmean(D.view_true[v][m_in, j])
                      - np.nanmean(D.view_true[v][m_out, j]))
                dsl = (D.view_sel[v][m_in, j].mean() - D.view_sel[v][m_out, j].mean())
            rec["DeltaU_%s" % v] = float(du)
            rec["DeltaS_%s" % v] = float(dsl)
            if np.isfinite(du) and du > bestv:
                best, bestv = v, du
        vs = [rec["DeltaU_%s" % v] for v in C.VIEWS]
        pos = sum(1 for x in vs if np.isfinite(x) and x > 0)
        rec["dominant_view"] = best
        rec["n_views_positive"] = pos
        rec["view_profile"] = ("broad" if pos == 3 else
                               ("mainly_%s" % best if pos <= 2 else "mixed"))
        vrows.append(rec)
    VS = pd.DataFrame(vrows)
    C.write_csv(VS, out / "view_specific_specialization.csv.gz", gz=True)
    log("view profile of specialists: %s"
        % (VS["view_profile"].value_counts().to_dict() if len(VS) else {}))

    # --------------------------------------------- predicted vs true utility
    prows = []
    for lvl, T in (("family", FAM), ("subfamily", SUB)):
        for g, s in T.groupby("group"):
            rec = {"level": lvl, "group": g,
                   "parent_family": s["parent_family"].iloc[0],
                   "n_datasets": int(s["n_datasets"].iloc[0])}
            for key, m in (("all", s), ("Original", s[s.metric_group == "Original"]),
                           ("Established", s[s.metric_group == "Established"]),
                           ("New46", s[s.metric_group == "New46"])):
                if len(m) >= 4:
                    rec["spearman_true_pred_%s" % key] = float(
                        m["U_group"].corr(m["P_group"], method="spearman"))
                    tt = set(m.nsmallest(5, "utility_rank_group")["metric_id"])
                    pp = set(m.nlargest(5, "P_group")["metric_id"])
                    rec["true_top5_recall_%s" % key] = len(tt & pp) / max(len(tt), 1)
                    rec["mean_pred_error_%s" % key] = float(m["pred_error_group"].mean())
                else:
                    for k in ("spearman_true_pred", "true_top5_recall", "mean_pred_error"):
                        rec["%s_%s" % (k, key)] = np.nan
            prows.append(rec)
    PT = pd.DataFrame(prows)
    C.write_csv(PT[PT.level == "family"], out / "family_predicted_vs_true.csv")
    C.write_csv(PT[PT.level == "subfamily"], out / "subfamily_predicted_vs_true.csv")

    # ------------------------------------------------------------ downstream
    GF, GS, gprov = DS.stage1_gain(repo, log)
    C.write_csv(GF, out / "clustopt_family_downstream_gain.csv")
    C.write_csv(GS, out / "clustopt_subfamily_downstream_gain.csv")
    C.write_json(gprov, out / "downstream_source.json")

    # -------------------------------------------------- family signature build
    oracle = pd.read_csv(C.ext(repo / C.OUT_REL / "new46_unique_win_coverage.csv"))
    famname = (pd.read_csv(C.ext(repo / C.OOF_REL), usecols=["family_id", "family_name"])
               .drop_duplicates().set_index("family_name")["family_id"].to_dict())
    oracle["group"] = oracle["key"].map(lambda k: famname.get(k, k))
    ocov = oracle[oracle.level == "family"].set_index("group")["new46_best_frac"].to_dict()
    ocov_sub = oracle[oracle.level == "subfamily"].set_index("key")["new46_best_frac"].to_dict()

    sig = []
    for g, s in FAM.groupby("group"):
        n46f = s[s.metric_group == "New46"]
        a = fa[fa.group == g]; bq = fb[fb.group == g]
        miss = n46[(n46.group == g) & (n46.classification == "B_missed_specialist")]
        selslots = s["S_group"].sum()
        core_share = s[s.metric_id.isin(core)]["S_group"].sum()
        uniq = a[a.unique_specialist]["metric_id"]
        uniq_share = s[s.metric_id.isin(uniq)]["S_group"].sum()
        new_share = n46f["S_group"].sum()
        rec = {
            "family": g, "n_datasets": int(s["n_datasets"].iloc[0]),
            "core_metrics_present": "|".join(sorted(
                s[s.metric_id.isin(core) & (s.utility_rank_group <= 10)]["metric_id"])),
            "top5_new46_by_utility": "|".join(
                n46f.nsmallest(5, "utility_rank_group")["metric_id"]),
            "tier_a_specialists": "|".join(sorted(a["metric_id"])),
            "tier_b_specialists": "|".join(sorted(bq["metric_id"])),
            "missed_specialists": "|".join(sorted(miss["metric_id"])),
            "n_core_present": int(s[s.metric_id.isin(core)
                                    & (s.utility_rank_group <= 10)].shape[0]),
            "n_tier_a": int(len(a)), "n_tier_a_unique": int(a["unique_specialist"].sum()),
            "n_tier_b": int(len(bq)), "n_missed": int(len(miss)),
            "new46_selection_share": float(new_share / selslots) if selslots else np.nan,
            "core_selection_share": float(core_share / selslots) if selslots else np.nan,
            "unique_specialist_selection_share": (float(uniq_share / selslots)
                                                  if selslots else np.nan),
            "new46_oracle_win_rate": float(ocov.get(g, np.nan)),
            "max_new46_utility_uplift": float(n46f["DeltaU"].max()),
            "mean_positive_new46_uplift": float(n46f.loc[n46f.DeltaU > 0, "DeltaU"].mean()),
        }
        sig.append(rec)
    SIG = pd.DataFrame(sig)
    SIG = SIG.merge(RF[["group", "new46_true_top5_share", "true_top5_recall",
                        "new46_true_top5_recall"]].rename(columns={"group": "family"}),
                    on="family", how="left")
    SIG = SIG.merge(AF[["group", "spearman_util_sel_all", "spearman_util_sel_New46"]]
                    .rename(columns={"group": "family"}), on="family", how="left")
    SIG = SIG.merge(GF[["family", "mean_delta", "ci_low", "ci_high", "wins",
                        "losses", "ties", "wilcoxon_q_bh"]]
                    .rename(columns={"mean_delta": "full60_minus_head14",
                                     "ci_low": "gain_ci_low",
                                     "ci_high": "gain_ci_high"}),
                    on="family", how="left")
    C.write_csv(SIG, out / "family_signature.csv")
    C.write_json(SIG.to_dict("records"), out / "family_signature.json")

    # ------------------------------------------------ subfamily signature
    ssig = []
    for g, s in SUB.groupby("group"):
        n46f = s[s.metric_group == "New46"]
        a = sa[sa.group == g]; bq = sb[sb.group == g]
        selslots = s["S_group"].sum()
        ssig.append({
            "subfamily": g, "parent_family": s["parent_family"].iloc[0],
            "n_datasets": int(s["n_datasets"].iloc[0]),
            "eligible_for_inference": bool(s["eligible_for_inference"].iloc[0]),
            "top5_new46_by_utility": "|".join(
                n46f.nsmallest(5, "utility_rank_group")["metric_id"]),
            "tier_a_specialists": "|".join(sorted(a["metric_id"])),
            "tier_b_specialists": "|".join(sorted(bq["metric_id"])),
            "subfamily_only_specialists": "|".join(sorted(
                a[a.subfamily_only]["metric_id"])) if len(a) else "",
            "n_tier_a": int(len(a)), "n_tier_b": int(len(bq)),
            "n_subfamily_only": int(a["subfamily_only"].sum()) if len(a) else 0,
            "new46_selection_share": (float(n46f["S_group"].sum() / selslots)
                                      if selslots else np.nan),
            "new46_oracle_win_rate": float(ocov_sub.get(g, np.nan)),
            "max_new46_utility_uplift": float(n46f["DeltaU"].max()),
        })
    SSIG = pd.DataFrame(ssig).merge(
        GS[["subfamily", "mean_delta", "wins", "losses", "ties"]]
        .rename(columns={"mean_delta": "full60_minus_head14"}),
        on="subfamily", how="left")
    C.write_csv(SSIG, out / "subfamily_signature.csv.gz", gz=True)
    SSIG.to_json(C.ext(out / "subfamily_signature.json.gz"), orient="records",
                 compression="gzip")

    # ------------------------------------------------------- associations
    ASSOC, joined = DS.association(
        SIG.rename(columns={"family": "group"}), GF, log)
    C.write_csv(ASSOC, out / "specialization_downstream_association.csv")
    HA = DS.hier_association(SSIG.rename(columns={"subfamily": "group"}), GS, log)
    C.write_csv(HA, out / "subfamily_downstream_association_hierarchical.csv")

    # ------------------------------------------------------- cross-method
    MF, MS = DS.ml2dac_capture(repo, fa, fb, SIG, log)
    C.write_csv(MF, out / "ml2dac_family_specialist_capture.csv")
    C.write_csv(MS, out / "ml2dac_subfamily_specialist_capture.csv.gz", gz=True)
    AC = DS.autoclust_family(repo, log)
    C.write_csv(AC, out / "autoclust_family_group_reliance.csv")

    X = SIG.merge(MF[["family", "new46_cvi_star_share", "new46_selected_share",
                      "recall_New46", "cvi_star_accuracy"]]
                  .rename(columns={"new46_cvi_star_share": "ml2dac_new46_cvi_star_share",
                                   "new46_selected_share": "ml2dac_new46_selected_share",
                                   "recall_New46": "ml2dac_new46_recall",
                                   "cvi_star_accuracy": "ml2dac_cvi_star_accuracy"}),
                  on="family", how="left")
    acp = AC.pivot_table(index="family", columns="group", values="rmse_increase_pct")
    acp.columns = ["autoclust_reliance_%s" % c for c in acp.columns]
    X = X.merge(acp.reset_index(), on="family", how="left")
    C.write_csv(X, out / "cross_method_family_mechanism.csv")
    XS = SSIG.merge(MS[["subfamily", "new46_cvi_star_share", "new46_selected_share",
                        "cvi_star_accuracy"]], on="subfamily", how="left")
    XS["autoclust_reliance"] = np.nan
    XS["autoclust_na_reason"] = ("subfamily grouped reliance not computed: the "
                                 "held-out Split-1 population gives too few "
                                 "independent datasets per subfamily")
    C.write_csv(XS, out / "cross_method_subfamily_mechanism.csv.gz", gz=True)

    # ------------------------------------------------------ pattern vs image
    def pvi(T: pd.DataFrame, key: str) -> pd.DataFrame:
        rows = []
        for g, s in T.groupby("group"):
            for sg in ("New46-Pattern", "New46-Image"):
                m = s[s.subgroup == sg]
                if not len(m):
                    continue
                rows.append({
                    key: g, "parent_family": s["parent_family"].iloc[0],
                    "subgroup": sg, "n_metrics": int(len(m)),
                    "mean_utility": float(m["U_group"].mean()),
                    "best_metric": str(m.nsmallest(1, "utility_rank_group")["metric_id"].iloc[0]),
                    "best_utility": float(m["U_group"].max()),
                    "top5_count": int((m["utility_rank_group"] <= 5).sum()),
                    "selection_share": float(m["S_group"].sum() / s["S_group"].sum()),
                    "mean_enrichment": float(m.loc[np.isfinite(m["enrichment"]),
                                                   "enrichment"].mean()),
                    "mean_DeltaU": float(m["DeltaU"].mean()),
                })
        return pd.DataFrame(rows)

    PVF = pvi(FAM, "family")
    PVS = pvi(SUB, "subfamily")
    PVF["n_tier_a"] = [int(((fa.group == r.family) & (fa.subgroup == r.subgroup)).sum())
                       for r in PVF.itertuples()]
    PVS["n_tier_a"] = [int(((sa.group == r.subfamily) & (sa.subgroup == r.subgroup)).sum())
                       for r in PVS.itertuples()]
    C.write_csv(PVF, out / "pattern_vs_image_family.csv")
    C.write_csv(PVS, out / "pattern_vs_image_subfamily.csv.gz", gz=True)

    # ---------------------------------------------------------- verdict
    n_fam_A = int((SIG["n_tier_a_unique"] > 0).sum())
    n_fam_B = int((SIG["n_tier_b"] > 0).sum())
    n_sub_A = int(SSIG.groupby("subfamily")["n_tier_a"].max().gt(0).sum())
    n_sub_B = int(SSIG.groupby("subfamily")["n_tier_b"].max().gt(0).sum())
    align = float(AF["spearman_util_sel_all"].mean())
    align_new = float(AF["spearman_util_sel_New46"].mean())
    assoc_sel = float(ASSOC.loc[ASSOC.signal == "new46_selection_share",
                                "spearman_vs_full60_gain"].iloc[0]) if len(ASSOC) else np.nan
    have_core = len(core) > 0
    have_fam = n_fam_A >= 6
    have_sub = n_sub_A >= 20
    if have_core and have_fam and align > 0.3:
        verdict = "SUPPORTED"
    elif have_core and (n_fam_A >= 1 or n_sub_A >= 1):
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    V = {
        "hypothesis": "shared global core + family/subfamily-specific specialists",
        "verdict": verdict,
        "decided_by": "predeclared quantities only; no narrative adjustment",
        "core": {"definition": coredef["definition"], "metrics": core,
                 "n": len(core), "sensitivity": coredef["sensitivity"]},
        "family": {
            "n_families": int(len(SIG)),
            "tier_a_pairs": int(len(fa)),
            "tier_a_unique_pairs": int(fa["unique_specialist"].sum()),
            "tier_b_pairs": int(len(fb)),
            "families_with_tier_a_unique": n_fam_A,
            "families_with_tier_b": n_fam_B,
            "median_specialists_per_family": float(SIG["n_tier_a"].median()),
            "families_without_any_new46_specialist": int(
                ((SIG["n_tier_a"] == 0) & (SIG["n_tier_b"] == 0)).sum()),
        },
        "subfamily": {
            "n_subfamilies": int(len(SSIG)),
            "eligible": int(scnt.eligible_for_inference.sum()),
            "tier_a_pairs": int(len(sa)), "tier_b_pairs": int(len(sb)),
            "subfamilies_with_tier_a": n_sub_A, "subfamilies_with_tier_b": n_sub_B,
            "subfamily_only_tier_a": int(sa["subfamily_only"].sum()) if len(sa) else 0,
        },
        "selection_composition": {
            "core_share": float(SIG["core_selection_share"].mean()),
            "unique_specialist_share": float(SIG["unique_specialist_selection_share"].mean()),
            "new46_share": float(SIG["new46_selection_share"].mean()),
        },
        "alignment": {"family_mean_spearman_all": align,
                      "family_mean_spearman_new46": align_new},
        "downstream_association": {
            "new46_selection_share_vs_full60_gain": assoc_sel,
            "n_families": 12, "evidence_level": "2 (association, descriptive)"},
        "jaccard": {"utility_top5": ju, "selection_top5": js,
                    "specialist_sets": jspec},
        "evidence_levels": {
            "1_causal_bundle": "Full60 vs Head14; AutoClust/ML2DAC A-arm contrasts",
            "2_association": "everything computed in Stage 3C",
            "3_single_metric_causal": "NOT AVAILABLE"},
    }
    C.write_json(V, out / "hypothesis_verdict.json")
    log("VERDICT: %s | core=%d | families with unique Tier-A: %d/12 | "
        "subfamilies with Tier-A: %d/86" % (verdict, len(core), n_fam_A, n_sub_A))

    # -------------------------------------------------------- provenance
    prov = dict(D.provenance)
    prov.update({
        "stage": "3C", "created_at": C.now(), "source_commit": commit,
        "analysis_only": True, "clustering_runs": 0, "models_trained": 0,
        "mkr_rebuilds": 0, "external_datasets": 0,
        "metric_inventory_modified": False,
        "stage3b_outputs_overwritten": False,
        "downstream": gprov,
        "bootstrap": {"family_B": SA.BOOT_FAMILY, "subfamily_B": SA.BOOT_SUBFAMILY,
                      "seed": SA.SEED},
        "permutation": {"family": SA.PERM_FAMILY, "subfamily": SA.PERM_SUBFAMILY,
                        "seed": SA.SEED,
                        "family_scheme": "shuffle family labels across datasets, "
                                         "sizes preserved",
                        "subfamily_scheme": "shuffle subfamily labels WITHIN each "
                                            "parent family only"},
        "min_n_subfamily_inference": SA.MIN_N_SUBFAMILY,
        "population_note": (
            "specialisation is measured on Splits 2-16 (OOF utility population); "
            "the Full60-vs-Head14 downstream gain is measured on Split 1. The two "
            "signals are therefore independent by construction, and their "
            "association is a family/subfamily-level statement, never per-dataset."),
        "evidence_levels": V["evidence_levels"],
    })
    C.write_json(prov, out / "provenance.json")
    C.write_json({
        "stage": "3C", "experiment": "family/subfamily metric specialization",
        "created_at": C.now(), "source_commit": commit,
        "n_families": 12, "n_subfamilies": int(len(SSIG)),
        "n_live_metrics": len(D.all_live),
        "n_metrics_utility_space": len(D.metrics),
        "verdict": verdict, "core_size": len(core),
    }, out / "experiment_manifest.json")
    log("Stage 3C complete -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
