"""Build the paper's LaTeX tables from the evidence tables. Presentation only: no recomputation.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_paths import (AC, EVI, FAM_PLAIN, IDBCFG, MAC, REPO, RRPREP, S1, S2,
                       S2B5,
                       SR, TAB, TEXNAME, record, write_prov)

EV = json.loads((EVI / "evidence_summary.json").read_text())


def esc(s):
    s = str(s)
    for a, b in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def code(s):
    """Typewriter; long identifiers may break after an underscore."""
    return r"\code{%s}" % esc(s).replace("\\_", "\\_\\allowbreak{}")


def w(name, lines):
    (TAB / (name + ".tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  table:", name)


def sig(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "--"
    if p < 1e-4:
        return "$<10^{-4}$"
    return "%.3g" % p


def metric_master():
    """Metric master table with a derived full-coverage rank column."""
    mm = pd.read_csv(SR / "metric_master_table.csv")
    mm["fc_rank"] = np.nan
    fc = mm[mm.full_coverage].sort_values("available_case_utility_macro",
                                          ascending=False)
    mm.loc[fc.index, "fc_rank"] = np.arange(1, len(fc) + 1)
    return mm


# ============================================================ MAIN TABLE 1
def qfmt(q):
    """BH-adjusted q for a table cell."""
    if q is None or not np.isfinite(q):
        return "--"
    return "$<10^{-4}$" if q < 1e-4 else "%.4f" % q


EXTBASE = REPO / "results_analysis" / "clustering_repository" / \
    "external_baselines"


def tab_setting():
    L = [r"\begin{table}[t]", r"\centering\footnotesize",
         r"\caption{\textbf{Criterion adaptation across automated clustering "
         r"systems.} Adapting the clustering criterion to the dataset is "
         r"established prior work; \clustopt{} differs in the granularity and "
         r"semantics of the adapted object, not in whether one is adapted. "
         r"``Granularity'' is what the learned component emits per dataset.}",
         r"\label{tab:setting}",
         # the System column must hold "AutoML4Clust" -- an unbreakable
         # 12-character word -- on one line, or it overflows its box and
         # visually joins the next cell ("AutoML4Clustno"). The table had
         # ~40pt of unused width, so the extra 0.04 comes out of that slack
         # and every prose column keeps its v15 width, leaving the row
         # heights and therefore the page break unchanged.
         r"\begin{tabular}{@{}p{0.145\linewidth}p{0.115\linewidth}"
         r"p{0.265\linewidth}p{0.15\linewidth}p{0.135\linewidth}@{}}",
         r"\toprule",
         r"System & Criterion adapted? & Learned component and granularity & "
         r"Criterion applies to & Also adapted \\",
         r"\midrule",
         r"AutoML4Clust & no & index supplied by the user & --- & search "
         r"budget, algorithm \\",
         r"\addlinespace[1.5pt]",
         r"\citet{muravyov2017metaclust} & yes & meta-learned prediction of "
         r"\emph{one} validation index & configuration search & algorithm \\",
         r"\addlinespace[1.5pt]",
         r"cSmartML & yes & meta-learned selection of algorithm \emph{and} "
         r"evaluation criteria; a criterion \emph{set} per dataset, whose "
         r"members enter as separate objectives & evolutionary tuning against "
         r"multiple objectives & algorithm, hyperparameters \\",
         r"\addlinespace[1.5pt]",
         r"AutoCluster & yes & meta-learned algorithm shortlist; tuned under "
         r"\emph{several} CVIs separately, partitions then combined by a "
         r"\emph{fixed} majority-voting rule & each grid search under its own "
         r"CVI & algorithm, output ensemble \\",
         r"\addlinespace[1.5pt]",
         r"ML2DAC & yes & meta-learned choice of \emph{one} CVI $+$ warm start "
         r"and a reduced search space & Bayesian optimisation & search region, "
         r"algorithm \\",
         r"\addlinespace[1.5pt]",
         r"AutoClust & yes, but \emph{globally} & $k$-NN algorithm selection "
         r"over CVI-valued meta-features $+$ one \emph{global} network from a "
         r"candidate's CVI vector to predicted ARI & Bayesian optimisation "
         r"(TPE) & algorithm, hyperparameters \\",
         r"\addlinespace[2.5pt]",
         r"\clustopt{} & yes & predicted \emph{continuous utility profile} "
         r"over the whole index vocabulary; objective composed from a weighted "
         r"top-$k$ of that profile, recomposed per dataset view & configuration "
         r"search & slate reranking (learned separately) \\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_setting", L)


# ============================================================ MAIN TABLE 2
def tab_controlled():
    prog = pd.read_csv(EVI / "controlled_progression.csv")
    m = pd.read_csv(S2 / "autoclust_matched_comparison.csv").iloc[0]
    v21 = pd.read_csv(S2 / "v2_vs_v1.csv").iloc[0]
    v2f = pd.read_csv(S2 / "v2_vs_fixed.csv").iloc[0]
    cp = EV["controlled_progression"]

    # the best mean among DEPLOYABLE rows (the oracle row is a ceiling)
    dep = prog[prog["key"] != "D"]
    best_dep = float(dep["mean"].max())

    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{\textbf{Controlled Split-1 replay} (1{,}055 datasets, "
         r"mean Best-View ARI, percentile bootstrap intervals). The "
         r"first two rows are the same policy on the same replay, so they "
         r"isolate reranking exactly. \textbf{Bold} marks the best deployable "
         r"mean; the policy oracle is a ceiling, not a system. Every paired "
         r"contrast is given in the text and in "
         r"Appendix~\ref{app:controlled}.}",
         r"\label{tab:controlled}",
         r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
         r"Configuration & Mean & 95\% interval \\", r"\midrule"]
    for _, r in prog.iterrows():
        ci = ("$[%.3f,\\,%.3f]$" % (r["lo"], r["hi"])
              if np.isfinite(r["lo"]) else "--")
        mean = ("\\textbf{%.4f}" % r["mean"]
                if (r["key"] != "D" and float(r["mean"]) == best_dep)
                else "%.4f" % r["mean"])
        L.append("%s & %s & %s \\\\" % (esc(r["label"]), mean, ci))
    L += [r"\addlinespace[2pt]",
          "AutoClust Extended --- matched search domain & %.4f & -- \\\\"
          % float(m.autoclust_reference_value),
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_controlled", L)
    record("tab_controlled",
           ["four_level_online_table.csv", "per_policy_results.csv",
            "v2_vs_v1.csv", "v2_vs_fixed.csv",
            "autoclust_matched_comparison.csv"],
           "stored means, intervals and p-values")


# ============================================================ MAIN TABLE 3
def tab_external():
    mm = pd.read_csv(SR / "method_master_table.csv").set_index("short")
    pc = pd.read_csv(SR / "paired_comparisons.csv")
    tv = pd.read_csv(SR / "runtime_performance_tradeoff.csv").set_index("short")

    def find(a, b):
        for _, r in pc.iterrows():
            if a in str(r.method_a) and b in str(r.method_b):
                return r
        return None

    L = [r"\begin{table}[t]", r"\centering\footnotesize",
         r"\caption{\textbf{Independent benchmark}, all 14 arms (50 datasets, "
         r"mean Best-View ARI, bootstrap over the 40 dependence groups; median "
         r"three-view runtime, Appendix~\ref{app:protocol}). \textbf{Bold} "
         r"marks the largest deployable point estimate; paired statistical "
         r"conclusions are reported separately, and bold does not denote "
         r"significance. \emph{Code} is the arm label used in Figure~\ref{fig:fig4_external}. In neither baseline is the best arm the extended one. "
         r"Full-precision intervals, the source-balanced macro and every paired "
         r"contrast with win/tie/loss counts are in "
         r"Appendix~\ref{app:external-arms}.}",
         r"\label{tab:external}",
         r"\begin{tabular}{@{}l@{\hspace{4pt}}lcr@{\hspace{5pt}}r@{\hspace{1.4em}}l@{\hspace{4pt}}lcr@{\hspace{5pt}}r@{}}", r"\toprule",
         r"Code & Arm & Mean & 95\% int. & RT & "
         r"Code & Arm & Mean & 95\% int. & RT \\",
         r"\midrule"]

    # the block header already names the system, so rows carry only the arm
    SHORT = {"C5": r"\textsc{ppv2}$+$R", "C4": r"\textsc{knn}$+$R",
             "C3": r"\textsc{mlp}$+$R", "C2": r"\textsc{ppv1}",
             "C1": r"\textsc{knn}", "C0": r"\textsc{mlp}",
             "A0": "original", "A1": r"$+$established",
             "A2": r"$+$New46", "A3": "extended",
             "M0": "original", "M1": r"$+$established",
             "M2": r"$+$New46", "M3": "extended"}

    def cell(k):
        """One arm as five aligned cells, led by the code used in Figure 4."""
        r = mm.loc[k]
        nm, val = SHORT[k], "%.4f" % r.bestview_ari_mean
        cd = k
        if k == "C5":
            cd = r"\textbf{%s}" % k
            nm, val = r"\textbf{%s}" % nm, r"\textbf{%s}" % val
        return "%s & %s & %s & {\\scriptsize$[%.2f,%.2f]$} & %.0f" % (
            cd, nm, val, r.bestview_ci_lo, r.bestview_ci_hi,
            tv.loc[k, "three_view_runtime_median"])

    def head(g):
        return r"\emph{%s} & & & & " % g

    # left block: ClustOpt (6). right block: AutoClust (4) then ML2DAC (4).
    left = [head(r"\clustopt{}")] + [cell(k) for k in
                                     ("C5", "C4", "C3", "C2", "C1", "C0")]
    right = ([head("AutoClust")] + [cell(k) for k in ("A0", "A1", "A2", "A3")]
             + [head("ML2DAC")] + [cell(k) for k in ("M0", "M1", "M2", "M3")])
    blank = " & & & & "
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else blank
        b = right[i] if i < len(right) else blank
        L.append("%s & %s \\\\" % (a, b))

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_external", L)
    record("tab_external", ["method_master_table.csv", "paired_comparisons.csv",
                            "runtime_performance_tradeoff.csv"],
           "stored means, group-aware intervals and medians, all 14 arms")


# ============================================================ MAIN TABLE 4
def tab_metrics():
    mm = metric_master()
    core = EV["stage3"]["core_metrics"]
    grp = mm.groupby("metric_group").available_case_utility_macro.agg(
        ["mean", "size"])
    show = ["cvdd", "silhouette", "noise_aware_silhouette", "gridness_fft_acf",
            "cvnn", "dcsi", "cdbw"]
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{\textbf{Index utility on the independent corpus.} Utility "
         r"is available-case: computed on the units where the index is "
         r"defined, and never coverage-adjusted. Rank is over all 67 primary "
         r"indices; FC-rank is over the 49 with complete coverage. The three "
         r"indices marked $\bullet$ are the generalist core identified on the "
         r"development repository. The full ranking is in "
         r"Appendix~\ref{app:metrics}.}",
         r"\label{tab:metrics}",
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Index & Group & Utility & Coverage & Rank & FC-rank \\",
         r"\midrule"]
    d = mm.set_index("metric_id")
    for m in show:
        if m not in d.index:
            continue
        r = d.loc[m]
        mark = r"$\bullet$~" if m in core else ""
        fc = ("%d" % r.fc_rank) if np.isfinite(r.fc_rank) else "--"
        L.append("%s%s & %s & %.4f & %d/150 & %d & %s \\\\" % (
            mark, code(m), r.metric_group, r.available_case_utility_macro,
            int(r.units_with_valid_utility), int(r.available_case_macro_rank),
            fc))
    L += [r"\midrule",
          r"\multicolumn{6}{@{}l}{\emph{Group means over all 67 primary "
          r"indices}} \\", r"\addlinespace[1.5pt]"]
    for g in ("Original", "Established", "New46", "Modern"):
        if g in grp.index:
            L.append("%s (%d indices) & & %.4f & & & \\\\" % (
                g, int(grp.loc[g, "size"]), grp.loc[g, "mean"]))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_metrics", L)
    record("tab_metrics", ["metric_master_table.csv"],
           "stored available-case utilities, coverage and ranks")


# ======================================================= APPENDIX TABLES
def tab_app_stage1():
    L = []
    for tag, sub, vc, cap in (
            ("offline", "offline_fixed32", "mean_best_view_ari",
             "fixed budget of 32 configurations"),
            ("online", "online_fixed50", "mean_ari",
             "online search, 50 evaluations")):
        s = pd.read_csv(S1 / sub / ("%s_2x3_overall_summary.csv" % tag))
        c = pd.read_csv(S1 / sub / ("%s_2x3_paired_contrasts.csv" % tag))
        L += [r"\begin{table}[htbp]", r"\centering\small",
              r"\caption{Stage-1 factorial, %s. Arm codes: H/F = Head14 / "
              r"Full60; U/P/O = uniform / predicted / oracle weighting.}" % cap,
              r"\label{tab:app-stage1-%s}" % tag,
              r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
              r"Inventory & Weighting & Mean & 95\% interval & Deployable \\",
              r"\midrule"]
        for _, r in s.iterrows():
            L.append("%s & %s & %.4f & $[%.3f,\\,%.3f]$ & %s \\\\" % (
                r.metric_inventory, r.utility_strategy, r[vc], r.mean_ci_lo,
                r.mean_ci_hi, "yes" if r.deployable else "no (oracle)"))
        L += [r"\midrule",
              r"Contrast & Mean $\Delta$ & 95\% interval & Holm $p$ & "
              r"Rejected \\", r"\addlinespace[1.5pt]"]
        cc = c.dropna(subset=["hypothesis_id"])
        for _, r in cc.iterrows():
            L.append("%s & $%+.4f$ & $[%+.4f,\\,%+.4f]$ & %s & %s \\\\" % (
                esc(r.hypothesis_id), r.mean_delta, r.mean_delta_ci_low,
                r.mean_delta_ci_high, sig(r.holm_p),
                "yes" if r.holm_reject else "no"))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    w("tab_app_stage1", L)


def tab_app_policies():
    p = pd.read_csv(EVI / "controlled_policy_prepost.csv")
    p = p.sort_values("reranked_mean_bestview", ascending=False)
    fixed = EV["controlled_progression"]["pre_reranker_policy"]
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{All 20 frozen policies on the controlled Split-1 replay, "
         r"before and after reranking. $\dagger$ marks the "
         r"development-selected fixed policy used as the deployable "
         r"baseline. Reranking improves every policy; the gain is largest "
         r"where the pre-reranker objective is weakest.}",
         r"\label{tab:app-policies}",
         r"\footnotesize",
         r"\begin{tabular}{@{}p{0.30\linewidth}ccccc@{}}", r"\toprule",
         r"Policy & No rerank & Reranked & Gain & 95\% interval & "
         r"Slate oracle \\", r"\midrule"]
    for _, r in p.iterrows():
        mark = r"$\dagger$" if r.policy_id == fixed else ""
        L.append("%s%s & %.4f & %.4f & $%+.4f$ & $[%+.3f,\\,%+.3f]$ & %.4f \\\\"
                 % (code(r.policy_id), mark, r.original_mean_bestview,
                    r.reranked_mean_bestview, r.absolute_gain, r.gain_ci_lo,
                    r.gain_ci_hi, r.visited_oracle_mean_bestview))
    L += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    w("tab_app_policies", L)


def tab_app_family():
    g = pd.read_csv(EVI / "controlled_family_deltas.csv")
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Controlled per-family means (descriptive; no family-level "
         r"inference is claimed). The comparator is \emph{AutoClust "
         r"Extended} under the matched search domain; families are ordered by "
         r"the \ppv{2}$-$AutoClust difference. The last column is the "
         r"remaining \emph{policy-selection} headroom to the reranked "
         r"\emph{policy} oracle across the evaluated policy portfolio; it is "
         r"not a gap on any single searched slate.}",
         r"\label{tab:app-family}",
         r"\begin{tabular}{@{}p{0.20\linewidth}cccccc@{}}", r"\toprule",
         r"Family & $n$ & Fixed+R & \ppv{2}+R & AutoClust & "
         r"$\Delta$ vs AC & Oracle$-$\ppv{2} \\", r"\midrule"]
    for _, r in g.iterrows():
        L.append("%s & %d & %.4f & %.4f & %.4f & $%+.4f$ & $%+.4f$ \\\\" % (
            esc(r.family), int(r.n), r.fixed, r.v2, r.autoclust, r.d_v2_ac,
            r.d_vbs_v2))
    L += [r"\midrule",
          r"\multicolumn{7}{@{}p{0.95\linewidth}}{\footnotesize Against "
          r"matched \emph{AutoClust Extended}, %d of 12 differences lie "
          r"within $\pm0.01$ ARI; the point estimate is positive in %d "
          r"families and negative in %d, and the largest difference in "
          r"either direction is $%.3f$. Signs are reported descriptively, "
          r"not as wins.} \\"
          % (EV["controlled_families"]["n_within_0.01_v2_vs_autoclust"],
             EV["controlled_families"]["n_clustopt_ahead"],
             EV["controlled_families"]["n_autoclust_ahead"],
             EV["controlled_families"]["max_abs_delta"]),
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_app_family", L)


def tab_app_repo():
    f = pd.read_csv(EVI / "repo_family_summary.csv")
    s = pd.read_csv(EVI / "repo_subfamily_summary.csv")
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{The controlled repository: 12 structural families over "
         r"the whole 16-split corpus.}", r"\label{tab:app-repo-family}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Family & Subfamilies & Datasets & Points (min--max) & "
         r"Median points & $k$ \\", r"\midrule"]
    for _, r in f.sort_values("family").iterrows():
        L.append("%s & %d & %d & %d--%d & %d & %d--%d \\\\" % (
            esc(r.family), int(r.n_subfamilies), int(r.n_datasets),
            int(r.n_points_min), int(r.n_points_max),
            int(r.n_points_median), int(r.k_min), int(r.k_max)))
    L += [r"\midrule",
          "Total & %d & %d & %d--%d & & %d--%d \\\\" % (
              int(f.n_subfamilies.sum()), int(f.n_datasets.sum()),
              int(f.n_points_min.min()), int(f.n_points_max.max()),
              int(f.k_min.min()), int(f.k_max.max())),
          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    L += [r"\footnotesize",
          r"\begin{longtable}{@{}p{0.19\linewidth}p{0.24\linewidth}rp{0.12\linewidth}p{0.19\linewidth}@{}}",
          r"\caption{All 86 subfamilies of the controlled repository.}"
          r"\label{tab:app-repo-subfamily} \\", r"\toprule",
          r"Family & Subfamily & Datasets & Points & Difficulty levels \\",
          r"\midrule", r"\endfirsthead", r"\toprule",
          r"Family & Subfamily & Datasets & Points & Difficulty levels \\",
          r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in s.sort_values(["family", "subfamily_id"]).iterrows():
        L.append("%s & %s & %d & %d--%d & %s \\\\" % (
            esc(r.family), code(r.subfamily_id), int(r.n_datasets),
            int(r.n_points_min), int(r.n_points_max),
            esc(str(r.difficulties).replace("very_hard", "v.hard"))))
    L += [r"\end{longtable}", r"\normalsize"]
    w("tab_app_repo", L)


def tab_app_arms():
    a = pd.read_csv(EVI / "arm_catalogue_stage4.csv")
    pol = pd.read_csv(EVI / "policy_set.csv")
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{The 14 frozen external arms. Within each baseline family "
         r"the only thing that varies is the index inventory, so that column "
         r"is what distinguishes A0--A3 and M0--M3; live counts are the "
         r"indices actually computable on this corpus, against 7/17/53/63 "
         r"requested. Note the two inventories are not nested: a baseline's "
         r"``Original'' is the 7 classical indices, whereas \headfourteen{} "
         r"and \fullsixty{} contain only 4 of those 7 "
         r"(Appendix~\ref{app:internal}). Trace groups: C0/C3 share "
         r"one physical search, C1/C4 share another, C2 and C5 each execute "
         r"their own, A0--A3 share one, M0--M3 share one. Sharing is "
         r"\emph{within} a group only; no candidate slate is shared across "
         r"systems.}", r"\label{tab:app-arms}",
         r"\footnotesize",
         r"\begin{tabular}{@{}cp{0.11\linewidth}p{0.20\linewidth}"
         r"p{0.11\linewidth}cp{0.10\linewidth}p{0.12\linewidth}@{}}",
         r"\toprule",
         r"Code & System & Index inventory (live) & Utility source & $k$ & "
         r"Policy & Rerank \\", r"\midrule"]
    INV = {"A0": r"Original (7)", "A1": r"$+$Established (17)",
           "A2": r"$+$New46 (51)", "A3": r"Extended (61)",
           "M0": r"Original (7)", "M1": r"$+$Established (17)",
           "M2": r"$+$New46 (51)", "M3": r"Extended (61)"}
    for _, r in a.iterrows():
        inv = INV.get(r.code, r"\fullsixty{} (60)")
        L.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
            r.code, r.framework, inv, esc(r.utility_source), esc(r.top_k),
            esc(r.policy), esc(r.reranker)))
    L += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}",
          "", r"\begin{table}[htbp]", r"\centering\footnotesize",
          r"\caption{The frozen policy set: 2 utility predictors $\times$ 5 "
          r"subset rules $\times$ 2 weighting schemes $=$ 20 policies, served "
          r"by 10 rerankers (one per predictor $\times$ subset rule).}",
          r"\label{tab:app-policyset}",
          r"\begin{tabular}{@{}p{0.34\linewidth}lll@{}}", r"\toprule",
          r"Policy & Predictor & Subset rule & Weighting \\", r"\midrule"]
    for _, r in pol.sort_values(["utility_source", "k_mode",
                                 "weighting_mode"]).iterrows():
        L.append("%s & %s & %s & %s \\\\" % (
            code(r.policy_id), r.utility_source.upper(), esc(r.k_mode),
            esc(r.weighting_mode)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_app_arms", L)


def tab_app_predictor():
    c = pd.read_csv(EVI / "predictor_head14_vs_full60.csv").set_index("metric")
    v = pd.read_csv(EVI / "predictor_by_view.csv")
    rows = [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "$R^2$"),
            ("spearman_mean", "Spearman $\\rho$"),
            ("kendall_mean", "Kendall $\\tau$"),
            ("pairwise_ranking_accuracy", "Pairwise ranking accuracy"),
            ("ndcg@all", "NDCG (all)"), ("ndcg@10", "NDCG@10"),
            ("ndcg@5", "NDCG@5"), ("ndcg@3", "NDCG@3"),
            ("top10_overlap", "Top-10 overlap"),
            ("top5_overlap", "Top-5 overlap"),
            ("top3_overlap", "Top-3 overlap"),
            ("top1_accuracy", "Top-1 accuracy"),
            ("rank_displacement_mean", "Mean rank displacement")]
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Utility-prediction quality on the held-out Split-1 "
         r"records ($n=3{,}165$ dataset--view records). \textbf{The two "
         r"columns are not comparable as a quality claim}: every top-$k$ and "
         r"NDCG@$k$ quantity is computed over inventories of different size, "
         r"so $k=10$ covers $71\%$ of \headfourteen{} but only $17\%$ of "
         r"\fullsixty{}. Inventory-independent quantities (MAE, RMSE, "
         r"Spearman, pairwise accuracy, NDCG over all indices) are "
         r"comparable.}", r"\label{tab:app-predictor}",
         r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Quantity & \fullsixty{} & \headfourteen{} & Comparable? \\",
         r"\midrule"]
    indep = {"mae", "rmse", "r2", "spearman_mean", "kendall_mean",
             "pairwise_ranking_accuracy", "ndcg@all"}
    for k, lab in rows:
        if k not in c.index:
            continue
        r = c.loc[k]
        h = ("%.4f" % r.head14_new) if np.isfinite(r.head14_new) else "--"
        L.append("%s & %.4f & %s & %s \\\\" % (
            lab, r.full60_historical, h, "yes" if k in indep else "no"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
          r"\begin{table}[htbp]", r"\centering\small",
          r"\caption{\fullsixty{} predictor quality by view.}",
          r"\label{tab:app-predictor-view}",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r"View & MAE & Top-1 accuracy & NDCG@10 & Records \\", r"\midrule"]
    for _, r in v.iterrows():
        L.append("%s & %.4f & %.4f & %.4f & %d \\\\" % (
            code(r.view_id), r["mae"], r["top1_accuracy"], r["ndcg@10"],
            int(r.n_samples)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_app_predictor", L)


def tab_app_stage3():
    fa = pd.read_csv(EVI / "stage3_family_tierA.csv")
    sa = pd.read_csv(EVI / "stage3_subfamily_tierA.csv")
    al = pd.read_csv(EVI / "stage3_alignment.csv")
    pv = pd.read_csv(EVI / "stage3_pattern_vs_image.csv")
    rc = pd.read_csv(EVI / "stage3_family_recall.csv")

    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{All 14 Tier-A family-level specialisation relations. "
         r"$\Delta U$ is the utility uplift of the index inside the family "
         r"against the same index outside it; $q$ is the "
         r"Benjamini--Hochberg-adjusted value within the pre-specified test "
         r"family. Enrichment is the ratio of in-family to out-of-family "
         r"selection rate.}", r"\label{tab:app-tierA-family}",
         r"\footnotesize",
         r"\begin{tabular}{@{}p{0.15\linewidth}p{0.19\linewidth}rrp{0.135\linewidth}rr@{}}",
         r"\toprule",
         r"Family & Index & $n$ & $\Delta U$ & 95\% interval & $q$ & Enrichment \\",
         r"\midrule"]
    for _, r in fa.sort_values(["group", "metric_id"]).iterrows():
        L.append("%s & %s & %d & $%+.4f$ & $[%+.3f,\\,%+.3f]$ & %s & %.2f \\\\" % (
            esc(FAM_PLAIN.get(r.group, r.group)), code(r.metric_id),
            int(r.n_datasets), r.DeltaU, r.DeltaU_ci_low, r.DeltaU_ci_high,
            qfmt(r.DeltaU_q_bh), r.enrichment))
    L += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}",
          "", r"\scriptsize",
          r"\begin{longtable}{@{}p{0.21\linewidth}p{0.13\linewidth}"
          r"p{0.18\linewidth}rrp{0.14\linewidth}c@{}}",
          r"\caption{All 42 Tier-A subfamily-level specialisation relations; "
          r"the comparator is the sibling subfamilies of the same family. "
          r"``Sub-only'' marks the 27 relations that are invisible at family "
          r"level.}\label{tab:app-tierA-subfamily} \\", r"\toprule",
          r"Subfamily & Family & Index & $n$ & $\Delta U$ & 95\% interval & "
          r"Sub-only \\", r"\midrule", r"\endfirsthead", r"\toprule",
          r"Subfamily & Family & Index & $n$ & $\Delta U$ & 95\% interval & "
          r"Sub-only \\", r"\midrule", r"\endhead", r"\bottomrule",
          r"\endfoot"]
    for _, r in sa.sort_values(["parent_family", "group"]).iterrows():
        L.append("%s & %s & %s & %d & $%+.4f$ & $[%+.3f,\\,%+.3f]$ & %s \\\\" % (
            code(r.group), esc(FAM_PLAIN.get(r.parent_family, r.parent_family)),
            code(r.metric_id), int(r.n_datasets), r.DeltaU, r.DeltaU_ci_low,
            r.DeltaU_ci_high, "yes" if r.subfamily_only else "--"))
    L += [r"\end{longtable}", r"\normalsize", "",
          r"\begin{table}[htbp]", r"\centering\footnotesize",
          r"\caption{Alignment between an index's true utility and how often "
          r"\clustopt{} selects it, within each family (Spearman over the 60 "
          r"indices of the deployed inventory), together with how much of the "
          r"true top-5 the deployed selector recovers.}",
          r"\label{tab:app-alignment}",
          r"\begin{tabular}{@{}p{0.26\linewidth}ccccc@{}}", r"\toprule",
          r"Family & $n$ & $\rho$ all & $\rho$ New46 & Top-5 recall & "
          r"Top-1 hit \\", r"\midrule"]
    rcx = rc.set_index("group")
    for _, r in al.iterrows():
        q = rcx.loc[r.group]
        L.append("%s & %d & %.3f & %.3f & %.3f & %.3f \\\\" % (
            esc(FAM_PLAIN.get(r.group, r.group)), int(r.n_datasets),
            r.spearman_util_sel_all, r.spearman_util_sel_New46,
            q.true_top5_recall, q.true_top1_hit))
    L += [r"\midrule",
          "Mean & & %.3f & %.3f & %.3f & %.3f \\\\" % (
              al.spearman_util_sel_all.mean(),
              al.spearman_util_sel_New46.mean(),
              rc.true_top5_recall.mean(), rc.true_top1_hit.mean()),
          r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
          r"\begin{table}[htbp]", r"\centering\small",
          r"\caption{Pattern-based against image-based new indices, pooled "
          r"over the 12 families.}", r"\label{tab:app-pattern-image}",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r"Subgroup & Indices & Mean utility & Selection share & "
          r"Tier-A relations \\", r"\midrule"]
    for g, sub in pv.groupby("subgroup"):
        L.append("%s & %d & %.4f & %.4f & %d \\\\" % (
            esc(g), int(sub.n_metrics.iloc[0]), sub.mean_utility.mean(),
            sub.selection_share.mean(), int(sub.n_tier_a.sum())))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_app_stage3", L)


def tab_app_external_arms():
    mm = pd.read_csv(SR / "method_master_table.csv")
    pc = pd.read_csv(SR / "paired_comparisons.csv")
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{All 14 external arms on the independent benchmark.}",
         r"\label{tab:app-external-arms}",
         r"\footnotesize",
         r"\begin{tabular}{@{}cp{0.24\linewidth}rp{0.17\linewidth}rrr@{}}",
         r"\toprule",
         r"Code & Arm & Best-View & 95\% interval & Single & "
         r"Src-bal. & Runtime \\", r"\midrule"]
    tv = pd.read_csv(SR / "runtime_performance_tradeoff.csv").set_index("short")
    for _, r in mm.iterrows():
        L.append("%s & %s & %.4f & $[%.3f,\\,%.3f]$ & %.4f & %.4f & %.1f \\\\"
                 % (r.short, esc(r.arm_description), r.bestview_ari_mean,
                    r.bestview_ci_lo, r.bestview_ci_hi, r.xy_2d_ari_mean,
                    r.bestview_source_balanced,
                    tv.loc[r.short, "three_view_runtime_median"]))
    L += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}",
          "", r"\begin{table}[htbp]", r"\centering\footnotesize",
          r"\caption{Every \emph{distinct} pre-specified external paired "
          r"contrast on Best-View ARI; intervals resample the 40 dependence "
          r"groups. The frozen comparison file lists \code{C5 PPv2\_R vs C2 "
          r"PPv1\_noR} twice, once under a second interpretive label; it is "
          r"one numerical comparison and appears here once. Note that it "
          r"changes the policy target \emph{and} adds reranking, so it is not "
          r"an isolated reranker effect: the isolated reranker contrasts are "
          r"C3$-$C0 and C4$-$C1.}",
          r"\label{tab:app-external-paired}",
          r"\begin{tabular}{@{}p{0.30\linewidth}ccccc@{}}", r"\toprule",
          r"Contrast & Mean $\Delta$ & 95\% interval & $p$ & W/T/L & "
          r"Rank-biserial \\", r"\midrule"]
    _seen_stats = set()
    for _, r in pc.iterrows():
        key = (round(float(r.mean_delta), 10), round(float(r.ci_lo), 10),
               round(float(r.ci_hi), 10), int(r.wins), int(r.ties),
               int(r.losses))
        if key in _seen_stats:
            continue          # same numerical comparison under a second label
        _seen_stats.add(key)
        L.append("%s & $%+.4f$ & $[%+.3f,\\,%+.3f]$ & %s & %d/%d/%d & %.3f \\\\"
                 % (esc(r.contrast), r.mean_delta, r.ci_lo, r.ci_hi,
                    sig(r.p_value), int(r.wins), int(r.ties), int(r.losses),
                    r.rank_biserial))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_app_external_arms", L)


def tab_app_corpus():
    """N.1 corpus metadata and N.2 headline results, as two tables."""
    m = pd.read_csv(EVI / "external_corpus_metadata.csv")
    p = pd.read_csv(EVI / "external_per_dataset.csv").drop(
        columns=["source"], errors="ignore")
    man = pd.read_csv(IDBCFG / "dataset_manifest.csv")
    keep = ["dataset_id", "n_classes", "licence", "openml_data_id",
            "generator", "difficulty", "seed", "representation_checksum"]
    man = man[[c for c in keep if c in man.columns]]
    d = m.merge(man, on="dataset_id", how="left",
                suffixes=("", "_man")).merge(p, on="dataset_id", how="left")

    src = {"fcps": "FCPS", "sklearn_shapes": "scikit-learn",
           "real_pca": "real (PCA)"}

    def ident(r):
        """The identifier that actually reconstructs this dataset."""
        oid = r.get("openml_data_id")
        if pd.notna(oid):
            return "OpenML %d" % int(oid)
        g, diff, sd = r.get("generator"), r.get("difficulty"), r.get("seed")
        if pd.notna(g):
            parts = [str(g)]
            if pd.notna(diff):
                parts.append(str(diff))
            if pd.notna(sd):
                parts.append("seed %d" % int(sd))
            return esc(", ".join(parts))
        if str(r.get("source")) == "fcps":
            return "FCPS suite"
        # one real-tabular source ships with scikit-learn and has no OpenML id
        lic = str(r.get("licence", ""))
        if "scikit-learn" in lic:
            return esc("scikit-learn bundled")
        return "--"

    def chk(r):
        c = r.get("representation_checksum")
        return code(str(c)[:8]) if pd.notna(c) else "--"

    def redist(r):
        v = str(r.get("redistribution", ""))
        if "ALLOWED" in v:
            return "allowed"
        if "RESTRICTED" in v:
            return "restricted"
        return "unclear"

    # ------------------------------------------------------------ N.1
    L = [r"\scriptsize",
         r"\begin{longtable}{@{}p{0.185\linewidth}p{0.075\linewidth}"
         r"p{0.130\linewidth}rrr@{\hspace{4pt}}c@{\hspace{6pt}}"
         r"p{0.105\linewidth}@{\hspace{6pt}}p{0.072\linewidth}@{}}",
         r"\caption{\textbf{N.1 --- the independent external corpus: "
         r"reproducibility metadata.} ``Identifier'' is what reconstructs the "
         r"dataset: an OpenML data id for real tabular sources, the generator "
         r"with its difficulty level and seed for the scikit-learn shapes, and "
         r"the named FCPS suite otherwise. $K$ is the number of ground-truth "
         r"classes. ``Dim'' is the dimensionality after cleaning and before "
         r"the projection to two dimensions; PCA is applied wherever that "
         r"exceeds two, and the recorded total explained variance is given "
         r"with it. ``Checksum'' is the first eight hex characters of the frozen "
         r"two-dimensional representation's checksum, which every system "
         r"loaded. ``Redist.'' records whether the \emph{raw source} may be "
         r"redistributed; prepared arrays are not released for any source, and "
         r"28 of the 50 sources are restricted. Per-dataset results are in "
         r"Table~\ref{tab:app-corpus-results}.}\label{tab:app-corpus} \\",
         r"\toprule",
         r"Dataset & Source & Identifier & $n$ & $K$ & Dim & PCA (EV) & "
         r"Checksum & Redist. \\",
         r"\midrule", r"\endfirsthead", r"\toprule",
         r"Dataset & Source & Identifier & $n$ & $K$ & Dim & PCA (EV) & "
         r"Checksum & Redist. \\",
         r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in d.sort_values(["source", "dataset_id"]).iterrows():
        ev = r.get("explained_variance_total")
        pca = ("yes (%.2f)" % ev) if (r.pca_applied and pd.notna(ev)) else (
            "yes" if r.pca_applied else "no")
        L.append("%s & %s & %s & %d & %s & %s & %s & %s & %s \\\\" % (
            code(r.dataset_id), src.get(r.source, r.source), ident(r),
            int(r.n_samples),
            ("%d" % r.n_classes) if pd.notna(r.get("n_classes")) else "--",
            ("%d" % r.dim_after_cleaning
             if np.isfinite(r.dim_after_cleaning) else "--"),
            pca, chk(r), redist(r)))
    L += [r"\end{longtable}", r"\normalsize"]

    # ------------------------------------------------------------ N.2
    L += ["", r"\scriptsize",
          r"\begin{longtable}{@{}p{0.34\linewidth}p{0.13\linewidth}rrrr@{}}",
          r"\caption{\textbf{N.2 --- headline per-dataset results.} Best-View "
          r"ARI on each external dataset for the deployed system and the three "
          r"reference arms. Metadata for the same datasets is in "
          r"Table~\ref{tab:app-corpus}.}\label{tab:app-corpus-results} \\",
          r"\toprule",
          r"Dataset & Source & C5 & C4 & A0 & M1 \\",
          r"\midrule", r"\endfirsthead", r"\toprule",
          r"Dataset & Source & C5 & C4 & A0 & M1 \\",
          r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in d.sort_values(["source", "dataset_id"]).iterrows():
        L.append("%s & %s & %.3f & %.3f & %.3f & %.3f \\\\" % (
            code(r.dataset_id), src.get(r.source, r.source),
            r.C5, r.C4, r.A0, r.M1))
    L += [r"\end{longtable}", r"\normalsize"]
    w("tab_app_corpus", L)
    record("tab_app_corpus",
           ["external_corpus_metadata.csv",
            "Independent_Domain_Benchmark/configs/dataset_manifest.csv",
            "external_per_dataset.csv"],
           "stored corpus metadata, identifiers, checksums and stored "
           "per-dataset Best-View ARIs")
def tab_app_runtime():
    r = pd.read_csv(SR / "runtime_method_summary.csv")
    tv = pd.read_csv(SR / "runtime_performance_tradeoff.csv").set_index("short")
    xy = pd.read_csv(SR / "runtime_xy2d_tradeoff.csv").set_index("short")
    dc = pd.read_csv(SR / "runtime_cost_decomposition.csv").set_index("short")
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Runtime by scope. \textbf{Pooled} is the median over all "
         r"$50\times3$ dataset--view records; \textbf{single view} is the "
         r"median over the \code{xy\_2d} records only; \textbf{three-view} is "
         r"the median over datasets of the summed cost of all three views, "
         r"which is the scope Best-View ARI requires. The three columns "
         r"answer different questions and are never mixed. Only physical "
         r"search and arm scoring carry non-zero values in the persisted "
         r"records; meta-feature extraction and model initialisation are "
         r"logged as zero and are not inferred.}",
         r"\label{tab:app-runtime}",
         r"\footnotesize",
         r"\begin{tabular}{@{}cccccc@{}}", r"\toprule",
         r"Code & Pooled (s) & Single view (s) & Three-view (s) & "
         r"Search (s) & Scoring (s) \\", r"\midrule"]
    for _, x in r.sort_values("short").iterrows():
        k = x.short
        L.append("%s & %.2f & %.2f & %.2f & %.2f & %.2f \\\\" % (
            k, x.pooled_median, xy.loc[k, "xy_2d_runtime_median"],
            tv.loc[k, "three_view_runtime_median"],
            dc.loc[k, "physical_median_s"], dc.loc[k, "scoring_median_s"]))
    L += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    w("tab_app_runtime", L)


def tab_app_metric_ranking():
    mm = metric_master().sort_values("available_case_macro_rank")
    L = [r"\footnotesize",
         r"\begin{longtable}{@{}cp{0.30\linewidth}lccccc@{}}",
         r"\caption{All 67 primary indices on the canonical candidate basis, "
         r"ordered by available-case utility. Coverage is the number of the "
         r"150 dataset--view units on which the index is defined; utility and "
         r"coverage are separate columns and no coverage-adjusted score is "
         r"formed.}\label{tab:app-metric-ranking} \\", r"\toprule",
         r"Rank & Index & Group & Utility & Coverage & FC-rank & Top-5 & "
         r"Top-10 \\", r"\midrule", r"\endfirsthead", r"\toprule",
         r"Rank & Index & Group & Utility & Coverage & FC-rank & Top-5 & "
         r"Top-10 \\", r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in mm.iterrows():
        fc = ("%d" % r.fc_rank) if np.isfinite(r.fc_rank) else "--"
        L.append("%d & %s & %s & %.4f & %d & %s & %d & %d \\\\" % (
            int(r.available_case_macro_rank), code(r.metric_id),
            r.metric_group, r.available_case_utility_macro,
            int(r.units_with_valid_utility), fc, int(r.top5_count),
            int(r.top10_count)))
    L += [r"\end{longtable}", r"\normalsize"]
    w("tab_app_metric_ranking", L)


def tab_app_new46():
    mm = pd.read_csv(SR / "new46_specialist_table.csv")
    mm = mm.sort_values("available_case_utility_macro", ascending=False)
    L = [r"\footnotesize",
         r"\begin{longtable}{@{}p{0.30\linewidth}cccccccc@{}}",
         r"\caption{The 46 newly designed structural indices, individually. "
         r"Utility is available-case on the canonical basis; coverage is the "
         r"number of the 150 dataset--view units on which the index is "
         r"defined; \emph{rank} is over all 67 primary indices; the last four "
         r"columns count the units on which the index is ranked first, or "
         r"appears in the top three, five or ten. The group mean is the lowest "
         r"of the four groups while several individual indices are the single "
         r"best available choice on particular units; both readings come from "
         r"this table.}\label{tab:app-new46} \\", r"\toprule",
         r"Index & Utility & FCPS & sk-learn & real & Cov. & Rank & \#1 & "
         r"Top-5 \\", r"\midrule", r"\endfirsthead", r"\toprule",
         r"Index & Utility & FCPS & sk-learn & real & Cov. & Rank & \#1 & "
         r"Top-5 \\", r"\midrule", r"\endhead", r"\bottomrule",
         r"\endfoot"]
    for _, r in mm.iterrows():
        L.append("%s & %.4f & %.3f & %.3f & %.3f & %d & %d & %d & %d \\\\" % (
            code(r.metric_id), r.available_case_utility_macro,
            r.utility_fcps, r.utility_sklearn_shapes, r.utility_real_pca,
            int(r.units_with_valid_utility),
            int(r.available_case_macro_rank), int(r.rank1), int(r.top5)))
    L += [r"\end{longtable}", r"\normalsize"]
    w("tab_app_new46", L)


def tab_app_search():
    sp = json.loads((EVI / "search_space.json").read_text())
    L = [r"\footnotesize",
         r"\begin{longtable}{@{}p{0.10\linewidth}p{0.22\linewidth}"
         r"p{0.60\linewidth}@{}}",
         r"\caption{The frozen configuration search space, per view. The true "
         r"cluster count never enters: every $k$-based algorithm searches "
         r"$k\in\{2,3,4,5\}$. Constrained-HDBSCAN blocks fix the component "
         r"count to $n$ and search the remaining parameters."
         r"}\label{tab:app-search} \\", r"\toprule",
         r"View & Algorithm & Hyperparameters \\", r"\midrule",
         r"\endfirsthead", r"\toprule",
         r"View & Algorithm & Hyperparameters \\", r"\midrule", r"\endhead",
         r"\bottomrule", r"\endfoot"]

    def fmt(spec):
        out = []
        for k, v in spec.items():
            if v["type"] == "categorical":
                out.append("%s $\\in$ \\{%s\\}" % (
                    esc(k), ", ".join(esc(x) for x in v["choices"])))
            elif v["low"] == v["high"]:
                out.append("%s $=$ %s" % (esc(k), esc(v["low"])))
            else:
                out.append("%s $\\in [%s,%s]$" % (esc(k), v["low"], v["high"]))
        return "; ".join(out)

    for view, spec in sp.items():
        first = True
        for alg, params in spec.items():
            L.append("%s & %s & %s \\\\" % (
                code(view) if first else "", code(alg), fmt(params)))
            first = False
        L.append(r"\addlinespace[2pt]")
    L += [r"\end{longtable}", r"\normalsize"]
    w("tab_app_search", L)


# ================================================================= macros
def macros():
    mm = pd.read_csv(SR / "method_master_table.csv").set_index("short")
    tv = pd.read_csv(SR / "runtime_performance_tradeoff.csv").set_index("short")
    prog = pd.read_csv(EVI / "controlled_progression.csv").set_index("key")
    cf = EV["controlled_families"]
    cp = EV["controlled_progression"]
    s3 = EV["stage3"]
    L = [r"%% auto-generated by scripts/build_v12_tables.py -- do not edit"]

    def m(n, v):
        L.append(r"\newcommand{\%s}{%s}" % (n, v))

    m("CtrlPre", "%.3f" % prog.loc["pre", "mean"])
    m("CtrlFixed", "%.3f" % prog.loc["A", "mean"])
    m("CtrlVone", "%.3f" % prog.loc["B", "mean"])
    m("CtrlVtwo", "%.3f" % prog.loc["C", "mean"])
    m("CtrlVBS", "%.3f" % prog.loc["D", "mean"])
    m("CtrlAC", "%.4f" % cp["autoclust_matched"])
    m("RerankCtrl", "%+.4f" % cp["pre_reranker_gain"])
    m("FamWithin", "%d" % cf["n_within_0.01_v2_vs_autoclust"])
    m("FamAhead", "%d" % cf["n_clustopt_ahead"])
    m("FamBehind", "%d" % cf["n_autoclust_ahead"])
    m("FamMaxAbs", "%.3f" % cf["max_abs_delta"])
    m("FamDeint", "%+.4f" % cf["deinterleaving_delta"])
    for k in ("C5", "C4", "C3", "A0", "M1"):
        m("Ext" + {"C5": "Cfive", "C4": "Cfour", "C3": "Cthree",
                   "A0": "Azero", "M1": "Mone"}[k],
          "%.3f" % mm.loc[k, "bestview_ari_mean"])
        m("Rt" + {"C5": "Cfive", "C4": "Cfour", "C3": "Cthree",
                  "A0": "Azero", "M1": "Mone"}[k],
          "%.1f" % tv.loc[k, "three_view_runtime_median"])
    m("TierAFam", "%d" % s3["family"]["tier_a_pairs"])
    m("TierAFamUniq", "%d" % s3["family"]["tier_a_unique_pairs"])
    m("TierAFamCov", "%d" % s3["family"]["families_with_tier_a_unique"])
    # families spanned by ALL Tier-A family relations (not only the
    # unique-specialist ones), read from the frozen relation table
    m("TierAFamSpan", "%d" % pd.read_csv(
        EVI / "stage3_family_tierA.csv")["group"].nunique())
    m("TierASub", "%d" % s3["subfamily"]["tier_a_pairs"])
    m("TierASubOnly", "%d" % s3["subfamily"]["subfamily_only_tier_a"])
    m("AlignAll", "%.3f" % s3["alignment"]["family_mean_spearman_all"])
    m("AlignNew", "%.3f" % s3["alignment"]["family_mean_spearman_new46"])
    m("NewRecall", "%.2f" % s3["new46_true_top5_recall_mean"])
    m("CoreShare", "%.0f" % (100 * s3["selection_composition"]["core_share"]))
    m("NewShare", "%.0f" % (100 * s3["selection_composition"]["new46_share"]))
    (MAC / "numbers.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("  macros: numbers.tex (%d)" % (len(L) - 1))


def tab_app_policy_family():
    """Family x policy selection frequency for PPv2 on the controlled replay."""
    sel = pd.read_csv(S2 / "split1_v2_policy_selections.csv.gz")
    fam = pd.read_csv(S1 / "offline_fixed32" / "offline_2x3_dataset_results.csv",
                      usecols=["dataset_id", "family"]).drop_duplicates(
                          "dataset_id")
    d = sel.merge(fam, on="dataset_id", how="left")
    assert d.family.notna().all()
    t = (d.groupby(["family", "selected_regime_id"]).size()
         .unstack(fill_value=0))
    t = t.div(t.sum(axis=1), axis=0)
    order = list(t.sum().sort_values(ascending=False).index)
    t = t[order]
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{What \ppv{2} selects, by structural family: the fraction of "
         r"the family's dataset--view records assigned to each policy regime on "
         r"the controlled Split-1 replay ($3{,}165$ records). Regimes name the "
         r"predictor and the subset rule; the weighting is raw throughout "
         r"(Appendix~\ref{app:null}). Rows sum to one.}",
         r"\label{tab:app-policy-family}",
         r"\scriptsize",
         r"\begin{tabular}{@{}p{0.17\linewidth}%s@{}}"
         % ("r" * len(order)), r"\toprule",
         "Family & " + " & ".join(
             esc(c.replace("_", "").replace("TOP", "T")
                 .replace("DYNAMIC", "dyn").lower())
             for c in order) + r" \\", r"\midrule"]
    for f, r in t.iterrows():
        L.append("%s & %s \\\\" % (
            esc(FAM_PLAIN.get(f, f)),
            " & ".join("%.2f" % v for v in r.values)))
    L += [r"\midrule",
          "All & " + " & ".join("%.2f" % v for v in
                                (d.selected_regime_id.value_counts(
                                    normalize=True).reindex(order).fillna(0)
                                 .values)) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    w("tab_app_policy_family", L)


def tab_app_controlled_baseline_arms():
    """The eight CONTROLLED (Stage-3A) baseline inventory arms, marked.

    Authority: HISTORICAL_RESULTS_USAGE_RULES.md; CLAIM_EVIDENCE_MATRIX C27.
    These are shown so a reader can see they exist and why they are not used
    causally. No claim in this paper rests on them.
    """
    ac = pd.read_csv(EXTBASE / "AutoClust_metric_decomposition_split1" /
                     "four_arm_global_summary.csv")
    ml = pd.read_csv(EXTBASE / "ML2DAC_metric_decomposition_split1" /
                     "four_arm_global_summary.csv")
    INV = ["Original", r"$+$Established", r"$+$New46", "Extended"]

    L = [r"\begin{table}[htbp]", r"\centering\footnotesize",
         r"\caption{\textbf{Historical controlled baseline inventory arms} "
         r"(Stage 3A; 1{,}055 synthetic datasets, matched search domain, mean "
         r"Best-View ARI). Shown for completeness because the runs exist and "
         r"are frozen, and read as \emph{legacy descriptive references only}. "
         r"\textbf{They are not clean evidence of an index-inventory effect, "
         r"and no inferential or causal claim in this paper relies on them.} "
         r"$\dagger$ the AutoClust arms are \emph{not paired}: 0 of 324 "
         r"comparison keys had identical ordered candidate slates and 324/324 "
         r"differed, so a difference between arms cannot be attributed to the "
         r"inventory rather than to the different search each arm ran. "
         r"$\ddagger$ the ML2DAC arms carry that defect \emph{and} a scorer "
         r"defect: the predicted index was unevaluable for 54 of 61 "
         r"classifier classes, so it had no influence on selection at all. "
         r"The properly paired replacement is the external benchmark "
         r"(Table~\ref{tab:external}), where the ordering is different: there "
         r"neither baseline's best arm is its extended one.}",
         r"\label{tab:app-controlled-baseline-arms}",
         r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"Arm & Index inventory & Requested & Live & Best-View mean ARI \\",
         r"\midrule"]
    for _, r in ac.iterrows():
        i = int(str(r.arm)[1])
        L.append("A%d $\\dagger$ & %s & %d & %d & %.4f \\\\" % (
            i, INV[i], int(r.requested_metric_count),
            int(r.live_metric_count), float(r.bestview_mean_ari)))
    L.append(r"\addlinespace[2pt]")
    for _, r in ml.iterrows():
        i = int(str(r.arm)[1])
        L.append("M%d $\\ddagger$ & %s & %d & %d & %.4f \\\\" % (
            i, INV[i], int(r.requested_metric_count),
            int(r.live_metric_count), float(r.bestview_mean_ari)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_app_controlled_baseline_arms", L)
    record("tab_app_controlled_baseline_arms",
           ["AutoClust_metric_decomposition_split1/four_arm_global_summary.csv",
            "ML2DAC_metric_decomposition_split1/four_arm_global_summary.csv"],
           "stored per-arm Best-View means and live index counts")


def tab_hist_sweep():
    """Compact historical synthetic inventory sweep (2 x 4)."""
    ac = pd.read_csv(EXTBASE / "AutoClust_metric_decomposition_split1" /
                     "four_arm_global_summary.csv")
    ml = pd.read_csv(EXTBASE / "ML2DAC_metric_decomposition_split1" /
                     "four_arm_global_summary.csv")
    a = [float(v) for v in ac.bestview_mean_ari]
    m = [float(v) for v in ml.bestview_mean_ari]

    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{\textbf{Historical synthetic inventory sweep} (Stage 3A; "
         r"the same 1{,}055 controlled datasets, mean Best-View ARI). Context "
         r"only: $\dagger$ unpaired candidate slates across arms; $\ddagger$ "
         r"legacy scoring semantics later found defective. \emph{No inferential or "
         r"causal claim in this paper relies on these eight values}; provenance "
         r"in Appendix~\ref{app:null}.}",
         r"\label{tab:histsweep}",
         r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"Historical arm & Original & $+$Established & $+$New46 & Extended \\",
         r"\midrule",
         "AutoClust\\,$\\dagger$ & %.4f & %.4f & %.4f & %.4f \\\\" % tuple(a),
         "ML2DAC\\,$\\ddagger$ & %.4f & %.4f & %.4f & %.4f \\\\" % tuple(m),
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_hist_sweep", L)
    record("tab_hist_sweep",
           ["AutoClust_metric_decomposition_split1/four_arm_global_summary.csv",
            "ML2DAC_metric_decomposition_split1/four_arm_global_summary.csv"],
           "stored per-arm Best-View means, no recomputation")


if __name__ == "__main__":
    print("building LaTeX tables ...")
    tab_setting(); tab_controlled(); tab_hist_sweep(); tab_external(); tab_metrics()
    tab_app_stage1(); tab_app_policies(); tab_app_family(); tab_app_repo()
    tab_app_arms();
    tab_app_controlled_baseline_arms(); tab_app_predictor(); tab_app_stage3()
    tab_app_policy_family()
    tab_app_external_arms(); tab_app_corpus(); tab_app_runtime()
    tab_app_metric_ranking(); tab_app_new46(); tab_app_search()
    macros()
    write_prov("asset_provenance_v12_tables.json")
    print("done")
