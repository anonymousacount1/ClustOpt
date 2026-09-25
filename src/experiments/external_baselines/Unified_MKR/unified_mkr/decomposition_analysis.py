"""Shared 2x2 metric-inventory decomposition analysis for the external baselines.

ANALYSIS ONLY. Every number is derived from artefacts this module does not own:
the canonical dataset-level frame produced by the standard paper-analysis
pipeline, plus the frozen variant specs and derived validation reports for
provenance. No per-dataset result tree is copied, and no arm is recomputed --
A0 and A3 are read exactly as the historical runs left them.

The design, with ``O`` the baseline's original CVIs, ``E`` the 10 established
additions and ``N`` the 46 newly designed ClustOpt indices::

    A0 = O            A1 = O u E
    A2 = O u N        A3 = O u E u N

Both baselines are analysed by this one module so their statistical conventions
are identical by construction: the same 10,000-sample percentile bootstrap, the
same Wilcoxon signed-rank test, the same tie tolerance and the same dataset-level
BestView endpoint.

Derived factorial quantities are computed **per dataset** and then summarised, so
the interaction and the two main effects carry the same paired uncertainty as any
direct contrast. Reporting a factorial term as a point estimate alone is exactly
the omission this module exists to prevent.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import io_utils

PA_REL = "results_analysis/clustering_repository/paper_analysis_v2"
CANONICAL_FRAME = "13_raw_and_reproducibility/canonical_dataset_level_frame.parquet"
SPEC_REL = "experiments/external_baselines/Unified_MKR/variant_specs"
DERIVED_REL = "experiments/external_baselines/Unified_MKR/derived"

ARM_LABEL: Dict[str, str] = {
    "A0": "Original", "A1": "Original + Established",
    "A2": "Original + New46", "A3": "Extended",
}
#: pre-declared paired comparisons, fixed before any value is read
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("A1", "A0"), ("A2", "A0"), ("A3", "A1"),
    ("A3", "A2"), ("A2", "A1"), ("A3", "A0"),
)
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260905
TOL = 1e-12
GZ = {"index": False, "compression": "gzip"}

#: (semantic arm, METHOD_ID, derived variant dir, requested, live, spec file)
ArmSpec = Tuple[str, str, str, int, int, Optional[str]]

BASELINES: Dict[str, Dict[str, Any]] = {
    "ml2dac": {
        "derived_root": "ml2dac_split1",
        "model_artifacts": ["cvi_classifier.pkl", "imputer.pkl",
                            "nearest_neighbor_index.pkl",
                            "warmstart_repository.parquet", "cvi_star.parquet"],
        "metric_dependent_components": ["CVI*", "CVI rank correlations",
                                        "RandomForest CVI classifier"],
        "mechanism": ("ML2DAC predicts ONE CVI per dataset and optimises that "
                      "single index, so enlarging the inventory enlarges the "
                      "classifier's label space"),
        "search_budget": {"n_warmstarts": 25, "n_optimizer": 25, "total": 50},
        "arms": (
            ("A0", "ML2DAC_Original_InDomain_same_search_space",
             "original_cvis", 7, 7, None),
            ("A1", "ML2DAC_OriginalPlusEstablished_InDomain_same_search_space",
             "original_plus_established", 17, 17, "original_plus_established.json"),
            ("A2", "ML2DAC_OriginalPlusNew46_InDomain_same_search_space",
             "original_plus_new46", 53, 51, "original_plus_new46.json"),
            ("A3", "ML2DAC_Extended_InDomain_same_search_space",
             "extended_cvis", 63, 61, None),
        ),
    },
    "autoclust": {
        "derived_root": "autoclust_split1",
        "model_artifacts": ["ari_predictor.pkl", "algorithm_selector.pkl",
                            "ari_regression_dataset.parquet",
                            "algorithm_labels.parquet"],
        "metric_dependent_components": ["ARI-predictor MLP (60,30,10)"],
        "mechanism": ("AutoClust feeds the COMPLETE CVI vector to a learned "
                      "ARI-predictor MLP on every trial, so enlarging the "
                      "inventory widens the regression input rather than a "
                      "label space"),
        "search_budget": {"n_evaluations": 50, "total": 50},
        "arms": (
            ("A0", "AutoClust_Original_InDomain_same_search_space",
             "original_cvis", 7, 7, None),
            ("A1", "AutoClust_OriginalPlusEstablished_InDomain_same_search_space",
             "original_plus_established", 17, 17, "original_plus_established.json"),
            ("A2", "AutoClust_OriginalPlusNew46_InDomain_same_search_space",
             "original_plus_new46", 53, 51, "original_plus_new46.json"),
            ("A3", "AutoClust_Extended_InDomain_same_search_space",
             "extended_cvis", 63, 61, None),
        ),
    },
}


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------- frame
def load_arm_frame(repo: Path, arms: Sequence[ArmSpec]) -> pd.DataFrame:
    """Dataset-level BestView rows for the four arms, from the standard frame."""
    f = repo / PA_REL / CANONICAL_FRAME
    df = pd.read_parquet(io_utils.ext(f))
    want = {m: a for a, m, _, _, _, _ in arms}
    sub = df[df["method_name"].isin(want)].copy()
    sub["arm"] = sub["method_name"].map(want)
    keep = [c for c in ("arm", "method_name", "dataset_id", "family", "family_id",
                        "subfamily", "difficulty", "best_view_ari", "best_view_id",
                        "status", "best_view_runtime_sec",
                        "total_runtime_all_views_sec", "selected_k", "true_k")
            if c in sub.columns]
    return sub[keep].sort_values(["arm", "dataset_id"]).reset_index(drop=True)


def pivot_bestview(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset, one column per arm. Inner join == paired by design."""
    p = frame.pivot_table(index="dataset_id", columns="arm",
                          values="best_view_ari", aggfunc="first")
    meta = (frame[["dataset_id", "family", "subfamily"]]
            .drop_duplicates("dataset_id").set_index("dataset_id"))
    return meta.join(p, how="inner").reset_index()


# ---------------------------------------------------------------- statistics
def bootstrap_ci(d: np.ndarray, *, n: int = BOOTSTRAP_N,
                 seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    if d.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _wilcoxon(d: np.ndarray):
    """Signed-rank test on the paired differences; ``None`` when all are zero."""
    from scipy import stats as sps
    if not np.any(d != 0):
        return None
    try:
        return sps.wilcoxon(d)
    except Exception:                                              # noqa: BLE001
        return None


def summarise_paired(d: np.ndarray, label: str, **extra: Any) -> Dict[str, Any]:
    """One row of paired-difference statistics, project-standard conventions."""
    lo, hi = bootstrap_ci(d)
    w = _wilcoxon(d)
    row: Dict[str, Any] = {
        "quantity": label,
        "n": int(d.size),
        "mean": float(d.mean()) if d.size else float("nan"),
        "median": float(np.median(d)) if d.size else float("nan"),
        "std": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
        "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "wilcoxon_stat": float(w.statistic) if w is not None else float("nan"),
        "wilcoxon_p": float(w.pvalue) if w is not None else float("nan"),
        "n_positive": int((d > TOL).sum()),
        "n_negative": int((d < -TOL).sum()),
        "n_tied": int((np.abs(d) <= TOL).sum()),
        "cohens_dz": (float(d.mean() / d.std(ddof=1))
                      if d.size > 1 and d.std(ddof=1) > 0 else float("nan")),
        "bootstrap_samples": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
    }
    row.update(extra)
    return row


def paired_rows(piv: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a, b in PAIRS:
        x, y = piv[a].to_numpy(float), piv[b].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        r = summarise_paired(
            x[m] - y[m], "%s_vs_%s" % (a, b),
            comparison="%s_vs_%s" % (a, b), arm_a=a, arm_b=b,
            label_a=ARM_LABEL[a], label_b=ARM_LABEL[b],
            n_paired=int(m.sum()),
            mean_a=float(x[m].mean()), mean_b=float(y[m].mean()))
        r["mean_paired_delta"] = r["mean"]
        r["median_paired_delta"] = r["median"]
        r["wins"], r["losses"], r["ties"] = r["n_positive"], r["n_negative"], r["n_tied"]
        out.append(r)
    return out


def per_dataset_factorials(piv: pd.DataFrame) -> pd.DataFrame:
    """The three derived factorial quantities, evaluated on every dataset."""
    A0, A1 = piv["A0"].to_numpy(float), piv["A1"].to_numpy(float)
    A2, A3 = piv["A2"].to_numpy(float), piv["A3"].to_numpy(float)
    return pd.DataFrame({
        "dataset_id": piv["dataset_id"], "family": piv["family"],
        "subfamily": piv["subfamily"],
        "interaction": A3 - A2 - A1 + A0,
        "established_main": ((A1 - A0) + (A3 - A2)) / 2.0,
        "new46_main": ((A2 - A0) + (A3 - A1)) / 2.0,
        "established_on_original": A1 - A0,
        "new46_on_original": A2 - A0,
        "new46_given_established": A3 - A1,
        "established_given_new46": A3 - A2,
        "total_shift": A3 - A0,
    })


def factorial_uncertainty(pdf: pd.DataFrame) -> List[Dict[str, Any]]:
    """Paired uncertainty for every derived factorial quantity."""
    cols = ["interaction", "established_main", "new46_main",
            "established_on_original", "new46_on_original",
            "new46_given_established", "established_given_new46", "total_shift"]
    return [summarise_paired(pdf[c].to_numpy(float), c) for c in cols]


def factorial_point(means: Dict[str, float]) -> Dict[str, Any]:
    A0, A1, A2, A3 = means["A0"], means["A1"], means["A2"], means["A3"]
    return {
        "definition": ("dataset-level BestView means; O=original, E=established "
                       "additions (10), N=new46 (46 requested / 44 evaluable)"),
        "arm_means": {"A0": A0, "A1": A1, "A2": A2, "A3": A3},
        "Established_on_Original": A1 - A0,
        "New46_on_Original": A2 - A0,
        "New46_given_Established": A3 - A1,
        "Established_given_New46": A3 - A2,
        "Interaction": A3 - A2 - A1 + A0,
        "MainEffect_Established": ((A1 - A0) + (A3 - A2)) / 2.0,
        "MainEffect_New46": ((A2 - A0) + (A3 - A1)) / 2.0,
        "TotalShift_A3_minus_A0": A3 - A0,
        "identity_check_total_equals_sum_of_main_effects": float(
            (A3 - A0) - (((A1 - A0) + (A3 - A2)) / 2.0
                         + ((A2 - A0) + (A3 - A1)) / 2.0)),
        "uncertainty_reported_in": "factorial_uncertainty.csv",
        "causal_scope": ("controlled metric-inventory ablation inside one frozen "
                         "baseline's InDomain matched-search-space protocol; the "
                         "result is specific to how THAT baseline consumes CVIs"),
    }


def group_summary(piv: pd.DataFrame, by: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, g in piv.groupby(by):
        r: Dict[str, Any] = {by: key, "n_datasets": int(len(g))}
        for a in ("A0", "A1", "A2", "A3"):
            r["mean_%s" % a] = float(g[a].mean())
            r["median_%s" % a] = float(g[a].median())
        r["A1_minus_A0"] = r["mean_A1"] - r["mean_A0"]
        r["A2_minus_A0"] = r["mean_A2"] - r["mean_A0"]
        r["A3_minus_A1"] = r["mean_A3"] - r["mean_A1"]
        r["A3_minus_A2"] = r["mean_A3"] - r["mean_A2"]
        r["A3_minus_A0"] = r["mean_A3"] - r["mean_A0"]
        r["interaction"] = r["mean_A3"] - r["mean_A2"] - r["mean_A1"] + r["mean_A0"]
        r["established_main"] = (r["A1_minus_A0"] + r["A3_minus_A2"]) / 2.0
        r["new46_main"] = (r["A2_minus_A0"] + r["A3_minus_A1"]) / 2.0
        rows.append(r)
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


# ------------------------------------------------------------------- driver
def run(repo: Path, baseline: str, out: Path, commit: str,
        log: Callable[[str], None]) -> int:
    cfg = BASELINES[baseline]
    arms: Sequence[ArmSpec] = cfg["arms"]
    frame = load_arm_frame(repo, arms)
    log("loaded %d arm/dataset rows from the standard canonical frame" % len(frame))

    # ---- coverage audit (blocking) ----------------------------------------
    cov_rows, sets = [], {}
    for arm, mid, vdir, req, live, spec in arms:
        g = frame[frame["arm"] == arm]
        ok = g[np.isfinite(pd.to_numeric(g["best_view_ari"], errors="coerce"))]
        sets[arm] = set(ok["dataset_id"])
        cov_rows.append({
            "baseline": baseline, "arm": arm, "semantic": ARM_LABEL[arm],
            "method_id": mid, "n_rows": int(len(g)),
            "n_finite_bestview": int(len(ok)),
            "n_datasets": int(g["dataset_id"].nunique()),
            "requested_metrics": req, "live_metrics": live,
            "derived_variant": vdir})
    COV = pd.DataFrame(cov_rows)
    io_utils.write_csv_atomic(out / "coverage_audit.csv", COV)
    base = sets.get("A0", set())
    mism = {a: len(sets[a] ^ base) for a in sets}
    log("coverage %s | symmetric differences vs A0: %s"
        % (COV["n_finite_bestview"].tolist(), mism))
    if any(v for v in mism.values()) or len(base) != 1055:
        io_utils.write_json_atomic(out / "coverage_blocker.json", {
            "n_by_arm": {a: len(sets[a]) for a in sets},
            "sym_diff_vs_A0": mism, "expected": 1055})
        log("BLOCKER: arm coverage is not identical across all four arms")
        return 2

    piv = pivot_bestview(frame).dropna(subset=["A0", "A1", "A2", "A3"]).reset_index(drop=True)
    log("paired dataset-level table: %d datasets x 4 arms" % len(piv))
    piv.to_csv(io_utils.ext(out / "four_arm_dataset_results.csv.gz"), **GZ)

    means = {a: float(piv[a].mean()) for a in ("A0", "A1", "A2", "A3")}

    # ---- global summary ----------------------------------------------------
    grows = []
    for arm, mid, vdir, req, live, spec in arms:
        v = piv[arm].to_numpy(float)
        rtv = pd.to_numeric(frame[frame["arm"] == arm]
                            .get("total_runtime_all_views_sec"),
                            errors="coerce").dropna()
        grows.append({
            "baseline": baseline, "arm": arm, "semantic_variant": ARM_LABEL[arm],
            "method_id": mid, "requested_metric_count": req,
            "live_metric_count": live, "n_datasets": int(v.size),
            "bestview_mean_ari": float(v.mean()),
            "bestview_median_ari": float(np.median(v)),
            "bestview_std_ari": float(v.std(ddof=1)),
            "bestview_q25_ari": float(np.quantile(v, 0.25)),
            "bestview_q75_ari": float(np.quantile(v, 0.75)),
            "bestview_min_ari": float(v.min()), "bestview_max_ari": float(v.max()),
            "total_runtime_all_views_mean_sec": (float(rtv.mean())
                                                 if len(rtv) else float("nan"))})
    G = pd.DataFrame(grows)
    io_utils.write_csv_atomic(out / "four_arm_global_summary.csv", G)
    log("BestView means: " + ", ".join("%s=%.7f" % (r["arm"], r["bestview_mean_ari"])
                                       for _, r in G.iterrows()))

    # ---- factorial point estimates + per-dataset uncertainty --------------
    fx = factorial_point(means)
    io_utils.write_json_atomic(out / "factorial_effects.json", fx)
    PDF = per_dataset_factorials(piv)
    PDF.to_csv(io_utils.ext(out / "per_dataset_factorial_terms.csv.gz"), **GZ)
    FU = pd.DataFrame(factorial_uncertainty(PDF))
    FU.insert(0, "baseline", baseline)
    io_utils.write_csv_atomic(out / "factorial_uncertainty.csv", FU)
    for _, r in FU[FU["quantity"].isin(
            ["interaction", "established_main", "new46_main"])].iterrows():
        log("%-18s %+.7f  CI [%+.7f, %+.7f]  p=%.4g  +/-/= %d/%d/%d"
            % (r["quantity"], r["mean"], r["bootstrap_ci_low"],
               r["bootstrap_ci_high"], r["wilcoxon_p"],
               r["n_positive"], r["n_negative"], r["n_tied"]))

    P = pd.DataFrame(paired_rows(piv))
    P.insert(0, "baseline", baseline)
    io_utils.write_csv_atomic(out / "paired_comparisons.csv", P)

    io_utils.write_csv_atomic(out / "four_arm_family_summary.csv",
                              group_summary(piv, "family"))
    io_utils.write_csv_atomic(out / "four_arm_subfamily_summary.csv",
                              group_summary(piv, "subfamily"))

    # ---- runtime (within-baseline only) -----------------------------------
    rrows = []
    for arm, mid, vdir, req, live, spec in arms:
        g = frame[frame["arm"] == arm]
        for col, tag in (("total_runtime_all_views_sec", "total_all_views"),
                         ("best_view_runtime_sec", "best_view_only")):
            s = pd.to_numeric(g.get(col), errors="coerce").dropna()
            if not len(s):
                continue
            rrows.append({
                "baseline": baseline, "arm": arm,
                "semantic_variant": ARM_LABEL[arm], "method_id": mid,
                "live_metric_count": live, "runtime_field": tag, "n": int(len(s)),
                "mean_sec": float(s.mean()), "median_sec": float(s.median()),
                "p25_sec": float(s.quantile(0.25)), "p75_sec": float(s.quantile(0.75)),
                "p90_sec": float(s.quantile(0.90)), "p95_sec": float(s.quantile(0.95)),
                "total_cpu_hours": float(s.sum()) / 3600.0})
    io_utils.write_csv_atomic(out / "runtime_comparison.csv", pd.DataFrame(rrows))

    # ---- provenance --------------------------------------------------------
    prov: Dict[str, Any] = {
        "baseline": baseline, "stage": "3A-1/3A-2", "created_at": now(),
        "source_commit": commit, "analysis_only": True,
        "mechanism": cfg["mechanism"],
        "metric_dependent_components": cfg["metric_dependent_components"],
        "authoritative_sources": {
            "dataset_level_frame": str(repo / PA_REL / CANONICAL_FRAME),
            "per_dataset_pattern": ("results_analysis/clustering_repository/"
                                    "analyzed_data/<family>/subfamilies/"
                                    "<subfamily>/<dataset_id>/experiments/"
                                    "split_01/<METHOD_ID>/{1d_x,1d_y,2d}/result.json"),
            "standard_aggregate_root": str(repo / PA_REL)},
        "split_1_status": ("common external-baseline evaluation split for all four "
                           "arms; Split-1 results have informed earlier project "
                           "decisions, so this is a controlled matched Split-1 "
                           "replay, NOT a pristine untouched final holdout. All "
                           "metric partitions were frozen independently of the "
                           "A1/A2 outcomes."),
        "arms": [],
    }
    for arm, mid, vdir, req, live, spec in arms:
        d = repo / DERIVED_REL / cfg["derived_root"] / vdir
        vr = io_utils.read_json(d / "validation_report.json") or {}
        sp = io_utils.read_json(repo / SPEC_REL / spec) if spec else None
        prov["arms"].append({
            "arm": arm, "semantic_variant": ARM_LABEL[arm], "method_id": mid,
            "requested_metric_count": req, "live_metric_count": live,
            "metric_manifest_hash": (sp or {}).get("manifest_hash"),
            "metric_manifest_path": (str(repo / SPEC_REL / spec) if spec
                                     else "historical build (no Stage-3A1 spec)"),
            "derived_repository": str(d),
            "model_artifacts": cfg["model_artifacts"],
            "train_splits": (vr.get("counts") or {}).get("train_splits"),
            "heldout_split": (vr.get("counts") or {}).get("heldout_split"),
            "random_seed": 1234,
            "leakage_free": (vr.get("leakage") or {}).get("no_leakage"),
            "all_checks_pass": vr.get("all_checks_pass"),
            "search_space_identifier": "clustopt_phaseE2_map",
            "search_budget": cfg["search_budget"],
            "workers_used_for_evaluation": 8,
            "evaluated_datasets": int(COV[COV["arm"] == arm]["n_datasets"].iloc[0]),
            "success_count": int(COV[COV["arm"] == arm]["n_finite_bestview"].iloc[0]),
            "failure_count": int(COV[COV["arm"] == arm]["n_datasets"].iloc[0]
                                 - COV[COV["arm"] == arm]["n_finite_bestview"].iloc[0])})
    prov["identical_dataset_set"] = True
    prov["n_datasets"] = int(len(piv))
    io_utils.write_json_atomic(out / "provenance.json", prov)

    man = {
        "baseline": baseline, "stage": "3A-1/3A-2",
        "experiment": "%s metric-inventory decomposition" % baseline,
        "design": "2x2 factorial over {Established additions} x {New46}",
        "arms": {a: {"method_id": m, "requested": rq, "live": lv,
                     "semantic": ARM_LABEL[a]} for a, m, _, rq, lv, _ in arms},
        "primary_metric": "dataset-level BestView ARI over the 3 views",
        "n_datasets": int(len(piv)), "global_means": means,
        "factorial_effects": fx,
        "factorial_uncertainty": FU.to_dict("records"),
        "paired_comparisons": P.drop(columns=["baseline"]).to_dict("records"),
        "runtime_note": ("within-%s only: all four arms share the identical "
                         "timing boundary. No cross-framework (ClustOpt) runtime "
                         "claim is made here." % baseline),
        "dead_metric_statement": ("46 newly designed metrics, of which 44 are "
                                  "evaluable on this training repository and 2 "
                                  "are deterministically excluded as all-NaN"),
        "outcome_driven_changes": "none",
        "source_commit": commit, "created_at": now(),
    }
    io_utils.write_json_atomic(out / "experiment_manifest.json", man)
    write_readme(out, baseline, G, fx, FU, P, man, cfg)
    log("analysis package written to %s" % out)
    return 0


def write_readme(out: Path, baseline: str, G: pd.DataFrame, fx: Dict[str, Any],
                 FU: pd.DataFrame, P: pd.DataFrame, man: Dict[str, Any],
                 cfg: Dict[str, Any]) -> None:
    def row(a: str) -> pd.Series:
        return G[G["arm"] == a].iloc[0]
    L = [
        "# %s Metric-Inventory Decomposition (Split 1)" % baseline.upper(), "",
        "**Analysis only.** Every value is derived from the authoritative "
        "per-dataset Split-1 `result.json` files via the standard "
        "`paper_analysis_v2` pipeline. No raw result tree is duplicated here, and "
        "A0/A3 are the historical runs, not recomputations.", "",
        "Mechanism: %s." % cfg["mechanism"], "",
        "## Arms", "",
        "| arm | inventory | METHOD_ID | requested | live | BestView mean ARI |",
        "|---|---|---|---:|---:|---:|",
    ]
    for a in ("A0", "A1", "A2", "A3"):
        r = row(a)
        L.append("| %s | %s | `%s` | %d | %d | **%.7f** |"
                 % (a, r["semantic_variant"], r["method_id"],
                    r["requested_metric_count"], r["live_metric_count"],
                    r["bestview_mean_ari"]))
    L += ["", "n = %d datasets, identical across all four arms." % man["n_datasets"],
          "", "## Factorial effects with paired uncertainty", "",
          "| quantity | mean | 95 % CI | Wilcoxon p | +/-/tie |",
          "|---|---:|---|---:|---|"]
    for _, r in FU.iterrows():
        L.append("| %s | %+.7f | [%+.7f, %+.7f] | %.4g | %d/%d/%d |"
                 % (r["quantity"], r["mean"], r["bootstrap_ci_low"],
                    r["bootstrap_ci_high"], r["wilcoxon_p"],
                    r["n_positive"], r["n_negative"], r["n_tied"]))
    L += ["", "## Paired comparisons", "",
          "| comparison | mean delta | 95 % CI | Wilcoxon p | W/L/T |",
          "|---|---:|---|---:|---|"]
    for _, r in P.iterrows():
        L.append("| %s | %+.7f | [%+.7f, %+.7f] | %.4g | %d/%d/%d |"
                 % (r["comparison"], r["mean_paired_delta"],
                    r["bootstrap_ci_low"], r["bootstrap_ci_high"],
                    r["wilcoxon_p"], r["wins"], r["losses"], r["ties"]))
    L += ["", "## Dead metrics", "", man["dead_metric_statement"] + ".", "",
          "## Runtime scope", "", man["runtime_note"], "",
          "## Split-1 status", "",
          "Common external-baseline evaluation split for all four arms. Split-1 "
          "results have informed earlier project decisions, so this is a "
          "**controlled matched Split-1 replay**, not a pristine untouched final "
          "holdout. All metric partitions were frozen independently of the A1/A2 "
          "outcomes.", ""]
    io_utils.write_text_atomic(out / "README.md", "\n".join(L))
