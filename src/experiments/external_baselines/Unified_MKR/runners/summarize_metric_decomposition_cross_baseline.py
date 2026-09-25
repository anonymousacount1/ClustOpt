"""Stage 3A-2: cross-baseline view of the two four-arm metric decompositions.

READ-ONLY over the two per-baseline packages. This module **stacks** the ML2DAC
and AutoClust results side by side; it never pools or averages them. The two
baselines consume a metric inventory by structurally different mechanisms --
ML2DAC selects a single CVI per dataset, AutoClust regresses on the complete CVI
vector -- so a mean over the two effects would not estimate any quantity that
exists. Every emitted row therefore carries its own ``baseline`` key, and the
only cross-baseline quantity computed is an explicit *difference* between the two
mechanisms on the same contrast, reported as a mechanism contrast and never as a
pooled effect.

The per-dataset factorial terms of the two baselines ARE defined on the same
1,055 Split-1 datasets, so the difference of a contrast is itself paired; it is
summarised with the same bootstrap and signed-rank conventions used inside each
package.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from unified_mkr import decomposition_analysis as DA  # noqa: E402
from unified_mkr import io_utils  # noqa: E402

EXT_REL = "results_analysis/clustering_repository/external_baselines"
PKG = {"ml2dac": "ML2DAC_metric_decomposition_split1",
       "autoclust": "AutoClust_metric_decomposition_split1"}
OUT_REL = EXT_REL + "/metric_decomposition_split1"
BASELINES = ("ml2dac", "autoclust")


def _load(repo: Path, b: str, name: str) -> pd.DataFrame:
    return pd.read_csv(io_utils.ext(repo / EXT_REL / PKG[b] / name))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="cross-baseline decomposition summary")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else io_utils.find_repo_root()
    io_utils.ensure_repo_on_path(repo)
    out = io_utils.ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 3A-2  CROSS-BASELINE METRIC-DECOMPOSITION SUMMARY")
    print("=" * 78, flush=True)

    # ---- 1. stacked arm/factorial summary ---------------------------------
    fac_rows: List[Dict[str, Any]] = []
    for b in BASELINES:
        G = _load(repo, b, "four_arm_global_summary.csv")
        FU = _load(repo, b, "factorial_uncertainty.csv")
        means = {r["arm"]: float(r["bestview_mean_ari"]) for _, r in G.iterrows()}
        live = {r["arm"]: int(r["live_metric_count"]) for _, r in G.iterrows()}
        for _, r in FU.iterrows():
            fac_rows.append({
                "baseline": b,
                "mechanism": DA.BASELINES[b]["mechanism"],
                "quantity": r["quantity"], "mean": float(r["mean"]),
                "median": float(r["median"]),
                "ci_low": float(r["bootstrap_ci_low"]),
                "ci_high": float(r["bootstrap_ci_high"]),
                "wilcoxon_p": float(r["wilcoxon_p"]),
                "n_positive": int(r["n_positive"]),
                "n_negative": int(r["n_negative"]),
                "n_tied": int(r["n_tied"]),
                "ci_excludes_zero": bool(float(r["bootstrap_ci_low"]) > 0
                                         or float(r["bootstrap_ci_high"]) < 0),
                "A0_mean": means.get("A0"), "A1_mean": means.get("A1"),
                "A2_mean": means.get("A2"), "A3_mean": means.get("A3"),
                "A0_live_metrics": live.get("A0"), "A3_live_metrics": live.get("A3"),
                "pooled_across_baselines": False})
    FAC = pd.DataFrame(fac_rows)
    io_utils.write_csv_atomic(out / "baseline_factorial_summary.csv", FAC)
    print("[cross] stacked %d factorial rows over %d baselines"
          % (len(FAC), FAC["baseline"].nunique()), flush=True)

    # ---- 2. stacked pairwise contrasts ------------------------------------
    pw = pd.concat([_load(repo, b, "paired_comparisons.csv").assign(baseline=b)
                    for b in BASELINES], ignore_index=True)
    pw["pooled_across_baselines"] = False
    io_utils.write_csv_atomic(out / "baseline_pairwise_effects.csv", pw)
    print("[cross] stacked %d pairwise rows" % len(pw), flush=True)

    # ---- 3. mechanism contrast: AutoClust term MINUS ML2DAC term -----------
    # Paired per dataset. This is a DIFFERENCE between mechanisms, not a pooled
    # effect: it answers "does the same inventory change act differently on the
    # two consumption mechanisms", which is the question Stage 3A exists to ask.
    pdf = {b: pd.read_csv(io_utils.ext(repo / EXT_REL / PKG[b]
                                       / "per_dataset_factorial_terms.csv.gz"))
           for b in BASELINES}
    a, m = pdf["autoclust"], pdf["ml2dac"]
    j = a.merge(m, on="dataset_id", suffixes=("_ac", "_ml"))
    assert len(j) == len(a) == len(m), "baselines do not share the dataset set"
    terms = [c for c in a.columns
             if c not in ("dataset_id", "family", "subfamily")]
    mrows = []
    for t in terms:
        d = j["%s_ac" % t].to_numpy(float) - j["%s_ml" % t].to_numpy(float)
        r = DA.summarise_paired(d, t)
        r.update({"contrast": "autoclust_minus_ml2dac", "n_datasets": int(d.size),
                  "autoclust_mean": float(a[t].mean()),
                  "ml2dac_mean": float(m[t].mean()),
                  "pooled_across_baselines": False})
        mrows.append(r)
    MC = pd.DataFrame(mrows)
    io_utils.write_csv_atomic(out / "mechanism_contrast.csv", MC)
    for _, r in MC.iterrows():
        print("[cross] %-24s AC %+.6f  ML %+.6f  diff %+.6f CI [%+.6f, %+.6f] p=%.4g"
              % (r["quantity"], r["autoclust_mean"], r["ml2dac_mean"], r["mean"],
                 r["bootstrap_ci_low"], r["bootstrap_ci_high"], r["wilcoxon_p"]),
              flush=True)

    # ---- 4. provenance -----------------------------------------------------
    prov = {
        "stage": "3A-2", "scope": "cross-baseline summary (stacked, never pooled)",
        "created_at": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_commit": commit, "analysis_only": True,
        "source_packages": {b: str(repo / EXT_REL / PKG[b]) for b in BASELINES},
        "pooling_policy": (
            "AutoClust and ML2DAC effects are NEVER averaged or pooled. The two "
            "baselines consume the CVI inventory by structurally different "
            "mechanisms, so a pooled effect would not estimate any well-defined "
            "quantity. Rows are stacked with an explicit baseline key; the only "
            "cross-baseline quantity is a paired DIFFERENCE, labelled a mechanism "
            "contrast."),
        "mechanisms": {b: DA.BASELINES[b]["mechanism"] for b in BASELINES},
        "shared_dataset_set": True, "n_datasets": int(len(j)),
        "statistical_conventions": {
            "bootstrap": "10,000-sample percentile, seed %d" % DA.BOOTSTRAP_SEED,
            "test": "Wilcoxon signed-rank, two-sided",
            "tie_tolerance": DA.TOL,
            "endpoint": "dataset-level BestView ARI over the 3 views"},
        "split_1_status": (
            "common external-baseline evaluation split; Split-1 results have "
            "informed earlier project decisions, so this is a controlled matched "
            "Split-1 replay, NOT a pristine untouched final holdout."),
    }
    io_utils.write_json_atomic(out / "provenance.json", prov)
    _readme(out, FAC, MC, prov)
    print("[cross] written to %s" % out, flush=True)
    return 0


def _readme(out: Path, FAC: pd.DataFrame, MC: pd.DataFrame,
            prov: Dict[str, Any]) -> None:
    def blk(b: str) -> List[str]:
        d = FAC[FAC["baseline"] == b]
        L = ["| quantity | mean | 95 % CI | Wilcoxon *p* | CI excludes 0 |",
             "|---|---:|---|---:|---|"]
        for _, r in d.iterrows():
            L.append("| %s | %+.6f | [%+.6f, %+.6f] | %.3g | %s |"
                     % (r["quantity"], r["mean"], r["ci_low"], r["ci_high"],
                        r["wilcoxon_p"], "yes" if r["ci_excludes_zero"] else "no"))
        return L

    L = ["# Metric-Inventory Decomposition - Cross-Baseline Summary (Split 1)", "",
         "**The two baselines are reported side by side and are never pooled.**",
         "", prov["pooling_policy"], ""]
    for b in ("ml2dac", "autoclust"):
        r0 = FAC[FAC["baseline"] == b].iloc[0]
        L += ["## %s" % b.upper(), "",
              "*Mechanism:* %s." % prov["mechanisms"][b], "",
              "Arm means - A0 %.6f, A1 %.6f, A2 %.6f, A3 %.6f."
              % (r0["A0_mean"], r0["A1_mean"], r0["A2_mean"], r0["A3_mean"]), ""]
        L += blk(b) + [""]
    L += ["## Mechanism contrast (AutoClust minus ML2DAC, paired per dataset)", "",
          "A *difference*, not a pooled effect. It asks whether the same "
          "inventory change acts differently on the two consumption mechanisms.",
          "",
          "| quantity | AutoClust | ML2DAC | difference | 95 % CI | *p* |",
          "|---|---:|---:|---:|---|---:|"]
    for _, r in MC.iterrows():
        L.append("| %s | %+.6f | %+.6f | %+.6f | [%+.6f, %+.6f] | %.3g |"
                 % (r["quantity"], r["autoclust_mean"], r["ml2dac_mean"],
                    r["mean"], r["bootstrap_ci_low"], r["bootstrap_ci_high"],
                    r["wilcoxon_p"]))
    L += ["", "## Files", "",
          "- `baseline_factorial_summary.csv` - stacked factorial terms + arm means",
          "- `baseline_pairwise_effects.csv` - stacked direct arm contrasts",
          "- `mechanism_contrast.csv` - paired AutoClust minus ML2DAC differences",
          "- `provenance.json`", "",
          "*%s*" % prov["split_1_status"], ""]
    io_utils.write_text_atomic(out / "README.md", "\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
