"""Stage 2C-0: inventory of reusable single-exclusion OOF reranker units.

A Stage-2C target unit is one (regime, held-out split) pair whose dedicated C2
reranker was trained WITHOUT that split. Several earlier stages already produced
exactly such units; this module finds them, verifies compatibility field by field
against the frozen Candidate-Reranker contract, and reports what is missing.

Compatibility is never inferred from a directory name. A unit is reusable only if
its own ``unit_manifest.json`` matches on every one of: dedicated single-regime
training, C2_LOCAL_UTIL context, 347 features, HGBR config hash, T2 slate-regret
target, eligibility-definition hash, completion status, leakage checks, and the
absence of Split 1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E501
    _ext, read_json,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR
from models.ClustOpt_Candidate_Reranker.training.final_models import (
    policy_mapping as PM,
)
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (
    unified_feature_builder as UFB,
)

RERANKER_IDS: Tuple[str, ...] = PM.RERANKER_IDS
OUTER_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
STAGE2C_REL = ("results_analysis/clustopt_policy_predictor/"
               "stage2c_reranker_aware_targets")

# Where a compatible dedicated-C2 unit may already live, in priority order.
# ``regimes=None`` means the source may supply any regime.
SOURCES: Tuple[Dict[str, Any], ...] = (
    {"stage": "stage2b4",
     "rel": ("results_analysis/clustopt_candidate_reranker/"
             "stage2b4_same_context_controls/outer_fold_predictions"),
     "dir_of": lambda cid: cid, "regimes": None},
    {"stage": "stage2b2",
     "rel": ("results_analysis/clustopt_candidate_reranker/"
             "stage2b2_context_ablation/outer_fold_predictions"),
     "dir_of": lambda cid: "C2_LOCAL_UTIL", "regimes": ("MLP_TOP10",)},
    {"stage": "stage2c",
     "rel": STAGE2C_REL + "/outer_fold_predictions",
     "dir_of": lambda cid: cid, "regimes": None},
)
# Known-incompatible source, recorded explicitly rather than silently ignored.
INCOMPATIBLE = {
    "stage2b1": ("results_analysis/clustopt_candidate_reranker/"
                 "stage2b1_masking_feasibility/outer_fold_predictions"),
}


def required_contract() -> Dict[str, Any]:
    return {"context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
            "model_config_hash": MR.config_hash(),
            "target": "T2_SLATE_REGRET",
            "eligibility_hash": CE.definition_hash(),
            "utility_source_train": "stage2a3b0 pair-exclusion (S \\ {s,t})",
            "utility_source_test": "stage2a1 single-exclusion (S \\ {s})"}


def manifest_regime(m: Dict[str, Any]) -> Optional[str]:
    """Regime the unit was TRAINED for, across both manifest dialects.

    Stage-2B4 records ``utility_source``/``k_mode``; Stage-2B2 records a single
    ``regime`` string such as ``mlp__top10``. Neither the directory name nor the
    ``condition_id`` is trusted.
    """
    if m.get("utility_source") and m.get("k_mode"):
        return PM.reranker_id(str(m["utility_source"]), str(m["k_mode"]))
    reg = m.get("regime")
    if isinstance(reg, str) and "__" in reg:
        src, km = reg.split("__", 1)
        return PM.reranker_id(src, km)
    return None


def _context_of(m: Dict[str, Any]) -> Optional[str]:
    return m.get("context") or m.get("condition_id")


def check_unit(manifest: Optional[Dict[str, Any]], cid: str, split: int
               ) -> Tuple[bool, str]:
    """(compatible, reason). Every mismatch is named, not summarised away."""
    if not manifest:
        return False, "absent"
    m = dict(manifest)
    m["context"] = _context_of(m)
    bad = [k for k, v in required_contract().items() if m.get(k) != v]
    if m.get("status") != "complete":
        bad.append("status")
    if not m.get("leakage_checks_pass"):
        bad.append("leakage_checks_pass")
    if m.get("split_1_used"):
        bad.append("split_1_used")
    if int(m.get("outer_split", -1)) != int(split):
        bad.append("outer_split")
    got = manifest_regime(m)
    if got != cid:
        bad.append("regime_identity(%s)" % got)
    # "dedicated" means one regime per model. Stage-2B4 says so in ``training``;
    # Stage-2B2's context arms are single-regime by construction (one ``regime``
    # field, no regime one-hot). A UNIFIED multi-regime unit has neither.
    if got is None:
        bad.append("not_single_regime")
    return (not bad), ("compatible" if not bad else "mismatch:" + ",".join(bad))


def locate(repo: Path, cid: str, split: int) -> Dict[str, Any]:
    """Resolve one (regime, split) unit against every candidate source."""
    attempts: List[str] = []
    for src in SOURCES:
        if src["regimes"] is not None and cid not in src["regimes"]:
            continue
        d = (repo / src["rel"] / src["dir_of"](cid) / ("fold_%02d" % split))
        m = read_json(d / "unit_manifest.json")
        ok, why = check_unit(m, cid, split)
        attempts.append("%s:%s" % (src["stage"], why))
        if ok and Path(_ext(d / "slate_selections.csv.gz")).exists():
            return {"regime_id": cid, "split_id": split, "status": "EXISTS",
                    "source_stage": src["stage"],
                    "unit_dir": str(d), "unit_hash": m.get("unit_hash"),
                    "n_train_rows": m.get("n_train_rows"),
                    "n_train_slates": m.get("n_train_slates"),
                    "fit_sec": (m.get("fit_info") or {}).get("fit_sec"),
                    "utility_source_train": m.get("utility_source_train"),
                    "utility_source_test": m.get("utility_source_test"),
                    "bestview": (m.get("metrics") or {}).get(
                        "mean_bestview_selected_ari"),
                    "provenance": "; ".join(attempts)}
    inc = []
    for stage, rel in INCOMPATIBLE.items():
        d = repo / rel / cid / ("fold_%02d" % split)
        m = read_json(d / "unit_manifest.json")
        if m:
            inc.append("%s(n_features=%s,context=%s)"
                       % (stage, m.get("n_features"), m.get("context")))
    return {"regime_id": cid, "split_id": split,
            "status": "INCOMPATIBLE" if inc else "MISSING",
            "source_stage": None, "unit_dir": None, "unit_hash": None,
            "n_train_rows": None, "n_train_slates": None, "fit_sec": None,
            "utility_source_train": None, "utility_source_test": None,
            "bestview": None,
            "provenance": "; ".join(attempts + inc) or "no source"}


def build(repo: Path) -> pd.DataFrame:
    rows = [locate(repo, cid, s) for cid in RERANKER_IDS for s in OUTER_SPLITS]
    return pd.DataFrame(rows)


def summarise(inv: pd.DataFrame) -> Dict[str, Any]:
    need = [r for r in RERANKER_IDS
            if not inv[(inv["regime_id"] == r)
                       & (inv["status"] == "EXISTS")].shape[0] == len(OUTER_SPLITS)]
    fits = inv["fit_sec"].dropna()
    return {
        "n_units_required": int(len(RERANKER_IDS) * len(OUTER_SPLITS)),
        "n_exists": int((inv["status"] == "EXISTS").sum()),
        "n_missing": int((inv["status"] != "EXISTS").sum()),
        "reuse_by_stage": inv[inv["status"] == "EXISTS"]["source_stage"]
        .value_counts().to_dict(),
        "regimes_fully_covered": [r for r in RERANKER_IDS if r not in need],
        "regimes_needing_fits": need,
        "measured_fit_sec_mean": float(fits.mean()) if len(fits) else None,
        "measured_fit_sec_median": float(fits.median()) if len(fits) else None,
        "measured_fit_sec_max": float(fits.max()) if len(fits) else None,
        "contract": required_contract(),
        "policy_semantics_hash": PM.policy_semantics_hash(PM.build_policy_set()),
    }


__all__ = ["RERANKER_IDS", "OUTER_SPLITS", "STAGE2C_REL", "build",
           "summarise", "locate", "check_unit", "required_contract"]
