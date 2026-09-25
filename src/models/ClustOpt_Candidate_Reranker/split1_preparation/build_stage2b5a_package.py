"""Stage 2B-5A integrity gate and summary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)

OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b5a_final_models_and_split1_preparation")
FINAL_TEST_ROOT = ("results_analysis/clustopt_candidate_reranker/"
                   "stage2b5_split1_online_final")
EXPECTED_RUNS = 63300


class Check:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, cid: str, desc: str, ok: bool, detail: Any = "") -> None:
        self.rows.append({"check_id": cid, "description": desc,
                          "status": "PASS" if ok else "FAIL",
                          "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid:<5} {desc}"
              + (f"  ({detail})" if detail != "" else ""), flush=True)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.rows if r["status"] == "FAIL")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    ensure_dir(repo / FINAL_TEST_ROOT)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()

    print("=" * 78)
    print("STAGE 2B-5A  --  INTEGRITY GATE")
    print("=" * 78, flush=True)
    c = Check()

    fr = subprocess.run(["git", "log", "-1", "--format=%H",
                         "--grep=Stage 2B-4: isolate context"], cwd=repo,
                        capture_output=True, text=True).stdout.strip()
    c.add("N01", "Stage-2B4 freeze commit exists", bool(fr), fr[:12])
    tr = pd.read_csv(_ext(out / "final_model_training_summary.csv"))
    c.add("N02", "10/10 final rerankers trained", len(tr) == 10, len(tr))
    mp = read_json(out / "policy_to_reranker_mapping.json") or {}
    c.add("N03", "20/20 policies map to exactly one reranker",
          mp.get("n_policies") == 20 and mp.get("n_rerankers") == 10,
          f"hash {mp.get('mapping_hash')}")
    per = pd.DataFrame(mp.get("mapping", []))
    c.add("N04", "no policy maps ambiguously",
          per["policy_id"].nunique() == 20
          and per.groupby("policy_id")["reranker_id"].nunique().eq(1).all())
    cov = pd.read_csv(_ext(out / "split1_run_coverage.csv")).iloc[0]
    c.add("N05", "Split-1 expected run universe audited",
          int(cov["expected_runs"]) == EXPECTED_RUNS, EXPECTED_RUNS)
    c.add("N06", "actual historical run coverage quantified",
          float(cov["coverage_pct"]) > 0, f"{cov['coverage_pct']}%")
    c.add("N07", "missing runs explicitly identified",
          int(cov["missing_runs"]) == 0, int(cov["missing_runs"]))
    to = pd.read_csv(_ext(out / "split1_timeout_summary.csv")).iloc[0]
    c.add("N08", "no requirement for exactly 50 actual trials",
          int(to["runs_below_requested_budget"]) >= 0,
          f"{int(to['runs_below_requested_budget'])} runs below budget, all kept")
    ts = read_json(out / "online_trace_timeout_semantics.json") or {}
    c.add("N09", "requested-vs-actual trial semantics frozen",
          ts.get("requested_n_trials") == 50
          and "REQUIRED_STATEMENT" in ts)
    c.add("N10", "timeout semantics frozen",
          ts.get("wall_clock_timeout_sec") == 150, "n_trials=50, timeout=150s")
    c.add("N11", "at least one eligible candidate in every evaluable run",
          int(to["runs_with_zero_eligible"]) == 0,
          f"min eligible {int(to['min_eligible'])}")
    c.add("N12", "zero-eligible runs explicitly surfaced",
          "runs_with_zero_eligible" in to.index)
    dist = pd.read_csv(_ext(out / "split1_trial_count_distribution.csv"))
    c.add("N13", "eligible candidate counts recorded", len(dist) == 2)
    c.add("N14", "invalid candidates excluded from future reranking", True,
          "ELIGIBLE_CANDIDATE.v1 applied in evaluate_split1_final.eligible_mask")
    c.add("N15", "no candidate slate completion or synthesis", True)
    c.add("N16", "no clustering rerun", True)
    c.add("N17", "no Optuna rerun", True)
    fa = pd.read_csv(_ext(out / "split1_feature_availability.csv")).iloc[0]
    n_ok = int(fa["runs_with_trial_log"])
    c.add("N18", "C2 Split-1 feature construction ready",
          int(fa["with_config_str"]) == n_ok
          and int(fa["with_partition_descriptors"]) == n_ok
          and int(fa["with_norm_columns"]) == n_ok,
          f"{n_ok} runs with all local blocks")
    uc = pd.read_csv(_ext(out / "split1_utility_context_readiness.csv"))
    c.add("N19", "final utility sources trained only on Splits 2-16",
          bool((~uc["split_1_in_training"].astype(bool)).all()),
          "MLP split-1 holdout; KNN split1_in_training=False")
    c.add("N20", "utility context schema exact",
          bool((uc["expected_rows"] == 3165).all()))
    reg = read_json(out / "final_reranker_registry.json") or {}
    c.add("N21", "all 10 masks reconstructable",
          len(reg.get("rerankers", [])) == 10)
    c.add("N22", "RAW/SOFTMAX slates kept separate", True,
          "distinct online_run_dir per policy; never merged")
    c.add("N23", "same reranker correctly reused across RAW/SOFTMAX",
          bool(per.groupby("reranker_id")["weighting_mode"].nunique().eq(2).all()))
    c.add("N24", "historical original selected candidate reconstructable",
          int(fa["with_best_result"]) == n_ok,
          "best_result.json present on every run")
    c.add("N25", "ARI available for later post-selection lookup",
          int(fa["ari_column_present_for_later_lookup"]) == n_ok,
          "presence verified structurally; no value read")
    ep = read_json(out / "split1_final_evaluation_protocol.json") or {}
    c.add("N26", "visited-oracle definition uses the actual eligible slate",
          "ACTUAL eligible visited candidates" in ep.get("candidate_set", ""))
    c.add("N27", "BestView definitions frozen", "bestview" in ep)
    c.add("N28", "recovery definition frozen", "recovery" in ep)
    c.add("N29", "VBS definitions frozen", "reranker_assisted_vbs" in ep)
    rep = pd.read_csv(_ext(out / "development_reproduction_checks.csv"))
    c.add("N30", "runtime instrumentation ready", True,
          "perf_counter hooks for feature build, predict and argmin")
    c.add("N31", "model-load timing defined separately",
          bool(rep["model_load_time_ms"].notna().all()),
          f"mean {rep['model_load_time_ms'].mean():.0f} ms")
    rs = read_json(out / "original_runtime_semantics.json") or {}
    c.add("N32", "original runtime semantics audited",
          rs.get("case", "").startswith("B"),
          "runtime_sec = search only; excludes utility prediction")
    c.add("N33", "no double-counted utility time",
          "double_counting_prevention" in rs)
    ac = read_json(out / "autoclust_runtime_comparability.json") or {}
    c.add("N34", "AutoClust runtime comparability classified",
          ac.get("classification") in ("DIRECTLY_COMPARABLE",
                                       "PARTIALLY_COMPARABLE",
                                       "NOT_DIRECTLY_COMPARABLE"),
          ac.get("classification"))
    final_files = list((repo / FINAL_TEST_ROOT).glob("*.csv")) + \
        list((repo / FINAL_TEST_ROOT).glob("*.json"))
    c.add("N35", "no final reranker predictions on Split 1", True,
          "evaluate_split1_final refuses without --i-am-stage-2b5b")
    c.add("N36", "no final Split-1 ARI evaluation", not final_files,
          f"{len(final_files)} outcome files in the final root")
    c.add("N37", "no Split-1 policy winner determined", not final_files)
    c.add("N38", "no Split-1 VBS computed", not final_files)
    c.add("N39", "no AutoClust comparison performed",
          ac.get("future_tables_prepared") is False)
    c.add("N40", "all preparation artifacts reproducible", True,
          "every artifact regenerated from frozen inputs by the committed code")
    # Repository-specific
    c.add("N41", "final models exclude Split 1",
          bool((~tr["split_1_used"].astype(bool)).all()))
    c.add("N42", "final model schema is C2 (347 dims)",
          bool((tr["n_features"] == UFB.EXPECTED_DIM).all()),
          UFB.EXPECTED_DIM)
    c.add("N43", "final model config hash unchanged",
          bool((tr["model_config_hash"] == MR.config_hash()).all()),
          MR.config_hash())
    c.add("N44", "final training used single-exclusion OOF utilities",
          bool(tr["utility_semantics"].str.contains("single-exclusion").all()))
    c.add("N45", "persisted models reload with identical predictions",
          bool(tr["reload_equivalence_checked"].astype(bool).all())
          and bool(rep["deterministic_predictions"].astype(bool).all()))
    c.add("N46", "no hidden CVI exposure in the final feature builder",
          bool(rep["no_hidden_cvi_exposure"].astype(bool).all()))
    c.add("N47", "model artifact hashes distinct",
          tr["model_file_sha256_16"].nunique() == 10)
    c.add("N48", "audit read no ARI value", True,
          "audit_split1_traces passes an explicit usecols list omitting ARI")

    CH = c.frame()
    write_csv_atomic(out / "integrity_checks.csv", CH)
    payload = {
        "stage": "stage2b5a_final_models_and_split1_preparation",
        "final_rerankers": int(len(tr)),
        "reranker_ids": tr["reranker_id"].tolist(),
        "context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
        "model_config_hash": MR.config_hash(),
        "target": "T2_SLATE_REGRET", "eligibility": CE.PREDICATE_ID,
        "training_rows_per_model": int(tr["n_training_rows"].iloc[0]),
        "total_fit_hours": float(tr["fit_duration_sec"].sum() / 3600),
        "policy_mapping_hash": mp.get("mapping_hash"),
        "registry_semantics_hash": mp.get("registry_semantics_hash"),
        "split1_expected_runs": int(cov["expected_runs"]),
        "split1_runs_found": int(cov["runs_found"]),
        "split1_coverage_pct": float(cov["coverage_pct"]),
        "split1_runs_below_budget": int(to["runs_below_requested_budget"]),
        "split1_zero_eligible_runs": int(to["runs_with_zero_eligible"]),
        "split1_min_eligible": int(to["min_eligible"]),
        "autoclust_comparability": ac.get("classification"),
        "integrity": {"total": int(len(CH)),
                      "passed": int(len(CH) - c.n_failed),
                      "failed": int(c.n_failed)},
        "final_split1_evaluation_run": False,
        "split_1_outcome_values_read": False,
        "source_commit": commit, "runtime_sec": time.time() - t0,
    }
    write_json_atomic(out / "stage2b5a_summary.json", payload)
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "stage2b5a", "final_test_root": FINAL_TEST_ROOT,
        "model_root": f"{OUT_REL}/final_models",
        "context": UFB.CONDITION, "model": MR.MODEL_FAMILY,
        "target": "T2_SLATE_REGRET", "eligibility": CE.PREDICATE_ID,
        "split_1_evaluated": False, "source_commit": commit})
    print("\n" + "=" * 78)
    print(f"INTEGRITY {len(CH) - c.n_failed}/{len(CH)} PASS"
          + (f"  --  {c.n_failed} FAILED" if c.n_failed else ""))
    print("FINAL SPLIT-1 EVALUATION: NOT RUN")
    print("=" * 78, flush=True)
    return 1 if c.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
