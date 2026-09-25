"""Stage 2B-5B evaluator -- PREPARED, NOT EXECUTED IN STAGE 2B-5A.

Running this module performs the final Split-1 online reranker evaluation. It is
deliberately gated: it refuses to run without ``--i-am-stage-2b5b``, so it cannot
be triggered by accident while Stage 2B-5A is the authorised scope.

Discipline it enforces:
  * candidates come from the ACTUAL eligible visited slate -- never extrapolated
    to 50, never synthesised, never re-run;
  * a run with 10 eligible candidates ranks those 10;
  * the reranked candidate id is frozen BEFORE its ARI is read;
  * RAW and SOFTMAX share a reranker but never share a candidate slate;
  * reranker overhead is timed with ``time.perf_counter`` and excludes ARI
    lookup, which is evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)

PREP_REL = ("results_analysis/clustopt_candidate_reranker/"
            "stage2b5a_final_models_and_split1_preparation")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b5_split1_online_final")
TIE_TOL = 1e-12


class FinalRerankerBundle:
    """The ten frozen models, loaded ONCE. Load time is measured separately."""

    def __init__(self, prep: Path) -> None:
        import joblib
        self.models: Dict[str, Any] = {}
        self.load_ms: Dict[str, float] = {}
        for rid in PM.RERANKER_IDS:
            t0 = time.perf_counter()
            self.models[rid] = joblib.load(
                _ext(prep / "final_models" / rid / "model.joblib"))
            self.load_ms[rid] = (time.perf_counter() - t0) * 1000.0

    def total_load_ms(self) -> float:
        return float(sum(self.load_ms.values()))


def eligible_mask(df: pd.DataFrame) -> np.ndarray:
    """ELIGIBLE_CANDIDATE.v1 applied to a historical online trial log."""
    return df["valid"].map(CS.is_valid_flag).to_numpy(bool)


def select_reranked(pred: np.ndarray, eligible: np.ndarray) -> int:
    """argmin predicted regret over eligible candidates; ties -> lowest index.

    Returns a POSITION in the trial log. ARI is not consulted.
    """
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        raise AssertionError("run has zero eligible candidates")
    v = pred[idx]
    tied = idx[np.abs(v - v.min()) <= TIE_TOL]
    return int(tied[0])


def evaluate_run(bundle: FinalRerankerBundle, reranker_id: str,
                 trial_log: pd.DataFrame, features: np.ndarray,
                 ari_column: np.ndarray) -> Dict[str, Any]:
    """One (dataset, view, policy). ARI is read only after the id is frozen."""
    elig = eligible_mask(trial_log)
    t0 = time.perf_counter()
    pred = np.asarray(bundle.models[reranker_id].predict(features),
                      dtype=np.float64)
    predict_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    chosen = select_reranked(pred, elig)
    argmin_ms = (time.perf_counter() - t1) * 1000.0
    if not elig[chosen]:
        raise AssertionError("an ineligible candidate was selected")

    # ---- identity frozen above; ARI read only now (evaluation-only) --------
    reranked_ari = float(ari_column[chosen])
    visited_oracle = float(np.nanmax(ari_column[elig]))
    return {"reranked_index": int(chosen), "reranked_ari": reranked_ari,
            "visited_oracle_ari": visited_oracle,
            "n_eligible": int(elig.sum()), "n_trial_records": int(len(trial_log)),
            "reranker_predict_time_ms": predict_ms,
            "reranker_argmin_time_ms": argmin_ms}


def recovery(reranked: float, original: float, oracle: float,
             tol: float = 1e-9) -> Tuple[float, bool]:
    den = oracle - original
    if den <= tol:
        return float("nan"), False
    return float((reranked - original) / den), True


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2B-5B final Split-1 online reranker evaluation")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--i-am-stage-2b5b", action="store_true",
                   help="explicit authorisation; without it this refuses to run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit-subfamilies", type=int, default=None)
    args = p.parse_args(argv)

    if not args.i_am_stage_2b5b:
        print("REFUSING TO RUN.\n"
              "This is the FINAL Split-1 evaluation. Stage 2B-5A is preparation "
              "only.\nRe-invoke with --i-am-stage-2b5b once Stage 2B-5B is "
              "explicitly authorised.", flush=True)
        return 2

    raise SystemExit(
        "Stage 2B-5B execution body is intentionally not implemented in the "
        "Stage-2B5A commit. The protocol, schemas, model bundle, selection "
        "discipline and timing hooks above are frozen; the driver loop over the "
        "63,300 historical runs is added when Stage 2B-5B is authorised.")


if __name__ == "__main__":
    raise SystemExit(main())
