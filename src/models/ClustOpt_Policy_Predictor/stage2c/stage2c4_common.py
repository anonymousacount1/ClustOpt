"""Stage 2C-4A: shared vocabulary for the online RAW/SOFTMAX development batch.

One place defines what a Stage-2C4 *scientific row* is, how its two twins are
named, where its OOF utility U_{-s} comes from and where its online outputs land.
Every Stage-2C4 module imports from here so the frozen sample, the twin contract
and the path conventions cannot drift between preparation, execution, target
building, training and gate verification.

Nothing scientific is decided here; the regime set, the K semantics and the
weighting semantics are all read from the frozen production registries
(``policy_mapping`` / ``oof_policy_replay``) and asserted against them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (  # noqa: E501
    MethodSpec,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E501
    _ext, read_json,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS
from models.ClustOpt_Candidate_Reranker.training.final_models import (
    policy_mapping as PM,
)
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI

# ---------------------------------------------------------------- identities
STAGE = "2C-4A"
FROZEN_SAMPLE_HASH = "26d3f5f393b5724d"
STAGE2C4P_COMMIT = "c624157ddb0ba177a09ef801fe38921f390ade15"

OUT_REL = ("results_analysis/clustopt_policy_predictor/"
           "stage2c4_online_weighting_mode_dev")
NESTED_REL = ("results_analysis/clustopt_policy_predictor/"
              "stage2a3b0_nested_crossfit")
OOF_UTILITY_REL = ("results_analysis/clustering_repository/stage2/"
                   "oof_utility_splits2_16")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
ANALYZED_REL = "results_analysis/clustering_repository/analyzed_data"
CONFIG_SUBDIR = "stage2c4"

#: Output split folder becomes ``experiments/split_XX_<SPLIT_DIR_SUFFIX>/``.
SPLIT_DIR_SUFFIX = "stage2c4_online_weighting_dev"

VIEWS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")
OUTER_SPLITS: Tuple[int, ...] = RI.OUTER_SPLITS
RERANKER_IDS: Tuple[str, ...] = PM.RERANKER_IDS
MODES: Tuple[str, ...] = ("RAW", "SOFTMAX")

#: Frozen online-search contract, taken from the Stage-2B5 semantics record and
#: from the phaseD/phaseE2 configs those runs were produced with.
N_TRIALS = 50
TIMEOUT_SEC = 150
BASE_SEED = 42
SOFTMAX_TEMPERATURE = 0.5

WEIGHTING_IMPL: Dict[str, str] = {"RAW": "normalized_positive",
                                  "SOFTMAX": "softmax"}

GZ = {"index": False, "compression": "gzip"}


# ------------------------------------------------------------------- regimes
def k_mode_of(regime_id: str) -> str:
    return regime_id.split("_", 1)[1].lower()


def source_of(regime_id: str) -> str:
    return regime_id.split("_", 1)[0].lower()


def top_k_of(regime_id: str) -> Any:
    km = k_mode_of(regime_id)
    return "dynamic" if km == "dynamic" else int(km[3:])


def method_name(regime_id: str, mode: str) -> str:
    """``stage2c4_mlp_top5_raw`` / ``stage2c4_knn_dynamic_softmax_t05``."""
    suffix = "raw" if mode == "RAW" else "softmax_t05"
    return "stage2c4_%s_%s_%s" % (source_of(regime_id), k_mode_of(regime_id),
                                  suffix)


def method_spec(regime_id: str, mode: str) -> MethodSpec:
    name = method_name(regime_id, mode)
    return MethodSpec(name, source_of(regime_id),
                      "%s/%s" % (CONFIG_SUBDIR, name))


def all_method_specs() -> List[MethodSpec]:
    return [method_spec(r, m) for r in RERANKER_IDS for m in MODES]


def regime_of_method(name: str) -> Tuple[str, str]:
    """Inverse of :func:`method_name`. Raises on an unknown method."""
    for r in RERANKER_IDS:
        for m in MODES:
            if method_name(r, m) == name:
                return r, m
    raise KeyError(name)


def assert_registry_alignment() -> Dict[str, Any]:
    """The 20 Stage-2C4 methods must mirror the frozen 20-policy registry.

    Same ten regimes, same RAW/SOFTMAX pairing, same K semantics and the same
    softmax temperature. Checked against ``policy_mapping.build_mapping`` so an
    edit to the production registry breaks this stage loudly instead of silently
    producing a different experiment.
    """
    mapping = pd.DataFrame(PM.build_mapping())
    got: Dict[Tuple[str, str], str] = {}
    for _, r in mapping.iterrows():
        mode = "RAW" if r["weighting_mode"] == "RAW" else "SOFTMAX"
        got[(r["reranker_id"], mode)] = r["policy_id"]
    want = {(r, m) for r in RERANKER_IDS for m in MODES}
    if set(got) != want:
        raise AssertionError("registry regime/mode grid mismatch")
    for (rid, mode), pid in got.items():
        row = mapping[mapping["policy_id"] == pid].iloc[0]
        if row["k_mode"] != k_mode_of(rid) or row["utility_source"] != source_of(rid):
            raise AssertionError("regime decomposition mismatch for %s" % pid)
        if row["weighting_impl"] != WEIGHTING_IMPL[mode]:
            raise AssertionError("weighting impl mismatch for %s" % pid)
    return {
        "n_regimes": len(RERANKER_IDS), "n_methods": 2 * len(RERANKER_IDS),
        "policy_semantics_hash": PM.registry_semantics_hash(),
        "mapping_hash": PM.mapping_hash(PM.build_mapping()),
        "twin_reference_policy": {"%s|%s" % k: v for k, v in sorted(got.items())},
    }


# -------------------------------------------------------------------- sample
def sample_payload_hash(sample: pd.DataFrame) -> str:
    """The frozen-sample hash, computed exactly as ``stage2c4_freeze_sample``."""
    payload = json.dumps(
        sample[["split_id", "rank", "dataset_id", "view_id"]]
        .sort_values(["split_id", "rank", "view_id"])
        .to_dict("records"), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_sample(repo: Path, *, n: Optional[int] = None) -> pd.DataFrame:
    """The frozen ordered development sample, hash-verified. Never regenerated.

    ``n`` restricts to ranks ``1..n`` (the pre-registered runtime fallback); the
    hash is always verified on the FULL frozen table first, so a smaller batch is
    provably a prefix of the same frozen sample rather than a new one.
    """
    f = Path(repo) / OUT_REL / "stage2c4_dev_sample.csv"
    if not Path(_ext(f)).exists():
        raise SystemExit("frozen sample missing: %s (do NOT regenerate)" % f)
    s = pd.read_csv(_ext(f))
    got = sample_payload_hash(s)
    if got != FROZEN_SAMPLE_HASH:
        raise SystemExit("frozen sample hash %s != %s -- refusing to proceed"
                         % (got, FROZEN_SAMPLE_HASH))
    if (s["split_id"] == 1).any():
        raise SystemExit("Split 1 present in the development sample")
    if n is not None:
        s = s[s["rank"] <= int(n)].reset_index(drop=True)
    return s


def dataset_dir(repo: Path, family_id: str, subfamily_id: str,
                dataset_id: str) -> Path:
    return (Path(repo) / ANALYZED_REL / str(family_id) / "subfamilies"
            / str(subfamily_id) / str(dataset_id))


def split_dir_name(split_id: int) -> str:
    return "split_%02d_%s" % (int(split_id), SPLIT_DIR_SUFFIX)


def online_run_dir(ds_dir: Path, split_id: int, method: str, view_id: str) -> Path:
    return (Path(ds_dir) / "experiments" / split_dir_name(split_id)
            / method / view_id)


# ------------------------------------------------------------- OOF utilities
def injected_utility_rel(view_id: str, source: str) -> str:
    """Path of the injected U_{-s} vector, RELATIVE to the dataset folder."""
    return "utility/%s/oof_utility_stage2c4_%s.json" % (view_id, source)


def metric_names(repo: Path) -> List[str]:
    return CS.load_metric_names(Path(repo))


def load_oof_utilities(repo: Path, split_id: int, names: Sequence[str]):
    """``load_oof_index`` for one held-out split, with its own leakage checks."""
    from models.Clustering_Repository_Builder.experiments.experiment_execution.oof_policy_replay import (  # noqa: E501
        load_oof_index,
    )
    return load_oof_index(Path(repo) / OOF_UTILITY_REL, int(split_id), names)


def util120(idx, dataset_id: str, view_id: str) -> np.ndarray:
    """The frozen C2 utility block: 60 MLP then 60 KNN, as in ``_fold_arrays``."""
    return np.concatenate([idx.get("mlp", dataset_id, view_id),
                           idx.get("knn", dataset_id, view_id)]).astype(np.float64)


def vector_hash(v: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(v, dtype=np.float64).tobytes()
                          ).hexdigest()[:16]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(_ext(p), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def sha_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------- rows
def scientific_rows(sample: pd.DataFrame) -> pd.DataFrame:
    """One row per (split, dataset, view) with its frozen OOF-v2 regime."""
    r = sample.copy()
    r["regime_id"] = r["oof_v2_regime"]
    r["source"] = r["regime_id"].map(source_of)
    r["k_mode"] = r["regime_id"].map(k_mode_of)
    r["method_raw"] = [method_name(g, "RAW") for g in r["regime_id"]]
    r["method_softmax"] = [method_name(g, "SOFTMAX") for g in r["regime_id"]]
    r["row_id"] = (r["dataset_id"].astype(str) + "|" + r["view_id"].astype(str))
    return r


__all__ = [
    "STAGE", "FROZEN_SAMPLE_HASH", "STAGE2C4P_COMMIT", "OUT_REL", "NESTED_REL",
    "OOF_UTILITY_REL", "SPLIT_CSV_REL", "ANALYZED_REL", "CONFIG_SUBDIR",
    "SPLIT_DIR_SUFFIX", "VIEWS", "OUTER_SPLITS", "RERANKER_IDS", "MODES",
    "N_TRIALS", "TIMEOUT_SEC", "BASE_SEED", "SOFTMAX_TEMPERATURE",
    "WEIGHTING_IMPL", "GZ", "k_mode_of", "source_of", "top_k_of",
    "method_name", "method_spec", "all_method_specs", "regime_of_method",
    "assert_registry_alignment", "sample_payload_hash", "load_sample",
    "dataset_dir", "split_dir_name", "online_run_dir", "injected_utility_rel",
    "metric_names", "load_oof_utilities", "util120", "vector_hash", "sha_file",
    "sha_frame", "scientific_rows",
]
