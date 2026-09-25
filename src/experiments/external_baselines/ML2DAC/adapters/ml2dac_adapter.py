"""The ML2DAC adapter -- the single integration point with the external project.

Design contract
---------------
* This module is the ONLY layer that imports ML2DAC. All ML2DAC imports are
  *lazy* (inside :meth:`fit_predict`) so importing this module never fails when
  the (py3.9, smac==1.4.0, ...) stack is absent. Import/artifact failures are
  reported as a structured ``failed`` result, never raised.
* ML2DAC's application phase returns a SMAC optimizer object, not labels
  directly; labels are read from its run history
  (``opt.get_incumbent_stats()["labels"]``). A reconstruction fallback re-applies
  the best configuration via ``ClusteringCS`` if needed.
* ML2DAC StandardScales the input internally, so the adapter passes the RAW
  partition space (no pre-normalisation -> no double scaling).
* Fairness: ML2DAC receives ONLY the partition space. ``true_labels=None`` is
  passed through; no true K, no metric utilities, no family metadata.

Budget
------
ML2DAC's SMAC scenario enforces BOTH an iteration budget
(``n_optimizer_loops`` -> ``runcount-limit``) AND a wall-clock budget
(``time_limit`` -> ``wallclock-limit``). So ``time_budget_sec`` maps directly to
a real wall-clock limit (unlike AutoML4Clust). ``random_state`` is fixed
internally to 1234 (deterministic).

Artifacts
---------
The full ML2DAC baseline requires the trained MetaKnowledgeRepository. If it is
missing and ``fail_on_missing_mkr`` / ``artifact_policy`` requires it, the
adapter returns ``status="failed"`` with ``failure_kind="artifacts_missing"``
and a clear message -- it never silently runs a degraded variant.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.path_utils import (
    default_mkr_path,
    ensure_ml2dac_on_path,
    external_ml2dac_exists,
    external_ml2dac_src,
)
from ..utils.safe_execution import build_failure
from .artifact_locator import DEFAULT_MF_SET_INDEX, locate_artifacts

_NOISE_SENTINELS = (-2,)  # ML2DAC uses -2 when labels are missing (cutoff/error)


@contextmanager
def _chdir(target: Path):
    """Temporarily change CWD (SMAC scatters ``smac/BO/`` output into CWD)."""
    prev = os.getcwd()
    try:
        os.makedirs(str(target), exist_ok=True)
        os.chdir(str(target))
        yield
    finally:
        try:
            os.chdir(prev)
        except Exception:
            pass


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class ML2DACAdapter:
    """Meta-learning AutoClustering via the external ML2DAC project.

    Parameters
    ----------
    config:
        Resolved configuration dict (see ``configs/ml2dac_full_meta_learning.json``).
        Recognised keys: ``use_meta_cvi_selection`` (bool), ``use_warmstarting``
        (bool), ``use_algorithm_reduction`` (bool), ``time_budget_sec`` (wall-clock
        seconds), ``n_optimizer_loops`` (int budget), ``n_warmstarts`` (int),
        ``n_similar_datasets`` (int), ``mf_set_index`` (int, default 4),
        ``mkr_path`` (str|null), ``artifact_policy``
        ("require_pretrained_or_fail" | "allow_degraded"), ``fail_on_missing_mkr``
        (bool), ``fail_on_invalid_clustering`` (bool), ``cvi`` (optional fixed CVI
        abbrev when meta CVI selection is off).
    repo_root:
        Repository root; used to locate ``external/ml2dac/src`` and the MKR.
    work_dir:
        Optional scratch dir for SMAC output. Defaults to a per-call temp folder.
    """

    METHOD_NAME = "ML2DAC"

    def __init__(self, config: Dict[str, Any], repo_root: str | Path,
                 work_dir: Optional[str | Path] = None) -> None:
        self.config = dict(config or {})
        self.repo_root = Path(repo_root)
        self.work_dir = Path(work_dir) if work_dir is not None else None

        self.use_meta_cvi_selection = bool(self.config.get("use_meta_cvi_selection", True))
        self.use_warmstarting = bool(self.config.get("use_warmstarting", True))
        self.use_algorithm_reduction = bool(self.config.get("use_algorithm_reduction", True))
        self.time_budget_sec = self.config.get("time_budget_sec", 120)
        self.n_optimizer_loops = int(self.config.get("n_optimizer_loops") or 100)
        self.n_warmstarts = int(self.config.get("n_warmstarts") if self.config.get("n_warmstarts") is not None else 25)
        self.n_similar_datasets = int(self.config.get("n_similar_datasets") or 1)
        self.mf_set_index = int(self.config.get("mf_set_index", DEFAULT_MF_SET_INDEX))
        self.mkr_path_cfg = self.config.get("mkr_path")
        self.artifact_policy = str(self.config.get("artifact_policy", "require_pretrained_or_fail"))
        self.fail_on_missing_mkr = bool(self.config.get("fail_on_missing_mkr", True))
        self.fail_on_invalid_clustering = bool(self.config.get("fail_on_invalid_clustering", False))
        self.fixed_cvi = self.config.get("cvi")  # only used when meta CVI selection is off
        self.enforce_timeout = bool(self.config.get("enforce_wallclock_timeout", False))

    # ------------------------------------------------------------------ public
    def fit_predict(
        self,
        partition_space: np.ndarray,
        evaluation_space: Optional[np.ndarray] = None,
        *,
        record_type: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run ML2DAC on ``partition_space``; return a normalised result dict."""
        t0 = time.perf_counter()
        mkr = Path(self.mkr_path_cfg) if self.mkr_path_cfg else default_mkr_path(self.repo_root)
        artifact_status = locate_artifacts(self.repo_root, mkr_path=mkr)

        base = self._base_result(record_type, artifact_status)

        # --- input validation --------------------------------------------
        X = np.asarray(partition_space, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] < 3:
            err = ValueError(f"partition_space has unusable shape {X.shape} (need >=3 rows, 2-D).")
            return self._fail(base, err, t0, stage="input_validation")

        # --- artifact gate ------------------------------------------------
        if not artifact_status.get("ml2dac_src_exists"):
            err = ImportError(
                "external/ml2dac not found (expected "
                f"{external_ml2dac_src(self.repo_root)}). "
                "Run: git submodule update --init --recursive"
            )
            return self._fail(base, err, t0, stage="locate_external",
                              failure_kind="missing_repo")

        if not artifact_status.get("has_pretrained_artifacts"):
            if self.fail_on_missing_mkr or self.artifact_policy == "require_pretrained_or_fail":
                missing = artifact_status.get("missing_required_artifacts")
                err = RuntimeError(
                    "ML2DAC MetaKnowledgeRepository artifacts are missing: "
                    f"{missing}. The full ML2DAC baseline requires the trained MKR "
                    "(evaluated_configs.csv, meta-feature kdtree, CVI classifier). "
                    "Build it via external/ml2dac LearningPhase.py or provide "
                    "mkr_path, then re-run. (Did 'git lfs pull' fetch the CSVs?)"
                )
                return self._fail(base, err, t0, stage="artifact_check",
                                  failure_kind="artifacts_missing")
            base["details"] = {"note": "artifact_policy=allow_degraded but degraded "
                                       "ML2DAC is not implemented; failing."}
            err = RuntimeError("ML2DAC cannot run without the MKR; degraded mode unsupported.")
            return self._fail(base, err, t0, stage="artifact_check",
                              failure_kind="artifacts_missing")

        try:
            return self._run_ml2dac(X, evaluation_space, record_type, dataset_id, mkr, base, t0)
        except Exception as exc:  # noqa: BLE001 - never propagate
            return self._fail(base, exc, t0, stage="fit_predict")

    # ----------------------------------------------------------------- private
    def _run_ml2dac(self, X, evaluation_space, record_type, dataset_id, mkr, base, t0):
        ensure_ml2dac_on_path(self.repo_root)
        try:
            from MetaLearning.ApplicationPhase import ApplicationPhase  # type: ignore
            from MetaLearning import MetaFeatureExtractor  # type: ignore
            from Optimizer.OptimizerSMAC import SMACOptimizer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="import_ml2dac")

        # Resolve meta-feature set + CVI mode.
        try:
            mf_sets = MetaFeatureExtractor.meta_feature_sets
            mf_set = mf_sets[self.mf_set_index]
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="resolve_mf_set")

        # Warmstart / algorithm-reduction policy (faithful to ML2DAC constraints).
        n_warmstarts = self.n_warmstarts if self.use_warmstarting else 0
        limit_cs = bool(self.use_algorithm_reduction)
        if limit_cs and n_warmstarts <= 0:
            # ML2DAC forbids limit_cs without warmstarts; respect that.
            limit_cs = False
        n_loops = max(self.n_optimizer_loops, n_warmstarts)

        cvi_arg: Any = "predict"
        if not self.use_meta_cvi_selection:
            cvi_arg = self._resolve_fixed_cvi()  # may raise -> handled below
            if cvi_arg is None:
                err = ValueError("use_meta_cvi_selection=false requires a valid 'cvi' abbrev in config.")
                return self._fail(base, err, t0, stage="resolve_cvi")

        ds_name = f"{dataset_id or 'dataset'}__{record_type or 'rec'}"
        work_dir = self.work_dir or (
            Path(os.environ.get("TEMP", "/tmp")) / "ml2dac_smac"
            / f"{ds_name}_{os.getpid()}"
        )

        # --- run the application phase ------------------------------------
        try:
            with _chdir(work_dir):
                app = ApplicationPhase(mkr_path=Path(mkr), mf_set=mf_set)
                opt_instance, info = app.optimize_with_meta_learning(
                    X,
                    dataset_name=ds_name,
                    n_warmstarts=n_warmstarts,
                    n_optimizer_loops=n_loops,
                    cvi=cvi_arg,
                    limit_cs=limit_cs,
                    time_limit=int(self.time_budget_sec) if self.time_budget_sec else 120 * 60,
                    optimizer=SMACOptimizer,
                    n_similar_datasets=self.n_similar_datasets,
                )
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="application_phase")

        # --- extract labels (with reconstruction fallback) ----------------
        try:
            labels = self._extract_labels(opt_instance, X)
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="extract_labels")
        if labels is None:
            err = RuntimeError("Could not extract or reconstruct labels from ML2DAC result")
            return self._fail(base, err, t0, stage="extract_labels",
                              error_message="Could not extract or reconstruct labels from ML2DAC result")

        labels = np.asarray(labels).reshape(-1)
        if labels.shape[0] != X.shape[0]:
            err = RuntimeError(f"Label length {labels.shape[0]} != n_points {X.shape[0]}")
            return self._fail(base, err, t0, stage="extract_labels")

        selected_k = int(len({int(u) for u in np.unique(labels)} - {-1}))
        if self.fail_on_invalid_clustering and selected_k < 2:
            err = RuntimeError(f"Invalid clustering: only {selected_k} cluster(s) (fail_on_invalid_clustering=True).")
            return self._fail(base, err, t0, stage="validate_clustering")

        # --- assemble result ----------------------------------------------
        try:
            best_config = opt_instance.get_incumbent().get_dictionary()
        except Exception:
            best_config = {}
        selected_algorithm = best_config.get("algorithm")
        selected_cvi = info.get("cvi")
        selected_algorithms = info.get("algorithms")
        if isinstance(selected_algorithms, str):
            selected_algorithms = [selected_algorithms]

        raw_result = {
            "additional_info": _to_serializable(info),
            "best_configuration": _to_serializable(best_config),
            "n_warmstarts": n_warmstarts,
            "n_optimizer_loops": n_loops,
            "limit_cs": limit_cs,
            "mf_set": _to_serializable(mf_set),
        }
        try:
            rh = opt_instance.get_runhistory_df()
            raw_result["n_configs_evaluated"] = int(len(rh))
        except Exception:
            raw_result["n_configs_evaluated"] = None

        runtime = time.perf_counter() - t0
        base.update({
            "status": "success",
            "labels": labels.tolist(),
            "selected_k": selected_k,
            "selected_cvi": selected_cvi,
            "selected_algorithms": _to_serializable(selected_algorithms),
            "selected_algorithm": selected_algorithm,
            "warmstart_configurations": None,  # not exposed by ML2DAC API
            "warmstart_count": int(n_warmstarts),
            "best_configuration": _to_serializable(best_config),
            "raw_ml2dac_result": raw_result,
            "runtime_sec": float(runtime),
            "error": None,
            "details": {
                "mode": self.config.get("mode", "full_meta_learning"),
                "use_meta_cvi_selection": self.use_meta_cvi_selection,
                "use_warmstarting": self.use_warmstarting,
                "use_algorithm_reduction": self.use_algorithm_reduction,
                "n_optimizer_loops": n_loops,
                "n_warmstarts": n_warmstarts,
                "n_similar_datasets": self.n_similar_datasets,
                "mf_set_index": self.mf_set_index,
                "time_budget_sec": self.time_budget_sec,
                "n_points": int(X.shape[0]),
                "partition_dim": int(X.shape[1]),
                "evaluation_dim": (
                    int(np.asarray(evaluation_space).shape[1])
                    if evaluation_space is not None and np.asarray(evaluation_space).ndim == 2
                    else None
                ),
                "mf_time_sec": info.get("mf time"),
                "similar_dataset": _to_serializable(info.get("similar dataset")),
                "budget_note": "time_budget_sec enforced as SMAC wallclock-limit; "
                               "n_optimizer_loops is the runcount-limit.",
                "note": "ML2DAC StandardScales input internally; raw partition "
                        "space passed (no double scaling).",
            },
        })
        return base

    def _extract_labels(self, opt_instance, X) -> Optional[np.ndarray]:
        """Get labels from the incumbent run-history; reconstruct if degenerate."""
        labels = None
        try:
            stats = opt_instance.get_incumbent_stats()
            labels = np.asarray(stats.get("labels"))
        except Exception:
            labels = None

        def _bad(lab) -> bool:
            if lab is None:
                return True
            lab = np.asarray(lab).reshape(-1)
            if lab.size != X.shape[0]:
                return True
            uniq = set(int(u) for u in np.unique(lab))
            return uniq.issubset(set(_NOISE_SENTINELS))  # all -2 -> invalid

        if not _bad(labels):
            return labels

        # Reconstruction: re-apply the best configuration via ClusteringCS.
        try:
            from sklearn.preprocessing import StandardScaler
            from ClusteringCS import ClusteringCS  # type: ignore
            best_config = opt_instance.get_incumbent()
            best_dict = best_config.get_dictionary()
            algo = best_dict.get("algorithm")
            algo_map = ClusteringCS.get_ALGORITHMS_MAP()
            if algo not in algo_map:
                return None
            scaled = StandardScaler().fit_transform(X)
            recon = algo_map[algo].execute_config(scaled, best_config)
            recon = np.asarray(recon).reshape(-1)
            if recon.size == X.shape[0]:
                return recon
        except Exception:
            return None
        return None

    def _resolve_fixed_cvi(self):
        """Resolve a fixed CVI object from its abbrev (only when meta selection off)."""
        if not self.fixed_cvi:
            return None
        try:
            from ClusterValidityIndices.CVIHandler import CVICollection  # type: ignore
            return CVICollection.get_cvi_by_abbrev(str(self.fixed_cvi))
        except Exception:
            return None

    def _base_result(self, record_type, artifact_status) -> Dict[str, Any]:
        return {
            "method": self.METHOD_NAME,
            "record_type": record_type,
            "status": "failed",
            "labels": None,
            "selected_k": None,
            "selected_cvi": None,
            "selected_algorithms": None,
            "selected_algorithm": None,
            "warmstart_configurations": None,
            "warmstart_count": None,
            "best_configuration": None,
            "raw_ml2dac_result": None,
            "runtime_sec": 0.0,
            "artifact_status": artifact_status,
            "error": None,
        }

    def _fail(self, base, error, t0, *, stage, error_message=None,
              failure_kind=None) -> Dict[str, Any]:
        failure = build_failure(error, runtime_sec=time.perf_counter() - t0, stage=stage)
        base.update(failure)
        base["error"] = error_message or failure.get("error_message")
        if failure_kind:
            base["failure_kind"] = failure_kind
        return base


__all__ = ["ML2DACAdapter"]
