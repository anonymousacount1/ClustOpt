"""The AutoML4Clust adapter -- the single integration point with the external project.

Design contract
---------------
* This module is the ONLY layer that imports AutoML4Clust. All AutoML4Clust
  imports are *lazy* (performed inside :meth:`fit_predict`) so merely constructing
  the adapter -- or importing this module -- never fails when the heavy, pinned,
  legacy dependency stack (smac / hpbandster / ConfigSpace / sklearn-extra /
  pyclustering) is absent. Import failures are reported as a structured
  ``failed`` result with full diagnostics, never raised.
* AutoML4Clust returns only the best *configuration* ``{k, algorithm}`` -- NOT
  labels. The adapter reconstructs labels by re-applying that configuration to
  the (same) partition space via AutoML4Clust's own ``run_algorithm``.
* Fairness: AutoML4Clust receives ONLY the partition space. No true labels, no
  true K, no ClustOpt metric utilities, no family metadata.

Budget
------
AutoML4Clust's native budget is an *iteration count* (``n_loops`` -> the SMAC
``runcount-limit``), not wall-clock time. ``time_budget_sec`` in the config is
recorded for provenance and (optionally) used as a soft thread-based safety
timeout, but the deterministic, comparable budget is ``n_loops``. See the README
for the full discussion.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.path_utils import (
    ensure_automl4clust_on_path,
    external_automl4clust_exists,
    external_automl4clust_src,
)
from ..utils.safe_execution import build_failure

# Metric name (config) -> MetricCollection attribute name.
_METRIC_ATTR = {
    "silhouette": "SILHOUETTE",
    "calinski_harabasz": "CALINSKI_HARABASZ",
    "calinski-harabasz": "CALINSKI_HARABASZ",
    "davies_bouldin": "DAVIES_BOULDIN",
    "davies-bouldin": "DAVIES_BOULDIN",
}

_DEFAULT_N_LOOPS = 60


@contextmanager
def _chdir(target: Path):
    """Temporarily change CWD (SMAC scatters ``smac3-output`` dirs into CWD)."""
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
    """Best-effort conversion of AutoML4Clust result objects to JSON-safe data."""
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


class AutoML4ClustAdapter:
    """Black-box AutoClustering via the external AutoML4Clust project.

    Parameters
    ----------
    config:
        Resolved configuration dict (see ``configs/automl4clust_smac_2d.json``).
        Recognised keys: ``optimizer`` ("SMAC" | "Random"), ``metric``
        ("silhouette" | "calinski_harabasz" | "davies_bouldin"), ``n_loops``
        (int optimizer budget), ``time_budget_sec`` (provenance / soft timeout),
        ``random_state`` (int), ``normalize_input`` (bool), ``use_warmstarting``
        (bool), ``fail_on_invalid_clustering`` (bool), ``k_range`` ([lo, hi] or
        null=auto), ``algorithms`` (list or null=AutoML4Clust default 4).
    repo_root:
        Repository root; used to locate ``external/Automl4Clust/src``.
    work_dir:
        Optional directory for SMAC scratch output. Defaults to a per-call temp
        folder under the system temp dir.
    """

    METHOD_NAME = "AutoML4Clust"

    def __init__(self, config: Dict[str, Any], repo_root: str | Path,
                 work_dir: Optional[str | Path] = None) -> None:
        self.config = dict(config or {})
        self.repo_root = Path(repo_root)
        self.work_dir = Path(work_dir) if work_dir is not None else None

        self.optimizer_name = str(self.config.get("optimizer", "SMAC"))
        self.metric_name = str(self.config.get("metric", "silhouette")).lower()
        self.n_loops = int(self.config.get("n_loops") or _DEFAULT_N_LOOPS)
        self.time_budget_sec = self.config.get("time_budget_sec")
        self.random_state = self.config.get("random_state", 42)
        self.normalize_input = bool(self.config.get("normalize_input", True))
        self.use_warmstarting = bool(self.config.get("use_warmstarting", False))
        self.fail_on_invalid_clustering = bool(
            self.config.get("fail_on_invalid_clustering", False)
        )
        self.k_range = self.config.get("k_range")           # [lo, hi] or None
        self.algorithms = self.config.get("algorithms")     # list or None
        # Soft wall-clock timeout: disabled unless explicitly enabled.
        self.enforce_timeout = bool(self.config.get("enforce_wallclock_timeout", False))

    # ------------------------------------------------------------------ public
    def fit_predict(
        self,
        partition_space: np.ndarray,
        evaluation_space: Optional[np.ndarray] = None,
        *,
        record_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run AutoML4Clust on ``partition_space`` and return a normalised result.

        ``evaluation_space`` (the full 2-D space) is accepted for interface
        completeness and recorded in the result; external metrics (ARI/NMI/AMI)
        are computed by the caller from the returned labels vs. true labels, so
        the clustering itself only ever sees ``partition_space``.
        """
        t0 = time.perf_counter()
        base: Dict[str, Any] = {
            "method": self.METHOD_NAME,
            "optimizer": self.optimizer_name,
            "record_type": record_type,
            "status": "failed",
            "labels": None,
            "selected_k": None,
            "best_configuration": None,
            "raw_automl4clust_result": None,
            "runtime_sec": 0.0,
            "error": None,
        }

        X = np.asarray(partition_space, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] < 3:
            err = ValueError(
                f"partition_space has unusable shape {X.shape} (need >=3 rows, 2-D)."
            )
            return self._fail(base, err, t0, stage="input_validation")

        timeout = float(self.time_budget_sec) if (self.enforce_timeout and self.time_budget_sec) else None

        try:
            if timeout:
                from ..utils.safe_execution import safe_call
                result, failure, runtime = safe_call(
                    self._run_automl4clust, X, evaluation_space, record_type,
                    timeout_sec=timeout,
                )
                if failure is not None:
                    base.update(failure)
                    base["error"] = failure.get("error_message")
                    base["runtime_sec"] = runtime
                    return base
                return result
            return self._run_automl4clust(X, evaluation_space, record_type)
        except Exception as exc:  # noqa: BLE001 - never propagate
            return self._fail(base, exc, t0, stage="fit_predict")

    # ----------------------------------------------------------------- private
    def _run_automl4clust(
        self,
        X: np.ndarray,
        evaluation_space: Optional[np.ndarray],
        record_type: Optional[str],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        base: Dict[str, Any] = {
            "method": self.METHOD_NAME,
            "optimizer": self.optimizer_name,
            "record_type": record_type,
            "status": "failed",
            "labels": None,
            "selected_k": None,
            "best_configuration": None,
            "raw_automl4clust_result": None,
            "runtime_sec": 0.0,
            "error": None,
        }

        # --- 1. Ensure the external project is importable -------------------
        if not external_automl4clust_exists(self.repo_root):
            err = ImportError(
                "external/Automl4Clust not found (expected "
                f"{external_automl4clust_src(self.repo_root)}). Did the git "
                "submodule get initialised? Run: git submodule update --init --recursive"
            )
            return self._fail(base, err, t0, stage="locate_external")

        ensure_automl4clust_on_path(self.repo_root)

        try:
            from Optimizer.Optimizer import SMACOptimizer, RandomOptimizer  # type: ignore
            from Metrics.MetricHandler import MetricCollection  # type: ignore
            from Algorithm import ClusteringAlgorithms  # type: ignore
            from Algorithm.ClusteringAlgorithms import run_algorithm  # type: ignore
            import ConfigSpace as CS  # type: ignore
            import ConfigSpace.hyperparameters as CSH  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="import_automl4clust")

        # Apply runtime compatibility shims for known upstream bugs WITHOUT
        # touching the external source (see _apply_upstream_shims).
        try:
            self._apply_upstream_shims(MetricCollection)
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="apply_upstream_shims")

        # --- 2. Prepare the (optionally normalised) clustering input -------
        X_cluster = X
        if self.normalize_input:
            try:
                from sklearn.preprocessing import StandardScaler
                X_cluster = StandardScaler().fit_transform(X)
            except Exception as exc:  # noqa: BLE001
                return self._fail(base, exc, t0, stage="normalize_input")

        # --- 3. Resolve metric + configuration space -----------------------
        metric_attr = _METRIC_ATTR.get(self.metric_name)
        if metric_attr is None:
            err = ValueError(
                f"Unknown metric '{self.metric_name}'. Expected one of "
                f"{sorted(set(_METRIC_ATTR))}."
            )
            return self._fail(base, err, t0, stage="resolve_metric")
        metric = getattr(MetricCollection, metric_attr)

        try:
            cs = self._build_configspace(X_cluster, ClusteringAlgorithms, CS, CSH)
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="build_configspace")

        # --- 4. Run the optimizer ------------------------------------------
        work_dir = self.work_dir or (
            Path(os.environ.get("TEMP", "/tmp"))
            / "automl4clust_smac"
            / f"{record_type or 'rec'}_{os.getpid()}"
        )
        try:
            with _chdir(work_dir):
                optimizer = self._make_optimizer(
                    SMACOptimizer, RandomOptimizer, X_cluster, metric, cs,
                )
                raw_result = optimizer.optimize()
                best_cfg = optimizer.get_best_configuration()
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="optimize")

        # --- 5. Reconstruct labels from the best configuration -------------
        try:
            best_k = int(best_cfg["k"])
            best_algo = best_cfg.get("algorithm") or ClusteringAlgorithms.KMEANS_ALGORITHM
            clustering_result = run_algorithm(best_algo, X_cluster, k=best_k)
            labels = clustering_result.labels
        except Exception as exc:  # noqa: BLE001
            return self._fail(base, exc, t0, stage="reconstruct_labels")

        if labels is None:
            err = RuntimeError("Could not extract labels from AutoML4Clust result")
            return self._fail(base, err, t0, stage="reconstruct_labels",
                              error_message="Could not extract labels from AutoML4Clust result")

        labels = np.asarray(labels).reshape(-1)
        if labels.shape[0] != X_cluster.shape[0]:
            err = RuntimeError(
                f"Reconstructed label length {labels.shape[0]} != n_points "
                f"{X_cluster.shape[0]}"
            )
            return self._fail(base, err, t0, stage="reconstruct_labels")

        selected_k = int(len({int(u) for u in np.unique(labels)} - {-1}))

        # Optional strictness: treat a single-cluster / all-noise result as a
        # failure when the config requests it (default: keep + report).
        if self.fail_on_invalid_clustering and selected_k < 2:
            err = RuntimeError(
                f"Invalid clustering: only {selected_k} cluster(s) produced "
                f"(fail_on_invalid_clustering=True)."
            )
            return self._fail(base, err, t0, stage="validate_clustering")

        runtime = time.perf_counter() - t0
        return {
            "method": self.METHOD_NAME,
            "optimizer": self.optimizer_name,
            "record_type": record_type,
            "status": "success",
            "labels": labels.tolist(),
            "selected_k": selected_k,
            "best_configuration": {"k": best_k, "algorithm": str(best_algo)},
            "raw_automl4clust_result": _to_serializable(raw_result),
            "runtime_sec": float(runtime),
            "error": None,
            "details": {
                "metric": self.metric_name,
                "n_loops": self.n_loops,
                "normalize_input": self.normalize_input,
                "n_points": int(X_cluster.shape[0]),
                "partition_dim": int(X_cluster.shape[1]),
                "evaluation_dim": (
                    int(np.asarray(evaluation_space).shape[1])
                    if evaluation_space is not None
                    and np.asarray(evaluation_space).ndim == 2
                    else None
                ),
                "k_range_used": self._resolved_k_range,
                "time_budget_sec": self.time_budget_sec,
                "budget_note": "Enforced budget is n_loops (iterations); "
                               "time_budget_sec is provenance only.",
            },
        }

    @staticmethod
    def _apply_upstream_shims(MetricCollection) -> None:
        """Patch known upstream AutoML4Clust bugs at runtime (no source edits).

        Bug: ``Metrics/MetricHandler.py::Metric.score_metric`` references
        ``MetricCollection.SILHOUETTE_SAMPLE_10``, but that attribute is never
        defined on the class (it is commented out in the metric lists). Because
        ``score_metric`` checks ``MetricCollection.SILHOUETTE_SAMPLE_10.name``
        on its first line, *every* metric evaluation raises
        ``AttributeError`` and SMAC aborts on the first run.

        We add the missing attribute as a ``Metric`` with a name that does NOT
        collide with the real metrics (Silhouette / Calinski-Harabasz /
        Davies-Bouldin), so the special-case branch is simply skipped and normal
        scoring proceeds. On Linux, pynisher runs target evaluations in forked
        subprocesses that inherit this class-level patch.
        """
        if getattr(MetricCollection, "SILHOUETTE_SAMPLE_10", None) is not None:
            return
        from Metrics.MetricHandler import Metric, MetricType  # type: ignore
        from sklearn import metrics as _skm  # type: ignore
        MetricCollection.SILHOUETTE_SAMPLE_10 = Metric(
            "Sampled Silhouette 10", _skm.silhouette_score, MetricType.INTERNAL,
            sample_size=0.1,
        )

    def _build_configspace(self, X_cluster, ClusteringAlgorithms, CS, CSH):
        """Construct the ConfigSpace (k range + algorithm choices).

        Mirrors AutoML4Clust's default (k in [2, n/10], all 4 k-center algos)
        but clamps the upper bound so tiny datasets remain valid.
        """
        n = int(X_cluster.shape[0])
        if self.k_range and len(self.k_range) == 2:
            lo, hi = int(self.k_range[0]), int(self.k_range[1])
        else:
            lo, hi = 2, max(2, n // 10)
        lo = max(2, lo)
        hi = max(lo + 1, min(hi, max(2, n - 1)))
        self._resolved_k_range = [lo, hi]

        algorithms = self.algorithms or list(ClusteringAlgorithms.algorithms)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CSH.CategoricalHyperparameter("algorithm", choices=list(algorithms))
        )
        cs.add_hyperparameter(
            CSH.UniformIntegerHyperparameter("k", lower=lo, upper=hi)
        )
        return cs

    def _make_optimizer(self, SMACOptimizer, RandomOptimizer, X_cluster, metric, cs):
        warmstart = None  # fairness: no warmstart configs derived from anything
        name = self.optimizer_name.strip().lower()
        if name in ("smac", "bo", "bayes"):
            return SMACOptimizer(
                dataset=X_cluster, metric=metric, cs=cs,
                n_loops=self.n_loops, warmstart_configs=warmstart,
            )
        if name in ("random", "rs"):
            # Documented fallback. NOTE: upstream RandomOptimizer has a known
            # bug (references a global `search_space`); SMAC is the supported
            # primary optimizer. Kept for completeness / future upstream fix.
            return RandomOptimizer(
                dataset=X_cluster, metric=metric, cs=cs,
                n_loops=self.n_loops, warmstart_configs=warmstart,
            )
        raise ValueError(
            f"Unsupported optimizer '{self.optimizer_name}'. Use 'SMAC' (primary) "
            f"or 'Random' (fallback)."
        )

    _resolved_k_range: Optional[List[int]] = None

    def _fail(self, base: Dict[str, Any], error: BaseException, t0: float,
              *, stage: str, error_message: Optional[str] = None) -> Dict[str, Any]:
        failure = build_failure(error, runtime_sec=time.perf_counter() - t0, stage=stage)
        base.update(failure)
        base["error"] = error_message or failure.get("error_message")
        return base


__all__ = ["AutoML4ClustAdapter"]
