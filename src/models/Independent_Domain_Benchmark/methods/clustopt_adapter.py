"""Adapter for the frozen ClustOpt implementation -- the six external arms.

This module ORCHESTRATES. Every scientific component it needs already exists and
is called, never reimplemented:

===============================  ==========================================
step                             frozen source
===============================  ==========================================
meta-features                    ``features_extraction.extract_features_for_view``
MLP / KNN utility                the ``regressor_dynamic`` / ``knn_dynamic``
                                 resolvers, reached through the normal config
Top-K + RAW weighting            ``dynamic_metric_selection.weighting``
policy inference                 the frozen PP models + ``target_transforms``
policy -> regime                 ``oof_policy_replay.build_policy_set``
search                           ``clustopt_execution.run_experiment_clustopt``
final J selection                ``clustopt_execution._best_row``
reranker features                ``online_feature_builder.build_online_X``
reranker selection               ``evaluate_split1_final.select_reranked``
===============================  ==========================================

**Labels never enter.** ``run`` takes ``X`` and identifiers. The ClustOpt search
objective is the aggregated CVI score; ground truth only ever attached an ARI
column to the log, and this adapter does not attach it. There is no ``y``, no
true K and no ARI anywhere in the adapter API or in its output.

**Trace sharing.** C0/C3 and C1/C4 differ only in the FINAL selector, so they
execute one physical search and two selections. The cache is keyed by the frozen
trace identity, so the sharing is structural rather than an optimisation that
could silently stop holding. C2 and C5 are never shared: PP v1 and PP v2 may
choose different policies, so their searches legitimately differ.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import MethodAdapter, MethodResult
from .clustopt_trace import (AdapterError, ClustOptFailure, ClustOptTrace,
                             assert_no_ground_truth, partition_hash)

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

#: the six frozen arms. Every field here must agree with method_registry.json.
ARM_SPEC: Dict[str, Dict[str, Any]] = {
    "IDB_ClustOpt_C0_MLP_TOP5_RAW_noR": {
        "utility_source": "mlp", "top_k": 5, "weighting": "normalized_positive",
        "policy_predictor": None, "reranker": None, "selector": "J"},
    "IDB_ClustOpt_C1_KNN_TOP10_RAW_noR": {
        "utility_source": "knn", "top_k": 10, "weighting": "normalized_positive",
        "policy_predictor": None, "reranker": None, "selector": "J"},
    "IDB_ClustOpt_C2_PPv1_noR": {
        "utility_source": "policy", "top_k": None, "weighting": "policy_selected",
        "policy_predictor": "PP_V1", "reranker": None, "selector": "J"},
    "IDB_ClustOpt_C3_MLP_TOP5_RAW_R": {
        "utility_source": "mlp", "top_k": 5, "weighting": "normalized_positive",
        "policy_predictor": None, "reranker": "MLP_TOP5", "selector": "RERANKER"},
    "IDB_ClustOpt_C4_KNN_TOP10_RAW_R": {
        "utility_source": "knn", "top_k": 10, "weighting": "normalized_positive",
        "policy_predictor": None, "reranker": "KNN_TOP10", "selector": "RERANKER"},
    "IDB_ClustOpt_C5_PPv2_R": {
        "utility_source": "policy", "top_k": None, "weighting": "policy_selected",
        "policy_predictor": "PP_V2", "reranker": "per_policy",
        "selector": "RERANKER"},
}

MLP_RUN_DIR = "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout"
#: the frozen Phase-E1 KNN model directory, as named by the Phase-E2 config
#: generator. method_registry.json records the ablation RUN directory in its
#: ``utility_model_run_dir`` field, which holds no models; see
#: configs/clustopt_adapter_source_audit.json.
KNN_MODEL_DIR = "results_analysis/metric_utility_knn/phase_e1/models"
PP_ROOT = ("results_analysis/clustopt_policy_predictor/"
           "stage4b_materialised_final_models")
RERANKER_PREP = ("results_analysis/clustopt_candidate_reranker/"
                 "stage2b5a_final_models_and_split1_preparation")

CVI_GENERIC = {"remove_noise": True, "noise_label": -1, "min_clusters": 2}


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _vec_hash(v: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(v, dtype=np.float64).tobytes()
                          ).hexdigest()[:16]


# --------------------------------------------------------------- artifacts
class ClustOptArtifacts:
    """Resolve, hash and cache every frozen artifact the six arms need."""

    def __init__(self, repo: Path = _REPO) -> None:
        self.repo = Path(repo)
        self.hashes: Dict[str, str] = {}
        self._pp: Dict[str, Any] = {}
        self._rerankers: Dict[str, Any] = {}
        self._loaded = False

    def _one(self, pattern_dir: Path, pattern: str, what: str) -> Path:
        cands = sorted(pattern_dir.glob(pattern)) if pattern_dir.is_dir() else []
        if not cands:
            raise AdapterError(ClustOptFailure.ARTIFACT_MISSING,
                               "%s: no match for %s/%s" % (what, pattern_dir, pattern))
        if len(cands) > 1:
            raise AdapterError(ClustOptFailure.ARTIFACT_AMBIGUOUS,
                               "%s resolves to %d files" % (what, len(cands)))
        return cands[0]

    def resolve(self) -> Dict[str, str]:
        """Load nothing heavy; just prove every artifact exists and hash it."""
        mlp = self.repo / MLP_RUN_DIR
        if not mlp.is_dir():
            raise AdapterError(ClustOptFailure.ARTIFACT_MISSING, "MLP run dir")
        self.hashes["mlp_config"] = _sha16(mlp / "config.json")
        self.hashes["mlp_fold0"] = _sha16(mlp / "folds/fold_0/best_model.pt")

        knn = self.repo / KNN_MODEL_DIR
        if not knn.is_dir():
            raise AdapterError(ClustOptFailure.ARTIFACT_MISSING, "KNN model dir")
        self.hashes["knn_feature_schema"] = _sha16(knn / "feature_schema.json")
        self.hashes["knn_metric_schema"] = _sha16(knn / "utility_metric_schema.json")
        self.hashes["knn_metadata"] = _sha16(knn / "final_knn_metadata.json")

        # Policy-selector models (PP_V1 / PP_V2) are resolved lazily by _resolve_pp(),
        # only for the arms that use them (C2 / C5). They are not distributed with the
        # public release; C0, C1, C3 and C4 never need them.

        from models.ClustOpt_Candidate_Reranker.training.final_models import (
            policy_mapping as PM)
        fm = self.repo / RERANKER_PREP / "final_models"
        for rid in PM.RERANKER_IDS:
            f = self._one(fm / rid, "model.joblib*", "reranker %s" % rid)
            self._rerankers[rid] = {"path": f}
            self.hashes["reranker_%s" % rid] = _sha16(f)
        self._loaded = True
        return dict(self.hashes)

    def _resolve_pp(self, arm: str) -> None:
        """Resolve and hash one policy-selector model (the original checks, unchanged)."""
        if arm in self._pp:
            return
        d = self.repo / PP_ROOT / ("PP_%s" % arm)
        f = self._one(d, "model.joblib*", "PP_%s" % arm)
        meta = json.loads((d / "artifact_metadata.json").read_text("utf-8"))
        self._pp[arm] = {"path": f, "meta": meta}
        self.hashes["pp_%s" % arm.lower()] = meta["model_sha256_16"]

    # -- lazy heavyweight loads, so an arm only pays for what it uses
    def pp_model(self, arm: str):
        import joblib
        self._resolve_pp(arm)
        e = self._pp[arm]
        if "model" not in e:
            e["model"] = joblib.load(e["path"])
            for est in (e["model"], getattr(e["model"], "estimator", None)):
                if est is not None and hasattr(est, "n_jobs"):
                    est.n_jobs = 1          # frozen PP inference policy
        return e["model"]

    def pp_meta(self, arm: str) -> Dict[str, Any]:
        self._resolve_pp(arm)
        return self._pp[arm]["meta"]

    def reranker(self, rid: str):
        import joblib
        e = self._rerankers.get(rid)
        if e is None:
            raise AdapterError(ClustOptFailure.ARTIFACT_MISSING,
                               "reranker %s" % rid)
        if "model" not in e:
            e["model"] = joblib.load(e["path"])
        return e["model"]


# ------------------------------------------------------------ meta-features
def extract_meta_features(X: np.ndarray, dataset_id: str, view_id: str,
                          work_dir: Path) -> Path:
    """Write the dataset folder the frozen resolvers expect, from X alone.

    The resolvers read ``<dataset_dir>/features/features_records.csv``. Rather
    than bypass them, the adapter produces exactly that file with the project's
    own extractor and its own writer, so the utility predictors receive a record
    built by the same code that built their training records.

    ``labels=None`` is passed deliberately: the extractor takes ground truth only
    to copy audit fields, and an external benchmark has none to give.
    """
    from models.Clustering_Repository_Builder.features_extraction import (
        dataset_loader as DL, feature_extractor as FE, output_writer as OW,
        view_builder as VB)

    ds = work_dir / dataset_id
    ds.mkdir(parents=True, exist_ok=True)
    A = np.asarray(X, dtype=float)
    names = ["x", "y"][:A.shape[1]] or [f"f{i}" for i in range(A.shape[1])]
    df = pd.DataFrame(A, columns=names)
    loaded = DL.LoadedDataset(
        dataset_dir=ds, dataset_id=dataset_id, data_df=df, X_original=A,
        labels=None, feature_names=list(names), metadata={},
        replay_config={}, structural_ground_truth=None, render_summary=None)
    try:
        views = VB.build_views(loaded)
        records = [FE.extract_features_for_view(loaded=loaded, view=v)[0]
                   for v in views]
        OW.write_dataset_outputs(dataset_dir=ds, records=records, timing={},
                                 overwrite=True)
    except Exception as exc:
        raise AdapterError(ClustOptFailure.META_FEATURE_EXTRACTION_FAILED,
                           "%s: %s" % (type(exc).__name__, str(exc)[:200]))
    out = ds / "features" / "features_records.csv"
    if not out.is_file():
        raise AdapterError(ClustOptFailure.META_FEATURE_EXTRACTION_FAILED,
                           "extractor produced no features_records.csv")
    return ds


# ------------------------------------------------------------------ config
def build_config(view_id: str, utility_source: str, top_k: Any, weighting: str,
                 seed: int, budget: int, softmax_temperature: float = 0.5,
                 k_policy: Optional[str] = None) -> Dict[str, Any]:
    """A ClustOpt config identical in shape to the frozen Stage-1/Phase-E2 ones.

    Only the objective block varies between arms; the clustering search space,
    the trial budget and the no-early-stop Optuna block are the frozen shared
    protocol.
    """
    from models.ClustOpt.configs.experiments.generate_experiment_configs import (
        build_search_space)

    if utility_source == "mlp":
        cvi = {"type": "regressor_dynamic",
               "params": {"model_run_dir": MLP_RUN_DIR,
                          "checkpoint_policy": "fold_ensemble"}}
    elif utility_source == "knn":
        cvi = {"type": "knn_dynamic",
               "params": {"knn_model_dir": KNN_MODEL_DIR}}
    else:
        raise AdapterError(ClustOptFailure.POLICY_UNRESOLVED,
                           "unknown utility source %r" % utility_source)
    cvi["params"].update({
        "top_k": top_k, "weighting": weighting,
        "metric_name_prefix": "utility__",
        "feature_source": "features/features_records.csv",
        "view_aware": True,
        # no silent substitution: a resolver failure must surface as a failure
        "fallback_on_error": False,
    })
    if weighting == "softmax":
        cvi["params"]["softmax_temperature"] = float(softmax_temperature)
    if k_policy:
        cvi["params"]["k_policy"] = k_policy
    cvi["params_generic"] = dict(CVI_GENERIC)
    return {"search_space": build_search_space(view_id),
            "search_algorithm": {"type": "optuna",
                                 "params": {"n_trials": int(budget),
                                            "seed": int(seed)}},
            "cvi": cvi}


# ------------------------------------------------------------------- search
def run_search(*, X_view: np.ndarray, X_full: np.ndarray, dataset_dir: Path,
               view_id: str, config: Dict[str, Any], work_dir: Path,
               trace_identity: str, dataset_id: str, seed: int,
               utility_source: str, top_k: Any, weighting: str,
               policy_id: Optional[str]) -> ClustOptTrace:
    """Execute ONE ClustOpt search and package it as a trace. No labels."""
    from models.Clustering_Repository_Builder.experiments.experiment_execution import (  # noqa: E501
        clustopt_execution as CE)

    cfg_path = work_dir / ("config_%s_%s.json" % (trace_identity, view_id))
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    try:
        outcome = CE.run_experiment_clustopt(
            config_path=cfg_path, dataset_dir=dataset_dir, view_id=view_id,
            X_decision=X_view, X_full=X_full,
            y_true=None,                      # <- the label-free contract
            runtime_block={}, attach_profiler=False)
    except Exception as exc:
        raise AdapterError(ClustOptFailure.SEARCH_FAILED,
                           "%s: %s" % (type(exc).__name__, str(exc)[:200]))

    log = outcome.results_df
    if log is None or len(log) == 0:
        raise AdapterError(ClustOptFailure.NO_VALID_CANDIDATE, "empty search log")
    assert_no_ground_truth(log)

    hashes: List[Optional[str]] = []
    from models.ClustOpt import ClusteringOptimizationBuilder
    builder = ClusteringOptimizationBuilder(cfg_path)
    space = builder.build_search_space()
    for _, row in log.iterrows():
        lab = CE._predict_best_labels(space, row, np.asarray(X_view))
        hashes.append(None if lab is None else partition_hash(lab))

    return ClustOptTrace(
        trace_id="%s|%s|%s" % (trace_identity, dataset_id, view_id),
        trace_identity=trace_identity, dataset_id=dataset_id, view_id=view_id,
        search_seed=int(seed), utility_source=utility_source, top_k=top_k,
        weighting=weighting, policy_id=policy_id,
        selected_metrics=list(outcome.selected_metrics),
        selected_metric_weights=dict(outcome.selected_metric_weights),
        utility_vector_hash=None, trial_log=log.reset_index(drop=True),
        n_evaluations=int(outcome.n_trials_completed),
        requested_trials=outcome.requested_trials,
        valid_trials=outcome.valid_trials, failed_trials=outcome.failed_trials,
        partition_hashes=hashes, runtime_sec=float(outcome.runtime_sec),
        extra={"selected_algorithm": outcome.selected_algorithm,
               "used_fallback": bool(outcome.used_fallback),
               "search_space_object": space})


# ---------------------------------------------------------------- selectors
def select_by_objective(trace: ClustOptTrace) -> int:
    """The original ClustOpt final selection: the highest aggregate score."""
    from models.Clustering_Repository_Builder.experiments.experiment_execution import (  # noqa: E501
        clustopt_execution as CE)
    row = CE._best_row(trace.trial_log)
    if row is None:
        raise AdapterError(ClustOptFailure.NO_VALID_CANDIDATE,
                           "no candidate with a finite objective")
    return int(trace.trial_log.index.get_loc(row.name))


def select_by_reranker(trace: ClustOptTrace, reranker_id: str,
                       artifacts: ClustOptArtifacts,
                       utility_vector: np.ndarray,
                       source: str, k_mode: str, actual_k: int) -> Tuple[int, Dict]:
    """The deployed final selection: argmin predicted regret over eligibles."""
    from models.ClustOpt_Candidate_Reranker.split1_preparation import (
        online_feature_builder as OFB, evaluate_split1_final as ESF)

    metric_names = _metric_names(artifacts)
    try:
        Xr, _, diag = OFB.build_online_X(
            trial_log=trace.trial_log, view_id=trace.view_id,
            metric_names=metric_names, utility_vector=utility_vector,
            source=source, k_mode=k_mode, actual_k=int(actual_k))
    except Exception as exc:
        raise AdapterError(ClustOptFailure.RERANKER_SCHEMA_MISMATCH,
                           "%s: %s" % (type(exc).__name__, str(exc)[:200]))
    if Xr.shape[1] != OFB.EXPECTED_DIM:
        raise AdapterError(ClustOptFailure.RERANKER_SCHEMA_MISMATCH,
                           "built %d columns, expected %d"
                           % (Xr.shape[1], OFB.EXPECTED_DIM))
    model = artifacts.reranker(reranker_id)
    try:
        pred = np.asarray(model.predict(Xr), dtype=np.float64)
    except Exception as exc:
        raise AdapterError(ClustOptFailure.RERANKER_FAILED,
                           "%s: %s" % (type(exc).__name__, str(exc)[:200]))
    elig = ESF.eligible_mask(trace.trial_log)
    if not elig.any():
        raise AdapterError(ClustOptFailure.NO_VALID_CANDIDATE,
                           "no eligible candidate for the reranker")
    chosen = ESF.select_reranked(pred, elig)
    return int(chosen), {"n_eligible": int(elig.sum()),
                         "reranker_feature_dim": int(Xr.shape[1]), **diag}


def _metric_names(artifacts: ClustOptArtifacts) -> List[str]:
    from models.ClustOpt_Policy_Predictor.data import feature_schema as FSCH
    return list(FSCH.load_metric_names(artifacts.repo))


# ------------------------------------------------------------------ adapter
class ClustOptAdapter(MethodAdapter):
    """One of the six frozen ClustOpt arms."""

    framework = "ClustOpt"

    #: process-wide, keyed by (trace_identity, dataset_id, view_id) so C0/C3 and
    #: C1/C4 share one physical search
    _TRACE_CACHE: Dict[Tuple[str, str, str], ClustOptTrace] = {}

    def __init__(self, method_id: str, *, repo: Path = _REPO,
                 work_dir: Optional[Path] = None) -> None:
        if method_id not in ARM_SPEC:
            raise AdapterError(ClustOptFailure.POLICY_UNRESOLVED,
                               "unknown ClustOpt method_id %r" % method_id)
        self.method_id = method_id
        self.spec = dict(ARM_SPEC[method_id])
        self.repo = Path(repo)
        self.artifacts = ClustOptArtifacts(self.repo)
        self._work = Path(work_dir) if work_dir else None
        self._own_work = work_dir is None
        self._context_cache: Dict[Tuple[str, str], Any] = {}

    # ------------------------------------------------------------ lifecycle
    def initialise(self) -> Dict[str, str]:
        h = self.artifacts.resolve()
        if self.spec["policy_predictor"]:            # C2 / C5 only
            self.artifacts._resolve_pp(self.spec["policy_predictor"].split("_")[-1])
            h = dict(self.artifacts.hashes)
        needed = ["mlp_config", "mlp_fold0"]
        if self.spec["utility_source"] == "knn":
            needed += ["knn_feature_schema", "knn_metric_schema"]
        if self.spec["policy_predictor"]:
            needed.append("pp_%s" % self.spec["policy_predictor"].lower()[-2:])
        missing = [k for k in needed if k not in h]
        if missing:
            raise AdapterError(ClustOptFailure.ARTIFACT_MISSING, str(missing))
        return h

    def _workdir(self) -> Path:
        if self._work is None:
            self._work = Path(tempfile.mkdtemp(prefix="idb_clustopt_"))
        self._work.mkdir(parents=True, exist_ok=True)
        return self._work

    def close(self) -> None:
        if self._own_work and self._work and self._work.exists():
            shutil.rmtree(self._work, ignore_errors=True)

    # -------------------------------------------------------------- policy
    def resolve_policy(self, meta_row: pd.DataFrame, util_row: pd.DataFrame,
                       view_id: str) -> Dict[str, Any]:
        """PP inference -> one of the 20 frozen policies. Magnitudes unused."""
        from models.ClustOpt_Policy_Predictor.stage2c import (
            run_final_v2_split1_replay as SRC)
        from models.ClustOpt_Policy_Predictor.training import (
            target_transforms as TT)

        arm = self.spec["policy_predictor"].split("_")[-1]     # V1 / V2
        model = self.artifacts.pp_model(arm)
        meta_dim = self.artifacts.pp_meta(arm)["input_dim"]
        Xp, _ = SRC.build_X(meta_row, util_row, pd.Series([view_id]))
        if Xp.shape[1] != meta_dim:
            raise AdapterError(
                ClustOptFailure.FEATURE_SCHEMA_MISMATCH,
                "PP expects %d features, built %d" % (meta_dim, Xp.shape[1]))
        pred = np.asarray(model.predict(Xp), dtype=np.float64)
        idx = int(np.asarray(TT.select_policy(pred, "CENTERED")).ravel()[0])
        table = policy_table()
        if idx < 0 or idx >= len(table):
            raise AdapterError(ClustOptFailure.POLICY_UNRESOLVED,
                               "policy index %d outside the frozen table" % idx)
        row = dict(table[idx])
        row["policy_index"] = idx
        return row

    # ----------------------------------------------------------------- run
    def run(self, X: np.ndarray, *, seed: int, k_min: int, k_max: int,
            budget: int, dataset_id: str = "smoke", view_id: str = "xy_2d",
            dataset_dir: Optional[Path] = None) -> MethodResult:
        """Execute the arm. ``X`` and identifiers only -- never labels."""
        work = self._workdir()
        if dataset_dir is None:
            dataset_dir = extract_meta_features(X, dataset_id, view_id, work)
        X_view = view_matrix(X, view_id)

        spec = self.spec
        policy = None
        if spec["policy_predictor"]:
            meta_row, util_row, uvec = self.context_frames(dataset_dir, view_id)
            policy = self.resolve_policy(meta_row, util_row, view_id)
            usource, top_k = policy["source"], policy["top_k"]
            weighting = policy["weighting"]
            k_policy = policy.get("k_policy")
            softmax_t = policy.get("softmax_temperature") or 0.5
        else:
            usource, top_k = spec["utility_source"], spec["top_k"]
            weighting, k_policy, softmax_t = spec["weighting"], None, 0.5
            uvec = None

        from ..search.protocol import seed_identity
        identity = seed_identity(self.method_id)
        key = (identity, dataset_id, view_id)
        trace = self._TRACE_CACHE.get(key)
        if trace is None:
            cfg = build_config(view_id, usource, top_k, weighting, seed, budget,
                               softmax_t, k_policy)
            trace = run_search(
                X_view=X_view, X_full=np.asarray(X, dtype=float),
                dataset_dir=dataset_dir, view_id=view_id, config=cfg,
                work_dir=work, trace_identity=identity, dataset_id=dataset_id,
                seed=seed, utility_source=usource, top_k=top_k,
                weighting=weighting,
                policy_id=(policy or {}).get("policy_id"))
            self._TRACE_CACHE[key] = trace

        # ---- final selection
        if spec["selector"] == "J":
            pos, sel_extra = select_by_objective(trace), {}
        else:
            if uvec is None:
                _, _, uvec = self.context_frames(dataset_dir, view_id)
            rid = spec["reranker"]
            if rid == "per_policy":
                if policy is None:
                    raise AdapterError(ClustOptFailure.POLICY_UNRESOLVED,
                                       "per-policy reranker without a policy")
                rid = policy["reranker_id"]
            src = "mlp" if rid.startswith("MLP") else "knn"
            k_mode = rid.split("_", 1)[1].lower()
            actual_k = _actual_k(top_k, trace)
            pos, sel_extra = select_by_reranker(
                trace, rid, self.artifacts, uvec, src, k_mode, actual_k)
            sel_extra["reranker_id"] = rid

        row = trace.trial_log.iloc[pos]
        labels = self._labels_for(trace, row, X_view)
        return MethodResult(
            labels=labels,
            selected_algorithm=str(row.get("algorithm")),
            best_configuration={"config_str": str(row.get("config_str"))},
            n_evaluations=trace.n_evaluations,
            extra={
                "method_id": self.method_id, "dataset_id": dataset_id,
                "view": view_id, "search_trace_id": trace.trace_id,
                "trace_identity": trace.trace_identity,
                "trace_fingerprint": trace.trace_fingerprint(),
                "search_seed": trace.search_seed,
                "final_selection_mechanism": spec["selector"],
                "selected_candidate_index": int(pos),
                "selected_metric_ids": list(trace.selected_metrics),
                "selected_metric_weights": dict(trace.selected_metric_weights),
                "policy": policy, "artifact_hashes": dict(self.artifacts.hashes),
                "final_labels_hash": partition_hash(labels)
                if labels is not None else None,
                "selection_detail": sel_extra,
                "runtime_sec_placeholder": None,
            })

    # ------------------------------------------------------------- helpers
    def context_frames(self, dataset_dir: Path, view_id: str):
        """One-row META and 120-dim utility frames for PP and the reranker.

        The utility context is always BOTH predictors' vectors, because that is
        what the frozen 347-dim reranker representation and the 373-dim PP input
        were built from -- even for an arm whose search used only one of them.
        """
        from models.ClustOpt_Policy_Predictor.data import feature_schema as FSCH
        from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (  # noqa: E501
            knn_predictor_provider as KP, loaders as LD,
            regressor_predictor as RP)

        key = (str(dataset_dir), view_id)
        cached = self._context_cache.get(key)
        if cached is not None:
            return cached

        meta_cols = list(FSCH.load_meta_feature_columns(self.repo))
        metric_names = list(FSCH.load_metric_names(self.repo))
        meta_row = pd.DataFrame(
            [LD.load_feature_vector(dataset_dir, view_id, meta_cols)],
            columns=meta_cols)

        mlp = RP.get_regressor_predictor(str(self.repo / MLP_RUN_DIR),
                                         checkpoint_policy="fold_ensemble")
        mfeat = LD.load_feature_vector(dataset_dir, view_id, mlp.feature_columns)
        mu = mlp.predict(mfeat)

        knn = KP.get_knn_provider(str(self.repo / KNN_MODEL_DIR))
        kfeat = LD.load_feature_vector(dataset_dir, view_id,
                                       list(knn.feature_columns))
        ku, _dbg = knn.predict(kfeat, view_id)

        mvec = np.array([float(mu.get(m, np.nan)) for m in metric_names])
        kvec = np.array([float(ku.get(m, np.nan)) for m in metric_names])
        uvec = np.concatenate([mvec, kvec])
        util_row = pd.DataFrame(
            [uvec], columns=FSCH.oof_feature_columns(metric_names))
        out = (meta_row, util_row, uvec)
        self._context_cache[key] = out
        return out

    @staticmethod
    def _labels_for(trace: ClustOptTrace, row, X_view: np.ndarray):
        """Re-instantiate the selected candidate to recover its partition."""
        from models.Clustering_Repository_Builder.experiments.experiment_execution import (  # noqa: E501
            clustopt_execution as CE)
        space = trace.extra.get("search_space_object")
        if space is None:
            raise AdapterError(ClustOptFailure.NO_VALID_CANDIDATE,
                               "trace carries no search space to re-instantiate")
        labels = CE._predict_best_labels(space, row, X_view)
        if labels is None:
            raise AdapterError(ClustOptFailure.NO_VALID_CANDIDATE,
                               "selected candidate could not be re-fitted")
        return labels


def _actual_k(top_k: Any, trace: ClustOptTrace) -> int:
    """The K regime value the reranker representation records."""
    if isinstance(top_k, int):
        return int(top_k)
    n = len(trace.selected_metrics)
    return int(n) if n else 0


def view_matrix(X: np.ndarray, view_id: str) -> np.ndarray:
    """x_only / y_only / xy_2d projections of a 2-D point cloud."""
    A = np.asarray(X, dtype=float)
    if view_id == "xy_2d":
        return A
    if view_id == "x_only":
        return A[:, [0]]
    if view_id == "y_only":
        return A[:, [1]]
    raise AdapterError(ClustOptFailure.FEATURE_SCHEMA_MISMATCH,
                       "unknown view %r" % view_id)


def policy_table() -> List[Dict[str, Any]]:
    """The frozen 20-row policy resolution table, in target order."""
    from models.Clustering_Repository_Builder.experiments.experiment_execution.oof_policy_replay import (  # noqa: E501
        build_policy_set)
    from models.ClustOpt_Candidate_Reranker.training.final_models import (
        policy_mapping as PM)
    rows = []
    for i, p in enumerate(build_policy_set()):
        km = PM._k_mode(p.top_k)
        rows.append({"policy_index": i, "policy_id": p.policy_id,
                     "short_label": p.short_label, "source": p.source,
                     "top_k": p.top_k, "weighting": p.weighting,
                     "k_policy": p.k_policy,
                     "softmax_temperature": p.softmax_temperature,
                     "k_mode": km,
                     "reranker_id": PM.reranker_id(p.source, km)})
    return rows


def build_adapters(**kw) -> Dict[str, "ClustOptAdapter"]:
    """All six arms, sharing one artifact resolution and one trace cache."""
    return {mid: ClustOptAdapter(mid, **kw) for mid in ARM_SPEC}
