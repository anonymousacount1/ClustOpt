"""ML2DAC and AutoClust meta-feature extraction per view record.

Both use the **original ML2DAC source**
(``external/ml2dac/src/MetaLearning/MetaFeatureExtractor.py``):

* **ML2DAC** meta-features = pymfe ``["statistical", "info-theory", "general"]``
  (mf_set index 4, the paper's strongest set, matching the shipped classifier).
  Requires ``pymfe``.
* **AutoClust** meta-features = landmarking via MeanShift + the 7 internal CVIs
  (``extract_landmarking(X, "meanshift")``). Requires only the ML2DAC CVIHandler
  + scikit-learn, so it runs in the ClustOpt interpreter.

Graceful degradation: when a backend (e.g. pymfe) is missing, the extractor
returns ``status="unavailable"`` with the error and no feature values — the
build continues and later runs (e.g. under the ML2DAC env) can fill them in.
All meta-features are extracted on the view's decision-space ``X`` so every
view is an independent record.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# ML2DAC's strongest pymfe meta-feature set (matches the shipped MKR classifier).
ML2DAC_MF_SET = ["statistical", "info-theory", "general"]

# AutoClust landmarking uses MeanShift, which is O(n^2) and dominates cost on
# multi-thousand-point datasets. Cap its input with a deterministic subsample
# (fixed seed -> reproducible) to bound the meta-feature stage. Set to 0/None to
# disable subsampling and use the full view.
LANDMARK_MAX_SAMPLES = 1500
LANDMARK_SUBSAMPLE_SEED = 1234


def _maybe_subsample(X: np.ndarray, max_samples: Optional[int], seed: int) -> np.ndarray:
    X = np.asarray(X)
    if not max_samples or X.shape[0] <= max_samples:
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=int(max_samples), replace=False)
    idx.sort()
    return X[idx]


@dataclass
class MetaFeatureResult:
    status: str  # "success" | "unavailable" | "failed" | "skipped"
    runtime_sec: float = 0.0
    error: str = ""
    features: Dict[str, float] = field(default_factory=dict)


def _import_mfe(repo_root: Optional[str]):
    if repo_root is not None:
        from .io_utils import ensure_ml2dac_on_path

        ensure_ml2dac_on_path(repo_root)
    from MetaLearning import MetaFeatureExtractor  # type: ignore

    return MetaFeatureExtractor


def extract_ml2dac(
    X: np.ndarray, *, repo_root: Optional[str] = None, prefix: str = "ml2dac_"
) -> MetaFeatureResult:
    t0 = time.perf_counter()
    try:
        mfe = _import_mfe(repo_root)
    except Exception as exc:  # noqa: BLE001 - missing pymfe / source
        return MetaFeatureResult("unavailable", 0.0, f"{type(exc).__name__}: {exc}"[:300])
    try:
        names, values = mfe.extract_meta_features(np.asarray(X), ML2DAC_MF_SET)
        feats = {
            f"{prefix}{n}": float(v)
            for n, v in zip(names, np.asarray(values, dtype=float))
        }
        return MetaFeatureResult("success", time.perf_counter() - t0, "", feats)
    except Exception as exc:  # noqa: BLE001
        return MetaFeatureResult(
            "failed", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"[:300]
        )


def _import_cvi_collection(repo_root: Optional[str]):
    """Import ML2DAC's CVICollection directly (avoids pymfe at module top).

    ``MetaFeatureExtractor`` imports ``pymfe`` at module level, so it cannot be
    imported when pymfe is absent. The AutoClust landmarking logic itself only
    needs the CVIHandler, which imports cleanly, so we reuse that and replicate
    ``extract_landmarking`` verbatim (MeanShift + the 7 internal CVIs).
    """
    if repo_root is not None:
        from .io_utils import ensure_ml2dac_on_path

        ensure_ml2dac_on_path(repo_root)
    from ClusterValidityIndices.CVIHandler import CVICollection  # type: ignore

    return CVICollection


def extract_autoclust(
    X: np.ndarray,
    *,
    repo_root: Optional[str] = None,
    prefix: str = "autoclust_",
    max_samples: Optional[int] = LANDMARK_MAX_SAMPLES,
    seed: int = LANDMARK_SUBSAMPLE_SEED,
) -> MetaFeatureResult:
    t0 = time.perf_counter()
    try:
        from sklearn.cluster import MeanShift

        CVICollection = _import_cvi_collection(repo_root)
    except Exception as exc:  # noqa: BLE001
        return MetaFeatureResult("unavailable", 0.0, f"{type(exc).__name__}: {exc}"[:300])
    try:
        Xs = _maybe_subsample(X, max_samples, seed)
        # Verbatim ML2DAC landmarking: MeanShift labels scored by all 7 CVIs.
        labels = MeanShift().fit_predict(Xs)
        single_cluster = np.unique(labels).size == 1
        feats: Dict[str, float] = {}
        for metric in CVICollection.internal_cvis:
            name = metric.get_abbrev()
            if single_cluster:
                score = float("nan")
            else:
                try:
                    score = float(metric.score_cvi(Xs, labels))
                except Exception:  # noqa: BLE001
                    score = float("nan")
            feats[f"{prefix}{name}"] = 0.0 if not np.isfinite(score) else score
        return MetaFeatureResult("success", time.perf_counter() - t0, "", feats)
    except Exception as exc:  # noqa: BLE001
        return MetaFeatureResult(
            "failed", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"[:300]
        )
