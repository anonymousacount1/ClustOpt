"""Fit one (information condition x outer fold) and select on the outer test.

Order of operations is the guarantee: fit on training rows, predict test regret,
argmin over ELIGIBLE candidates, freeze the chosen candidate, and only then read
its ARI. No outer-test value reaches preprocessing, fitting or selection.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import evaluate_fold as EV
from . import feature_builder as FB
from . import model_registry as MR


def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                *, return_model: bool = False
                ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Frozen HGBR. No scaling: the model is histogram-based and NaN-native.

    ``return_model`` is additive and defaults to False, so every historical
    caller is byte-for-byte unaffected. Stage 2C-4P sets it to persist the
    estimator that this path already fits, rather than reimplementing the fit.
    """
    model = MR.build_model()
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = np.asarray(model.predict(X_te), dtype=np.float64)
    info: Dict[str, Any] = {
        "fit_sec": fit_sec, "predict_sec": time.perf_counter() - t1,
        "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "n_iter_": int(getattr(model, "n_iter_", MR.HGBR_CONFIG["max_iter"])),
    }
    if return_model:
        info["model"] = model
    return pred, info


def run_unit(*, condition: Dict[str, Any], X_tr, y_tr, X_te,
             eligible_te: np.ndarray, slate_starts: np.ndarray,
             slate_sizes: np.ndarray, ari_te: np.ndarray,
             oracle_te: np.ndarray) -> Dict[str, Any]:
    """One execution unit: fit, predict, select, then read ARI."""
    pred, info = fit_predict(X_tr, y_tr, X_te)

    # --- selection from PREDICTIONS ONLY -----------------------------------
    chosen = EV.select_per_slate(pred, eligible_te, slate_starts, slate_sizes,
                                 minimise=True)
    if not eligible_te[chosen].all():
        raise AssertionError("an ineligible candidate was selected")

    # --- only now is ARI read ----------------------------------------------
    sel_ari = ari_te[chosen]
    return {"pred": pred, "chosen": chosen, "selected_ari": sel_ari,
            "slate_oracle": oracle_te, "fit_info": info}


def target_regret(ari: np.ndarray, oracle_per_row: np.ndarray) -> np.ndarray:
    """T2 slate regret, the single frozen Stage-2B1 target.

    Defined on ELIGIBLE candidates only. The slate optimum is the max over
    eligible candidates, so an INELIGIBLE candidate can carry negative regret --
    on the 33 exception slates a degenerate invalid partition scores ARI 0 while
    every eligible candidate is slightly negative. Ineligible rows are therefore
    excluded from training rather than clipped: the model never scores them at
    selection time, so training on them would spend capacity on candidates the
    system can never return.
    """
    r = (oracle_per_row - ari).astype(np.float32)
    if np.nanmin(r) < -1e-6:
        raise AssertionError(
            "negative slate regret in the training target -- ineligible "
            "candidates must be filtered out before calling this")
    return r
