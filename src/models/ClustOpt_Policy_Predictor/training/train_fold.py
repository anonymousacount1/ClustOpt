"""Fit one (configuration x outer fold) and evaluate it under test-target discipline.

Discipline enforced structurally: the model produces its 20 outputs, the policy is
selected by argmax/argmin on those outputs alone, and only then is the outer-test
ARI of the selected policy retrieved. No outer-test value touches preprocessing,
fitting, early stopping or selection.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import feature_sets as FSET
from . import model_registry as MR
from . import target_transforms as TT


def config_id(feature_set: str, target: str, view_arch: str,
              family: str) -> str:
    return f"{family}__{view_arch}__{feature_set}__{target}"


def config_hash(feature_set: str, target: str, view_arch: str,
                family: str) -> str:
    payload = {"feature_set": feature_set, "target": target,
               "view_arch": view_arch, "family": family,
               "model_config": MR.config_of(family), "seed": MR.GLOBAL_SEED}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str,
                   separators=(",", ":")).encode()).hexdigest()[:16]


def _fit_predict(Xtr, ytr, Xte, n_cont: int, family: str,
                 groups) -> Tuple[np.ndarray, Dict[str, Any], Any]:
    """Fit on training rows only; return test predictions + fit info + model."""
    info: Dict[str, Any] = {}
    scaler = None
    if MR.config_of(family)["requires_scaling"]:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr[:, :n_cont])       # TRAIN ONLY
        Xtr = np.hstack([scaler.transform(Xtr[:, :n_cont]), Xtr[:, n_cont:]])
        Xte = np.hstack([scaler.transform(Xte[:, :n_cont]), Xte[:, n_cont:]])
        info["scaler"] = "StandardScaler on continuous block, fit on train only"

    t0 = time.perf_counter()
    if family == "RIDGE":
        model = MR.build_ridge().fit(Xtr, ytr)
    elif family == "EXTRATREES":
        model = MR.build_extratrees().fit(Xtr, ytr)
    elif family == "MLP":
        model = MR.PolicyMLP(in_dim=Xtr.shape[1], out_dim=ytr.shape[1])
        info.update(model.fit(Xtr, ytr, groups))
    else:
        raise ValueError(family)
    info["fit_sec"] = time.perf_counter() - t0
    pred = np.asarray(model.predict(Xte), dtype=np.float64)
    return pred, info, (model, scaler)


def run_fold(root: Path, s: int, feature_set: str, target: str,
             view_arch: str, family: str,
             fold: Optional[Dict[str, pd.DataFrame]] = None
             ) -> Dict[str, Any]:
    """Train + evaluate one configuration on outer fold ``s``."""
    F = fold if fold is not None else FSET.load_fold(root, s)
    tr_ident, te_ident = F["train_ident"], F["test_ident"]
    ari_tr = F["train_ari"].to_numpy(np.float32)
    ari_te = F["test_ari"].to_numpy(np.float64)
    y_tr = TT.transform(ari_tr, target)

    t0 = time.perf_counter()
    pred = np.empty((len(te_ident), 20), dtype=np.float64)
    infos: List[Dict[str, Any]] = []

    if view_arch == "UNIFIED":
        Xtr, n_cont, _ = FSET.build_X(F["train_meta"], F["train_util"],
                                      feature_set, tr_ident["view_id"], view_arch)
        Xte, _, _ = FSET.build_X(F["test_meta"], F["test_util"], feature_set,
                                 te_ident["view_id"], view_arch)
        p, info, _ = _fit_predict(Xtr, y_tr, Xte, n_cont, family,
                                  tr_ident["dataset_id"].to_numpy())
        pred[:] = p
        infos.append({"view": "ALL", **info})
    else:                                   # SEPARATE: one model per view
        Xtr_all, n_cont, _ = FSET.build_X(F["train_meta"], F["train_util"],
                                          feature_set, tr_ident["view_id"],
                                          view_arch)
        Xte_all, _, _ = FSET.build_X(F["test_meta"], F["test_util"], feature_set,
                                     te_ident["view_id"], view_arch)
        for v in FSET.VIEW_IDS:
            mtr = (tr_ident["view_id"] == v).to_numpy()
            mte = (te_ident["view_id"] == v).to_numpy()
            p, info, _ = _fit_predict(Xtr_all[mtr], y_tr[mtr], Xte_all[mte],
                                      n_cont, family,
                                      tr_ident.loc[mtr, "dataset_id"].to_numpy())
            pred[mte] = p
            infos.append({"view": v, **info})

    # ---- policy selected from PREDICTIONS ONLY; ARI retrieved only after -----
    sel = TT.select_policy(pred, target)
    sel_ari = ari_te[np.arange(len(sel)), sel]
    view_oracle = ari_te.max(axis=1)
    out = pd.DataFrame({
        "dataset_id": te_ident["dataset_id"].to_numpy(),
        "split_id": te_ident["split_id"].to_numpy() if "split_id" in te_ident
        else s,
        "view_id": te_ident["view_id"].to_numpy(),
        "selected_policy_index": sel,
        "selected_ari": sel_ari,
        "view_oracle_ari": view_oracle,
        "view_regret": view_oracle - sel_ari,
    })
    for j in range(20):
        out[f"pred_{j:02d}"] = pred[:, j]
    return {"predictions": out, "fit_info": infos,
            "runtime_sec": time.perf_counter() - t0,
            "n_train": int(len(tr_ident)), "n_test": int(len(te_ident)),
            "input_dim": int(FSET.expected_dim(feature_set, view_arch)),
            "output_dim": 20}
