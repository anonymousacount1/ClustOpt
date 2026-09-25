"""Frozen model families M1/M2/M3. No hyperparameter search anywhere.

Every configuration below is fixed before any outer-test value is observed, and
none is tuned on outer-test performance.

Documented deviation (M2): the brief recommends ``min_samples_leaf = 2``. A probe
on outer fold 02 (44,907 x 370, 20 outputs) measured a **3.4 GB** forest at 400
trees with that setting, against 6.1 GB free RAM on this machine. ``min_samples_leaf
= 8`` gives 742 MB with a ~40 s fit and is used instead. This is a resource
constraint, decided from a memory probe and not from any outer-test result.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np

MODEL_FAMILIES: Tuple[str, ...] = ("RIDGE", "EXTRATREES", "MLP")
GLOBAL_SEED = 42

RIDGE_CONFIG: Dict[str, Any] = {
    "alpha": 1.0, "fit_intercept": True, "requires_scaling": True,
}
EXTRATREES_CONFIG: Dict[str, Any] = {
    "n_estimators": 400,
    "min_samples_leaf": 8,          # documented deviation from 2 -- see module docstring
    "min_samples_leaf_recommended": 2,
    "deviation_reason": ("400 trees x min_samples_leaf=2 on 44.9k x 370 with 20 "
                         "multi-outputs measured 3.4 GB per forest; only 6.1 GB "
                         "RAM free. msl=8 gives 742 MB at ~40 s fit."),
    "max_features": "sqrt", "random_state": GLOBAL_SEED,
    "requires_scaling": False,
}
MLP_CONFIG: Dict[str, Any] = {
    "hidden": (256, 128, 64), "activation": "GELU", "norm": "LayerNorm",
    "dropout": (0.10, 0.10, 0.0), "output_activation": "linear",
    "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
    "batch_size": 512, "max_epochs": 200, "early_stopping_patience": 15,
    "loss": "SmoothL1", "seed": GLOBAL_SEED,
    "val_fraction": 0.10, "val_grouping": "dataset_id",
    "requires_scaling": True,
}


def safe_n_jobs() -> int:
    """Conservative CPU cap: 4 physical cores on this machine."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def build_ridge():
    from sklearn.linear_model import Ridge
    return Ridge(alpha=RIDGE_CONFIG["alpha"],
                 fit_intercept=RIDGE_CONFIG["fit_intercept"])


def build_extratrees():
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(
        n_estimators=EXTRATREES_CONFIG["n_estimators"],
        min_samples_leaf=EXTRATREES_CONFIG["min_samples_leaf"],
        max_features=EXTRATREES_CONFIG["max_features"],
        random_state=EXTRATREES_CONFIG["random_state"],
        n_jobs=safe_n_jobs())


class PolicyMLP:
    """Modest multi-output MLP with a LINEAR output head.

    Linear rather than sigmoid because ARI can be negative, centered targets are
    signed, and regret selection only needs relative values.
    """

    def __init__(self, in_dim: int, out_dim: int = 20) -> None:
        import torch
        import torch.nn as nn
        torch.manual_seed(GLOBAL_SEED)
        h1, h2, h3 = MLP_CONFIG["hidden"]
        d1, d2, _ = MLP_CONFIG["dropout"]
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.GELU(), nn.LayerNorm(h1), nn.Dropout(d1),
            nn.Linear(h1, h2), nn.GELU(), nn.LayerNorm(h2), nn.Dropout(d2),
            nn.Linear(h2, h3), nn.GELU(), nn.LayerNorm(h3),
            nn.Linear(h3, out_dim),
        )
        self.in_dim, self.out_dim = in_dim, out_dim

    def fit(self, X, y, groups) -> Dict[str, Any]:
        """Train with early stopping on a group-safe split of TRAINING rows only."""
        import torch
        from sklearn.model_selection import GroupShuffleSplit
        from torch.utils.data import DataLoader, TensorDataset
        gss = GroupShuffleSplit(n_splits=1, test_size=MLP_CONFIG["val_fraction"],
                                random_state=GLOBAL_SEED)
        tr_i, va_i = next(gss.split(X, y, groups=groups))
        dev = torch.device("cpu")
        Xt = torch.tensor(X[tr_i], dtype=torch.float32)
        yt = torch.tensor(y[tr_i], dtype=torch.float32)
        Xv = torch.tensor(X[va_i], dtype=torch.float32)
        yv = torch.tensor(y[va_i], dtype=torch.float32)
        dl = DataLoader(TensorDataset(Xt, yt),
                        batch_size=MLP_CONFIG["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(GLOBAL_SEED))
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=MLP_CONFIG["learning_rate"],
                                weight_decay=MLP_CONFIG["weight_decay"])
        lossf = torch.nn.SmoothL1Loss()
        best, best_state, bad, ep = float("inf"), None, 0, 0
        for ep in range(1, MLP_CONFIG["max_epochs"] + 1):
            self.net.train()
            for xb, yb in dl:
                opt.zero_grad()
                lossf(self.net(xb), yb).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(lossf(self.net(Xv), yv))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= MLP_CONFIG["early_stopping_patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {"epochs_run": ep, "best_val_loss": best,
                "n_train": int(len(tr_i)), "n_val": int(len(va_i))}

    def predict(self, X) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy()


def config_of(family: str) -> Dict[str, Any]:
    return {"RIDGE": RIDGE_CONFIG, "EXTRATREES": EXTRATREES_CONFIG,
            "MLP": MLP_CONFIG}[family]
