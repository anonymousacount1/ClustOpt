"""Stage 3B step 7: what the AutoClust ARI predictor relies on.

AutoClust never names a metric, so there is no such thing as an AutoClust
"selection frequency" and none is invented here. What can be measured is how
much its already-trained ARI-predictor MLP *depends* on each metric group, by
permuting that group's columns and watching prediction quality fall.

Evaluation population. The A3 predictor was trained on Splits 2-16; **Split 1
was never seen by it**. The master MKR nevertheless stores the full CVI vector
and the realised ARI for every Split-1 (record, configuration) pair, so Split 1
gives a genuinely held-out, dataset-disjoint evaluation population of 101,280
rows. That is strictly better than reconstructing scikit-learn's internal
``validation_fraction`` split, which is row-level and would put configurations
of the same dataset on both sides.

What this is and is not. Permutation reliance is a statement about the *model's*
dependence on an input, i.e. predictive reliance. It is NOT a causal claim about
downstream ARI: the causal evidence is the A0/A1/A2/A3 bundle ablation, which
already exists. Because CVIs are strongly correlated, a low individual score can
mean "redundant", not "useless" -- a metric whose information is available
elsewhere can be permuted with little loss while still being informative.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import common as C

DERIVED = "experiments/external_baselines/Unified_MKR/derived/autoclust_split1/extended_cvis"
N_REPEATS = 5


def load_model(repo: Path) -> Tuple[Any, List[str], Dict[str, Any]]:
    import joblib
    d = repo / DERIVED
    meta = json.loads((d / "ari_predictor_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(d / "ari_predictor.pkl")
    return model, list(meta["feature_columns"]), meta


def heldout_population(repo: Path, feats: List[str], log) -> pd.DataFrame:
    """Split-1 (record, config) rows with the full CVI vector and realised ARI."""
    M = repo / C.MKR_REL / "master"
    rec = pd.read_parquet(M / "records.parquet",
                          columns=["record_id", "dataset_id", "split_id", "family",
                                   "subfamily", "view_type"])
    ids = set(rec[rec.split_id == 1]["record_id"])
    ev = pd.read_parquet(M / "evaluations.parquet",
                         columns=["record_id", "config_id", "ari", "valid_result"] + feats)
    ev = ev[ev["record_id"].isin(ids)].copy()
    ev = ev[np.isfinite(pd.to_numeric(ev["ari"], errors="coerce"))]
    ev = ev.merge(rec[["record_id", "dataset_id", "family", "subfamily"]],
                  on="record_id", how="left")
    log("AutoClust held-out population: %d Split-1 rows over %d datasets "
        "(never seen in training)" % (len(ev), ev.dataset_id.nunique()))
    return ev


def _predict(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def _score(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    resid = y - p
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(resid ** 2))),
            "mae": float(np.mean(np.abs(resid))),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "spearman": float(pd.Series(p).corr(pd.Series(y), method="spearman"))}


def reliance(repo: Path, A: pd.DataFrame, log, max_rows: int = 0
             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    model, feats, meta = load_model(repo)
    ev = heldout_population(repo, feats, log)
    if max_rows and len(ev) > max_rows:
        # deterministic, frozen BEFORE any reliance value is inspected
        ev = ev.sample(n=max_rows, random_state=C.PERM_SEED).reset_index(drop=True)
        log("subsampled to %d rows (seed %d, frozen before inspection)"
            % (len(ev), C.PERM_SEED))

    X = ev[feats].to_numpy(float)
    y = ev["ari"].to_numpy(float)
    base_pred = _predict(model, X)
    base = _score(y, base_pred)
    log("baseline on held-out Split 1: rmse=%.5f r2=%.4f spearman=%.4f"
        % (base["rmse"], base["r2"], base["spearman"]))

    grp = A.set_index("metric_id")["group"].to_dict()
    sub = A.set_index("metric_id")["subgroup"].to_dict()
    groups: Dict[str, List[int]] = {}
    for i, f in enumerate(feats):
        g = grp.get(f, "Unknown")
        key = sub.get(f) or g
        groups.setdefault(key if key else g, []).append(i)
        if g == "New46":
            groups.setdefault("New46", []).append(i)
        groups.setdefault("ALL_%s" % g, []).append(i)

    rng = np.random.default_rng(C.PERM_SEED)

    def perm_score(idx: List[int]) -> Dict[str, float]:
        accs = []
        for _ in range(N_REPEATS):
            Xp = X.copy()
            order = rng.permutation(Xp.shape[0])
            Xp[:, idx] = Xp[order][:, idx]
            accs.append(_score(y, _predict(model, Xp)))
        return {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}

    grows = []
    for key in ("ALL_Original", "ALL_Established", "New46", "New46-Pattern",
                "New46-Image"):
        idx = sorted(set(groups.get(key, [])))
        if not idx:
            continue
        s = perm_score(idx)
        grows.append({
            "group": key.replace("ALL_", ""), "n_features_permuted": len(idx),
            "baseline_rmse": base["rmse"], "permuted_rmse": s["rmse"],
            "rmse_increase": s["rmse"] - base["rmse"],
            "rmse_increase_pct": 100.0 * (s["rmse"] - base["rmse"]) / base["rmse"],
            "baseline_r2": base["r2"], "permuted_r2": s["r2"],
            "r2_drop": base["r2"] - s["r2"],
            "baseline_spearman": base["spearman"], "permuted_spearman": s["spearman"],
            "spearman_drop": base["spearman"] - s["spearman"],
            "evidence_level": "2 (model reliance, NOT causal ARI contribution)",
        })
    Gp = pd.DataFrame(grows)

    irows = []
    for i, f in enumerate(feats):
        s = perm_score([i])
        irows.append({
            "metric_id": f, "group": grp.get(f, "Unknown"), "subgroup": sub.get(f, ""),
            "baseline_rmse": base["rmse"], "permuted_rmse": s["rmse"],
            "rmse_increase": s["rmse"] - base["rmse"],
            "rmse_increase_pct": 100.0 * (s["rmse"] - base["rmse"]) / base["rmse"],
            "r2_drop": base["r2"] - s["r2"],
            "spearman_drop": base["spearman"] - s["spearman"],
            "evidence_level": "2 (model reliance, NOT causal ARI contribution)",
        })
    Ip = pd.DataFrame(irows).sort_values("rmse_increase", ascending=False).reset_index(drop=True)
    Ip["reliance_rank"] = np.arange(1, len(Ip) + 1)

    frows = []
    for fam, gsel in ev.groupby("family"):
        m = ev["family"].to_numpy() == fam
        Xf, yf = X[m], y[m]
        bf = _score(yf, _predict(model, Xf))
        for key in ("ALL_Original", "ALL_Established", "New46-Pattern", "New46-Image"):
            idx = sorted(set(groups.get(key, [])))
            if not idx:
                continue
            accs = []
            for _ in range(2):
                Xp = Xf.copy()
                order = rng.permutation(Xp.shape[0])
                Xp[:, idx] = Xp[order][:, idx]
                accs.append(_score(yf, _predict(model, Xp)))
            s = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
            frows.append({
                "family": fam, "group": key.replace("ALL_", ""),
                "n_rows": int(m.sum()), "baseline_rmse": bf["rmse"],
                "permuted_rmse": s["rmse"],
                "rmse_increase": s["rmse"] - bf["rmse"],
                "rmse_increase_pct": 100.0 * (s["rmse"] - bf["rmse"]) / bf["rmse"],
            })
    Fp = pd.DataFrame(frows)

    prov = {
        "model": str(repo / DERIVED / "ari_predictor.pkl"),
        "model_retrained": False, "model_tuned": False,
        "architecture": meta.get("architecture"),
        "trained_on_splits": list(range(2, 17)),
        "evaluation_population": {
            "source": "master evaluations.parquet, split_id == 1",
            "never_seen_in_training": True,
            "dataset_disjoint_from_training": True,
            "n_rows": int(len(ev)), "n_datasets": int(ev.dataset_id.nunique()),
            "rationale": ("Split 1 is a genuinely held-out, dataset-grouped "
                          "population; sklearn's internal validation_fraction "
                          "split is row-level and would leak configurations of the "
                          "same dataset across the boundary"),
        },
        "permutation": {"n_repeats": N_REPEATS, "seed": C.PERM_SEED,
                        "family_level_repeats": 2},
        "baseline_scores": base,
        "interpretation": (
            "Predictive reliance of the already-trained ARI predictor. NOT a "
            "causal single-metric ARI contribution. CVIs are correlated, so a low "
            "individual score may indicate redundancy rather than uselessness."),
    }
    for _, r in Gp.iterrows():
        log("reliance %-14s +%.5f RMSE (%.2f%%), r2 drop %.4f"
            % (r["group"], r["rmse_increase"], r["rmse_increase_pct"], r["r2_drop"]))
    return Gp, Ip, Fp, prov
