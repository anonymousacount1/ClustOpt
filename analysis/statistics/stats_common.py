"""Shared statistics for the analysis scripts in this folder.

Bootstrap intervals, Wilcoxon signed-rank tests, rank-biserial effects and win/tie/loss counts,
with the same conventions as the external benchmark's result tables (10,000 bootstrap draws,
base seed 1234). Pure aggregation over released records; nothing is re-run.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

N_BOOT = 10000
BASE_SEED = 1234
TIE_EPS = 1e-3          # the audit brief's tie rule: |delta| < 0.001


def group_bootstrap_ci(values: np.ndarray, groups: np.ndarray, *,
                       n_boot: int = N_BOOT, seed: int = BASE_SEED,
                       stat=np.mean) -> Dict[str, Optional[float]]:
    """Percentile bootstrap that resamples DEPENDENCE GROUPS, not rows."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    values, groups = values[ok], np.asarray(groups)[ok]
    if values.size == 0:
        return {"lo": None, "hi": None, "n": 0, "n_groups": 0}
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_group[g] for g in pick])
        out[b] = stat(values[sel])
    return {"lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)),
            "n": int(values.size), "n_groups": int(uniq.size)}


def row_bootstrap_ci(values: np.ndarray, *, n_boot: int = N_BOOT,
                     seed: int = BASE_SEED, stat=np.mean) -> Dict[str, Optional[float]]:
    """Percentile bootstrap over rows (the controlled split has no dependence
    groups: every synthetic dataset is an independent draw from the generator)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    n = values.size
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        out[b] = stat(values[rng.integers(0, n, size=n)])
    return {"lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)), "n": int(n)}


def wilcoxon_paired(delta: np.ndarray) -> Dict[str, Any]:
    """Two-sided Wilcoxon signed-rank + matched-pairs rank-biserial.

    NOTE on counting: wins/ties/losses here use EXACT zero, matching the frozen
    Stage-4 package. The audit brief additionally asks for a |delta| < 0.001 tie
    rule; that is reported separately by `wtl_eps`.
    """
    from scipy.stats import wilcoxon
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    res: Dict[str, Any] = {"n_pairs": int(d.size), "n_nonzero": int(nz.size),
                           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                           "ties": int((d == 0).sum())}
    if nz.size < 1:
        res.update({"statistic": None, "p_value": None,
                    "rank_biserial": None, "valid": False,
                    "note": "no non-zero differences; Wilcoxon not valid"})
        return res
    try:
        st = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
        ranks = pd.Series(np.abs(nz)).rank().to_numpy()
        wp = float(ranks[nz > 0].sum())
        wm = float(ranks[nz < 0].sum())
        rb = (wp - wm) / (wp + wm) if (wp + wm) > 0 else None
        res.update({"statistic": float(st.statistic), "p_value": float(st.pvalue),
                    "rank_biserial": rb, "valid": True})
    except Exception as exc:  # noqa: BLE001
        res.update({"statistic": None, "p_value": None, "rank_biserial": None,
                    "valid": False, "note": str(exc)[:120]})
    return res


def wtl_eps(delta: np.ndarray, eps: float = TIE_EPS) -> Dict[str, int]:
    """Win/tie/loss under the audit brief's |delta| < eps tie rule."""
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    return {"wins_eps": int((d >= eps).sum()),
            "ties_eps": int((np.abs(d) < eps).sum()),
            "losses_eps": int((d <= -eps).sum())}


def paired_block(a: np.ndarray, b: np.ndarray, *, groups: Optional[np.ndarray] = None,
                 seed: int = BASE_SEED) -> Dict[str, Any]:
    """Full paired contrast a - b at machine precision."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    out: Dict[str, Any] = {
        "n": int(d.size),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_delta": float(np.mean(d)),
        "median_delta": float(np.median(d)),
    }
    if groups is not None:
        ci = group_bootstrap_ci(d, groups, seed=seed)
        out["ci_lo"], out["ci_hi"] = ci["lo"], ci["hi"]
        out["n_dependence_groups"] = ci["n_groups"]
        out["bootstrap_unit"] = "dependence_group"
    else:
        ci = row_bootstrap_ci(d, seed=seed)
        out["ci_lo"], out["ci_hi"] = ci["lo"], ci["hi"]
        out["n_dependence_groups"] = None
        out["bootstrap_unit"] = "dataset_row"
    out.update(wilcoxon_paired(d))
    out.update(wtl_eps(d))
    return out
