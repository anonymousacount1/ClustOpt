"""Stage 3B shared foundations: paths, canonical partitions, statistics.

ANALYSIS ONLY. Nothing in this package fits, tunes, or rebuilds anything. Every
number is derived from artefacts that already existed before Stage 3B started.

Two conventions matter enough to state once, here, and never vary:

* **The dataset is the inferential unit.** The repository stores three views per
  dataset and those views are strongly correlated, so treating them as three
  observations would inflate n threefold. Every confidence interval and test in
  this package resamples *datasets*; view-level numbers are reported separately
  and only ever descriptively.
* **The utility definition is not ours to choose.** It is read from the frozen
  ClustOpt implementation (``compute_metric_utility.compute_metric_utilities``)
  and recorded verbatim in ``metric_utility_definition.json``.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_MKR_PKG = REPO / "experiments" / "external_baselines" / "Unified_MKR"
if str(_MKR_PKG) not in sys.path:
    sys.path.insert(0, str(_MKR_PKG))

OUT_REL = "results_analysis/clustering_repository/metric_analysis_existing_domain"
MKR_REL = "experiments/external_baselines/Unified_MKR"
OOF_REL = ("results_analysis/clustering_repository/stage2/oof_utility_splits2_16/"
           "combined/oof_mlp_utilities.csv.gz")
OOF_KNN_REL = ("results_analysis/clustering_repository/stage2/oof_utility_splits2_16/"
               "combined/oof_knn_utilities.csv.gz")
STAGE1_REL = "results_analysis/clustering_repository/stage1_2x3_aggregation"
EXT_REL = "results_analysis/clustering_repository/external_baselines"
ANALYZED_REL = "results_analysis/clustering_repository/analyzed_data"

#: frozen ClustOpt deployment semantics, taken from the Stage-1 arm identifier
#: ``stage1_full60_mlp_top5_raw_fixed50`` -- Top-5 by predicted utility, RAW
#: (normalised-positive) weighting. This package never invents a selection rule.
CLUSTOPT_TOP_K = 5
CLUSTOPT_WEIGHTING = "normalized_positive"
CLUSTOPT_ARM_ID = "stage1_full60_mlp_top5_raw_fixed50"

BOOT_B = 1000
BOOT_SEED = 20260909
PERM_SEED = 1234
VIEWS = ("x_only", "y_only", "xy_2d")


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def out_dir(repo: Path = REPO) -> Path:
    d = repo / OUT_REL
    d.mkdir(parents=True, exist_ok=True)
    return d


def ext(p: Path) -> str:
    """Windows long-path-safe string form."""
    s = str(p)
    if len(s) > 240 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def write_csv(df: pd.DataFrame, path: Path, gz: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        df.to_csv(ext(path), index=False, compression="gzip")
    else:
        df.to_csv(ext(path), index=False)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(ext(path)).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Optional[dict]:
    p = Path(ext(path))
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ------------------------------------------------------------------ partitions
def partitions(repo: Path = REPO) -> Dict[str, List[str]]:
    """Canonical ORIGINAL / ESTABLISHED / NEW46 sets, from the frozen resolver."""
    from unified_mkr import metric_partitions as MP  # noqa: E402
    reg = pd.read_parquet(repo / MKR_REL / "master" / "metric_registry.parquet")
    p = MP.resolve_partitions(repo, reg)
    return {
        "original": sorted(p["original"]),
        "established": sorted(p["established"]),
        "new46": sorted(p["new46"]),
        "dead": sorted(MP.EXPECTED_DEAD),
    }


def implementation_subgroups(repo: Path = REPO) -> Dict[str, str]:
    """metric -> implementation subpackage (core_cvi/structure/pattern/image).

    This is the *implementation's own* classification, read from the ClustOpt
    metric package layout, not a grouping invented for this analysis. It is what
    gives New46 its Pattern vs Image split.
    """
    base = repo / "models/ClustOpt/cluster_validity_indices/metrics"
    out: Dict[str, str] = {}
    for sub in ("core_cvi", "structure_base", "pattern_base", "image_base"):
        d = base / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.py")):
            if f.stem == "__init__":
                continue
            out[f.stem] = sub
    return out


def head_groups() -> Dict[str, str]:
    """metric -> multi-head MLP head group (classic_cvi, geometry_shape, ...)."""
    from models.metric_utility_mlp.head_groups import METRIC_HEAD_GROUPS  # noqa: E402
    return {m: g for g, ms in METRIC_HEAD_GROUPS.items() for m in ms}


#: module-name -> emitted metric name, where the implementation file and the
#: metric identifier differ. Derived by inspection of the frozen registry; kept
#: explicit so the Pattern/Image assignment is auditable rather than fuzzy.
MODULE_TO_METRIC = {
    "peak_to_background_line_ratio": "peak_to_background_hough",
    "Density_Based_Clustering_Validation_Index": "dbcv",
    "Noise_Aware_Silhouette": "noise_aware_silhouette",
    "S_Dbw_Index": "s_dbw",
    "Silhouette": "silhouette",
    "Average_cluster_compactness": "avg_pca_isotropy",
    "Entropy_cluster_sizes": "cluster_size_balance_entropy",
    "Local_Neighborhood_Purity": "neighborhood_purity",
}


def subgroup_for(metric: str, impl: Dict[str, str], group: str) -> str:
    """Pattern/Image subgroup for a New46 metric; '' for the other groups."""
    if group != "New46":
        return ""
    sub = impl.get(metric)
    if sub is None:
        for mod, met in MODULE_TO_METRIC.items():
            if met == metric:
                sub = impl.get(mod)
                break
    if sub == "pattern_base":
        return "New46-Pattern"
    if sub == "image_base":
        return "New46-Image"
    return "New46-Unclassified"


def group_for(metric: str, P: Dict[str, List[str]]) -> str:
    if metric in P["original"]:
        return "Original"
    if metric in P["established"]:
        return "Established"
    if metric in P["new46"]:
        return "New46"
    return "Unknown"


# ------------------------------------------------------------------ statistics
def dataset_level(df: pd.DataFrame, value_cols: Sequence[str],
                  key: str = "dataset_id") -> pd.DataFrame:
    """Collapse the three views of a dataset to one row (mean over views).

    Averaging is the neutral choice: it neither favours the view a metric
    happens to suit nor discards the other two, and it makes the dataset the
    unit of inference as required.
    """
    return df.groupby(key, sort=True)[list(value_cols)].mean().reset_index()


def boot_ci_matrix(M: np.ndarray, *, b: int = BOOT_B, seed: int = BOOT_SEED,
                   alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile CI of the column means, resampling ROWS (datasets).

    ``M`` is (n_datasets, n_metrics) and may contain NaN; column means ignore
    NaN, so a metric that is undefined on some datasets is summarised over the
    datasets where it is defined.
    """
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    reps = np.empty((b, M.shape[1]), dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, n)
        with np.errstate(invalid="ignore"):
            reps[i] = np.nanmean(M[idx], axis=0)
    lo = np.nanpercentile(reps, 100 * alpha / 2.0, axis=0)
    hi = np.nanpercentile(reps, 100 * (1 - alpha / 2.0), axis=0)
    return lo, hi


def boot_ci_vector(x: np.ndarray, *, b: int = BOOT_B, seed: int = BOOT_SEED,
                   alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    reps = np.array([np.mean(rng.choice(x, x.size, replace=True)) for _ in range(b)])
    return (float(np.percentile(reps, 100 * alpha / 2.0)),
            float(np.percentile(reps, 100 * (1 - alpha / 2.0))))


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values (NaN-safe, order preserved)."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    q[ok] = out
    return q


def wilcoxon_p(d: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or np.all(np.abs(d) < 1e-12):
        return float("nan")
    try:
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


# ------------------------------------------------------------------- OOF table
def load_oof(repo: Path = REPO, knn: bool = False) -> pd.DataFrame:
    """The leakage-free OOF utility table for Splits 2-16.

    Columns ``true_<metric>`` are the canonical ClustOpt utilities; ``pred_<m>``
    are the cross-fitted out-of-fold predictions of the deployed utility model.
    One row per (dataset, view); 48,039 rows over 16,013 datasets.
    """
    f = repo / (OOF_KNN_REL if knn else OOF_REL)
    return pd.read_csv(ext(f))


def metric_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    return sorted(c for c in df.columns if c.startswith(prefix))


def strip(cols: Iterable[str], prefix: str) -> List[str]:
    return [c[len(prefix):] for c in cols]
