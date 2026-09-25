"""Pair-specific replay of the frozen 20 policies over the persisted fixed-32 pools.

**No clustering.** For every pair-exclusion utility vector the objective ``J`` is
evaluated over the already-stored 32 candidates and the argmax is taken; the
persisted ARI is read only afterwards.

Work is pivoted by TARGET SPLIT rather than by pair. Every dataset/view in split
``t`` appears in 14 pair sources ({s,t} for each s != t), so loading its candidate
pool once and scoring all 14 x 20 policy evaluations against it reduces pool loads
from 672,546 to 48,039.

Scoring uses a vectorised form of the production ``score_candidates``:
``J = M_norm @ w`` over the ``*_norm`` columns. Equivalence with the production
function is asserted on a sample before the bulk run, not assumed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.dynamic_k import (
    select_dynamic_top_k,
)
from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.weighting import (
    compute_weights, select_top_k,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.offline_candidate_execution import (  # noqa: E501
    load_candidate_set, score_candidates,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E501
    _ext, ensure_dir, path_exists,
)

from ..data import target_schema as TS
from .pair_utility_builder import SPLITS, VIEWS, pair_dir

EXPECTED_CANDIDATES = 32


def policy_specs() -> List[Dict[str, Any]]:
    """The frozen 20 policies as (id, source, top_k, weighting) records."""
    out = []
    for m in TS.policy_manifest():
        out.append({
            "policy_id": m["policy_id"], "index": m["index"],
            "source": "mlp" if m["source_family"] == "MLP" else "knn",
            "top_k": ("dynamic" if m["dynamic"] else int(m["fixed_metric_count"])),
            "weighting": ("normalized_positive" if m["weighting"] == "RAW"
                          else "softmax"),
        })
    return out


def _candidate_matrix(dataset_dir: Path, view: str,
                      metric_names: Sequence[str]):
    """(frame, M_norm 32x60, ARI vector, usable mask). Loaded once per row."""
    cand = load_candidate_set(dataset_dir, view, expected_count=EXPECTED_CANDIDATES)
    frame = cand["frame"]
    cols = [f"{m}_norm" for m in metric_names]
    M = frame[cols].to_numpy(np.float64)
    ari = pd.to_numeric(frame["ARI"], errors="coerce").to_numpy(np.float64)
    usable = np.isfinite(M).all(axis=1)
    return frame, M, ari, usable, cand.get("fingerprint")


def _select(u: Dict[str, float], spec: Dict[str, Any]):
    if spec["top_k"] == "dynamic":
        sel, k, _ = select_dynamic_top_k(u)
    else:
        sel, k = select_top_k(u, int(spec["top_k"])), int(spec["top_k"])
    return sel, compute_weights(sel, spec["weighting"], 0.5), k


def verify_vectorised_scoring(dataset_dir: Path, view: str,
                              metric_names: Sequence[str],
                              utilities: Dict[str, float],
                              specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Assert J = M_norm @ w reproduces the production score_candidates exactly."""
    frame, M, ari, usable, _ = _candidate_matrix(dataset_dir, view, metric_names)
    order = {m: i for i, m in enumerate(metric_names)}
    n = ok = 0
    worst = 0.0
    for spec in specs:
        sel, w, _ = _select(utilities, spec)
        wv = np.zeros(len(metric_names))
        for name, val in w.items():
            wv[order[name]] = val
        Jv = M @ wv
        Jp = score_candidates(frame, w)
        for i, jp in enumerate(Jp):
            n += 1
            if jp is None:
                ok += int(not usable[i])
                continue
            d = abs(float(jp) - float(Jv[i]))
            worst = max(worst, d)
            ok += int(d <= 1e-12)
    return {"n_compared": n, "n_exact": ok, "max_abs_deviation": worst,
            "equivalent": ok == n}


def replay_rows(dataset_dir: Path, view: str, metric_names: Sequence[str],
                util_by_pair: Dict[Tuple[int, int], Dict[str, Dict[str, float]]],
                specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All (pair x 20 policy) outcomes for ONE dataset/view. Pool loaded once."""
    frame, M, ari, usable, fp = _candidate_matrix(dataset_dir, view, metric_names)
    order = {m: i for i, m in enumerate(metric_names)}
    neg = np.where(usable, 0.0, -np.inf)
    rows: List[Dict[str, Any]] = []
    for pair, by_source in util_by_pair.items():
        aris = np.empty(len(specs)); ks = np.empty(len(specs), dtype=int)
        for j, spec in enumerate(specs):
            u = by_source[spec["source"]]
            sel, w, k = _select(u, spec)
            wv = np.zeros(len(metric_names))
            for name, val in w.items():
                wv[order[name]] = val
            J = M @ wv + neg                     # unusable candidates -> -inf
            b = int(np.argmax(J))
            aris[j] = ari[b]                     # ARI read only after argmax
            ks[j] = k
        vbs = float(aris.max())
        best = np.isclose(aris, vbs, atol=1e-12)
        row: Dict[str, Any] = {
            "excluded_pair": f"{pair[0]:02d}_{pair[1]:02d}",
            "excluded_a": pair[0], "excluded_b": pair[1],
            "view_policy_vbs": vbs,
            "n_exact_winners": int(best.sum()),
            "winner_set": "|".join(specs[j]["policy_id"]
                                   for j in np.flatnonzero(best)),
            "candidate_fingerprint": fp,
        }
        for j, spec in enumerate(specs):
            row[f"{TS.ARI_PREFIX}{spec['policy_id']}"] = float(aris[j])
            row[f"{TS.REGRET_PREFIX}{spec['policy_id']}"] = float(vbs - aris[j])
        srt = np.sort(aris)
        row["top1_minus_top2_policy_score"] = float(srt[-1] - srt[-2])
        lower = aris[~best]
        row["top1_minus_second_distinct_score"] = (
            float(vbs - lower.max()) if lower.size else TS.ALL_TIED_SENTINEL)
        reg = vbs - aris
        row["n_within_0_001"] = int((reg < 0.001).sum())
        row["n_within_0_005"] = int((reg < 0.005).sum())
        row["n_within_0_01"] = int((reg < 0.01).sum())
        rows.append(row)
    return rows


def load_pair_utilities(root: Path, target_split: int, metric_names: Sequence[str]
                        ) -> Dict[Tuple[int, int], pd.DataFrame]:
    """The 14 pair-exclusion utility tables that predict ``target_split``."""
    out: Dict[Tuple[int, int], pd.DataFrame] = {}
    for s in SPLITS:
        if s == target_split:
            continue
        a, b = (min(s, target_split), max(s, target_split))
        d = pair_dir(root, a, b)
        mlp = pd.read_csv(_ext(d / "mlp" / f"predictions_split_{target_split:02d}.csv.gz"))
        knn = pd.read_csv(_ext(d / "knn" / f"predictions_split_{target_split:02d}.csv.gz"))
        mlp = mlp.rename(columns={f"pred_{m}": f"mlp__{m}" for m in metric_names})
        knn = knn.rename(columns={m: f"knn__{m}" for m in metric_names})
        out[(a, b)] = mlp.merge(knn, on=["dataset_id", "view_id"], how="inner")
    if len(out) != 14:
        raise ValueError(f"split {target_split}: expected 14 pair sources, "
                         f"found {len(out)}")
    return out
