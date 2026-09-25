"""K-handling, warmstart filtering, and the lightweight in-domain optimizer.

The optimizer is a seeded random search over the warmstart algorithms with K
restricted to ``2..5`` for K-based algorithms (n_clusters / n_components).
DBSCAN / HDBSCAN (no explicit K) sample their native parameters. This is a
documented, py3.12-compatible, ML2DAC-compatible search — NOT the original SMAC
Bayesian optimizer.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

K_PARAM_NAMES = ("n_clusters", "n_components", "k")
K_BASED_ALGORITHMS = {"kmeans", "minibatch_kmeans", "gmm", "agglomerative", "birch"}


def config_k(hyperparameters: dict) -> Optional[int]:
    for key in K_PARAM_NAMES:
        if key in hyperparameters:
            try:
                return int(hyperparameters[key])
            except Exception:
                return None
    return None


def is_k_based(algorithm: str, hyperparameters: dict) -> bool:
    return algorithm in K_BASED_ALGORITHMS or config_k(hyperparameters) is not None


def warmstart_within_k(algorithm: str, hyperparameters: dict, k_range: Tuple[int, int]) -> bool:
    """True if this warmstart config is allowed under the K restriction."""
    if not is_k_based(algorithm, hyperparameters):
        return True  # DBSCAN/HDBSCAN etc. always kept
    k = config_k(hyperparameters)
    if k is None:
        return True
    return k_range[0] <= k <= k_range[1]


def select_warmstarts(
    ranked_configs: List[dict], k_range: Tuple[int, int], n_warmstarts: int
) -> Tuple[List[dict], int]:
    """Take the top ``n_warmstarts`` valid warmstarts (K-filtered), filling down
    the ranking. Returns (selected, n_skipped_due_to_k_range)."""
    selected: List[dict] = []
    skipped = 0
    for cfg in ranked_configs:
        hp = json.loads(cfg["hyperparameters_json"])
        if warmstart_within_k(cfg["algorithm"], hp, k_range):
            cfg = dict(cfg)
            cfg["hyperparameters"] = hp
            selected.append(cfg)
            if len(selected) >= n_warmstarts:
                break
        else:
            skipped += 1
    return selected, skipped


# --- optimizer sampling ----------------------------------------------------

def _sample_params(algorithm: str, rng, k_range: Tuple[int, int]) -> Optional[dict]:
    lo, hi = k_range
    if algorithm in ("kmeans", "minibatch_kmeans"):
        p = {"n_clusters": int(rng.integers(lo, hi + 1)),
             "init": str(rng.choice(["k-means++", "random"])),
             "random_state": int(rng.integers(0, 10000))}
        if algorithm == "minibatch_kmeans":
            p["batch_size"] = int(rng.choice([50, 100, 200]))
        return p
    if algorithm == "gmm":
        return {"n_components": int(rng.integers(lo, hi + 1)),
                "covariance_type": str(rng.choice(["full", "tied", "diag", "spherical"])),
                "random_state": int(rng.integers(0, 10000))}
    if algorithm == "agglomerative":
        return {"n_clusters": int(rng.integers(lo, hi + 1)),
                "linkage": str(rng.choice(["ward", "complete", "average", "single"]))}
    if algorithm == "birch":
        return {"n_clusters": int(rng.integers(lo, hi + 1)),
                "threshold": round(float(rng.uniform(0.1, 0.9)), 3),
                "branching_factor": int(rng.choice([25, 50, 75]))}
    if algorithm == "dbscan":
        return {"eps": round(float(rng.uniform(0.05, 1.0)), 3),
                "min_samples": int(rng.integers(3, 16))}
    if algorithm == "hdbscan":
        return {"min_cluster_size": int(rng.integers(10, 151)),
                "min_samples": int(rng.integers(3, 16))}
    return None


def sample_optimizer_configs(
    algorithms: List[str], n_iter: int, k_range: Tuple[int, int], seed: int,
    avoid: Optional[set] = None,
) -> List[dict]:
    """Sample ``n_iter`` configs over ``algorithms`` (round-robin), K in k_range.

    Deterministic given ``seed``. Skips exact duplicates of ``avoid`` and of each
    other where possible (falls back to allowing repeats if the space is small).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    avoid = set(avoid or set())
    algos = [a for a in algorithms if _sample_params(a, rng, k_range) is not None] or list(algorithms)
    out: List[dict] = []
    seen = set(avoid)
    attempts = 0
    i = 0
    while len(out) < n_iter and attempts < n_iter * 50:
        attempts += 1
        algo = algos[i % len(algos)]
        i += 1
        params = _sample_params(algo, rng, k_range)
        if params is None:
            continue
        key = algo + "|" + json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append({"algorithm": algo, "hyperparameters": params})
    # If still short (tiny space), allow repeats to reach n_iter live evaluations.
    while len(out) < n_iter and algos:
        algo = algos[len(out) % len(algos)]
        params = _sample_params(algo, rng, k_range)
        if params is not None:
            out.append({"algorithm": algo, "hyperparameters": params})
        else:
            break
    return out


# --------------------------------------------------------------------------- #
# Constrained ("same search space") sampling / snapping.
#
# Used ONLY by the ``*_same_search_space`` in-domain variants. The candidate
# clustering configuration space (algorithms + per-parameter options) is the
# ClustOpt search map (``build_search_space(view)``), while ML2DAC's meta-ranking
# (predicted CVI, nearest neighbour, warmstart ranking + K filter) is untouched:
#   * optimizer configs are sampled from the ClustOpt grid instead of the
#     hardcoded ML2DAC ranges (:func:`sample_optimizer_configs_constrained`), and
#   * selected warmstarts are projected (snapped) onto the nearest valid ClustOpt
#     grid config (:func:`snap_to_space`), preserving algorithm + rank order.
# The ClustOpt ``space`` dict is ``{algorithm: {param: {type, low/high|choices}}}``.
# --------------------------------------------------------------------------- #

def space_algorithms(space: dict) -> List[str]:
    return list(space.keys())


def _param_is_k(pname: str) -> bool:
    return pname in ("n_clusters", "n_components")


def sample_from_space(algorithm: str, space: dict, rng, k_range: Tuple[int, int]) -> Optional[dict]:
    """Sample one config for ``algorithm`` from the ClustOpt ``space`` grid.

    K-based parameters are drawn from ``[low, high] ∩ k_range``. Returns ``None``
    if the algorithm is not part of the ClustOpt space.
    """
    algo_space = space.get(algorithm)
    if not algo_space:
        return None
    params: dict = {}
    for pname, spec in algo_space.items():
        ptype = spec.get("type")
        if ptype == "int":
            lo, hi = int(spec["low"]), int(spec["high"])
            if _param_is_k(pname):
                lo, hi = max(lo, k_range[0]), min(hi, k_range[1])
            if hi < lo:
                lo, hi = lo, lo
            params[pname] = int(rng.integers(lo, hi + 1))
        elif ptype == "float":
            params[pname] = round(float(rng.uniform(float(spec["low"]), float(spec["high"]))), 6)
        elif ptype == "categorical":
            choices = list(spec["choices"])
            params[pname] = choices[int(rng.integers(0, len(choices)))]
    return params


def sample_optimizer_configs_constrained(
    algorithms: List[str], space: dict, n_iter: int, k_range: Tuple[int, int],
    seed: int, avoid: Optional[set] = None,
) -> List[dict]:
    """Like :func:`sample_optimizer_configs` but draws from the ClustOpt ``space``.

    Round-robins over ``algorithms`` (restricted to those present in ``space``),
    K in ``k_range``. Deterministic given ``seed``; skips exact duplicates where
    the (small, categorical) grid allows.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    avoid = set(avoid or set())
    algos = [a for a in algorithms if a in space] or [a for a in space_algorithms(space)]
    out: List[dict] = []
    seen = set(avoid)
    attempts = 0
    i = 0
    while len(out) < n_iter and attempts < n_iter * 50:
        attempts += 1
        algo = algos[i % len(algos)]
        i += 1
        params = sample_from_space(algo, space, rng, k_range)
        if params is None:
            continue
        key = algo + "|" + json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append({"algorithm": algo, "hyperparameters": params})
    # tiny grid: allow repeats to reach the requested number of live evaluations.
    while len(out) < n_iter and algos:
        algo = algos[len(out) % len(algos)]
        params = sample_from_space(algo, space, rng, k_range)
        if params is not None:
            out.append({"algorithm": algo, "hyperparameters": params})
        else:
            break
    return out


def _nearest_choice(value, choices):
    """Nearest categorical choice to ``value`` (numeric nearest; else exact or first)."""
    if value in choices:
        return value
    numeric = all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in choices)
    if numeric and isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(choices, key=lambda c: abs(float(c) - float(value)))
    return choices[0]


def snap_to_space(algorithm: str, hyperparameters: dict, space: dict,
                  k_range: Tuple[int, int]) -> Optional[dict]:
    """Project a warmstart config onto the nearest valid ClustOpt ``space`` config.

    Produces a COMPLETE valid config for ``algorithm`` (all grid parameters
    filled): known params are snapped to the grid; params absent from the
    warmstart take the grid's lower/first value. Returns ``None`` if the
    algorithm is not part of the ClustOpt space (caller drops the warmstart).
    """
    algo_space = space.get(algorithm)
    if not algo_space:
        return None
    snapped: dict = {}
    for pname, spec in algo_space.items():
        ptype = spec.get("type")
        val = hyperparameters.get(pname)
        if ptype == "int":
            lo, hi = int(spec["low"]), int(spec["high"])
            if _param_is_k(pname):
                lo, hi = max(lo, k_range[0]), min(hi, k_range[1])
            if hi < lo:
                hi = lo
            if val is None:
                snapped[pname] = lo
            else:
                try:
                    snapped[pname] = int(min(max(round(float(val)), lo), hi))
                except (TypeError, ValueError):
                    snapped[pname] = lo
        elif ptype == "float":
            lo, hi = float(spec["low"]), float(spec["high"])
            try:
                v = float(val) if val is not None else lo
            except (TypeError, ValueError):
                v = lo
            snapped[pname] = round(min(max(v, lo), hi), 6)
        elif ptype == "categorical":
            choices = list(spec["choices"])
            snapped[pname] = _nearest_choice(val if val is not None else choices[0], choices)
    return snapped
