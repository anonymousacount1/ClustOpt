"""Modern comparator CVIs for EXTERNAL METRIC VALIDATION ONLY.

None of these is ever added to the ClustOpt, AutoClust or ML2DAC learned
inventories. They exist so the external metric study can compare the project's
46 designed indices against strong modern baselines.

Provenance and licensing (full detail in ``configs/modern_cvi_provenance.json``):

* **CDbw** -- PyPI ``cdbw`` 0.2 (Rubinsky, Lashkov, Eistrikh-Heller), MIT,
  Copyright (c) 2019 Alexander Lashkov. The installed package is called for its
  geometry (representatives, shrinking, densities); the separation step is
  re-implemented from the package code with the three documented corrections in
  :data:`CDBW_PAPER_CORRECTIONS`. Requires ``cdbw==0.2``. Full MIT notice in
  ``THIRD_PARTY_NOTICES.md``.
* **CVNN** -- Liu et al. (2013). Not implemented in this release: computed on
  request by the external R package ``fpc`` via ``cvnn_fpc_adapter`` (fpc is not
  distributed). CVNN is not part of CLUSTOPT's 60-index inventory.
* **CVDD** -- Hu & Zhong (2019); a port of the author implementation
  ``hulianyu/CVDD`` @ 32cabe0c, MIT, Copyright (c) 2019 Lianyu Hu. Full MIT
  notice in ``THIRD_PARTY_NOTICES.md``.
* **DCSI** -- Gauss et al. (2024). The author repository declares **no licence**,
  so this implementation is written from the paper's mathematics only and the
  repository is consulted purely as an external oracle.

Every metric returns a :class:`MetricResult` that is either a finite value or an
explicit ``INVALID`` with a reason. Formulas are never altered to force uniform
edge behaviour across metrics -- different published indices legitimately differ
on noise, singletons and degenerate partitions, and flattening that would
misrepresent them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: direction of improvement, frozen from the source publications
DIRECTION = {"cdbw": "maximize", "cvnn": "minimize",
             "cvdd": "maximize", "dcsi": "maximize"}
#: frozen parameters, pinned before any external outcome was observed
FROZEN_PARAMS = {
    "cdbw": {"metric": "euclidean", "alg_noise": "comb",
             "intra_dens_inf": False, "s": 3},
    "cvnn": {"k": 5},
    "cvdd": {"K": 7, "duplicate_policy": "INVALID(CVDD_DUPLICATE_POINT_MST_AMBIGUITY)"},
    "dcsi": {"MinPts": 5, "distance": "euclidean", "multi_class": "mean pairwise"},
}


@dataclass
class MetricResult:
    metric_id: str
    value: Optional[float]
    direction: str
    valid: bool
    invalid_reason: str = ""
    runtime_sec: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def oriented(self) -> Optional[float]:
        """Higher-is-better form, for uniform downstream ranking."""
        if not self.valid or self.value is None:
            return None
        return -self.value if self.direction == "minimize" else self.value


def _clean(X: np.ndarray, labels: np.ndarray):
    X = np.asarray(X, dtype=float)
    y = np.asarray(labels).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X, y


def _basic_invalid(X: np.ndarray, y: np.ndarray, *, need_k2: bool = True
                   ) -> Optional[str]:
    if X.shape[0] != y.shape[0]:
        return "length_mismatch"
    real = y[y >= 0]
    k = len(np.unique(real))
    if need_k2 and k < 2:
        return "fewer_than_two_clusters"
    if real.size == 0:
        return "all_points_are_noise"
    return None


# --------------------------------------------------------------------- CDbw
#: Corrections applied to PyPI ``cdbw`` 0.2 so the value follows the published
#: mathematics (Halkidi & Vazirgiannis 2008) and is intrinsically invariant to
#: how clusters are named. Full audit: ``configs/cdbw_definition_audit.json``.
CDBW_PAPER_CORRECTIONS = {
    "C1_stdev": "the neighbourhood radius must be the AVERAGE standard "
                "deviation over the considered clusters. prep() overwrites a "
                "scalar inside a double loop, so it returns the standard "
                "deviation of the HIGHEST-INDEXED cluster only. Affects "
                "compactness, cohesion and separation, and is the first "
                "internal quantity that changes under a label permutation.",
    "C2_separation_distance": "Sep aggregates, for each cluster, the distance "
                              "to its CLOSEST OTHER cluster. separation() "
                              "minimises over lower-indexed clusters only, so "
                              "cluster 0 contributes 0 to a sum divided by K.",
    "C3_inter_density": "Inter_dens is the mean over clusters of each "
                        "cluster's maximum pairwise inter-density. "
                        "separation() uses the single global maximum over all "
                        "pairs divided by K. This one is label-invariant, so "
                        "it is a faithfulness correction rather than a cause "
                        "of the ordering defect.",
}
#: The corrected index no longer needs an ordering convention, but the package's
#: helpers index arrays by label, so labels are made contiguous and 0-based.
#: That relabelling is now provably neutral -- see
#: ``validation/test_cdbw_permutation_invariance.py``.
CDBW_LABEL_ORDER_POLICY = "not_required_paper_faithful_is_permutation_invariant"
#: retained ONLY so the diagnostic package-compatibility mode is reproducible
CDBW_COMPAT_LABEL_ORDER_POLICY = "canonical_centroid_lexicographic"


def cdbw_canonical_labels(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Renumber clusters 0..K-1 by a partition-intrinsic canonical order.

    Only the diagnostic ``cdbw_package_compat`` needs this: the raw package is
    order-dependent, so without a rule it has no single value to report. The
    primary ``cdbw`` does not use it.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    out = np.full(y.shape, -1, dtype=int)
    real = y >= 0
    keys = []
    for c in np.unique(y[real]):
        m = y == c
        keys.append((tuple(X[m].mean(axis=0).tolist()), int(m.sum()),
                     int(np.flatnonzero(m)[0]), c))
    for new_id, k in enumerate(sorted(keys, key=lambda t: t[:3])):
        out[y == k[3]] = new_id
    return out


def _cdbw_contiguous(y: np.ndarray) -> np.ndarray:
    """0-based contiguous labels, noise preserved as -1."""
    y = np.asarray(y).ravel().astype(int)
    out = y.copy()
    real = y >= 0
    _, comp = np.unique(y[real], return_inverse=True)
    out[real] = comp
    return out


def _cdbw_paper_stdev(X, y, n_clusters, dimension):
    """C1: the average standard deviation over the considered clusters.

    Per-cluster spread is computed exactly as the package computes it, so the
    only change is the aggregation.
    """
    per = []
    for i in range(n_clusters):
        sd = np.std(X[y == i], axis=0)
        per.append(float(np.sqrt(float(np.dot(sd, sd)) / dimension)))
    # sorted before reduction so the floating-point summation order is a
    # function of the VALUES, not of how the clusters happened to be numbered
    return float(np.mean(sorted(per))), per


def _cdbw_paper_separation(X, y, n_clusters, stdev, middle_point, dist_min,
                           n_cl_rep, n_points_in_cl, coord_in_cl, metric):
    """C2 + C3. Per-pair quantities are ported from the MIT ``cdbw`` 0.2 package
    (Copyright (c) 2019 Alexander Lashkov); the aggregation is corrected (C2, C3).

    Raises ValueError for a pair with no mutually closest representatives.
    """
    from scipy.spatial.distance import cdist
    dk = np.zeros((n_clusters, n_clusters), dtype=float)
    dens = np.zeros((n_clusters, n_clusters), dtype=float)
    for i in range(n_clusters):
        for j in range(i):
            m = int(n_cl_rep[(i, j)])
            if m == 0:
                raise ValueError("cdbw_no_mutual_representative_pair")
            mp = np.asarray(middle_point[(i, j)], dtype=float)
            dm = np.asarray(dist_min[(i, j)], dtype=float).ravel()
            pts = np.vstack([coord_in_cl[i, 0:n_points_in_cl[i], :],
                             coord_in_cl[j, 0:n_points_in_cl[j], :]])
            card = (cdist(mp, pts, metric=metric) < stdev).sum(axis=1)
            dk[i, j] = dk[j, i] = float(dm.sum()) / m
            dens[i, j] = dens[j, i] = (
                float((card * dm).sum())
                / (stdev * (n_points_in_cl[i] + n_points_in_cl[j])))
    off = ~np.eye(n_clusters, dtype=bool)
    # sorted for the same reason as in _cdbw_paper_stdev
    inter_dens = float(np.mean(sorted(dens[i][off[i]].max()
                                      for i in range(n_clusters))))
    dist_m = float(np.mean(sorted(dk[i][off[i]].min()
                                  for i in range(n_clusters))))
    sep = dist_m / (1.0 + inter_dens)
    return sep, {"inter_dens": inter_dens, "dist_m": dist_m,
                 "pairwise_distance": dk.tolist(),
                 "pairwise_inter_density": dens.tolist()}


def cdbw_parts(X, labels, s: int = 3, metric: str = "euclidean"):
    """Paper-faithful compactness, cohesion, separation and CDbw, plus internals.

    The geometry (representatives, shrunk representatives, per-cluster density,
    mutually closest representative pairs) is the MIT ``cdbw`` package's own
    code, reused with attribution. Only the three aggregations recorded in
    :data:`CDBW_PAPER_CORRECTIONS` are replaced.
    """
    from cdbw import cdbw as _pkg
    X = np.asarray(X, dtype=float)
    y = _cdbw_contiguous(labels)
    y = np.asarray(_pkg.comb_noise_lab(y))          # frozen alg_noise='comb'
    (n_clusters, _pkg_stdev, dimension, n_points_in_cl, n_max, coord_in_cl,
     labels_in_cl) = _pkg.prep(X, y)
    stdev, per_cluster_sd = _cdbw_paper_stdev(X, y, n_clusters, dimension)
    mean_arr, n_rep, n_rep_max, rep_in_cl = _pkg.rep(
        n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl)
    distvec = _pkg.gen_dist_func(metric)
    middle_point, dist_min, n_cl_rep = _pkg.closest_rep(
        X, n_clusters, rep_in_cl, n_rep, metric, distvec)
    a_rep_shell = _pkg.art_rep(X, n_clusters, rep_in_cl, n_rep, n_rep_max,
                               mean_arr, s, dimension)
    compact, cohesion = _pkg.compactness(
        n_clusters, stdev, a_rep_shell, n_rep, n_points_in_cl, s, coord_in_cl,
        n_max, n_rep_max, metric)
    sep, comps = _cdbw_paper_separation(
        X, y, n_clusters, stdev, middle_point, dist_min, n_cl_rep,
        n_points_in_cl, coord_in_cl, metric)
    return {"compactness": float(compact), "cohesion": float(cohesion),
            "separation": float(sep),
            "cdbw": float(compact) * float(cohesion) * float(sep),
            "stdev_paper": stdev, "stdev_package": float(_pkg_stdev),
            "per_cluster_sd": per_cluster_sd, "n_clusters": int(n_clusters),
            **comps}


def cdbw(X, labels, **kw) -> MetricResult:
    """CDbw (Halkidi & Vazirgiannis 2008), paper-faithful and label-invariant.

    Higher is better. The reference package is reused for geometry; three
    aggregations are corrected so the value follows the published mathematics
    and depends only on ``(X, partition)``.
    """
    X, y = _clean(X, labels)
    p = dict(FROZEN_PARAMS["cdbw"]); p.update(kw)
    bad = _basic_invalid(X, y)
    prov = {"reference": "PyPI cdbw==0.2 (MIT), geometry reused with attribution",
            "relationship": "paper-faithful correction layer",
            "definition": "Halkidi & Vazirgiannis (2008)",
            "corrections": sorted(CDBW_PAPER_CORRECTIONS),
            "label_order_policy": CDBW_LABEL_ORDER_POLICY, "params": p}
    if bad:
        return MetricResult("cdbw", None, DIRECTION["cdbw"], False, bad,
                            provenance=prov)
    t = time.perf_counter()
    try:
        parts = cdbw_parts(X, y, s=p["s"], metric=p["metric"])
    except ValueError as e:
        reason = str(e)
        if "only 1 point" in reason:
            reason = "cdbw_cluster_with_single_representative"
        elif "cdbw_no_mutual_representative_pair" not in reason:
            reason = "reference_error:ValueError"
        return MetricResult("cdbw", None, DIRECTION["cdbw"], False, reason,
                            time.perf_counter() - t, prov)
    except Exception as e:
        return MetricResult("cdbw", None, DIRECTION["cdbw"], False,
                            "reference_error:%s" % type(e).__name__,
                            time.perf_counter() - t, prov)
    v = parts["cdbw"]
    if not np.isfinite(v) or not np.isfinite(parts["compactness"]):
        # the package silently returns 0 here; an unscoreable input is INVALID
        return MetricResult("cdbw", None, DIRECTION["cdbw"], False,
                            "cdbw_non_finite_density", time.perf_counter() - t,
                            prov)
    return MetricResult("cdbw", float(v), DIRECTION["cdbw"], True, "",
                        time.perf_counter() - t,
                        dict(prov, components={k: parts[k] for k in
                             ("compactness", "cohesion", "separation",
                              "stdev_paper", "inter_dens", "dist_m")}))


def cdbw_package_compat(X, labels, **kw) -> MetricResult:
    """DIAGNOSTIC ONLY. The raw PyPI ``cdbw`` 0.2 value.

    Not a comparator, never in the primary registry. The package is not
    invariant to cluster naming, so a canonical ordering is imposed purely to
    make this diagnostic reproducible.
    """
    X, y = _clean(X, labels)
    p = dict(FROZEN_PARAMS["cdbw"]); p.update(kw)
    prov = {"reference": "PyPI cdbw==0.2 (MIT)", "relationship": "raw package",
            "role": "DIAGNOSTIC sensitivity variant -- NOT a comparator",
            "label_order_policy": CDBW_COMPAT_LABEL_ORDER_POLICY, "params": p}
    bad = _basic_invalid(X, y)
    if bad:
        return MetricResult("cdbw_package_compat", None, DIRECTION["cdbw"],
                            False, bad, provenance=prov)
    t = time.perf_counter()
    try:
        from cdbw import CDbw
        v = float(CDbw(X, cdbw_canonical_labels(X, y), **p))
    except Exception as e:
        return MetricResult("cdbw_package_compat", None, DIRECTION["cdbw"],
                            False, "reference_error:%s" % type(e).__name__,
                            time.perf_counter() - t, prov)
    if not np.isfinite(v):
        return MetricResult("cdbw_package_compat", None, DIRECTION["cdbw"],
                            False, "non_finite_value",
                            time.perf_counter() - t, prov)
    return MetricResult("cdbw_package_compat", v, DIRECTION["cdbw"], True, "",
                        time.perf_counter() - t, prov)


# --------------------------------------------------------------------- CVNN
# CVNN is not implemented in this release. It is computed, when wanted, by the
# external R package fpc through cvnn_fpc_adapter (no fpc code is included).
from .cvnn_fpc_adapter import cvnn, cvnn_index  # noqa: E402,F401


# --------------------------------------------------------------------- CVDD
# Port of the author implementation, github.com/hulianyu/CVDD @ 32cabe0c
# (MIT License, Copyright (c) 2019 Lianyu Hu), per Hu & Zhong (2019),
# "An Internal Validity Index Based on Density-Involved Distance", IEEE Access
# 7:40038-40051. Ported with attribution as the MIT licence permits; semantics
# are preserved exactly, including the author's error fallbacks. The full MIT
# licence text is reproduced in THIRD_PARTY_NOTICES.md.
CVDD_K = 7


def _minimax_path_distance(W: np.ndarray) -> np.ndarray:
    """Verbatim port of the author's ``fast_PathbasedDist`` (MIT, L. Hu 2018).

    This deliberately reproduces the author's traversal rather than substituting
    a mathematically cleaner bottleneck-distance routine. The two agree whenever
    the minimum spanning tree is unique, but the author's "Nodes each other"
    block fills some entries from the ORIGINAL weight matrix rather than from the
    running path maximum, which makes the result order-dependent when edges tie.
    Exact duplicate points create such ties, so matching the reference here
    requires matching its traversal exactly.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    W = np.asarray(W, dtype=float)
    N = W.shape[0]
    if N <= 1:
        return np.zeros((N, N))
    mst = minimum_spanning_tree(W).toarray()
    pairs = np.argwhere(mst > 0)
    if len(pairs) < N - 1:                      # zero-weight MST edges
        m2 = np.maximum(mst, mst.T)
        seen = set()
        pairs = []
        for i in range(N):
            for j in np.nonzero(m2[i])[0]:
                if (min(i, j), max(i, j)) not in seen:
                    seen.add((min(i, j), max(i, j)))
                    pairs.append((i, j))
        pairs = np.array(pairs) if pairs else np.zeros((0, 2), int)

    PathbasedW = np.zeros((N, N))
    Degrees = np.zeros(N, dtype=int)
    Connections = np.zeros((N, N), dtype=int)
    for a, b in pairs:
        Connections[a, Degrees[a]] = b
        Degrees[a] += 1
        Connections[b, Degrees[b]] = a
        Degrees[b] += 1

    start = N - 1
    for i in range(N):                          # first node of degree 1
        if Degrees[i] == 1:
            start = i
            break
    CloseT = np.zeros(N, dtype=int)
    OpenT = np.zeros(N, dtype=int)
    cursor_C = 0
    cursor_T = 0
    CloseT[cursor_C] = start
    Visited = np.zeros(N, dtype=int)
    Visited[start] = 1
    while cursor_C < N - 1:
        cur = CloseT[cursor_C]
        Nodes1 = Connections[cur, :Degrees[cur]]
        Nodes = [int(x) for x in Nodes1 if Visited[x] == 0]
        for nd in Nodes:
            for j in range(cursor_C + 1):
                PathbasedW[CloseT[j], nd] = max(PathbasedW[CloseT[j], cur],
                                                W[cur, nd])
                PathbasedW[nd, CloseT[j]] = PathbasedW[CloseT[j], nd]
            for j in range(cursor_T):
                PathbasedW[OpenT[j], nd] = max(PathbasedW[OpenT[j], cur],
                                               W[cur, nd])
                PathbasedW[nd, OpenT[j]] = PathbasedW[OpenT[j], nd]
        for a in range(len(Nodes) - 1):
            for b in range(a + 1, len(Nodes)):
                PathbasedW[Nodes[a], Nodes[b]] = max(W[Nodes[a], cur],
                                                     W[Nodes[b], cur])
                PathbasedW[Nodes[b], Nodes[a]] = PathbasedW[Nodes[a], Nodes[b]]
        for nd in Nodes:
            OpenT[cursor_T] = nd
            cursor_T += 1
        cursor_C += 1
        CloseT[cursor_C] = OpenT[cursor_T - 1]
        Visited[OpenT[cursor_T - 1]] = 1
        cursor_T -= 1
    return PathbasedW


def density_involved_distance(d: np.ndarray, K: int = CVDD_K) -> np.ndarray:
    """`Density_involved_distance(d, K)` from the author implementation."""
    n = d.shape[0]
    dd = np.array(d, dtype=float, copy=True)
    np.fill_diagonal(dd, np.inf)
    knn = np.argsort(dd, axis=0, kind="stable")[:K, :].T   # column-wise, as MATLAB
    Den = np.array([d[j, knn[j]].sum() / K for j in range(n)], dtype=float)
    mx = Den.max()
    fDen = Den / mx if mx > 0 else np.zeros_like(Den)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rel = Den[:, None] / Den[None, :]
    tmp1 = Rel + Rel.T
    fRel = 1.0 - np.exp(-np.abs(tmp1 - 2.0))
    nD = Den[:, None] + Den[None, :]
    drD = d + nD * fRel
    conD = _minimax_path_distance(drD)
    return conD * np.sqrt(fDen[:, None] * fDen[None, :])


def cvdd(X, labels, K: int = CVDD_K) -> MetricResult:
    """CVDD = sum_i sep[i] / sum_i com[i]; higher is better."""
    from scipy.spatial.distance import cdist
    X, y = _clean(X, labels)
    prov = {"reference": "hulianyu/CVDD @ 32cabe0cf876ccd4e04e44dda84b0ad96be88bbc (MIT)",
            "relationship": "port with attribution", "params": {"K": K},
            "citation": "Hu & Zhong (2019), IEEE Access 7:40038-40051"}
    bad = _basic_invalid(X, y)
    if bad:
        return MetricResult("cvdd", None, DIRECTION["cvdd"], False, bad,
                            provenance=prov)
    # FROZEN APPLICABILITY RULE (declared before any external prevalence was
    # inspected). The density-involved distance is built on a path-based
    # (minimax) distance derived from a minimum spanning tree. Exact duplicate
    # points create zero-weight edges, the MST is then non-unique, and the
    # reference procedure defines no implementation-independent tie-break --
    # different valid MSTs give different CVDD values. No canonical value
    # therefore exists for such an input, so CVDD is INVALID here. Exact row
    # equality is used deliberately: an epsilon would introduce a further
    # arbitrary metric parameter. This invalidates CVDD ONLY -- never the
    # dataset, the view, another CVI, a method run, or candidate generation.
    if len(np.unique(X, axis=0)) < X.shape[0]:
        return MetricResult("cvdd", None, DIRECTION["cvdd"], False,
                            "CVDD_DUPLICATE_POINT_MST_AMBIGUITY",
                            provenance=dict(prov, n_points=int(X.shape[0]),
                                            n_unique_points=int(len(np.unique(X, axis=0)))))
    t = time.perf_counter()
    try:
        d = cdist(X, X)
        DD = density_involved_distance(d, K)
        uniq = np.unique(y)
        sep_l, com_l = [], []
        for j in uniq:
            a = y == j
            b = ~a
            n = int(a.sum())
            sep_l.append(float(DD[np.ix_(a, b)].min()) if b.any() else 0.0)
            try:
                Ci = _minimax_path_distance(d[np.ix_(a, a)])
                # MATLAB std2 is the N-1 normalised std over all elements, and
                # MATLAB returns 0 (not NaN) for a single element, which is the
                # singleton-cluster case. numpy with ddof=1 would give NaN, so
                # the MATLAB convention is reproduced explicitly.
                sd = float(np.std(Ci, ddof=1)) if Ci.size > 1 else 0.0
                com_l.append(sd / n * float(Ci.mean()))
            except Exception:
                com_l.append(float(max(com_l)) if com_l else 0.0)
        sep, com = float(np.sum(sep_l)), float(np.sum(com_l))
        v = sep / com if com != 0 else float("nan")
    except Exception as e:
        # the author returns 0 when Rel fails on total overlap (zero distances)
        return MetricResult("cvdd", 0.0, DIRECTION["cvdd"], True,
                            "author_fallback:%s" % type(e).__name__,
                            time.perf_counter() - t, prov)
    if not np.isfinite(v):
        return MetricResult("cvdd", None, DIRECTION["cvdd"], False,
                            "non_finite_value", time.perf_counter() - t, prov)
    return MetricResult("cvdd", v, DIRECTION["cvdd"], True, "",
                        time.perf_counter() - t,
                        dict(prov, sep=sep, com=com))


# --------------------------------------------------------------------- DCSI
# Independent implementation from the published definition in
# Gauss, Scheipl & Herrmann, "DCSI -- An improved measure of cluster
# separability based on separation and connectedness", arXiv:2310.12806v4
# (CC BY 4.0). The author repository (JanaGauss/dcsi) declares NO licence and is
# therefore used ONLY as an external numerical oracle: no line of it is copied,
# vendored or translated here. Full definition audit:
# configs/dcsi_definition_audit.json
DCSI_MINPTS = 5


def _dcsi_epsilon(dc: np.ndarray, minpts: int) -> float:
    """eps_i = median over class points of the distance to the (2*MinPts)-th
    within-class nearest neighbour (Definition, 'Choice of parameters')."""
    n = dc.shape[0]
    if n < 2:
        return 0.0
    k = 2 * minpts
    order = np.sort(dc, axis=1)          # column 0 is the self-distance (0)
    idx = min(k, n - 1)                  # continuous extension for small classes
    return float(np.median(order[:, idx]))


def _dcsi_core_points(dc: np.ndarray, minpts: int) -> np.ndarray:
    """Boolean mask of core points: |N_eps(x)| >= MinPts, self excluded."""
    eps = _dcsi_epsilon(dc, minpts)
    n = dc.shape[0]
    within = (dc <= eps)
    np.fill_diagonal(within, False)      # x' in C_i \ {x}
    return within.sum(axis=1) >= minpts


def _dcsi_connectedness(dc: np.ndarray) -> float:
    """Largest edge weight of an MST built on the given (core) points.

    The maximum MST edge is identical for every minimum spanning tree even when
    the MST is not unique, so -- unlike CVDD -- duplicates create no ambiguity.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    n = dc.shape[0]
    if n <= 1:
        return 0.0
    mst = minimum_spanning_tree(dc).toarray()
    return float(mst.max()) if mst.size else 0.0


#: Separation modes. ONE shared implementation, so the exact difference between
#: the published definition and the released reference code is auditable in a
#: single place rather than duplicated.
SEP_MODE_PAPER = "PAPER"
SEP_MODE_AUTHOR = "AUTHOR_COMPAT"


def dcsi_pair(X, y, a, b, minpts: int = DCSI_MINPTS,
              sep_mode: str = SEP_MODE_PAPER):
    """(value, invalid_reason, components) for one pair of classes.

    ``sep_mode``:

    * ``PAPER``        -- Sep = min distance between core points (Definition 3.2).
    * ``AUTHOR_COMPAT`` -- Sep = 2 x the above. The released reference code fills
      both triangles of its separation matrix and then adds the transpose, which
      doubles every entry; this mode reproduces that observable behaviour for
      diagnostic comparison only. It is NOT a primary comparator.
    """
    from scipy.spatial.distance import cdist
    if sep_mode not in (SEP_MODE_PAPER, SEP_MODE_AUTHOR):
        raise ValueError("unknown sep_mode %r" % sep_mode)
    Xa, Xb = X[y == a], X[y == b]
    if len(Xa) == 0 or len(Xb) == 0:
        return None, "dcsi_empty_class", {}
    da, db = cdist(Xa, Xa), cdist(Xb, Xb)
    ca, cb = _dcsi_core_points(da, minpts), _dcsi_core_points(db, minpts)
    if ca.sum() == 0 or cb.sum() == 0:
        return None, "dcsi_no_core_points", {}
    sep_paper = float(cdist(Xa[ca], Xb[cb]).min())
    conn_a = _dcsi_connectedness(da[np.ix_(ca, ca)])
    conn_b = _dcsi_connectedness(db[np.ix_(cb, cb)])
    conn = max(conn_a, conn_b)
    sep = sep_paper * (2.0 if sep_mode == SEP_MODE_AUTHOR else 1.0)
    comp = {"sep_paper": sep_paper, "sep_used": sep, "conn_a": conn_a,
            "conn_b": conn_b, "conn": conn, "n_core_a": int(ca.sum()),
            "n_core_b": int(cb.sum())}
    if conn == 0.0 and sep == 0.0:
        return None, "dcsi_indeterminate_zero_over_zero", comp
    if conn == 0.0:
        comp["q"] = float("inf")
        return 1.0, "", comp
    q = sep / conn
    comp["q"] = q
    return float(q / (1.0 + q)), "", comp


def dcsi(X, labels, minpts: int = DCSI_MINPTS,
         sep_mode: str = SEP_MODE_PAPER) -> MetricResult:
    """DCSI(X) = mean of the K(K-1)/2 pairwise DCSI values. Higher is better.

    NOTE ON RANKING. The pairwise score q/(1+q) is strictly increasing in q, so
    for K=2 the PAPER and AUTHOR_COMPAT modes rank candidates identically. This
    does NOT extend to K>=3: the multiclass value is the MEAN of a nonlinear
    transform, and mean(f(q)) and mean(g(q)) can order candidates differently.
    See ``validation/test_dcsi_variant_ranking.py`` for a worked reversal. The
    two modes must never be described as globally ranking-equivalent.
    """
    X, y = _clean(X, labels)
    mid = "dcsi" if sep_mode == SEP_MODE_PAPER else "dcsi_author_compat"
    prov = {"reference": "arXiv:2310.12806v4 (CC BY 4.0); oracle JanaGauss/dcsi "
                         "@ 15ba6f7bd89151e64b009dc4329f678f416ddc6c",
            "relationship": "independent implementation from the paper "
                            "(no author source copied, vendored or translated)",
            "sep_mode": sep_mode,
            "role": ("PRIMARY scientific comparator" if sep_mode == SEP_MODE_PAPER
                     else "DIAGNOSTIC sensitivity variant -- NOT a comparator"),
            "params": {"MinPts": minpts, "distance": "euclidean",
                       "multi_class": "mean pairwise"}}
    bad = _basic_invalid(X, y)
    if bad:
        return MetricResult(mid, None, DIRECTION["dcsi"], False, bad,
                            provenance=prov)
    t = time.perf_counter()
    classes = [c for c in np.unique(y) if c >= 0]   # noise label is not a class
    vals, reasons, comps = [], [], []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            v, r, c = dcsi_pair(X, y, classes[i], classes[j], minpts, sep_mode)
            c = dict(c, pair=(int(classes[i]), int(classes[j])))
            comps.append(c)
            if v is None:
                reasons.append("%s|%s:%s" % (classes[i], classes[j], r))
            else:
                vals.append(v)
    # The paper aggregates over ALL K(K-1)/2 pairs. If any pair is undefined the
    # mean over the survivors is a DIFFERENT estimand, and the paper is silent on
    # substituting it, so the whole partition is INVALID rather than silently
    # rescored on a subset.
    if reasons:
        return MetricResult(mid, None, DIRECTION["dcsi"], False,
                            "dcsi_pair_undefined:%s" % reasons[0].split(":")[-1],
                            time.perf_counter() - t,
                            dict(prov, pair_failures=reasons,
                                 n_pairs_valid=len(vals), components=comps))
    return MetricResult(mid, float(np.mean(vals)), DIRECTION["dcsi"], True, "",
                        time.perf_counter() - t,
                        dict(prov, n_pairs=len(vals), pair_values=vals,
                             pair_failures=reasons, components=comps))


def dcsi_components(X, labels, minpts: int = DCSI_MINPTS) -> Dict[str, Any]:
    """Per-class and per-pair internals, for component-level oracle comparison.

    Aggregates are never compared alone: a matching total can hide two
    compensating component errors, so core points, connectedness and separation
    are checked individually against the reference.
    """
    from scipy.spatial.distance import cdist
    X, y = _clean(X, labels)
    classes = [int(c) for c in np.unique(y) if c >= 0]
    per_class, dists, cores = {}, {}, {}
    for c in classes:
        idx = np.flatnonzero(y == c)
        d = cdist(X[idx], X[idx])
        m = _dcsi_core_points(d, minpts)
        dists[c], cores[c] = d, m
        per_class[c] = {
            "n": int(idx.size),
            "epsilon": _dcsi_epsilon(d, minpts),
            "n_core": int(m.sum()),
            #: 0-based positions into the FULL input, sorted
            "core_indices": np.sort(idx[m]).tolist(),
            "conn": _dcsi_connectedness(d[np.ix_(m, m)]),
        }
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = classes[i], classes[j]
            ia = np.flatnonzero(y == a)[cores[a]]
            ib = np.flatnonzero(y == b)[cores[b]]
            rec = {"pair": (a, b)}
            if ia.size == 0 or ib.size == 0:
                rec["invalid_reason"] = "dcsi_no_core_points"
                pairs.append(rec)
                continue
            sep_paper = float(cdist(X[ia], X[ib]).min())
            conn = max(per_class[a]["conn"], per_class[b]["conn"])
            rec.update({"sep_paper": sep_paper,
                        "sep_author_compat": 2.0 * sep_paper, "conn": conn})
            for mode, sep in (("paper", sep_paper),
                              ("author_compat", 2.0 * sep_paper)):
                if conn == 0.0:
                    rec["q_" + mode] = float("inf")
                    rec["dcsi_" + mode] = None if sep == 0.0 else 1.0
                else:
                    q = sep / conn
                    rec["q_" + mode] = q
                    rec["dcsi_" + mode] = q / (1.0 + q)
            pairs.append(rec)
    def _agg(mode):
        v = [p["dcsi_" + mode] for p in pairs if p.get("dcsi_" + mode) is not None]
        return float(np.mean(v)) if v else None
    return {"minpts": minpts, "classes": classes, "per_class": per_class,
            "pairs": pairs, "aggregate_paper": _agg("paper"),
            "aggregate_author_compat": _agg("author_compat")}


def dcsi_author_compat(X, labels, minpts: int = DCSI_MINPTS) -> MetricResult:
    """DIAGNOSTIC ONLY. Reproduces the released reference implementation.

    Not a modern comparator, never enters the primary registry, never
    contributes oracle coverage or a primary FDR test family.
    """
    return dcsi(X, labels, minpts=minpts, sep_mode=SEP_MODE_AUTHOR)
