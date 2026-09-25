"""L block — landmarking features (30 features, optional).

Lightweight unsupervised probes. We deliberately treat HDBSCAN as optional —
if the package isn't available we fill those features with NaN and continue.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..config import (
    EPS,
    L_AGGLO_K_VALUES,
    L_GMM_K_VALUES,
    L_KMEANS_K_VALUES,
    L_MAX_POINTS_FOR_LANDMARKING,
)
from ..feature_schema import FEATURE_COLUMNS_L
from ..utils.geometry_utils import sample_indices
from ..utils.numeric_utils import EPS as NUM_EPS, safe_div


def _nan_record() -> dict[str, float]:
    return {col: np.nan for col in FEATURE_COLUMNS_L}


def _silhouette_or_nan(X: np.ndarray, labels: np.ndarray, max_points: int = 1500) -> float:
    from sklearn.metrics import silhouette_score

    unique = set(int(v) for v in labels if v >= 0)
    if len(unique) < 2:
        return np.nan
    try:
        if X.shape[0] > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], size=max_points, replace=False)
            score = silhouette_score(X[idx], labels[idx])
        else:
            score = silhouette_score(X, labels)
        return float(score)
    except Exception:
        return np.nan


def compute_block_l(X_view: np.ndarray, view_dim: int, *, skip: bool, rng: np.random.Generator) -> dict[str, float]:
    out = _nan_record()
    if skip:
        return out

    mask = np.all(np.isfinite(X_view), axis=1)
    X = X_view[mask]
    if X.shape[0] < 30:
        return out

    if X.shape[0] > L_MAX_POINTS_FOR_LANDMARKING:
        idx = rng.choice(X.shape[0], size=L_MAX_POINTS_FOR_LANDMARKING, replace=False)
        X = X[idx]

    n = X.shape[0]

    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # KMeans + silhouette across k.
        km_silhouettes: dict[int, float] = {}
        km_inertias: dict[int, float] = {}
        for k in L_KMEANS_K_VALUES:
            try:
                km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
                km_silhouettes[k] = _silhouette_or_nan(X, km.labels_)
                km_inertias[k] = float(km.inertia_)
            except Exception:
                km_silhouettes[k] = np.nan
                km_inertias[k] = np.nan
            out[f"L_kmeans_k{k}_silhouette"] = km_silhouettes[k]

        valid_sil = {k: v for k, v in km_silhouettes.items() if np.isfinite(v)}
        if valid_sil:
            best_k = max(valid_sil, key=valid_sil.get)
            out["L_kmeans_best_k_by_silhouette"] = float(best_k)
            out["L_kmeans_best_silhouette"] = float(valid_sil[best_k])

        valid_inertia = [(k, v) for k, v in sorted(km_inertias.items()) if np.isfinite(v)]
        if len(valid_inertia) >= 2:
            ks = np.array([k for k, _ in valid_inertia], dtype=float)
            vals = np.array([v for _, v in valid_inertia], dtype=float)
            if vals.std(ddof=0) > NUM_EPS and ks.std(ddof=0) > NUM_EPS:
                slope, _ = np.polyfit(ks, vals, 1)
                out["L_kmeans_inertia_slope_2_to_5"] = float(slope)

        # GMM BIC.
        gmm_bics: dict[int, float] = {}
        for k in L_GMM_K_VALUES:
            try:
                gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit(X)
                gmm_bics[k] = float(gmm.bic(X))
            except Exception:
                gmm_bics[k] = np.nan
            out[f"L_gmm_k{k}_bic"] = gmm_bics[k]

        valid_bic = {k: v for k, v in gmm_bics.items() if np.isfinite(v)}
        if valid_bic:
            best_k = min(valid_bic, key=valid_bic.get)
            out["L_gmm_best_k_by_bic"] = float(best_k)
            vals = np.array(list(valid_bic.values()))
            denom = abs(vals.mean()) if abs(vals.mean()) > NUM_EPS else 1.0
            out["L_gmm_bic_range_norm"] = float((vals.max() - vals.min()) / denom)

        # DBSCAN at three eps presets, derived from kth-nearest-neighbor distance.
        from sklearn.neighbors import NearestNeighbors

        try:
            nn = NearestNeighbors(n_neighbors=min(10, max(2, n - 1))).fit(X)
            dists, _ = nn.kneighbors(X)
            base_eps = float(np.percentile(dists[:, -1], 80))
        except Exception:
            base_eps = float(np.std(X) * 0.1 + EPS)

        for preset, eps_mult, key_n, key_noise in [
            ("default", 1.0, "L_dbscan_default_n_clusters", "L_dbscan_default_noise_ratio"),
            ("loose", 2.0, "L_dbscan_loose_n_clusters", "L_dbscan_loose_noise_ratio"),
            ("strict", 0.5, "L_dbscan_strict_n_clusters", "L_dbscan_strict_noise_ratio"),
        ]:
            eps_val = max(base_eps * eps_mult, EPS)
            try:
                labels = DBSCAN(eps=eps_val, min_samples=5).fit_predict(X)
                n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
                noise = float(np.mean(labels == -1))
            except Exception:
                n_clusters = 0
                noise = np.nan
                labels = None
            out[key_n] = float(n_clusters)
            out[key_noise] = noise
            if preset == "default" and labels is not None:
                out["L_dbscan_default_silhouette_if_valid"] = _silhouette_or_nan(X, labels)

        # HDBSCAN (optional). Prefer sklearn.cluster.HDBSCAN (1.3+); fall back to hdbscan package.
        hdb_available = True
        hdb_labels = None
        hdb_persistences = None
        try:
            from sklearn.cluster import HDBSCAN as SKHDBSCAN

            try:
                hdb = SKHDBSCAN(min_cluster_size=20).fit(X)
                hdb_labels = hdb.labels_
                hdb_persistences = getattr(hdb, "cluster_persistence_", None)
            except Exception:
                hdb_available = False
        except ImportError:
            try:
                import hdbscan  # type: ignore

                clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
                hdb_labels = clusterer.fit_predict(X)
                hdb_persistences = getattr(clusterer, "cluster_persistence_", None)
            except ImportError:
                hdb_available = False
            except Exception:
                hdb_available = False

        if hdb_available and hdb_labels is not None:
            n_clusters = int(len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0))
            out["L_hdbscan_light_n_clusters"] = float(n_clusters)
            out["L_hdbscan_light_noise_ratio"] = float(np.mean(hdb_labels == -1))
            if hdb_persistences is not None and len(hdb_persistences) > 0:
                arr = np.asarray(hdb_persistences, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    out["L_hdbscan_light_cluster_persistence_mean"] = float(arr.mean())

        # Agglomerative silhouette across k.
        agglo_sils: dict[int, float] = {}
        for k in L_AGGLO_K_VALUES:
            try:
                if n < k + 1:
                    score = np.nan
                else:
                    agg = AgglomerativeClustering(n_clusters=k).fit(X)
                    score = _silhouette_or_nan(X, agg.labels_)
            except Exception:
                score = np.nan
            agglo_sils[k] = score
            out[f"L_agglomerative_k{k}_silhouette"] = score

    # Aggregate landmark scores.
    score_pool: list[float] = []
    for k, v in km_silhouettes.items():
        if np.isfinite(v):
            score_pool.append(v)
    for k, v in agglo_sils.items():
        if np.isfinite(v):
            score_pool.append(v)
    sil_default = out.get("L_dbscan_default_silhouette_if_valid", np.nan)
    if np.isfinite(sil_default):
        score_pool.append(sil_default)
    if score_pool:
        out["L_landmark_best_internal_score"] = float(np.max(score_pool))
        out["L_landmark_score_variance"] = float(np.var(score_pool, ddof=0))

    # Disagreement among algorithm preferred-k outputs.
    chosen_ks: list[float] = []
    for key in ("L_kmeans_best_k_by_silhouette", "L_gmm_best_k_by_bic",
                "L_dbscan_default_n_clusters", "L_hdbscan_light_n_clusters"):
        val = out.get(key, np.nan)
        if np.isfinite(val):
            chosen_ks.append(float(val))
    if chosen_ks:
        out["L_landmark_algorithm_disagreement"] = float(np.var(chosen_ks, ddof=0))

    return out
