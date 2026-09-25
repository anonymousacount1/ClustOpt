from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from models.HYBRID_SCM.io.config_io import DatasetConfig, ComponentConfig
from .components import get_component, apply_component_operators
from .priors import sample_from_spec
from .rasterizer import resolve_rendering_spec
from .semantic_metadata import compute_semantic_pattern_factors


UINT32_MAX = np.iinfo(np.uint32).max


def _topological_order(components: List[ComponentConfig]) -> List[ComponentConfig]:
    name_to_comp = {c.name: c for c in components}
    temp = set()
    perm = set()
    out: List[ComponentConfig] = []

    def visit(name: str) -> None:
        if name in perm:
            return
        if name in temp:
            raise ValueError(f"Cycle detected around component '{name}'.")
        temp.add(name)
        comp = name_to_comp[name]
        for p in comp.parents:
            visit(p)
        temp.remove(name)
        perm.add(name)
        out.append(comp)

    for c in components:
        visit(c.name)
    return out


def _spawn_rng(master_rng: np.random.Generator, frozen_seed: int | None = None) -> tuple[int, np.random.Generator]:
    seed = int(frozen_seed) if frozen_seed is not None else int(master_rng.integers(0, UINT32_MAX, dtype=np.uint32))
    return seed, np.random.default_rng(seed)


def _as_feature_sigma(spec: Any, n_features: int) -> np.ndarray:
    if np.isscalar(spec):
        return np.full((n_features,), float(spec), dtype=float)
    arr = np.asarray(spec, dtype=float).reshape(-1)
    if arr.size != n_features:
        raise ValueError(f"Expected anisotropic_jitter_sigma to have length {n_features}, got {arr.size}.")
    return arr


def _cluster_centroids(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for lab in np.unique(y):
        mask = y == lab
        if np.any(mask):
            out[int(lab)] = X[mask].mean(axis=0)
    return out



DEPRECATED_GLOBAL_TO_DATASET_KEYS = {
    "anisotropic_jitter_sigma",
    "point_dropout_rate",
    "duplicate_rate",
    "duplicate_jitter_sigma",
    "local_contamination_rate",
    "local_contamination_sigma",
    "bridge_segments",
    "hole_punching",
    "cluster_size_imbalance",
    "density_imbalance",
}


def _resolve_effective_runtime_specs(cfg: DatasetConfig) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    gd = dict(cfg.global_difficulty or {})
    dp = dict(cfg.dataset_perturbations or {})
    migrations: Dict[str, Any] = {"global_to_dataset": {}}

    for key in sorted(DEPRECATED_GLOBAL_TO_DATASET_KEYS):
        if key in gd:
            if key not in dp:
                dp[key] = gd[key]
            migrations["global_to_dataset"][key] = {
                "value": gd[key],
                "applied_from": "global_difficulty",
                "effective_target": "dataset_perturbations",
            }
            gd.pop(key, None)

    if not migrations["global_to_dataset"]:
        migrations = {}
    return gd, dp, migrations


def _resolve_effective_rendering_spec(cfg: DatasetConfig) -> Dict[str, Any]:
    return resolve_rendering_spec(cfg.rendering or {}, cfg.feature_names, bounds=cfg.bounds)


def _apply_global_difficulty(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    gd_spec: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    gd = dict(gd_spec if gd_spec is not None else (cfg.global_difficulty or {}))
    if not gd:
        return X, y, {}

    low = np.asarray(cfg.bounds["low"], dtype=float)
    high = np.asarray(cfg.bounds["high"], dtype=float)
    info: Dict[str, Any] = {}

    global_jitter = float(gd.get("global_jitter_sigma", 0.0))
    if global_jitter > 0:
        X = np.clip(X + rng.normal(0.0, global_jitter, size=X.shape), low, high)
        info["global_jitter_sigma"] = global_jitter

    feature_jitter_spec = gd.get("feature_jitter_sigma", None)
    if feature_jitter_spec is not None:
        sigmas = _as_feature_sigma(feature_jitter_spec, cfg.n_features)
        if np.any(sigmas > 0):
            X = np.clip(X + rng.normal(0.0, sigmas, size=X.shape), low, high)
            info["feature_jitter_sigma"] = sigmas.tolist()

    overlap_jitter = float(gd.get("overlap_jitter_sigma", 0.0))
    if overlap_jitter > 0:
        X = np.clip(X + rng.normal(0.0, overlap_jitter, size=X.shape), low, high)
        info["overlap_jitter_sigma"] = overlap_jitter

    label_flip_rate = float(gd.get("label_flip_rate", 0.0))
    if label_flip_rate > 0 and len(np.unique(y)) > 1:
        flip_mask = rng.random(len(y)) < label_flip_rate
        uniq = np.unique(y)
        flip_count = 0
        for idx in np.where(flip_mask)[0]:
            cand = uniq[uniq != y[idx]]
            if cand.size > 0:
                y[idx] = int(cand[rng.integers(0, cand.size)])
                flip_count += 1
        info["label_flip_rate"] = label_flip_rate
        info["label_flip_count"] = flip_count

    return X, y, info


def _apply_postprocess(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    spec = cfg.postprocess or {}
    if not spec:
        return X, y, {}

    kind = str(spec.get("kind", "none")).lower()
    if kind in {"", "none"}:
        return X, y, {}

    low = np.asarray(cfg.bounds["low"], dtype=float)
    high = np.asarray(cfg.bounds["high"], dtype=float)

    if kind != "pri_time_pipeline":
        raise ValueError(f"Unknown postprocess kind '{kind}'.")

    pri_dim = int(spec.get("pri_dim", 0))
    toa_dim = int(spec.get("toa_dim", 1 if cfg.n_features > 1 else 0))
    pri_scale = float(spec.get("pri_scale", 1.0))
    start_toa = float(spec.get("start_toa", low[toa_dim]))
    jitter_sigma = float(spec.get("toa_jitter_sigma", 0.0))
    sort_output = bool(spec.get("sort_by_toa", True))
    drop_rate = float(spec.get("missing_rate", 0.0))
    duplicate_rate = float(spec.get("duplicate_rate", 0.0))
    duplicate_jitter_sigma = float(spec.get("duplicate_jitter_sigma", 0.0))

    pri = np.maximum(X[:, pri_dim] * pri_scale, 1e-12)
    order = np.arange(len(X))
    if bool(spec.get("sort_by_pri", False)):
        order = np.argsort(pri)
        pri = pri[order]
        X = X[order]
        y = y[order]

    toa = start_toa + np.cumsum(pri)
    if jitter_sigma > 0:
        toa = toa + rng.normal(0.0, jitter_sigma, size=toa.shape)

    X[:, toa_dim] = np.clip(toa, low[toa_dim], high[toa_dim])

    if drop_rate > 0:
        keep = rng.random(len(X)) >= drop_rate
        X = X[keep]
        y = y[keep]

    dup_count = 0
    if duplicate_rate > 0 and len(X) > 0:
        mask = rng.random(len(X)) < duplicate_rate
        dup_count = int(mask.sum())
        if dup_count > 0:
            X_dup = X[mask].copy()
            if duplicate_jitter_sigma > 0:
                X_dup[:, toa_dim] += rng.normal(0.0, duplicate_jitter_sigma, size=dup_count)
            X_dup = np.clip(X_dup, low, high)
            y_dup = y[mask].copy()
            X = np.vstack([X, X_dup])
            y = np.concatenate([y, y_dup])

    if sort_output and len(X) > 0:
        ord2 = np.argsort(X[:, toa_dim])
        X = X[ord2]
        y = y[ord2]

    info = {
        "kind": kind,
        "pri_dim": pri_dim,
        "toa_dim": toa_dim,
        "missing_rate": drop_rate,
        "duplicate_rate": duplicate_rate,
        "duplicate_count": dup_count,
        "toa_jitter_sigma": jitter_sigma,
    }
    return X, y, info


def _apply_cluster_size_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    imbalance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if imbalance <= 0 or len(X) == 0:
        return X, y, {}
    labels = np.unique(y)
    if labels.size <= 1:
        return X, y, {}

    keep_idx = []
    ratios: Dict[int, float] = {}
    min_keep = max(0.12, 1.0 - float(imbalance))
    max_keep = 1.0
    for lab in labels:
        idx = np.where(y == lab)[0]
        rate = float(rng.uniform(min_keep, max_keep))
        ratios[int(lab)] = rate
        if rate >= 0.999:
            keep_idx.append(idx)
        else:
            mask = rng.random(idx.size) < rate
            if np.any(mask):
                keep_idx.append(idx[mask])
    keep = np.concatenate(keep_idx) if keep_idx else np.zeros((0,), dtype=int)
    return X[keep], y[keep], {"cluster_size_imbalance": imbalance, "keep_ratios": ratios}


def _apply_density_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    imbalance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if imbalance <= 0 or len(X) == 0:
        return X, y, {}

    keep_mask = np.ones(len(X), dtype=bool)
    mode_per_label: Dict[int, str] = {}
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if idx.size < 12:
            continue
        Xi = X[idx]
        ctr = Xi.mean(axis=0, keepdims=True)
        d = np.linalg.norm(Xi - ctr, axis=1)
        if np.allclose(d.max(), d.min()):
            continue
        d = (d - d.min()) / max(1e-12, d.max() - d.min())
        mode = "center_dense" if bool(rng.integers(0, 2)) else "edge_dense"
        mode_per_label[int(lab)] = mode
        if mode == "center_dense":
            keep_prob = 1.0 - imbalance * d
        else:
            keep_prob = 1.0 - imbalance * (1.0 - d)
        keep_prob = np.clip(keep_prob, 0.15, 1.0)
        keep_mask[idx] = rng.random(idx.size) < keep_prob

    return X[keep_mask], y[keep_mask], {"density_imbalance": imbalance, "modes": mode_per_label}


def _apply_point_dropout(
    X: np.ndarray,
    y: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if rate <= 0 or len(X) == 0:
        return X, y, {}
    keep = rng.random(len(X)) >= rate
    return X[keep], y[keep], {"point_dropout_rate": rate, "dropped_points": int((~keep).sum())}


def _apply_hole_punching(
    X: np.ndarray,
    y: np.ndarray,
    spec: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(X) == 0 or not spec:
        return X, y, {}

    if isinstance(spec, (int, float)):
        rate = float(spec)
        radius_scale = 0.18
        n_holes = 1
    else:
        rate = float(spec.get("rate", 0.0))
        radius_scale = float(spec.get("radius_scale", 0.18))
        n_holes = int(spec.get("n_holes", 1))
    if rate <= 0:
        return X, y, {}

    keep = np.ones(len(X), dtype=bool)
    total_removed = 0
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if idx.size < 20:
            continue
        Xi = X[idx]
        span = np.maximum(1e-12, Xi.max(axis=0) - Xi.min(axis=0))
        radius = max(1e-12, radius_scale * float(np.linalg.norm(span)))
        lab_keep = np.ones(idx.size, dtype=bool)
        for _ in range(max(1, n_holes)):
            center = Xi[rng.integers(0, idx.size)]
            d = np.linalg.norm(Xi - center, axis=1)
            cand = np.where(d <= radius)[0]
            if cand.size == 0:
                continue
            remove_count = int(np.ceil(rate * cand.size))
            if remove_count <= 0:
                continue
            chosen = cand[np.argsort(d[cand])[:remove_count]]
            lab_keep[chosen] = False
        keep[idx] = lab_keep
        total_removed += int((~lab_keep).sum())

    return X[keep], y[keep], {
        "hole_punching": True,
        "removed_points": total_removed,
        "rate": rate,
        "radius_scale": radius_scale,
        "n_holes": n_holes,
    }


def _apply_bridge_segments(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    spec: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(X) == 0 or not spec:
        return X, y, {}

    if isinstance(spec, (int, float)):
        n_bridges = int(spec)
        points_per_bridge = 80
        sigma = 0.03
    else:
        n_bridges = int(spec.get("n_bridges", 0))
        points_per_bridge = int(spec.get("points_per_bridge", 80))
        sigma = float(spec.get("sigma", 0.03))
    if n_bridges <= 0:
        return X, y, {}

    labels = np.unique(y)
    if labels.size <= 1:
        return X, y, {}
    cents = _cluster_centroids(X, y)
    low = np.asarray(cfg.bounds["low"], dtype=float)
    high = np.asarray(cfg.bounds["high"], dtype=float)

    bridge_parts = []
    pair_info = []
    for _ in range(n_bridges):
        a, b = rng.choice(labels, size=2, replace=False)
        ca, cb = cents[int(a)], cents[int(b)]
        t = np.linspace(0.0, 1.0, max(2, points_per_bridge))
        bridge = (1.0 - t[:, None]) * ca[None, :] + t[:, None] * cb[None, :]
        noise = rng.normal(0.0, sigma, size=bridge.shape)
        bridge = np.clip(bridge + noise, low, high)
        bridge_parts.append(bridge)
        pair_info.append([int(a), int(b)])

    Xb = np.vstack(bridge_parts)
    yb = np.full((len(Xb),), int(cfg.noise_label), dtype=int)
    X = np.vstack([X, Xb])
    y = np.concatenate([y, yb])
    return X, y, {
        "bridge_segments": True,
        "n_bridges": n_bridges,
        "points_per_bridge": points_per_bridge,
        "sigma": sigma,
        "pairs": pair_info,
        "added_points": int(len(Xb)),
    }


def _apply_local_contamination(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    rate: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if rate <= 0 or sigma <= 0 or len(X) == 0:
        return X, y, {}
    n_new = int(np.ceil(rate * len(X)))
    if n_new <= 0:
        return X, y, {}
    anchor_idx = rng.integers(0, len(X), size=n_new)
    anchors = X[anchor_idx]
    low = np.asarray(cfg.bounds["low"], dtype=float)
    high = np.asarray(cfg.bounds["high"], dtype=float)
    Xn = np.clip(anchors + rng.normal(0.0, sigma, size=anchors.shape), low, high)
    yn = np.full((n_new,), int(cfg.noise_label), dtype=int)
    X = np.vstack([X, Xn])
    y = np.concatenate([y, yn])
    return X, y, {
        "local_contamination_rate": rate,
        "local_contamination_sigma": sigma,
        "added_points": n_new,
    }


def _apply_duplicate_points(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    rate: float,
    jitter_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if rate <= 0 or len(X) == 0:
        return X, y, {}
    mask = rng.random(len(X)) < rate
    dup_count = int(mask.sum())
    if dup_count <= 0:
        return X, y, {"duplicate_rate": rate, "duplicate_count": 0, "duplicate_jitter_sigma": jitter_sigma}
    low = np.asarray(cfg.bounds["low"], dtype=float)
    high = np.asarray(cfg.bounds["high"], dtype=float)
    Xdup = X[mask].copy()
    if jitter_sigma > 0:
        Xdup = Xdup + rng.normal(0.0, jitter_sigma, size=Xdup.shape)
    Xdup = np.clip(Xdup, low, high)
    ydup = y[mask].copy()
    X = np.vstack([X, Xdup])
    y = np.concatenate([y, ydup])
    return X, y, {"duplicate_rate": rate, "duplicate_count": dup_count, "duplicate_jitter_sigma": jitter_sigma}


def _apply_anisotropic_jitter(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    sigma_spec: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if sigma_spec is None or len(X) == 0:
        return X, y, {}
    sigmas = _as_feature_sigma(sigma_spec, cfg.n_features)
    if np.all(sigmas <= 0):
        return X, y, {}
    low = np.asarray(cfg.bounds["low"], dtype=float)
    high = np.asarray(cfg.bounds["high"], dtype=float)
    X = np.clip(X + rng.normal(0.0, sigmas, size=X.shape), low, high)
    return X, y, {"anisotropic_jitter_sigma": sigmas.tolist()}


def _apply_dataset_perturbations(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    dp_spec: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    dp = dict(dp_spec if dp_spec is not None else (cfg.dataset_perturbations or {}))
    if not dp:
        return X, y, {}

    info: Dict[str, Any] = {}

    X, y, sub = _apply_cluster_size_imbalance(X, y, float(dp.get("cluster_size_imbalance", 0.0)), rng)
    info.update(sub)

    X, y, sub = _apply_density_imbalance(X, y, float(dp.get("density_imbalance", 0.0)), rng)
    info.update(sub)

    X, y, sub = _apply_point_dropout(X, y, float(dp.get("point_dropout_rate", 0.0)), rng)
    info.update(sub)

    X, y, sub = _apply_hole_punching(X, y, dp.get("hole_punching", 0.0), rng)
    info.update(sub)

    X, y, sub = _apply_bridge_segments(X, y, cfg, dp.get("bridge_segments", 0), rng)
    info.update(sub)

    X, y, sub = _apply_local_contamination(
        X,
        y,
        cfg,
        float(dp.get("local_contamination_rate", 0.0)),
        float(dp.get("local_contamination_sigma", 0.0)),
        rng,
    )
    info.update(sub)

    X, y, sub = _apply_duplicate_points(
        X,
        y,
        cfg,
        float(dp.get("duplicate_rate", 0.0)),
        float(dp.get("duplicate_jitter_sigma", 0.0)),
        rng,
    )
    info.update(sub)

    X, y, sub = _apply_anisotropic_jitter(X, y, cfg, dp.get("anisotropic_jitter_sigma", None), rng)
    info.update(sub)

    info["final_n_points"] = int(len(X))
    label_counts = {int(l): int((y == l).sum()) for l in np.unique(y)}
    info["final_label_counts"] = label_counts
    return X, y, info


def _summarize_pattern_factors(component_summaries: List[Dict[str, Any]], perturb_info: Dict[str, Any]) -> Dict[str, Any]:
    families = [c["type"] for c in component_summaries]
    family_counts: Dict[str, int] = {}
    for fam in families:
        family_counts[fam] = family_counts.get(fam, 0) + 1

    operator_summary: Dict[str, Dict[str, int]] = {}
    for c in component_summaries:
        ops = c.get("operators_used", {}) or {}
        for k, v in ops.items():
            operator_summary.setdefault(k, {})
            operator_summary[k][str(v)] = operator_summary[k].get(str(v), 0) + 1

    active_perturbations = sorted([
        k for k, v in perturb_info.items()
        if k not in {"final_n_points", "final_label_counts"} and v not in ({}, None, False, 0, 0.0)
    ])

    return {
        "shape_family_counts": family_counts,
        "operator_summary": operator_summary,
        "active_dataset_perturbations": active_perturbations,
        "connectivity_mode": "fragmented" if any(k in perturb_info for k in ["point_dropout_rate", "hole_punching"]) else "continuous",
        "density_mode": "imbalanced" if any(k in perturb_info for k in ["density_imbalance", "cluster_size_imbalance"]) else "balanced",
        "contamination_mode": "contaminated" if any(k in perturb_info for k in ["local_contamination_rate", "bridge_segments"]) else "clean",
        "texture_mode": "texture_rich" if any(f in family_counts for f in ["ridge_field", "blob_field", "fractal_dust", "grid_lattice"]) else "shape_only",
    }


def generate_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    master_rng = np.random.default_rng(cfg.seed)

    low = np.array(cfg.bounds["low"], dtype=float)
    high = np.array(cfg.bounds["high"], dtype=float)
    topo = _topological_order(cfg.components)

    comp_params_used: Dict[str, Dict[str, Any]] = {}
    comp_seed_info: Dict[str, Dict[str, int]] = {}

    next_label = 0
    X_parts = []
    y_parts = []
    summaries = []

    for c in topo:
        comp = get_component(c.type)

        if c.type == "noise" and cfg.noise_label_mode == "collapse":
            label = cfg.noise_label
        elif c.label is None:
            label = next_label
            next_label += 1
        else:
            label = int(c.label)
            next_label = max(next_label, label + 1)

        parent_params = {p: comp_params_used[p] for p in c.parents}

        frozen_seed_info = None
        if cfg.replay_mode == "frozen":
            frozen_seed_info = c.params.get("_internal_seeds", None)

        param_seed, param_rng = _spawn_rng(master_rng, None if frozen_seed_info is None else frozen_seed_info.get("param_seed"))
        point_seed, point_rng = _spawn_rng(master_rng, None if frozen_seed_info is None else frozen_seed_info.get("point_seed"))
        noise_seed, noise_rng = _spawn_rng(master_rng, None if frozen_seed_info is None else frozen_seed_info.get("noise_seed"))

        params_used = comp.sample_params(param_rng, cfg.n_features, c.params, parent_params)
        comp_params_used[c.name] = params_used
        comp_seed_info[c.name] = {
            "param_seed": int(param_seed),
            "point_seed": int(point_seed),
            "noise_seed": int(noise_seed),
        }

        Xc = comp.sample_points(
            rng=point_rng,
            n_points=c.n_points,
            n_features=cfg.n_features,
            bounds_low=low,
            bounds_high=high,
            params_used=params_used,
            parent_params=parent_params,
        )

        Xc, operator_runtime = apply_component_operators(
            rng=point_rng,
            X=Xc,
            comp_type=c.type,
            params_used=params_used,
            operators=c.operators,
            bounds_low=low,
            bounds_high=high,
        )

        if c.noise == "gaussian":
            sigma = float(sample_from_spec(noise_rng, c.noise_params.get("sigma", 0.0)))
            if sigma > 0:
                Xc = np.clip(Xc + noise_rng.normal(0.0, sigma, size=Xc.shape), low, high)
            noise_used = {"type": "gaussian", "sigma": sigma}
        elif c.noise == "none":
            noise_used = {"type": "none"}
        else:
            raise ValueError(f"Unknown component noise: {c.noise}")

        yc = np.full((Xc.shape[0],), label, dtype=int)
        X_parts.append(Xc)
        y_parts.append(yc)

        summaries.append(
            {
                "name": c.name,
                "type": c.type,
                "n_points_requested": int(c.n_points),
                "n_points_generated": int(Xc.shape[0]),
                "label": int(label),
                "parents": list(c.parents),
                "params_used": params_used,
                "operators_used": dict((operator_runtime or {}).get("used", c.operators or {})),
                "operator_runtime": dict(operator_runtime or {}),
                "noise_used": noise_used,
                "internal_seeds": comp_seed_info[c.name],
            }
        )

    X = np.vstack(X_parts) if X_parts else np.zeros((0, cfg.n_features), dtype=float)
    y = np.concatenate(y_parts) if y_parts else np.zeros((0,), dtype=int)

    effective_gd, effective_dp, config_migrations = _resolve_effective_runtime_specs(cfg)

    global_seed_spec = None if cfg.replay_mode != "frozen" else (cfg.postprocess or {}).get("_global_seed")
    global_seed, global_rng = _spawn_rng(master_rng, global_seed_spec)

    X, y, perturb_info = _apply_dataset_perturbations(X, y, cfg, global_rng, dp_spec=effective_dp)
    X, y, global_difficulty_info = _apply_global_difficulty(X, y, cfg, global_rng, gd_spec=effective_gd)
    X, y, postprocess_info = _apply_postprocess(X, y, cfg, global_rng)

    pattern_factors = _summarize_pattern_factors(summaries, perturb_info)
    semantic_pattern_factors = compute_semantic_pattern_factors(
        summaries,
        perturb_info
    )
    effective_rendering = _resolve_effective_rendering_spec(cfg)

    meta = {
        "generator": "HYBRID_SCM_V2",
        "name": cfg.name,
        "seed": cfg.seed,
        "n_features": cfg.n_features,
        "feature_names": cfg.feature_names,
        "bounds": {"low": cfg.bounds["low"], "high": cfg.bounds["high"]},
        "params_used": {"components": summaries},
        "noise_label_mode": cfg.noise_label_mode,
        "effective_dataset_perturbations_spec": effective_dp,
        "effective_global_difficulty_spec": effective_gd,
        "effective_postprocess_spec": dict(cfg.postprocess or {}),
        "effective_rendering_spec": effective_rendering,
        "config_migrations": config_migrations,
        "dataset_perturbations_used": perturb_info,
        "global_difficulty": global_difficulty_info,
        "postprocess": postprocess_info,
        "pattern_factors": pattern_factors,
        "pattern_factors_semantic": semantic_pattern_factors,
        "structural_targets": dict((semantic_pattern_factors.get("structural_targets") or {})),
        "rendering": {
            "enabled": bool(effective_rendering.get("enabled", False)),
            "axis_map": effective_rendering.get("axis_map"),
            "render_mode": effective_rendering.get("render_mode"),
            "image_size": effective_rendering.get("image_size"),
            "normalize": effective_rendering.get("normalize"),
            "threshold": effective_rendering.get("threshold"),
        },
        "tags": dict(cfg.tags or {}),
        "internal_global_seed": int(global_seed),
        "run_signature": f"HYBRID_V2|seed={cfg.seed}|components={len(summaries)}|replay={cfg.replay_mode}",
    }
    return X, y, meta, cfg.feature_names

# =========================
# Stage 1-4 extensions
# =========================
from .structural_graph import build_structural_ground_truth

_old_generate_dataset = generate_dataset


def generate_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str]]:  # type: ignore[override]
    X, y, meta, feature_names = _old_generate_dataset(cfg)
    try:
        structural_ground_truth = build_structural_ground_truth(
            X=X,
            y=y,
            component_summaries=list((meta.get("params_used") or {}).get("components") or []),
            feature_names=feature_names,
            bounds=meta.get("bounds") or {"low": cfg.bounds["low"], "high": cfg.bounds["high"]},
            effective_rendering_spec=meta.get("effective_rendering_spec") or {},
            noise_label=int(cfg.noise_label),
        )
    except Exception as exc:
        structural_ground_truth = {"error": str(exc)}

    meta["structural_ground_truth"] = structural_ground_truth
    if "error" not in structural_ground_truth:
        merged_targets = dict(meta.get("structural_targets") or {})
        merged_targets.update({
            "true_components_count": structural_ground_truth.get("true_components_count"),
            "true_holes_count": structural_ground_truth.get("true_holes_count"),
            "true_junction_count": structural_ground_truth.get("true_junction_count"),
            "true_endpoints_count": structural_ground_truth.get("true_endpoints_count"),
            "true_parallel_groups": structural_ground_truth.get("true_parallel_groups"),
            "true_ladder_rungs": structural_ground_truth.get("true_ladder_rungs"),
        })
        meta["structural_targets"] = merged_targets
        sem = dict(meta.get("pattern_factors_semantic") or {})
        sem["structural_targets"] = merged_targets
        sem["topology_ground_truth_available"] = True
        sem["graph_exports_available"] = True
        meta["pattern_factors_semantic"] = sem
        pf = dict(meta.get("pattern_factors") or {})
        pf["topology_mode"] = "ground_truth_exported"
        pf["texture_support_mode"] = "expanded" if any(k in (sem.get("shape_family_counts") or {}) for k in ["oriented_texture_field", "micro_blob_texture", "striped_texture", "checker_texture", "speckle_texture", "anisotropic_grain_field"]) else pf.get("texture_mode", "shape_only")
        meta["pattern_factors"] = pf
    meta["run_signature"] = f"{meta.get('run_signature', 'HYBRID_V2')}|stage14"
    return X, y, meta, feature_names
