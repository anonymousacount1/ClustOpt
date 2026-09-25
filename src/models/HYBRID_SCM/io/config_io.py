from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class ComponentConfig:
    name: str
    type: str
    n_points: int
    parents: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    operators: Dict[str, Any] = field(default_factory=dict)
    noise: str = "gaussian"
    noise_params: Dict[str, Any] = field(default_factory=dict)
    label: Optional[int] = None


@dataclass
class DatasetConfig:
    name: str
    seed: int
    n_features: int
    feature_names: List[str]
    bounds: Dict[str, List[float]]
    components: List[ComponentConfig]

    noise_label: int = -1
    noise_label_mode: str = "collapse"   # collapse | preserve
    replay_mode: str = "fresh"           # fresh | frozen
    strict_validation: bool = True
    postprocess: Dict[str, Any] = field(default_factory=dict)
    global_difficulty: Dict[str, Any] = field(default_factory=dict)
    rendering: Dict[str, Any] = field(default_factory=dict)
    dataset_perturbations: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BundleConfig:
    bundle_name: str
    datasets: List[DatasetConfig]


VALID_COMPONENT_TYPES: Set[str] = {
    "band",
    "blob",
    "noise",
    "curve",
    "ladder",
    "parabola",
    "circle",
    "parallel_bands",
    "arch_chain",
    "piecewise_polyline",
    "ring_annulus",
    "polygon",
    "grid_lattice",
    "branching_tree",
    "meander_curve",
    "ellipse",
    "rectangle_frame",
    "junction",
    "spiral",
    "ridge_field",
    "blob_field",
    "fractal_dust",
}

ALLOWED_COMPONENT_KEYS: Dict[str, Set[str]] = {
    "band": {
        "dims", "slope", "intercept", "x_low", "x_high", "intervals", "interval_generation",
        "thickness", "irrelevant", "other_sigma",
        "x_sampling", "x_bins_per_interval", "x_jitter_sigma", "y_quantization_step", "y_quantization_origin",
    },
    "ladder": {
        "dims", "slope", "base_intercept", "n_rungs", "rung_spacing", "segment_x_len",
        "x_center_low", "x_center_high", "thickness", "irrelevant", "other_sigma",
    },
    "parallel_bands": {
        "dims", "slope", "intercept0", "n_bands", "spacing", "x_low", "x_high",
        "intervals", "interval_generation", "thickness", "irrelevant", "other_sigma",
        "x_sampling", "x_bins_per_interval", "x_jitter_sigma", "band_allocation", "band_probs",
        "y_quantization_step", "y_quantization_origin",
    },
    "parabola": {
        "dims", "a", "x0", "b", "x_low", "x_high", "intervals", "interval_generation",
        "thickness", "irrelevant", "other_sigma",
    },
    "arch_chain": {
        "dims", "x_low", "x_high", "intervals", "interval_generation",
        "y_base", "mode", "amp_min", "amp_max", "amp_schedule",
        "grow_power", "grow_floor", "hump_power",
        "sharpness_min", "sharpness_max",
        "thickness", "thickness_growth", "thickness_schedule",
        "flat_tail_frac", "cap_to_bounds", "bounds_margin",
        "irrelevant", "other_sigma",
        "base", "trend_slope", "segments",
    },
    "circle": {
        "dims", "cx", "cy", "r", "angle_low", "angle_high", "thickness", "irrelevant", "other_sigma",
    },
    "blob": {"mu", "sigma"},
    "noise": {
        "mode", "low", "high", "roi", "strength", "min_abs_offset", "max_abs_offset",
        "x_low", "x_high", "side", "offset_min", "offset_max", "y_jitter_sigma",
        "x_source", "x_expand", "intervals",
    },
    "curve": set(),
    "piecewise_polyline": {
        "dims", "vertices", "n_segments", "segment_lengths", "angles", "corner_rounding",
        "base_thickness", "irrelevant", "other_sigma",
    },
    "ring_annulus": {
        "dims", "cx", "cy", "inner_radius", "outer_radius", "angle_low", "angle_high",
        "ellipse_ratio", "hole_offset", "rotation", "irrelevant", "other_sigma",
    },
    "polygon": {
        "dims", "n_sides", "radius", "rotation", "corner_rounding", "boundary_thickness",
        "fill_mode", "irrelevant", "other_sigma",
    },
    "grid_lattice": {
        "dims", "n_vertical", "n_horizontal", "spacing_x", "spacing_y", "line_thickness",
        "jitter", "missing_lines_prob", "irrelevant", "other_sigma",
    },
    "branching_tree": {
        "dims", "root", "branching_factor", "depth", "angle_spread", "segment_length",
        "length_decay", "segment_thickness", "pruning_probability", "angle_jitter", "irrelevant", "other_sigma",
    },
    "meander_curve": {
        "dims", "x_low", "x_high", "y_center", "amplitude", "frequency", "phase",
        "amplitude_drift", "frequency_drift", "n_anchors", "anchor_jitter", "base_thickness",
        "irrelevant", "other_sigma",
    },
    "ellipse": {
        "dims", "cx", "cy", "rx", "ry", "rotation", "angle_low", "angle_high", "thickness", "fill", "irrelevant", "other_sigma",
    },
    "rectangle_frame": {
        "dims", "cx", "cy", "width", "height", "rotation", "frame_thickness", "fill", "irrelevant", "other_sigma",
    },
    "junction": {
        "dims", "cx", "cy", "arm_lengths", "angles", "thickness", "irrelevant", "other_sigma",
    },
    "spiral": {
        "dims", "cx", "cy", "r_start", "r_end", "turns", "angle_offset", "thickness", "irrelevant", "other_sigma",
    },
    "ridge_field": {
        "dims", "cx", "cy", "width", "height", "n_ridges", "orientation", "spacing", "ridge_sigma", "jitter", "irrelevant", "other_sigma",
    },
    "blob_field": {
        "dims", "cx", "cy", "width", "height", "n_blobs", "blob_sigma_min", "blob_sigma_max", "strength_jitter", "irrelevant", "other_sigma",
    },
    "fractal_dust": {
        "dims", "cx", "cy", "base_scale", "depth", "children_per_level", "decay", "jitter", "irrelevant", "other_sigma",
    },
}

ALLOWED_NOISE_TYPES = {"gaussian", "none"}
ALLOWED_TOP_LEVEL_KEYS = {
    "name", "seed", "n_features", "feature_names", "bounds", "components",
    "noise_label", "noise_label_mode", "replay_mode", "strict_validation",
    "postprocess", "global_difficulty", "rendering", "dataset_perturbations", "tags",
}

VALID_OPERATOR_KEYS = {
    "sampling_profile",
    "thickness_profile",
    "fragmentation_profile",
    "roughness_profile",
    "fill_mode",
}

VALID_SAMPLING_PROFILES = {"uniform", "edge_heavy", "center_heavy", "bursty", "periodic_sparse"}
VALID_THICKNESS_PROFILES = {"constant", "linear_up", "linear_down", "center_bulge", "bimodal"}
VALID_FRAGMENTATION_PROFILES = {"none", "random_gaps", "regular_gaps", "dropout_windows", "alternating_presence"}
VALID_ROUGHNESS_PROFILES = {"smooth", "gaussian_edge", "periodic_wobble", "jagged"}
VALID_FILL_MODES = {"native", "boundary_only", "ribbon", "shell", "filled"}

VALID_DATASET_PERTURBATION_KEYS = {
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

VALID_RENDERING_KEYS = {
    "enabled",
    "image_size",
    "render_mode",
    "sigma_px",
    "threshold",
    "normalize",
    "morph_open",
    "morph_close",
    "axis_map",
    "per_cluster_masks",
    "axis_limits",
    "binary_blur_sigma",
}

VALID_GLOBAL_DIFFICULTY_KEYS = {
    "overlap_jitter_sigma",
    "label_flip_rate",
    "global_jitter_sigma",
    "feature_jitter_sigma",
}

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

VALID_POSTPROCESS_KINDS = {"none", "pri_time_pipeline"}
VALID_POSTPROCESS_COMMON_KEYS = {"kind", "_global_seed"}
VALID_PRI_TIME_PIPELINE_KEYS = VALID_POSTPROCESS_COMMON_KEYS | {
    "pri_dim", "toa_dim", "pri_scale", "start_toa", "toa_jitter_sigma",
    "sort_by_toa", "missing_rate", "duplicate_rate", "duplicate_jitter_sigma",
    "sort_by_pri",
}


def _require_keys(d: Dict[str, Any], keys: List[str], where: str) -> None:
    for k in keys:
        if k not in d:
            raise KeyError(f"Missing key '{k}' in {where}.")


def _build_component(c: Dict[str, Any], where: str) -> ComponentConfig:
    _require_keys(c, ["name", "type", "n_points"], where)
    return ComponentConfig(
        name=c["name"],
        type=c["type"],
        n_points=int(c["n_points"]),
        parents=list(c.get("parents", [])),
        params=dict(c.get("params", {})),
        operators=dict(c.get("operators", {})),
        noise=c.get("noise", "gaussian"),
        noise_params=dict(c.get("noise_params", {})),
        label=c.get("label", None),
    )


def load_dataset_config(path: Path) -> DatasetConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(data, ["name", "seed", "n_features", "feature_names", "bounds", "components"], str(path))
    comps = [_build_component(c, f"component in {path}") for c in data["components"]]
    cfg = DatasetConfig(
        name=str(data["name"]),
        seed=int(data["seed"]),
        n_features=int(data["n_features"]),
        feature_names=list(data["feature_names"]),
        bounds=dict(data["bounds"]),
        components=comps,
        noise_label=int(data.get("noise_label", -1)),
        noise_label_mode=str(data.get("noise_label_mode", "collapse")),
        replay_mode=str(data.get("replay_mode", "fresh")),
        strict_validation=bool(data.get("strict_validation", True)),
        postprocess=dict(data.get("postprocess", {})),
        global_difficulty=dict(data.get("global_difficulty", {})),
        rendering=dict(data.get("rendering", {})),
        dataset_perturbations=dict(data.get("dataset_perturbations", {})),
        tags=dict(data.get("tags", {})),
    )
    _validate_dataset(cfg, raw_data=data)
    return cfg


def load_bundle_config(path: Path) -> BundleConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(data, ["bundle_name", "datasets"], str(path))

    datasets: List[DatasetConfig] = []
    for item in data["datasets"]:
        if isinstance(item, str):
            datasets.append(load_dataset_config(Path(item)))
        elif isinstance(item, dict):
            datasets.append(_load_dataset_inline(item, path))
        else:
            raise TypeError("bundle.datasets items must be either a path string or inline dict.")
    return BundleConfig(bundle_name=str(data["bundle_name"]), datasets=datasets)


def _load_dataset_inline(d: Dict[str, Any], bundle_path: Path) -> DatasetConfig:
    _require_keys(d, ["name", "seed", "n_features", "feature_names", "bounds", "components"], f"inline dataset in {bundle_path}")
    comps = [_build_component(c, f"component in inline dataset in {bundle_path}") for c in d["components"]]
    cfg = DatasetConfig(
        name=str(d["name"]),
        seed=int(d["seed"]),
        n_features=int(d["n_features"]),
        feature_names=list(d["feature_names"]),
        bounds=dict(d["bounds"]),
        components=comps,
        noise_label=int(d.get("noise_label", -1)),
        noise_label_mode=str(d.get("noise_label_mode", "collapse")),
        replay_mode=str(d.get("replay_mode", "fresh")),
        strict_validation=bool(d.get("strict_validation", True)),
        postprocess=dict(d.get("postprocess", {})),
        global_difficulty=dict(d.get("global_difficulty", {})),
        rendering=dict(d.get("rendering", {})),
        dataset_perturbations=dict(d.get("dataset_perturbations", {})),
        tags=dict(d.get("tags", {})),
    )
    _validate_dataset(cfg, raw_data=d)
    return cfg


def load_replay_config(path: Path) -> DatasetConfig:
    return load_dataset_config(path)


def _check_unknown_top_keys(raw_data: Dict[str, Any]) -> None:
    extra = set(raw_data.keys()) - ALLOWED_TOP_LEVEL_KEYS
    if extra:
        raise ValueError(f"Unknown top-level dataset keys: {sorted(extra)}")


def _check_unknown_component_params(cfg: DatasetConfig) -> None:
    for c in cfg.components:
        allowed = ALLOWED_COMPONENT_KEYS.get(c.type)
        if allowed is None:
            continue
        extra = set((c.params or {}).keys()) - allowed
        if extra:
            raise ValueError(f"Component '{c.name}' of type '{c.type}' has unsupported params: {sorted(extra)}")


def _validate_operators(cfg: DatasetConfig) -> None:
    for c in cfg.components:
        ops = c.operators or {}
        extra = set(ops.keys()) - VALID_OPERATOR_KEYS
        if extra:
            raise ValueError(f"Component '{c.name}' has unsupported operators: {sorted(extra)}")

        if "sampling_profile" in ops and ops["sampling_profile"] not in VALID_SAMPLING_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid sampling_profile '{ops['sampling_profile']}'")
        if "thickness_profile" in ops and ops["thickness_profile"] not in VALID_THICKNESS_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid thickness_profile '{ops['thickness_profile']}'")
        if "fragmentation_profile" in ops and ops["fragmentation_profile"] not in VALID_FRAGMENTATION_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid fragmentation_profile '{ops['fragmentation_profile']}'")
        if "roughness_profile" in ops and ops["roughness_profile"] not in VALID_ROUGHNESS_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid roughness_profile '{ops['roughness_profile']}'")
        if "fill_mode" in ops and ops["fill_mode"] not in VALID_FILL_MODES:
            raise ValueError(f"Component '{c.name}' has invalid fill_mode '{ops['fill_mode']}'")


def _validate_rendering(cfg: DatasetConfig) -> None:
    rendering = cfg.rendering or {}
    extra = set(rendering.keys()) - VALID_RENDERING_KEYS
    if extra:
        raise ValueError(f"Unsupported rendering keys: {sorted(extra)}")

    if not rendering:
        return

    if "enabled" in rendering and not isinstance(rendering["enabled"], bool):
        raise ValueError("rendering.enabled must be boolean.")

    if "image_size" in rendering:
        sz = rendering["image_size"]
        if not isinstance(sz, (list, tuple)) or len(sz) != 2:
            raise ValueError("rendering.image_size must be a 2-item list/tuple.")
        if int(sz[0]) <= 0 or int(sz[1]) <= 0:
            raise ValueError("rendering.image_size values must be positive.")

    if "render_mode" in rendering and str(rendering["render_mode"]).lower() not in {"hard_bin", "gaussian_splat"}:
        raise ValueError("rendering.render_mode must be 'hard_bin' or 'gaussian_splat'.")

    if "normalize" in rendering and str(rendering["normalize"]).lower() not in {"minmax", "percentile", "none"}:
        raise ValueError("rendering.normalize must be 'minmax', 'percentile', or 'none'.")

    if "threshold" in rendering:
        thr = float(rendering["threshold"])
        if not (0.0 <= thr <= 1.0):
            raise ValueError("rendering.threshold must be in [0,1].")

    for key in ["sigma_px", "morph_open", "morph_close", "binary_blur_sigma"]:
        if key in rendering:
            float(rendering[key])

    if "per_cluster_masks" in rendering and not isinstance(rendering["per_cluster_masks"], bool):
        raise ValueError("rendering.per_cluster_masks must be boolean.")

    if "axis_map" in rendering:
        am = rendering["axis_map"]
        if not (am is None or isinstance(am, (list, tuple, dict))):
            raise ValueError("rendering.axis_map must be None, list/tuple, or dict.")

    if "axis_limits" in rendering:
        al = rendering["axis_limits"]
        if not isinstance(al, dict) or "x" not in al or "y" not in al:
            raise ValueError("rendering.axis_limits must be {'x':[low,high], 'y':[low,high]}.")


def _validate_global_difficulty(cfg: DatasetConfig) -> None:
    gd = cfg.global_difficulty or {}
    extra = set(gd.keys()) - (VALID_GLOBAL_DIFFICULTY_KEYS | DEPRECATED_GLOBAL_TO_DATASET_KEYS)
    if extra:
        raise ValueError(f"Unsupported global_difficulty keys: {sorted(extra)}")

    def _check_prob(name: str) -> None:
        if name in gd:
            v = float(gd[name])
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"global_difficulty.{name} must be in [0,1].")

    _check_prob("label_flip_rate")


def _validate_postprocess(cfg: DatasetConfig) -> None:
    spec = cfg.postprocess or {}
    if not spec:
        return

    kind = str(spec.get("kind", "none")).lower()
    if kind not in VALID_POSTPROCESS_KINDS:
        raise ValueError(f"Unsupported postprocess kind '{kind}'.")

    if kind in {"", "none"}:
        allowed = VALID_POSTPROCESS_COMMON_KEYS
    elif kind == "pri_time_pipeline":
        allowed = VALID_PRI_TIME_PIPELINE_KEYS
    else:
        allowed = VALID_POSTPROCESS_COMMON_KEYS

    extra = set(spec.keys()) - allowed
    if extra:
        raise ValueError(f"Unsupported postprocess keys for kind '{kind}': {sorted(extra)}")


def _validate_dataset_perturbations(cfg: DatasetConfig) -> None:
    dp = cfg.dataset_perturbations or {}
    extra = set(dp.keys()) - VALID_DATASET_PERTURBATION_KEYS
    if extra:
        raise ValueError(f"Unsupported dataset_perturbations keys: {sorted(extra)}")

    def _check_prob(name: str) -> None:
        if name in dp:
            v = float(dp[name])
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"dataset_perturbations.{name} must be in [0,1].")

    for key in [
        "point_dropout_rate", "duplicate_rate", "local_contamination_rate",
        "cluster_size_imbalance", "density_imbalance",
    ]:
        _check_prob(key)

    if "bridge_segments" in dp:
        v = dp["bridge_segments"]
        if not isinstance(v, (int, float, dict)):
            raise ValueError("dataset_perturbations.bridge_segments must be number or dict.")
    if "hole_punching" in dp:
        v = dp["hole_punching"]
        if not isinstance(v, (int, float, dict)):
            raise ValueError("dataset_perturbations.hole_punching must be number or dict.")


def _validate_cycle_free(components: List[ComponentConfig]) -> None:
    name_to_comp = {c.name: c for c in components}
    temp_mark: Set[str] = set()
    perm_mark: Set[str] = set()

    def visit(name: str, stack: List[str]) -> None:
        if name in perm_mark:
            return
        if name in temp_mark:
            raise ValueError(f"Cycle detected in component graph: {' -> '.join(stack + [name])}")
        temp_mark.add(name)
        comp = name_to_comp[name]
        for p in comp.parents:
            visit(p, stack + [name])
        temp_mark.remove(name)
        perm_mark.add(name)

    for c in components:
        visit(c.name, [])


def _validate_dataset(cfg: DatasetConfig, raw_data: Optional[Dict[str, Any]] = None) -> None:
    if raw_data is not None and cfg.strict_validation:
        _check_unknown_top_keys(raw_data)

    if cfg.n_features <= 0:
        raise ValueError("n_features must be positive.")
    if len(cfg.feature_names) != cfg.n_features:
        raise ValueError("feature_names length must equal n_features.")
    if "low" not in cfg.bounds or "high" not in cfg.bounds:
        raise ValueError("bounds must contain 'low' and 'high'.")
    if len(cfg.bounds["low"]) != cfg.n_features or len(cfg.bounds["high"]) != cfg.n_features:
        raise ValueError("bounds low/high must have length n_features.")

    names = [c.name for c in cfg.components]
    if len(names) != len(set(names)):
        raise ValueError("Component names must be unique.")
    name_set = set(names)
    for c in cfg.components:
        if c.type not in VALID_COMPONENT_TYPES:
            raise ValueError(f"Unknown component type '{c.type}'. Supported: {sorted(VALID_COMPONENT_TYPES)}")
        if c.noise not in ALLOWED_NOISE_TYPES:
            raise ValueError(f"Unknown component noise '{c.noise}'. Supported: {sorted(ALLOWED_NOISE_TYPES)}")
        for p in c.parents:
            if p not in name_set:
                raise ValueError(f"Component '{c.name}' refers to unknown parent '{p}'.")
        if c.n_points <= 0:
            raise ValueError(f"Component '{c.name}' must have n_points > 0.")

    if cfg.noise_label_mode not in {"collapse", "preserve"}:
        raise ValueError("noise_label_mode must be 'collapse' or 'preserve'.")
    if cfg.replay_mode not in {"fresh", "frozen"}:
        raise ValueError("replay_mode must be 'fresh' or 'frozen'.")

    _validate_cycle_free(cfg.components)
    _validate_operators(cfg)
    _validate_rendering(cfg)
    _validate_global_difficulty(cfg)
    _validate_dataset_perturbations(cfg)
    _validate_postprocess(cfg)

    if cfg.strict_validation:
        _check_unknown_component_params(cfg)

# =========================
# Stage 1-4 extensions
# =========================
VALID_COMPONENT_TYPES.update({
    "oriented_texture_field",
    "micro_blob_texture",
    "striped_texture",
    "checker_texture",
    "speckle_texture",
    "anisotropic_grain_field",
})

ALLOWED_COMPONENT_KEYS.update({
    "oriented_texture_field": {
        "dims", "cx", "cy", "width", "height", "orientation", "n_ridges", "ridge_spacing",
        "ridge_sigma", "cross_sigma", "phase_jitter", "strength_jitter", "irrelevant", "other_sigma",
    },
    "micro_blob_texture": {
        "dims", "cx", "cy", "width", "height", "n_micro_blobs", "blob_sigma_min", "blob_sigma_max",
        "strength_jitter", "clustered_centers", "irrelevant", "other_sigma",
    },
    "striped_texture": {
        "dims", "cx", "cy", "width", "height", "orientation", "n_stripes", "stripe_spacing",
        "stripe_sigma", "phase_offset", "duty_cycle", "irrelevant", "other_sigma",
    },
    "checker_texture": {
        "dims", "cx", "cy", "width", "height", "cell_size_x", "cell_size_y", "orientation",
        "on_probability", "jitter", "irrelevant", "other_sigma",
    },
    "speckle_texture": {
        "dims", "cx", "cy", "width", "height", "speckle_rate", "clustered_fraction", "cluster_sigma",
        "irrelevant", "other_sigma",
    },
    "anisotropic_grain_field": {
        "dims", "cx", "cy", "width", "height", "n_grains", "grain_sigma_long", "grain_sigma_short",
        "orientation", "orientation_jitter", "strength_jitter", "irrelevant", "other_sigma",
    },
})

VALID_OPERATOR_KEYS.update({
    "symmetry_profile",
    "periodicity_profile",
    "junction_profile",
    "curvature_profile",
    "thickness_modulation_profile",
    "topology_profile",
    "occlusion_profile",
    "texture_profile",
})

VALID_SYMMETRY_PROFILES = {"none", "weak", "axial_x", "axial_y", "central", "rotational_2", "rotational_4"}
VALID_PERIODICITY_PROFILES = {"none", "uniform_periodic", "jittered_periodic", "locally_periodic", "broken_periodic"}
VALID_JUNCTION_PROFILES = {"none", "weak", "strong", "branch_biased", "center_dense"}
VALID_CURVATURE_PROFILES = {"none", "flatten", "amplify", "sinusoidal_warp", "piecewise_kink"}
VALID_THICKNESS_MODULATION_PROFILES = {"none", "monotone_increase", "monotone_decrease", "center_heavy", "bimodal", "strong_bimodal"}
VALID_TOPOLOGY_PROFILES = {"none", "connected", "disconnected", "looped", "bridged", "tree_like", "fragmented"}
VALID_OCCLUSION_PROFILES = {"none", "left_occluded", "right_occluded", "center_occluded", "random_patch"}
VALID_TEXTURE_PROFILES = {"none", "smooth", "striped", "granular", "speckled", "oriented", "multiscale"}

VALID_RENDERING_KEYS.update({
    "extra_views",
    "edge_sigma",
    "skeletonize",
    "distance_transform",
    "local_thickness",
    "gradient_orientation",
    "cluster_id_raster",
})


def _validate_operators(cfg: DatasetConfig) -> None:  # type: ignore[override]
    for c in cfg.components:
        ops = c.operators or {}
        extra = set(ops.keys()) - VALID_OPERATOR_KEYS
        if extra:
            raise ValueError(f"Component '{c.name}' has unsupported operators: {sorted(extra)}")

        if "sampling_profile" in ops and ops["sampling_profile"] not in VALID_SAMPLING_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid sampling_profile '{ops['sampling_profile']}'")
        if "thickness_profile" in ops and ops["thickness_profile"] not in VALID_THICKNESS_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid thickness_profile '{ops['thickness_profile']}'")
        if "fragmentation_profile" in ops and ops["fragmentation_profile"] not in VALID_FRAGMENTATION_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid fragmentation_profile '{ops['fragmentation_profile']}'")
        if "roughness_profile" in ops and ops["roughness_profile"] not in VALID_ROUGHNESS_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid roughness_profile '{ops['roughness_profile']}'")
        if "fill_mode" in ops and ops["fill_mode"] not in VALID_FILL_MODES:
            raise ValueError(f"Component '{c.name}' has invalid fill_mode '{ops['fill_mode']}'")
        if "symmetry_profile" in ops and ops["symmetry_profile"] not in VALID_SYMMETRY_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid symmetry_profile '{ops['symmetry_profile']}'")
        if "periodicity_profile" in ops and ops["periodicity_profile"] not in VALID_PERIODICITY_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid periodicity_profile '{ops['periodicity_profile']}'")
        if "junction_profile" in ops and ops["junction_profile"] not in VALID_JUNCTION_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid junction_profile '{ops['junction_profile']}'")
        if "curvature_profile" in ops and ops["curvature_profile"] not in VALID_CURVATURE_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid curvature_profile '{ops['curvature_profile']}'")
        if "thickness_modulation_profile" in ops and ops["thickness_modulation_profile"] not in VALID_THICKNESS_MODULATION_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid thickness_modulation_profile '{ops['thickness_modulation_profile']}'")
        if "topology_profile" in ops and ops["topology_profile"] not in VALID_TOPOLOGY_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid topology_profile '{ops['topology_profile']}'")
        if "occlusion_profile" in ops and ops["occlusion_profile"] not in VALID_OCCLUSION_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid occlusion_profile '{ops['occlusion_profile']}'")
        if "texture_profile" in ops and ops["texture_profile"] not in VALID_TEXTURE_PROFILES:
            raise ValueError(f"Component '{c.name}' has invalid texture_profile '{ops['texture_profile']}'")



def _validate_rendering(cfg: DatasetConfig) -> None:  # type: ignore[override]
    rendering = cfg.rendering or {}
    extra = set(rendering.keys()) - VALID_RENDERING_KEYS
    if extra:
        raise ValueError(f"Unsupported rendering keys: {sorted(extra)}")

    if not rendering:
        return

    if "enabled" in rendering and not isinstance(rendering["enabled"], bool):
        raise ValueError("rendering.enabled must be boolean.")

    if "image_size" in rendering:
        sz = rendering["image_size"]
        if not isinstance(sz, (list, tuple)) or len(sz) != 2:
            raise ValueError("rendering.image_size must be a 2-item list/tuple.")
        if int(sz[0]) <= 0 or int(sz[1]) <= 0:
            raise ValueError("rendering.image_size values must be positive.")

    if "render_mode" in rendering and str(rendering["render_mode"]).lower() not in {"hard_bin", "gaussian_splat"}:
        raise ValueError("rendering.render_mode must be 'hard_bin' or 'gaussian_splat'.")

    if "normalize" in rendering and str(rendering["normalize"]).lower() not in {"minmax", "percentile", "none"}:
        raise ValueError("rendering.normalize must be 'minmax', 'percentile', or 'none'.")

    if "threshold" in rendering:
        thr = float(rendering["threshold"])
        if not (0.0 <= thr <= 1.0):
            raise ValueError("rendering.threshold must be in [0,1].")

    for key in ["sigma_px", "morph_open", "morph_close", "binary_blur_sigma", "edge_sigma"]:
        if key in rendering:
            float(rendering[key])

    for key in ["per_cluster_masks", "skeletonize", "distance_transform", "local_thickness", "gradient_orientation", "cluster_id_raster"]:
        if key in rendering and not isinstance(rendering[key], bool):
            raise ValueError(f"rendering.{key} must be boolean.")

    if "extra_views" in rendering:
        ev = rendering["extra_views"]
        if not isinstance(ev, dict):
            raise ValueError("rendering.extra_views must be a dict of boolean flags.")

    if "axis_map" in rendering:
        am = rendering["axis_map"]
        if not (am is None or isinstance(am, (list, tuple, dict))):
            raise ValueError("rendering.axis_map must be None, list/tuple, or dict.")

    if "axis_limits" in rendering:
        al = rendering["axis_limits"]
        if not isinstance(al, dict) or "x" not in al or "y" not in al:
            raise ValueError("rendering.axis_limits must be {'x':[low,high], 'y':[low,high]}.")
