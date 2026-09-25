from __future__ import annotations
from typing import Dict, Any, List


def _family_counts(families: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for f in families:
        out[f] = out.get(f, 0) + 1
    return out


def compute_semantic_pattern_factors(
    component_summaries: List[Dict[str, Any]],
    perturb_info: Dict[str, Any],
) -> Dict[str, Any]:

    families = [c["type"] for c in component_summaries]
    family_counts = _family_counts(families)

    operators = {}
    for c in component_summaries:
        for k, v in (c.get("operators_used") or {}).items():
            operators.setdefault(k, []).append(v)

    has_bridges = "bridge_segments" in perturb_info
    has_holes = "hole_punching" in perturb_info
    has_dropout = "point_dropout_rate" in perturb_info
    density_imb = "density_imbalance" in perturb_info
    size_imb = "cluster_size_imbalance" in perturb_info

    if has_bridges:
        connectivity = "connected_with_bridges"
    elif has_dropout:
        connectivity = "fragmented"
    else:
        connectivity = "independent_components"

    hole_structure = "holes_present" if has_holes else "solid"

    curved_types = {"arch_chain", "meander_curve", "ring_annulus", "circle", "ellipse", "spiral"}
    if any(t in curved_types for t in families):
        curvature_mode = "curved"
    else:
        curvature_mode = "linear"

    symmetric_types = {"polygon", "ring_annulus", "grid_lattice", "rectangle_frame", "ellipse"}
    if any(t in symmetric_types for t in families):
        symmetry = "moderate"
    else:
        symmetry = "low"

    thickness = "thin"
    if "thickness_profile" in operators or "base_thickness" in operators:
        thickness = "variable"

    fragmentation = "fragmented" if has_dropout or has_holes else "continuous"
    roughness = "rough" if "anisotropic_jitter_sigma" in perturb_info or "roughness_profile" in operators else "smooth"
    density_profile = "imbalanced" if density_imb or size_imb else "balanced"

    if any(f in families for f in ["grid_lattice", "ridge_field"]):
        fill_mode = "grid"
    elif any(f in families for f in ["polygon", "rectangle_frame"]):
        fill_mode = "polygonal"
    elif any(f in families for f in ["blob_field", "fractal_dust"]):
        fill_mode = "texture_field"
    else:
        fill_mode = "sparse_structure"

    structural_targets = {
        "expected_components_lower_bound": int(len(component_summaries)),
        "expected_holes_lower_bound": int(family_counts.get("ring_annulus", 0) + family_counts.get("rectangle_frame", 0) + family_counts.get("ellipse", 0) * 0),
        "expected_junctions_lower_bound": int(family_counts.get("junction", 0) + family_counts.get("branching_tree", 0)),
        "expected_bands_lower_bound": int(family_counts.get("band", 0) + family_counts.get("parallel_bands", 0) + family_counts.get("ridge_field", 0)),
        "expected_blob_regions_lower_bound": int(family_counts.get("blob", 0) + family_counts.get("blob_field", 0) + family_counts.get("fractal_dust", 0)),
        "expected_symmetry_axes_lower_bound": int(
            family_counts.get("circle", 0) + family_counts.get("ellipse", 0) + family_counts.get("rectangle_frame", 0) + family_counts.get("grid_lattice", 0)
        ),
        "texture_regime": (
            "texture_rich" if any(f in families for f in ["ridge_field", "blob_field", "fractal_dust", "grid_lattice"]) else "shape_only"
        ),
    }
    if has_holes:
        structural_targets["hole_creation_mode"] = "punched_or_intrinsic"
    if has_bridges:
        structural_targets["component_linking_mode"] = "bridged"

    return {
        "shape_family": families,
        "shape_family_counts": family_counts,
        "connectivity_mode": connectivity,
        "hole_structure": hole_structure,
        "symmetry_level": symmetry,
        "curvature_mode": curvature_mode,
        "thickness_mode": thickness,
        "fragmentation_mode": fragmentation,
        "roughness_mode": roughness,
        "density_profile": density_profile,
        "fill_mode": fill_mode,
        "structural_targets": structural_targets,
    }

# =========================
# Stage 1-4 extensions
# =========================

def compute_semantic_pattern_factors(component_summaries: List[Dict[str, Any]], perturb_info: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    families = [c["type"] for c in component_summaries]
    family_counts = _family_counts(families)

    operators: Dict[str, List[str]] = {}
    for c in component_summaries:
        for k, v in (c.get("operators_used") or {}).items():
            operators.setdefault(k, []).append(str(v))

    has_bridges = "bridge_segments" in perturb_info
    has_holes = "hole_punching" in perturb_info
    has_dropout = "point_dropout_rate" in perturb_info
    density_imb = "density_imbalance" in perturb_info
    size_imb = "cluster_size_imbalance" in perturb_info
    contaminated = any(k in perturb_info for k in ["local_contamination_rate", "duplicate_rate", "duplicate_count"]) or has_bridges

    if has_bridges:
        connectivity = "connected_with_bridges"
    elif has_dropout:
        connectivity = "fragmented"
    else:
        connectivity = "independent_components"

    hole_structure = "holes_present" if has_holes or any(f in families for f in ["ring_annulus", "rectangle_frame"]) else "solid"

    curved_types = {"arch_chain", "meander_curve", "ring_annulus", "circle", "ellipse", "spiral", "parabola"}
    texture_types = {"ridge_field", "blob_field", "fractal_dust", "oriented_texture_field", "micro_blob_texture", "striped_texture", "checker_texture", "speckle_texture", "anisotropic_grain_field"}
    if any(t in curved_types for t in families):
        curvature_mode = "curved"
    elif any("curvature_profile" in (c.get("operators_used") or {}) and (c.get("operators_used") or {}).get("curvature_profile") not in {"none", "flatten"} for c in component_summaries):
        curvature_mode = "operator_curved"
    else:
        curvature_mode = "linear"

    symmetry_score = 0
    symmetry_score += sum(family_counts.get(k, 0) for k in ["circle", "ellipse", "rectangle_frame", "grid_lattice", "checker_texture", "polygon"])
    symmetry_score += sum(1 for vals in operators.get("symmetry_profile", []) if vals not in {"none", "weak"})
    symmetry = "high" if symmetry_score >= 3 else ("moderate" if symmetry_score >= 1 else "low")

    periodicity_score = sum(family_counts.get(k, 0) for k in ["grid_lattice", "checker_texture", "striped_texture", "parallel_bands", "ladder", "oriented_texture_field"])
    periodicity_score += sum(1 for vals in operators.get("periodicity_profile", []) if vals not in {"none"})
    periodicity_mode = "strong" if periodicity_score >= 3 else ("moderate" if periodicity_score >= 1 else "weak")

    thickness = "variable" if (
        "thickness_profile" in operators or
        any("thickness_modulation_profile" in (c.get("operators_used") or {}) and (c.get("operators_used") or {}).get("thickness_modulation_profile") not in {"none"} for c in component_summaries)
    ) else "thin"

    fragmentation = "fragmented" if has_dropout or has_holes or any(v not in {"none"} for v in operators.get("fragmentation_profile", [])) else "continuous"
    roughness = "rough" if "anisotropic_jitter_sigma" in perturb_info or any(v not in {"smooth", "none"} for v in operators.get("roughness_profile", [])) else "smooth"
    density_profile = "imbalanced" if density_imb or size_imb else "balanced"

    if any(f in families for f in ["checker_texture", "grid_lattice"]):
        fill_mode = "grid"
    elif any(f in families for f in ["polygon", "rectangle_frame"]):
        fill_mode = "polygonal"
    elif any(f in families for f in texture_types):
        fill_mode = "texture_field"
    else:
        fill_mode = "sparse_structure"

    texture_regime = "texture_rich" if any(f in families for f in texture_types) or any(v not in {"none", "smooth"} for v in operators.get("texture_profile", [])) else "shape_only"

    structural_targets = {
        "expected_components_lower_bound": int(len(component_summaries)),
        "expected_holes_lower_bound": int(family_counts.get("ring_annulus", 0) + family_counts.get("rectangle_frame", 0)),
        "expected_junctions_lower_bound": int(family_counts.get("junction", 0) + family_counts.get("branching_tree", 0)),
        "expected_bands_lower_bound": int(family_counts.get("band", 0) + family_counts.get("parallel_bands", 0) + family_counts.get("ridge_field", 0) + family_counts.get("striped_texture", 0)),
        "expected_blob_regions_lower_bound": int(family_counts.get("blob", 0) + family_counts.get("blob_field", 0) + family_counts.get("fractal_dust", 0) + family_counts.get("micro_blob_texture", 0)),
        "expected_symmetry_axes_lower_bound": int(sum(family_counts.get(k, 0) for k in ["circle", "ellipse", "rectangle_frame", "grid_lattice", "checker_texture"])),
        "expected_periodic_structures_lower_bound": int(sum(family_counts.get(k, 0) for k in ["grid_lattice", "checker_texture", "striped_texture", "parallel_bands", "ladder", "oriented_texture_field"])),
        "expected_texture_regions_lower_bound": int(sum(family_counts.get(k, 0) for k in texture_types)),
        "texture_regime": texture_regime,
    }
    if has_holes:
        structural_targets["hole_creation_mode"] = "punched_or_intrinsic"
    if has_bridges:
        structural_targets["component_linking_mode"] = "bridged"

    return {
        "shape_family": families,
        "shape_family_counts": family_counts,
        "connectivity_mode": connectivity,
        "hole_structure": hole_structure,
        "symmetry_level": symmetry,
        "periodicity_level": periodicity_mode,
        "curvature_mode": curvature_mode,
        "thickness_mode": thickness,
        "fragmentation_mode": fragmentation,
        "roughness_mode": roughness,
        "density_profile": density_profile,
        "fill_mode": fill_mode,
        "texture_mode": texture_regime,
        "contamination_mode": "contaminated" if contaminated else "clean",
        "operator_histograms": {k: {vv: vals.count(vv) for vv in sorted(set(vals))} for k, vals in operators.items()},
        "structural_targets": structural_targets,
    }
