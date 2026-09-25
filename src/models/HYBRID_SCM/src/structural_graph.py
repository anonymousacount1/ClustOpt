from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt, label, binary_erosion, convolve
except Exception:  # pragma: no cover
    distance_transform_edt = None
    label = None
    binary_erosion = None
    convolve = None

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover
    skeletonize = None

from .rasterizer import render_dataset


_CURVED = {"arch_chain", "meander_curve", "ring_annulus", "circle", "ellipse", "spiral", "parabola"}
_LINEAR = {"band", "parallel_bands", "ladder", "piecewise_polyline", "grid_lattice", "rectangle_frame", "junction", "branching_tree", "ridge_field", "striped_texture", "checker_texture", "oriented_texture_field"}
_TEXTURE = {"ridge_field", "blob_field", "fractal_dust", "oriented_texture_field", "micro_blob_texture", "striped_texture", "checker_texture", "speckle_texture", "anisotropic_grain_field"}


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _estimate_intrinsic_targets(component_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    true_holes = 0
    true_junctions = 0
    true_endpoints = 0
    true_parallel_groups = 0
    true_ladder_rungs = 0
    true_periodic_components = 0
    symmetry_axes: List[Dict[str, Any]] = []
    blob_centers: List[List[float]] = []
    ridge_orientations: List[float] = []
    contour_nodes: List[Dict[str, Any]] = []
    contour_edges: List[Dict[str, Any]] = []

    for comp in component_summaries:
        ctype = comp.get("type", "")
        params = comp.get("params_used", {}) or {}
        cname = comp.get("name", ctype)

        if ctype == "ring_annulus":
            true_holes += 1
            symmetry_axes.extend([
                {"component": cname, "kind": "major_minor", "axis": "x"},
                {"component": cname, "kind": "major_minor", "axis": "y"},
            ])
        elif ctype == "rectangle_frame":
            true_holes += 1
            symmetry_axes.extend([
                {"component": cname, "kind": "reflection", "axis": "x"},
                {"component": cname, "kind": "reflection", "axis": "y"},
            ])
        elif ctype in {"circle", "ellipse", "polygon", "grid_lattice", "checker_texture"}:
            symmetry_axes.extend([
                {"component": cname, "kind": "reflection", "axis": "x"},
                {"component": cname, "kind": "reflection", "axis": "y"},
            ])

        if ctype == "junction":
            true_junctions += 1
            true_endpoints += len(params.get("angles", []) or [])
        elif ctype == "branching_tree":
            edges = params.get("edges", []) or []
            verts = params.get("vertices", []) or []
            deg = {idx: 0 for idx in range(len(verts))}
            for a, b in edges:
                deg[int(a)] = deg.get(int(a), 0) + 1
                deg[int(b)] = deg.get(int(b), 0) + 1
            true_junctions += sum(1 for d in deg.values() if d >= 3)
            true_endpoints += sum(1 for d in deg.values() if d == 1)
        elif ctype in {"band", "parabola", "piecewise_polyline", "meander_curve", "spiral"}:
            true_endpoints += 2

        if ctype == "parallel_bands":
            true_parallel_groups += max(1, _safe_int(params.get("n_bands", 1)))
        elif ctype == "ridge_field":
            true_parallel_groups += max(1, _safe_int(params.get("n_ridges", 1)))
            ridge_orientations.extend([float(params.get("orientation", 0.0))] * max(1, _safe_int(params.get("n_ridges", 1))))
        elif ctype == "striped_texture":
            true_parallel_groups += max(1, _safe_int(params.get("n_stripes", 1)))
            ridge_orientations.extend([float(params.get("orientation", 0.0))] * max(1, _safe_int(params.get("n_stripes", 1))))

        if ctype == "ladder":
            true_ladder_rungs += max(1, _safe_int(params.get("n_rungs", 1)))
            true_parallel_groups += 2

        if ctype in {"checker_texture", "grid_lattice", "striped_texture", "oriented_texture_field"}:
            true_periodic_components += 1

        if ctype == "blob_field":
            blob_centers.extend([[float(x), float(y)] for x, y in (params.get("centers", []) or [])])
        elif ctype == "micro_blob_texture":
            blob_centers.extend([[float(x), float(y)] for x, y in (params.get("blob_centers", []) or [])])

        contour_nodes.append({
            "component": cname,
            "type": ctype,
            "family": "curved" if ctype in _CURVED else ("linear" if ctype in _LINEAR else "texture"),
        })
        for parent in comp.get("parents", []) or []:
            contour_edges.append({"from": parent, "to": cname, "relation": "parent"})

    return {
        "true_holes_count_intrinsic": int(true_holes),
        "true_junction_count_intrinsic": int(true_junctions),
        "true_endpoints_count_intrinsic": int(true_endpoints),
        "true_parallel_groups": int(true_parallel_groups),
        "true_ladder_rungs": int(true_ladder_rungs),
        "true_periodic_components": int(true_periodic_components),
        "true_symmetry_axes": symmetry_axes,
        "true_blob_centers": blob_centers,
        "true_ridge_orientations": ridge_orientations,
        "true_contour_graph": {"nodes": contour_nodes, "edges": contour_edges},
    }



def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    if convolve is None:
        out = np.zeros(mask.shape, dtype=int)
        H, W = mask.shape
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                src_i = slice(max(0, -di), min(H, H - di))
                src_j = slice(max(0, -dj), min(W, W - dj))
                dst_i = slice(max(0, di), min(H, H + di))
                dst_j = slice(max(0, dj), min(W, W + dj))
                out[dst_i, dst_j] += mask[src_i, src_j]
        return out
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    return convolve(mask.astype(int), kernel, mode="constant", cval=0)



def _raster_topology(rendering: Dict[str, Any]) -> Dict[str, Any]:
    mask = np.asarray(rendering.get("binary_mask", np.zeros((0, 0), dtype=bool)), dtype=bool)
    if mask.size == 0:
        return {
            "true_components_count_raster": 0,
            "true_holes_count_raster": 0,
            "true_junction_count_raster": 0,
            "true_endpoints_count_raster": 0,
            "true_skeleton_graph": {"nodes": [], "edges": [], "pixel_count": 0},
        }

    if label is not None:
        labeled, n_comp = label(mask)
        inv_labeled, n_bg = label(~mask)
        holes = max(0, int(n_bg) - 1)
    else:
        n_comp = int(mask.any())
        holes = 0

    if skeletonize is not None:
        skel = skeletonize(mask)
    else:
        if distance_transform_edt is not None:
            dt = distance_transform_edt(mask)
            skel = mask & (dt >= np.maximum(1.0, np.percentile(dt[mask], 80) if np.any(mask) else 1.0))
        else:
            skel = mask.copy()

    nbh = _neighbor_count(skel)
    endpoints = np.argwhere(skel & (nbh == 1))
    junctions = np.argwhere(skel & (nbh >= 3))

    nodes = []
    for idx, (r, c) in enumerate(endpoints[:256]):
        nodes.append({"id": f"e{idx}", "kind": "endpoint", "rc": [int(r), int(c)]})
    base = len(nodes)
    for idx, (r, c) in enumerate(junctions[:256]):
        nodes.append({"id": f"j{idx}", "kind": "junction", "rc": [int(r), int(c)]})

    return {
        "true_components_count_raster": int(n_comp),
        "true_holes_count_raster": int(holes),
        "true_junction_count_raster": int(len(junctions)),
        "true_endpoints_count_raster": int(len(endpoints)),
        "true_skeleton_graph": {
            "nodes": nodes,
            "edges": [],
            "pixel_count": int(np.sum(skel)),
        },
    }



def build_structural_ground_truth(
    X: np.ndarray,
    y: np.ndarray,
    component_summaries: List[Dict[str, Any]],
    feature_names: List[str],
    bounds: Dict[str, Any],
    effective_rendering_spec: Dict[str, Any] | None,
    noise_label: int = -1,
) -> Dict[str, Any]:
    intrinsic = _estimate_intrinsic_targets(component_summaries)
    rendering_spec = dict(effective_rendering_spec or {})
    rendering_spec["enabled"] = True
    rendering_spec.setdefault("extra_views", {})
    rendering_spec["extra_views"] = {
        "distance_transform": True,
        "skeleton_mask": True,
        "edge_map": True,
        "gradient_orientation_map": True,
        "local_thickness_map": True,
        "cluster_id_raster": True,
    }
    render_out = render_dataset(X=X, y=y, feature_names=feature_names, rendering_spec=rendering_spec)
    raster = _raster_topology(render_out)

    non_noise_labels = [int(l) for l in np.unique(y) if int(l) != int(noise_label)]
    family_counts: Dict[str, int] = {}
    for comp in component_summaries:
        family_counts[comp.get("type", "unknown")] = family_counts.get(comp.get("type", "unknown"), 0) + 1

    out = {
        "true_components_count": int(max(len(non_noise_labels), raster.get("true_components_count_raster", 0))),
        "true_holes_count": int(max(intrinsic.get("true_holes_count_intrinsic", 0), raster.get("true_holes_count_raster", 0))),
        "true_junction_count": int(max(intrinsic.get("true_junction_count_intrinsic", 0), raster.get("true_junction_count_raster", 0))),
        "true_endpoints_count": int(max(intrinsic.get("true_endpoints_count_intrinsic", 0), raster.get("true_endpoints_count_raster", 0))),
        "true_symmetry_axes": intrinsic.get("true_symmetry_axes", []),
        "true_periodicity": {
            "periodic_component_count": int(intrinsic.get("true_periodic_components", 0)),
            "families_supporting_periodicity": [k for k in family_counts if k in {"checker_texture", "grid_lattice", "striped_texture", "oriented_texture_field", "parallel_bands", "ladder"}],
        },
        "true_parallel_groups": int(intrinsic.get("true_parallel_groups", 0)),
        "true_ladder_rungs": int(intrinsic.get("true_ladder_rungs", 0)),
        "true_blob_centers": intrinsic.get("true_blob_centers", []),
        "true_ridge_orientations": intrinsic.get("true_ridge_orientations", []),
        "true_contour_graph": intrinsic.get("true_contour_graph", {}),
        "true_skeleton_graph": raster.get("true_skeleton_graph", {}),
        "family_counts": family_counts,
        "noise_label": int(noise_label),
        "raster_support_summary": render_out.get("summary", {}),
    }
    out.update(intrinsic)
    out.update(raster)
    return out
