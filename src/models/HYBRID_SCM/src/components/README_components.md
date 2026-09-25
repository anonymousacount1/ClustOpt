# Components — HYBRID_SCM shape & structure primitives

Each **component** is a self-contained generator for one structural primitive —
a blob, a ring, a band, a spiral, a texture field, … A dataset is *composed* by
listing several components in its config; the generator samples each one's
parameters and points, applies operators, and unions the results into the final
`(X, y)`. Components are the **structural vocabulary** of the project: they define
the space of structures over which the metric-utility mapping is learned.

Parent generator: [README_HYBRID_SCM.md](../../README_HYBRID_SCM.md).

---

## Research role: a compositional generative model of structure

The components form a small **generative grammar**. A dataset is a labelled union
of typed primitives:

```
D = ⋃ᵢ Componentᵢ(paramsᵢ),   y assigns each point its component label

structural family of D  ⇐  the multiset of component types chosen
true k, holes, junctions ⇐  derived from the components and their composition
```

Because structure is *built from labelled parts*, ground truth is exact by
construction and the structural regime is an explicit, controllable variable —
exactly the property the rest of the pipeline needs in order to measure metric
utility against known structure. Composition also yields controlled difficulty:
overlap two blobs, bridge two rings, or drop points from an arc, without ever
losing track of the truth.

---

## The contract

All components subclass `Component` (`base.py`) and register themselves into a
name→class registry at import (`__init__.py`); the generator resolves a config's
`"type"` via `get_component(type)`.

| Method | Contract |
|--------|----------|
| `sample_params(rng, n_features, param_spec, parent_params)` | draw geometric parameters from priors (`priors.py`); may reference a parent component |
| `sample_points(rng, n_points, n_features, low, high, params_used, parent_params)` | generate the points for those parameters |

Shared mechanics:

* **`operators.py`** — post-generation transforms (rotation, scaling, jitter,
  thickness profiles, quantisation, …), recorded in the component summary.
* **`_helpers.py`** — common sampling/geometry utilities.
* **`base.py`** — the `Component` base class and the registry (`register`,
  `get_component`).

Components can declare **parents**: the generator topologically sorts them so a
child (noise under a band, a junction between two arcs) can position itself
relative to an already-sampled parent — enabling *relational* structure, not just
independent shapes.

---

## The primitive catalogue

| Category | Components | Builds structure like |
|----------|-----------|------------------------|
| **Blobs / ellipses** | `blob`, `ellipse`, `blob_field` | convex Gaussian clusters, elliptical clusters, fields of small blobs |
| **Curved / circular** | `circle`, `ring_annulus`, `arch_chain`, `parabola`, `spiral`, `meander_curve` | rings, hollow annuli, arch chains, parabolic curves, spirals, serpentine curves |
| **Linear / piecewise** | `band`, `parallel_bands`, `piecewise_polyline`, `ladder` | straight bands, parallel stripes, multi-segment polylines, ladder rungs |
| **Polygonal** | `polygon`, `rectangle_frame`, `grid_lattice`, `junction`, `branching_tree` | filled polygons, hollow frames, regular lattices, junctions, tree/fork structures |
| **Texture / field** | `ridge_field`, `oriented_texture_field`, `anisotropic_grain_field`, `micro_blob_texture`, `striped_texture`, `checker_texture`, `speckle_texture`, `fractal_dust` | ridge waves, oriented/anisotropic grain, micro-blob and stripe/checker textures, speckle, fractal dust |
| **Noise** | `noise` | uniform / structured noise; clouds near or under other components |

### Designed correspondence with the metric library

The primitives map deliberately onto the validity metrics' specialties, so each
metric has datasets that genuinely exercise it:

| Primitive group | Metrics it exercises |
|-----------------|----------------------|
| rings / arcs / circles | `arc_circle_fit`, `hollow_score`, `euler_holes_count`, `circularity_compactness` |
| bands / ladders / lattices | `parallel_bands`, `ladderness`, `gridness_2d`, `gridness_fft_acf` |
| polylines / corners | `piecewise_linearity`, `corner_sharpness`, `turning_points_count` |
| texture fields | `glcm_haralick`, `lbp_stationarity`, `fractal_dimension`, `ridge_strength` |
| blobs | `silhouette`, `calinski_harabasz`, `davies_bouldin` |

This alignment is what lets the metric-coverage suite build a credible
positive/negative/borderline/**adversarial** scenario for every metric, and what
guarantees the repository contains structure where each metric's utility can
actually peak.

---

## Example (config fragment)

```json
{
  "components": [
    { "name": "r1", "type": "ring_annulus", "n_points": 800,
      "params": { "radius": 0.6, "thickness": 0.05 } },
    { "name": "b1", "type": "blob", "n_points": 400,
      "params": { "sigma": 0.05 }, "noise": "gaussian" }
  ]
}
```

The generator samples `r1` and `b1`, assigns each its own label, applies operators/
noise, and unions them — a solid blob inside a hollow ring, with exact labels and
structural ground truth recorded.

---

## Main contributions

* A **typed, parent-aware compositional grammar** that generates structurally
  diverse datasets with exact, by-construction ground truth.
* A **catalogue co-designed with the validity-metric library**, ensuring every
  metric has structures that exercise (and adversarially probe) it.

## Limitations

* **Vocabulary-bounded.** Only structures expressible as unions of these
  primitives (+ operators/perturbations) can be generated.
* **2-D.** Primitives and operators are planar.
* **Prior-dependent.** The realism/variety of each primitive is bounded by its
  sampling priors in `priors.py`.

To add a primitive: create `my_shape.py` implementing `sample_params` /
`sample_points`, call `register("my_shape", MyShape)`, and import it in
`__init__.py`. No generator changes are needed.
