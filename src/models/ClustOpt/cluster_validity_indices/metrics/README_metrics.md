# Metric Library (~60 cluster-validity metrics)

The pool of validity metrics ClustOpt can use as its objective. Each metric maps a
partition `(X, labels)` to a score in `[0,1]` (after normalisation) where higher =
"this partition looks good *according to this metric's notion of structure*". The
research premise is that **these notions disagree**, and the right one depends on
the dataset.

These metric names are exactly the regressor's 60 prediction targets
(`utility__<metric>`) and the per-metric folders validated by
`models/HYBRID_SCM/configs/metric_coverage_suite`.

Parent layer: [README_cluster_validity_indices.md](../README_cluster_validity_indices.md).

---

## Why metric diversity is necessary

A clustering-validity metric is an *operational definition* of "good clustering".
Any single definition is a hypothesis about the data's geometry:

| Metric class | Implicit hypothesis | Fails when… |
|--------------|---------------------|-------------|
| **Compactness / separation** (Silhouette, Calinski–Harabasz, Davies–Bouldin) | clusters are compact, convex, roughly spherical blobs | clusters are curved, hollow, or elongated (rings, arcs, stripes) — the *correct* partition scores *worse* |
| **Density-based** (DBCV, S_Dbw) | clusters are connected high-density regions separated by low density | density is uniform or structure is thin/linear |
| **Topology / connectivity** | structure is defined by holes, components, junctions | shape is solid and simply connected |
| **Image / morphology / texture** | structure is best read from the rendered *shape*, not point statistics | rendering is degenerate or clusters are tiny |

Because each class is wrong somewhere, *no single metric can serve as a universal
objective.* The library exists so that, for any structural regime, **some** metric
encodes the right notion of validity — and the project's job
([dynamic_metric_selection](../dynamic_metric_selection/README_dynamic_metric_selection.md))
is to learn *which*.

---

## Scientific positioning relative to standard CVIs

* **Beyond compactness.** Classic CVIs are retained as `core_cvi/` (and as
  baselines), but treated as *one family among several*, not the default.
* **Beyond point statistics.** `image_base/` imports computer-vision reasoning
  (Hough, GLCM, LBP, fractal, Euler number) so morphology and texture become
  first-class validity signals — properties that point-distance CVIs cannot express.
* **Shape-explicit.** `pattern_base/` fits arcs, lines, parabolas, ladders, rings
  *directly in point space*, encoding domain shape priors that no generic CVI has.

---

## The four families

Organised by *how* they read structure. Registered in `metrics_registry.py`.

| Family (folder) | ~count | How it reads the partition | Probes datasets like |
|-----------------|--------|----------------------------|----------------------|
| **`core_cvi/`**        | 6  | classic distance/density CVIs over point coordinates | convex blobs, separated clusters |
| **`structure_base/`**  | 8  | per-cluster geometry & cross-cluster organisation (PCA, neighbourhood, dispersion, size balance) | compactness, balance, neighbourhood purity |
| **`pattern_base/`**    | ~22 | shape & curve fitting **in point space** (no rasterisation) | arcs, lines, parabolas, ladders, rings, periodicity |
| **`image_base/`**      | ~24 | **rasterise** each cluster to an image, then computer-vision / texture analysis | morphology, texture, topology, edges |

### core_cvi (classic)
`silhouette`, `calinski_harabasz`, `davies_bouldin`, `dbcv`, `s_dbw`,
`noise_aware_silhouette`. Fast, well-understood, strong on convex/separated
clusters; the natural **baselines**.

### structure_base (cluster organisation)
`avg_pca_isotropy`, `cluster_size_balance_entropy`,
`cluster_size_imbalance_entropy`, `neighborhood_purity`,
`neighborhood_purity_multi_k` (k∈{5,10,20}), `min_intercluster_distance`,
`avg_within_cluster_dispersion`, `min_centroid_distance`. Geometric "health"
signals that hold in any dimension.

### pattern_base (point-space shape/curve fitting)
`arc_circle_fit`, `ladderness`, `parallel_bands`, `parabolicity`,
`periodicity_1d`, `line_straightness`, `piecewise_linearity`,
`alpha_shape_compactness`, `bimodality_anti_thickness`, `convexity_ratio`,
`curvature_consistency`, `turning_points_count`, `direction_entropy`,
`gridness_2d`, `mst_smoothness`, `ribbon_thickness`, `axial_symmetry`,
`connectivity_components`, `corner_sharpness`, `eccentricity_ratio`,
`hollow_score`, `radial_uniformity`. These test "does this cluster look like an
arc / line / parabola / ladder / ring?" and are what let the search succeed on the
non-convex families.

### image_base (rasterise → vision/texture)
`hough_line_strength`, `peak_to_background_hough`,
`orientation_histogram_entropy`, `edge_coherence`, `hough_arc_circle_strength`,
`convexity_ratio_image_based`, `ellipse_fit_score`, `circularity_compactness`,
`polygonality`, `rectangularity`, `morphological_band_count`,
`skeleton_connectivity`, `contour_graph_connectivity`, `ridge_strength`,
`thickness_uniformity`, `glcm_haralick`, `lbp_stationarity`, `fractal_dimension`,
`gridness_fft_acf`, `symmetry_score`, `euler_holes_count`,
`connected_components_count`, `blobness_log_dog`, `corner_junction_density`.
These capture properties hard to express analytically (texture, morphology,
topology) by borrowing from computer vision.

---

## How image_base works

`image_base/utils/image_utils.py` + `image_metric_base.py`:

```
cluster points Xᵢ
   ↓ rasterize_points (RasterParams: grid 256, binary|density, blur, dilate; global bounds)
binary / density image
   ↓ ImagePerClusterMetric._cluster_score(Xᵢ, bounds)   (CV operator: Hough / GLCM / Euler / …)
per-cluster score
   ↓ size-weighted average over clusters
metric value
```

Bounds are taken over **all** clusters so geometry is comparable; each image metric
carries a `min_cluster_points` guard (small clusters can't form a meaningful image
and score neutrally). Example: `hough_line_strength` runs Canny + probabilistic
Hough and returns the fraction of edge length explained by straight lines.

---

## Main contributions

* A **broad, four-paradigm validity-metric library** (classic, structural,
  shape-fitting, image/topology) under one normalised interface.
* **Image-based clustering metrics** that transfer computer-vision shape/texture
  analysis into clustering validity — signals unavailable to point-distance CVIs.

## Novelty positioning

Unlike silhouette-style or density-only validity frameworks, this library treats
"good clustering" as *plural*: it provides explicit shape-, morphology-, and
topology-aware metrics so that the correct partition of a ring, lattice, or texture
field can actually be *rewarded* by some member of `M`.

## Limitations

* **Redundancy.** Several metrics are correlated (e.g. multiple convexity/circle
  signals); they are not statistically independent.
* **Rendering dependence.** Image metrics need a usable raster; degenerate
  clusters fall back to neutral scores.
* **Cost.** Image/texture metrics are the expensive members of `M`, motivating
  ClustOpt's fast metric mode.

---

## Contract recap

Each metric (`BaseMetric`) exposes `name`, `space` (`decision`/`full`),
`higher_is_better`, `evaluate`, `normalize`, and `safe_*` wrappers; the registry in
`metrics_registry.py` instantiates each with its tuned hyper-parameters. The CVI
evaluators ([README_cluster_validity_indices.md](../README_cluster_validity_indices.md))
call them through evaluate → normalise → aggregate, reusing context caches so a
multi-trial search stays affordable.
