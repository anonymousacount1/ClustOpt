# Feature Extraction — the label-free dataset descriptor φ(D)

Computes a fixed, **250-number description** `φ(D)` of every dataset, using *only*
what an unlabeled observer could see (point coordinates + rendered images). In the
problem formulation (see
[README_Clustering_Repository_Builder.md](../README_Clustering_Repository_Builder.md))
`φ(D)` is the **input** to the learned metric-utility predictor `f`: the model
must decide which validity metrics to trust *from appearance alone*, so `φ` must
never depend on ground-truth labels.

```
Input  : one raw 2-D dataset folder (x, y [, cluster_id], rendering/*.png)
Output : 3 rows (x_only, y_only, xy_2d) × 250 features, written to features/
```

---

## Formal role

```
φ : D ↦ ℝ²⁵⁰      a deterministic, label-free embedding of a dataset

Requirements:
  (label-freeness)   φ(D) is computable without y(D)  — identical offline & online
  (completeness)     φ(D) always has all 250 coordinates (NaN → 0.0 fill)
  (expressiveness)   φ separates structural regimes that change argmaxₘ U(D,m)
```

The third requirement is the scientific crux: `f = g ∘ φ` can only recover the
structure→metric mapping if `φ` actually exposes the distinctions (convex vs.
annular, linear vs. curved, periodic, holey, textured, uni- vs. multi-modal) that
*cause* one metric to outperform another. The six feature blocks are designed to
span exactly those axes of variation.

---

## Why label-free, and identical offline/online

The downstream goal is to predict, for a brand-new **unlabeled** dataset, which
metric to trust. For that to be honest and deployable:

1. **Strictly label-free.** `cluster_id`, cluster masks, and structural ground
   truth are forbidden inputs. The visual block reads only the *global* rendering
   (grayscale/binary/edge/skeleton/distance) — never per-cluster rasters — so the
   answer cannot leak through the image channel.
2. **Phase-invariant.** `φ` is computed by the *same* code path offline (to build
   training inputs) and online (at inference). There is no train-only feature.
3. **Fixed schema, no empty cells.** `feature_schema.py` is the single source of
   truth: every record carries all 250 feature columns + 19 audit columns in a
   fixed order; an uncomputable feature becomes `NaN` → filled `0.0`; a block that
   raises is caught so one bad feature never voids a row.

### The three views

| view_id  | What the point-statistics blocks see | Why |
|----------|--------------------------------------|-----|
| `x_only` | x-axis only (1-D)                    | is the structure separable on one axis? |
| `y_only` | y-axis only (1-D)                    | the orthogonal projection |
| `xy_2d`  | full 2-D plane                       | the full structure |

Geometry/visual blocks always retain the full 2-D context, so image and shape
features stay meaningful even in 1-D views — the partition-space / evaluation-space
separation begins here and is preserved through utility generation and the
experiments.

---

## The six feature blocks (250 total)

| Block | Module | # | What it measures | Why it matters for metric choice |
|-------|--------|---|------------------|----------------------------------|
| **A** | `block_a_partition_stats.py` | 30 | univariate stats of the **selected view**: counts, finite/duplicate ratios, mean/std/min/max, quantiles, IQR, MAD, skew, kurtosis, histogram entropy (32/64 bins), peak ratio, outlier fractions (IQR & z>3), tail mass | distribution shape on the axis the search may use |
| **B** | `block_b_evaluation_stats.py` | 45 | bivariate stats of the **full 2-D** data: per-axis stats, bbox/aspect/density, covariance trace/det/condition, Pearson correlation, PCA eigenvalues/explained-ratio/anisotropy, centroid distances, 16×16 & 32×32 density-grid occupancy/entropy/Gini | global spread & anisotropy of the whole cloud |
| **G** | `block_g_geometry_structure.py` | 60 | geometry **without clustering**: kNN distances (k=1,3,5,10), local density, MST edge stats & long-edge ratio, convex-hull & alpha-shape area/compactness, linearity/planarity, curve smoothness, direction & orientation entropy, radial / ring-likeness, circle/parabola/line fits (R²), periodicity (ACF & FFT), multimodality | linear, curved, annular, periodic, or fragmented? |
| **V** | `block_v_visual_raster.py` | 60 | **image-derived** features from the global rendering: foreground fraction, grayscale stats, connected components, holes & Euler number, edges, skeleton (endpoints/junctions), distance-transform / local-thickness stats, orientation histograms, projection periodicity, FFT energy bands, texture (LBP, GLCM), fractal box-count, LoG blob response | morphology & texture that mirror the image-based metrics |
| **R** | `block_r_relation_features.py` | 25 | **dependency between the two original axes**: Pearson/Spearman/Kendall, mutual information (16/32 bins, normalized), linear & poly2/poly3 R² (x→y, y→x), monotonicity, conditional variance, joint/marginal entropies | functional vs. independent axes (curves, bands) |
| **L** | `block_l_landmarking.py` | 30 | **cheap unsupervised probes** ("landmarking"): quick KMeans (k=2–5) silhouette & inertia, GMM BIC, DBSCAN/HDBSCAN cluster-count & noise ratio, Agglomerative silhouettes, consensus/disagreement | a meta-learning prior: what do off-the-shelf clusterers already say? |

Block letters reflect development order; CSV prefixes follow the block (`A_*`,
`B_*`, `G_*`, `V_*`, `R_*`, `L_*`). The L block is optional (`--skip-landmarking`);
HDBSCAN is handled gracefully if absent.

The blocks deliberately span **four complementary epistemologies** of structure —
statistical (A,B), analytic-geometric (G,R), visual/morphological (V), and
empirical-probing (L) — so that no single way of "seeing" the data dominates the
descriptor.

---

## Main contributions

* A **label-free, fixed-schema, leakage-controlled descriptor** purpose-built so a
  learner can recover *which validity metric fits a dataset's structure*.
* A **multi-paradigm feature design** (statistics + analytic geometry + computer
  vision + unsupervised landmarking) aligned, by construction, with the four
  metric families it must help select among.
* **View-aware extraction** that materialises the partition/evaluation-space
  separation as data, tripling the supervision signal per generated dataset.

## Novelty positioning

Unlike traditional clustering meta-features (which mostly summarise global
statistics for algorithm selection), this descriptor is engineered to be
*metric-discriminative* and explicitly **decouples the projection used for
partitioning from the geometry used for evaluation**, and adds image- and
topology-aware channels matched to image-based validity metrics.

---

## Module layout & flow

```
features_extraction/
  run_extract_subfamily_features.py  # CLI: discover datasets → extract → summarise
  dataset_loader.py                  # load x/y + metadata + rendering for one folder
  view_builder.py                    # build the 3 views (x_only / y_only / xy_2d)
  feature_extractor.py               # orchestrate: run all 6 blocks per view
  feature_schema.py                  # AUTHORITATIVE 250-feature + 19-audit schema
  blocks/                            # block_a … block_l (one file per block)
  utils/                             # numeric / image / geometry / validation helpers
  output_writer.py                   # atomic CSV / JSONL / Parquet writers
  summary_writer.py                  # subfamily-level CSV / JSON / MD summaries
  progress_tracker.py                # per-dataset progress, ETA, block timings
```

```
per dataset:
  dataset_loader → view_builder → feature_extractor(blocks A,B,G,V,R,L per view)
        → feature_schema.ensure_complete_feature_record → output_writer
```

### Output

```
<dataset>/features/
  features_records.csv     # 3 rows: x_only, y_only, xy_2d
  features_records.jsonl   # line-format backup
  features_metadata.json   # per-block timings, NaN counts, status

<subfamily>/
  subfamily_features_all.csv      # all datasets concatenated
  subfamily_features_all.parquet  # (if deps available)
  subfamily_feature_extraction_summary.{json,md}
  subfamily_feature_schema.json
  failed_datasets.csv             # ids + reasons (if any)
```

### Audit columns (19, not model inputs)

`dataset_id, record_id, view_mode, view_dim, source_dataset_dir, family_id,
family_name, subfamily_id, difficulty, cluster_count_from_generator,
dataset_seed, family_seed, n_points, feature_names_original, selected_features,
has_labels, has_rendering, has_structural_ground_truth, extraction_status`.

They travel with the features for grouping/splitting and reporting but are never
fed to the model.

---

## Limitations

* **2-D and rendering-dependent.** The V block presupposes a rendered image;
  poor or missing renderings degrade the morphological channel.
* **Fixed 250-dim schema.** New structural regimes may need new features; the
  schema is authoritative and intentionally static within a study.
* **Landmarking cost/variance.** The L block runs real clusterers; it adds cost
  and a little stochasticity, hence the opt-out flag.

---

## CLI

```powershell
python -m models.Clustering_Repository_Builder.features_extraction.run_extract_subfamily_features `
  --subfamily-root "<REPO>\...\analyzed_data\<family>\subfamilies\<subfamily>" `
  --overwrite
```

Resumable (skips datasets whose `features/` is already complete) unless
`--overwrite`; `--skip-landmarking` drops the L block for speed.
