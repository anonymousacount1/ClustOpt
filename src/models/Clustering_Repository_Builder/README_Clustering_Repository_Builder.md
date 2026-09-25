# Clustering Repository Builder

*Master document for the data-generation and supervision half of the project.*

This subsystem turns a large library of synthetic 2-D point-cloud datasets into a
single, machine-learning-ready table that pairs **what a dataset looks like**
(label-free meta-features) with **which clustering-validity metrics actually work
on it** (metric-utility targets). That table is the supervision for the
Metric-Utility MLP (`metric_utility_mlp`), which in turn powers the *dynamic
metric selection* studied in `ClustOpt`.

---

## Research problem formulation

Unsupervised clustering optimization must choose a partition without ground-truth
labels, using an internal **cluster-validity index (CVI)** as a surrogate
objective. The recurring failure is that *no single CVI is correct for every
structure*: compactness-based indices (Silhouette, Calinski–Harabasz,
Davies–Bouldin) reward convex blobs and actively penalise the correct partition
of rings, arcs, spirals, or stripes. The choice of validity metric is therefore
itself a latent decision variable that should depend on the data.

We formalise this as a **metric-utility meta-learning** problem.

```
Given
  D                an unlabeled dataset (a 2-D point cloud),
  φ(D) ∈ ℝ²⁵⁰      a label-free meta-feature description of D,
  M = {m₁,…,m₆₀}   a library of cluster-validity metrics,
  S                a clustering search space (algorithms × hyper-parameters),
  and, OFFLINE ONLY, ground-truth labels y(D),

define for a candidate partition π and metric m
  sₘ(π) ∈ [0,1]    the normalised score metric m assigns to π,
  Q(π)   = ARI(π, y(D))   the external (ground-truth) quality of π,

and define the UTILITY of metric m on D as the rank-agreement between the way m
orders the candidate partitions visited during a search and the way ground-truth
ARI orders them:
  U(D, m) = Agree( {sₘ(π)}_{π∈trace(S,D)} , {Q(π)}_{π∈trace(S,D)} ) ∈ [0,1].

Stacking over M gives the utility vector  u(D) = (U(D,m₁),…,U(D,m₆₀)) ∈ [0,1]⁶⁰.

LEARN a predictor
  f : φ(D) ↦ û(D)
that approximates u(D) from label-free features alone, and USE it online to build
a per-dataset clustering objective
  O_D(π) = Σ_{m∈top-k(û(D))} wₘ · sₘ(π),   wₘ = softmax(û/T),

such that  argmax_{π∈S} O_D(π)  yields higher Q than any fixed-CVI baseline.
```

The research goal of this subsystem specifically is to **manufacture
`(φ(D), u(D))` pairs at scale** — the labelled examples `f` is trained on — and
to assemble them into one unified table.

---

## Why predict *metric utility* rather than the algorithm directly

A natural alternative is to learn `D ↦ best clustering algorithm` (classical
algorithm-selection / meta-learning). We deliberately do not. Predicting metric
utility:

* **Targets the objective, not the optimiser.** The bottleneck in unsupervised
  clustering is the *evaluation* surrogate, not the catalogue of algorithms. By
  shaping the objective, any algorithm in `S` can win if it is genuinely correct.
* **Is a smoother, better-posed learning target.** Utility is a graded,
  rank-based quantity in `[0,1]`, supporting ranking-aware supervision; "the best
  algorithm" is a brittle categorical label that ignores near-ties and
  hyper-parameter effects.
* **Composes.** Several metrics can be combined into one objective (Top-K +
  weighting), whereas a single predicted algorithm cannot be "blended."
* **Decouples generation from exploitation.** The learned objective is reusable
  across search strategies and budgets.

---

## Offline vs. online

The ground-truth labels appear **only offline**, purely as supervision/evaluation
signal — never as an input to the online clustering of a new dataset.

```
OFFLINE  (this subsystem + metric_utility_mlp)        ONLINE / inference (ClustOpt)
─────────────────────────────────────────────        ───────────────────────────────
HYBRID_SCM generates D with known y(D)                receive a new, UNLABELED D
   ↓                                                     ↓
features_extraction:  φ(D)        (label-free)        features_extraction: φ(D)
   ↓                                                     ↓
utility_generation:  run S, score M vs ARI            metric_utility_mlp: û(D)=f(φ(D))
   → u(D)            (uses y(D))                          ↓
   ↓                                                  dynamic_metric_selection:
training_dataset_builder: (φ(D), u(D)) rows              top-k(û)+softmax → O_D
   ↓                                                     ↓
metric_utility_mlp: learn f                           ClustOpt: argmax_π O_D(π)
   ↓                                                     ↓
experiments: evaluate f vs baselines (uses y(D))      return partition π̂  (no labels used)
```

---

## Global pipeline

```
HYBRID_SCM  (controlled synthetic generator, known ground truth)
   ↓        12 families × subfamilies × {easy,medium,hard} × true-k
Clustering Repository  (≈17k datasets, 2-D + labels + renderings)
   ↓
[1] features_extraction      →  φ(D): 250 label-free features × 3 views
   ↓
[2] utility_generation       →  u(D): 60 metric-utility targets × 3 views   (runs ClustOpt vs ARI)
   ↓
[3] training_dataset_builder →  unified_training_dataset.csv  (φ ⨝ u)
   ↓
metric_utility_mlp           →  f: φ(D) → û(D)
   ↓
dynamic_metric_selection     →  top-k + softmax → per-dataset objective O_D
   ↓
ClustOpt                     →  argmax_π O_D(π)
   ↓
[4] experiments              →  dynamic selection vs fixed-CVI baselines (ARI, k-accuracy)
```

Each **raw dataset contributes three "views"** — `x_only`, `y_only`, `xy_2d` —
sharing one `dataset_id` and always kept together when splitting (no leakage).

---

## Main contributions (of this subsystem)

* **Utility-supervised clustering optimization.** A concrete, scalable procedure
  to convert ground-truth-scored clustering searches into a continuous,
  rank-based *metric-utility* supervision signal `u(D)`.
* **A large, structurally-stratified synthetic clustering repository** (12
  families spanning convex, curved, hollow, linear, polygonal, lattice, texture,
  and radar-like regimes) purpose-built to make different validity metrics
  succeed and fail.
* **Partition/evaluation-space separation** realised end-to-end through the
  three-view design, enabling metric learning over both 1-D projections and the
  full 2-D structure.
* **A reproducible, failure-isolated data-engineering pipeline** (atomic writes,
  per-(dataset,view) resume, checkpointing) that makes the supervision set
  rebuildable and auditable.

## Novelty positioning

* **Unlike fixed-CVI clustering pipelines**, the objective is not a constant but a
  *learned, per-dataset* combination of metrics.
* **Unlike classical AutoClustering / algorithm-selection meta-learning**, the
  learned target is the *evaluation objective* (metric utility), not the
  optimiser (algorithm/hyper-parameters).
* **Unlike standard clustering-validity studies**, metrics are judged by their
  *rank agreement with ground truth across a search trace*, producing graded
  utility rather than a single "best index" verdict.

---

## Why three views (partition space ≠ evaluation space)

Each dataset is processed as three views:

| view_id  | Partition space (what may be clustered) | Why |
|----------|------------------------------------------|-----|
| `x_only` | x-axis only (1-D)                        | tests whether structure is recoverable from a single projection |
| `y_only` | y-axis only (1-D)                        | the orthogonal projection — different metrics may win |
| `xy_2d`  | full 2-D plane                           | the full structure |

The key methodological choice — carried through every downstream stage — is that
the **partition space** (the coordinates a clustering algorithm fits on) is kept
*separate* from the **evaluation space** (the coordinates a metric scores on). In
the 1-D views the partition is decided on one axis, yet geometry/image metrics
still evaluate against the *true* 2-D structure. This (a) multiplies the number
of `(φ, u)` examples per generated dataset, (b) creates regimes where the "best"
metric genuinely differs across views, and (c) lets the learner observe how
*projection* changes which metric is trustworthy — a signal that would be invisible
if every dataset were only ever evaluated in its native 2-D form.

---

## Internal components

| Folder | Stage | What it produces | Doc |
|--------|-------|------------------|-----|
| `features_extraction/`       | [1] | 250 label-free meta-features per (dataset, view) | [README_features_extraction.md](features_extraction/README_features_extraction.md) |
| `utility_generation/`        | [2] | 60 metric-utility targets per (dataset, view) via ClustOpt | [README_utility_generation.md](utility_generation/README_utility_generation.md) |
| `training_dataset_builder/`  | [3] | `unified_training_dataset.csv` (features ⨝ utilities) | — |
| `experiments/`               | [4] | AutoClustering evaluation: dynamic selection vs baselines | [README_experiments.md](experiments/README_experiments.md) |

All stages are **resumable, atomic-write, and failure-isolated**: work is
checkpointed per (dataset, view); a crash mid-write never leaves a half-file that
resume mistakes for "done"; a single dataset failure is logged and skipped, never
aborting the run. This is a methodological requirement, not just engineering: the
supervision set must be exactly rebuildable for the learned `f` to be reproducible.

### [3] training_dataset_builder

A leakage-aware join. For every dataset it reads the 3 feature rows
(`features/features_records.csv`) and attaches the matching `utility__<metric>`
columns from `utility/<view>/utility_vector.json`. Columns are ordered **19
audit/id → 250 features → 60 utility targets** (targets sorted alphabetically).
Datasets missing a utility vector are skipped and recorded in
`missing_artifacts.csv`, so the build never fails on partial data.

Output: `results_analysis/clustering_repository/analyzed_data/unified_training_dataset/`
→ `unified_training_dataset.csv` (~51,204 rows × 329 columns) + `build_metadata.json`.

---

## The data: families, subfamilies, difficulty, k

Raw datasets are generated by `HYBRID_SCM` and analysed under
`results_analysis/clustering_repository/analyzed_data/<family>/subfamilies/<subfamily>/<dataset>/`.
The twelve **families** are chosen as distinct *structural regimes*, each stressing
a different region of the metric library:

| Family | Structural regime it probes |
|--------|------------------------------|
| `standard_clustering_blobs`        | convex Gaussian blobs (where classic CVIs are strong) |
| `arcs_circles_rings`               | curved / annular structure (defeats compactness CVIs) |
| `hollow_topological_components`    | holes / non-trivial topology (Euler number, components) |
| `linear_bands_parallel_stripes`    | straight parallel bands / stripes |
| `parabolic_curved_chains`          | parabolic and curved chains |
| `piecewise_polylines_corners`      | multi-segment polylines with corners |
| `polygons_rectangles_frames`       | filled / hollow polygons and frames |
| `radial_spiral_meander`            | spirals and meandering curves |
| `ladder_grid_lattice`              | ladders and regular lattices (periodicity / grid metrics) |
| `texture_density_fields`           | texture / density fields (GLCM, LBP, fractal) |
| `hybrid_mixed_geometry`            | adversarial mixtures of the above |
| `deinterleaving_pri_toa_amp_like`  | radar-pulse-like (PRI/TOA/amplitude) deinterleaving structure |

Why this matters for *metric learning*: a predictor of metric utility can only
learn the structure→metric mapping if the training distribution actually contains
structures where that mapping varies. The families are the experimental design
that guarantees this coverage. Within each family, **subfamilies** are parametric
variants (e.g. `annuli_variable_thickness`); each dataset also carries
**difficulty** ∈ {easy, medium, hard} and a generator **true-k**, used *only* to
score results, never as a search input.

A dataset folder holds the raw `data_data.csv` (`x, y, cluster_id`),
`data_metadata.json` (generator parameters, seeds, per-component descriptors),
rendering artefacts (grayscale / binary / distance / skeleton / edge images, read
by the visual feature block and image metrics), and the per-stage outputs
(`features/`, `utility/`, `experiments/`).

---

## Limitations

* **2-D scope.** Every stage assumes 2-D point clouds; the image-based features
  and metrics, and the three-view construction, are specific to the plane.
* **Synthetic-only supervision.** Utilities are learned from `HYBRID_SCM`
  datasets; synthetic-to-real transfer is not established here.
* **Search-trace dependence.** Utility is defined over the partitions a *given*
  search visits, so it inherits any bias of the search space `S` and budget —
  a metric is only credited for structure the search actually explores.
* **Computational cost.** Generating `u(D)` requires a full clustering search per
  (dataset, view); this is the dominant offline cost of the pipeline.
* **Metric redundancy.** `M` contains correlated metrics, so utility targets are
  not independent — relevant when interpreting per-metric predictions.

---

## How to run (typical order)

```powershell
# [1] features for one subfamily
python -m models.Clustering_Repository_Builder.features_extraction.run_extract_subfamily_features `
  --subfamily-root "<REPO>\...\analyzed_data\<family>\subfamilies\<subfamily>" --overwrite

# [2] utility targets for the same subfamily (runs ClustOpt under the hood)
python -m models.Clustering_Repository_Builder.utility_generation.run_subfamily_utility_generation `
  --repo-root "<REPO>" --subfamily-dir "<...\subfamily>" --metric-mode fast --max-workers 8 --resume

# [3] join everything into the unified training table
python -m models.Clustering_Repository_Builder.training_dataset_builder.build_training_dataset `
  --repo-root "<REPO>" --analyzed-data-dir "<REPO>\...\analyzed_data"

# [4] (after the MLP is trained) evaluate dynamic selection vs baselines
```

See each sub-folder's README for flags, output layouts, and resume semantics.
