# HYBRID_SCM — Synthetic Cluster-Map Generator

HYBRID_SCM is the **data factory** of the project: a parametric, fully
reproducible generator of 2-D point-cloud clustering datasets with rich,
*controlled* geometric and topological structure — blobs, rings, arcs, lines,
ladders, polygons, spirals, textures, and adversarial mixtures of them. It
produces both the large **clustering repository** the rest of the project learns
from, and a **metric-coverage test suite** that validates the metric library.

```
declarative config (components + perturbations + difficulty)  ──►  X, y, metadata, rendering
```

---

## Research role: a controlled benchmarking environment

The project studies *which validity metric fits which structure*. Answering that
empirically requires datasets where (a) the ground truth is known exactly, (b) the
structural regime is **controllable and labelled**, and (c) generation is
reproducible to the seed. HYBRID_SCM is therefore positioned not as a toy data
helper but as a **controlled benchmarking environment for clustering-validity
research** — the experimental apparatus that makes the structure→metric mapping
observable and measurable.

```
HYBRID_SCM provides the independent variable (structure)
  and the ground truth (labels + semantic + structural targets)
  → enabling utility to be measured (offline) and metrics to be validated.
```

### Why a custom generator (vs. sklearn blobs/moons)

Off-the-shelf generators cover only convex blobs and a few 2-cluster shapes; they
omit rings, hollow topologies, ladders, lattices, textures, and radar-like
deinterleaving structure, and they record neither the *semantic pattern factors*
nor the *structural ground truth* this project needs. HYBRID_SCM is built to:

1. **Span structural regimes** — to construct the 12 repository families, each
   chosen so some metrics succeed and others fail.
2. **Control difficulty** — overlap, bridges, hole-punching, dropout,
   contamination, jitter, label-flip — so a regime can be made easy → hard while
   the truth stays known.
3. **Record everything** — exact seeds, per-component parameters, semantic factors
   (connectivity / holes / curvature / symmetry / thickness / …), and a structural
   ground truth (component / hole / junction / endpoint counts) for fine-grained,
   honest evaluation.
4. **Render** to images (grayscale / binary / edge / skeleton / distance /
   local-thickness) so the image-based features and image-based metrics have a
   raster to read.

### Why semantic + structural ground truth

Cluster labels alone do not say *what kind* of structure a dataset has. Recording
semantic pattern factors and structural counts turns the repository into a
*labelled testbed*: it lets analyses ask "does the regressor select ring-aware
metrics on datasets that are actually annular?" and lets the coverage suite assert
expected metric behaviour against known structure.

### Why adversarial structures

A metric can appear correct by responding to a *correlate* of the target structure
rather than the structure itself. Adversarial scenarios (e.g. two blobs bridged
into a fake connected component) are deliberately crafted look-alikes; a metric
that is fooled by them is not genuinely measuring what it claims. They are the
negative controls that make metric validation rigorous.

---

## How a dataset is built (offline generation)

A dataset is **composed** from primitive *components* (a blob, a ring, a band, a
spiral, …), each generating points and a label, then jointly perturbed and
roughened. `src/generator.py::generate_dataset(cfg)`:

```
seed master RNG
  → topologically sort components (a child may reference its parent's params)
  → for each component:
        spawn independent param / point / noise RNGs
        sample_params()  → sample_points()  → apply operators  → add noise
        accumulate X, y, per-component summary
  → dataset-level perturbations  (bridges, hole-punching, dropout,
                                  local contamination, size/density imbalance)
  → global difficulty            (global/feature/overlap jitter, label flip)
  → postprocess                  (e.g. PRI/TOA pipeline for radar-like data)
  → semantic pattern factors     (connectivity / holes / curvature / symmetry / …)
  → structural ground truth      (components, holes, junctions, endpoints, …)
  → (X, y, metadata, feature_names)
```

Saved by `io/save_utils.py` as the CSV data table, a metadata JSON (`params_used`,
pattern factors, structural targets), a **replay config** (frozen seeds for exact
reproduction), pairplots, and rendering artefacts.

---

## Layout

```
HYBRID_SCM/
  run_generate.py            # CLI: --config / --bundle / --replay
  src/
    generator.py             # generate_dataset(): the composition pipeline
    components/              # the shape/topology primitives  → README_components.md
    priors.py                # parameter sampling priors / specs
    semantic_metadata.py     # compute the semantic pattern factors
    rasterizer.py            # rendering / image generation
  io/                        # config_io (DatasetConfig / ComponentConfig), save_utils
  family_generation/         # build whole families/subfamilies for the repository
  configs/
    repository_generation/families/*.json   # the 12 repository families
    metric_coverage_suite/                   # per-metric coverage tests (see below)
  results_analysis/          # example generated runs + pairplots
```

A `DatasetConfig` declares `seed`, `n_features`, `bounds`, a list of
`ComponentConfig`s (`type`, `n_points`, `params`, `operators`, `noise`, optional
`parents`/`label`), plus optional `dataset_perturbations`, `global_difficulty`,
`postprocess`, and `rendering` blocks.

---

## The metric-coverage suite — validating the metric library

`configs/metric_coverage_suite/` is a **unit-test harness for the ~60 validity
metrics** in `models/ClustOpt`. It answers:

> Does each metric actually peak on the structure it is supposed to detect — and
> *not* get fooled by look-alikes?

One folder per metric, each with four hand-built scenarios:

| Case | Intent | Expectation |
|------|--------|-------------|
| `positive_01`   | a clear instance of the target structure | metric scores **high** (per its polarity) |
| `negative_01`   | the anti-pattern / target absent          | metric scores **low** |
| `borderline_01` | a marginal instance                       | score lands **between** positive and negative |
| `adversarial_01`| crafted to mislead                        | does **not** out-score the positive case |

```
HYBRID_SCM generates each scenario
   ↓
ClustOpt METRIC_REGISTRY:  safe_evaluate → safe_normalize
   ↓
checks: polarity · score spread · borderline ordering · adversarial robustness
        (per-family tolerances from manifest.json: fam_cluster, fam_circle, fam_ring, …)
   ↓
verdict: pass / review   (case_results.csv, expectation_report.json, summary.md)
```

This is the only place HYBRID_SCM calls into ClustOpt, and only to *test* metrics
— never during normal generation. See `configs/metric_coverage_suite/README.md`.

---

## Relationship to the rest of the project

```
HYBRID_SCM ──generates──►  raw datasets ──►  Clustering_Repository_Builder ──►  unified training set ──►  metric_utility_mlp
     │
     └──metric_coverage_suite──►  reuses ClustOpt's METRIC_REGISTRY to validate each metric
```

HYBRID_SCM and ClustOpt are decoupled: HYBRID_SCM produces `(X, y)`; ClustOpt's
metrics evaluate it.

---

## Main contributions

* **A controlled, structurally-diverse synthetic clustering repository** with exact
  labels plus *semantic* and *structural* ground truth — the experimental design
  that makes metric-utility learning possible.
* **A compositional generative model** (typed, parent-aware components +
  perturbations + difficulty controls) that spans convex, curved, hollow, linear,
  polygonal, lattice, texture, and radar-like regimes.
* **A metric coverage-validation suite** with positive/negative/borderline/
  **adversarial** controls, turning "does this metric work?" into an automated,
  per-metric verdict.

## Novelty positioning

Unlike standard synthetic clustering generators (convex blobs, a few 2-cluster
shapes, labels only), HYBRID_SCM is a *benchmarking environment*: it controls and
records the structural regime, supports adversarial negative controls, and is
explicitly co-designed with the validity-metric library it is used to evaluate.

## Limitations

* **2-D only.** Components, rendering, and structural targets are planar.
* **Synthetic.** It is a controlled proxy for real data; realism is bounded by the
  component vocabulary and the chosen priors.
* **Coverage ⊆ component expressiveness.** Structures outside the component set
  cannot be generated, so the learned structure→metric mapping is conditioned on
  this design space.

---

## CLI

```powershell
python -m models.HYBRID_SCM.run_generate --config  <dataset_config.json>
python -m models.HYBRID_SCM.run_generate --bundle  <bundle.json>          # many at once
python -m models.HYBRID_SCM.run_generate --replay  <replay_config.json>   # exact reproduction
```
