# Cluster Validity Indices (CVI)

The **objective layer** of ClustOpt — the part that, with no access to
ground-truth labels, scores how good a candidate partition is. This is where the
project's central idea is implemented: instead of one fixed validity index, a
*library of ~60 metrics* and a mechanism to **choose and combine them per
dataset**.

Parent engine: [README_ClustOpt.md](../README_ClustOpt.md).

---

## Research framing

A clustering search needs a scalar to maximise. The classical choice is a single
CVI, which encodes *one* notion of "good clustering" — usually compact, convex,
well-separated blobs — and is provably mismatched to non-convex structure.

```
Problem:  the validity objective is a latent decision variable.
          argmaxₘ  fit(m, structure(D))  is not constant across datasets.

This layer provides:
  (i)  a diverse metric library  M = {m₁,…,m₆₀}   (metrics/)
  (ii) evaluators that turn a chosen subset of M into one scalar objective:
          O(π) = Σ_{m} wₘ · sₘ(π)
       where the weight vector w is fixed (generic) or data-dependent (dynamic).
```

The CVI layer is thus the seam between *defining* candidate notions of validity
(the metric library) and *deciding which to apply* (the evaluators).

---

## Why a whole CVI subsystem

A single CVI bakes in one geometry of "good". To automatically cluster *arbitrary*
2-D structure — rings, arcs, stripes, spirals, textures — the objective must be
able to reward thin curves, hollow annuli, periodic lattices, textured fields, and
straight bands, and a mechanism must decide *which* to reward where. This
subsystem supplies the metrics (`metrics/`) and the deciding mechanism
(`dynamic_metric_selection/`).

Every metric is normalised to a common `[0,1]` scale and wrapped in
`safe_evaluate`/`safe_normalize`, so heterogeneous metrics are directly combinable
and one failing metric never aborts a search.

---

## Architecture

```
cluster_validity_indices/
  cvi_base.py            # BaseCVI: evaluate → normalize_scores → aggregate_score
  cvi_registry.py        # {"generic", "regressor_dynamic", "oracle_dynamic"}
  cvi_implementations/
    cvi_generic.py       # GenericCVIEvaluator (fixed metrics + weights)
    cvi_dynamic.py       # Regressor/Oracle dynamic evaluators (resolve → delegate to generic)
  metrics/               # the ~60-metric library + registry  → README_metrics.md
  dynamic_metric_selection/  # MLP prediction → top-k → weights → README_dynamic_metric_selection.md
```

### The three-stage objective (every CVI)

```
candidate labels π
   ↓ evaluate(π, X_decision, X_full)     →  {metric: raw value}   (each metric reads its space)
   ↓ normalize_scores(raw, π)            →  {metric: value ∈ [0,1]} (deterministic per metric)
   ↓ aggregate_score(normalized)         →  scalar = Σ wₘ·sₘ / Σ wₘ
objective value
```

`params` control validity preconditions: `remove_noise` (drop label `-1` before
scoring), `noise_label`, and `min_clusters` (reject partitions with too few
clusters via `-inf`).

---

## The three evaluator types

| `cvi.type` | How metrics & weights are chosen | Role in the study |
|------------|----------------------------------|-------------------|
| **`generic`** | fixed at config time: `metrics: {name: weight}` | **baselines** — e.g. `{silhouette: 1.0}`, or all classic CVIs uniform |
| **`regressor_dynamic`** | MLP predicts a utility per metric for *this* dataset → **top-k** → softmax weights | the **proposed method** |
| **`oracle_dynamic`** | same selection, from the *measured* utility vector | **upper bound** — best achievable if prediction were perfect |

`generic` is the shared engine: both dynamic evaluators **resolve** their chosen
metrics/weights once per dataset, then delegate scoring to an internal
`GenericCVIEvaluator`. All three thus share identical evaluate/normalise/aggregate
behaviour — they differ *only* in which metrics get non-zero weight. This is a
deliberate design choice: it guarantees that any performance difference comes from
*selection*, not from a different scoring mechanism.

### generic vs regressor_dynamic in one line

* `generic`: "use these metrics with these weights for **every** dataset."
* `regressor_dynamic`: "for **this** dataset's appearance, the regressor says
  metrics *m₁…mₖ* are most trustworthy — use them, weighted by predicted utility."

The dynamic path is fault-tolerant: if regressor resolution fails it falls back to
a configured fixed metric (`fallback_metrics`, e.g. Silhouette), so a search never
dies on a prediction error.

---

## How metrics are defined

Every metric subclasses `BaseMetric` (`metrics/metrics_base.py`):

| Attribute / method | Meaning |
|--------------------|---------|
| `name`             | registry key (e.g. `arc_circle_fit`); also the regressor target name |
| `space`            | `"decision"` or `"full"` — which data view it scores on |
| `higher_is_better` | optimisation direction (normalisation flips "lower-is-better" indices) |
| `evaluate(X, labels)` | raw, possibly unbounded value |
| `normalize(raw, ctx)` | deterministic map to `[0,1]` |
| `safe_evaluate / safe_normalize` | exception-safe wrappers (a failing metric yields a neutral score, never crashes the search) |

Metrics opt into the cached `GlobalMetricContext` / `PartitionMetricContext` via an
`evaluate_ctx(...)` path to avoid recomputing distances, rasters, hulls, and
skeletons across thousands of candidates.

---

## Main contributions

* **Geometry-, image-, and topology-aware clustering evaluation** unified behind
  one normalised `BaseCVI` interface, so structurally diverse notions of validity
  become directly composable.
* **A pluggable objective** with three interchangeable evaluators (fixed / learned
  / oracle) sharing identical scoring, isolating the effect of metric *selection*.

## Novelty positioning

Unlike traditional clustering-validity frameworks that expose a fixed menu of
indices, this layer treats validity as a *combinable, per-dataset-selectable*
quantity and provides the machinery to learn the combination — turning "which CVI?"
from a manual choice into a data-driven one.

## Limitations

* **Normalisation choices** influence the aggregated objective; a metric's `[0,1]`
  mapping is a modelling decision, not a canonical truth.
* **Metric redundancy.** `M` contains correlated metrics, so weights are not
  independently interpretable.
* **2-D image metrics** require a usable rasterisation; degenerate clusters fall
  back to neutral scores.

See **[README_metrics.md](metrics/README_metrics.md)** for the full library and
**[README_dynamic_metric_selection.md](dynamic_metric_selection/README_dynamic_metric_selection.md)**
for the regressor → selection → weighting flow.
