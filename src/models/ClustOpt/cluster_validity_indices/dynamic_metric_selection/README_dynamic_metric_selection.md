# Dynamic Metric Selection

The bridge that turns the trained Metric-Utility MLP into a live ClustOpt
objective. This is where the project's hypothesis becomes an operational
mechanism: **read the dataset's label-free meta-features → predict each metric's
utility → keep the few most useful metrics → weight them → hand that combination to
the search as its objective.**

Used by the `regressor_dynamic` (and `oracle_dynamic`) CVI evaluators in
`../cvi_implementations/cvi_dynamic.py`. Parent layer:
[README_cluster_validity_indices.md](../README_cluster_validity_indices.md).

---

## This is dynamic objective shaping, not algorithm prediction

A crucial distinction:

```
algorithm selection (NOT this):   f(φ(D)) → "use HDBSCAN with these params"
objective shaping  (THIS):        f(φ(D)) → weights w over metrics M
                                  → O_D(π) = Σ_{m∈top-k} wₘ · sₘ(π)
                                  → ClustOpt: argmax_{π∈S} O_D(π)
```

The predictor never names an algorithm, a partition, or a cluster count. It only
reshapes **what the search optimises**. *Any* algorithm in the search space can
then win, provided it produces a partition the selected metrics rank highly. This
keeps the optimiser a controlled variable and makes the intervention purely about
*the definition of "good"* for this dataset.

---

## Research problem (this module's slice)

```
Online, given an UNLABELED dataset D:
  û(D) = f(φ(D))                         predicted metric-utility vector ∈ [0,1]⁶⁰
  Sₖ   = top-k(û)                        the k highest-utility metrics
  wₘ   = softmax(û_m / T)  for m ∈ Sₖ    weights (temperature T)
  O_D(π) = Σ_{m∈Sₖ} wₘ · sₘ(π)          the per-dataset clustering objective
```

The design questions this module answers experimentally: **how many** metrics to
trust (`k`) and **how sharply** to concentrate weight on the best ones (`T`).

---

## The flow

```
features φ(D)   (for this dataset + view)
   ↓  RegressorPredictor.predict()
û(D) = {metric: predicted utility ∈ [0,1]}   (60 values)
   ↓  select_top_k(k)
top-k metrics by utility   (deterministic tie-break by name)
   ↓  compute_weights(weighting, T)
{metric: weight}
   ↓
GenericCVIEvaluator   →   dynamic CVI objective O_D
   ↓
Optuna / brute-force search   →   argmax_π O_D(π)
```

| File | Responsibility |
|------|----------------|
| `regressor_predictor.py` | load the trained MLP run dir and predict utilities from a feature vector |
| `resolvers.py`           | orchestrate predict → select → weight; return a `ResolvedMetrics` (weights, selected, raw utilities, debug); also the oracle resolver |
| `weighting.py`           | `select_top_k` + `compute_weights` (softmax / normalized_positive) |
| `naming.py`              | map regressor target names (`utility__<metric>`) ↔ metric registry keys |
| `loaders.py`             | load the feature vector (view-aware) or an oracle utility vector |

### 1. Predict (`RegressorPredictor`)
Loads a trained run directory (`config.json`, `artifacts/{feature,target}_columns.json`,
`head_mapping.json`, per-fold `best_model.pt` + `x_scaler.pkl`). Aligns incoming
features to the training column order, scales them, runs the MLP, and — under
`checkpoint_policy: "fold_ensemble"` — **averages predictions across folds** (or
uses a single fold). Returns `{metric_name: utility}`.

### 2. Select (`select_top_k`)
Sort metrics by predicted utility (descending; ties broken alphabetically for
reproducibility; non-finite → bottom) and keep the top **k**.

### 3. Weight (`compute_weights`)

| `weighting` | Formula | Effect |
|-------------|---------|--------|
| `softmax` (default) | `softmax(score / T)` | temperature `T` controls sharpness; low `T` ⇒ concentrate weight on the very best metric |
| `normalized_positive` | clip to `[0,∞)`, normalise to sum 1 | linear, zero weight to non-positive utilities |

---

## Why Top-K weighting rather than single-metric selection

Selecting only the single argmax metric (`k=1`) is brittle: prediction noise can
flip the top choice, and many structures are best judged by *several complementary*
signals (e.g. a ring scores on both `hollow_score` and `arc_circle_fit`). A
weighted committee of the top-k:

* **hedges against prediction error** — a mistaken #1 is diluted by correct
  runners-up;
* **captures multi-faceted structure** — combines compatible notions of validity;
* **degrades gracefully** — weight mass tracks predicted confidence rather than
  collapsing to a hard choice.

### Why softmax, and why temperature

Softmax turns predicted utilities into a normalised, positive weight distribution
whose *sharpness* is tunable by temperature `T`:

* low `T` → near-argmax (trust the single best metric);
* high `T` → near-uniform over the top-k (hedge widely).

`T` therefore interpolates between "decisive" and "cautious" objective shaping
*without* changing `k`, making the confidence-vs-robustness trade-off an explicit,
sweepable knob.

---

## The experiment grid: `regressor_top{1,3,5,10}_softmax_t05`

The named configs vary only **k** at fixed temperature `T = 0.5`:

| Config | k | Meaning |
|--------|---|---------|
| `regressor_top1_softmax_t05`  | 1  | trust only the single best-predicted metric |
| `regressor_top3_softmax_t05`  | 3  | small committee |
| `regressor_top5_softmax_t05`  | 5  | medium committee |
| `regressor_top10_softmax_t05` | 10 | broad committee |

This sweep answers a clean research question: **is it better to bet on the one
metric the regressor likes most, or to hedge across several?** The
[experiments](../../../Clustering_Repository_Builder/experiments/README_experiments.md)
pipeline compares all four against fixed-CVI baselines and the oracle ceiling.

---

## Config (inside `cvi.params`)

```json
{
  "model_run_dir": "results_analysis/mlp/combined",
  "checkpoint_policy": "fold_ensemble",
  "top_k": 5,
  "weighting": "softmax",
  "softmax_temperature": 0.5,
  "metric_name_prefix": "utility__",
  "feature_source": "features/features_records.csv",
  "view_aware": true,
  "fallback_on_error": true,
  "fallback_metrics": { "silhouette": 1.0 }
}
```

* `view_aware: true` selects the feature row matching the search's view
  (`x_only`/`y_only`/`xy_2d`).
* `metric_name_prefix` strips `utility__` so predicted names line up with metric
  registry keys (`naming.py` resolves the few that differ).
* `fallback_*` guarantees the search still runs (on a fixed metric) if prediction
  or loading fails.

The `oracle_dynamic` variant uses the same selection/weighting but sources
utilities from a stored `utility_vector.json` — the upper bound on what perfect
prediction would achieve.

---

## Main contributions

* **Dynamic, per-dataset objective shaping** for clustering: a learned predictor
  reconfigures the validity objective at inference from label-free features alone.
* **Top-K + temperature-controlled softmax weighting** as a robust selection
  mechanism, with `k` and `T` as explicit, interpretable trade-off knobs.

## Novelty positioning

Unlike fixed-CVI pipelines (constant objective) and unlike algorithm-selection
meta-learning (predict the optimiser), this module predicts *how to evaluate*, then
lets the existing optimiser act on the reshaped objective.

## Limitations

* **Inherits the regressor.** Selection quality is bounded by `f`; systematic
  prediction bias propagates into the objective (the oracle variant quantifies the
  gap).
* **No predictive uncertainty.** Weights are point estimates; the module does not
  yet express confidence in its selection.
* **Fixed `T` in shipped configs.** The grid sweeps `k` at `T=0.5`; the full
  `(k, T)` surface is not exhaustively explored.
