# Metric Utility MLP

A self-contained PyTorch pipeline that predicts the **utility vector of clustering
metrics** from extracted dataset meta-features. This model *is* the project's
learned hypothesis `f`: it maps "what a dataset looks like" → "which validity
metrics to trust on it".

```
Input : 250 dataset meta-features          φ(D)
Output: 60 metric-utility scores in [0, 1] û(D) ≈ u(D)
```

The model is trained and evaluated with **group-safe 5-fold cross-validation** and
judged as a regression model, a **ranking** model, and a **top-metric selector**.

> **Role in the pipeline.** Inputs are the 250 label-free meta-features from
> [features_extraction](../Clustering_Repository_Builder/features_extraction/README_features_extraction.md);
> targets are the 60 metric-utility scores from
> [utility_generation](../Clustering_Repository_Builder/utility_generation/README_utility_generation.md);
> both are joined by the
> [Clustering Repository Builder](../Clustering_Repository_Builder/README_Clustering_Repository_Builder.md).
> Once trained, the model is loaded at clustering time by ClustOpt's
> [dynamic_metric_selection](../ClustOpt/cluster_validity_indices/dynamic_metric_selection/README_dynamic_metric_selection.md)
> and evaluated end-to-end in the
> [AutoClustering experiments](../Clustering_Repository_Builder/experiments/README_experiments.md).
> The six output heads mirror the families in
> [metrics](../ClustOpt/cluster_validity_indices/metrics/README_metrics.md).

---

## Research problem (this module's slice)

```
Learn      f : ℝ²⁵⁰ → [0,1]⁶⁰,   f(φ(D)) = û(D)
from data  { (φ(Dᵢ), u(Dᵢ)) }   (the unified training set)
so that    the metrics ranked highest by û(D) are the metrics that actually
           steer a clustering search toward the ground-truth-best partition.
```

`f` is the **offline-trained** core of the system; at inference it is consumed by
dynamic metric selection. It never sees ground-truth labels — its targets `u(D)`
already distil that supervision (computed offline by utility generation).

### Why predicting utility — and why ranking quality, not exact regression

The downstream use of `û(D)` is to **rank** metrics and keep the top-k for the
clustering objective. Therefore:

* what matters is the **ordering** of the 60 utilities, not their exact values;
* a model with low MSE but poor top-end ranking would select the *wrong* metrics
  despite "accurate" numbers;
* so `f` is supervised and judged as a **ranking / top-metric selector**, with a
  loss that explicitly rewards correct ordering and correct top-k membership.

This is the central modelling choice and the reason the loss is not plain MSE.

---

## Data

Unified training CSV (already produced by the repository builder):

```
results_analysis/clustering_repository/analyzed_data/unified_training_dataset/unified_training_dataset.csv
```

| group    | count | notes |
| -------- | ----- | ----- |
| id/meta  | 19    | not model inputs (`dataset_id`, `family_name`, `view_mode`, ...) |
| features | 250   | model inputs (`A_view_*`, `B_*`, ... `L_landmark_*`) |
| targets  | 60    | `utility__<metric>` columns, all in `[0, 1]` |

Each raw dataset contributes **exactly three rows** (`x_only`, `y_only`, `xy_2d`)
sharing one `dataset_id`. These rows are kept together by `GroupKFold(dataset_id)`
so no dataset leaks across train/val/test.

---

## Leakage-safe split strategy — and why it is essential

* Outer test folds: `GroupKFold(n_splits=5)` over `dataset_id` (20% test each).
* Inner validation: `GroupShuffleSplit` over the train+val groups (15% val).
* Every fold is asserted leakage-free (no `dataset_id` in two splits) and a split
  summary is saved.

**Why GroupKFold (not plain KFold).** The three views of one dataset are highly
correlated — same generator, same structure, near-identical features. A random row
split would place, say, `x_only` in train and `xy_2d` in test, letting the model
*memorise the dataset* rather than learn the structure→utility mapping. Reported
performance would then be optimistic and would **not** transfer to genuinely new
datasets at inference. Grouping by `dataset_id` makes the test fold a true
out-of-dataset estimate — the only number that reflects the online setting, where
every dataset is unseen. Leakage prevention is not hygiene here; it is what makes
the evaluation *mean* what we need it to mean.

---

## Model — why multi-head

* **Multi-head MLP (default)** — shared trunk `250→512→512→256→256` (BatchNorm +
  GELU + Dropout) feeding six metric-family heads (`classic_cvi`, `geometry_shape`,
  `curves_lines_arcs`, `image_morphology`, `texture_frequency`,
  `topology_connectivity_neighborhood`). Head outputs are scattered back into the
  canonical target order, then `sigmoid`. The mapping is saved as `head_mapping.json` /
  `target_head_mapping.json`.
* **Single-head MLP (ablation)** — trunk → `Linear(256, 60)` → `sigmoid`.

The 60 metrics fall into **families with shared structure** (the same four/six
groupings used across the project). A shared trunk learns general
structure-description features; per-family heads then specialise — metrics within a
family (e.g. all texture metrics) co-vary and benefit from a common sub-space,
while families that respond to unrelated structure are decoupled. This is a
standard multi-task inductive bias: share what is common, separate what is not. The
single-head variant exists precisely to test whether that bias helps.

---

## Loss — why ranking and Top-K terms

```
Loss = 0.60 * MSE + 0.25 * PairwiseRankingLoss + 0.15 * TopKWeightedMSE
```

`--loss-mode mse_only` disables the ranking terms for ablation.

| Term | What it optimises | Why it is present |
|------|-------------------|-------------------|
| **MSE** | per-metric value accuracy | keeps predictions calibrated and bounded; a stable regression backbone |
| **PairwiseRankingLoss** | for metric pairs, predict the correct *order* of utilities | the search compares metrics pairwise; ordering is the property that actually drives selection |
| **TopKWeightedMSE** | extra weight on the highest-utility metrics | the top-k metrics are the ones used in the objective — errors there cost far more than errors on irrelevant low-utility metrics |

In short, the loss is shaped to the *use*: get the **ordering** and the **top of
the list** right, not merely the average value. The `mse_only` ablation quantifies
how much the ranking/Top-K terms contribute.

---

## Metrics

Regression (MSE/RMSE/MAE/R², per-metric), Top-1, Top-K overlap/ordered/utility
(K=3,5,10), ranking (Spearman, Kendall, pairwise accuracy, rank displacement), and
NDCG@{3,5,10,All}. Metrics are also computed grouped by family, subfamily, view,
difficulty and cluster count — so we can see *where* (which structural regimes) the
predictor is reliable. The ranking/Top-K/NDCG metrics, not the regression error,
are the primary success criteria, mirroring the downstream selection use.

---

## Main contributions

* **Metric-utility meta-learning**: a model that predicts, from label-free
  features, *which clustering-validity metrics to trust* on an unseen dataset.
* **Ranking-aware supervision** (pairwise + Top-K losses, ranking/NDCG evaluation)
  aligning training and assessment with the downstream top-k selection use.
* **A family-structured multi-head architecture** with a shared structural trunk,
  plus single-head and `mse_only` ablations to isolate each design choice.
* **A rigorously leakage-controlled protocol** (`GroupKFold(dataset_id)`) that
  yields an honest out-of-dataset estimate of inference-time behaviour.

## Novelty positioning

Unlike classical clustering meta-learning that predicts the *best algorithm*, this
model predicts the *evaluation objective* (metric utility) and is trained/judged as
a ranker — a learned, structure-conditioned answer to "which validity metric should
I optimise?"

## Limitations

* **Targets are noisy & redundant.** `u(D)` is search-trace-dependent and contains
  correlated metrics, capping achievable per-metric accuracy.
* **Point predictions only.** `f` outputs no uncertainty, so dynamic selection
  cannot yet weigh confidence.
* **Synthetic training distribution.** Generalisation is established within the
  `HYBRID_SCM` family distribution; real-data transfer is untested.
* **Per-view model.** It predicts per view and does not itself choose the best view.

---

## Usage

Full run (PowerShell):

```powershell
python -m models.metric_utility_mlp.run_train_cv `
  --csv-path "results_analysis\clustering_repository\analyzed_data\unified_training_dataset\unified_training_dataset.csv" `
  --repo-root "." `
  --run-name "metric_utility_mlp_v1" `
  --n-splits 5 --batch-size 512 --epochs 300 `
  --lr 0.001 --weight-decay 0.0001 --patience 25 --device auto
```

Quick smoke test (a few epochs, one fold, row-capped):

```powershell
python -m models.metric_utility_mlp.run_train_cv --debug
```

Useful flags: `--limit-rows N`, `--folds-to-run 0 1`,
`--model-type {multi_head_mlp,single_head_mlp}`,
`--loss-mode {combined,mse_only}`, `--config run.json`.

Loss weights can be overridden directly on the CLI (each flag only overrides the
config/default when provided): `--mse-weight`, `--pairwise-rank-weight`,
`--topk-mse-weight`, `--topk`, `--pairwise-margin`, `--max-pairs-per-sample`,
`--tie-epsilon`, `--topk-extra-weight`. The effective loss weights are echoed in
the `[cli] effective configuration` summary at startup.

---

## Recommended Loss Experiments

Three ready-made configs live in `models/metric_utility_mlp/configs/`. They change
**only the loss weights** (`mse_weight`, `pairwise_rank_weight`, `topk_mse_weight`);
everything else stays at the defaults. They test whether stronger ranking / Top-K
emphasis improves Top-K metric selection (baseline: MAE ≈ 0.084, Spearman ≈ 0.759,
NDCG@10 ≈ 0.908, Top-1 ≈ 0.229, Top-10 overlap ≈ 0.602).

| config | mse | pairwise | topk | purpose |
| --- | --- | --- | --- | --- |
| `mlp_baseline_combined.json`   | 0.60 | 0.25 | 0.15 | **Baseline** — reproduces the previous combined-loss run. |
| `mlp_topk_balanced.json`       | 0.40 | 0.30 | 0.30 | **Balanced** — more ranking and Top-K emphasis, solid regression. |
| `mlp_topk_aggressive.json`     | 0.30 | 0.30 | 0.40 | **Aggressive** — strongest Top-K optimisation; regression may regress slightly. |

All three use `topk = 10`. Regenerate them with:

```
python -m models.metric_utility_mlp.configs.create_default_configs
```

### Run an experiment (Windows PowerShell)

```powershell
python -u -m models.metric_utility_mlp.run_train_cv `
  --config "models\metric_utility_mlp\configs\mlp_topk_balanced.json" `
  --run-name "metric_utility_mlp_topk_balanced" `
  --device auto
```

### Run an experiment (Colab / Linux)

```bash
python -u -m models.metric_utility_mlp.run_train_cv \
  --config "/content/ClustOpt/models/metric_utility_mlp/configs/mlp_topk_balanced.json" \
  --csv-path "/content/ClustOpt/results_analysis/clustering_repository/analyzed_data/unified_training_dataset/unified_training_dataset.csv" \
  --repo-root "/content/ClustOpt" \
  --run-name "metric_utility_mlp_topk_balanced_colab" \
  --device cuda
```

### CLI-only override (no config file)

```bash
python -u -m models.metric_utility_mlp.run_train_cv \
  --csv-path ".../unified_training_dataset.csv" \
  --repo-root "..." \
  --run-name "metric_utility_mlp_topk_balanced_cli" \
  --mse-weight 0.40 \
  --pairwise-rank-weight 0.30 \
  --topk-mse-weight 0.30 \
  --topk 10 \
  --device cuda
```

Swap in `mlp_baseline_combined.json` or `mlp_topk_aggressive.json` (or change the
three weight flags) to run the other settings. When both a `--config` and
loss-weight flags are given, the **CLI flags win**.

---

## Output layout

```
results_analysis/mlp/<timestamp>__<run_name>/
  config.json  run_summary.md  fold_results.csv  overall_results.csv
  folds/fold_<k>/
    best_model.pt  last_model.pt  x_scaler.pkl
    feature_columns.json  target_columns.json  head_mapping.json
    split_summary.json  train_log.csv  val_log.csv
    test_predictions.csv  test_metrics.json  test_metrics.md
    per_metric_errors.csv  grouped_metrics/  plots/
  plots/        (cross-fold comparison plots)
  artifacts/    all_fold_test_predictions.csv  final_oof_predictions.csv
                feature_columns.json  target_columns.json
                model_architecture.txt  head_mapping.json  cv_split_summary.csv
```

---

## Reloading a trained fold

```python
import torch, pickle
from models.metric_utility_mlp.config import RunConfig
from models.metric_utility_mlp.head_groups import build_head_mapping
from models.metric_utility_mlp.model import build_model

cfg = RunConfig.from_json("results_analysis/mlp/<run>/config.json")
import json
targets = json.load(open("results_analysis/mlp/<run>/artifacts/target_columns.json"))
heads = build_head_mapping(targets, cfg.data.target_prefix)["heads"]
model = build_model(cfg.model, heads)
ckpt = torch.load("results_analysis/mlp/<run>/folds/fold_0/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
scaler = pickle.load(open("results_analysis/mlp/<run>/folds/fold_0/x_scaler.pkl", "rb"))
```

The package is isolated under `models/metric_utility_mlp/` and does not import or
modify ClustOpt, HYBRID_SCM, feature extraction or the repository builder. It is
consumed only through its saved run directory, by
[dynamic_metric_selection](../ClustOpt/cluster_validity_indices/dynamic_metric_selection/README_dynamic_metric_selection.md).
