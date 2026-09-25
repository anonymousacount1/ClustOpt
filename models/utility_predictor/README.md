# Utility-profile predictors

These models map the 250 label-free meta-features of a dataset view to a predicted utility for each validity index.

| Model | Deployed in | Output | Training data | Final model / config |
|---|---|---|---|---|
| `knn/` | **CLUSTOPT as evaluated externally (C1, C4)**; controlled configuration `configs/clustopt/knn_profile_top10/` | 60 utilities (distance-weighted mean of the 15 nearest training records) | splits 2–16; one reference set per view | `knn/<view>/{training_features,training_utilities}.npy`, `preprocessor.pkl`; configuration `n015__standard__euclidean__inverse_distance` in `knn/selected_configuration.json` |
| `neural_full60/` | neural-profile arms (C0, C3; controlled `configs/clustopt/neural_profile_top5/`); conditioned arm of the inventory factorial | 60 utilities | splits 2–16, held-out split 1 | `config.json`, `folds/fold_0/best_model.pt`, `x_scaler.pkl` |
| `neural_head14/` | 14-index arm of the inventory factorial (Tables 10–11) | 14 utilities | same | same layout |

- **Inputs.** `folds/fold_0/feature_columns.json` (250 meta-features, computed by `clustopt.meta_features.extract` into `features/features_records.csv`).
- **Targets.** `target_columns.json` (`utility__<index>`). A utility is the agreement of an index's ranking of 32 candidate partitions with their ground-truth ARI ranking.
- **Neural architecture.** A shared MLP trunk (512-512-256-256, batch norm, dropout 0.15/0.15/0.1/0.1) with per-group heads (`head_mapping.json`).
- **Neural loss.** 0.6 MSE + 0.25 pairwise ranking + 0.15 top-10 MSE.
- **Neural training.** Batch 512, up to 300 epochs, early stopping after 25, weight decay 1e-4, seed 42.
- **Training code.** `python experiments/controlled_benchmark/run.py train-neural` / `train-knn`.
- **Evaluation.** `results/controlled/predictor_evaluation/`.

Load the predictors with `clustopt.utility_prediction.load_knn()` and `clustopt.utility_prediction.load_neural("neural_full60")`. To run CLUSTOPT as reported end to end, use `clustopt.pipeline`. Configurations that use the models by their recorded location need `python scripts/prepare_workspace.py` once.
