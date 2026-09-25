# Trained models

| Folder | Content |
|---|---|
| `utility_predictor/neural_full60/` | neural utility-profile predictor over 60 indices (`config.json`, `folds/fold_0/best_model.pt`, input scaler, feature/target/head metadata; `artifacts/feature_columns.json` is an identical copy of the fold's feature list, at the path the reranker context reads). Profile of arms C0/C3; also half of the reranker's input context, so C4 uses it too |
| `utility_predictor/neural_head14/` | the same architecture over the 14 established indices (inventory ablation) |
| `utility_predictor/knn/` | k-NN predictor: one reference set of training meta-features and utilities per view, plus the selected configuration. **The utility profile of CLUSTOPT as reported (C4)** |
| `candidate_reranker/<POLICY>/model.joblib` | one reranker per selection policy (MLP/KNN profile x top-1/3/5/10/dynamic); `reranker_registry.json` and `policy_to_reranker_mapping.json` |

All models were trained on splits 2-16 only.

The policy-selector models (auxiliary arms C2, C5) are not included, so those two arms cannot be re-run. Their per-dataset outputs are in `results/external/per_dataset/final_partitions_by_unit.jsonl.gz` (arms C2, C5), and their training code is in `src/models/ClustOpt_Policy_Predictor`.
