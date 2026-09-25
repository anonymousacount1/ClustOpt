# Candidate rerankers

After the search, a reranker scores every candidate partition the search visited and returns the best-scored one instead of the objective's maximiser.

| | |
|---|---|
| **Deployed in** | CLUSTOPT with reranking: external arm **C4** uses `KNN_TOP10` and C3 uses `MLP_TOP5`; the controlled "CLUSTOPT, reranked" row of Table 1 uses `KNN_TOP10`. The policy-selector arm C5 uses the reranker of whichever policy it selects |
| **One model per selection regime** | `{MLP,KNN}_{TOP1,TOP3,TOP5,TOP10,DYNAMIC}/model.joblib` (scikit-learn `HistGradientBoostingRegressor`). Which policies each serves: `policy_to_reranker_mapping.json`; file hashes and sizes: `reranker_registry.json` |
| **Features** | 347 label-free features per candidate, describing the candidate within its visited slate and the dataset view (definitions in `src/models/ClustOpt_Candidate_Reranker/data/`) |
| **Target** | slate regret: how far the candidate's ARI is below the best candidate of the same slate (ground truth is used only in training) |
| **Training data** | visited candidates of the controlled splits 2–16 (1,161,522 rows; `training_population.csv`, `model_training_summary.csv`). Held-out split 1 was never used |
| **Evaluation** | `results/controlled/reranking/` (Tables 13–14) and `results/external/paired_statistics/reranker_summary.csv` |

Load a reranker with `clustopt.reranking.load("KNN_TOP10")`. Training code is in `src/models/ClustOpt_Candidate_Reranker/training/`.
