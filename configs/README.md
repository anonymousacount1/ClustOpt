# Configurations

These are byte-identical copies of the frozen configurations, under descriptive names. The originals, which are the ones used by the batch runners, are in `src/models/ClustOpt/configs/experiments/` and `src/models/Independent_Domain_Benchmark/configs/`. The mapping is in `docs/internal_to_release_mapping.md`.

| Folder | Content |
|---|---|
| `clustopt/knn_profile_top10/` | **CLUSTOPT as reported (arm C4)**: search with the k-NN predicted profile, top-10, raw weights, one file per view; reranking with `KNN_TOP10` follows the search (`clustopt.pipeline`) |
| `clustopt/neural_profile_top5/` | the neural-profile variant (neural predicted profile, top-5, raw weights; arms C0/C3; the conditioned arm of the inventory factorial) |
| `clustopt/ablations/` | subset-size and weighting variants; `oracle_*` use the true utility profile (ceiling) |
| `clustopt/candidate_bank_32/` | the fixed 32-configuration candidate bank used for utility targets and the index analysis |
| `criterion_baselines/` | single-index searches, uniform weighting, random search, static generalist core, best single index, mean profile |
| `controlled_benchmark/inventory_factorial/` | inventory x criterion factorial (60 vs 14 indices x uniform / predicted / oracle) |
| `controlled_benchmark/weighting_ablation/` | raw vs softmax weighting of the selected indices |
| `external_benchmark/` | protocol, arm registry and index registries of the external benchmark |
| `baselines/` | ML2DAC and AutoML4Clust adapter settings. `baselines/autoclust/` belongs to AutoClust's native adapter, which is **not** included and **not** reported; the reported AutoClust arms are configured as described in `baselines/README.md` |
