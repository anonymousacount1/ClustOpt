# Implementation

This folder is the scientific implementation under its original module names. Keeping those names preserves imports, frozen configuration files and the configuration hashes recorded with the results.

The public entry points are in `clustopt/` (CLUSTOPT as reported: `clustopt.pipeline`) and `experiments/`. File names inside `src/` that mention development stages (`run_stage4c_*`, `stage2c/`, `phaseD_*`, `e1_*`) are internal; they are not part of the public interface. The mapping between released files and their original locations is in `docs/internal_to_release_mapping.md`.

| Package | Content |
|---|---|
| `models/ClustOpt` | search, 60 validity indices, dynamic (predicted-profile) objectives, configurations |
| `models/HYBRID_SCM` | the synthetic dataset generator |
| `models/Clustering_Repository_Builder` | utility targets, meta-features, splits, experiment execution and aggregation |
| `models/metric_utility_mlp`, `models/metric_utility_knn` | utility-profile predictors |
| `models/ClustOpt_Candidate_Reranker`, `models/ClustOpt_Policy_Predictor` | reranker and policy selector |
| `models/Independent_Domain_Benchmark` | the external benchmark |
| `experiments/external_baselines` | our AutoClust / ML2DAC / AutoML4Clust adapters and the meta-knowledge builder |
| `experiments/metric_atlas` | index attribution and family-specialisation analysis |
| `scripts/stage1`, `scripts/stage2` | factorial-experiment and reranker-data drivers |

Run `python scripts/prepare_workspace.py` once, so that configurations naming the trained models by their recorded location find them.
