# Baselines

This page is the canonical description of how the baselines were run.

## What is reported

| Paper arm | System and index inventory | Benchmark | Our code | Configuration | Results |
|---|---|---|---|---|---|
| AutoClust (matched) | AutoClust (Poulakis et al., ICDM 2020), original / +established / +new / extended inventory | controlled | `src/experiments/external_baselines/AutoClust/indomain/`, run by `src/experiments/external_baselines/AutoClust/runners/run_autoclust_indomain_all.py` | `--variant AutoClust_<inventory>_InDomain_same_search_space`: the `_same_search_space` suffix restricts AutoClust's search to CLUSTOPT's search space (the one in `configs/clustopt/knn_profile_top10/`); meta-knowledge from `src/experiments/external_baselines/Unified_MKR/` on splits 2–16 | `results/controlled/matched_autoclust/`, `results/controlled/baseline_inventory_decomposition/autoclust/` |
| A0–A3 | same four inventories | external | `src/models/Independent_Domain_Benchmark/methods/autoclust*.py` | `configs/external_benchmark/method_registry.json` | `results/external/` |
| ML2DAC (matched) | ML2DAC (Treder-Tschechlov et al., EDBT 2023), four inventories | controlled | `src/experiments/external_baselines/ML2DAC/`, `src/experiments/external_baselines/Unified_MKR/` | `configs/baselines/ml2dac/` | `results/controlled/baseline_inventory_decomposition/ml2dac/` |
| M0–M3 | same four inventories | external | `src/models/Independent_Domain_Benchmark/methods/ml2dac*.py` | `configs/external_benchmark/method_registry.json` | `results/external/` |
| AutoML4Clust | AutoML4Clust (Tschechlov et al., EDBT 2021), silhouette, its own search space and budget (supplementary, unmatched) | controlled only | `src/experiments/external_baselines/AutoML4Clust/` | `configs/baselines/automl4clust/` | `results/controlled/criterion_baselines/canonical_frame/controlled_criterion_baselines.csv` |

### What we changed relative to the original systems
- AutoClust and ML2DAC search CLUSTOPT's configuration space, with the same budget and views.
- Their meta-knowledge is rebuilt on our training splits (2–16) only.
- Their selection logic follows the published descriptions.
- The ML2DAC-specific indices (Dunn, Coggins-Jain, COP) are evaluated with the upstream ML2DAC implementation.
- AutoClust's native, unconstrained-search variant is not reported, and its adapter is not included. The files that describe it and that remain for provenance (`configs/baselines/autoclust/`, `src/experiments/external_baselines/AutoClust/{configs,setup}/` and `README_AutoClust_baseline.md`) are **not** used by the reported arms. The replay result files contain one auxiliary row for this variant (`AutoClust_Extended_InDomain`, 0.6171); see `results/controlled/README.md`.

## Upstream code (fetch; not included)

The upstream repositories declare no licence, so no upstream code, model or data is redistributed here.

| Repository | Commit | Needed for |
|---|---|---|
| `https://github.com/tschechlovdev/ml2dac` | `c946b9eaf58d9e7a566e95f785303da850747653` | all ML2DAC arms; the AutoClust +established / extended arms (ML2DAC-only indices) |
| `https://github.com/tschechlovdev/Automl4Clust` | `d67351fd1b1e97aaf351ce491e8a976dcbcffc51` | AutoML4Clust |

```bash
git clone https://github.com/tschechlovdev/ml2dac.git src/external/ml2dac
git -C src/external/ml2dac checkout c946b9eaf58d9e7a566e95f785303da850747653
git clone https://github.com/tschechlovdev/Automl4Clust.git src/external/Automl4Clust
git -C src/external/Automl4Clust checkout d67351fd1b1e97aaf351ce491e8a976dcbcffc51
```

## Rebuild order (controlled benchmark)

1. **Regenerate the corpus.** Then compute utility targets and meta-features (`docs/reproduction.md`, Level 3).
2. **ML2DAC environment.** Create it with Python 3.9: `src/experiments/external_baselines/ML2DAC/setup/` (conda recipe and install scripts).
3. **Rebuild the meta-knowledge** (about 9 GB) with the builder in `src/experiments/external_baselines/Unified_MKR/runners/`.
4. **Run the in-domain arms.** Use `src/experiments/external_baselines/{AutoClust,ML2DAC}/runners/run_*_indomain_all.py` (see `--help`).

The external-benchmark arms run through `python experiments/external_benchmark/run.py arms`.
