# Reproduction guide

There are three levels, from cheapest to most expensive. Every command below is run from the repository root.

## Level 1 — the reported numbers, from the released results (minutes)

```bash
python scripts/print_paper_table.py all        # Tables 1, 2, 15, 16, 17 and the §5.3 comparison, as printed in the paper
python scripts/reproduce_paper_artifacts.py    # evidence tables, LaTeX tables and figure files -> analysis/output/paper/
python analysis/statistics/criterion_baselines.py
python analysis/statistics/controlled_paired_statistics.py
```

`docs/paper_results_index.md` lists the released file behind every table and figure, and its **Figure map** says how each figure of the paper was produced.

- **Output.** `scripts/reproduce_paper_artifacts.py` rebuilds the numbers of the tables and the figure files that the paper includes. Four figures of the paper (Figs. 1, 2, 4, 8) are drawn directly in the manuscript from released data; see the Figure map. The typeset manuscript remains the authority for layout.
- **Gallery.** The one-dataset-per-family gallery uses its frozen, deterministic selection.
- **Predictor quality and utility-selection alignment (Table 22).** `analysis/statistics/predictor_alignment.py` is **not** a Level-1 command. Its first part (predictor metrics on the held-out split) runs. Its second part needs the out-of-fold utility predictions of the training splits 2–16, which derive from the predictor training table; that table is intentionally not redistributed, so the script stops there. The frozen outputs of both parts are released in `results/controlled/predictor_alignment/`:
  - `predictor_metrics.csv`;
  - `selection_alignment.csv`;
  - `selection_alignment_per_family.csv`;
  - `predictor_alignment_provenance.json`.

## Level 2 — verify reported rows from frozen per-dataset outputs (minutes)

| Benchmark | Per-dataset data |
|---|---|
| Controlled | `results/controlled/per_dataset/controlled_best_view_results.parquet` (every arm × dataset) and `controlled_run_results.parquet` (× view); the seeded sweep in `results/controlled/criterion_baselines/seeded_sweep/dataset_results.csv` |
| External | `results/external/per_dataset/final_partitions_by_unit.jsonl.gz`: for every dataset × view × arm, the returned partition, the selected configuration and the selected indices |

To re-score the external partitions against the reconstructed ground truth, which rebuilds Table 26 and all arm means including the policy-selector arms, without any trained model:

```bash
python scripts/reconstruct_external_datasets.py
python analysis/statistics/verify_external_from_partitions.py
```

## Run CLUSTOPT as reported on a new dataset (minutes)

```bash
python -m clustopt.pipeline --input <file.csv with columns x, y> --output labels.csv
```

This is arm C4 (k-NN top-10 utility profile, search, `KNN_TOP10` reranking), with the external protocol's fixed seed and budget; see the README. `python scripts/smoke_test.py` (step J) checks that it reproduces 12 published external partitions exactly.

## Level 3 — re-run the experiments (hours to days)

```bash
python scripts/prepare_workspace.py                        # once: links the released models for the frozen configs
```

### Controlled benchmark

```bash
python experiments/controlled_benchmark/run.py generate --family standard_clustering_blobs     # one family (about 1,000 datasets); ×12 families
python experiments/controlled_benchmark/run.py features --subfamily-root <generated subfamily folder>
python experiments/controlled_benchmark/run.py utilities      -- --help   # utility targets
python experiments/controlled_benchmark/run.py training-table -- --help   # predictor training table
python experiments/controlled_benchmark/run.py train-neural   -- --help   # or use models/utility_predictor
python experiments/controlled_benchmark/run.py train-knn      -- --help
python experiments/controlled_benchmark/run.py arm --config configs/clustopt/neural_profile_top5/xy_2d.json \
       --dataset-dir <dataset folder> --view xy_2d
python experiments/controlled_benchmark/run.py batch -- --help            # arms over a subfamily and split
```

- **Seeds and splits.** Generation seeds are derived from `data/family_configs/`, and expanding them reproduces all 17,068 dataset ids (`data/synthetic_generator/dataset_catalogue.csv`). Split membership is in `data/splits/`.
- **Arm configurations.** They are in `configs/`: CLUSTOPT in `clustopt/`, the criterion baselines in `criterion_baselines/`, the inventory factorial in `controlled_benchmark/inventory_factorial/`.

### External benchmark

```bash
python experiments/external_benchmark/run.py reconstruct
python experiments/external_benchmark/run.py archive
python experiments/external_benchmark/run.py arms  -- --workers 4          # prints RUN_ID
python experiments/external_benchmark/run.py banks -- --workers 4
python experiments/external_benchmark/run.py score          --run-id RUN_ID
python experiments/external_benchmark/run.py repair-indices --run-id RUN_ID
python experiments/external_benchmark/run.py index-utility  --run-id RUN_ID
python experiments/external_benchmark/run.py core-analysis  --run-id RUN_ID
python experiments/external_benchmark/run.py tables         --run-id RUN_ID
python experiments/external_benchmark/run.py runtime        --run-id RUN_ID
```

- **Arms that can be re-run.** The CLUSTOPT arms C0, C1, C3 and **C4** run from this release. The baseline arms (A0–A3, M0–M3) also need the upstream code; see `baselines/README.md`.
- **Arms that cannot be re-run.** The auxiliary policy-selector arms **C2 and C5** need the policy-selector models, which are not distributed; for them, the released per-dataset partitions (`results/external/per_dataset/final_partitions_by_unit.jsonl.gz`) are the evidence (Level 2).
- **CVNN.** The CVNN comparator needs R and fpc: `analysis/index_analysis/reproduce_cvnn_with_fpc.py`.

**Criterion-baseline construction.** The scripts in `experiments/controlled_benchmark/criterion_baselines/` (development-set baselines, config construction, aggregation) are included for transparency. They expect the regenerated repository and its utility targets.

## Environment

- **Main environment.** Python 3.12 with `requirements.txt` (pinned) or `requirements-lock.txt` (full closure). The scikit-learn and NumPy pins are needed for the released scalers to unpickle identically.
- **ML2DAC-based baselines.** These use a separate Python 3.9 environment (`baselines/README.md`).
- **Unit tests.** `python -m pytest src/models/metric_utility_knn/tests src/models/Clustering_Repository_Builder/experiments/paper_analysis/tests` (requires `pytest`).

## Known numerical limitations

- **Ties.** Floating-point summation order can resolve exact ties between candidates differently across machines. Aggregates are unaffected at reported precision.
- **PCA.** The external preparation includes a PCA, and a different linear-algebra backend could change the last bits; `reconstruct` verifies every dataset against its checksum. The release was checked with 50/50 checksums matching in a fresh environment.
