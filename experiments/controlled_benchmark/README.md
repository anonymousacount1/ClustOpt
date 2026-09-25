# Controlled benchmark

Run it with `python experiments/controlled_benchmark/run.py --help`. Available steps: `generate`, `features`, `utilities`, `training-table`, `splits`, `train-neural`, `train-knn`, `arm`, `batch`. The step order is in [`docs/reproduction.md`](../../docs/reproduction.md), Level 3.

- **Arm configurations:** `configs/` (see [`configs/README.md`](../../configs/README.md)).
- **Frozen results:** [`results/controlled/`](../../results/controlled/README.md).

`criterion_baselines/` holds the scripts that built the criterion-baseline arms of the seeded sweep (development-set baselines, configuration construction, batch driver, aggregation). They are included for transparency and expect the regenerated repository with its utility targets.
