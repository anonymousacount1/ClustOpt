# Data

| Folder | Content |
|---|---|
| `family_configs/` | the 12 family generation plans (all subfamily parameterizations, family seed, difficulty and point-count policies) |
| `synthetic_generator/dataset_catalogue.csv` | all 17,068 generated datasets: `dataset_id`, family, subfamily, cluster count, difficulty, point count, `family_seed`, `dataset_seed` (the seeds are exactly those derived from `family_configs/`) |
| `splits/` | assignment of the 17,068 datasets to the 16 splits, with stratification summaries (its `n_points` column is the planned point count used for stratification; the generated count is in `dataset_catalogue.csv`) |
| `external_benchmark/` | identifiers, preparation parameters and checksums of the 50 external datasets; no data values |

The generator code is `src/models/HYBRID_SCM`. Generated datasets are not shipped; see `docs/reproduction.md`.
