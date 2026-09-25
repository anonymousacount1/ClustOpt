# Unified Raw MKR Builder (Experiment 3, Phase 1)

Builds **one unified raw meta-knowledge repository** per subfamily, from which the
four in-domain derived repositories can later be constructed (NOT in this phase):

1. ML2DAC (original CVIs) · 2. ML2DAC + ClustOpt metrics ·
3. AutoClust (original CVIs) · 4. AutoClust + ClustOpt metrics

See [`UNIFIED_MKR_DESIGN.md`](UNIFIED_MKR_DESIGN.md) for the authoritative spec.

## What it does

For every dataset view (`x_only`, `y_only`, `xy_2d`) it:

1. Resolves **exactly 32 clustering configurations** from the ClustOpt utility
   maps, replacing every `HDBSCAN_CONSTRAINED` config (see below).
2. Runs each configuration (reusing ClustOpt's clustering wrappers) and stores
   the cluster labels as `.npy`.
3. Computes external metrics (ARI / AMI / NMI / FMI) against ground truth.
4. Computes the **de-duplicated union** of internal metrics:
   60 ClustOpt utility-map metrics + 3 ML2DAC-only CVIs (Dunn / Coggins-Jain /
   COP). The four CVIs shared with ClustOpt (CH/DBI/SIL/DBCV) are computed once.
5. Extracts **ML2DAC** (pymfe) and **AutoClust** (MeanShift landmarking)
   meta-features per view record.

## Run

```bash
python experiments/external_baselines/Unified_MKR/runners/run_subfamily_unified_mkr.py \
  --subfamily-root "results_analysis/clustering_repository/analyzed_data/arcs_circles_rings/subfamilies/annuli_variable_thickness" \
  --output-root    "experiments/external_baselines/Unified_MKR/outputs" \
  --n-jobs 8 --resume
```

Key flags: `--config-root` (default `models/ClustOpt/configs/utilities_maps`),
`--split-assignment-csv` (default repo standard), `--metafeatures {auto,on,off}`,
`--overwrite`, `--limit-datasets N`, `--dry-run`, `--n-jobs 1` (debug, no pool).

## Outputs (`<output-root>/<family>/<subfamily>/`)

```
schema_version.json  build_config.json
records.parquet  metafeatures.parquet  configurations.parquet
metric_registry.parquet  evaluations.parquet  failures.parquet
labels/<record_id>/<config_id>.npy
logs/  validation/{integrity,coverage,cache}_report.json
_datasets/<dataset_id>/...        # per-dataset staging (resume granularity)
```

`evaluations.parquet` has one row per `(record_id, config_id)` with external
metrics + every unified internal metric column. Labels live only as `.npy`
(never inside Parquet).

## HDBSCAN_CONSTRAINED replacement

The 4 constrained-HDBSCAN slots per view are removed and re-assigned one each,
round-robin across the other algorithms already in the map (sorted by name →
`agglomerative`, `birch`, `dbscan`, `gmm` each gain one). Each replacement
extends that algorithm's primary numeric hyper-parameter past its current max
using the algorithm's own step size — no new algorithm or hyper-parameter
dimension is invented. Provenance (`was_replaced`, `original_config_id`,
`replacement_reason`) is stored in `configurations.parquet`. Total stays 32.

## Environment note

Runs in the ClustOpt env (Python 3.12). Clustering, all 60 ClustOpt metrics,
external metrics, the 3 ML2DAC-only CVIs, and **AutoClust** meta-features all run
natively. **ML2DAC meta-features need `pymfe`**, which is intentionally kept out
of the ClustOpt env (it forces pandas/numpy upgrades). Instead a dedicated
minimal venv `.mfenv` holds pymfe:

```bash
bash experiments/external_baselines/Unified_MKR/setup/create_mf_env.sh   # or .ps1 on Windows
```

When `.mfenv` exists, the subfamily runner **auto-runs** the ML2DAC meta-feature
pass at the end of the build (no manual env switching). It can also be run
standalone via `runners/run_ml2dac_metafeatures_pass.py`, pointed at any output
dir. Without `.mfenv`, ML2DAC meta-features are left `ml2dac_status="unavailable"`
(NaN) and everything else still completes.

`pyarrow` is required in the ClustOpt env for the Parquet outputs. See
`COVERAGE_AND_READINESS.md` for the full ML2DAC/AutoClust coverage proof and
`CONFIG_SPACE_REPORT.md` for the resolved 32-config tables.
