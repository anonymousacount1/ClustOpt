# External data policy

**We redistribute no external data**: no raw source file, cleaned matrix, PCA projection or ground-truth vector.

Each of the 50 external datasets is reconstructed locally from its public identifier and verified against the checksum of the frozen two-dimensional representation that every system consumed.

## Identifiers and preparation

`data/external_benchmark/dataset_manifest.{csv,json}` holds, per dataset:
- the source;
- the identifier:
  - the FCPS file name;
  - the OpenML data id;
  - the scikit-learn loader;
  - or the scikit-learn generator with its difficulty level and seed;
- the preparation parameters;
- the `representation_checksum`.

`data/external_benchmark/prepared_representation_manifest.json` records the preparation of every dataset (cleaning, standardisation, PCA explained variance) together with its checksum.

## Reconstruction

```bash
python scripts/reconstruct_external_datasets.py                 # fetch + prepare + verify all 50
python scripts/reconstruct_external_datasets.py --verify-only   # verify only, write nothing
```

Arrays are written only to your local `src/results_analysis/independent_domain_benchmark/publication/corpus/`. Never commit them.

**Sources**
- FCPS files are downloaded from the public mirror named in the manifest.
- OpenML datasets are fetched with `sklearn.datasets.fetch_openml(data_id=…)`.
- The breast-cancer (WDBC) dataset is the UCI Breast Cancer Wisconsin (Diagnostic) dataset (CC BY 4.0), loaded through the copy bundled with scikit-learn 1.6.0.
- The scikit-learn generator datasets are generated deterministically.

The preparation includes a PCA for 26 real sources. Under a different linear-algebra backend the projection of a high-dimensional source can differ in the last bits, and the checksum reports this rather than accepting it.

## Licences and citations

See `DATA_SOURCES.md`. Every dataset must be used under its provider's terms and cited as listed.

The manifest fields `licence` / `redistribution` are strings written automatically by the preparation code. They are kept unchanged because the files are covered by recorded hashes, but they are **not** licence determinations: `DATA_SOURCES.md` is authoritative.

## What is released about external data

- **Our outputs on it.** Per-dataset results, the partitions each system returned (cluster assignments only, no feature values), and the 32-candidate partitions used by the index analysis.
- **Checksums.**
