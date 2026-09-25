# Analysis

| Folder | Content |
|---|---|
| `paper/` | builders for the paper's evidence tables, LaTeX tables and figures (`python scripts/reproduce_paper_artifacts.py`) |
| `statistics/` | paired statistics and criterion baselines. `criterion_baselines.py` and `controlled_paired_statistics.py` run on `results/`. `external_paired_statistics.py` needs the reconstructed external data. `predictor_alignment.py` runs only its first part: the second needs training-split predictions that are not redistributed; its frozen outputs are in `results/controlled/predictor_alignment/` |
| `index_analysis/` | optional CVNN reproduction with R fpc |

`release_paths.py` and `legacy_path_map.json` resolve the input locations named in the scripts to the released files. The scientific logic of the scripts is unchanged. Outputs go to `analysis/output/`.
