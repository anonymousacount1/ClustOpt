# CLUSTOPT

**Tailoring the clustering objective to the data: utility-guided validity-index selection and candidate reranking.**
Code, trained models and result tables for the anonymous submission.

## What CLUSTOPT is

Choosing a clustering configuration without labels needs a criterion, and no single cluster-validity index works across geometries. CLUSTOPT composes the criterion *per dataset*:

1. **Meta-features.** Label-free meta-features describe the dataset view.
2. **Utility profile.** A learned predictor turns them into a *utility profile*: for each of 60 validity indices, how well that index would rank candidate partitions of this data.
3. **Search objective.** The search maximises a weighted combination of the indices the profile rates highest.
4. **Reranking.** A learned reranker returns the final partition from among the candidates the search visited.

The configuration reported in the paper as CLUSTOPT (arm **C4**) uses the k-NN utility predictor, keeps the 10 indices it rates highest, and reranks with the `KNN_TOP10` reranker. [`clustopt.pipeline`](#run-clustopt-on-a-dataset) runs exactly this.

The 60 indices include 46 new structural indices (22 pattern-based, 24 image-based). The component-to-code map is in [`docs/methodology.md`](docs/methodology.md).

## Where are the paper's results?

**[`docs/paper_results_index.md`](docs/paper_results_index.md)** maps every table and figure to the released files, the command that prints or rebuilds it, and its protocol. For example:

```bash
python scripts/print_paper_table.py table1        # also: table2, table15, table16, table17, section5_3, all
python scripts/reproduce_paper_artifacts.py       # evidence tables, tables and figures from results/
```

## What is included, and what is not

| Folder | Content |
|---|---|
| `clustopt/` | public Python interface (see [Code layout](#code-layout)) |
| `src/` | the frozen research implementation (see [Code layout](#code-layout)) |
| `data/` | generator configurations for 12 families / 86 subfamilies; the catalogue and seeds of all 17,068 generated datasets; the 16 splits; the external-benchmark manifest (identifiers, preparation, checksums) |
| `models/` | [utility-profile predictors](models/utility_predictor/README.md) and [candidate rerankers](models/candidate_reranker/README.md) |
| `configs/` | configurations of CLUSTOPT, the criterion baselines, the controlled and external benchmarks and the baselines |
| `results/` | frozen results behind every reported number: [controlled](results/controlled/README.md), [external](results/external/README.md) |
| `analysis/`, `scripts/` | scripts that print or rebuild tables and figures, verify results, set up and smoke-test |
| `experiments/` | public entry points for re-running the benchmarks |
| `baselines/` | [how AutoClust, ML2DAC and AutoML4Clust were run](baselines/README.md) |

**Not included:**
- **External data**, raw or prepared ([data policy](docs/external_data.md)). The 50 external datasets are rebuilt from their public sources and verified against checksums.
- **The generated corpus.** It is regenerated from the generator, configurations and seeds.
- **Upstream baseline code.** It declares no licence; fetch it at the pinned commits (see [baselines](baselines/README.md)).
- **The policy-selector models.** They belong to the auxiliary arms C2 and C5, which therefore cannot be re-run from this release; their per-dataset partitions are included. CLUSTOPT as reported (C4) does not use them.

## Install and smoke test

Python 3.12, CPU only.

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python scripts/prepare_workspace.py                  # once
python scripts/smoke_test.py                         # about 10 minutes; writes smoke_test_report.json
```

The smoke test covers:
- the imports;
- generating a synthetic dataset from a released configuration and running a CLUSTOPT search on it;
- loading the predictors and a reranker;
- reconstructing an external dataset against its checksum;
- rebuilding tables and figures from `results/`;
- running CLUSTOPT as reported (C4, and C1 without reranking) through `clustopt.pipeline` on two external datasets × three views, and checking that all 12 partitions equal the published ones.

## Run CLUSTOPT on a dataset

`clustopt.pipeline` runs CLUSTOPT as reported in the paper (arm C4) on one two-dimensional dataset:
1. meta-features;
2. the k-NN utility profile, keeping the top 10 indices;
3. a 50-evaluation search;
4. the `KNN_TOP10` reranker.

It calls the implementation that produced the external results, with the external protocol's fixed seed, so the output is deterministic. Ground truth is not used.

Start Python in the repository root, the folder that contains `clustopt/`. There is no installable package, so a script saved elsewhere must first add that folder to `sys.path`.

```python
from sklearn.datasets import make_moons
from clustopt import pipeline

X, _ = make_moons(n_samples=400, noise=0.06, random_state=0)   # any (n, 2) array
res = pipeline.run(X)                                         # C4; view "xy_2d"
print(res.algorithm, res.selected_indices, res.labels[:20])
```

The same from the command line, on a CSV with columns `x` and `y` (or a `.npy` array):

```bash
python -c "from sklearn.datasets import make_moons; import pandas as pd; X, _ = make_moons(400, noise=0.06, random_state=0); pd.DataFrame(X, columns=['x', 'y']).to_csv('moons.csv', index=False)"
python -m clustopt.pipeline --input moons.csv --output moons_labels.csv
```

Options:
- `view="x_only"` / `"y_only"` fits the partition on one coordinate.
- `reranked=False` gives arm C1: the same search, returning the objective's maximiser.

A run takes about half a minute on a laptop CPU. The released models are linked on first use (as `scripts/prepare_workspace.py` does).

## Code layout

| Folder | Role |
|---|---|
| `clustopt/` | **the supported public interface**: `pipeline` (CLUSTOPT as reported), `utility_prediction`, `reranking`, `validity_indices`, `meta_features`, `search`, `clustering`. Thin wrappers; they add no algorithm |
| `experiments/` | **the supported experiment entry points**: `controlled_benchmark/run.py`, `external_benchmark/run.py` |
| `analysis/`, `scripts/` | paper-result analysis, the table printer, figure/table rebuild, setup and the smoke test |
| `src/` | **the frozen research implementation**, under the module names it had when the reported results were produced. The names are kept because frozen configurations, recorded provenance and run identities refer to them. Internal file names such as `run_stage4c_*`, `stage2c/` or `phaseD_*` record development stages; they are not part of the public interface. Use the wrappers above |

## Reproducing more

[`docs/reproduction.md`](docs/reproduction.md) describes three levels:
1. reported numbers from the released results;
2. verification from per-dataset outputs;
3. full re-runs of the controlled benchmark (`experiments/controlled_benchmark/run.py`) and the external benchmark (`experiments/external_benchmark/run.py`).

## Licence and notices

- Code: [`LICENSE`](LICENSE).
- Third-party code: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- Dataset sources and citations: [`DATA_SOURCES.md`](DATA_SOURCES.md).
