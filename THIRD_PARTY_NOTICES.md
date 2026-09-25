# Third-party notices

This repository contains code derived from the third-party works below. Their licences apply to the corresponding portions and are reproduced in full.

## CVDD (Clustering Validation index based on Density-involved Distance)

- Location: `src/models/Independent_Domain_Benchmark/metric_validation/external_cvis.py` (CVDD section).
- Relationship: Python port of the author implementation `github.com/hulianyu/CVDD` at commit `32cabe0cf876ccd4e04e44dda84b0ad96be88bbc`, semantics preserved.
- Citation: L. Hu and C. Zhong, "An Internal Validity Index Based on Density-Involved Distance," *IEEE Access* 7:40038-40051, 2019.

```
MIT License

Copyright (c) 2019 Lianyu Hu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CDbw (Composed Density between and within clusters)

- Location: `src/models/Independent_Domain_Benchmark/metric_validation/external_cvis.py` (CDbw section).
- Relationship: calls the PyPI package `cdbw` 0.2 (S. Rubinsky, A. Lashkov, P. Eistrikh-Heller; `github.com/alashkov83/CDbw`) for representatives, shrinking and densities, and re-implements the separation step from the package code with three documented corrections (`CDBW_PAPER_CORRECTIONS`; audit in `src/models/Independent_Domain_Benchmark/configs/cdbw_definition_audit.json`). Requires `cdbw==0.2`.
- Citation: M. Halkidi and M. Vazirgiannis, "A density-based cluster validity approach using multi-representatives," *Pattern Recognition Letters* 29(6):773-786, 2008.

```
MIT License

Copyright (c) 2019 Alexander Lashkov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CVNN (Clustering Validation index based on Nearest Neighbours)

- Not implemented in this repository. CVNN is used only in the auxiliary external index analysis (it is not one of CLUSTOPT's 60 indices).
- `src/models/Independent_Domain_Benchmark/metric_validation/cvnn_fpc_adapter.py` calls the external R package **fpc** (C. Hennig, GPL) in a separate process. fpc is not distributed with this release and must be installed by the user. The frozen CVNN values are provided in `results/external/index_analysis/authoritative/cvnn_frozen_values.csv`, and fpc reproduces them on all 150 units.
- Citation: Y. Liu, Z. Li, H. Xiong, X. Gao, J. Wu, S. Wu, "Understanding and enhancement of internal clustering validation measures," *IEEE Trans. Cybernetics* 43(3):982-994, 2013. C. Hennig, *fpc: Flexible Procedures for Clustering*, R package.

## Methods cited but not distributed

- DCSI: independent implementation from J. Gauss, F. Scheipl, M. Herrmann, "DCSI -- An improved measure of cluster separability based on separation and connectedness" (2023/2024).
- DBCV: called through `hdbscan.validity.validity_index` (hdbscan, BSD-3-Clause); D. Moulavi et al., SDM 2014.
- ML2DAC and AutoML4Clust are **not** included; see `baselines/README.md`.
- AutoClust: our in-domain implementation follows Y. Poulakis, C. Doulkeridis, D. Kyriazis, "AutoClust: A Framework for Automated Clustering based on Cluster Validity Indices," ICDM 2020.

Runtime dependencies (scikit-learn, NumPy, SciPy, pandas, PyTorch, Optuna, hdbscan, scikit-image, shapely, networkx, OpenCV, cdbw, ...) are installed from `environment/` and are not redistributed.
