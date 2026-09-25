# Data sources

**No external dataset array is redistributed in this repository**: no raw source file, no cleaned matrix, no PCA projection and no ground-truth vector. Every external dataset is reconstructed locally from its public identifier by `src/models/Independent_Domain_Benchmark/datasets/` (FCPS download, OpenML `fetch_openml(data_id=...)`, scikit-learn bundled data, or a seeded scikit-learn generator) and prepared with the frozen parameters in `data/external_benchmark/dataset_manifest.{csv,json}`. The reconstruction is checked against the frozen `representation_checksum` of the two-dimensional array (`scripts/reconstruct_external_datasets.py` rebuilds each dataset locally and refuses a mismatch).

The synthetic controlled repository (17,068 datasets) is not shipped either; it is regenerated from `src/models/HYBRID_SCM` with the family/subfamily configurations and the per-dataset seeds recorded in `data/synthetic_generator/dataset_catalogue.csv`.

> **Superseded field.** `dataset_manifest.{csv,json}` (`licence`) and `data/external_benchmark/prepared_representation_manifest.json` (`redistribution`) carry licence strings written automatically by the preparation code. They are kept byte-identical because they are covered by recorded hashes, but they are **not licence determinations**: in particular the FCPS mirror declares no licence (not MIT), and the UCI-derived sources are CC BY 4.0. The table below is the audited status.

Using a dataset requires respecting its provider's terms and citing it as listed.

| Dataset | Source | Identifier | Provider licence (audited) | Cite |
|---|---|---|---|---|
| fcps_atom | FCPS via `github.com/deric/clustering-benchmark` | atom.arff | none declared by the mirror; ARFF header requires citation | A. Ultsch, *Clustering with SOM: U\*C*, Proc. Workshop on Self-Organizing Maps, Paris, 2005, pp. 31-37 |
| fcps_chainlink | FCPS via `github.com/deric/clustering-benchmark` | chainlink.arff | none declared by the mirror; ARFF header requires citation | A. Ultsch, *Clustering with SOM: U\*C*, Proc. Workshop on Self-Organizing Maps, Paris, 2005, pp. 31-37 |
| fcps_engytime | FCPS via `github.com/deric/clustering-benchmark` | engytime.arff | none declared by the mirror; ARFF header requires citation | A. Ultsch, *Clustering with SOM: U\*C*, Proc. Workshop on Self-Organizing Maps, Paris, 2005, pp. 31-37 |
| fcps_lsun | FCPS via `github.com/deric/clustering-benchmark` | lsun.arff | none declared by the mirror; ARFF header requires citation | A. Ultsch, *Clustering with SOM: U\*C*, Proc. Workshop on Self-Organizing Maps, Paris, 2005, pp. 31-37 |
| fcps_tetra | FCPS via `github.com/deric/clustering-benchmark` | tetra.arff | none declared by the mirror; ARFF header requires citation | A. Ultsch, *Clustering with SOM: U\*C*, Proc. Workshop on Self-Organizing Maps, Paris, 2005, pp. 31-37 |
| fcps_twodiamonds | FCPS via `github.com/deric/clustering-benchmark` | twodiamonds.arff | none declared by the mirror; ARFF header requires citation | A. Ultsch, *Clustering with SOM: U\*C*, Proc. Workshop on Self-Organizing Maps, Paris, 2005, pp. 31-37 |
| fcps_wingnut | FCPS via `github.com/deric/clustering-benchmark` | wingnut.arff | none declared by the mirror; ARFF header requires citation | A. Ultsch, *Clustering with SOM: U\*C*, Proc. Workshop on Self-Organizing Maps, Paris, 2005, pp. 31-37 |
| real_analcatdata_authorship | OpenML | OpenML data id 458 | not established (see provider) | J. S. Simonoff, *Analyzing Categorical Data*, Springer, 2003 |
| real_balance_scale | OpenML (UCI origin) | OpenML data id 11 | CC BY 4.0 (UCI Machine Learning Repository) | R. S. Siegler. *Balance Scale*. UCI Machine Learning Repository. doi:10.24432/C5488X |
| real_banknote | OpenML (UCI origin) | OpenML data id 1462 | CC BY 4.0 (UCI Machine Learning Repository) | V. Lohweg. *Banknote Authentication*. UCI Machine Learning Repository. doi:10.24432/C55P57 |
| real_blood_transfusion | OpenML (UCI origin) | OpenML data id 1464 | CC BY 4.0 (UCI Machine Learning Repository) | I.-C. Yeh. *Blood Transfusion Service Center*. UCI Machine Learning Repository. doi:10.24432/C5GS39 |
| real_breast_cancer_wdbc | UCI copy bundled with scikit-learn 1.6.0 | `sklearn.datasets.load_breast_cancer` | CC BY 4.0 (UCI Machine Learning Repository) | W. Wolberg, O. Mangasarian, N. Street, W. Street. *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. doi:10.24432/C5DW2B |
| real_car_evaluation | OpenML (UCI origin) | OpenML data id 21 | CC BY 4.0 (UCI Machine Learning Repository) | M. Bohanec. *Car Evaluation*. UCI Machine Learning Repository. doi:10.24432/C5JP48 |
| real_cardiotocography_3c | OpenML (UCI origin) | OpenML data id 1560 | CC BY 4.0 (UCI Machine Learning Repository) | D. Campos, J. Bernardes. *Cardiotocography*. UCI Machine Learning Repository. doi:10.24432/C51S4N |
| real_churn | OpenML | OpenML data id 40701 | not established (see provider) | OpenML data id 40701 |
| real_climate_model_crashes | OpenML (UCI origin) | OpenML data id 1467 | CC BY 4.0 (UCI Machine Learning Repository) | D. Lucas, R. Klein, J. Tannahill, D. Ivanova, S. Brandon, D. Domyancic, Y. Zhang. *Climate Model Simulation Crashes*. UCI Machine Learning Repository. doi:10.24432/C5HG71 |
| real_cmc | OpenML (UCI origin) | OpenML data id 23 | CC BY 4.0 (UCI Machine Learning Repository) | T.-S. Lim. *Contraceptive Method Choice*. UCI Machine Learning Repository. doi:10.24432/C59W2D |
| real_credit_approval | OpenML (UCI origin) | OpenML data id 29 | CC BY 4.0 (UCI Machine Learning Repository) | J. R. Quinlan. *Credit Approval*. UCI Machine Learning Repository. doi:10.24432/C5FS30 |
| real_diabetes_pima | OpenML | OpenML data id 37 | not established (see provider) | J. W. Smith et al., Proc. Symp. Computer Applications in Medical Care, 1988 |
| real_german_credit | OpenML (UCI origin) | OpenML data id 31 | CC BY 4.0 (UCI Machine Learning Repository) | H. Hofmann. *Statlog (German Credit Data)*. UCI Machine Learning Repository. doi:10.24432/C5NC77 |
| real_haberman | OpenML (UCI origin) | OpenML data id 43 | CC BY 4.0 (UCI Machine Learning Repository) | S. Haberman. *Haberman's Survival*. UCI Machine Learning Repository. doi:10.24432/C5XK51 |
| real_hill_valley | OpenML (UCI origin) | OpenML data id 1479 | CC BY 4.0 (UCI Machine Learning Repository) | L. Graham, F. Oppacher. *Hill-Valley*. UCI Machine Learning Repository. doi:10.24432/C5JC8P |
| real_ilpd_liver | OpenML (UCI origin) | OpenML data id 1480 | CC BY 4.0 (UCI Machine Learning Repository) | B. Ramana, N. Venkateswarlu. *ILPD (Indian Liver Patient Dataset)*. UCI Machine Learning Repository. doi:10.24432/C5D02C |
| real_ionosphere | OpenML (UCI origin) | OpenML data id 59 | CC BY 4.0 (UCI Machine Learning Repository) | V. Sigillito, S. Wing, L. Hutton, K. Baker. *Ionosphere*. UCI Machine Learning Repository. doi:10.24432/C5W01B |
| real_kc1_software | OpenML | OpenML data id 1067 | not established (see provider) | PROMISE repository (Sayyad Shirabad & Menzies, 2005); NASA MDP |
| real_kc2_software | OpenML | OpenML data id 1063 | not established (see provider) | PROMISE repository (Sayyad Shirabad & Menzies, 2005); NASA MDP |
| real_madelon | OpenML (UCI origin) | OpenML data id 1485 | CC BY 4.0 (UCI Machine Learning Repository) | I. Guyon. *Madelon*. UCI Machine Learning Repository. doi:10.24432/C5602H |
| real_ozone_level_8hr | OpenML (UCI origin) | OpenML data id 1487 | CC BY 4.0 (UCI Machine Learning Repository) | K. Zhang, W. Fan. *Ozone Level Detection*. UCI Machine Learning Repository. doi:10.24432/C5NG6W |
| real_pc1_software | OpenML | OpenML data id 1068 | not established (see provider) | PROMISE repository (Sayyad Shirabad & Menzies, 2005); NASA MDP |
| real_pc3_software | OpenML | OpenML data id 1050 | not established (see provider) | PROMISE repository (Sayyad Shirabad & Menzies, 2005); NASA MDP |
| real_pc4_software | OpenML | OpenML data id 1049 | not established (see provider) | PROMISE repository (Sayyad Shirabad & Menzies, 2005); NASA MDP |
| real_pollen | OpenML | OpenML data id 871 | not established (see provider) | OpenML data id 871 |
| real_qsar_biodeg | OpenML (UCI origin) | OpenML data id 1494 | CC BY 4.0 (UCI Machine Learning Repository) | K. Mansouri, T. Ringsted, D. Ballabio, R. Todeschini, V. Consonni. *QSAR biodegradation*. UCI Machine Learning Repository. doi:10.24432/C5H60M |
| real_seeds | OpenML (UCI origin) | OpenML data id 1499 | CC BY 4.0 (UCI Machine Learning Repository) | M. Charytanowicz, J. Niewczas, P. Kulczycki, P. Kowalski, S. Lukasik. *Seeds*. UCI Machine Learning Repository. doi:10.24432/C5H30K |
| real_sonar | OpenML (UCI origin) | OpenML data id 40 | CC BY 4.0 (UCI Machine Learning Repository) | T. Sejnowski, R. Gorman. *Connectionist Bench (Sonar, Mines vs. Rocks)*. UCI Machine Learning Repository. doi:10.24432/C5T01Q |
| sk_aniso_blobs_easy | scikit-learn generator | aniso_blobs / easy / seed 1234 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_aniso_blobs_hard | scikit-learn generator | aniso_blobs / hard / seed 1236 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_aniso_blobs_medium | scikit-learn generator | aniso_blobs / medium / seed 1235 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_blobs_easy | scikit-learn generator | blobs / easy / seed 1334 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_blobs_hard | scikit-learn generator | blobs / hard / seed 1336 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_blobs_medium | scikit-learn generator | blobs / medium / seed 1335 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_circles_easy | scikit-learn generator | circles / easy / seed 1434 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_circles_hard | scikit-learn generator | circles / hard / seed 1436 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_circles_medium | scikit-learn generator | circles / medium / seed 1435 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_moons_easy | scikit-learn generator | moons / easy / seed 1534 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_moons_hard | scikit-learn generator | moons / hard / seed 1536 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_moons_medium | scikit-learn generator | moons / medium / seed 1535 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_varied_blobs_easy | scikit-learn generator | varied_blobs / easy / seed 1634 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_varied_blobs_hard | scikit-learn generator | varied_blobs / hard / seed 1636 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |
| sk_varied_blobs_medium | scikit-learn generator | varied_blobs / medium / seed 1635 | synthetic, generated by this code | scikit-learn (Pedregosa et al., JMLR 2011) |

OpenML: J. Vanschoren, J. N. van Rijn, B. Bischl, L. Torgo, *OpenML: networked science in machine learning*, SIGKDD Explorations 15(2), 2013.
