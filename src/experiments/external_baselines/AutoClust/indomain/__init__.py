"""In-domain AutoClust execution phase (py3.12).

A faithful, AutoClust-compatible application phase that runs in the main py3.12
environment against the in-domain derived AutoClust repository
(``Unified_MKR/derived/autoclust_split1/{original_cvis,extended_cvis}``). It is
NOT the original SMAC-based AutoClust adapter; it reproduces AutoClust's two
application steps with our rebuilt models:

1. **Algorithm selection** — the in-domain k=1 KNeighbors classifier picks one
   clustering algorithm from the test record's MeanShift-landmarking
   meta-features (the ``autoclust_*`` features).
2. **HPO with the ARI-predictor MLP as the objective** — a lightweight, seeded
   random search over the selected algorithm's hyperparameters (K restricted to
   2..5 for K-based algorithms) where each config is scored by the in-domain MLP
   predicting ARI from the config's CVI vector. The incumbent is the config with
   the highest predicted ARI.

All configurations are clustered live; ground truth is used only for external
evaluation, never selection (the MLP objective never sees labels).
"""
