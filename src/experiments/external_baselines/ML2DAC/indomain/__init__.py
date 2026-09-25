"""In-domain ML2DAC execution phase (py3.12).

A faithful, ML2DAC-*compatible* application phase that runs in the main py3.12
environment against the in-domain derived MKR
(``Unified_MKR/derived/ml2dac_split1/original_cvis``) — it is NOT the original
paper ``ApplicationPhase`` (which needs py3.9 + SMAC). It reproduces ML2DAC's
three application steps with our rebuilt artifacts:

1. **CVI selection** — the in-domain RandomForest predicts one of the 7 CVIs
   from the test record's ML2DAC meta-features.
2. **Warmstarting** — the in-domain KDTree finds the nearest *train* record and
   seeds the search with that record's best-ARI configurations (K-filtered).
3. **Search** — a lightweight, seeded random search (documented, not SMAC) over
   the warmstart algorithms with K restricted to 2..5, optimising the predicted
   CVI.

All 50 configurations (25 warmstart + 25 search) are clustered **live** on the
test view; ground truth is used only for external evaluation, never selection.
"""
