"""Load and serve the in-domain AutoClust models (py3.12)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

# The 4 ClustOpt-implemented CVIs vs the 3 ML2DAC-only CVIs (for live computation).
CLUSTOPT_CVIS = {"calinski_harabasz", "davies_bouldin", "silhouette", "dbcv"}
ML2DAC_ONLY_CVIS = {"dunn_index", "coggins_jain_index", "cop"}


def _ext(p) -> str:
    s = str(p)
    if len(s) >= 240 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


class InDomainAutoClust:
    """In-domain AutoClust: k=1 algorithm selector + MLP ARI predictor."""

    def __init__(self, variant_dir: str | Path, master_metafeatures: str | Path):
        self.variant_dir = Path(variant_dir)
        self.selector = joblib.load(_ext(self.variant_dir / "algorithm_selector.pkl"))
        self.predictor = joblib.load(_ext(self.variant_dir / "ari_predictor.pkl"))

        fc = json.loads((self.variant_dir / "feature_columns.json").read_text(encoding="utf-8"))
        self.autoclust_features: List[str] = list(fc["autoclust_meta_feature_columns"])
        self.cvi_columns: List[str] = list(fc["ari_predictor_cvi_columns"])

        sm = json.loads((self.variant_dir / "algorithm_selector_metadata.json").read_text(encoding="utf-8"))
        self.selector_model_type = sm.get("model_type")
        self.train_split_ids = sm.get("train_split_ids")
        pm = json.loads((self.variant_dir / "ari_predictor_metadata.json").read_text(encoding="utf-8"))
        self.predictor_arch = pm.get("architecture", {}).get("hidden_layer_sizes")

        cand = pd.read_parquet(self.variant_dir / "cvi_candidates.parquet")
        self.candidate_cvis = list(cand[cand["included"]]["metric_id"])

        mf = pd.read_parquet(master_metafeatures)
        self._mf = mf.set_index("record_id")

    # ---- feature access -------------------------------------------------
    def has_record(self, record_id: str) -> bool:
        return record_id in self._mf.index

    def landmarking_features(self, record_id: str) -> np.ndarray:
        row = self._mf.loc[record_id, self.autoclust_features]
        return np.asarray(row, dtype=float).reshape(1, -1)

    # ---- AutoClust application steps -----------------------------------
    def predict_algorithm(self, landmarking_feat: np.ndarray) -> str:
        return str(self.selector.predict(landmarking_feat)[0])

    def predict_ari(self, cvi_vector: np.ndarray) -> float:
        """Predict ARI from a CVI vector (in self.cvi_columns order). NaNs are
        imputed by the predictor pipeline (median, train-fit)."""
        x = np.asarray(cvi_vector, dtype=float).reshape(1, -1)
        return float(self.predictor.predict(x)[0])
