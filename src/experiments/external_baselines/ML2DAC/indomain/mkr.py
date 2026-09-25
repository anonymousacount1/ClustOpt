"""Load and serve the in-domain ML2DAC Original MKR artifacts (py3.12)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# Direction of each of the 7 CVIs (higher-is-better after orientation). Used to
# pick the incumbent by the predicted CVI.
CVI_DIRECTION: Dict[str, str] = {
    "calinski_harabasz": "maximize", "davies_bouldin": "minimize",
    "silhouette": "maximize", "dbcv": "maximize", "dunn_index": "maximize",
    "coggins_jain_index": "maximize", "cop": "minimize",
}
CLUSTOPT_CVIS = {"calinski_harabasz", "davies_bouldin", "silhouette", "dbcv"}
ML2DAC_ONLY_CVIS = {"dunn_index", "coggins_jain_index", "cop"}


def _ext(p) -> str:
    s = str(p)
    if len(s) >= 240 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


class InDomainMKR:
    """The in-domain ML2DAC Original MKR: CVI classifier, NN index, warmstarts."""

    def __init__(self, variant_dir: str | Path, master_metafeatures: str | Path):
        self.variant_dir = Path(variant_dir)
        self.classifier = joblib.load(_ext(self.variant_dir / "cvi_classifier.pkl"))
        self.imputer = joblib.load(_ext(self.variant_dir / "imputer.pkl"))
        nn = joblib.load(_ext(self.variant_dir / "nearest_neighbor_index.pkl"))
        self.kdtree = nn["kdtree"]
        self.nn_record_ids = list(nn["train_record_ids"])
        self.feature_columns = list(nn["feature_columns"])

        cand = pd.read_parquet(self.variant_dir / "cvi_candidates.parquet")
        self.candidate_cvis = list(cand[cand["included"]]["metric_id"])

        # warmstart repository indexed by train record_id -> ranked configs
        wr = pd.read_parquet(self.variant_dir / "warmstart_repository.parquet")
        wr = wr.sort_values(["record_id", "rank_by_ari"], kind="mergesort")
        self._warm = {rid: g for rid, g in wr.groupby("record_id", sort=False)}

        # test record meta-features (split 1 etc.) from the master
        mf = pd.read_parquet(master_metafeatures)
        self._mf = mf.set_index("record_id")

        # provenance
        meta = json.loads((self.variant_dir / "cvi_classifier_metadata.json").read_text(encoding="utf-8"))
        self.train_split_ids = meta.get("train_split_ids")
        self.classifier_type = meta.get("classifier_type")

    # ---- feature access -------------------------------------------------
    def has_record(self, record_id: str) -> bool:
        return record_id in self._mf.index

    def features(self, record_id: str) -> np.ndarray:
        row = self._mf.loc[record_id, self.feature_columns]
        return np.asarray(row, dtype=float).reshape(1, -1)

    # ---- ML2DAC application steps --------------------------------------
    def predict_cvi(self, X_feat: np.ndarray) -> str:
        Xi = self.imputer.transform(X_feat)
        return str(self.classifier.predict(Xi)[0])

    def nearest_neighbor(self, X_feat: np.ndarray) -> str:
        Xi = self.imputer.transform(X_feat)
        _, idx = self.kdtree.query(Xi, k=1)
        return self.nn_record_ids[int(idx[0][0])]

    def warmstart_configs(self, nn_record_id: str) -> List[dict]:
        """Ranked (best-ARI-first) configs of the nearest train record."""
        g = self._warm.get(nn_record_id)
        if g is None:
            return []
        out = []
        for r in g.itertuples(index=False):
            out.append({
                "config_id": r.config_id, "algorithm": r.algorithm,
                "hyperparameters_json": r.hyperparameters_json,
                "rank_by_ari": int(r.rank_by_ari),
                "train_ari": float(r.ari) if r.ari == r.ari else None,
            })
        return out
