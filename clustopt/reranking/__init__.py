"""The released candidate rerankers (one per selection policy).

``models/candidate_reranker/<POLICY>/model.joblib`` with the registry in
``models/candidate_reranker/reranker_registry.json`` and the policy mapping in
``models/candidate_reranker/policy_to_reranker_mapping.json``.
"""
from __future__ import annotations

import json

from ..common import MODELS, ensure_src_on_path

ensure_src_on_path()
BASE = MODELS / "candidate_reranker"


def registry():
    return json.loads((BASE / "reranker_registry.json").read_text(encoding="utf-8"))


def available():
    return sorted(p.name for p in BASE.iterdir() if (p / "model.joblib").is_file())


def load(policy="KNN_TOP10"):
    import joblib
    return joblib.load(BASE / policy / "model.joblib")
