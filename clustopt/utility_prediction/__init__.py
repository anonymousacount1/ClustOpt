"""The released utility-profile predictors.

    models/utility_predictor/knn             k-NN predictor, 60 indices: the profile of CLUSTOPT as
                                             reported (arm C4, see clustopt.pipeline) and of arm C1
    models/utility_predictor/neural_full60   neural predictor, 60 indices: the neural-profile arms
                                             C0/C3; also half of the reranker's input context
    models/utility_predictor/neural_head14   neural predictor, 14 established indices (inventory ablation)
"""
from __future__ import annotations

from ..common import MODELS, ensure_src_on_path

ensure_src_on_path()


def load_neural(name="neural_full60", checkpoint_policy="best_single_fold"):
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.regressor_predictor import RegressorPredictor
    return RegressorPredictor(MODELS / "utility_predictor" / name, checkpoint_policy, 0)


def predict_neural(feature_vector, name="neural_full60"):
    """feature_vector: mapping feature-name -> value (see ``features_records.csv``)."""
    p = load_neural(name)
    return p.predict(feature_vector)


def knn_model_dir():
    return MODELS / "utility_predictor" / "knn"


def load_knn():
    """The released k-NN predictor (``predict(feature_vector, view_id) -> (utilities, debug)``)."""
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.knn_predictor_provider import (
        KNNUtilityProvider)
    return KNNUtilityProvider(knn_model_dir())
