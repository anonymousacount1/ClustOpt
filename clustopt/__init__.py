"""CLUSTOPT: utility-guided validity-index selection and candidate reranking.

Public entry points (thin wrappers over the implementation in ``src/``):

    clustopt.pipeline           CLUSTOPT as reported (arm C4) on one dataset: k-NN top-10 profile,
                                search, KNN_TOP10 reranking
    clustopt.validity_indices   the 60 validity indices used by CLUSTOPT
    clustopt.clustering         the clustering search space
    clustopt.search             run one search configuration (no reranking) on a generated dataset view
    clustopt.meta_features      label-free meta-features of a dataset
    clustopt.utility_prediction the released utility-profile predictors
    clustopt.reranking          the released candidate rerankers
"""
from .common import ROOT, SRC, VIEWS, ensure_src_on_path  # noqa: F401

__version__ = "1.0.0"
