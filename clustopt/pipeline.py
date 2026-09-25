"""CLUSTOPT as reported in the paper (arm C4), end to end, on one dataset.

C4 is the deployed CLUSTOPT method reported in the paper:

1. label-free meta-features of the dataset view (250 descriptors);
2. a utility profile from the released k-NN predictor (``models/utility_predictor/knn``),
   keeping the 10 indices with the highest predicted utility, with raw normalised weights;
3. an Optuna TPE search over the clustering search space, 50 evaluations;
4. the ``KNN_TOP10`` candidate reranker (``models/candidate_reranker/KNN_TOP10``), which
   returns the final partition from among the candidates the search visited. Its input
   includes both released utility profiles (k-NN and neural), so both predictors are loaded.

This module adds no algorithm. It calls the frozen implementation that produced the
external-benchmark results (``src/models/Independent_Domain_Benchmark/methods/clustopt_adapter.py``),
with the frozen external protocol as defaults: search seed 1234, 50 evaluations, 2 to 5
clusters (``src/models/Independent_Domain_Benchmark/search/protocol.py``). With these defaults
the result is deterministic for a given input. On the external benchmark datasets it
reproduces the published partitions (``results/external/per_dataset/final_partitions_by_unit.jsonl.gz``);
``scripts/smoke_test.py`` step J checks this.

``reranked=False`` gives arm C1: the same search, returning the maximiser of the search
objective instead of the reranker's choice.

Input
    ``X``: an ``(n, 2)`` array of the two coordinates (the released models were trained on
    two-dimensional data). ``view``: ``"xy_2d"`` (both coordinates, the default),
    ``"x_only"`` or ``"y_only"`` (the partition is fitted on one coordinate; the validity
    indices are still computed on both).
Output
    a :class:`Result` with ``labels`` (length ``n``, noise ``-1``), the selected algorithm and
    configuration, the selected validity indices and their weights, and the index of the
    chosen candidate. Ground truth is never used.

Example::

    from sklearn.datasets import make_moons
    from clustopt import pipeline
    X, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
    res = pipeline.run(X)                  # C4 on the xy_2d view
    print(res.algorithm, res.selected_indices, res.labels[:10])

Command line (a CSV file with columns ``x`` and ``y``, or a ``.npy`` array)::

    python -m clustopt.pipeline --input data.csv [--view xy_2d] [--output labels.csv]

The policy-selector arms C2 and C5 are not available here: their models are not
distributed with this release. C4 does not use them.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .common import ROOT, SRC, VIEWS, ensure_src_on_path

ensure_src_on_path()

ARMS = {True: "IDB_ClustOpt_C4_KNN_TOP10_RAW_R", False: "IDB_ClustOpt_C1_KNN_TOP10_RAW_noR"}
_WORKSPACE_LINKS = ("results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout",
                    "results_analysis/metric_utility_knn/phase_e1/models",
                    "results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models")


@dataclass
class Result:
    method: str                      # "C4" (reranked) or "C1" (objective maximiser)
    view: str
    labels: np.ndarray
    algorithm: str
    configuration: str
    selected_indices: List[str]
    index_weights: Dict[str, float]
    selected_candidate: int
    n_evaluations: int
    reranker: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)


def _ensure_workspace() -> None:
    """Link the released models to the locations the frozen code reads (idempotent)."""
    if not all((SRC / p).exists() for p in _WORKSPACE_LINKS):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_workspace.py")], check=True)


def protocol():
    """The frozen external protocol (seed, budget, cluster range)."""
    from models.Independent_Domain_Benchmark.search import protocol as P
    return {"seed": P.BASE_SEED, "budget": P.BUDGET_EVALUATIONS, "k_min": P.K_MIN, "k_max": P.K_MAX}


def run(X, view: str = "xy_2d", *, reranked: bool = True, dataset_id: Optional[str] = None,
        seed: Optional[int] = None, budget: Optional[int] = None) -> Result:
    """Run C4 (``reranked=True``) or C1 on one dataset view; see the module docstring."""
    if view not in VIEWS:
        raise ValueError("view must be one of %s" % (VIEWS,))
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be an (n, 2) array; got shape %s" % (X.shape,))
    _ensure_workspace()
    from models.Independent_Domain_Benchmark.methods import clustopt_adapter as CA

    p = protocol()
    # the adapter caches search traces per (dataset id, view): key them by content
    dataset_id = dataset_id or "input_" + hashlib.sha256(X.tobytes()).hexdigest()[:16]
    adapter = CA.ClustOptAdapter(ARMS[bool(reranked)])
    try:
        adapter.initialise()
        r = adapter.run(X, seed=p["seed"] if seed is None else int(seed),
                        budget=p["budget"] if budget is None else int(budget),
                        k_min=p["k_min"], k_max=p["k_max"], dataset_id=dataset_id, view_id=view)
    finally:
        adapter.close()
    e = r.extra
    return Result(method="C4" if reranked else "C1", view=view, labels=np.asarray(r.labels),
                  algorithm=r.selected_algorithm, configuration=r.best_configuration.get("config_str", ""),
                  selected_indices=list(e["selected_metric_ids"]),
                  index_weights=dict(e["selected_metric_weights"]),
                  selected_candidate=int(e["selected_candidate_index"]), n_evaluations=int(r.n_evaluations),
                  reranker=(e.get("selection_detail") or {}).get("reranker_id"),
                  details={"final_labels_hash": e.get("final_labels_hash"), "search_seed": e.get("search_seed"),
                           "dataset_id": dataset_id})


def _load(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import pandas as pd
    df = pd.read_csv(path)
    cols = ["x", "y"] if {"x", "y"} <= set(df.columns) else list(df.columns[:2])
    return df[cols].to_numpy(float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run CLUSTOPT (C4) on one dataset.")
    ap.add_argument("--input", required=True, help="CSV with columns x, y (or the first two columns), or .npy")
    ap.add_argument("--view", default="xy_2d", choices=list(VIEWS))
    ap.add_argument("--no-rerank", action="store_true", help="arm C1: the search objective's maximiser")
    ap.add_argument("--output", default=None, help="write one label per row to this CSV")
    a = ap.parse_args(argv)
    res = run(_load(Path(a.input)), a.view, reranked=not a.no_rerank)
    print("method            %s (view %s)" % (res.method, res.view))
    print("selected indices  %s" % ", ".join(res.selected_indices))
    print("algorithm         %s" % res.algorithm)
    print("configuration     %s" % res.configuration)
    print("chosen candidate  %d of %d evaluations (reranker: %s)" % (res.selected_candidate, res.n_evaluations,
                                                                      res.reranker))
    labs, counts = np.unique(res.labels, return_counts=True)
    print("clusters          %s" % dict(zip(labs.tolist(), counts.tolist())))
    if a.output:
        import pandas as pd
        pd.DataFrame({"label": res.labels}).to_csv(a.output, index=False)
        print("labels written to %s" % a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
