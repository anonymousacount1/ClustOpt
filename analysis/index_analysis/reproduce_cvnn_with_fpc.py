"""OPTIONAL: reproduce the CVNN column of the external index analysis with R fpc.

CVNN is not part of CLUSTOPT. The frozen values are in
``results/external/index_analysis/authoritative/cvnn_frozen_values.csv``. This script
recomputes them with the external R package fpc (GPL; install it yourself) on the
released 32-candidate banks and compares.

Prerequisites
  1. ``python scripts/reconstruct_external_datasets.py`` (rebuilds the inputs locally)
  2. R with fpc, reachable as ``Rscript`` or via ``CLUSTOPT_RSCRIPT`` (see
     ``src/models/Independent_Domain_Benchmark/metric_validation/cvnn_fpc_adapter.py``)

    python analysis/index_analysis/reproduce_cvnn_with_fpc.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import clustopt  # noqa: E402,F401


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    from Independent_Domain_Benchmark.analysis import fixed32_track as F32
    from Independent_Domain_Benchmark.datasets import corpus_archive as CA
    from Independent_Domain_Benchmark.execution.dataset_view import build_dataset_view
    from Independent_Domain_Benchmark.metric_validation import cvnn_fpc_adapter as AD
    man = json.loads((ROOT / "data/external_benchmark/dataset_manifest.json").read_text(encoding="utf-8"))
    src = {e["dataset_id"]: e["source"] for e in man["datasets"]}
    frozen = pd.read_csv(ROOT / "results/external/index_analysis/authoritative/cvnn_frozen_values.csv")
    banks = sorted((ROOT / "results/external/candidate_banks").glob("*.npz"))
    if a.limit:
        banks = banks[: a.limit]
    rows = []
    for b in banks:
        ds, view = b.stem.rsplit("__", 1)
        z = np.load(b)
        X2, _rec = CA.load_scientific_input(ds)
        v = build_dataset_view(dataset_id=ds, source=src[ds], view_id=view, X_full=X2)
        X = np.asarray(F32._space_array(v, "full"), dtype=float)
        res = AD.cvnn_index(X, [z["labels"][:, j] for j in range(z["labels"].shape[1])], k=5)
        for ci, r in zip(z["candidate_index"], res):
            rows.append({"dataset_id": ds, "view_id": view, "candidate_index": int(ci),
                         "valid": bool(r.valid), "cvnn_reproduced": r.value})
    out = pd.DataFrame(rows).merge(frozen, on=["dataset_id", "view_id", "candidate_index"], how="left")
    ok = out[out.status == "valid"]
    d = (ok.cvnn_reproduced - ok.cvnn_frozen).abs()
    print("units %d, valid values %d, max |diff| %.3g, n > 1e-9: %d"
          % (len(banks), len(ok), d.max(), int((d > 1e-9).sum())))
    (ROOT / "analysis/output").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "analysis/output/cvnn_fpc_reproduction.csv", index=False)
    return 0 if (d <= 1e-9).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
