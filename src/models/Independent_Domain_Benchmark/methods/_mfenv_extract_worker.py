"""Authoritative ML2DAC meta-feature extraction, executed by the isolated ``.mfenv``.

This script is deliberately tiny and imports nothing from the project. It is run
by the ``.mfenv`` interpreter -- the same interpreter that produced the frozen
master meta-feature parquet -- and it calls the AUTHORITATIVE extractor
``unified_mkr/pymfe_worker.py::_extract`` by file path, so the pymfe groups, the
``MFE.fit(X)`` call and the ``nan_to_num`` step are the frozen ones rather than a
convenient re-implementation.

``_build_decision_X`` is deliberately NOT used here: it is a CSV loader that
rebuilds a view from a repository dataset file, and the Stage-4 external path
supplies the decision-space ``X`` directly from the benchmark's own view builder.
The scientific identity of the meta-features lives in ``_extract``; the
equivalence of the two entry points is proven by a regression gate against the
stored master parquet.

Usage::

    python _mfenv_extract_worker.py <pymfe_worker.py> <X.npy> <out.json>
"""
import json
import sys
import time

import numpy as np


def main(worker_path: str, x_path: str, out_path: str) -> int:
    import importlib.util

    t0 = time.perf_counter()
    spec = importlib.util.spec_from_file_location("pymfe_worker", worker_path)
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)

    import pandas as pd
    import pymfe

    env = {"python": sys.version.split()[0], "pymfe": pymfe.__version__,
           "numpy": np.__version__, "pandas": pd.__version__}
    try:
        X = np.load(x_path)
        names, values = W._extract(np.asarray(X, dtype=float))
        out = {"status": "success", "env": env, "names": list(names),
               "values": [float(v) for v in values],
               "authoritative_callable": "%s.%s" % (W.__name__, W._extract.__name__),
               "groups": list(W.ML2DAC_MF_GROUPS),
               "runtime_sec": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001
        out = {"status": "failed", "env": env,
               "error": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
               "runtime_sec": time.perf_counter() - t0}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return 0 if out["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
