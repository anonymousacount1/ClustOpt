"""Self-contained pymfe meta-feature worker (runs in the isolated `.mfenv`).

This script is executed by a *separate* Python interpreter that has ``pymfe``
installed (see ``setup/create_mf_env``), so it must not import anything from the
``unified_mkr`` package or the ClustOpt project — only the stdlib + numpy +
pandas + pymfe. Communication is via JSON files (no pyarrow needed here).

Usage::

    python pymfe_worker.py <jobs.json> <out.json>

``jobs.json`` is a list of ``{"record_id", "data_path", "view_type"}``. For each
job it loads the dataset CSV, rebuilds the view's decision-space X exactly as
the ClustOpt view builder does, and runs the same pymfe call ML2DAC uses
(``MFE(groups=["statistical","info-theory","general"]).fit(X).extract()`` with
NaN -> 0). ``out.json`` maps record_id -> {status, error, runtime_sec, names,
values}.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

ML2DAC_MF_GROUPS = ["statistical", "info-theory", "general"]
_LABEL_NAMES = ("cluster_id", "label", "labels", "y_true")


def _build_decision_X(data_path: str, view_type: str) -> np.ndarray:
    df = pd.read_csv(data_path)
    cols = list(df.columns)
    feature_cols = [c for c in cols if c not in _LABEL_NAMES]
    lower = {c.lower(): c for c in feature_cols}
    x_col = lower.get("x", feature_cols[0])
    y_col = lower.get("y", feature_cols[1] if len(feature_cols) > 1 else feature_cols[0])
    if view_type == "x_only":
        sel = [x_col]
    elif view_type == "y_only":
        sel = [y_col]
    else:  # xy_2d
        sel = [x_col, y_col]
    return df[sel].to_numpy(dtype=float)


def _extract(X: np.ndarray):
    from pymfe.mfe import MFE

    mfe = MFE(groups=ML2DAC_MF_GROUPS)
    mfe.fit(X)
    names, values = mfe.extract()
    values = np.nan_to_num(np.asarray(values, dtype=float))
    return list(names), [float(v) for v in values]


def main(jobs_path: str, out_path: str) -> int:
    with open(jobs_path, "r", encoding="utf-8") as fh:
        jobs = json.load(fh)

    results = {}
    for job in jobs:
        rid = job["record_id"]
        t0 = time.perf_counter()
        try:
            X = _build_decision_X(job["data_path"], job["view_type"])
            names, values = _extract(X)
            results[rid] = {
                "status": "success",
                "error": "",
                "runtime_sec": time.perf_counter() - t0,
                "names": names,
                "values": values,
            }
        except Exception as exc:  # noqa: BLE001
            results[rid] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "runtime_sec": time.perf_counter() - t0,
                "names": [],
                "values": [],
            }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python pymfe_worker.py <jobs.json> <out.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
