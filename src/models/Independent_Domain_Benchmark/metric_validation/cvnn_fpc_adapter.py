"""CVNN through the external R package ``fpc`` (optional).

CVNN (Liu et al., 2013) is NOT part of CLUSTOPT's 60-index inventory. It is one
of four recent comparator indices scored only in the auxiliary external
index analysis (index-utility tables). This release contains no CVNN
implementation and no fpc code: this adapter writes the data and the candidate
partitions to a temporary folder and calls ``fpc::cvnn`` in a separate R
process. fpc (GPL) must be installed by the user.

Semantics reproduced from the frozen analysis (validated on all 150 external
units; see ``results/external/index_analysis/cvnn_frozen_values.csv``):
* partitions are passed to fpc as positive cluster codes 1..K (noise is its own
  code);
* CVNN = sep / max(sep) + comp / max(comp) over the candidate set, where a
  partition in which every point is noise does not enter the maxima;
* partitions with fewer than two clusters (or all noise) are reported invalid.

R command: ``CLUSTOPT_RSCRIPT`` (default ``Rscript``). For a container use a
template containing ``{work}``, e.g.
``docker run --rm -v {work}:/w r-base-with-fpc Rscript``; the work folder is
then mounted at ``/w``.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_R = r"""
args <- commandArgs(trailingOnly = TRUE)
w <- args[1]; k <- as.integer(args[2])
X <- as.matrix(read.csv(file.path(w, "X.csv"), header = FALSE))
L <- as.matrix(read.csv(file.path(w, "L.csv"), header = FALSE))
cls <- lapply(seq_len(ncol(L)), function(j) as.integer(factor(as.integer(L[, j]))))
res <- fpc::cvnn(d = dist(X), clusterings = cls, k = k)
write.csv(data.frame(sep = res$sep, comp = res$comp), file.path(w, "out.csv"), row.names = FALSE)
"""


def _run_fpc(X: np.ndarray, clusterings: Sequence[np.ndarray], k: int):
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        w = Path(td)
        np.savetxt(w / "X.csv", np.asarray(X, float).reshape(len(X), -1), delimiter=",", fmt="%.17g")
        np.savetxt(w / "L.csv", np.column_stack([np.asarray(c).ravel() for c in clusterings]),
                   delimiter=",", fmt="%d")
        (w / "run.R").write_text(_R, encoding="utf-8")
        cmd = os.environ.get("CLUSTOPT_RSCRIPT", "Rscript")
        if "{work}" in cmd:
            argv = shlex.split(cmd.format(work=str(w).replace("\\", "/"))) + ["/w/run.R", "/w", str(k)]
        else:
            argv = shlex.split(cmd) + [str(w / "run.R"), str(w), str(k)]
        p = subprocess.run(argv, capture_output=True, text=True,
                           env=dict(os.environ, MSYS_NO_PATHCONV="1"))
        if p.returncode != 0:
            raise RuntimeError("fpc::cvnn failed: %s" % p.stderr[-500:])
        out = pd.read_csv(w / "out.csv")
    return out["sep"].to_numpy(float), out["comp"].to_numpy(float)


def cvnn_index(X, clusterings: Sequence[np.ndarray], k: int = 5):
    """CVNN for a set of candidate partitions of one dataset (list of MetricResult)."""
    from .external_cvis import DIRECTION, MetricResult, _basic_invalid, _clean
    X = np.asarray(X, dtype=float)
    ys = [_clean(X, lab)[1] for lab in clusterings]
    bad = [_basic_invalid(X, y) for y in ys]
    structural = [_basic_invalid(X, y, need_k2=False) for y in ys]
    sep, comp = _run_fpc(X, ys, k)
    sep = np.where([s is None for s in structural], sep, np.nan)
    comp = np.where([s is None for s in structural], comp, np.nan)
    msep = np.nanmax(sep) if np.isfinite(sep).any() else np.nan
    mcomp = np.nanmax(comp) if np.isfinite(comp).any() else np.nan
    prov = {"reference": "Liu et al. 2013; computed by external R fpc::cvnn", "params": {"k": k},
            "normalisation": "sep/max(sep) + comp/max(comp) across the compared set"}
    out: List = []
    for i, b in enumerate(bad):
        if b:
            out.append(MetricResult("cvnn", None, DIRECTION["cvnn"], False, b, provenance=prov))
            continue
        v = float(sep[i] / msep + comp[i] / mcomp)
        out.append(MetricResult("cvnn", v, DIRECTION["cvnn"], bool(np.isfinite(v)),
                                "" if np.isfinite(v) else "non_finite_value",
                                provenance=dict(prov, sep=float(sep[i]), comp=float(comp[i]))))
    return out


def cvnn(X, labels, k: int = 5, *, comparison_set: Optional[Sequence] = None):
    """CVNN is only defined relative to a comparison set; see :func:`cvnn_index`."""
    from .external_cvis import DIRECTION, MetricResult
    if comparison_set is not None:
        return cvnn_index(X, [np.asarray(labels)] + list(comparison_set), k)[0]
    return MetricResult("cvnn", None, DIRECTION["cvnn"], False,
                        "cvnn_requires_a_comparison_set_for_normalisation",
                        provenance={"note": "use cvnn_index over the candidate bank"})
