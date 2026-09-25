"""Behavioural check of the Phase-C utility round trip.

Feeds the canonical utility function a frame shaped exactly the way Phase C
shapes it -- an ARI column, a valid column, and one <metric_id>_norm column per
primary metric -- and asserts that what comes back can actually be joined back
onto the registry ids and yields finite numbers.

INTERNAL / NON_SCIENTIFIC fixture. No external dataset, no real ARI.
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "models/ClustOpt/external_evaluation"))
warnings.filterwarnings("ignore")

import pandas as pd                                                   # noqa: E402
from compute_metric_utility import compute_metric_utilities           # noqa: E402

CFG = REPO / "models/Independent_Domain_Benchmark/configs"
reg = json.loads((CFG / "metric_analysis_registry.json").read_text(encoding="utf-8"))
primary = {e["canonical_metric_id"]: e for e in reg["entries"]}

UTILITY_VALUE_COLUMN = "utility_score"

rng = np.random.default_rng(20260911)
n = 32
ari = rng.uniform(0, 1, n)
cols = {"ARI": ari, "valid": [True] * n}
for i, mid in enumerate(sorted(primary)):
    # a few metrics track ARI, the rest are noise; both must round trip
    cols["%s_norm" % mid] = (ari + rng.normal(0, 0.05, n)) if i < 5 \
        else rng.uniform(0, 1, n)

fails = []
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "u.csv"
    pd.DataFrame(cols).to_csv(p, index=False)
    u = compute_metric_utilities(str(p))

    if "metric" not in u.columns:
        fails.append("no 'metric' column")
    if UTILITY_VALUE_COLUMN not in u.columns:
        fails.append("no %r column; got %s" % (UTILITY_VALUE_COLUMN,
                                               sorted(u.columns)))
    if "utility" in u.columns:
        fails.append("a 'utility' column unexpectedly exists")

    returned = {str(m) for m in u["metric"]}
    unknown = sorted(returned - set(primary))
    missing = sorted(set(primary) - returned)
    if unknown:
        fails.append("ids not in registry: %s" % unknown[:5])
    if missing:
        fails.append("registry ids absent from output: %s" % missing[:5])

    vals = u[UTILITY_VALUE_COLUMN].to_numpy(dtype=float) \
        if UTILITY_VALUE_COLUMN in u.columns else np.array([])
    if vals.size and not np.all(np.isfinite(vals)):
        fails.append("non-finite utility values")
    if vals.size and not ((vals >= 0).all() and (vals <= 1).all()):
        fails.append("utility outside [0,1]")

    # the old code path, demonstrated to be broken
    old_would_be = u.iloc[0].get("utility")
    if old_would_be is not None:
        fails.append("old path would have worked; the premise is wrong")

print("metrics returned      %d" % len(returned))
print("registry primary      %d" % len(primary))
print("utility range         [%.4f, %.4f]" % (float(vals.min()), float(vals.max())))
print("top by utility        %s" % u.nlargest(3, UTILITY_VALUE_COLUMN)["metric"].tolist())
print("old ur.get('utility') %r  <- would have become float(None) -> TypeError"
      % old_would_be)
print()
print("RESULT: %s" % ("PASS" if not fails else "FAIL " + "; ".join(fails)))
sys.exit(0 if not fails else 1)
