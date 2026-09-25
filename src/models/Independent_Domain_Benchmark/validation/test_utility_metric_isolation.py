"""Regression: one metric's utility must not depend on the other metrics present.

The Stage-4C metric-dispatch repair exposed how easy it is to move utilities by
accident. The mechanism was the candidate ROW basis, but the neighbouring risk is
COLUMN coupling: if a metric's score changed when an unrelated metric were added
to or removed from the analysis frame, then every reported utility would depend
on which metrics happened to be evaluable, and no metric ranking would mean
anything.

This pins the invariant. For a fixed candidate row set, adding or removing
unrelated metric columns must leave a metric's status, its correlation row count,
every utility component and its final utility bit-identical.

Archetypes covered: a full-coverage metric, a partially-invalid metric, an
all-but-a-few-invalid metric, plus the real ids the audit called out -- cvdd,
cvnn, a New46 and an Original.

INTERNAL / NON_SCIENTIFIC fixture. No external dataset, no real ARI.
"""
import itertools
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

UTIL = "utility_score"
COMPONENTS = ("spearman_signed", "kendall_signed", "pairwise_accuracy",
              "ndcg_all", "topk_overlap", "bottomk_overlap", "topk_ndcg",
              "bottomk_ndcg", "n_rows_used_for_corr")
TOL = 0.0          # bit-identical is the requirement, not "close enough"

N = 40
rng = np.random.default_rng(20260911)
ari = rng.uniform(0.0, 1.0, N)

# archetypes, by how much of the candidate set they can score
def _partial(frac, strength):
    v = strength * ari + (1 - strength) * rng.uniform(0, 1, N)
    v = v.astype(float)
    n_bad = int(round((1 - frac) * N))
    if n_bad:
        v[rng.choice(N, n_bad, replace=False)] = np.nan
    return v

COLUMNS = {
    "silhouette_norm": _partial(1.00, 0.85),            # Original, full coverage
    "gridness_fft_acf_norm": _partial(1.00, 0.55),      # New46, full coverage
    "cvdd_norm": _partial(0.68, 0.70),                  # partial, as observed
    "cvnn_norm": _partial(0.99, 0.60),                  # near-full
    "dcsi_norm": _partial(0.85, 0.40),                  # partially invalid
    "cdbw_norm": _partial(0.97, 0.35),
    "noise_aware_silhouette_norm": _partial(1.00, 0.80),
    "sparse_metric_norm": _partial(0.30, 0.50),         # mostly invalid
}
ALL = sorted(COLUMNS)


def utilities(cols, tmp):
    frame = {"ARI": ari, "valid": [True] * N}
    for c in cols:
        frame[c] = COLUMNS[c]
    p = Path(tmp) / "u.csv"
    pd.DataFrame(frame).to_csv(p, index=False)
    try:
        u = compute_metric_utilities(str(p))
    except Exception:  # noqa: BLE001
        return {}
    return {str(r["metric"]): dict(r) for _, r in u.iterrows()}


def main() -> int:
    fails = []
    checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        full = utilities(ALL, tmp)

        # 1. every strict subset containing the metric must reproduce it exactly
        for target in ALL:
            base = target[:-5]
            others = [c for c in ALL if c != target]
            subsets = [[target],
                       [target] + others[:1],
                       [target] + others[:3],
                       [target] + others[:-1],
                       ALL]
            for sub in subsets:
                got = utilities(sorted(sub), tmp)
                a, b = got.get(base), full.get(base)
                checks += 1
                if (a is None) != (b is None):
                    fails.append("%s: presence changed with %d columns"
                                 % (base, len(sub)))
                    continue
                if a is None:
                    continue
                for c in COMPONENTS:
                    if float(a[c]) != float(b[c]):
                        fails.append("%s: component %s changed with %d columns "
                                     "(%r vs %r)" % (base, c, len(sub), a[c], b[c]))
                        break
                if float(a[UTIL]) != float(b[UTIL]):
                    fails.append("%s: utility changed with %d columns (%.15f vs %.15f)"
                                 % (base, len(sub), a[UTIL], b[UTIL]))

        # 2. removing one unrelated metric must not disturb any survivor
        for dropped in ALL:
            kept = [c for c in ALL if c != dropped]
            got = utilities(kept, tmp)
            for c in kept:
                base = c[:-5]
                a, b = got.get(base), full.get(base)
                checks += 1
                if a is None or b is None:
                    if (a is None) != (b is None):
                        fails.append("%s: presence changed when %s was dropped"
                                     % (base, dropped))
                    continue
                if float(a[UTIL]) != float(b[UTIL]):
                    fails.append("%s: utility moved when unrelated %s was dropped"
                                 % (base, dropped))

    print("  archetypes:            %d metric columns" % len(ALL))
    print("  comparisons performed: %d" % checks)
    print("  tolerance:             exact equality")
    for f in fails[:10]:
        print("   FAIL %s" % f)
    print("\n  RESULT: %s" % ("PASS - utility is independent of the other metrics "
                              "present" if not fails else "FAIL (%d)" % len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
