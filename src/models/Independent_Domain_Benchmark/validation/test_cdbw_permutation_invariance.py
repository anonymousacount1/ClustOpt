"""Regression test: paper-faithful CDbw depends on the partition, not on names.

The reference package PyPI ``cdbw`` 0.2 fails this: renaming clusters moved its
value by up to 0.93 absolute. The corrected implementation
(``external_cvis.cdbw``) must be invariant by construction, so this suite is
deliberately harsh -- many random bijective relabellings per geometry, plus the
naming conventions a caller might realistically use.

Naming schemes exercised per fixture:

* ``contiguous``     -- 0..K-1
* ``one_based``      -- 1..K, the R convention
* ``sparse``         -- 17, 42, 103, 255, ... large and non-adjacent
* ``random_perm_NN`` -- 20 random bijections onto random large ids

Negative labels are NOT exercised as cluster ids: the frozen wrapper contract
reserves ``-1`` for noise, so a negative value is not a cluster name. The
handling of ``-1`` is checked separately -- it must be treated as noise
identically however the real clusters are named.

Run directly to (re)write ``cdbw_permutation_invariance_evidence.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT.parent))

from Independent_Domain_Benchmark.metric_validation import external_cvis as CV  # noqa: E402
from Independent_Domain_Benchmark.validation import cvi_fixtures as FX          # noqa: E402

#: geometries the suite must cover, chosen for shape diversity rather than score
GEOMETRIES = ("separated_3", "unequal_density_2", "ring_and_core",
              "separated_4", "elongated_3", "nested_rings_3",
              "mixed_density_4", "two_moons_2")
N_RANDOM_PERMUTATIONS = 20
SEED = 20260911
SPARSE_IDS = [17, 42, 103, 255, 511, 1024]
#: The claim is exact: a pure renaming cannot change the mathematical value.
#: What remains is IEEE-754 summation order inside the ONE reduction still done
#: by the reused package function (compactness sums over clusters in label
#: order). Our own aggregations reduce over sorted values, so they contribute
#: nothing. The requirement is therefore the frozen CDbw tolerance, and the
#: observed residual is reported so its size is on record rather than implied.
ATOL = FX.TOLERANCE["cdbw"]["atol"]


def _schemes(y: np.ndarray, rng: np.random.Generator):
    y = np.asarray(y).astype(int)
    real = y >= 0
    uniq, comp = np.unique(y[real], return_inverse=True)
    K = len(uniq)
    out = []

    def build(name, ids):
        z = y.copy()
        z[real] = np.asarray(ids, int)[comp]
        return (name, z)

    out.append(build("contiguous", range(K)))
    out.append(build("one_based", range(1, K + 1)))
    out.append(build("sparse", SPARSE_IDS[:K]))
    for t in range(N_RANDOM_PERMUTATIONS):
        ids = rng.choice(np.arange(0, 10_000), size=K, replace=False)
        out.append(build("random_perm_%02d" % t, ids))
    return out


def run() -> dict:
    rng = np.random.default_rng(SEED)
    F = FX.build()
    rows, worst = [], 0.0
    for fid in GEOMETRIES:
        X, y = F[fid]
        base = CV.cdbw(X, y)
        vals = []
        for name, z in _schemes(y, rng):
            r = CV.cdbw(X, z)
            if r.valid != base.valid or r.invalid_reason != base.invalid_reason:
                rows.append({"fixture_id": fid, "scheme": name,
                             "status": "FAIL",
                             "detail": "validity changed under renaming"})
                continue
            d = (0.0 if not base.valid
                 else abs(float(r.value) - float(base.value)))
            worst = max(worst, d)
            vals.append(d)
        rows.append({
            "fixture_id": fid, "n_schemes": len(vals),
            "n_random_permutations": N_RANDOM_PERMUTATIONS,
            "base_value": base.value, "base_valid": bool(base.valid),
            "max_abs_diff": float(max(vals)) if vals else None,
            "status": "PASS" if (vals and max(vals) <= ATOL) else "FAIL"})

    # noise must be handled identically however the real clusters are named
    X, y = F["separated_3"]
    yn = np.asarray(y).astype(int).copy()
    yn[:5] = -1
    noise_vals = []
    for name, z in _schemes(yn, rng):
        r = CV.cdbw(X, z)
        noise_vals.append(None if not r.valid else float(r.value))
    noise_ok = (len({v for v in noise_vals if v is not None}) <= 1
                and all(v is not None for v in noise_vals))
    rows.append({"fixture_id": "separated_3+noise", "scheme": "noise_handling",
                 "n_schemes": len(noise_vals),
                 "base_value": noise_vals[0],
                 "status": "PASS" if noise_ok else "FAIL"})

    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    return {"test": "cdbw_permutation_invariance",
            "claim_under_test": "paper-faithful CDbw depends only on (X, "
                                "partition), never on cluster names",
            "geometries": list(GEOMETRIES),
            "n_geometries": len(GEOMETRIES),
            "random_permutations_per_geometry": N_RANDOM_PERMUTATIONS,
            "naming_schemes": ["contiguous", "one_based", "sparse",
                               "random_perm x%d" % N_RANDOM_PERMUTATIONS],
            "negative_labels": "not exercised as cluster ids -- -1 is reserved "
                               "for noise by the frozen wrapper contract; noise "
                               "handling is checked separately",
            "atol": ATOL,
            "residual_source": "IEEE-754 summation order inside the reused "
                               "package compactness reduction",
            "total_evaluations": sum(r.get("n_schemes", 0) for r in rows),
            "max_abs_diff_overall": float(worst),
            "n_fail": n_fail, "rows": rows,
            "status": "PASS" if n_fail == 0 else "FAIL"}


def main() -> int:
    ev = run()
    out = _ROOT / "validation" / "cdbw_permutation_invariance_evidence.json"
    out.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    for r in ev["rows"]:
        print("  %-20s %-16s %s" % (r["fixture_id"], r.get("scheme", ""),
                                    r["status"]))
    print("\n  %d geometries x %d relabellings | evaluations %d | max abs diff %s"
          " -> %s" % (ev["n_geometries"], ev["random_permutations_per_geometry"],
                      ev["total_evaluations"], ev["max_abs_diff_overall"],
                      ev["status"]))
    return 0 if ev["status"] == "PASS" else 1


def test_cdbw_permutation_invariance():
    assert run()["status"] == "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
