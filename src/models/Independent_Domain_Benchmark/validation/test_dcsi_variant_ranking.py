"""Regression test: the two DCSI separation modes are NOT ranking-equivalent.

An earlier note in this project claimed that because both
``f(q) = q/(1+q)`` and ``g(q) = 2q/(1+2q)`` are strictly increasing in ``q``,
the PAPER and AUTHOR_COMPAT variants must rank candidate partitions
identically, so no conclusion could depend on the choice.

That claim is guaranteed only at the pairwise (K=2) level. The multiclass DCSI
is the MEAN of a nonlinear transform of the per-pair ``q`` values, and
``mean(f(q))`` and ``mean(g(q))`` can order two candidates differently as soon
as K >= 3. This module proves that, analytically and on real data produced by
the shipped implementation, so the claim cannot silently reappear.

Run directly to (re)write ``dcsi_variant_ranking_evidence.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT.parent))

from Independent_Domain_Benchmark.metric_validation import external_cvis as CV  # noqa: E402

#: the predeclared q-vectors, stated in the project_lead decision and verified
#: here rather than trusted
Q_A = np.array([0.05, 0.05, 0.50])
Q_B = np.array([0.10, 0.20, 0.20])

f_paper = lambda q: q / (1.0 + q)            # noqa: E731
g_compat = lambda q: 2.0 * q / (1.0 + 2.0 * q)  # noqa: E731


def check_pairwise_monotone(n: int = 200_001) -> dict:
    """K=2 equivalence DOES hold: both maps are strictly increasing in q."""
    q = np.linspace(1e-9, 1e4, n)
    return {"f_strictly_increasing": bool(np.all(np.diff(f_paper(q)) > 0)),
            "g_strictly_increasing": bool(np.all(np.diff(g_compat(q)) > 0)),
            "conclusion": "for K=2 the two modes rank candidates identically"}


def check_analytic_reversal() -> dict:
    """K=3 equivalence does NOT hold: an explicit order reversal."""
    a_p, b_p = float(f_paper(Q_A).mean()), float(f_paper(Q_B).mean())
    a_c, b_c = float(g_compat(Q_A).mean()), float(g_compat(Q_B).mean())
    return {"q_A": Q_A.tolist(), "q_B": Q_B.tolist(),
            "paper_mean_A": a_p, "paper_mean_B": b_p,
            "compat_mean_A": a_c, "compat_mean_B": b_c,
            "paper_order": "A>B" if a_p > b_p else "A<B",
            "compat_order": "A>B" if a_c > b_c else "A<B",
            "order_reverses": bool((a_p > b_p) != (a_c > b_c))}


def _synthetic_k3(rng) -> tuple:
    """A random 3-class configuration with controllable separation/spread."""
    scales = rng.uniform(0.10, 0.60, 3)
    centres = rng.uniform(-6.0, 6.0, (3, 2))
    sizes = rng.integers(30, 60, 3)
    X = np.vstack([rng.normal(centres[i], scales[i], (int(sizes[i]), 2))
                   for i in range(3)])
    y = np.repeat([1, 2, 3], sizes)
    return X, y


def check_empirical_reversal(n_trials: int = 400, seed: int = 20260911) -> dict:
    """The same reversal, through the shipped implementation on real inputs."""
    rng = np.random.default_rng(seed)
    cands = []
    for t in range(n_trials):
        X, y = _synthetic_k3(rng)
        p = CV.dcsi(X, y, sep_mode=CV.SEP_MODE_PAPER)
        c = CV.dcsi(X, y, sep_mode=CV.SEP_MODE_AUTHOR)
        if p.valid and c.valid:
            cands.append((t, float(p.value), float(c.value)))
    for i in range(len(cands)):
        for j in range(len(cands)):
            if i == j:
                continue
            _, pi, ci = cands[i]
            _, pj, cj = cands[j]
            if pi > pj and ci < cj:
                return {"found": True, "n_candidates": len(cands),
                        "candidate_i": {"trial": cands[i][0], "paper": pi,
                                        "author_compat": ci},
                        "candidate_j": {"trial": cands[j][0], "paper": pj,
                                        "author_compat": cj},
                        "paper_order": "i>j", "compat_order": "i<j",
                        "note": "produced by external_cvis.dcsi on K=3 data, "
                                "not by hand-chosen q values"}
    return {"found": False, "n_candidates": len(cands),
            "note": "no reversal at this search size; the analytic "
                    "counterexample remains the binding proof"}


def main() -> int:
    mono = check_pairwise_monotone()
    ana = check_analytic_reversal()
    emp = check_empirical_reversal()
    ok = (mono["f_strictly_increasing"] and mono["g_strictly_increasing"]
          and ana["order_reverses"])
    ev = {"test": "dcsi_variant_ranking",
          "claim_under_test": "PAPER and AUTHOR_COMPAT DCSI are universally "
                              "ranking-equivalent",
          "verdict": "REFUTED for K>=3; holds only for K=2",
          "pairwise_k2": mono, "analytic_k3_counterexample": ana,
          "empirical_k3_counterexample": emp,
          "status": "PASS" if ok else "FAIL"}
    out = _ROOT / "validation" / "dcsi_variant_ranking_evidence.json"
    out.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print(json.dumps(ev, indent=2))
    return 0 if ok else 1


def test_pairwise_monotone():
    m = check_pairwise_monotone()
    assert m["f_strictly_increasing"] and m["g_strictly_increasing"]


def test_multiclass_order_reverses():
    assert check_analytic_reversal()["order_reverses"] is True


if __name__ == "__main__":
    raise SystemExit(main())
