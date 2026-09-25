"""Guard: the matched AutoClust path must never route through the native sampler.

ML2DAC ships two config samplers. ``sample_optimizer_configs`` is the native one
and assigns ``random_state = rng.integers(0, 10000)`` -- a fresh random seed per
candidate. ``sample_optimizer_configs_constrained`` draws every parameter,
``random_state`` included, from the frozen ClustOpt grid.

The matched ``*_same_search_space`` protocol -- and therefore the Stage-4
external AutoClust path -- must use the CONSTRAINED sampler. Routing through the
native one would silently randomise the seeds the whole trace-sharing proof rests
on, and would do so without any visible error.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))

from experiments.external_baselines.ML2DAC.indomain import search as S  # noqa: E402


def test_native_sampler_randomises_random_state():
    """The hazard is real: the native path randomises random_state per candidate.

    The randomisation lives in the helper _sample_params, which the native
    sample_optimizer_configs calls and the constrained sampler does not.
    """
    helper = inspect.getsource(S._sample_params)

    assert '"random_state": int(rng.integers' in helper
    native = inspect.getsource(S.sample_optimizer_configs)
    assert "_sample_params" in native, "native sampler must reach the helper"


def test_constrained_sampler_avoids_the_randomising_helper():
    """The constrained sampler must not touch _sample_params at all."""
    src = inspect.getsource(S.sample_optimizer_configs_constrained)
    assert "_sample_params" not in src


def test_constrained_sampler_does_not_randomise_random_state():
    """The constrained sampler must take random_state from the grid only."""
    src = inspect.getsource(S.sample_optimizer_configs_constrained)
    assert '"random_state": int(rng.integers' not in src
    assert "sample_from_space" in src


def test_constrained_sampler_is_deterministic_given_seed():
    from models.ClustOpt.configs.experiments.generate_experiment_configs import (
        build_search_space)
    space = build_search_space("xy_2d")
    a = S.sample_optimizer_configs_constrained(["kmeans"], space, 20, (2, 5), 1234)
    b = S.sample_optimizer_configs_constrained(["kmeans"], space, 20, (2, 5), 1234)
    c = S.sample_optimizer_configs_constrained(["kmeans"], space, 20, (2, 5), 4321)
    assert a == b, "same seed must give the same slate"
    assert a != c, "a different seed must give a different slate"


def test_grid_random_state_is_pinned_not_sampled():
    from models.ClustOpt.configs.experiments.generate_experiment_configs import (
        build_search_space)
    space = build_search_space("xy_2d")
    got = S.sample_optimizer_configs_constrained(
        ["kmeans", "gmm", "minibatch_kmeans"], space, 30, (2, 5), 99)
    seen = {c["hyperparameters"].get("random_state") for c in got}
    assert seen == {42}, "grid random_state must be pinned, saw %s" % seen


def main() -> int:
    tests = [test_native_sampler_randomises_random_state,
             test_constrained_sampler_avoids_the_randomising_helper,
             test_constrained_sampler_does_not_randomise_random_state,
             test_constrained_sampler_is_deterministic_given_seed,
             test_grid_random_state_is_pinned_not_sampled]
    rows, fail = [], 0
    for t in tests:
        try:
            t()
            rows.append({"test": t.__name__, "status": "PASS"})
        except Exception as e:
            fail += 1
            rows.append({"test": t.__name__, "status": "FAIL",
                         "error": "%s: %s" % (type(e).__name__, str(e)[:160])})
        print("  %-52s %s" % (rows[-1]["test"], rows[-1]["status"]))
    out = {"suite": "autoclust_sampler_guard", "n": len(rows), "n_fail": fail,
           "required_sampler": "sample_optimizer_configs_constrained",
           "forbidden_sampler": "sample_optimizer_configs (native; randomises "
                                "random_state per candidate)",
           "rows": rows, "status": "PASS" if fail == 0 else "FAIL"}
    (_ROOT / "validation" / "autoclust_sampler_guard.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("  %d tests | %d fail -> %s" % (len(rows), fail, out["status"]))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
