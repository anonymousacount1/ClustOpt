"""Deterministic search-seed derivation for matched factorial experiments.

Historically the ClustOpt Optuna study was created without a sampler seed, so
the search trajectory was not reproducible (Stage-0 BLOCKER-1). The Stage-1 2x3
ablation needs the six arms to start from *identical* sampler randomness for a
given (dataset, view) so that any difference between arms is attributable to the
objective rather than to the draw.

Design constraints (Stage-1A):

* stable across Python processes -- ``hash()`` is salted per process (PYTHONHASHSEED)
  and must never be used;
* stable across Windows and Linux;
* the **arm / method id is deliberately not an input**, so all six arms share the
  same derived seed for the same dataset/view;
* different (dataset, view) pairs normally produce different seeds;
* the result must fit the sampler's accepted integer range.

The derivation is SHA-256 over a delimited UTF-8 payload, truncated to 32 bits
and masked to a non-negative int32. Optuna's ``TPESampler`` seeds a NumPy
generator, so the value must lie in ``[0, 2**32)``; masking to 31 bits keeps it
inside the stricter ``[0, 2**31)`` range accepted everywhere.

This is the *search* seed and is intentionally distinct from the MLP *training*
seed (``TrainingConfig.seed``), which governs model initialisation and the inner
train/validation split. The two must not be conflated.
"""
from __future__ import annotations

import hashlib
from typing import Optional

# Upper bound (exclusive) for derived seeds: non-negative signed 32-bit range.
SEED_MODULUS: int = 2 ** 31

# Sentinel used in configs to request per-record seed derivation.
AUTO_SEED: str = "auto"

# Stage-1 default base seed. Configs should state this explicitly
# (``"base_seed": 42``) rather than relying on the default, so a run's seed
# policy is readable from its own config snapshot.
DEFAULT_BASE_SEED: int = 42


def derive_seed(base_seed: int, dataset_id: str, view_id: str) -> int:
    """Derive a deterministic search seed for one (dataset, view).

    The arm/method id is *not* part of the payload: every arm evaluating the
    same (dataset, view) must receive the same seed.

    >>> derive_seed(42, "c2e_000808_1feb31de", "xy_2d") == derive_seed(
    ...     42, "c2e_000808_1feb31de", "xy_2d")
    True
    """
    payload = f"{int(base_seed)}|{dataset_id}|{view_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % SEED_MODULUS


def resolve_search_seed(
    seed_param: object,
    *,
    base_seed: object = None,
    dataset_id: Optional[str] = None,
    view_id: Optional[str] = None,
) -> Optional[int]:
    """Resolve a config ``seed`` value into a concrete seed (or ``None``).

    Accepted forms:

    ``None`` / absent
        No seeding -- the historical unseeded Optuna path.
    ``"auto"``
        Derive per record via :func:`derive_seed`; requires ``dataset_id`` and
        ``view_id``. ``base_seed`` defaults to :data:`DEFAULT_BASE_SEED`.
    ``int``
        Used verbatim (useful for tests and single-record reproductions).
    """
    if seed_param is None:
        return None

    if isinstance(seed_param, str):
        if seed_param.lower() != AUTO_SEED:
            raise ValueError(
                f"Unsupported search seed '{seed_param}'. Use an integer, "
                f"'{AUTO_SEED}', or omit the key for the unseeded path."
            )
        if not dataset_id or not view_id:
            raise ValueError(
                "seed='auto' requires both dataset_id and view_id to derive a "
                "deterministic per-record seed."
            )
        resolved_base = DEFAULT_BASE_SEED if base_seed is None else int(base_seed)
        return derive_seed(resolved_base, dataset_id, view_id)

    if isinstance(seed_param, bool):          # guard: bool is an int subclass
        raise ValueError("Search seed must be an int or 'auto', not a bool.")

    return int(seed_param) % SEED_MODULUS
