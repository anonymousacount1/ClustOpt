"""Static validator for the utility-search-map configs.

Loads each ``utility_map_*_<N>.json`` next to this file and counts the
number of clustering configurations the ClustOpt ``brute_force`` strategy
would enumerate. The math mirrors ``BruteForceSearch._discretise`` + the
``ClusteringAlgorithm.get_search_space`` mapping (``low == high`` -> 1
category; otherwise full integer enumeration or ``max_samples_per_param``
for floats).

The expected count is parsed from the filename suffix:

* ``utility_map_x_only_42.json`` -> expects total = 42
* ``utility_map_xy_2d_64.json``  -> expects total = 64

Run as a script::

    python validate_counts.py

Exits with code 0 iff every map matches its filename-declared total.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


FILENAME_RE = re.compile(r"^utility_map_.+_(?P<n>\d+)\.json$")


def _count_dim(definition: dict, max_samples_per_param: int) -> int:
    """Return the cardinality of a single parameter dimension."""
    t = definition["type"]
    if t == "categorical":
        return len(definition["choices"])
    if t == "int":
        low, high = int(definition["low"]), int(definition["high"])
        return 1 if low == high else (high - low + 1)
    if t == "float":
        low, high = float(definition["low"]), float(definition["high"])
        return 1 if low == high else int(max_samples_per_param)
    raise ValueError(f"Unsupported parameter type: {t!r}")


def count_configs(config_path: Path) -> tuple[int, dict[str, int]]:
    """Count expanded configurations and return ``(total, per_algorithm)``."""
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    sa = cfg.get("search_algorithm", {})
    if isinstance(sa, dict):
        max_samples = int(sa.get("params", {}).get("max_samples_per_param", 3))
    else:
        max_samples = int(cfg.get("search_algorithm_params", {}).get("max_samples_per_param", 3))

    breakdown: dict[str, int] = {}
    for algo_name, params in cfg["search_space"].items():
        product = 1
        for p_def in params.values():
            product *= _count_dim(p_def, max_samples)
        breakdown[algo_name] = product
    return sum(breakdown.values()), breakdown


def expected_count_from_filename(path: Path) -> int | None:
    m = FILENAME_RE.match(path.name)
    return int(m.group("n")) if m else None


def main() -> int:
    here = Path(__file__).parent
    maps = sorted(here.glob("utility_map_*.json"))
    if not maps:
        print("ERROR: no utility_map_*.json files found.", file=sys.stderr)
        return 2

    all_ok = True
    for cfg_path in maps:
        expected = expected_count_from_filename(cfg_path)
        total, breakdown = count_configs(cfg_path)
        if expected is None:
            status = f"NO-SUFFIX (total={total})"
        elif total == expected:
            status = f"OK (total={total}, expected={expected})"
        else:
            status = f"MISMATCH (total={total}, expected={expected})"
            all_ok = False
        print(f"{cfg_path.name}: {status}")
        for algo, n in breakdown.items():
            print(f"  {algo:30s}  {n:>3d}")
        print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
