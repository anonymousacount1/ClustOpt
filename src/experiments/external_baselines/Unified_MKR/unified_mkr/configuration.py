"""Resolve the 32 clustering configurations per view from the ClustOpt
utility-map JSON files, replacing every ``HDBSCAN_CONSTRAINED`` configuration.

The enumeration faithfully reuses ClustOpt's brute-force semantics
(``search_algorithm/.../Brute_Force_Search.py``): categorical choices are
enumerated exhaustively, integer ranges fully, and continuous ranges are
sampled with ``max_samples_per_param`` points. The utility maps are entirely
categorical, so the product of per-parameter choices reproduces exactly the
documented 32 configurations.

Replacement rule (see UNIFIED_MKR_DESIGN.md §7.2): the 4 constrained-HDBSCAN
slots are reassigned, one each, round-robin across the other algorithms
already present in the map (sorted by name, fairest distribution). Each new
configuration extends the *primary* (most-varied) numeric hyper-parameter of
its algorithm beyond the current maximum, using the algorithm's own step
size, so no new algorithm and no new hyper-parameter dimension is invented.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CONSTRAINED_PREFIX = "hdbscan_constrained"

_UTILITY_MAP_FILENAMES = {
    "x_only": "utility_map_x_only_32.json",
    "y_only": "utility_map_y_only_32.json",
    "xy_2d": "utility_map_xy_2d_32.json",
}


@dataclass
class ConfigSpec:
    """One concrete (algorithm, hyper-parameters) clustering configuration."""

    algorithm: str
    params: Dict[str, Any]
    original_config_id: str
    config_id: str = ""
    was_replaced: bool = False
    replacement_reason: str = ""

    def is_constrained(self) -> bool:
        return self.algorithm.startswith(CONSTRAINED_PREFIX)

    def to_search_config(self) -> Dict[str, Any]:
        """Build the ``{algorithm, algo__param: value}`` dict ClustOpt expects."""
        cfg: Dict[str, Any] = {"algorithm": self.algorithm}
        for k, v in self.params.items():
            cfg[f"{self.algorithm}__{k}"] = v
        return cfg

    def hyperparameters_json(self) -> str:
        return json.dumps(self.params, sort_keys=True, default=_json_default)


@dataclass
class ResolvedConfigurations:
    view_type: str
    source_json_path: str
    search_space_config: Dict[str, Any]
    configs: List[ConfigSpec] = field(default_factory=list)


def _json_default(v: Any) -> Any:
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return v


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #
def _discretise_param(param_def: Dict[str, Any], max_samples_per_param: int) -> List[Any]:
    ptype = param_def.get("type", "categorical")
    if ptype == "categorical":
        return list(param_def["choices"])
    if ptype in ("int", "integer"):
        return list(range(int(param_def["low"]), int(param_def["high"]) + 1))
    if ptype in ("float", "real"):
        import numpy as np

        return np.linspace(
            float(param_def["low"]), float(param_def["high"]), max_samples_per_param
        ).tolist()
    raise ValueError(f"Unsupported parameter type '{ptype}' in utility map.")


def enumerate_configs(
    search_space: Dict[str, Any],
    view_type: str,
    max_samples_per_param: int = 1,
) -> List[ConfigSpec]:
    """Enumerate the full cartesian product of the search space (the 32 configs)."""
    specs: List[ConfigSpec] = []
    idx = 0
    for algo_name, params_def in search_space.items():
        param_keys = list(params_def.keys())
        discretised = [
            _discretise_param(params_def[k], max_samples_per_param) for k in param_keys
        ]
        for combo in product(*discretised):
            params = dict(zip(param_keys, combo))
            specs.append(
                ConfigSpec(
                    algorithm=algo_name,
                    params=params,
                    original_config_id=f"{view_type}__orig_{idx:02d}",
                )
            )
            idx += 1
    return specs


# --------------------------------------------------------------------------- #
# HDBSCAN_CONSTRAINED replacement
# --------------------------------------------------------------------------- #
def _primary_numeric_param(algo: str, search_space: Dict[str, Any]) -> Optional[str]:
    """The numeric hyper-parameter with the most distinct choices for ``algo``."""
    params_def = search_space.get(algo, {})
    best_key, best_n = None, -1
    for key in sorted(params_def.keys()):
        pdef = params_def[key]
        if pdef.get("type", "categorical") != "categorical":
            # Treat int/float ranges as numeric too.
            best_key, best_n = key, max(best_n, 2)
            continue
        choices = pdef.get("choices", [])
        numeric = [c for c in choices if isinstance(c, (int, float)) and not isinstance(c, bool)]
        if len(numeric) > best_n and numeric:
            best_key, best_n = key, len(numeric)
    return best_key


def _base_params(algo: str, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic base hyper-parameters (first choice of each param)."""
    out: Dict[str, Any] = {}
    for key, pdef in search_space.get(algo, {}).items():
        if pdef.get("type", "categorical") == "categorical":
            out[key] = pdef["choices"][0]
        else:
            out[key] = pdef.get("low")
    return out


def _new_values(existing: List[Any], count: int) -> List[Any]:
    """``count`` in-style values extending ``existing`` past its maximum."""
    numeric = sorted(float(x) for x in existing)
    is_int = all(float(x).is_integer() for x in existing)
    if len(numeric) >= 2:
        diffs = [b - a for a, b in zip(numeric, numeric[1:]) if b > a]
        step = statistics.median(diffs) if diffs else 1.0
    else:
        step = 1.0
    if step <= 0:
        step = 1.0
    out: List[Any] = []
    cur = numeric[-1]
    for _ in range(count):
        cur += step
        out.append(int(round(cur)) if is_int else round(cur, 6))
    return out


def _candidate_pool(
    algo: str, search_space: Dict[str, Any], existing_specs: List[ConfigSpec], count: int
) -> List[Dict[str, Any]]:
    """Generate up to ``count`` new param dicts for ``algo`` (distinct from kept)."""
    primary = _primary_numeric_param(algo, search_space)
    if primary is None:
        return []
    pdef = search_space[algo][primary]
    if pdef.get("type", "categorical") == "categorical":
        existing_vals = list(pdef.get("choices", []))
    else:
        existing_vals = [pdef.get("low"), pdef.get("high")]
    base = _base_params(algo, search_space)
    seen = {
        json.dumps(s.params, sort_keys=True, default=_json_default)
        for s in existing_specs
        if s.algorithm == algo
    }
    pool: List[Dict[str, Any]] = []
    for val in _new_values(existing_vals, count):
        params = dict(base)
        params[primary] = val
        key = json.dumps(params, sort_keys=True, default=_json_default)
        if key not in seen:
            pool.append(params)
            seen.add(key)
    return pool


def replace_constrained(
    specs: List[ConfigSpec], search_space: Dict[str, Any], view_type: str
) -> List[ConfigSpec]:
    """Replace constrained-HDBSCAN configs, keeping exactly ``len(specs)`` total."""
    kept = [s for s in specs if not s.is_constrained()]
    to_replace = [s for s in specs if s.is_constrained()]
    if not to_replace:
        resolved = list(specs)
    else:
        non_constrained_algos = sorted({s.algorithm for s in kept})
        if not non_constrained_algos:
            raise ValueError(
                f"No non-constrained algorithm available to absorb replacements "
                f"for view '{view_type}'."
            )
        # Fair round-robin: one replacement at a time, cycling algorithms.
        pools: Dict[str, List[Dict[str, Any]]] = {
            a: _candidate_pool(a, search_space, kept, len(to_replace))
            for a in non_constrained_algos
        }
        replacements: List[ConfigSpec] = []
        ai = 0
        for orig in to_replace:
            placed = False
            for _ in range(len(non_constrained_algos)):
                algo = non_constrained_algos[ai % len(non_constrained_algos)]
                ai += 1
                if pools[algo]:
                    params = pools[algo].pop(0)
                    replacements.append(
                        ConfigSpec(
                            algorithm=algo,
                            params=params,
                            original_config_id=orig.original_config_id,
                            was_replaced=True,
                            replacement_reason=(
                                f"Replaced constrained config '{orig.algorithm}' "
                                f"with extra '{algo}' configuration (fair round-robin)."
                            ),
                        )
                    )
                    placed = True
                    break
            if not placed:
                raise ValueError(
                    f"Could not generate a replacement for '{orig.algorithm}' "
                    f"in view '{view_type}' (candidate pools exhausted)."
                )
        resolved = kept + replacements

    # Assign deterministic resolved config ids.
    for i, spec in enumerate(resolved):
        spec.config_id = f"{view_type}__cfg_{i:02d}"
    return resolved


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_utility_map(config_root: str | Path, view_type: str) -> Tuple[Dict[str, Any], Path]:
    fname = _UTILITY_MAP_FILENAMES.get(view_type)
    if fname is None:
        raise ValueError(f"Unknown view_type '{view_type}'.")
    path = Path(config_root) / fname
    if not path.is_file():
        raise FileNotFoundError(f"Utility map not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh), path


def resolve_view_configurations(
    config_root: str | Path, view_type: str
) -> ResolvedConfigurations:
    """Load, enumerate, and constrained-replace the 32 configs for one view."""
    raw, json_path = load_utility_map(config_root, view_type)
    search_space = raw["search_space"]
    sa_params = raw.get("search_algorithm", {}).get("params", {}) if isinstance(
        raw.get("search_algorithm"), dict
    ) else {}
    max_samples = int(sa_params.get("max_samples_per_param", 1))

    enumerated = enumerate_configs(search_space, view_type, max_samples)
    resolved = replace_constrained(enumerated, search_space, view_type)
    return ResolvedConfigurations(
        view_type=view_type,
        source_json_path=str(json_path),
        search_space_config=search_space,
        configs=resolved,
    )


def get_cvi_metric_names(config_root: str | Path) -> List[str]:
    """Final ClustOpt utility-map metric set (union across the three views)."""
    names: List[str] = []
    seen = set()
    for view_type in _UTILITY_MAP_FILENAMES:
        raw, _ = load_utility_map(config_root, view_type)
        for name in raw.get("cvi", {}).get("metrics", {}):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def get_cvi_params(config_root: str | Path, view_type: str = "xy_2d") -> Dict[str, Any]:
    """Noise-handling params from the utility map ``cvi`` block."""
    raw, _ = load_utility_map(config_root, view_type)
    return dict(raw.get("cvi", {}).get("params", {}))
