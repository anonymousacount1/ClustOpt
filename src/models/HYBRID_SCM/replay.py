from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def make_replay_config(cfg, meta: Dict[str, Any]) -> Dict[str, Any]:

    if not is_dataclass(cfg):
        raise TypeError("make_replay_config expects DatasetConfig dataclass.")

    cfg_dict = deepcopy(asdict(cfg))

    cfg_dict["replay_mode"] = "frozen"

    used = meta.get("params_used", {}).get("components", [])
    used_map = {c["name"]: c for c in used}

    for c in cfg_dict["components"]:

        u = used_map.get(c["name"])
        if not u:
            continue

        # freeze parameters
        c["params"] = deepcopy(u.get("params_used", {}))

        # freeze seeds
        c["params"]["_internal_seeds"] = deepcopy(
            u.get("internal_seeds", {})
        )

        # freeze operators
        c["operators"] = deepcopy(
            u.get("operators_used", c.get("operators", {}))
        )

        # freeze noise
        nu = u.get("noise_used", {})

        if nu.get("type") == "gaussian":

            c["noise"] = "gaussian"

            c["noise_params"] = {
                "sigma": nu.get("sigma", 0.0)
            }

        else:

            c["noise"] = "none"
            c["noise_params"] = {}

    # freeze global seed for postprocess
    cfg_dict.setdefault("postprocess", {})

    cfg_dict["postprocess"]["_global_seed"] = meta.get(
        "internal_global_seed"
    )

    # freeze dataset perturbations
    if "effective_dataset_perturbations_spec" in meta:

        cfg_dict["dataset_perturbations"] = deepcopy(
            meta["effective_dataset_perturbations_spec"]
        )

    # freeze global difficulty
    if "effective_global_difficulty_spec" in meta:

        cfg_dict["global_difficulty"] = deepcopy(
            meta["effective_global_difficulty_spec"]
        )

    # freeze rendering
    if "effective_rendering_spec" in meta:

        cfg_dict["rendering"] = deepcopy(
            meta["effective_rendering_spec"]
        )

    return cfg_dict