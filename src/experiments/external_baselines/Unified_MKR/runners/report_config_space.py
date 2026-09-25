r"""Emit + validate the resolved 32-configuration space for all three views.

Produces a Markdown report (full per-config table with original/replacement
provenance) and runs the configuration-space validation checks:
exactly 32 per view, no HDBSCAN_CONSTRAINED remaining, balanced replacement
distribution, no duplicate configurations, and instantiability of every config
(a proxy for "no invalid hyper-parameter values").

Usage:
    python experiments/external_baselines/Unified_MKR/runners/report_config_space.py \
      [--config-root models/ClustOpt/configs/utilities_maps] [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_THIS = Path(__file__).resolve()
for _p in [_THIS, *_THIS.parents]:
    if (_p / "models" / "ClustOpt").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from experiments.external_baselines.Unified_MKR.unified_mkr.configuration import (  # noqa: E402
    CONSTRAINED_PREFIX,
    resolve_view_configurations,
)
from experiments.external_baselines.Unified_MKR.unified_mkr.schemas import (  # noqa: E402
    N_CONFIGS_PER_VIEW,
    VIEW_TYPES,
)


def _validate_view(rc) -> dict:
    from models.ClustOpt.search_space.Clustering_Search_Space import ClusteringSearchSpace
    import numpy as np

    ss = ClusteringSearchSpace.from_dict(rc.search_space_config)
    n = len(rc.configs)
    constrained = [c for c in rc.configs if c.algorithm.startswith(CONSTRAINED_PREFIX)]
    # duplicate detection on (algorithm, sorted params)
    seen = [(c.algorithm, c.hyperparameters_json()) for c in rc.configs]
    dups = [k for k, v in Counter(seen).items() if v > 1]
    replaced = [c for c in rc.configs if c.was_replaced]
    repl_by_algo = Counter(c.algorithm for c in replaced)
    # instantiate + run each on tiny data (validity of hyper-parameter values)
    dim = 2 if rc.view_type == "xy_2d" else 1
    X = np.random.RandomState(0).randn(120, dim)
    bad = []
    for c in rc.configs:
        try:
            ss.instantiate_from_config(c.to_search_config()).fit_predict(X)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{c.config_id} ({c.algorithm}): {type(exc).__name__}")
    balance = max(repl_by_algo.values()) - min(repl_by_algo.values()) if repl_by_algo else 0
    return {
        "view_type": rc.view_type,
        "count": n,
        "count_ok": n == N_CONFIGS_PER_VIEW,
        "constrained_remaining": len(constrained),
        "duplicates": dups,
        "n_replaced": len(replaced),
        "replacement_by_algorithm": dict(repl_by_algo),
        "replacement_max_minus_min": balance,
        "uninstantiable": bad,
    }


def _table(rc) -> str:
    rows = [
        "| config_id | algorithm | hyperparameters | origin | replacement_reason |",
        "|---|---|---|---|---|",
    ]
    for c in rc.configs:
        origin = "replacement" if c.was_replaced else "original"
        reason = c.replacement_reason if c.was_replaced else ""
        rows.append(
            f"| {c.config_id} | {c.algorithm} | "
            f"`{c.hyperparameters_json()}` | {origin} | {reason} |"
        )
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config-root", default="models/ClustOpt/configs/utilities_maps")
    p.add_argument("--out", default=str(_THIS.parent.parent / "CONFIG_SPACE_REPORT.md"))
    args = p.parse_args()

    lines = ["# Resolved 32-Configuration Space Report", ""]
    all_checks = []
    for view_type in VIEW_TYPES:
        rc = resolve_view_configurations(args.config_root, view_type)
        checks = _validate_view(rc)
        all_checks.append(checks)
        lines += [
            f"## View `{view_type}`",
            "",
            f"- total configs: **{checks['count']}** (== 32: {checks['count_ok']})",
            f"- HDBSCAN_CONSTRAINED remaining: **{checks['constrained_remaining']}**",
            f"- replaced: **{checks['n_replaced']}** | by algorithm: {checks['replacement_by_algorithm']}",
            f"- replacement balance (max-min): {checks['replacement_max_minus_min']}",
            f"- duplicate configs: {checks['duplicates'] or 'none'}",
            f"- uninstantiable / invalid hyper-params: {checks['uninstantiable'] or 'none'}",
            "",
            _table(rc),
            "",
        ]

    overall_ok = all(
        c["count_ok"]
        and c["constrained_remaining"] == 0
        and not c["duplicates"]
        and not c["uninstantiable"]
        for c in all_checks
    )
    lines.insert(1, f"\n**Overall PASS: {overall_ok}**\n")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Overall PASS: {overall_ok}")
    print(json.dumps(all_checks, indent=2))
    print(f"\nReport written to {args.out}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
