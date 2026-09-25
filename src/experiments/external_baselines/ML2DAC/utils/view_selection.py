"""Best-view (record) selection for the ML2DAC baseline.

The dataset-level comparison metric is **Best View by ARI** (the same convention
AutoML4Clust uses and that ClustOpt's Best-View Mean ARI mirrors): of the three
record types run for a dataset, the dataset's headline result is taken from the
record whose external ARI is highest among the successful runs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def select_best_record_by_ari(
    record_payloads: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Pick the best record by ARI across the per-record result.json payloads.

    ``record_payloads`` maps record_type -> result.json payload (each carrying a
    ``metrics`` dict and ``status``). Returns a block with the best record's
    headline metrics, or Nones if nothing succeeded with a finite ARI.
    """
    best_record_type: Optional[str] = None
    best_ari = float("-inf")
    best_payload: Optional[Dict[str, Any]] = None

    for rtype, payload in record_payloads.items():
        if payload.get("status") != "success":
            continue
        metrics = payload.get("metrics", {}) or {}
        ari = metrics.get("ari")
        if ari is None:
            continue
        try:
            ari_f = float(ari)
        except (TypeError, ValueError):
            continue
        if ari_f == ari_f and ari_f > best_ari:  # not NaN
            best_ari = ari_f
            best_record_type = rtype
            best_payload = payload

    block: Dict[str, Any] = {
        "best_record_type": best_record_type,
        "best_ari": None,
        "best_nmi": None,
        "best_ami": None,
        "selected_k": None,
        "exact_k_match": None,
        "selected_cvi": None,
        "selected_algorithm": None,
        "warmstart_count": None,
    }
    if best_payload is not None:
        m = best_payload.get("metrics", {}) or {}
        block.update({
            "best_ari": m.get("ari"),
            "best_nmi": m.get("nmi"),
            "best_ami": m.get("ami"),
            "selected_k": best_payload.get("selected_k"),
            "exact_k_match": m.get("exact_k_match"),
            "selected_cvi": best_payload.get("selected_cvi"),
            "selected_algorithm": best_payload.get("selected_algorithm"),
            "warmstart_count": best_payload.get("warmstart_count"),
        })
    return block


__all__ = ["select_best_record_by_ari"]
