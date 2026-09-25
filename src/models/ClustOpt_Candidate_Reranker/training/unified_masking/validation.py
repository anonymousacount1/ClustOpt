"""Stage-2B3 integrity assertions."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

N_FOLDS = 15
N_REGIMES = 10
N_UNIFIED_FITS = 15
N_FOLD_REGIME_ROWS = 150
N_DATASETS = 16013
N_SLATES = 48039
DEDICATED_C2_TRAIN_ROWS = 1085658
CONTEXT_DIM = 347
N_UTIL = 120
FORBIDDEN_SPLIT = 1
OUTER_SPLITS = tuple(range(2, 17))


class Check:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, cid: str, desc: str, ok: bool, detail: Any = "") -> bool:
        self.rows.append({"check_id": cid, "description": desc,
                          "status": "PASS" if ok else "FAIL",
                          "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid:<5} {desc}"
              + (f"  ({detail})" if detail != "" else ""), flush=True)
        return ok

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.rows if r["status"] == "FAIL")
