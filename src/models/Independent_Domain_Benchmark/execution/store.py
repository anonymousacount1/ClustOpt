"""Stage-4 result store: atomic completion, immutable COMPLETE, strict resume.

The integrity model rests on persisted state, not on recomputation. Since some
floating metric kernels are not bitwise reproducible (see
``configs/numerical_reproducibility_policy.json``), "recompute and compare" is
not a validity test. Instead:

* a record becomes ``COMPLETE`` only by an atomic rename of a fully written
  temporary file, so a crash leaves either the previous valid record or clearly
  incomplete work -- never a partial file that reads as COMPLETE;
* a ``COMPLETE`` record whose provenance and content integrity check out is
  **authoritative** and is reused exactly, with zero scientific recomputation;
* a ``COMPLETE`` record that fails integrity is **never silently regenerated**:
  it raises, because replacing a frozen scientific output could change the
  result;
* a ``COMPLETE`` record whose provenance does not match the current frozen
  identity is not reused and not overwritten -- the caller is told to open a new
  run namespace.

Layer separation is deliberate: run manifest, dataset/view preparation, physical
traces, scientific arms and offline evaluation are distinct record kinds, and a
scientific arm record references its physical trace by id rather than copying
the candidate partitions.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from . import hashing as H

CONTRACT_VERSION = "idb_stage4_store_v1"


class State(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class Layer(str, Enum):
    RUN = "run"                       # manifest + environment
    PREPARATION = "preparation"       # dataset/view preparation metadata
    PHYSICAL = "physical"             # physical trace records
    SCIENTIFIC = "scientific"         # scientific arm outputs
    EVALUATION = "evaluation"         # offline evaluation, written later


class IntegrityError(RuntimeError):
    """A COMPLETE record is corrupt. Never auto-repaired."""


class ProvenanceMismatch(RuntimeError):
    """A COMPLETE record belongs to a different frozen identity."""


@dataclass(frozen=True)
class Reuse:
    """The outcome of asking the store whether stored work can be reused."""

    reusable: bool
    reason: str
    record: Optional[Mapping[str, Any]] = None


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Write fully to a temp file, fsync, then atomically replace.

    Returns the canonical content hash of what was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = H.canonical_json(payload)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)          # atomic on POSIX and on Windows
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return H.canonical_text_hash(text)


class ResultStore:
    """Layered, resumable, provenance-checked store."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.run_dir = self.root / run_id
        for layer in Layer:
            (self.run_dir / layer.value).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- paths
    def path(self, layer: Layer, key: str) -> Path:
        return self.run_dir / layer.value / ("%s.json" % key)

    # ------------------------------------------------------------ writing
    def mark(self, layer: Layer, key: str, state: State,
             payload: Optional[Mapping[str, Any]] = None) -> None:
        """Record a non-final state. Never produces a reusable record."""
        if state is State.COMPLETE:
            raise ValueError("use complete() to finalise a record")
        body = dict(payload or {})
        body.update({"state": state.value, "contract_version": CONTRACT_VERSION,
                     "run_id": self.run_id, "layer": layer.value, "key": key})
        atomic_write_json(self.path(layer, key), body)

    def complete(self, layer: Layer, key: str, *, provenance: Mapping[str, Any],
                 content: Mapping[str, Any]) -> Mapping[str, Any]:
        """Atomically finalise a record as COMPLETE and authoritative."""
        body: Dict[str, Any] = {
            "state": State.COMPLETE.value,
            "contract_version": CONTRACT_VERSION,
            "hash_conventions_version": H.HASH_CONVENTIONS_VERSION,
            "run_id": self.run_id, "layer": layer.value, "key": key,
            "provenance": dict(provenance), "content": dict(content),
        }
        body["provenance_hash"] = H.semantic_object_hash(body["provenance"])
        body["content_hash"] = H.semantic_object_hash(body["content"])
        atomic_write_json(self.path(layer, key), body)
        return body

    # ------------------------------------------------------------ reading
    def load(self, layer: Layer, key: str) -> Optional[Mapping[str, Any]]:
        p = self.path(layer, key)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise IntegrityError("malformed record %s/%s: %s"
                                 % (layer.value, key, type(exc).__name__))

    def check_integrity(self, layer: Layer, key: str,
                        rec: Mapping[str, Any]) -> None:
        """A COMPLETE record must be internally consistent. Raises, never repairs."""
        where = "%s/%s" % (layer.value, key)
        if rec.get("contract_version") != CONTRACT_VERSION:
            raise IntegrityError("%s: contract version %r != %r"
                                 % (where, rec.get("contract_version"),
                                    CONTRACT_VERSION))
        for field in ("provenance", "content", "provenance_hash", "content_hash"):
            if field not in rec:
                raise IntegrityError("%s: missing %s" % (where, field))
        if H.semantic_object_hash(rec["provenance"]) != rec["provenance_hash"]:
            raise IntegrityError("%s: provenance hash mismatch" % where)
        if H.semantic_object_hash(rec["content"]) != rec["content_hash"]:
            raise IntegrityError("%s: content hash mismatch" % where)
        for ref in rec["content"].get("referenced_files", []):
            if not (self.run_dir / ref).is_file():
                raise IntegrityError("%s: referenced file missing: %s" % (where, ref))

    def try_reuse(self, layer: Layer, key: str,
                  expected_provenance: Mapping[str, Any]) -> Reuse:
        """The single resume decision point.

        COMPLETE + integrity + provenance match -> reuse, zero recomputation.
        COMPLETE + integrity failure            -> raise (never auto-replace).
        COMPLETE + provenance mismatch          -> raise (open a new run).
        anything else                           -> recompute; it was never
                                                   authoritative.
        """
        rec = self.load(layer, key)
        if rec is None:
            return Reuse(False, "absent")
        state = rec.get("state")
        if state != State.COMPLETE.value:
            return Reuse(False, "not COMPLETE (%s)" % state)
        self.check_integrity(layer, key, rec)     # raises IntegrityError
        expected = H.semantic_object_hash(dict(expected_provenance))
        if expected != rec.get("provenance_hash"):
            diff = sorted(k for k in set(expected_provenance) | set(rec["provenance"])
                          if expected_provenance.get(k) != rec["provenance"].get(k))
            raise ProvenanceMismatch(
                "%s/%s belongs to a different frozen identity; differing keys: %s. "
                "Open a new run namespace rather than overwriting a frozen result."
                % (layer.value, key, diff))
        return Reuse(True, "authoritative COMPLETE", rec)

    # --------------------------------------------------------- inventory
    def keys(self, layer: Layer) -> list:
        return sorted(p.stem for p in (self.run_dir / layer.value).glob("*.json"))

    def count_complete(self, layer: Layer) -> int:
        n = 0
        for k in self.keys(layer):
            rec = self.load(layer, k)
            if rec and rec.get("state") == State.COMPLETE.value:
                n += 1
        return n
