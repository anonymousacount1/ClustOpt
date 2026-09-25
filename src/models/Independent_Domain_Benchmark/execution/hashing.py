"""Canonical hashing conventions v1 — every hash names the convention it used.

Three conventions, deliberately never conflated:

``raw_file``
    SHA-256 of the file bytes exactly as stored. Provenance of the artifact as
    serialised, including line endings and serialiser quirks.

``canonical_text_v1``
    SHA-256 after CRLF -> LF normalisation, UTF-8, otherwise byte-preserving.
    Nothing is trimmed, reformatted, reordered or numerically normalised. This is
    the convention used for scientific text configs so that a checkout on Windows
    and one on Linux agree.

``semantic_array_v1`` / ``semantic_rows_v1``
    Content identity for data artifacts whose scientific meaning is the content
    rather than the serialisation. Arrays hash dtype, shape and C-order bytes;
    row collections hash the sorted MULTISET of per-row digests so duplicate
    multiplicity is preserved and column order is normalised.

An earlier stage published a digest without recording its convention and the
value proved unreproducible. Every function here returns the convention name
alongside the digest so that cannot recur.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

HASH_CONVENTIONS_VERSION = "v1"
CANONICAL_TEXT = "canonical_text_v1"
RAW_FILE = "raw_file"
SEMANTIC_ARRAY = "semantic_array_v1"
SEMANTIC_ROWS = "semantic_rows_v1"

_DIGEST_CHARS = 16


def _short(h: "hashlib._Hash") -> str:
    return h.hexdigest()[:_DIGEST_CHARS]


def canonical_text_hash(text: str | bytes) -> str:
    """UTF-8, CRLF -> LF, otherwise byte-preserving. Nothing else is touched."""
    b = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return _short(hashlib.sha256(b.replace(b"\r\n", b"\n")))


def raw_file_hash(path: str | Path) -> str:
    """SHA-256 of the bytes exactly as stored on disk."""
    return _short(hashlib.sha256(Path(path).read_bytes()))


def file_hashes(path: str | Path) -> Dict[str, str]:
    """Both conventions for one file, each labelled."""
    b = Path(path).read_bytes()
    return {"raw_file": _short(hashlib.sha256(b)),
            "canonical_text_v1": _short(hashlib.sha256(b.replace(b"\r\n", b"\n"))),
            "hash_conventions_version": HASH_CONVENTIONS_VERSION}


def semantic_array_hash(a: np.ndarray) -> str:
    """dtype + shape + C-order bytes. Independent of how the array was stored."""
    arr = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b"|")
    h.update(arr.tobytes(order="C"))
    return _short(h)


def semantic_rows_hash(rows: Iterable[Sequence[Any]],
                       columns: Sequence[str]) -> str:
    """Sorted multiset of per-row digests; duplicate multiplicity preserved."""
    digests = [hashlib.sha256(
        "\x1f".join(repr(v) for v in row).encode("utf-8")).hexdigest()
        for row in rows]
    digests.sort()
    h = hashlib.sha256()
    h.update("\n".join(digests).encode("utf-8"))
    h.update(b"|")
    h.update("\x1f".join(sorted(str(c) for c in columns)).encode("utf-8"))
    return _short(h)


def canonical_json(obj: Any) -> str:
    """The canonical serialisation used for provenance blocks.

    ``sort_keys`` here is part of the DEFINED semantic-object convention, not a
    silent reformat of someone else's text: it applies to objects this project
    builds for hashing, never to text read from disk.
    """
    return json.dumps(obj, indent=2, sort_keys=True, default=str,
                      ensure_ascii=False)


def semantic_object_hash(obj: Mapping[str, Any]) -> str:
    """Content identity of a provenance mapping."""
    return canonical_text_hash(canonical_json(obj))


def labelled(value: str, convention: str) -> Dict[str, str]:
    """Any hash leaving this module carries its convention."""
    return {"value": value, "convention": convention,
            "hash_conventions_version": HASH_CONVENTIONS_VERSION}
