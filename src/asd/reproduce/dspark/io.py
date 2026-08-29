"""Small deterministic JSON/JSONL helpers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import canonical_json_bytes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: Path, payload: bytes) -> None:
    """Create a file atomically, refusing to replace different content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact with different bytes: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if immutable:
        write_immutable(path, payload)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def write_jsonl(path: Path, values: Iterable[Any], *, immutable: bool = False) -> None:
    payload = b"".join(canonical_json_bytes(value) for value in values)
    if immutable:
        write_immutable(path, payload)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL record must be an object at {path}:{line_number}"
                )
            values.append(value)
    return values
