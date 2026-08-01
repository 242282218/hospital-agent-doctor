from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_immutable_json(path: Path, value: Any) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite immutable artifact: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json(value)
    path.write_text(body, encoding="utf-8")
    return sha256(body.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()
