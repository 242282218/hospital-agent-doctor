from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from offline.artifacts import content_hash, file_hash, read_json, write_immutable_json


def ingest_episode(*, run_dir: Path, episode_dir: Path, episode_id: str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    trace_path = run_dir / "run.jsonl"
    seal_path = run_dir / "run.seal.json"
    if not trace_path.exists() or not seal_path.exists():
        raise FileNotFoundError("run.jsonl and run.seal.json are required")
    seal = read_json(seal_path)
    raw = trace_path.read_bytes()
    actual_hash = __import__("hashlib").sha256(raw).hexdigest()
    if seal.get("trace_hash") != actual_hash:
        raise ValueError("seal trace_hash mismatch")
    events = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    if int(seal.get("event_count") or -1) != len(events):
        raise ValueError("seal event_count mismatch")
    if any(isinstance(ev, Mapping) and ev.get("type") == "seal" for ev in events):
        raise ValueError("trace must not contain seal records")
    episode = {
        "schema_version": "episode/v1",
        "episode_id": episode_id,
        "run_id": seal.get("run_id"),
        "trace_hash": seal.get("trace_hash"),
        "event_count": len(events),
        "records": events,
        "seal": {
            "schema_version": seal.get("schema_version"),
            "trace_hash": seal.get("trace_hash"),
            "event_count": seal.get("event_count"),
            "sealed_at": seal.get("sealed_at"),
        },
    }
    episode_path = Path(episode_dir) / ("%s.json" % episode_id)
    digest = write_immutable_json(episode_path, episode)
    episode["episode_hash"] = digest
    return episode
