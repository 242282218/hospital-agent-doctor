from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from offline.artifacts import content_hash, file_hash, write_immutable_json


_SOURCE_FILES = ("evaluation_results.jsonl", "events.jsonl", "final_results.jsonl")


def scan_train_runs(train_outputs: Path) -> List[Dict[str, Any]]:
    """Build historical_runs entries for all train_* dirs with artifact hashes."""
    train_outputs = Path(train_outputs)
    if not train_outputs.is_dir():
        raise FileNotFoundError("train outputs not found: %s" % train_outputs)

    runs: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in train_outputs.iterdir() if path.is_dir()):
        run_id = run_dir.name
        if not run_id.startswith("train_"):
            continue
        artifact_hashes: Dict[str, str] = {}
        for name in _SOURCE_FILES:
            path = run_dir / name
            if path.exists():
                artifact_hashes[name] = file_hash(path)
        has_evaluation = "evaluation_results.jsonl" in artifact_hashes
        if not artifact_hashes:
            continue
        runs.append(
            {
                "run_id": run_id,
                "mode": "train",
                "has_evaluation": has_evaluation,
                "dataset_layer": "HistoricalReplay",
                "artifact_hashes": artifact_hashes,
            }
        )
    return runs


def build_train_trust_manifest(
    *,
    train_outputs: Path,
    base_manifest: Optional[Mapping[str, Any]] = None,
    only_with_evaluation: bool = False,
) -> Dict[str, Any]:
    """Content-addressable trust manifest for case-memory import.

    When base_manifest is provided, non-historical fields are preserved; historical_runs
    are replaced by a fresh scan of train_outputs (hash-verified at import time).
    """
    runs = scan_train_runs(train_outputs)
    if only_with_evaluation:
        runs = [item for item in runs if item.get("has_evaluation") is True]

    body: Dict[str, Any] = {}
    if base_manifest is not None:
        body = {key: value for key, value in dict(base_manifest).items() if key != "historical_runs"}
    body["schema_version"] = "train-trust-manifest/v1"
    body["historical_runs"] = runs
    body["train_outputs"] = str(Path(train_outputs).as_posix())
    body["trusted_train_run_count"] = sum(1 for item in runs if item.get("has_evaluation") is True)
    body["scanned_train_run_count"] = len(runs)
    body["manifest_hash"] = content_hash(
        {
            "schema_version": body["schema_version"],
            "historical_runs": runs,
            "trusted_train_run_count": body["trusted_train_run_count"],
        }
    )
    return body


def write_train_trust_manifest(
    manifest: Mapping[str, Any],
    *,
    path: Path,
) -> Dict[str, Any]:
    path = Path(path)
    write_immutable_json(path, dict(manifest))
    return {"path": str(path), "manifest_hash": manifest.get("manifest_hash"), "file_hash": file_hash(path)}
