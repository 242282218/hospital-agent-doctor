from __future__ import annotations

import argparse
import errno
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from offline.artifacts import canonical_json, content_hash
from offline.coverage_snapshot import CoverageInputs, build_coverage_snapshot


def _validate_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(snapshot)
    if value.get("schema_version") != "coverage_snapshot/v1":
        raise ValueError("invalid coverage snapshot schema")
    snapshot_id = value.get("snapshot_id")
    if not isinstance(snapshot_id, str) or len(snapshot_id) != 64:
        raise ValueError("coverage snapshot_id required")
    body = {key: item for key, item in value.items() if key != "snapshot_id"}
    if content_hash(body) != snapshot_id:
        raise ValueError("coverage snapshot_id mismatch")
    return value


def _published_snapshot_matches(path: Path, body: bytes) -> bool:
    try:
        return path.read_bytes() == body
    except FileNotFoundError:
        return False


def _exclusive_publish(temp_path: Path, path: Path) -> None:
    try:
        os.link(temp_path, path)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EPERM} and os.name == "nt":
            os.rename(temp_path, path)
            return
        raise


def write_coverage_snapshot(
    snapshot: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> Dict[str, Any]:
    value = _validate_snapshot(snapshot)
    path = Path(artifact_root) / value["snapshot_id"] / "coverage_snapshot.json"
    body = canonical_json(value).encode("utf-8")
    if _published_snapshot_matches(path, body):
        return {"snapshot_id": value["snapshot_id"], "path": str(path), "reused": True}
    if path.exists():
        raise FileExistsError("immutable coverage snapshot differs: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _exclusive_publish(temp_path, path)
            published = True
        except FileExistsError:
            if _published_snapshot_matches(path, body):
                return {"snapshot_id": value["snapshot_id"], "path": str(path), "reused": True}
            raise FileExistsError("immutable coverage snapshot differs: %s" % path) from None
    finally:
        if temp_path is not None and (published or temp_path.exists()):
            temp_path.unlink(missing_ok=True)

    return {"snapshot_id": value["snapshot_id"], "path": str(path), "reused": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an immutable offline coverage snapshot.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-outputs", type=Path, required=True)
    parser.add_argument("--test-outputs", type=Path, required=True)
    parser.add_argument("--trust-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--offline-question-root", type=Path, action="append", required=True)
    parser.add_argument("--pollution-receipt", type=Path, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = build_coverage_snapshot(
        CoverageInputs(
            project_root=args.project_root,
            train_outputs=args.train_outputs,
            test_outputs=args.test_outputs,
            trust_manifest=args.trust_manifest,
            registry_path=args.registry,
            release_manifest_path=args.release_manifest,
            offline_question_roots=tuple(args.offline_question_root),
            pollution_receipts=tuple(args.pollution_receipt),
        )
    )
    result = write_coverage_snapshot(snapshot, artifact_root=args.artifact_root)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
