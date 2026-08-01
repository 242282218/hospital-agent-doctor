from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Mapping, Set

from offline.artifacts import content_hash


_ALLOWED_POLLUTION_KINDS = {
    "cross_case_patch",
    "unsupported_clinical_fact",
    "answer_source_leak",
}
_REQUIRED_FIELDS = {
    "schema_version",
    "patient_id",
    "run_id",
    "evidence_path",
    "evidence_file_sha256",
    "evidence_excerpt",
    "evidence_excerpt_hash",
    "pollution_kind",
    "reviewer",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATIENT_RE = re.compile(r"^Patient_(?:Comorbid-)?\d+$")


def _project_relative_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("%s must be a POSIX project-relative path" % field)
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("%s must be a POSIX project-relative path" % field)
    return path


def _evidence_line(path: Path, source_bytes: bytes, *, line_number: Any) -> Dict[str, Any]:
    if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number < 1:
        raise ValueError("pollution evidence excerpt line must be a positive integer")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("pollution evidence must be valid UTF-8") from exc
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        lines = source_text.splitlines()
        if line_number > len(lines) or not lines[line_number - 1].strip():
            raise ValueError("pollution evidence excerpt line out of range")
        try:
            value = json.loads(lines[line_number - 1])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSONL pollution evidence line") from exc
    elif suffix == ".json":
        if line_number != 1:
            raise ValueError("JSON pollution evidence excerpt line must be 1")
        try:
            value = json.loads(source_text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON pollution evidence") from exc
    else:
        raise ValueError("pollution evidence must be JSON or JSONL")
    if not isinstance(value, dict):
        raise ValueError("pollution evidence line must be an object")
    return value


def _bound_patient_ids(value: Any) -> Set[str]:
    found: Set[str] = set()
    if isinstance(value, Mapping):
        for key in ("patient_id", "patientId"):
            patient_id = value.get(key)
            if isinstance(patient_id, str) and _PATIENT_RE.fullmatch(patient_id):
                found.add(patient_id)
        for item in value.values():
            found.update(_bound_patient_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_bound_patient_ids(item))
    return found


def _excerpt_parts(excerpt: Mapping[str, Any]) -> tuple[Any, Dict[str, Any]]:
    line_number = excerpt.get("line")
    row = excerpt.get("row")
    if row is None:
        source_row = {key: value for key, value in excerpt.items() if key != "line"}
    else:
        if set(excerpt) != {"line", "row"} or not isinstance(row, Mapping):
            raise ValueError("pollution evidence excerpt must contain line metadata and row content")
        source_row = dict(row)
    return line_number, source_row


def validate_pollution_receipt(
    receipt: Mapping[str, Any],
    *,
    project_root: Path,
    evidence_bytes: bytes | None = None,
) -> Dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ValueError("pollution receipt must be an object")
    fields = set(receipt)
    unknown = fields - _REQUIRED_FIELDS
    missing = _REQUIRED_FIELDS - fields
    if unknown:
        raise ValueError("pollution receipt unknown fields: %s" % sorted(unknown))
    if missing:
        raise ValueError("pollution receipt missing fields: %s" % sorted(missing))
    if receipt.get("schema_version") != "coverage-pollution-receipt/v1":
        raise ValueError("invalid pollution receipt schema_version")

    patient_id = receipt.get("patient_id")
    if not isinstance(patient_id, str) or not _PATIENT_RE.fullmatch(patient_id):
        raise ValueError("invalid pollution receipt patient_id")
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("pollution receipt run_id required")
    reviewer = receipt.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("pollution receipt reviewer required")
    pollution_kind = receipt.get("pollution_kind")
    if pollution_kind not in _ALLOWED_POLLUTION_KINDS:
        raise ValueError("unknown pollution kind")

    path = _project_relative_path(receipt.get("evidence_path"), field="evidence_path")
    project_root = Path(project_root).resolve()
    evidence_path = project_root.joinpath(*path.parts).resolve()
    try:
        evidence_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("evidence_path must be a POSIX project-relative path") from exc
    if not evidence_path.is_file():
        raise FileNotFoundError("pollution evidence file missing: %s" % path.as_posix())
    parts = path.parts
    if len(parts) >= 4 and parts[0:2] in {("outputs", "train"), ("outputs", "test")}:
        if parts[2] != run_id:
            raise ValueError("pollution evidence run_id mismatch")
    else:
        raise ValueError("pollution evidence path must bind run_id under outputs/train or outputs/test")

    expected_file_hash = receipt.get("evidence_file_sha256")
    if not isinstance(expected_file_hash, str) or not _SHA256_RE.fullmatch(expected_file_hash):
        raise ValueError("invalid evidence file sha256")
    source_bytes = evidence_path.read_bytes() if evidence_bytes is None else evidence_bytes
    if sha256(source_bytes).hexdigest() != expected_file_hash:
        raise ValueError("evidence file hash mismatch")

    excerpt = receipt.get("evidence_excerpt")
    if not isinstance(excerpt, Mapping) or not excerpt:
        raise ValueError("pollution evidence excerpt must be a non-empty object")
    line_number, excerpt_row = _excerpt_parts(excerpt)
    evidence_row = _evidence_line(evidence_path, source_bytes, line_number=line_number)
    if excerpt_row != evidence_row:
        raise ValueError("evidence excerpt does not match evidence line")
    if _bound_patient_ids(evidence_row) != {patient_id}:
        raise ValueError("pollution evidence patient_id mismatch")
    expected_excerpt_hash = receipt.get("evidence_excerpt_hash")
    if not isinstance(expected_excerpt_hash, str) or not _SHA256_RE.fullmatch(expected_excerpt_hash):
        raise ValueError("invalid evidence excerpt hash")
    if content_hash(excerpt) != expected_excerpt_hash:
        raise ValueError("evidence excerpt hash mismatch")

    return dict(receipt)
