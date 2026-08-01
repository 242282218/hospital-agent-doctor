from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Set, Tuple

from offline.artifacts import content_hash
from offline.coverage_pollution import validate_pollution_receipt


_PATIENT_PATTERN = r"Patient_(?:Comorbid-)?\d+"
_PATIENT_RE = re.compile(r"^%s$" % _PATIENT_PATTERN)
_PATIENT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])%s(?![A-Za-z0-9_-])" % _PATIENT_PATTERN
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIMARY_CLASSES = (
    "case-memory-covered",
    "manifest-anchored-batch-evaluated-provenance-only",
    "unanchored-evaluated",
    "final-only",
    "attempt-only",
    "offline-test-only",
)


@dataclass(frozen=True)
class CoverageInputs:
    project_root: Path
    train_outputs: Path
    test_outputs: Path
    trust_manifest: Path
    registry_path: Path
    release_manifest_path: Path
    offline_question_roots: tuple[Path, ...]
    pollution_receipts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _JsonlRow:
    value: Dict[str, Any]
    ref: Dict[str, Any]


@dataclass(frozen=True)
class _SourceBytes:
    path: Path
    relative: str
    data: bytes
    digest: str


def _project_path(path: Path, root: Path) -> str:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        parent = root.parent
        try:
            return PurePosixPath("..", path.relative_to(parent).as_posix()).as_posix()
        except ValueError as exc:
            raise ValueError("path must be inside project_root or its parent: %s" % path) from exc


def _input_path(path: Path, root: Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root).resolve() / path).resolve()


def _read_source(path: Path, *, root: Path) -> _SourceBytes:
    data = path.read_bytes()
    return _SourceBytes(
        path=path,
        relative=_project_path(path, root),
        data=data,
        digest=sha256(data).hexdigest(),
    )


def _source_ref(source: _SourceBytes, *, kind: str) -> Dict[str, Any]:
    return {"kind": kind, "path": source.relative, "file_sha256": source.digest}


def _source_text(source: _SourceBytes) -> str:
    try:
        return source.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("source must be valid UTF-8: %s" % source.relative) from exc


def _source_json(source: _SourceBytes) -> Any:
    try:
        return json.loads(_source_text(source))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON at %s" % source.relative) from exc


def _read_jsonl(source: _SourceBytes, *, kind: str) -> List[_JsonlRow]:
    rows: List[_JsonlRow] = []
    for line_number, raw in enumerate(_source_text(source).splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSONL at %s:%d" % (source.relative, line_number)) from exc
        if not isinstance(value, dict):
            raise ValueError("JSONL row must be an object at %s:%d" % (source.relative, line_number))
        rows.append(
            _JsonlRow(
                value=value,
                ref={
                    "kind": kind,
                    "path": source.relative,
                    "file_sha256": source.digest,
                    "line": line_number,
                    "row_hash": content_hash(value),
                },
            )
        )
    return rows


def _patient_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _PATIENT_RE.fullmatch(value) else None


def _patient_ids(value: Any) -> Set[str]:
    found: Set[str] = set()
    if isinstance(value, Mapping):
        patient_id = _patient_id(value.get("patient_id")) or _patient_id(value.get("patientId"))
        if patient_id:
            found.add(patient_id)
        for item in value.values():
            found.update(_patient_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_patient_ids(item))
    return found


def _validate_registry(
    registry_source: _SourceBytes,
    release_manifest_source: _SourceBytes,
) -> Tuple[Mapping[str, Any], List[Dict[str, Any]]]:
    registry = _source_json(registry_source)
    release = _source_json(release_manifest_source)
    if not isinstance(registry, Mapping) or registry.get("schema_version") != "verified-registry/v1":
        raise ValueError("invalid verified registry")
    registry_body = {key: value for key, value in registry.items() if key != "registry_hash"}
    if content_hash(registry_body) != registry.get("registry_hash"):
        raise ValueError("verified registry content hash mismatch")
    if not isinstance(release, Mapping) or release.get("registry_hash") != registry_source.digest:
        raise ValueError("release manifest registry hash mismatch")
    assets = registry.get("assets")
    if not isinstance(assets, list):
        raise ValueError("verified registry assets required")
    return registry, assets


def _validate_trust_manifest(manifest: Any) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("invalid trust manifest")
    if manifest.get("schema_version") != "architecture-baseline/v1":
        raise ValueError("invalid trust manifest schema_version")
    expected = manifest.get("manifest_hash")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
        raise ValueError("trust manifest hash required")
    if content_hash(body) != expected:
        raise ValueError("trust manifest hash mismatch")
    return manifest


def _manifest_runs(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    runs = manifest.get("historical_runs")
    if not isinstance(runs, list):
        raise ValueError("trust manifest historical_runs required")
    result: Dict[str, Mapping[str, Any]] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("trust manifest run must be an object")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("trust manifest run_id required")
        if run_id in result:
            raise ValueError("duplicate trust manifest run_id: %s" % run_id)
        result[run_id] = run
    return result


def _verify_anchored_run(
    run_dir: Path,
    run: Mapping[str, Any],
    sources: Mapping[str, _SourceBytes],
) -> None:
    hashes = run.get("artifact_hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("anchored artifact hashes required")
    names = {"events.jsonl", "final_results.jsonl"}
    if run.get("mode") == "train" and run.get("has_evaluation") is True:
        names.add("evaluation_results.jsonl")
    for name in sorted(names):
        expected = hashes.get(name)
        source = sources.get(name)
        if expected is None:
            if name == "evaluation_results.jsonl":
                raise ValueError("anchored evaluation hash required: %s" % run_dir.name)
            if source is None:
                continue
        if not isinstance(expected, str) or source is None or source.digest != expected:
            raise ValueError("anchored artifact hash mismatch: %s/%s" % (run_dir.name, name))


def _report_patient_ids(report: Mapping[str, Any]) -> Set[str]:
    details = report.get("treatment_details")
    if not isinstance(details, list):
        return set()
    result: Set[str] = set()
    for item in details:
        if not isinstance(item, Mapping):
            raise ValueError("batch evaluation treatment detail must be an object")
        patient_id = _patient_id(item.get("patient_id"))
        if not patient_id:
            raise ValueError("batch evaluation detail patient_id required")
        if patient_id in result:
            raise ValueError("duplicate patient in batch evaluation report: %s" % patient_id)
        result.add(patient_id)
    return result


def _validate_train_evaluations(
    *,
    evaluations: Iterable[_JsonlRow],
    events: Iterable[_JsonlRow],
    finals: Iterable[_JsonlRow],
    run_id: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    final_values: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in finals:
        patient_id = _patient_id(row.value.get("patient_id"))
        if patient_id and row.value.get("finished") is True:
            final_values[patient_id].append(row.value)

    request_finals: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    case_end_counts: Counter[str] = Counter()
    event_reports: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        value = row.value
        event_type = value.get("event_type")
        patient_id = _patient_id(value.get("patient_id"))
        payload = value.get("payload")
        payload_patient = _patient_id(payload.get("patient_id")) if isinstance(payload, Mapping) else None
        if event_type == "EVALUATION_REQUEST" and value.get("status") == "success":
            final_result = payload.get("final_result") if isinstance(payload, Mapping) else None
            final_patient = _patient_id(final_result.get("patient_id")) if isinstance(final_result, Mapping) else None
            if not patient_id or payload_patient != patient_id or final_patient != patient_id:
                raise ValueError("train evaluation request binding mismatch: %s" % run_id)
            request_finals[patient_id].append(final_result)
        elif event_type == "EVALUATION_RESULT" and value.get("status") == "success":
            report = payload.get("report") if isinstance(payload, Mapping) else None
            report_patient = _patient_id(report.get("patientId")) if isinstance(report, Mapping) else None
            if not patient_id or payload_patient != patient_id or report_patient != patient_id:
                raise ValueError("train evaluation patient binding mismatch: %s" % run_id)
            if not isinstance(report, Mapping) or report.get("status") != "evaluated":
                raise ValueError("train evaluation status must be evaluated: %s" % run_id)
            event_reports[patient_id].append(report)
        elif event_type == "CASE_END" and value.get("status") == "success":
            if (
                not patient_id
                or payload_patient != patient_id
                or not isinstance(payload, Mapping)
                or payload.get("finished") is not True
            ):
                raise ValueError("train case end binding mismatch: %s" % run_id)
            case_end_counts[patient_id] += 1

    validated: List[Tuple[str, Dict[str, Any]]] = []
    evaluation_counts: Counter[str] = Counter()
    for row in evaluations:
        value = row.value
        patient_id = _patient_id(value.get("patient_id"))
        report = value.get("report")
        report_patient = _patient_id(report.get("patientId")) if isinstance(report, Mapping) else None
        if not patient_id or report_patient != patient_id:
            raise ValueError("train evaluation patient binding mismatch: %s" % run_id)
        evaluation_counts[patient_id] += 1
        if evaluation_counts[patient_id] != 1:
            raise ValueError("train evaluation requires unique evaluation row: %s/%s" % (run_id, patient_id))
        if not isinstance(report, Mapping) or report.get("status") != "evaluated":
            raise ValueError("train evaluation status must be evaluated: %s" % run_id)
        finals_for_patient = final_values.get(patient_id, [])
        if len(finals_for_patient) != 1:
            raise ValueError("train evaluation requires unique finished final: %s/%s" % (run_id, patient_id))
        requests = request_finals.get(patient_id, [])
        if len(requests) != 1 or requests[0] != finals_for_patient[0]:
            raise ValueError("train evaluation request final mismatch: %s/%s" % (run_id, patient_id))
        if case_end_counts[patient_id] != 1:
            raise ValueError("train case end binding mismatch: %s/%s" % (run_id, patient_id))
        reports = event_reports.get(patient_id, [])
        if len(reports) != 1:
            raise ValueError("train evaluation requires unique successful event: %s/%s" % (run_id, patient_id))
        if reports[0] != report:
            raise ValueError("train evaluation report mismatch: %s/%s" % (run_id, patient_id))
        validated.append((patient_id, row.ref))
    return validated


def _successful_batch_report(
    events: Iterable[_JsonlRow],
    *,
    run_id: str,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any] | None]:
    matches: List[Tuple[Mapping[str, Any], Dict[str, Any]]] = []
    for row in events:
        value = row.value
        if value.get("event_type") != "BATCH_EVALUATION_RESULT" or value.get("status") != "success":
            continue
        payload = value.get("payload")
        report = payload.get("report") if isinstance(payload, Mapping) else None
        if not isinstance(report, Mapping):
            raise ValueError("successful batch evaluation result report required: %s" % run_id)
        matches.append((report, row.ref))
    if not matches:
        return None, None
    if len(matches) != 1:
        raise ValueError("expected unique successful BATCH_EVALUATION_RESULT: %s" % run_id)
    return matches[0]


def _add_ref(refs: DefaultDict[str, List[Dict[str, Any]]], patient_id: str, ref: Dict[str, Any]) -> None:
    refs[patient_id].append(dict(ref))


def _scan_outputs(
    *,
    outputs: Path,
    mode: str,
    project_root: Path,
    manifest_runs: Mapping[str, Mapping[str, Any]],
    attempts: Set[str],
    finished: Set[str],
    evaluated: Set[str],
    anchored_evaluated: Set[str],
    refs: DefaultDict[str, List[Dict[str, Any]]],
    sources: List[Dict[str, Any]],
    source_cache: MutableMapping[Path, _SourceBytes],
) -> Set[str]:
    all_ids: Set[str] = set()
    if not outputs.is_dir():
        raise FileNotFoundError("%s outputs missing: %s" % (mode, outputs))
    for run_dir in sorted((path for path in outputs.iterdir() if path.is_dir()), key=lambda path: path.name):
        run_id = run_dir.name
        events_path = run_dir / "events.jsonl"
        finals_path = run_dir / "final_results.jsonl"
        evaluations_path = run_dir / "evaluation_results.jsonl"
        report_path = run_dir / "final_results_eval_report.json"
        run_sources = {
            path.name: _read_source(path, root=project_root)
            for path in (events_path, finals_path, evaluations_path, report_path)
            if path.is_file()
        }
        source_cache.update({source.path.resolve(): source for source in run_sources.values()})

        anchored = manifest_runs.get(run_id)
        if anchored is not None:
            if anchored.get("mode") != mode:
                raise ValueError("anchored run mode mismatch: %s" % run_id)
            _verify_anchored_run(run_dir, anchored, run_sources)

        events_source = run_sources.get(events_path.name)
        finals_source = run_sources.get(finals_path.name)
        evaluations_source = run_sources.get(evaluations_path.name)
        report_source = run_sources.get(report_path.name)
        events = _read_jsonl(events_source, kind="%s-event" % mode) if events_source is not None else []
        finals = _read_jsonl(finals_source, kind="%s-final" % mode) if finals_source is not None else []
        evaluations = (
            _read_jsonl(evaluations_source, kind="%s-evaluation" % mode)
            if evaluations_source is not None
            else []
        )
        for source, kind in (
            (events_source, "%s-events-file" % mode),
            (finals_source, "%s-finals-file" % mode),
            (evaluations_source, "%s-evaluations-file" % mode),
            (report_source, "%s-evaluation-report" % mode),
        ):
            if source is not None:
                sources.append(_source_ref(source, kind=kind))

        for row in events:
            patient_id = _patient_id(row.value.get("patient_id"))
            if patient_id:
                attempts.add(patient_id)
                all_ids.add(patient_id)
                _add_ref(refs, patient_id, row.ref)
        finished_in_run: Set[str] = set()
        for row in finals:
            patient_id = _patient_id(row.value.get("patient_id"))
            if not patient_id:
                continue
            all_ids.add(patient_id)
            _add_ref(refs, patient_id, row.ref)
            if row.value.get("finished") is True:
                finished.add(patient_id)
                finished_in_run.add(patient_id)

        if mode == "train":
            if evaluations and anchored is None:
                raise ValueError("train evaluation must be manifest anchored: %s" % run_id)
            if anchored is not None and bool(evaluations) != (anchored.get("has_evaluation") is True):
                raise ValueError("train has_evaluation mismatch: %s" % run_id)
            for patient_id, evaluation_ref in _validate_train_evaluations(
                evaluations=evaluations,
                events=events,
                finals=finals,
                run_id=run_id,
            ):
                all_ids.add(patient_id)
                evaluated.add(patient_id)
                anchored_evaluated.add(patient_id)
                _add_ref(refs, patient_id, evaluation_ref)
        else:
            report, event_ref = _successful_batch_report(events, run_id=run_id)
            if report is not None:
                if report_source is not None and _source_json(report_source) != report:
                    raise ValueError("batch evaluation report file does not match event: %s" % run_id)
                report_ids = _report_patient_ids(report)
                for patient_id in sorted(report_ids & finished_in_run):
                    evaluated.add(patient_id)
                    all_ids.add(patient_id)
                    if anchored is not None:
                        anchored_evaluated.add(patient_id)
                    if event_ref is not None:
                        _add_ref(refs, patient_id, event_ref)
                    if report_source is not None:
                        _add_ref(
                            refs,
                            patient_id,
                            _source_ref(report_source, kind="test-evaluation-report"),
                        )
    return all_ids


def _scan_offline_questions(
    roots: Iterable[Path],
    *,
    project_root: Path,
    refs: DefaultDict[str, List[Dict[str, Any]]],
    sources: List[Dict[str, Any]],
) -> Set[str]:
    result: Set[str] = set()
    for root in sorted((_input_path(path, project_root) for path in roots), key=lambda path: path.as_posix()):
        if not root.is_dir():
            raise FileNotFoundError("offline question root missing: %s" % root)
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            source_bytes = _read_source(path, root=project_root)
            source = _source_ref(source_bytes, kind="offline-question")
            sources.append(source)
            ids = set(_PATIENT_TOKEN_RE.findall(_source_text(source_bytes)))
            for patient_id in ids:
                result.add(patient_id)
                _add_ref(refs, patient_id, source)
    return result


def _primary_class(
    patient_id: str,
    *,
    case_memory: Set[str],
    anchored_evaluated: Set[str],
    evaluated: Set[str],
    finished: Set[str],
    attempts: Set[str],
    offline: Set[str],
) -> str:
    if patient_id in case_memory:
        return _PRIMARY_CLASSES[0]
    if patient_id in anchored_evaluated:
        return _PRIMARY_CLASSES[1]
    if patient_id in evaluated:
        return _PRIMARY_CLASSES[2]
    if patient_id in finished:
        return _PRIMARY_CLASSES[3]
    if patient_id in attempts:
        return _PRIMARY_CLASSES[4]
    if patient_id in offline:
        return _PRIMARY_CLASSES[5]
    raise ValueError("patient has no coverage class: %s" % patient_id)


def build_coverage_snapshot(inputs: CoverageInputs) -> Dict[str, Any]:
    project_root = Path(inputs.project_root).resolve()
    train_outputs = _input_path(inputs.train_outputs, project_root)
    test_outputs = _input_path(inputs.test_outputs, project_root)
    trust_manifest_path = _input_path(inputs.trust_manifest, project_root)
    registry_path = _input_path(inputs.registry_path, project_root)
    release_manifest_path = _input_path(inputs.release_manifest_path, project_root)

    trust_manifest_source = _read_source(trust_manifest_path, root=project_root)
    registry_source = _read_source(registry_path, root=project_root)
    release_manifest_source = _read_source(release_manifest_path, root=project_root)
    trust_manifest = _validate_trust_manifest(_source_json(trust_manifest_source))
    manifest_runs = _manifest_runs(trust_manifest)
    _, assets = _validate_registry(registry_source, release_manifest_source)

    refs: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    sources: List[Dict[str, Any]] = [
        _source_ref(trust_manifest_source, kind="trust-manifest"),
        _source_ref(registry_source, kind="release-registry"),
        _source_ref(release_manifest_source, kind="release-manifest"),
    ]
    source_cache: Dict[Path, _SourceBytes] = {
        source.path.resolve(): source
        for source in (trust_manifest_source, registry_source, release_manifest_source)
    }
    case_memory: Set[str] = set()
    for asset in assets:
        if not isinstance(asset, Mapping) or asset.get("candidate_type") != "case_memory":
            continue
        content = asset.get("content")
        patient_id = _patient_id(content.get("patient_id")) if isinstance(content, Mapping) else None
        if not patient_id:
            raise ValueError("case-memory asset patient_id required")
        if patient_id in case_memory:
            raise ValueError("duplicate case-memory patient: %s" % patient_id)
        case_memory.add(patient_id)
        _add_ref(refs, patient_id, sources[1])

    attempts: Set[str] = set()
    finished: Set[str] = set()
    evaluated: Set[str] = set()
    anchored_evaluated: Set[str] = set()
    train_ids = _scan_outputs(
        outputs=train_outputs,
        mode="train",
        project_root=project_root,
        manifest_runs=manifest_runs,
        attempts=attempts,
        finished=finished,
        evaluated=evaluated,
        anchored_evaluated=anchored_evaluated,
        refs=refs,
        sources=sources,
        source_cache=source_cache,
    )
    test_ids = _scan_outputs(
        outputs=test_outputs,
        mode="test",
        project_root=project_root,
        manifest_runs=manifest_runs,
        attempts=attempts,
        finished=finished,
        evaluated=evaluated,
        anchored_evaluated=anchored_evaluated,
        refs=refs,
        sources=sources,
        source_cache=source_cache,
    )
    offline_ids = _scan_offline_questions(
        inputs.offline_question_roots,
        project_root=project_root,
        refs=refs,
        sources=sources,
    )

    occurrence_ids = case_memory | train_ids | test_ids | offline_ids
    polluted: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for receipt_input in inputs.pollution_receipts:
        receipt_path = _input_path(receipt_input, project_root)
        receipt_source = _read_source(receipt_path, root=project_root)
        receipt = _source_json(receipt_source)
        evidence_value = receipt.get("evidence_path")
        evidence_path = (
            project_root.joinpath(*PurePosixPath(evidence_value).parts).resolve()
            if isinstance(evidence_value, str)
            else project_root
        )
        evidence_source = source_cache.get(evidence_path)
        validated = validate_pollution_receipt(
            receipt,
            project_root=project_root,
            evidence_bytes=evidence_source.data if evidence_source is not None else None,
        )
        patient_id = validated["patient_id"]
        if patient_id not in occurrence_ids:
            raise ValueError("pollution receipt patient has no coverage occurrence: %s" % patient_id)
        polluted[patient_id].append(
            {
                "kind": validated["pollution_kind"],
                "reviewer": validated["reviewer"],
                "receipt_path": _project_path(receipt_path, project_root),
                "receipt_file_sha256": receipt_source.digest,
                "evidence_path": validated["evidence_path"],
                "evidence_file_sha256": validated["evidence_file_sha256"],
                "evidence_excerpt_hash": validated["evidence_excerpt_hash"],
            }
        )
        sources.append(_source_ref(receipt_source, kind="pollution-receipt"))

    patient_ids = occurrence_ids
    patients: List[Dict[str, Any]] = []
    for patient_id in sorted(patient_ids):
        item = {
            "patient_id": patient_id,
            "primary_class": _primary_class(
                patient_id,
                case_memory=case_memory,
                anchored_evaluated=anchored_evaluated,
                evaluated=evaluated,
                finished=finished,
                attempts=attempts,
                offline=offline_ids,
            ),
            "flags": {
                "batch_evaluated": patient_id in evaluated,
                "manifest_anchored": patient_id in anchored_evaluated,
                "offline_test_covered": patient_id in offline_ids,
                "polluted": patient_id in polluted,
                "test_seen": patient_id in test_ids,
                "train_seen": patient_id in train_ids,
            },
            "pollution_receipts": sorted(
                polluted.get(patient_id, []),
                key=lambda value: (value["receipt_path"], value["kind"], value["reviewer"]),
            ),
            "refs": sorted(
                refs.get(patient_id, []),
                key=lambda value: (
                    value["path"],
                    value.get("line", 0),
                    value["kind"],
                    value.get("row_hash", ""),
                ),
            ),
        }
        patients.append(item)

    primary_counts = Counter(item["primary_class"] for item in patients)
    source_values = sorted(
        {json.dumps(item, ensure_ascii=False, sort_keys=True): item for item in sources}.values(),
        key=lambda value: (value["path"], value["kind"], value["file_sha256"]),
    )
    body = {
        "schema_version": "coverage_snapshot/v1",
        "counts": {
            "unique_patients": len(patients),
            "primary": {name: primary_counts.get(name, 0) for name in _PRIMARY_CLASSES},
            "sources": {
                "test": len(test_ids),
                "train": len(train_ids),
                "offline": len(offline_ids),
                "test_offline_intersection": len(test_ids & offline_ids),
            },
            "flags": {
                "polluted": len(polluted),
                "offline_test_covered": len(offline_ids),
            },
        },
        "patients": patients,
        "patient_content_ids": {item["patient_id"]: content_hash(item) for item in patients},
        "sources": source_values,
    }
    return {**body, "snapshot_id": content_hash(body)}
