from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.architecture.freeze_baseline import (
    _run_command,
    build_deployment_bundle,
    REQUIRED_JSON_PATHS,
    build_release_identity,
    build_manifest,
    canonical_hash,
    check_manifest,
    collect_snapshot_files,
    release_identity_matches_manifest,
    require_snapshot_matches_head,
    validate_json_files,
    scan_python_source,
    sha256_file,
    snapshot_matches_commit,
    validate_base_image_lock,
    validate_dependency_lock,
    verify_manifest_hash,
    verify_release_identity,
    write_manifest,
)


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_collect_snapshot_files_excludes_runtime_outputs(tmp_path: Path) -> None:
    (tmp_path / ".dockerignore").write_text(".git\n", encoding="utf-8")
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "analyze_eval.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "outputs" / "test").mkdir(parents=True)
    (tmp_path / "outputs" / "test" / "result.json").write_text("{}", encoding="utf-8")

    files = collect_snapshot_files(tmp_path)

    assert [item["path"] for item in files] == [
        ".dockerignore",
        "agent/agent.py",
        "analyze_eval.py",
    ]


def test_collect_snapshot_files_normalizes_text_line_endings(tmp_path: Path) -> None:
    (tmp_path / "agent").mkdir()
    source = tmp_path / "agent" / "agent.py"
    source.write_bytes(b"VALUE = 1\r\nVALUE = 2\r\n")
    crlf_item = collect_snapshot_files(tmp_path)[0]

    source.write_bytes(b"VALUE = 1\nVALUE = 2\n")
    lf_item = collect_snapshot_files(tmp_path)[0]

    assert crlf_item["sha256"] == lf_item["sha256"]
    assert crlf_item["bytes"] == lf_item["bytes"]


def test_scan_python_source_reports_direct_action_and_patient_literal(tmp_path: Path) -> None:
    source = tmp_path / "legacy.py"
    source.write_text(
        "\n".join(
            [
                "async def run(self):",
                "    await self.actions.ask_patient('Patient_123', {})",
                "",
                "def append_eleventh_round_facts():",
                "    return None",
            ]
        ),
        encoding="utf-8",
    )

    report = scan_python_source([source], root=tmp_path)

    assert report["direct_action_calls"][0]["action"] == "ask_patient"
    assert report["patient_id_literals"][0]["value"] == "Patient_123"
    assert report["experiment_round_names"][0]["value"] == "append_eleventh_round_facts"


def test_build_manifest_has_required_sections_and_valid_hash(tmp_path: Path) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "data" / "ref_data").mkdir(parents=True)
    (tmp_path / "data" / "ref_data" / "diseases_catalog.json").write_text(
        json.dumps({"内科": ["疾病"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = build_manifest(
        tmp_path,
        verification={"pytest": {"status": "passed", "summary": "1 passed"}},
    )

    assert manifest["schema_version"] == "architecture-baseline/v1"
    assert manifest["snapshot"]["files"]
    assert manifest["catalog_hashes"]
    assert manifest["verification"]["pytest"]["status"] == "passed"
    assert verify_manifest_hash(manifest)


def test_build_manifest_indexes_historical_runs_without_patient_content(tmp_path: Path) -> None:
    run = tmp_path / "outputs" / "test" / "test_001"
    run.mkdir(parents=True)
    (run / "final_results.jsonl").write_text(
        '{"patient_id":"Patient_1"}\n',
        encoding="utf-8",
    )

    manifest = build_manifest(tmp_path, verification={})

    assert manifest["historical_runs"][0]["run_id"] == "test_001"
    assert "Patient_1" not in json.dumps(manifest, ensure_ascii=False)
    assert manifest["historical_runs"][0]["artifact_hashes"]["final_results.jsonl"]


def test_manifest_marks_failed_verification_as_not_releasable(tmp_path: Path) -> None:
    manifest = build_manifest(
        tmp_path,
        verification={"pytest": {"status": "failed", "summary": "1 failed"}},
    )

    assert manifest["baseline_gate"]["passed"] is False
    assert "pytest" in manifest["baseline_gate"]["failure_reasons"]


def test_manifest_requires_all_baseline_verification_results(tmp_path: Path) -> None:
    manifest = build_manifest(
        tmp_path,
        verification={"pytest": {"status": "passed", "summary": "1 passed"}},
    )

    assert manifest["baseline_gate"]["passed"] is False
    assert manifest["baseline_gate"]["failure_reasons"] == [
        "compileall",
        "json",
        "leakage_scan",
        "dependency_lock",
        "base_image_lock",
    ]


def test_deletion_ledger_entries_have_valid_lifecycle() -> None:
    path = Path("docs/架构迁移基线/legacy_deletion_ledger.json")
    ledger = json.loads(path.read_text(encoding="utf-8"))

    assert ledger["schema_version"] == "legacy-deletion-ledger/v1"
    assert ledger["entries"]
    for entry in ledger["entries"]:
        assert entry["owner_stage"] in {"2", "3", "4", "5", "6", "7A", "7B", "7C"}
        assert entry["status"] in {"pending", "shadowed", "deleted"}
        assert entry["legacy_symbols"]
        assert entry["replacement"]
        if entry["status"] == "shadowed":
            assert entry["shadowed_in_stage"] == entry["owner_stage"]
            assert entry["shadow_evidence"]
        if entry["status"] == "deleted":
            expected_stage = entry.get("final_deletion_stage", entry["owner_stage"])
            assert entry["deleted_in_stage"] == expected_stage
            assert entry["deletion_evidence"]


def test_stage_two_runtime_boundary_entries_are_deleted() -> None:
    ledger = json.loads(
        Path("docs/架构迁移基线/legacy_deletion_ledger.json").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in ledger["entries"]}

    assert entries["direct-sdk-actions"]["status"] == "deleted"
    assert entries["inline-evaluation"]["status"] == "deleted"
    assert entries["external-batch-evaluation-runner"]["status"] == "deleted"


def test_stage_three_dict_state_is_shadowed_not_deleted() -> None:
    # Historical name retained: after 7C cutover dict-case-state is fully deleted.
    ledger = json.loads(
        Path("docs/架构迁移基线/legacy_deletion_ledger.json").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in ledger["entries"]}
    entry = entries["dict-case-state"]

    assert entry["status"] == "deleted"
    assert entry["shadowed_in_stage"] == "3"
    assert entry["final_deletion_stage"] == "7C"
    assert entry["deleted_in_stage"] == "7C"


def test_closed_stage_ledger_entries_have_substitutes() -> None:
    ledger = json.loads(
        Path("docs/架构迁移基线/legacy_deletion_ledger.json").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in ledger["entries"]}
    for entry_id in (
        "legacy-intake-control",
        "legacy-diagnosis-exam-loop",
        "legacy-treatment-patch-chain",
        "markdown-runtime-memory",
        "round-named-regression-files",
    ):
        entry = entries[entry_id]
        assert entry["status"] == "deleted"
        assert entry["replacement"]
        assert entry["deletion_evidence"]


def test_check_manifest_detects_snapshot_change(tmp_path: Path) -> None:
    (tmp_path / "agent").mkdir()
    source = tmp_path / "agent" / "agent.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        build_manifest(
            tmp_path,
            verification={"pytest": {"status": "passed", "summary": "1 passed"}},
        ),
    )

    assert check_manifest(tmp_path, manifest_path)["snapshot_hash"] is True

    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert check_manifest(tmp_path, manifest_path)["snapshot_hash"] is False


def test_check_manifest_recomputes_gate_semantics(tmp_path: Path) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    verification = {
        name: {"status": "passed"}
        for name in (
            "pytest",
            "compileall",
            "json",
            "leakage_scan",
            "dependency_lock",
            "base_image_lock",
        )
    }
    manifest = build_manifest(tmp_path, verification=verification)
    manifest["verification"]["pytest"]["status"] = "failed"
    manifest["baseline_gate"] = {"passed": True, "failure_reasons": []}
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = canonical_hash(payload)
    write_manifest(manifest_path, manifest)

    assert check_manifest(tmp_path, manifest_path)["baseline_gate"] is False


def test_validate_json_files_reports_invalid_json(tmp_path: Path) -> None:
    for relative_path in REQUIRED_JSON_PATHS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "agent" / "knowledge" / "alias_map.json").write_text(
        "{",
        encoding="utf-8",
    )

    result = validate_json_files(tmp_path)

    assert result["status"] == "failed"
    assert result["missing_paths"] == []
    assert result["invalid_paths"] == ["agent/knowledge/alias_map.json"]


def test_validate_json_files_fails_when_required_files_are_missing(tmp_path: Path) -> None:
    result = validate_json_files(tmp_path)

    assert result["status"] == "failed"
    assert result["checked_count"] == 0
    assert result["missing_paths"] == [
        "agent/knowledge/alias_map.json",
        "agent/knowledge/diagnosis_exam_profiles.json",
        "agent/knowledge/exam_intent_map.json",
        "agent/knowledge/treatment_safety_profiles.json",
        "data/ref_data/departments.json",
        "data/ref_data/diseases_catalog.json",
        "data/ref_data/examinations_catalog.json",
    ]


def test_current_dependency_lock_is_complete_and_hashed() -> None:
    result = validate_dependency_lock(Path.cwd())

    assert result["status"] == "passed"
    assert result["missing_packages"] == []
    assert result["unhashed_packages"] == []


def test_base_image_lock_matches_official_dockerfile() -> None:
    result = validate_base_image_lock(Path.cwd())

    assert result["status"] == "passed"
    assert result["digest"].startswith("sha256:")


def test_git_state_preserves_unicode_and_excludes_manifest_itself(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "baseline@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Baseline Test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "README.md").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True)
    manifest_path = tmp_path / "docs" / "架构迁移基线" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "docs" / "中文证据.txt").write_text("evidence\n", encoding="utf-8")

    manifest = build_manifest(tmp_path, verification={})

    assert "docs/中文证据.txt" in manifest["git"]["dirty_paths"]
    assert "docs/架构迁移基线/manifest.json" not in manifest["git"]["dirty_paths"]


def test_run_command_removes_nondeterministic_pytest_duration(tmp_path: Path) -> None:
    result = _run_command(
        [
            sys.executable,
            "-c",
            "print('304 passed, 95 subtests passed in 6.01s')",
        ],
        root=tmp_path,
    )

    assert result["summary"] == "304 passed, 95 subtests passed"


def test_check_manifest_detects_new_historical_run(tmp_path: Path) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    first_run = tmp_path / "outputs" / "test" / "test_001"
    first_run.mkdir(parents=True)
    (first_run / "final_results.jsonl").write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        build_manifest(tmp_path, verification={}),
    )
    second_run = tmp_path / "outputs" / "test" / "test_002"
    second_run.mkdir(parents=True)
    (second_run / "final_results.jsonl").write_text("{}\n", encoding="utf-8")

    result = check_manifest(tmp_path, manifest_path)

    assert result["historical_runs"] is False


def test_deletion_ledger_covers_scanned_legacy_symbols() -> None:
    ledger = json.loads(
        Path("docs/架构迁移基线/legacy_deletion_ledger.json").read_text(encoding="utf-8")
    )
    symbols = {
        symbol
        for entry in ledger["entries"]
        for symbol in entry["legacy_symbols"]
    }
    round_named_test_files = {
        path.as_posix()
        for path in Path("tests").glob("test_*round*.py")
    }

    assert "test.py:evaluate_test_output:actions.batch_evaluation" in symbols
    assert "append_eleventh_round_facts" in symbols
    assert round_named_test_files <= symbols
    scan = scan_python_source(
        [*Path("agent").rglob("*.py"), *Path("tests").rglob("*.py")],
        root=Path.cwd(),
    )
    for finding in scan["experiment_round_names"]:
        path = finding["path"]
        value = finding["value"]
        if str(path).startswith("agent/"):
            assert value in symbols
            continue
        assert path in symbols or f"{path}:test_*_round_*" in symbols


def release_manifest_fixture() -> dict[str, object]:
    return {
        "git": {"head": "a" * 40},
        "snapshot": {
            "sha256": "b" * 64,
            "files": [
                {"path": "requirements.txt", "sha256": "c" * 64},
                {"path": "Dockerfile", "sha256": "d" * 64},
            ],
        },
        "base_image": {
            "repository": "registry.example/python",
            "tag": "3.9",
            "digest": "sha256:" + "f" * 64,
            "media_type": "application/vnd.docker.distribution.manifest.v2+json",
        },
        "locked_dockerfile_hash": "1" * 64,
    }


def test_build_release_identity_is_self_verifying() -> None:
    manifest = release_manifest_fixture()

    identity = build_release_identity(manifest, artifact_hash="e" * 64)

    assert identity["release_name"] == "release_R_stable"
    assert identity["code_commit"] == "a" * 40
    assert identity["dependency_lock_hash"] == "c" * 64
    assert identity["dockerfile_hash"] == "d" * 64
    assert identity["release_pack_hash"] != "b" * 64
    assert identity["artifact_or_image_hash"] == "e" * 64
    assert identity["base_image_digest"] == "sha256:" + "f" * 64
    assert identity["locked_dockerfile_hash"] == "1" * 64
    assert identity["artifact_kind"] == "docker-build-context"
    assert verify_release_identity(identity)
    assert release_identity_matches_manifest(identity, manifest, artifact_hash="e" * 64)


def test_release_identity_cross_checks_manifest_hashes() -> None:
    manifest = release_manifest_fixture()
    identity = build_release_identity(manifest, artifact_hash="e" * 64)
    identity["dependency_lock_hash"] = "e" * 64
    payload = {key: value for key, value in identity.items() if key != "identity_hash"}
    identity["identity_hash"] = canonical_hash(payload)

    assert verify_release_identity(identity)
    assert not release_identity_matches_manifest(
        identity,
        manifest,
        artifact_hash="e" * 64,
    )


def test_release_identity_rejects_restore_strategy_tampering() -> None:
    manifest = release_manifest_fixture()
    identity = build_release_identity(manifest, artifact_hash="e" * 64)
    identity["restore_strategy"] = "run arbitrary command"
    payload = {key: value for key, value in identity.items() if key != "identity_hash"}
    identity["identity_hash"] = canonical_hash(payload)

    assert verify_release_identity(identity)
    assert not release_identity_matches_manifest(
        identity,
        manifest,
        artifact_hash="e" * 64,
    )


def test_release_identity_rejects_unexpected_fields() -> None:
    manifest = release_manifest_fixture()
    identity = build_release_identity(manifest, artifact_hash="e" * 64)
    identity["unexpected_field"] = "not allowed"
    payload = {key: value for key, value in identity.items() if key != "identity_hash"}
    identity["identity_hash"] = canonical_hash(payload)

    assert verify_release_identity(identity)
    assert not release_identity_matches_manifest(
        identity,
        manifest,
        artifact_hash="e" * 64,
    )


def test_deployment_bundle_is_deterministic_and_contains_commit_files(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "baseline@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Baseline Test"],
        cwd=tmp_path,
        check=True,
    )
    files = {
        ".dockerignore": ".git/\n",
        "Dockerfile": "FROM python:3.9\nCOPY . /app\n",
        "requirements.txt": "example==1.0\n",
        "agent/agent.py": "VALUE = 1\n",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    snapshot_files = collect_snapshot_files(tmp_path)
    first_bundle = tmp_path / "first.zip"
    second_bundle = tmp_path / "second.zip"
    base_image = {
        "repository": "python",
        "tag": "3.9",
        "digest": "sha256:" + "f" * 64,
        "media_type": "application/vnd.docker.distribution.manifest.v2+json",
    }

    build_deployment_bundle(
        tmp_path,
        commit,
        snapshot_files,
        first_bundle,
        base_image=base_image,
    )
    build_deployment_bundle(
        tmp_path,
        commit,
        snapshot_files,
        second_bundle,
        base_image=base_image,
    )

    assert sha256_file(first_bundle) == sha256_file(second_bundle)
    with zipfile.ZipFile(first_bundle) as bundle:
        assert bundle.namelist() == sorted([*files, "Dockerfile.locked"])
        assert bundle.read("agent/agent.py") == b"VALUE = 1\n"
        assert bundle.read("Dockerfile.locked").startswith(
            b"FROM python@sha256:"
        )


def test_require_snapshot_matches_head_rejects_dirty_source(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "baseline@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Baseline Test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "agent").mkdir()
    source = tmp_path / "agent" / "agent.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "agent/agent.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    manifest = build_manifest(tmp_path, verification={})

    with pytest.raises(RuntimeError, match="snapshot does not match git head"):
        require_snapshot_matches_head(tmp_path, manifest)


def test_snapshot_matches_commit_detects_content_mismatch(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "baseline@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Baseline Test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "agent").mkdir()
    source = tmp_path / "agent" / "agent.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "agent/agent.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    matching_files = collect_snapshot_files(tmp_path)

    assert snapshot_matches_commit(tmp_path, commit, matching_files)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    mismatched_files = collect_snapshot_files(tmp_path)

    assert not snapshot_matches_commit(tmp_path, commit, mismatched_files)
