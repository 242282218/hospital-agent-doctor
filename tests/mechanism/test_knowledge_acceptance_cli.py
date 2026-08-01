from __future__ import annotations

import errno
import importlib
import json
import shutil
import socket
import subprocess
import sys
import urllib.request
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from offline.artifacts import canonical_json, content_hash
from scripts.knowledge.build_knowledge_candidates import (
    EXPECTED_RULES,
    build_knowledge_candidate_batch,
)


ARTIFACT_FILES = {
    "knowledge_rules.json",
    "knowledge_rule_controls.json",
    "offline_knowledge_acceptance.json",
    "acceptance_manifest.json",
}


def _cli() -> ModuleType:
    return importlib.import_module(
        "scripts.knowledge.compile_knowledge_candidates"
    )


def _write_project_inputs(project_root: Path) -> tuple[list[Path], list[Path], Path]:
    source_paths = [
        project_root / "docs" / "design.md",
        project_root / "docs" / "offline-question.md",
    ]
    test_paths = [
        project_root / "tests" / "test_diagnosis.py",
        project_root / "tests" / "test_treatment.py",
    ]
    for index, path in enumerate(source_paths + test_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local evidence %d\n" % index, encoding="utf-8")

    registry_path = project_root / "agent" / "knowledge" / "exam_intent_map.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "exam_intent_congenital_infection_organ_involvement",
                        "status": "verified",
                    }
                ]
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    catalog_path = project_root / "data" / "ref_data" / "diseases_catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "diseases": {
                    "心内科": ["原发性高血压"],
                    "耳鼻喉科": ["耳鸣"],
                    "骨科": ["跟骨骨折"],
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source_paths, test_paths, catalog_path


@pytest.fixture()
def run_inputs(tmp_path: Path) -> dict[str, Any]:
    project_root = tmp_path / "project"
    source_paths, test_paths, catalog_path = _write_project_inputs(project_root)
    candidate_result = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=source_paths,
        test_files=test_paths,
        artifact_root=project_root / "candidate-batches",
    )
    return {
        "project_root": project_root,
        "candidate_batch": Path(candidate_result["batch_dir"]),
        "catalog": catalog_path,
        "source_paths": source_paths,
        "test_paths": test_paths,
        "artifact_root": tmp_path / "acceptance-runs",
    }


def _build(run_inputs: dict[str, Any]) -> dict[str, Any]:
    return _cli().build_knowledge_acceptance_run(
        run_inputs["project_root"],
        run_inputs["candidate_batch"],
        run_inputs["catalog"],
    )


def _byte_tree(root: Path) -> dict[str, tuple[str, bytes]]:
    result: dict[str, tuple[str, bytes]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[relative] = ("symlink", str(path.readlink()).encode("utf-8"))
        elif path.is_dir():
            result[relative] = ("directory", b"")
        elif path.is_file():
            result[relative] = ("file", path.read_bytes())
        else:
            result[relative] = ("other", b"")
    return result


def _read_artifacts(path: Path) -> dict[str, Any]:
    return {
        name: json.loads((path / name).read_text(encoding="utf-8"))
        for name in ARTIFACT_FILES
    }


def _bundle_artifact_bytes(bundle: dict[str, Any]) -> dict[str, bytes]:
    return {
        "knowledge_rules.json": canonical_json(bundle["knowledge_rules"]).encode("utf-8"),
        "knowledge_rule_controls.json": canonical_json(
            bundle["knowledge_rule_controls"]
        ).encode("utf-8"),
        "offline_knowledge_acceptance.json": canonical_json(
            bundle["offline_knowledge_acceptance"]
        ).encode("utf-8"),
        "acceptance_manifest.json": canonical_json(
            bundle["acceptance_manifest"]
        ).encode("utf-8"),
    }


def _write_canonical_json(path: Path, payload: Any) -> None:
    path.write_text(canonical_json(payload), encoding="utf-8")


def _rehash_manifest(bundle: dict[str, Any]) -> None:
    manifest = bundle["acceptance_manifest"]
    manifest["run_id"] = content_hash(
        {key: value for key, value in manifest.items() if key != "run_id"}
    )


def test_module_exposes_the_three_offline_run_entrypoints() -> None:
    module = _cli()

    assert callable(module.build_knowledge_acceptance_run)
    assert callable(module.write_knowledge_acceptance_run)
    assert callable(module.main)


def test_build_uses_the_real_six_candidate_acceptance_path(
    run_inputs: dict[str, Any],
) -> None:
    bundle = _build(run_inputs)

    acceptance = bundle["offline_knowledge_acceptance"]
    controls = bundle["knowledge_rule_controls"]
    manifest = bundle["acceptance_manifest"]
    assert acceptance["status"] == "active_scope_passed"
    assert acceptance["release_gate_passed"] is False
    assert acceptance["scope"] == {
        "active_rule_ids": [
            "congenital_infection_differential",
            "symptom_over_background_condition",
        ],
        "evaluated_active_rule_ids": [
            "congenital_infection_differential",
            "symptom_over_background_condition",
        ],
        "audit_only_rule_ids": [
            "congenital_infection_organ_closure",
            "hfref_phase_ordered_treatment",
            "negative_culture_antibiotic_stewardship",
            "qt_tricyclic_structural_heart_risk",
        ],
        "unevaluated_rule_ids": [
            "congenital_infection_organ_closure",
            "hfref_phase_ordered_treatment",
            "negative_culture_antibiotic_stewardship",
            "qt_tricyclic_structural_heart_risk",
        ],
    }
    assert controls["control_count"] == 17
    assert acceptance["metrics"] == {
        "positive_hits": 8,
        "misses": 0,
        "false_positives": 0,
        "exceptions_preserved": 3,
        "exception_failures": 0,
        "idempotency_failures": 0,
        "control_failures": 0,
        "p0_count": 0,
        "treatment_active_rule_count": 0,
        "p0_applicable": False,
        "p0_status": "not_evaluated",
    }
    assert len(manifest["candidate_files"]) == len(EXPECTED_RULES) == 6
    assert [item["path"] for item in manifest["candidate_files"]] == sorted(
        item["path"] for item in manifest["candidate_files"]
    )
    assert manifest["batch_metadata_files"] == [
        {
            "path": "review_checklist.json",
            "sha256": sha256(
                (run_inputs["candidate_batch"] / "review_checklist.json").read_bytes()
            ).hexdigest(),
        },
        {
            "path": "source_receipt.json",
            "sha256": sha256(
                (run_inputs["candidate_batch"] / "source_receipt.json").read_bytes()
            ).hexdigest(),
        },
    ]


def test_build_derives_control_scope_from_compiled_active_rules(
    run_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _cli()
    captured_active_rule_ids: list[tuple[str, ...]] = []
    real_builder = module.build_active_rule_control_set
    real_compiler = module.compile_knowledge_rules

    def compile_with_differential_audit_only(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pack = deepcopy(real_compiler(*args, **kwargs))
        differential = next(
            rule
            for rule in pack["rules"]
            if rule["rule_id"] == "congenital_infection_differential"
        )
        differential["runtime"] = {
            "status": "audit_only",
            "stage": "diagnosis_candidates",
        }
        pack["rules_hash"] = content_hash(pack["rules"])
        return pack

    def capture_builder(
        compiled_rules_hash: str,
        *,
        disease_catalog_hash: str,
        active_rule_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        captured_active_rule_ids.append(tuple(active_rule_ids))
        return real_builder(
            compiled_rules_hash,
            disease_catalog_hash=disease_catalog_hash,
            active_rule_ids=active_rule_ids,
        )

    monkeypatch.setattr(
        module,
        "compile_knowledge_rules",
        compile_with_differential_audit_only,
    )
    monkeypatch.setattr(module, "build_active_rule_control_set", capture_builder)

    bundle = module.build_knowledge_acceptance_run(
        run_inputs["project_root"],
        run_inputs["candidate_batch"],
        run_inputs["catalog"],
    )

    expected = ("symptom_over_background_condition",)
    assert tuple(
        sorted(
            rule["rule_id"]
            for rule in bundle["knowledge_rules"]["rules"]
            if rule["runtime"]["status"] == "active"
        )
    ) == expected
    assert captured_active_rule_ids == [expected]
    assert bundle["offline_knowledge_acceptance"]["scope"]["active_rule_ids"] == list(
        expected
    )


def test_build_path_fails_closed_for_unknown_active_rule_id(
    run_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _cli()
    real_compiler = module.compile_knowledge_rules

    def compile_with_unknown_active_rule(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pack = deepcopy(real_compiler(*args, **kwargs))
        differential = next(
            rule
            for rule in pack["rules"]
            if rule["rule_id"] == "congenital_infection_differential"
        )
        differential["rule_id"] = "unknown_active_rule"
        pack["rules_hash"] = content_hash(pack["rules"])
        return pack

    monkeypatch.setattr(
        module,
        "compile_knowledge_rules",
        compile_with_unknown_active_rule,
    )

    with pytest.raises(ValueError, match="unknown active rule_id"):
        module.build_knowledge_acceptance_run(
            run_inputs["project_root"],
            run_inputs["candidate_batch"],
            run_inputs["catalog"],
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "empty_receipt",
        "checklist_status",
        "receipt_candidate_hashes",
        "checklist_rule_ids",
    ],
)
def test_batch_provenance_rejects_tampered_receipt_or_checklist(
    run_inputs: dict[str, Any],
    tamper: str,
) -> None:
    batch = run_inputs["candidate_batch"]
    receipt_path = batch / "source_receipt.json"
    checklist_path = batch / "review_checklist.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checklist = json.loads(checklist_path.read_text(encoding="utf-8"))
    if tamper == "empty_receipt":
        receipt = {}
    elif tamper == "checklist_status":
        checklist["approval_status"] = "approved"
    elif tamper == "receipt_candidate_hashes":
        receipt["candidate_hashes"].pop(next(iter(receipt["candidate_hashes"])))
    else:
        checklist["rule_ids"] = checklist["rule_ids"][:-1]
    _write_canonical_json(receipt_path, receipt)
    _write_canonical_json(checklist_path, checklist)

    with pytest.raises(ValueError):
        _build(run_inputs)


@pytest.mark.parametrize("extra_kind", ["file", "directory"])
def test_batch_provenance_rejects_extra_root_entries(
    run_inputs: dict[str, Any],
    extra_kind: str,
) -> None:
    extra = run_inputs["candidate_batch"] / "unexpected"
    if extra_kind == "file":
        extra.write_text("unexpected", encoding="utf-8")
    else:
        extra.mkdir()

    with pytest.raises(ValueError, match="batch|root|unexpected"):
        _build(run_inputs)


@pytest.mark.parametrize("filename_tamper", ["wrong_stem", "swapped"])
def test_candidate_filename_must_match_the_loaded_candidate_identity(
    run_inputs: dict[str, Any],
    filename_tamper: str,
) -> None:
    candidates_dir = run_inputs["candidate_batch"] / "candidates"
    paths = sorted(candidates_dir.glob("*.json"))
    if filename_tamper == "wrong_stem":
        paths[0].rename(candidates_dir / "wrong-rule-id.json")
    else:
        temporary = candidates_dir / "temporary.json"
        paths[0].rename(temporary)
        paths[1].rename(paths[0])
        temporary.rename(paths[1])

    with pytest.raises(ValueError, match="filename|candidate_id|rule_id"):
        _build(run_inputs)


def test_batch_directory_name_must_equal_the_authoritative_batch_id(
    run_inputs: dict[str, Any],
) -> None:
    original = run_inputs["candidate_batch"]
    mismatched = original.with_name("0" * 64)
    original.rename(mismatched)
    run_inputs["candidate_batch"] = mismatched

    with pytest.raises(ValueError, match="batch_id"):
        _build(run_inputs)


def test_supersedes_is_bound_into_metadata_and_changes_the_run_id(
    run_inputs: dict[str, Any],
) -> None:
    first = _build(run_inputs)
    second_result = build_knowledge_candidate_batch(
        project_root=run_inputs["project_root"],
        source_files=run_inputs["source_paths"],
        test_files=run_inputs["test_paths"],
        artifact_root=run_inputs["project_root"] / "candidate-batches",
        supersedes_batch_id="f" * 64,
    )
    second_inputs = {
        **run_inputs,
        "candidate_batch": Path(second_result["batch_dir"]),
    }
    second = _build(second_inputs)

    assert first["acceptance_manifest"]["run_id"] != second[
        "acceptance_manifest"
    ]["run_id"]
    assert first["acceptance_manifest"]["batch_metadata_files"] != second[
        "acceptance_manifest"
    ]["batch_metadata_files"]


def test_write_emits_only_four_canonical_files_with_a_complete_hash_chain(
    run_inputs: dict[str, Any],
) -> None:
    module = _cli()
    bundle = _build(run_inputs)

    result = module.write_knowledge_acceptance_run(
        bundle,
        run_inputs["artifact_root"],
    )

    run_path = Path(result["path"])
    assert result == {
        "run_id": bundle["acceptance_manifest"]["run_id"],
        "path": str(run_path),
        "reused": False,
    }
    assert {path.name for path in run_path.iterdir()} == ARTIFACT_FILES
    artifacts = _read_artifacts(run_path)
    for name, payload in artifacts.items():
        assert (run_path / name).read_bytes() == canonical_json(payload).encode("utf-8")

    manifest = artifacts["acceptance_manifest.json"]
    manifest_core = {key: value for key, value in manifest.items() if key != "run_id"}
    assert manifest["schema_version"] == "offline-knowledge-acceptance-run/v1"
    assert manifest["run_id"] == content_hash(manifest_core)
    assert manifest["acceptance_hash"] == artifacts[
        "offline_knowledge_acceptance.json"
    ]["acceptance_hash"]
    assert manifest["input_hashes"] == artifacts[
        "offline_knowledge_acceptance.json"
    ]["input_hashes"]
    assert manifest["acceptance_status"] == "active_scope_passed"
    assert manifest["release_gate_passed"] is False

    for name in ARTIFACT_FILES - {"acceptance_manifest.json"}:
        actual = sha256((run_path / name).read_bytes()).hexdigest()
        assert manifest["artifact_hashes"][name] == actual
    for item in manifest["candidate_files"]:
        candidate_path = run_inputs["candidate_batch"] / Path(item["path"])
        assert item["sha256"] == sha256(candidate_path.read_bytes()).hexdigest()
        assert len(item["candidate_hash"]) == 64
        assert len(item["effect_hash"]) == 64


def test_build_write_and_main_preserve_offline_side_effect_boundaries(
    run_inputs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = run_inputs["project_root"]
    releases_root = project_root / "releases"
    promotions_root = project_root / "outputs" / "offline" / "knowledge" / "promotions"
    decisions_root = project_root / "outputs" / "offline" / "knowledge" / "decisions"
    sentinel_files = {
        releases_root / "current.json": b'{"release":"production-unchanged"}\n',
        releases_root / "existing-release" / "manifest.json": b'{"status":"stable"}\n',
        releases_root / "promotion-records" / "approved.json": b'{"approved":true}\n',
        promotions_root / "pending.json": b'{"promotion":"pending"}\n',
        decisions_root / "approved.json": b'{"decision":"approved"}\n',
    }
    for path, body in sentinel_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    watched_roots = (releases_root, promotions_root, decisions_root)
    before = {str(path): _byte_tree(path) for path in watched_roots}
    pointer_before = (releases_root / "current.json").read_bytes()

    def forbidden_online_action(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("offline acceptance CLI attempted an online action")

    monkeypatch.setattr(socket, "socket", forbidden_online_action)
    monkeypatch.setattr(socket, "create_connection", forbidden_online_action)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_online_action)

    module = _cli()
    bundle = module.build_knowledge_acceptance_run(
        project_root,
        run_inputs["candidate_batch"],
        run_inputs["catalog"],
    )
    assert not run_inputs["artifact_root"].exists()
    written = module.write_knowledge_acceptance_run(
        bundle,
        run_inputs["artifact_root"],
    )
    assert written["reused"] is False
    main_artifact_root = tmp_path / "main-acceptance-runs"
    assert not main_artifact_root.exists()
    module.main(
        [
            "--project-root",
            str(project_root),
            "--candidate-batch",
            str(run_inputs["candidate_batch"]),
            "--disease-catalog",
            str(run_inputs["catalog"]),
            "--artifact-root",
            str(main_artifact_root),
        ]
    )

    main_result = json.loads(capsys.readouterr().out)
    assert main_result == {
        **written,
        "path": str(main_artifact_root / written["run_id"]),
        "reused": False,
    }
    assert (releases_root / "current.json").read_bytes() == pointer_before
    assert {str(path): _byte_tree(path) for path in watched_roots} == before
    for artifact_root in (run_inputs["artifact_root"], main_artifact_root):
        run_paths = list(artifact_root.iterdir())
        assert run_paths == [artifact_root / written["run_id"]]
        assert {path.name for path in run_paths[0].iterdir()} == ARTIFACT_FILES
        for name, payload in _read_artifacts(run_paths[0]).items():
            assert (run_paths[0] / name).read_bytes() == canonical_json(payload).encode(
                "utf-8"
            )


def test_build_and_write_never_change_the_candidate_batch(
    run_inputs: dict[str, Any],
) -> None:
    before = _byte_tree(run_inputs["candidate_batch"])

    bundle = _build(run_inputs)
    after_build = _byte_tree(run_inputs["candidate_batch"])
    _cli().write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])

    assert after_build == before
    assert _byte_tree(run_inputs["candidate_batch"]) == before


def test_same_bundle_reuses_the_exact_existing_run(
    run_inputs: dict[str, Any],
) -> None:
    module = _cli()
    bundle = _build(run_inputs)

    first = module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])
    second = module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])

    assert first["reused"] is False
    assert second == {**first, "reused": True}


def test_atomic_rename_no_replace_preserves_an_existing_empty_target(
    tmp_path: Path,
) -> None:
    module = _cli()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "payload.json").write_text("source", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module._rename_no_replace(source, target)

    assert (source / "payload.json").read_text(encoding="utf-8") == "source"
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_atomic_rename_fails_closed_when_the_platform_has_no_safe_primitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _cli()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    monkeypatch.setattr(module.sys, "platform", "unsupported-platform")

    def forbidden_fallback(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ordinary os.rename must not be used as a fallback")

    monkeypatch.setattr(module.os, "rename", forbidden_fallback)
    with pytest.raises(OSError) as exc_info:
        module._rename_no_replace(source, target)

    assert exc_info.value.errno == errno.ENOTSUP
    assert source.is_dir()
    assert not target.exists()


def test_link_like_helper_detects_windows_reparse_attributes_cross_platform() -> None:
    module = _cli()

    class ReparsePath:
        def is_symlink(self) -> bool:
            return False

        def stat(self, *, follow_symlinks: bool = True) -> Any:
            assert follow_symlinks is False
            return SimpleNamespace(st_file_attributes=0x00000400)

    assert module._is_link_like(ReparsePath()) is True


@pytest.mark.parametrize("reparse_entry", ["target", "artifact"])
def test_existing_run_validation_rejects_any_reparse_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reparse_entry: str,
) -> None:
    module = _cli()
    target = tmp_path / "run"
    target.mkdir()
    artifact = target / "artifact.json"
    artifact.write_bytes(b"{}")
    flagged = target if reparse_entry == "target" else artifact
    real_is_link_like = module._is_link_like

    def injected_reparse(path: Path) -> bool:
        if Path(path) == flagged:
            return True
        return real_is_link_like(path)

    monkeypatch.setattr(module, "_is_link_like", injected_reparse)

    with pytest.raises(FileExistsError, match="ordinary|reparse|link"):
        module._verify_existing(target, {"artifact.json": b"{}"})


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction behavior")
def test_candidate_directory_rejects_a_windows_junction(
    run_inputs: dict[str, Any],
) -> None:
    candidates = run_inputs["candidate_batch"] / "candidates"
    junction_target = run_inputs["project_root"] / "junction-target"
    shutil.copytree(candidates, junction_target)
    shutil.rmtree(candidates)
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(candidates), str(junction_target)],
        capture_output=True,
        text=False,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("mbcs", errors="replace")
        pytest.skip("junction creation is unavailable: %s" % stderr)

    with pytest.raises(ValueError, match="ordinary|reparse|link"):
        _build(run_inputs)


@pytest.mark.parametrize("racing_target", ["empty", "different"])
def test_publish_race_never_overwrites_a_conflicting_target_and_cleans_temp(
    run_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    racing_target: str,
) -> None:
    module = _cli()
    bundle = _build(run_inputs)
    expected = _bundle_artifact_bytes(bundle)
    run_id = bundle["acceptance_manifest"]["run_id"]
    target = run_inputs["artifact_root"] / run_id
    real_rename = module._rename_no_replace

    def create_racing_target(source: Path, destination: Path) -> None:
        destination = Path(destination)
        destination.mkdir()
        if racing_target == "different":
            for name, body in expected.items():
                if name == "knowledge_rules.json":
                    body += b" "
                (destination / name).write_bytes(body)
        real_rename(source, destination)

    monkeypatch.setattr(module, "_rename_no_replace", create_racing_target)

    with pytest.raises(FileExistsError):
        module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])

    assert target.is_dir()
    if racing_target == "empty":
        assert list(target.iterdir()) == []
    else:
        assert {path.name for path in target.iterdir()} == ARTIFACT_FILES
        assert (target / "knowledge_rules.json").read_bytes() == (
            expected["knowledge_rules.json"] + b" "
        )
    assert {path.name for path in run_inputs["artifact_root"].iterdir()} == {run_id}


def test_publish_race_reuses_an_exact_four_file_target_and_cleans_temp(
    run_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _cli()
    bundle = _build(run_inputs)
    expected = _bundle_artifact_bytes(bundle)
    run_id = bundle["acceptance_manifest"]["run_id"]
    target = run_inputs["artifact_root"] / run_id
    real_rename = module._rename_no_replace

    def publish_first(source: Path, destination: Path) -> None:
        destination = Path(destination)
        destination.mkdir()
        for name, body in expected.items():
            (destination / name).write_bytes(body)
        real_rename(source, destination)

    monkeypatch.setattr(module, "_rename_no_replace", publish_first)

    result = module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])

    assert result == {"run_id": run_id, "path": str(target), "reused": True}
    assert {path.name for path in target.iterdir()} == ARTIFACT_FILES
    assert {path.name for path in run_inputs["artifact_root"].iterdir()} == {run_id}


@pytest.mark.parametrize("changed_source", ["source_ref", "candidate"])
def test_post_publish_source_change_rolls_back_the_new_target(
    run_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    changed_source: str,
) -> None:
    module = _cli()
    bundle = _build(run_inputs)
    run_id = bundle["acceptance_manifest"]["run_id"]
    target = run_inputs["artifact_root"] / run_id
    real_publish = module._publish

    def publish_then_change(*args: Any, **kwargs: Any) -> bool:
        reused = real_publish(*args, **kwargs)
        if changed_source == "source_ref":
            run_inputs["source_paths"][0].write_text("changed-after-publish\n", encoding="utf-8")
        else:
            candidate = next(
                (run_inputs["candidate_batch"] / "candidates").glob("*.json")
            )
            candidate.write_bytes(candidate.read_bytes() + b"\n")
        return reused

    monkeypatch.setattr(module, "_publish", publish_then_change)

    with pytest.raises(ValueError):
        module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])

    assert not target.exists()
    assert list(run_inputs["artifact_root"].iterdir()) == []


def test_post_publish_failure_never_deletes_an_exact_reused_target(
    run_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _cli()
    bundle = _build(run_inputs)
    expected = _bundle_artifact_bytes(bundle)
    run_id = bundle["acceptance_manifest"]["run_id"]
    target = run_inputs["artifact_root"] / run_id

    def reuse_then_change(
        temp_path: Path,
        destination: Path,
        expected_bytes: dict[str, bytes],
    ) -> bool:
        del temp_path
        destination.mkdir()
        for name, body in expected_bytes.items():
            (destination / name).write_bytes(body)
        run_inputs["source_paths"][0].write_text("changed-after-reuse\n", encoding="utf-8")
        return True

    monkeypatch.setattr(module, "_publish", reuse_then_change)

    with pytest.raises(ValueError):
        module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])

    assert target.is_dir()
    assert {path.name for path in target.iterdir()} == ARTIFACT_FILES
    assert all((target / name).read_bytes() == body for name, body in expected.items())
    assert {path.name for path in run_inputs["artifact_root"].iterdir()} == {run_id}


@pytest.mark.parametrize("mutation", ["tampered", "missing", "extra"])
def test_existing_run_with_any_byte_or_file_set_difference_is_rejected(
    run_inputs: dict[str, Any],
    mutation: str,
) -> None:
    module = _cli()
    bundle = _build(run_inputs)
    result = module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])
    run_path = Path(result["path"])
    if mutation == "tampered":
        path = run_path / "knowledge_rules.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "missing":
        (run_path / "knowledge_rule_controls.json").unlink()
    else:
        (run_path / "decisions.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])


@pytest.mark.parametrize("nested", [False, True])
def test_artifact_root_cannot_equal_or_live_inside_the_candidate_batch(
    run_inputs: dict[str, Any],
    nested: bool,
) -> None:
    module = _cli()
    bundle = _build(run_inputs)
    before = _byte_tree(run_inputs["candidate_batch"])
    artifact_root = run_inputs["candidate_batch"]
    if nested:
        artifact_root = artifact_root / "acceptance-runs"

    with pytest.raises(ValueError, match="artifact_root"):
        module.write_knowledge_acceptance_run(bundle, artifact_root)

    assert _byte_tree(run_inputs["candidate_batch"]) == before


@pytest.mark.parametrize("invalid_layout", ["empty", "extra", "subdirectory"])
def test_candidate_directory_rejects_empty_extra_and_nested_entries(
    run_inputs: dict[str, Any],
    invalid_layout: str,
) -> None:
    candidates = run_inputs["candidate_batch"] / "candidates"
    if invalid_layout == "empty":
        for path in candidates.iterdir():
            path.unlink()
    elif invalid_layout == "extra":
        (candidates / "README.txt").write_text("not a candidate", encoding="utf-8")
    else:
        nested = candidates / "nested"
        nested.mkdir()
        (nested / "candidate.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="candidates"):
        _build(run_inputs)


@pytest.mark.parametrize("tamper", ["candidate_bytes", "source_ref", "catalog"])
def test_write_revalidates_source_bytes_changed_after_build(
    run_inputs: dict[str, Any],
    tamper: str,
) -> None:
    module = _cli()
    bundle = _build(run_inputs)
    if tamper == "candidate_bytes":
        candidate_path = next(
            (run_inputs["candidate_batch"] / "candidates").glob("*.json")
        )
        candidate_path.write_bytes(candidate_path.read_bytes() + b"\n")
    elif tamper == "source_ref":
        run_inputs["source_paths"][0].write_text("tampered\n", encoding="utf-8")
    else:
        run_inputs["catalog"].write_bytes(run_inputs["catalog"].read_bytes() + b"\n")

    with pytest.raises(ValueError):
        module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])

    target = run_inputs["artifact_root"] / bundle["acceptance_manifest"]["run_id"]
    assert not target.exists()


@pytest.mark.parametrize("tamper", ["run_id", "artifact_hash", "acceptance_hash"])
def test_writer_does_not_trust_bundle_hashes(
    run_inputs: dict[str, Any],
    tamper: str,
) -> None:
    module = _cli()
    bundle = deepcopy(_build(run_inputs))
    if tamper == "run_id":
        bundle["acceptance_manifest"]["run_id"] = "0" * 64
    elif tamper == "artifact_hash":
        bundle["acceptance_manifest"]["artifact_hashes"][
            "knowledge_rules.json"
        ] = "0" * 64
    else:
        bundle["offline_knowledge_acceptance"]["acceptance_hash"] = "0" * 64

    with pytest.raises(ValueError):
        module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])


def test_candidate_batch_tree_snapshot_is_json_round_trip_stable(
    run_inputs: dict[str, Any],
) -> None:
    tree = _build(run_inputs)["_source_state"]["candidate_batch_tree"]

    assert json.loads(canonical_json(tree)) == tree
    assert all(isinstance(entry, dict) for entry in tree.values())


@pytest.mark.parametrize(
    "invalid_schema",
    [
        "top_extra",
        "top_missing",
        "manifest_extra",
        "candidate_item_extra",
        "metadata_item_missing",
        "artifact_hash_extra",
        "source_state_extra",
        "source_state_missing",
        "candidate_bad_sha",
        "candidate_bad_path",
    ],
)
def test_writer_rejects_non_exact_bundle_schema_before_rebuild(
    run_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    invalid_schema: str,
) -> None:
    module = _cli()
    bundle = deepcopy(_build(run_inputs))
    if invalid_schema == "top_extra":
        bundle["unexpected"] = True
    elif invalid_schema == "top_missing":
        bundle.pop("knowledge_rule_controls")
    elif invalid_schema == "manifest_extra":
        bundle["acceptance_manifest"]["unexpected"] = True
        _rehash_manifest(bundle)
    elif invalid_schema == "candidate_item_extra":
        bundle["acceptance_manifest"]["candidate_files"][0]["unexpected"] = True
        _rehash_manifest(bundle)
    elif invalid_schema == "metadata_item_missing":
        bundle["acceptance_manifest"]["batch_metadata_files"][0].pop("sha256")
        _rehash_manifest(bundle)
    elif invalid_schema == "artifact_hash_extra":
        bundle["acceptance_manifest"]["artifact_hashes"]["extra.json"] = "0" * 64
        _rehash_manifest(bundle)
    elif invalid_schema == "source_state_extra":
        bundle["_source_state"]["unexpected"] = True
    elif invalid_schema == "source_state_missing":
        bundle["_source_state"].pop("bound_files")
    elif invalid_schema == "candidate_bad_sha":
        bundle["acceptance_manifest"]["candidate_files"][0]["sha256"] = "A" * 64
        _rehash_manifest(bundle)
    else:
        bundle["acceptance_manifest"]["candidate_files"][0]["path"] = "../escape.json"
        _rehash_manifest(bundle)

    def forbidden_rebuild(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid schema reached source rebuild")

    monkeypatch.setattr(module, "_rebuild_and_compare", forbidden_rebuild)
    with pytest.raises(ValueError):
        module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])


def test_writer_uses_an_entry_snapshot_when_the_caller_mutates_mid_write(
    run_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _cli()
    bundle = _build(run_inputs)
    expected_rules = canonical_json(bundle["knowledge_rules"]).encode("utf-8")
    manifest = deepcopy(bundle["acceptance_manifest"])
    real_rebuild = module._rebuild_and_compare

    def mutate_caller_after_rebuild(*args: Any, **kwargs: Any) -> Any:
        result = real_rebuild(*args, **kwargs)
        bundle["knowledge_rules"]["schema_version"] = "tampered-after-validation"
        return result

    monkeypatch.setattr(module, "_rebuild_and_compare", mutate_caller_after_rebuild)

    result = module.write_knowledge_acceptance_run(bundle, run_inputs["artifact_root"])

    rules_path = Path(result["path"]) / "knowledge_rules.json"
    assert rules_path.read_bytes() == expected_rules
    assert sha256(rules_path.read_bytes()).hexdigest() == manifest["artifact_hashes"][
        "knowledge_rules.json"
    ]


def test_main_prints_one_canonical_json_result(
    run_inputs: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _cli()

    module.main(
        [
            "--project-root",
            str(run_inputs["project_root"]),
            "--candidate-batch",
            str(run_inputs["candidate_batch"]),
            "--disease-catalog",
            str(run_inputs["catalog"]),
            "--artifact-root",
            str(run_inputs["artifact_root"]),
        ]
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert output == canonical_json(payload) + "\n"
    assert payload["reused"] is False
    assert Path(payload["path"]).parent == run_inputs["artifact_root"]


def test_build_rejects_a_batch_outside_the_project_root(
    run_inputs: dict[str, Any],
) -> None:
    smaller_root = run_inputs["project_root"] / "docs"

    with pytest.raises(ValueError, match="candidate_batch"):
        _cli().build_knowledge_acceptance_run(
            smaller_root,
            run_inputs["candidate_batch"],
            run_inputs["catalog"],
        )
