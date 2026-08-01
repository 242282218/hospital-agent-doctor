"""Freeze the current architecture baseline as deterministic JSON evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = "architecture-baseline/v1"
SNAPSHOT_PATTERNS = (
    ".dockerignore",
    "agent/**/*.py",
    "agent/knowledge/*.json",
    "tests/**/*.py",
    "scripts/**/*.py",
    "data/ref_data/*.json",
    "config.yaml",
    "requirements.txt",
    "base_image.lock.json",
    "Dockerfile",
    "*.py",
)
EXCLUDED_PARTS = {".venv", "outputs", "__pycache__", ".pytest_cache"}
TEXT_SNAPSHOT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_SNAPSHOT_NAMES = {
    ".dockerignore",
    "Dockerfile",
    "requirements.txt",
    "base_image.lock.json",
}
ACTION_NAMES = {
    "ask_patient",
    "order_examination",
    "prescribe_treatment",
    "evaluation",
    "batch_evaluation",
}
HISTORICAL_ARTIFACTS = (
    "final_results.jsonl",
    "evaluation_results.jsonl",
    "evaluation_result.json",
    "events.jsonl",
)
PATIENT_ID_PATTERN = re.compile(r"\bPatient_\d+\b")
EXPERIMENT_ROUND_PATTERN = re.compile(
    r"第(?:一|二|三|四|五|六|七|八|九|十|十一|十二|十三|十四|十五)轮"
)
EXPERIMENT_ROUND_IDENTIFIER_PATTERN = re.compile(
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth)_round",
    flags=re.IGNORECASE,
)
DEFAULT_MANIFEST_PATH = Path("docs") / "架构迁移基线" / "manifest.json"
DEFAULT_RELEASE_IDENTITY_PATH = (
    Path("releases") / "release_R_stable" / "release_identity.json"
)
DEFAULT_DEPLOYMENT_BUNDLE_PATH = (
    Path("releases") / "release_R_stable" / "deployment_bundle.zip"
)
RESTORE_STRATEGY = (
    "extract deployment_bundle.zip and run docker build -f Dockerfile.locked ."
)
REQUIRED_JSON_PATHS = (
    "agent/knowledge/alias_map.json",
    "agent/knowledge/diagnosis_exam_profiles.json",
    "agent/knowledge/exam_intent_map.json",
    "agent/knowledge/treatment_safety_profiles.json",
    "data/ref_data/departments.json",
    "data/ref_data/diseases_catalog.json",
    "data/ref_data/examinations_catalog.json",
)
REQUIRED_VERIFICATIONS = (
    "pytest",
    "compileall",
    "json",
    "leakage_scan",
    "dependency_lock",
    "base_image_lock",
)
REQUIRED_RUNTIME_PACKAGES = {
    "blinker",
    "click",
    "colorama",
    "flask",
    "hospital-agent-sdk",
    "itsdangerous",
    "jinja2",
    "markdown2",
    "markupsafe",
    "pyyaml",
    "werkzeug",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_snapshot_files(root: Path) -> list[dict[str, object]]:
    root = root.resolve()
    paths: set[Path] = set()
    for pattern in SNAPSHOT_PATTERNS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return [
        _snapshot_file_record(path, root)
        for path in sorted(paths)
        if not _is_excluded(path.relative_to(root))
    ]


def scan_python_source(paths: Iterable[Path], *, root: Path) -> dict[str, object]:
    direct_action_calls: list[dict[str, object]] = []
    patient_id_literals: list[dict[str, object]] = []
    experiment_round_names: list[dict[str, object]] = []
    functions_over_40_lines: list[dict[str, object]] = []
    files_over_150_lines: list[dict[str, object]] = []
    root = root.resolve()

    for path in sorted({item.resolve() for item in paths}):
        relative_path = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        if len(lines) > 150:
            files_over_150_lines.append({"path": relative_path, "lines": len(lines)})

        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ACTION_NAMES:
                    direct_action_calls.append(
                        {
                            "path": relative_path,
                            "line": node.lineno,
                            "action": node.func.attr,
                        }
                    )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                _append_literal_matches(
                    patient_id_literals,
                    PATIENT_ID_PATTERN,
                    node.value,
                    relative_path,
                    node.lineno,
                )
                _append_literal_matches(
                    experiment_round_names,
                    EXPERIMENT_ROUND_PATTERN,
                    node.value,
                    relative_path,
                    node.lineno,
                )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if EXPERIMENT_ROUND_IDENTIFIER_PATTERN.search(node.name):
                    experiment_round_names.append(
                        {
                            "path": relative_path,
                            "line": node.lineno,
                            "value": node.name,
                        }
                    )
                end_line = int(node.end_lineno or node.lineno)
                line_count = end_line - node.lineno + 1
                if line_count > 40:
                    functions_over_40_lines.append(
                        {
                            "path": relative_path,
                            "line": node.lineno,
                            "name": node.name,
                            "lines": line_count,
                        }
                    )

    return {
        "direct_action_calls": sorted(
            direct_action_calls,
            key=lambda item: (str(item["path"]), int(item["line"])),
        ),
        "patient_id_literals": sorted(
            patient_id_literals,
            key=lambda item: (str(item["path"]), int(item["line"]), str(item["value"])),
        ),
        "experiment_round_names": sorted(
            experiment_round_names,
            key=lambda item: (str(item["path"]), int(item["line"]), str(item["value"])),
        ),
        "functions_over_40_lines": sorted(
            functions_over_40_lines,
            key=lambda item: (-int(item["lines"]), str(item["path"]), int(item["line"])),
        ),
        "files_over_150_lines": sorted(
            files_over_150_lines,
            key=lambda item: (-int(item["lines"]), str(item["path"])),
        ),
    }


def build_manifest(
    root: Path,
    *,
    verification: dict[str, object],
) -> dict[str, object]:
    root = root.resolve()
    files = collect_snapshot_files(root)
    base_image = load_base_image_lock(root)
    locked_dockerfile = b""
    if (root / "Dockerfile").is_file() and base_image:
        try:
            locked_dockerfile = _locked_dockerfile_bytes(
                (root / "Dockerfile").read_bytes(),
                base_image,
            )
        except RuntimeError:
            locked_dockerfile = b""
    python_paths = [
        root / str(item["path"])
        for item in files
        if str(item["path"]).endswith(".py")
    ]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "git": _git_state(root),
        "snapshot": {
            "files": files,
            "sha256": canonical_hash(files),
        },
        "catalog_hashes": _hashes_under(files, "data/ref_data/"),
        "knowledge_hashes": _hashes_under(files, "agent/knowledge/"),
        "base_image": base_image,
        "locked_dockerfile_hash": hashlib.sha256(locked_dockerfile).hexdigest()
        if locked_dockerfile
        else "",
        "historical_runs": collect_historical_runs(root),
        "source_scan": scan_python_source(python_paths, root=root),
        "verification": verification,
        "baseline_gate": _baseline_gate(verification),
    }
    return attach_manifest_hash(payload)


def attach_manifest_hash(payload: dict[str, object]) -> dict[str, object]:
    manifest = dict(payload)
    manifest["manifest_hash"] = canonical_hash(payload)
    return manifest


def verify_manifest_hash(manifest: dict[str, object]) -> bool:
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return manifest.get("manifest_hash") == canonical_hash(payload)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def check_manifest(root: Path, manifest_path: Path) -> dict[str, bool]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_files = collect_snapshot_files(root)
    current_runs = collect_historical_runs(root)
    snapshot = manifest.get("snapshot")
    expected_hash = snapshot.get("sha256") if isinstance(snapshot, dict) else None
    gate = manifest.get("baseline_gate")
    verification = manifest.get("verification")
    expected_gate = _baseline_gate(
        verification if isinstance(verification, dict) else {}
    )
    recorded_runs = manifest.get("historical_runs")
    identity_path = root / DEFAULT_RELEASE_IDENTITY_PATH
    bundle_path = root / DEFAULT_DEPLOYMENT_BUNDLE_PATH
    identity_valid = False
    if identity_path.exists() and bundle_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        git = manifest.get("git")
        artifact_hash = sha256_file(bundle_path)
        identity_valid = (
            release_identity_matches_manifest(
                identity,
                manifest,
                artifact_hash=artifact_hash,
            )
            and isinstance(git, dict)
            and _git_commit_exists(root, str(identity.get("code_commit") or ""))
            and deployment_bundle_matches_snapshot(
                bundle_path,
                snapshot.get("files") if isinstance(snapshot.get("files"), list) else [],
                locked_dockerfile_hash=str(
                    manifest.get("locked_dockerfile_hash") or ""
                ),
            )
            and snapshot_matches_commit(
                root,
                str(identity.get("code_commit") or ""),
                snapshot.get("files") if isinstance(snapshot.get("files"), list) else [],
            )
        )
    return {
        "manifest_hash": verify_manifest_hash(manifest),
        "snapshot_hash": expected_hash == canonical_hash(current_files),
        "historical_runs": canonical_hash(recorded_runs) == canonical_hash(current_runs),
        "baseline_gate": bool(
            isinstance(gate, dict)
            and gate == expected_gate
            and gate.get("passed") is True
        ),
        "release_identity": identity_valid,
    }


def validate_json_files(root: Path) -> dict[str, object]:
    candidates = [root / relative_path for relative_path in REQUIRED_JSON_PATHS]
    missing_paths = [
        path.relative_to(root).as_posix()
        for path in candidates
        if not path.is_file()
    ]
    invalid_paths = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            invalid_paths.append(path.relative_to(root).as_posix())
    return {
        "status": "passed" if not missing_paths and not invalid_paths else "failed",
        "checked_count": len(candidates) - len(missing_paths),
        "missing_paths": missing_paths,
        "invalid_paths": invalid_paths,
    }


def validate_dependency_lock(root: Path) -> dict[str, object]:
    path = root / "requirements.txt"
    if not path.is_file():
        return {
            "status": "failed",
            "missing_packages": sorted(REQUIRED_RUNTIME_PACKAGES),
            "unhashed_packages": [],
        }
    packages: dict[str, bool] = {}
    for block in _requirement_blocks(path.read_text(encoding="utf-8")):
        package = block.split("==", 1)[0].strip().lower().replace("_", "-")
        if not package or package.startswith("--"):
            continue
        packages[package] = "--hash=sha256:" in block
    missing = sorted(REQUIRED_RUNTIME_PACKAGES - packages.keys())
    unhashed = sorted(package for package, hashed in packages.items() if not hashed)
    return {
        "status": "passed" if not missing and not unhashed else "failed",
        "missing_packages": missing,
        "unhashed_packages": unhashed,
        "package_count": len(packages),
    }


def load_base_image_lock(root: Path) -> dict[str, str]:
    path = root / "base_image.lock.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        key: str(value.get(key) or "")
        for key in ("repository", "tag", "digest", "media_type")
    }


def validate_base_image_lock(root: Path) -> dict[str, object]:
    lock = load_base_image_lock(root)
    dockerfile = root / "Dockerfile"
    from_line = ""
    if dockerfile.is_file():
        from_line = next(
            (
                line.strip()
                for line in dockerfile.read_text(encoding="utf-8").splitlines()
                if line.strip().upper().startswith("FROM ")
            ),
            "",
        )
    expected_from = "FROM %s:%s" % (
        lock.get("repository", ""),
        lock.get("tag", ""),
    )
    digest = lock.get("digest", "")
    pinned_from = "FROM %s@%s" % (lock.get("repository", ""), digest)
    passed = bool(
        from_line in {expected_from, pinned_from}
        and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        and lock.get("media_type")
    )
    return {
        "status": "passed" if passed else "failed",
        "dockerfile_from": from_line,
        "expected_from": expected_from,
        "digest": digest,
    }


def build_release_identity(
    manifest: dict[str, object],
    *,
    artifact_hash: str,
) -> dict[str, object]:
    git = manifest.get("git")
    snapshot = manifest.get("snapshot")
    if not isinstance(git, dict) or not isinstance(snapshot, dict):
        raise ValueError("manifest must contain git and snapshot objects")
    base_image = manifest.get("base_image")
    if not isinstance(base_image, dict):
        raise ValueError("manifest must contain base_image object")
    payload: dict[str, object] = {
        "schema_version": "release-identity/v1",
        "release_name": "release_R_stable",
        "code_commit": str(git.get("head") or ""),
        "dependency_lock_hash": _snapshot_hash_for(snapshot, "requirements.txt"),
        "dockerfile_hash": _snapshot_hash_for(snapshot, "Dockerfile"),
        "release_pack_hash": _release_pack_hash(snapshot),
        "artifact_or_image_hash": artifact_hash,
        "artifact_kind": "docker-build-context",
        "base_image_digest": str(base_image.get("digest") or ""),
        "locked_dockerfile_hash": str(
            manifest.get("locked_dockerfile_hash") or ""
        ),
        "runtime_schema_version": "legacy-baseline/v1",
        "restore_strategy": RESTORE_STRATEGY,
    }
    identity = dict(payload)
    identity["identity_hash"] = canonical_hash(payload)
    return identity


def verify_release_identity(identity: dict[str, object]) -> bool:
    payload = {key: value for key, value in identity.items() if key != "identity_hash"}
    return identity.get("identity_hash") == canonical_hash(payload)


def release_identity_matches_manifest(
    identity: dict[str, object],
    manifest: dict[str, object],
    *,
    artifact_hash: str,
) -> bool:
    git = manifest.get("git")
    snapshot = manifest.get("snapshot")
    if not verify_release_identity(identity):
        return False
    if not isinstance(git, dict) or not isinstance(snapshot, dict):
        return False
    base_image = manifest.get("base_image")
    if not isinstance(base_image, dict):
        return False
    expected = {
        "schema_version": "release-identity/v1",
        "release_name": "release_R_stable",
        "code_commit": str(git.get("head") or ""),
        "dependency_lock_hash": _snapshot_hash_for(snapshot, "requirements.txt"),
        "dockerfile_hash": _snapshot_hash_for(snapshot, "Dockerfile"),
        "release_pack_hash": _release_pack_hash(snapshot),
        "artifact_or_image_hash": artifact_hash,
        "artifact_kind": "docker-build-context",
        "base_image_digest": str(base_image.get("digest") or ""),
        "locked_dockerfile_hash": str(
            manifest.get("locked_dockerfile_hash") or ""
        ),
        "runtime_schema_version": "legacy-baseline/v1",
        "restore_strategy": RESTORE_STRATEGY,
    }
    return set(identity) == {*expected, "identity_hash"} and all(
        identity.get(key) == value for key, value in expected.items()
    )


def build_deployment_bundle(
    root: Path,
    commit: str,
    snapshot_files: list[dict[str, object]],
    output_path: Path,
    *,
    base_image: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        str(item.get("path") or ""): _git_blob(
            root,
            commit,
            str(item.get("path") or ""),
        )
        for item in snapshot_files
    }
    entries["Dockerfile.locked"] = _locked_dockerfile_bytes(
        entries.get("Dockerfile", b""),
        base_image,
    )
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as bundle:
        for relative_path in sorted(entries):
            info = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, entries[relative_path])


def deployment_bundle_matches_snapshot(
    bundle_path: Path,
    snapshot_files: list[dict[str, object]],
    *,
    locked_dockerfile_hash: str,
) -> bool:
    expected_names = sorted(
        [*(str(item.get("path") or "") for item in snapshot_files), "Dockerfile.locked"]
    )
    try:
        with zipfile.ZipFile(bundle_path) as bundle:
            if bundle.namelist() != expected_names:
                return False
            for item in snapshot_files:
                relative_path = str(item.get("path") or "")
                content = _normalize_snapshot_bytes(
                    bundle.read(relative_path),
                    relative_path,
                )
                if hashlib.sha256(content).hexdigest() != item.get("sha256"):
                    return False
                if len(content) != item.get("bytes"):
                    return False
            locked_content = bundle.read("Dockerfile.locked")
            if hashlib.sha256(locked_content).hexdigest() != locked_dockerfile_hash:
                return False
    except (OSError, KeyError, zipfile.BadZipFile):
        return False
    return True


def snapshot_matches_commit(
    root: Path,
    commit: str,
    snapshot_files: list[dict[str, object]],
) -> bool:
    if not commit or not snapshot_files:
        return False
    for item in snapshot_files:
        relative_path = str(item.get("path") or "")
        expected_hash = str(item.get("sha256") or "")
        completed = subprocess.run(
            ["git", "show", "%s:%s" % (commit, relative_path)],
            cwd=root,
            capture_output=True,
        )
        if completed.returncode != 0:
            return False
        content = _normalize_snapshot_bytes(completed.stdout, relative_path)
        if hashlib.sha256(content).hexdigest() != expected_hash:
            return False
    return True


def require_snapshot_matches_head(
    root: Path,
    manifest: dict[str, object],
) -> None:
    git = manifest.get("git")
    snapshot = manifest.get("snapshot")
    if not isinstance(git, dict) or not isinstance(snapshot, dict):
        raise RuntimeError("manifest is missing git or snapshot")
    files = snapshot.get("files")
    commit = str(git.get("head") or "")
    if not isinstance(files, list) or not snapshot_matches_commit(root, commit, files):
        raise RuntimeError("snapshot does not match git head")


def collect_historical_runs(root: Path) -> list[dict[str, object]]:
    runs = []
    for mode in ("train", "test"):
        mode_dir = root / "outputs" / mode
        if not mode_dir.exists():
            continue
        for run_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
            artifact_hashes = {
                name: sha256_file(run_dir / name)
                for name in HISTORICAL_ARTIFACTS
                if (run_dir / name).is_file()
            }
            runs.append(
                {
                    "run_id": run_dir.name,
                    "mode": mode,
                    "dataset_layer": "HistoricalReplay",
                    "artifact_hashes": artifact_hashes,
                    "has_evaluation": any(
                        name.startswith("evaluation") for name in artifact_hashes
                    ),
                }
            )
    return runs


def _is_excluded(path: Path) -> bool:
    return bool(EXCLUDED_PARTS.intersection(path.parts))


def _snapshot_file_record(path: Path, root: Path) -> dict[str, object]:
    relative_path = path.relative_to(root).as_posix()
    content = _normalize_snapshot_bytes(path.read_bytes(), relative_path)
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _append_literal_matches(
    target: list[dict[str, object]],
    pattern: re.Pattern[str],
    value: str,
    path: str,
    line: int,
) -> None:
    for match in pattern.finditer(value):
        target.append({"path": path, "line": line, "value": match.group(0)})


def _hashes_under(
    files: list[dict[str, object]],
    prefix: str,
) -> dict[str, str]:
    return {
        str(item["path"]): str(item["sha256"])
        for item in files
        if str(item["path"]).startswith(prefix)
    }


def _snapshot_hash_for(snapshot: dict[str, object], relative_path: str) -> str:
    files = snapshot.get("files")
    if not isinstance(files, list):
        return ""
    for item in files:
        if isinstance(item, dict) and item.get("path") == relative_path:
            return str(item.get("sha256") or "")
    return ""


def _release_pack_hash(snapshot: dict[str, object]) -> str:
    files = snapshot.get("files")
    if not isinstance(files, list):
        return canonical_hash([])
    pack_files = [
        item
        for item in files
        if isinstance(item, dict)
        and (
            item.get("path") in {"config.yaml", "agent/prompt.py"}
            or str(item.get("path") or "").startswith("agent/knowledge/")
            or str(item.get("path") or "").startswith("data/ref_data/")
        )
    ]
    return canonical_hash(sorted(pack_files, key=lambda item: str(item.get("path") or "")))


def _normalize_snapshot_bytes(content: bytes, relative_path: str) -> bytes:
    path = Path(relative_path)
    if path.suffix.lower() not in TEXT_SNAPSHOT_SUFFIXES and path.name not in TEXT_SNAPSHOT_NAMES:
        return content
    return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _requirement_blocks(text: str) -> list[str]:
    blocks = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        current = ("%s %s" % (current, line)).strip()
        if current.endswith("\\"):
            current = current[:-1].strip()
            continue
        blocks.append(current)
        current = ""
    if current:
        blocks.append(current)
    return blocks


def _locked_dockerfile_bytes(
    dockerfile_content: bytes,
    base_image: dict[str, str],
) -> bytes:
    normalized = _normalize_snapshot_bytes(dockerfile_content, "Dockerfile")
    text = normalized.decode("utf-8")
    expected = "FROM %s:%s" % (
        base_image.get("repository", ""),
        base_image.get("tag", ""),
    )
    replacement = "FROM %s@%s" % (
        base_image.get("repository", ""),
        base_image.get("digest", ""),
    )
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == expected:
            lines[index] = replacement
            suffix = "\n" if text.endswith("\n") else ""
            return ("\n".join(lines) + suffix).encode("utf-8")
    raise RuntimeError("Dockerfile FROM does not match base image lock")


def _git_blob(root: Path, commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", "%s:%s" % (commit, relative_path)],
        cwd=root,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("missing git blob %s:%s" % (commit, relative_path))
    return completed.stdout


def _git_state(root: Path) -> dict[str, object]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-c",
                "core.quotePath=false",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "head": "", "dirty_paths": []}
    ignored_dirty_paths = {
        DEFAULT_MANIFEST_PATH.as_posix(),
        DEFAULT_RELEASE_IDENTITY_PATH.as_posix(),
        DEFAULT_DEPLOYMENT_BUNDLE_PATH.as_posix(),
    }
    dirty_paths = sorted(
        path
        for line in status
        if len(line) >= 4
        for path in [line[3:].replace("\\", "/")]
        if path not in ignored_dirty_paths
    )
    return {
        "available": True,
        "head": head,
        "dirty_paths": dirty_paths,
    }


def _baseline_gate(verification: dict[str, object]) -> dict[str, object]:
    failed = [
        name
        for name in REQUIRED_VERIFICATIONS
        for result in [verification.get(name)]
        if not isinstance(result, dict) or result.get("status") != "passed"
    ]
    return {
        "passed": not failed,
        "failure_reasons": failed,
    }


def run_verification(root: Path) -> dict[str, object]:
    python = sys.executable
    pytest_result = _run_command(
        [python, "-m", "pytest", "-q"],
        root=root,
    )
    compile_result = _run_command(
        [python, "-m", "compileall", "-q", "agent", "tests", "scripts"],
        root=root,
    )
    json_result = validate_json_files(root)
    dependency_lock_result = validate_dependency_lock(root)
    base_image_lock_result = validate_base_image_lock(root)
    production_paths = sorted((root / "agent").rglob("*.py"))
    source_scan = scan_python_source(production_paths, root=root)
    leakage_findings = source_scan["patient_id_literals"]
    leakage_result = {
        "status": "passed" if not leakage_findings else "failed",
        "patient_id_literal_count": len(leakage_findings),
        "findings": leakage_findings,
    }
    return {
        "pytest": pytest_result,
        "compileall": compile_result,
        "json": json_result,
        "leakage_scan": leakage_result,
        "dependency_lock": dependency_lock_result,
        "base_image_lock": base_image_lock_result,
    }


def _run_command(command: list[str], *, root: Path) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    non_empty_lines = [line.strip() for line in output.splitlines() if line.strip()]
    summary = non_empty_lines[-1] if non_empty_lines else "exit code 0"
    summary = re.sub(r"\s+in\s+\d+(?:\.\d+)?s$", "", summary)
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "exit_code": completed.returncode,
        "summary": summary,
    }


def _git_commit_exists(root: Path, commit: str) -> bool:
    if not commit:
        return False
    completed = subprocess.run(
        ["git", "cat-file", "-e", "%s^{commit}" % commit],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _print_check_result(result: dict[str, bool]) -> None:
    print("manifest hash: %s" % ("PASS" if result["manifest_hash"] else "FAIL"))
    print("snapshot hash: %s" % ("PASS" if result["snapshot_hash"] else "FAIL"))
    print("historical runs: %s" % ("PASS" if result["historical_runs"] else "FAIL"))
    print("baseline gate: %s" % ("PASS" if result["baseline_gate"] else "FAIL"))
    print("release identity: %s" % ("PASS" if result["release_identity"] else "FAIL"))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check"))
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    manifest_path = root / DEFAULT_MANIFEST_PATH
    if args.command == "generate":
        verification = run_verification(root)
        manifest = build_manifest(root, verification=verification)
        try:
            require_snapshot_matches_head(root, manifest)
        except RuntimeError as exc:
            print("baseline generation refused: %s" % exc)
            return 1
        snapshot = manifest["snapshot"]
        git = manifest["git"]
        if not isinstance(snapshot, dict) or not isinstance(git, dict):
            raise RuntimeError("generated manifest is missing git or snapshot")
        snapshot_files = snapshot.get("files")
        if not isinstance(snapshot_files, list):
            raise RuntimeError("generated manifest snapshot has no files")
        bundle_path = root / DEFAULT_DEPLOYMENT_BUNDLE_PATH
        build_deployment_bundle(
            root,
            str(git.get("head") or ""),
            snapshot_files,
            bundle_path,
            base_image=manifest.get("base_image")
            if isinstance(manifest.get("base_image"), dict)
            else {},
        )
        write_manifest(manifest_path, manifest)
        write_manifest(
            root / DEFAULT_RELEASE_IDENTITY_PATH,
            build_release_identity(
                manifest,
                artifact_hash=sha256_file(bundle_path),
            ),
        )
        print("baseline manifest written: %s" % DEFAULT_MANIFEST_PATH.as_posix())
        gate = manifest["baseline_gate"]
        passed = bool(isinstance(gate, dict) and gate.get("passed") is True)
        print("baseline gate: %s" % ("PASS" if passed else "FAIL"))
        return 0 if passed else 1

    result = check_manifest(root, manifest_path)
    _print_check_result(result)
    return 0 if all(result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
