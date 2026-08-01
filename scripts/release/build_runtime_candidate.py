"""Build an immutable strict runtime candidate without switching current."""

from __future__ import annotations

import argparse
from hashlib import sha256
import sys
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from agent.clinical.authority_policy import parse_clinical_authority_policy
from agent.prompt import REQUIRED_RUNTIME_PROMPT_KEYS
from agent.runtime.release_loader import load_current_release
from offline.artifacts import content_hash, file_hash, read_json, write_immutable_json
from offline.release import build_candidate_pack


def _runtime_code_hashes() -> Dict[str, str]:
    """Freeze every importable runtime module instead of a brittle hand-picked list."""
    hashes: Dict[str, str] = {}
    for path in sorted((BASE_DIR / "agent").rglob("*.py")):
        relative = path.relative_to(BASE_DIR).as_posix()
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        hashes[relative] = sha256(data).hexdigest()
    if not hashes:
        raise ValueError("runtime code tree is empty")
    return hashes


def _runtime_prompt_pack() -> Dict[str, str]:
    import agent.prompt as prompt

    pack = {}
    for key in REQUIRED_RUNTIME_PROMPT_KEYS:
        value = getattr(prompt, key, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("runtime prompt unavailable: %s" % key)
        pack[key] = value
    return pack


def _runtime_knowledge_hashes() -> Dict[str, str]:
    knowledge_dir = BASE_DIR / "agent" / "knowledge"
    hashes = {
        path.name: file_hash(path)
        for path in sorted(knowledge_dir.glob("*.json"))
    }
    if "exam_axis_evidence_contract.json" not in hashes:
        raise ValueError("runtime exam axis evidence contract is missing")
    return hashes


def _portable_release_dir(target: Path) -> str:
    resolved = target.resolve()
    try:
        return resolved.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _pointer_release_dir(target: Path, pointer_path: Path) -> str:
    resolved = target.resolve()
    pointer_parent = Path(pointer_path).resolve().parent
    if pointer_parent.name == "releases":
        try:
            relative = resolved.relative_to(pointer_parent.parent)
        except ValueError as exc:
            raise ValueError("production runtime candidate must live under releases") from exc
        return relative.as_posix()
    try:
        return resolved.relative_to(pointer_parent).as_posix()
    except ValueError:
        return str(resolved)


def _load_source_release(pointer: Path):
    try:
        return load_current_release(pointer, require_runtime_code_pins=False)
    except TypeError as exc:
        if "require_runtime_code_pins" not in str(exc):
            raise
        return load_current_release(pointer)


def build_candidate(*, source_pointer: Path, output_root: Path, name: str, authority: str) -> dict:
    policy = parse_clinical_authority_policy(authority)
    if policy.values() != ("legacy", "legacy", "legacy", "legacy"):
        raise ValueError("A6 runtime only supports legacy authority policy")
    source = _load_source_release(source_pointer)
    code_files = _runtime_code_hashes()
    seed = content_hash(
        {
            "source_pack_hash": source.pointer.get("pack_hash"),
            "runtime_code_hash": content_hash(code_files),
            "authority_policy_hash": policy.identity_hash,
        }
    )
    target = output_root / ("%s-%s" % (name, seed[:12]))
    stable_pointer = output_root / ("%s-pointer.json" % name)
    if target.exists() or stable_pointer.exists():
        raise FileExistsError("refusing to overwrite runtime candidate")
    manifest = build_candidate_pack(
        release_dir=target,
        code_commit="",
        code_tree_hash=content_hash(code_files),
        prompt_pack=_runtime_prompt_pack(),
        policy_pack=dict(source.policy_pack),
        registry=dict(source.registry),
        knowledge_hashes=_runtime_knowledge_hashes(),
        catalog_hashes=dict(source.manifest.get("catalog_hashes") or {}),
        control_report_hashes=dict(source.manifest.get("control_report_hashes") or {}),
        knowledge_rule_pack=read_json(
            BASE_DIR / "agent" / "knowledge" / "clinical_pattern_rules.json"
        ),
        runtime_code_files=code_files,
        authority_policy=authority,
        authority_policy_hash=policy.identity_hash,
        schema_version="clinical-runtime/v2",
    )
    pointer = {
        "schema_version": "release-pointer/v1",
        "release_dir": _pointer_release_dir(target, stable_pointer),
        "pack_hash": file_hash(target / "release_manifest.json"),
        "runtime_schema_version": manifest["schema_version"],
    }
    write_immutable_json(stable_pointer, pointer)
    return {
        "release_dir": str(target),
        "pointer": str(stable_pointer),
        "pack_hash": pointer["pack_hash"],
        "runtime_code_hash": manifest["runtime_code_hash"],
        "prompt_pack_hash": manifest["prompt_pack_hash"],
        "authority_policy_hash": policy.identity_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pointer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--authority", required=True)
    args = parser.parse_args()
    report = build_candidate(
        source_pointer=args.source_pointer,
        output_root=args.output_root,
        name=args.name,
        authority=args.authority,
    )
    import json

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
