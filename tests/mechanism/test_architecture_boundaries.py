from __future__ import annotations

import ast
from pathlib import Path

from scripts.architecture.freeze_baseline import scan_python_source


def test_agent_does_not_import_offline() -> None:
    offenders = []
    for path in Path("agent").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "offline" or alias.name.startswith("offline."):
                        offenders.append(path.as_posix())
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "offline" or node.module.startswith("offline."):
                    offenders.append(path.as_posix())
    assert offenders == []


def test_sdk_actions_only_in_adapter() -> None:
    scan = scan_python_source(Path("agent").rglob("*.py"), root=Path.cwd())
    assert {item["path"] for item in scan["direct_action_calls"]} == {
        "agent/runtime/sdk_adapter.py"
    }


def test_entrypoint_line_count_under_150() -> None:
    lines = Path("agent/agent.py").read_text(encoding="utf-8").splitlines()
    assert len(lines) < 150


def test_no_patient_id_literals_in_agent() -> None:
    scan = scan_python_source(Path("agent").rglob("*.py"), root=Path.cwd())
    assert scan["patient_id_literals"] == []


def test_clinical_policy_modules_have_no_sdk_io_or_llm_dependencies() -> None:
    """S8 boundary: pure policy modules stay free of SDK/I/O/LLM side effects."""
    pure_modules = [
        Path("agent/clinical/exam_rule_closure.py"),
        Path("agent/clinical/treatment_review_policy.py"),
        Path("agent/clinical/exam_budget_policy.py"),
        Path("agent/clinical/legacy_hypotheses.py"),
        Path("agent/observability/runtime_events.py"),
    ]
    forbidden_import_prefixes = (
        "hospital_agent_sdk",
        "agent.runtime.sdk_adapter",
    )
    forbidden_literals = (
        "releases/current.json",
        "_call_llm",
        "outputs/",
    )
    for path in pure_modules:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(
                        alias.name == prefix or alias.name.startswith(prefix + ".")
                        for prefix in forbidden_import_prefixes
                    ), path.as_posix()
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(
                    node.module == prefix or node.module.startswith(prefix + ".")
                    for prefix in forbidden_import_prefixes
                ), path.as_posix()
        for literal in forbidden_literals:
            assert literal not in source, "%s contains %s" % (path.as_posix(), literal)


def test_agent_entrypoint_remains_thin_composition_root() -> None:
    path = Path("agent/agent.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert len(source.splitlines()) < 150
    assert not [node for node in tree.body if isinstance(node, ast.ClassDef)]
    # No clinical helpers in composition root.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert node.name in {
                "load_release_if_present",
                "build_agent",
                "main",
            }
