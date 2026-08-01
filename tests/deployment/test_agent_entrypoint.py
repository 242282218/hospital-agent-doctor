from __future__ import annotations

import ast
from pathlib import Path

import agent.agent as entrypoint
from agent.legacy_orchestrator import MyDoctorAgent
from agent.knowledge.typed_rule_engine import empty_compiled_rule_pack
from scripts.architecture.freeze_baseline import scan_python_source


def test_build_agent_wires_config_and_memory(monkeypatch, tmp_path: Path) -> None:
    config = {"log_llm_prompts": False}
    memory = object()
    monkeypatch.setattr(entrypoint, "load_config", lambda path: config)
    monkeypatch.setattr(entrypoint, "build_memory", lambda value, **_: memory)
    # Isolate from the live release pointer so this composition test stays release-free.
    monkeypatch.setattr(entrypoint, "load_release_if_present", lambda pointer: None)

    agent = entrypoint.build_agent(tmp_path / "config.yaml")

    assert isinstance(agent, MyDoctorAgent)
    assert agent.config is config
    assert agent.memory is memory
    assert agent.rule_pack == empty_compiled_rule_pack()


def test_build_agent_injects_release_rule_pack_without_copying_rules_to_config(
    monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    config = {"log_llm_prompts": False}
    rule_pack = empty_compiled_rule_pack()
    release = SimpleNamespace(
        pointer={"pack_hash": "a" * 64},
        manifest={"schema_version": "clinical-runtime/v1"},
        policy_pack={},
        prompt_pack={},
        registry={"assets": []},
        knowledge_rule_pack=rule_pack,
    )
    monkeypatch.setattr(entrypoint, "load_config", lambda path: config)
    monkeypatch.setattr(entrypoint, "build_memory", lambda value, **_: object())
    monkeypatch.setattr(entrypoint, "load_release_if_present", lambda pointer: release)

    agent = entrypoint.build_agent(tmp_path / "config.yaml")

    assert agent.rule_pack is rule_pack
    assert config["release_pack"]["typed_rules_active"] is False
    assert config["release_pack"]["typed_rule_count"] == 0
    assert "rules" not in config["release_pack"]


def test_build_agent_uses_explicit_release_pointer(monkeypatch, tmp_path: Path) -> None:
    config = {"log_llm_prompts": False}
    seen = []
    pointer = tmp_path / "candidate-pointer.json"

    monkeypatch.setattr(entrypoint, "load_config", lambda path: config)
    monkeypatch.setattr(entrypoint, "build_memory", lambda value, **_: object())
    monkeypatch.setattr(
        entrypoint,
        "load_release_if_present",
        lambda value: seen.append(value) or None,
    )

    entrypoint.build_agent(tmp_path / "config.yaml", release_pointer=pointer)

    assert seen == [pointer]


def test_main_starts_sdk_builder(monkeypatch) -> None:
    sentinel_agent = object()
    calls = []

    class FakeBuilder:
        def __init__(self, agent: object) -> None:
            calls.append(("init", agent))

        def start(self) -> None:
            calls.append(("start",))

    monkeypatch.setattr(entrypoint, "build_agent", lambda: sentinel_agent)
    monkeypatch.setattr(entrypoint, "AgentBuilder", FakeBuilder)

    entrypoint.main()

    assert calls == [("init", sentinel_agent), ("start",)]


def test_agent_module_is_thin_composition_root() -> None:
    path = Path("agent/agent.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert len(source.splitlines()) < 150
    assert not [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert "select_diagnosis_axes" not in source
    assert "final_verifier" not in source


def test_sdk_action_methods_exist_only_in_sdk_adapter() -> None:
    scan = scan_python_source(Path("agent").rglob("*.py"), root=Path.cwd())
    findings = scan["direct_action_calls"]

    assert {item["path"] for item in findings} == {"agent/runtime/sdk_adapter.py"}
    assert {item["action"] for item in findings} == {
        "ask_patient",
        "order_examination",
        "prescribe_treatment",
        "evaluation",
    }


def test_tests_no_longer_import_clinical_helpers_from_entrypoint() -> None:
    offenders = []
    for path in Path("tests").glob("test_*.py"):
        if "from agent.agent import" in path.read_text(encoding="utf-8"):
            offenders.append(path.as_posix())
    assert offenders == []
