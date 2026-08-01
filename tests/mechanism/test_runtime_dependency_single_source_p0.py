"""P0: Composition Root and single runtime RulePack source."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

import train as train_mod
from agent.knowledge.typed_rule_engine import CompiledRulePack, empty_compiled_rule_pack
from agent.legacy_orchestrator import (
    MyDoctorAgent,
    load_packaged_rule_pack_for_offline_use,
    select_diagnosis_axes,
    validate_axis_consult,
)


ROOT = Path(__file__).resolve().parents[2]
POINTER = ROOT / "releases" / "current.json"


def test_train_entrypoint_builds_agent_through_composition_root(monkeypatch) -> None:
    calls: List[str] = []

    class SentinelAgent:
        def run_train(self):
            calls.append("run_train")
            return {"status": "ok"}

    monkeypatch.setattr(train_mod, "build_agent", lambda: SentinelAgent(), raising=False)
    train_mod.main()
    assert calls == ["run_train"]


def test_packaged_fallback_does_not_read_current_pointer(monkeypatch, tmp_path: Path) -> None:
    reads: List[Path] = []
    original_read = Path.read_text

    def tracked_read(self: Path, *args: Any, **kwargs: Any) -> str:
        reads.append(Path(self))
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read)
    pack = load_packaged_rule_pack_for_offline_use()
    assert isinstance(pack, CompiledRulePack)
    assert pack.rule_count >= 0
    assert not any(path.resolve() == POINTER.resolve() for path in reads)


def test_empty_pack_keeps_axis_and_closure_empty_together() -> None:
    empty = empty_compiled_rule_pack()
    axes = select_diagnosis_axes(
        {"symptom_clusters": [{"label": "发热", "evidence": "发热三天"}]},
        rule_pack=empty,
    )
    # Hardcoded non-typed axes may still appear; typed engine output must stay empty.
    assert empty.rule_count == 0
    assert all(
        not str(axis.get("source") or "").startswith("typed")
        for axis in axes
    )


def test_diagnostic_axis_consult_uses_agent_rule_pack_object(monkeypatch) -> None:
    sentinel = empty_compiled_rule_pack()
    captured: Dict[str, Any] = {}

    def fake_validate(
        raw_consult: Dict[str, Any],
        *,
        case_state: Dict[str, Any],
        official_diseases,
        alias_rules=None,
        rule_pack: Optional[CompiledRulePack] = None,
    ) -> Dict[str, Any]:
        captured["validate_pack"] = rule_pack
        return {
            "intake_facts": {},
            "diagnosis_axes": [],
            "risk_summary": "",
        }

    def fake_select(
        intake_facts: Dict[str, Any],
        llm_axes=None,
        *,
        case_state=None,
        rule_pack: Optional[CompiledRulePack] = None,
    ):
        captured["select_pack"] = rule_pack
        return []

    agent = MyDoctorAgent(config={}, memory=None, rule_pack=sentinel)
    monkeypatch.setattr(
        "agent.legacy_orchestrator.validate_axis_consult",
        fake_validate,
    )
    monkeypatch.setattr(
        "agent.legacy_orchestrator.select_diagnosis_axes",
        fake_select,
    )

    import asyncio

    async def fake_llm(**kwargs):
        return {"intake_facts": {}, "diagnosis_axes": [], "risk_summary": ""}

    monkeypatch.setattr(agent, "_call_llm", fake_llm)
    asyncio.run(
        agent._diagnostic_axis_consult(
            case_state={
                "chat_history": [],
                "ordered_examinations": [],
                "examination_results": {},
                "exam_decision_trace": [],
            },
            disease_candidates=[],
            memory_notes=[],
            patient_id="offline-only",
            prompt_name="diagnostic_axis_consult",
        )
    )

    assert captured["validate_pack"] is sentinel
    assert captured["select_pack"] is sentinel


def test_case_runtime_does_not_second_load_current_release(monkeypatch) -> None:
    """Composition Root may load once; production helpers must not re-read pointer."""
    load_calls: List[str] = []

    from agent.runtime import release_loader

    real_load = release_loader.load_current_release

    def counting_load(pointer_path):
        load_calls.append(str(pointer_path))
        return real_load(pointer_path)

    monkeypatch.setattr(release_loader, "load_current_release", counting_load)
    monkeypatch.setattr(
        "agent.runtime.release_loader.load_current_release",
        counting_load,
    )

    # Direct production-style select must use explicit pack, never pointer.
    from agent.agent import build_agent

    agent = build_agent()
    first_loads = list(load_calls)
    axes = select_diagnosis_axes(
        {"symptom_clusters": [{"label": "病例", "evidence": "发热"}]},
        case_state={
            "chat_history": [{"from": "patient", "text": "发热"}],
            "ordered_examinations": [],
            "examination_results": {},
            "exam_decision_trace": [],
        },
        rule_pack=agent.rule_pack,
    )
    assert isinstance(axes, list)
    # No additional load_current_release during select with explicit pack.
    assert load_calls == first_loads
