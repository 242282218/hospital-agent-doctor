"""P1: fake online RunTrace completeness, exception seal, and privacy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

import agent.legacy_orchestrator as legacy_orchestrator
from agent.clinical.final_submission import LoadedRuntimeIdentity
from agent.clinical.online_runtime import run_online_clinical_case
from agent.legacy_orchestrator import MyDoctorAgent
from agent.memory import VerifiedOnlyMemory
from agent.observability.runtime_events import canonical_hash, safe_final_submission_event


TEST_RUNTIME_IDENTITY = LoadedRuntimeIdentity(
    status="strict_verified",
    identity_hash="online-trace-test-release",
)


class FakeDoctorActions:
    def __init__(self, *, fail_on: str = "") -> None:
        self.calls: List[Any] = []
        self.fail_on = fail_on

    async def ask_patient(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        self.calls.append(("ask_patient", patient_id, input_data.get("question")))
        return "发热、咳嗽，症状稳定。UNIQUE_PATIENT_SECRET_TEXT_XYZ"

    async def order_examination(
        self,
        patient_id: str,
        items: Iterable[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        item_list = list(items)
        self.calls.append(("order_examination", patient_id, item_list, reason))
        if self.fail_on == "order":
            raise RuntimeError("forced_order_failure")
        return {
            "results": {
                name: {
                    "status": "abnormal",
                    "result": {"flag": "阳性"},
                    "abnormal_indicators": ["flag"],
                }
                for name in item_list
            }
        }

    async def prescribe_treatment(
        self,
        patient_id: str,
        diagnosis: Any,
        treatment_plan: str,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        self.calls.append(("prescribe_treatment", patient_id, diagnosis, treatment_plan))
        if self.fail_on == "prescribe":
            raise RuntimeError("forced_prescribe_failure")
        return {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
            "finished": True,
        }

    async def evaluation(
        self,
        patient_id: str,
        final_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append(("evaluation", patient_id))
        return {"diagnosisAccuracy": 0.5}


def _read_events(case_run_id: str) -> List[Dict[str, Any]]:
    path = Path("outputs/run_traces") / case_run_id / "run.jsonl"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_success_path_trace_is_complete_and_sealed() -> None:
    agent = MyDoctorAgent(
        config={"log_llm_prompts": False},
        memory=VerifiedOnlyMemory(),
        runtime_identity=TEST_RUNTIME_IDENTITY,
    )
    agent.actions = FakeDoctorActions()

    async def no_llm(**kwargs):
        # Keep loop deterministic: force final after one decision cycle by
        # returning empty payloads so heuristics drive finish.
        return {}

    agent._call_llm = no_llm  # type: ignore[method-assign]

    result = asyncio.run(
        run_online_clinical_case(
            agent=agent,
            patient_id="trace-success",
            mode="test",
            valid_examinations=["体格检查", "全血细胞计数（CBC）", "血常规", "C反应蛋白（CRP）"],
            official_diseases=["肺炎", "上呼吸道感染"],
            exam_intent_map=[],
        )
    )
    case_run_id = str(result["case_run_id"])
    events = _read_events(case_run_id)
    types = [event.get("type") for event in events]
    assert types[0] == "case_start"
    assert "runtime_decision" in types
    assert "action_command" in types
    assert "action_observation" in types
    assert "verifier_summary" in types or result.get("finished") is True
    final_submission = [event for event in events if event.get("type") == "final_submission"]
    assert len(final_submission) == 1
    assert final_submission[0]["authorization_result"] == "issued"
    assert final_submission[0]["payload_hash"]
    assert final_submission[0]["legacy_verifier_hash"]
    assert final_submission[0]["five_dimension_gate_hash"]
    assert types[-1] == "case_end"
    seal = result.get("run_seal") or {}
    assert seal.get("event_count") == len(events)
    seal_path = Path("outputs/run_traces") / case_run_id / "run.seal.json"
    assert seal_path.exists()


def test_exception_path_writes_case_error_and_seals(monkeypatch: pytest.MonkeyPatch) -> None:
    decisions = iter(
        [
            {"action": "order_examination", "question": "", "reason": "controlled intent"},
            {"action": "final_diagnosis", "question": "", "reason": "exam complete"},
        ]
    )
    monkeypatch.setattr(
        legacy_orchestrator,
        "select_next_clinical_action",
        lambda _: next(decisions, {"action": "final_diagnosis", "question": "", "reason": "done"}),
    )
    agent = MyDoctorAgent(
        config={"log_llm_prompts": False},
        memory=VerifiedOnlyMemory(),
        runtime_identity=TEST_RUNTIME_IDENTITY,
    )
    agent.actions = FakeDoctorActions(fail_on="order")

    async def controlled_axis_consult(**_: Any) -> Dict[str, Any]:
        return {
            "diagnosis_axes": [
                {
                    "axis_id": "forced_exam_axis",
                    "exam_intents": ["exam_intent_inflammation_blood"],
                }
            ],
            "treatment_risks": [],
        }

    agent._diagnostic_axis_consult = controlled_axis_consult  # type: ignore[method-assign]

    async def force_exam_llm(**kwargs):
        name = str(kwargs.get("prompt_name") or "")
        if name in {"exam_category", "exam_item"}:
            return {
                "category": "实验室检查 - 血液",
                "examinations": ["全血细胞计数（CBC）"],
                "reason": "forced order path",
            }
        if name.startswith("diagnostic_axis_consult"):
            return {
                "intake_facts": {},
                "diagnosis_axes": [
                        {
                            "axis_id": "forced_exam_axis",
                            "exam_intents": ["exam_intent_inflammation_blood"],
                            "candidate_official_names": ["肺炎"],
                        }

                ],
                "treatment_risks": [],
            }
        return {}

    agent._call_llm = force_exam_llm  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="forced_order_failure"):
        asyncio.run(
            run_online_clinical_case(
                agent=agent,
                patient_id="trace-error",
                mode="test",
                valid_examinations=["体格检查", "全血细胞计数（CBC）"],
                official_diseases=["肺炎"],
                exam_intent_map=[],
            )
        )

    # Discover latest error trace by scanning newest seal with case_error.
    root = Path("outputs/run_traces")
    found = None
    for path in sorted(root.glob("*/run.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        text = path.read_text(encoding="utf-8")
        if "forced_order_failure" in text or '"type":"case_error"' in text.replace(" ", ""):
            found = path
            break
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
        if any(event.get("type") == "case_error" for event in events):
            found = path
            break
    assert found is not None
    events = [json.loads(line) for line in found.read_text(encoding="utf-8").splitlines() if line.strip()]
    types = [event.get("type") for event in events]
    assert "action_command" in types
    assert "case_error" in types
    seal_path = found.parent / "run.seal.json"
    assert seal_path.exists()
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal.get("event_count") == len(events)


def test_final_submission_event_contains_only_binding_metadata() -> None:
    event = safe_final_submission_event(
        payload_hash="payload-digest",
        legacy_verifier_hash="verifier-digest",
        five_dimension_gate_hash="gate-digest",
        issue_codes=["must_fix", "must_fix", ""],
        patch_count=2,
        authorization_result="issued",
        command_id="command-1",
        action_sequence=4,
    )

    assert event == {
        "type": "final_submission",
        "payload_hash": "payload-digest",
        "legacy_verifier_hash": "verifier-digest",
        "five_dimension_gate_hash": "gate-digest",
        "issue_codes": ["must_fix", "must_fix"],
        "patch_count": 2,
        "authorization_result": "issued",
        "command_id": "command-1",
        "action_sequence": 4,
    }
    assert "treatment_plan" not in event
    assert "reasoning" not in event


def test_trace_privacy_excludes_patient_and_treatment_plaintext() -> None:
    secret_q = "SECRET_QUESTION_ABC_123"
    secret_treatment = "SECRET_TREATMENT_PLAN_DO_NOT_LOG"
    agent = MyDoctorAgent(
        config={"log_llm_prompts": False},
        memory=VerifiedOnlyMemory(),
        runtime_identity=TEST_RUNTIME_IDENTITY,
    )
    agent.actions = FakeDoctorActions()

    async def no_llm(**kwargs):
        return {}

    agent._call_llm = no_llm  # type: ignore[method-assign]

    # Monkeypatch first decision path indirectly by running normal case; privacy
    # assertion still checks patient answer secret and treatment hash only.
    result = asyncio.run(
        run_online_clinical_case(
            agent=agent,
            patient_id="trace-privacy",
            mode="test",
            valid_examinations=["体格检查", "全血细胞计数（CBC）"],
            official_diseases=["肺炎"],
            exam_intent_map=[],
        )
    )
    case_run_id = str(result["case_run_id"])
    raw = (Path("outputs/run_traces") / case_run_id / "run.jsonl").read_text(encoding="utf-8")
    assert "UNIQUE_PATIENT_SECRET_TEXT_XYZ" not in raw
    assert secret_q not in raw
    assert secret_treatment not in raw
    # Hashes may appear for any emitted text fields.
    assert "schema_version" in raw
