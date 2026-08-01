"""T10: one bounded, de-anchored independent diagnosis review for low-confidence cases.

The review exists to catch anchoring on a first guess. It therefore must never
see the first diagnosis or its reasoning, must never run when the code gate is
already confident, and may use one main call plus one budgeted JSON repair.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List

import pytest

from agent.prompt import DIAGNOSIS_INDEPENDENT_REVIEW_PROMPT
from agent.legacy_orchestrator import (
    DiagnosisConfidenceReport,
    LOW_CONFIDENCE_REASON_CODES,
    assess_diagnosis_confidence,
    independent_review_specific_exam_cross_organ_conflict,
)


def _candidate(name: str, score: int, **extra: Any) -> Dict[str, Any]:
    row = {
        "disease": name,
        "score": score,
        "source": "catalog_match",
        "evidence_polarity": "positive",
    }
    row.update(extra)
    return row


# --- Step 2: the confidence report is a pure, closed-enum decision -------------


def test_reason_codes_are_a_closed_set() -> None:
    assert LOW_CONFIDENCE_REASON_CODES == frozenset(
        {
            "top2_margin_low",
            "dominant_axis_conflict",
            "high_risk_axis_unclosed",
            "reasoning_rejects_selected",
            "specific_exam_cross_organ_conflict",
        }
    )


def test_clear_margin_and_consistent_pool_is_high_confidence() -> None:
    report = assess_diagnosis_confidence(
        candidates=[_candidate("卡波西水痘样疹", 40), _candidate("湿疹", 10)],
        selected_diagnosis="卡波西水痘样疹",
        pool_consistent=True,
        safe_escalation_required=False,
        dominant_axis_closed=True,
    )
    assert isinstance(report, DiagnosisConfidenceReport)
    assert report.level == "high"
    assert report.reason_codes == ()
    assert report.top_two_margin == 30
    assert report.top_score == 40


def test_empty_candidate_pool_is_low_confidence() -> None:
    report = assess_diagnosis_confidence(
        candidates=[],
        selected_diagnosis="",
        pool_consistent=True,
        safe_escalation_required=False,
        dominant_axis_closed=True,
    )
    assert report.level == "low"
    assert "top2_margin_low" in report.reason_codes


def test_narrow_top_two_margin_is_low_confidence() -> None:
    report = assess_diagnosis_confidence(
        candidates=[_candidate("A", 20), _candidate("B", 15)],
        selected_diagnosis="A",
        pool_consistent=True,
        safe_escalation_required=False,
        dominant_axis_closed=True,
    )
    assert report.level == "low"
    assert "top2_margin_low" in report.reason_codes
    assert report.top_two_margin == 5


def test_inconsistent_pool_is_low_confidence() -> None:
    report = assess_diagnosis_confidence(
        candidates=[_candidate("A", 40), _candidate("B", 5)],
        selected_diagnosis="A",
        pool_consistent=False,
        safe_escalation_required=False,
        dominant_axis_closed=True,
    )
    assert report.level == "low"
    assert "dominant_axis_conflict" in report.reason_codes


def test_safe_escalation_and_unclosed_axis_are_low_confidence() -> None:
    report = assess_diagnosis_confidence(
        candidates=[_candidate("A", 40), _candidate("B", 5)],
        selected_diagnosis="A",
        pool_consistent=True,
        safe_escalation_required=True,
        dominant_axis_closed=False,
    )
    assert report.level == "low"
    assert "high_risk_axis_unclosed" in report.reason_codes


def test_report_reason_codes_stay_inside_the_closed_set() -> None:
    report = assess_diagnosis_confidence(
        candidates=[],
        selected_diagnosis="",
        pool_consistent=False,
        safe_escalation_required=True,
        dominant_axis_closed=False,
    )
    assert set(report.reason_codes) <= LOW_CONFIDENCE_REASON_CODES


def test_reasoning_rejection_triggers_low_confidence_review() -> None:
    report = assess_diagnosis_confidence(
        candidates=[_candidate("A", 40), _candidate("B", 5)],
        selected_diagnosis="A",
        pool_consistent=True,
        safe_escalation_required=False,
        dominant_axis_closed=True,
        reasoning_rejects_selected=True,
    )
    assert report.level == "low"
    assert report.reason_codes == ("reasoning_rejects_selected",)


def test_cross_organ_conflict_rejects_legacy_opaque_result() -> None:
    case_state = _case_state(
        examination_results={
            "特异检查": {
                "status": "abnormal",
                "result": {
                    "cross_organ_conflict": True,
                    "conflicting_axis_id": "pediatric_congenital_glaucoma",
                },
            }
        }
    )
    assert not independent_review_specific_exam_cross_organ_conflict(
        case_state, [{"axis_id": "pediatric_congenital_glaucoma"}]
    )


def test_cross_organ_conflict_requires_contract_bound_structured_finding() -> None:
    case_state = _case_state(
        examination_results={
            "特异检查": {
                "status": "abnormal",
                "result": {"opaque": True},
                "structured_findings": [
                    {
                        "schema_version": "exam-axis-evidence-contract/v1",
                        "finding_code": "controlled_respiratory_finding_against_ocular_axis",
                        "polarity": "present",
                        "target_system_id": "respiratory",
                        "source_evidence_id": "sdk:exam:controlled:001",
                    }
                ],
            }
        }
    )
    conflict = independent_review_specific_exam_cross_organ_conflict(
        case_state, [{"axis_id": "pediatric_congenital_glaucoma"}]
    )
    assert conflict
    report = assess_diagnosis_confidence(
        candidates=[_candidate("A", 40), _candidate("B", 5)],
        selected_diagnosis="A",
        pool_consistent=True,
        safe_escalation_required=False,
        dominant_axis_closed=True,
        specific_exam_cross_organ_conflict=conflict,
    )
    assert report.reason_codes == ("specific_exam_cross_organ_conflict",)


@pytest.mark.parametrize(
    "finding, status, axes",
    [
        (
            {
                "schema_version": "exam-axis-evidence-contract/v1",
                "finding_code": "controlled_respiratory_finding_against_ocular_axis",
                "polarity": "present",
                "target_system_id": "respiratory",
                "source_evidence_id": "",
            },
            "abnormal",
            [{"axis_id": "pediatric_congenital_glaucoma"}],
        ),
        (
            {
                "schema_version": "exam-axis-evidence-contract/v1",
                "finding_code": "unknown_finding",
                "polarity": "present",
                "target_system_id": "respiratory",
                "source_evidence_id": "sdk:exam:controlled:001",
            },
            "abnormal",
            [{"axis_id": "pediatric_congenital_glaucoma"}],
        ),
        (
            {
                "schema_version": "exam-axis-evidence-contract/v1",
                "finding_code": "controlled_respiratory_finding_against_ocular_axis",
                "polarity": "present",
                "target_system_id": "respiratory",
                "source_evidence_id": "sdk:exam:controlled:001",
            },
            "invalid",
            [{"axis_id": "pediatric_congenital_glaucoma"}],
        ),
        (
            {
                "schema_version": "exam-axis-evidence-contract/v1",
                "finding_code": "controlled_respiratory_finding_against_ocular_axis",
                "polarity": "present",
                "target_system_id": "respiratory",
                "source_evidence_id": "sdk:exam:controlled:001",
            },
            "abnormal",
            [{"axis_id": "inactive_axis"}],
        ),
    ],
)
def test_cross_organ_conflict_fails_closed_for_unverified_inputs(
    finding: Dict[str, str],
    status: str,
    axes: List[Dict[str, str]],
) -> None:
    case_state = _case_state(
        examination_results={
            "特异检查": {
                "status": status,
                "result": {"opaque": True},
                "structured_findings": [finding],
            }
        }
    )
    assert not independent_review_specific_exam_cross_organ_conflict(case_state, axes)


# --- Step 3: the prompt must be de-anchored ------------------------------------


def test_review_prompt_never_asks_for_the_first_diagnosis() -> None:
    text = DIAGNOSIS_INDEPENDENT_REVIEW_PROMPT
    for forbidden in ("初步诊断", "首次诊断", "原诊断", "treatment_plan", "reasoning"):
        assert forbidden not in text, forbidden
    for required in (
        "recommended_diagnosis",
        "supporting_evidence",
        "contradicting_evidence",
        "confidence",
    ):
        assert required in text, required


def test_review_prompt_placeholders_are_structured_only() -> None:
    text = DIAGNOSIS_INDEPENDENT_REVIEW_PROMPT
    for placeholder in (
        "$official_candidates",
        "$case_summary",
        "$examination_results",
        "$evidence_catalog",
        "$candidate_evidence",
    ):
        assert placeholder in text
    assert "{" not in text.replace("请只输出 JSON：\n{", "")
    # No slot may carry the first answer.
    assert "$diagnosis" not in text
    assert "$treatment_plan" not in text


# --- Steps 4-5: bounded call, code-gated replacement ---------------------------


import asyncio  # noqa: E402

from agent.legacy_orchestrator import MyDoctorAgent  # noqa: E402


def _agent() -> MyDoctorAgent:
    return MyDoctorAgent(config={"log_llm_prompts": False}, memory=None)


class _ReviewProvider:
    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def call(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({"prompt": prompt, "temperature": temperature})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return str(item)


def _case_state(**extra: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "patient_id": "Patient_01061",
        "mode": "test",
        "chat_history": [
            {"from": "doctor", "text": "哪里不适"},
            {"from": "patient", "text": "免疫抑制状态，出现成簇水疱伴发热。"},
        ],
        "ordered_examinations": ["体格检查"],
        "invalid_examinations": [],
        "examination_results": {
            "体格检查": {"status": "abnormal", "result": {"summary": "成簇水疱"}},
        },
        "decision_trace": [],
        "exam_decision_trace": [],
    }
    state.update(extra)
    return state


def _run_review(
    agent: MyDoctorAgent,
    *,
    report: DiagnosisConfidenceReport,
    candidates: List[Dict[str, Any]],
    selected: str,
    llm: Any,
) -> Dict[str, Any]:
    agent._call_llm = llm  # type: ignore[method-assign]
    case_state = _case_state()
    result = asyncio.run(
        agent._run_independent_diagnosis_review(
            case_state=case_state,
            confidence=report,
            disease_candidates=candidates,
            selected_diagnosis=selected,
            diagnosis_axes=[],
            patient_id="Patient_01061",
        )
    )
    return {"diagnosis": result, "case_state": case_state}


def _low_report() -> DiagnosisConfidenceReport:
    return assess_diagnosis_confidence(
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected_diagnosis="卡波西水痘样疹",
        pool_consistent=True,
        safe_escalation_required=False,
        dominant_axis_closed=True,
    )


def _high_report() -> DiagnosisConfidenceReport:
    return assess_diagnosis_confidence(
        candidates=[_candidate("卡波西水痘样疹", 40), _candidate("湿疹", 5)],
        selected_diagnosis="卡波西水痘样疹",
        pool_consistent=True,
        safe_escalation_required=False,
        dominant_axis_closed=True,
    )


def test_high_confidence_case_never_calls_the_provider() -> None:
    agent = _agent()
    calls: List[str] = []

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        calls.append(str(kwargs.get("prompt_name")))
        raise AssertionError("high-confidence case must not call the provider")

    out = _run_review(
        agent,
        report=_high_report(),
        candidates=[_candidate("卡波西水痘样疹", 40), _candidate("湿疹", 5)],
        selected="卡波西水痘样疹",
        llm=llm,
    )
    assert calls == []
    assert out["diagnosis"] == "卡波西水痘样疹"
    review = out["case_state"].get("diagnosis_independent_review")
    assert review["triggered"] is False
    assert review["provider_calls"] == 0


def test_low_confidence_case_calls_the_provider_exactly_once() -> None:
    agent = _agent()
    calls: List[Dict[str, Any]] = []

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {
            "recommended_diagnosis": "湿疹",
            "supporting_evidence_ids": ["patient:a188ebedaf33f790"],
            "contradicting_evidence_ids": [],
            "confidence": "high",
        }

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=llm,
    )
    assert len(calls) == 1
    assert calls[0]["prompt_name"] == "diagnosis_independent_review"
    review = out["case_state"]["diagnosis_independent_review"]
    assert review["triggered"] is True
    assert review["provider_calls"] == 1
    assert review["before"] == "卡波西水痘样疹"
    assert out["diagnosis"] == "湿疹"
    assert review["accepted"] is True
    assert review["after"] == "湿疹"


def test_review_prompt_payload_excludes_the_first_diagnosis() -> None:
    agent = _agent()
    seen: Dict[str, Any] = {}

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        seen["prompt"] = kwargs.get("prompt")
        return {"recommended_diagnosis": "", "supporting_evidence": [], "contradicting_evidence": [], "confidence": "low"}

    _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=llm,
    )
    prompt = str(seen["prompt"])
    # Candidates legitimately appear; the *selected* label must not be marked as chosen.
    for forbidden in ("初步诊断", "首次诊断", "原诊断", "已选诊断"):
        assert forbidden not in prompt, forbidden
    assert "免疫抑制状态，出现成簇水疱伴发热。" in prompt
    assert "patient:a188ebedaf33f790" in prompt
    assert "candidate_evidence" not in prompt


def test_out_of_pool_review_result_is_rejected() -> None:
    agent = _agent()

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        return {
            "recommended_diagnosis": "完全不在候选池的疾病",
            "supporting_evidence": ["x"],
            "contradicting_evidence": [],
            "confidence": "high",
        }

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=llm,
    )
    assert out["diagnosis"] == "卡波西水痘样疹"
    review = out["case_state"]["diagnosis_independent_review"]
    assert review["accepted"] is False


def test_in_pool_review_with_fabricated_evidence_id_is_rejected() -> None:
    agent = _agent()

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        return {
            "recommended_diagnosis": "湿疹",
            "supporting_evidence_ids": ["fabricated-evidence-id"],
            "contradicting_evidence_ids": [],
            "confidence": "high",
        }

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=llm,
    )
    assert out["diagnosis"] == "卡波西水痘样疹"
    assert out["case_state"]["diagnosis_independent_review"]["accepted"] is False


def test_review_rejects_evidence_that_only_supports_another_candidate() -> None:
    agent = _agent()
    candidates = [_candidate("卡波西水痘样疹", 20), _candidate("湿疹", 15)]
    diagnosis_axes = [
        {
            "axis_id": "viral_axis",
            "candidate_official_names": ["卡波西水痘样疹"],
            "evidence": ["成簇水疱"],
        }
    ]

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        return {
            "recommended_diagnosis": "湿疹",
            "supporting_evidence_ids": ["axis:viral_axis:support:1"],
            "contradicting_evidence_ids": [],
            "confidence": "high",
        }

    agent._call_llm = llm  # type: ignore[method-assign]
    case_state = _case_state()
    result = asyncio.run(
        agent._run_independent_diagnosis_review(
            case_state=case_state,
            confidence=_low_report(),
            disease_candidates=candidates,
            selected_diagnosis="卡波西水痘样疹",
            diagnosis_axes=diagnosis_axes,
            patient_id="Patient_01061",
        )
    )
    assert result == "卡波西水痘样疹"
    assert case_state["diagnosis_independent_review"]["accepted"] is False


def test_exhausted_budget_skips_the_review() -> None:
    agent = _agent()
    agent.llm_hard_cap = 1
    agent.llm_calls_used = 1
    calls: List[str] = []

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        calls.append("called")
        raise AssertionError("must not call provider without budget")

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=llm,
    )
    assert calls == []
    review = out["case_state"]["diagnosis_independent_review"]
    assert review["triggered"] is False
    assert "budget_exhausted" in review["skip_reasons"]


def test_inconsistent_candidate_pool_triggers_one_review() -> None:
    agent = _agent()
    calls: List[str] = []

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        calls.append(str(kwargs["prompt_name"]))
        return {"recommended_diagnosis": "", "confidence": "low"}

    report = assess_diagnosis_confidence(
        candidates=[_candidate("卡波西水痘样疹", 40), _candidate("湿疹", 5)],
        selected_diagnosis="卡波西水痘样疹",
        pool_consistent=False,
        safe_escalation_required=False,
        dominant_axis_closed=True,
    )
    out = _run_review(
        agent,
        report=report,
        candidates=[_candidate("卡波西水痘样疹", 40), _candidate("湿疹", 5)],
        selected="卡波西水痘样疹",
        llm=llm,
    )
    assert calls == ["diagnosis_independent_review"]
    assert out["case_state"]["diagnosis_independent_review"]["reason_codes"] == [
        "dominant_axis_conflict"
    ]


def test_review_runs_at_most_once_per_case() -> None:
    agent = _agent()
    count = {"n": 0}

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        count["n"] += 1
        return {
            "recommended_diagnosis": "湿疹",
            "supporting_evidence_ids": ["patient:a188ebedaf33f790"],
            "contradicting_evidence_ids": [],
            "confidence": "high",
        }

    agent._call_llm = llm  # type: ignore[method-assign]
    case_state = _case_state()
    candidates = [_candidate("卡波西水痘样疹", 20), _candidate("湿疹", 15)]
    for _ in range(3):
        asyncio.run(
            agent._run_independent_diagnosis_review(
                case_state=case_state,
                confidence=_low_report(),
                disease_candidates=candidates,
                selected_diagnosis="卡波西水痘样疹",
                diagnosis_axes=[],
                patient_id="Patient_01061",
            )
        )
    assert count["n"] == 1
    assert case_state["diagnosis_independent_review"]["provider_calls"] == 1


def test_review_disables_transient_main_retry() -> None:
    agent = _agent()
    provider = _ReviewProvider([TimeoutError("transient")])
    agent.llm = provider  # type: ignore[assignment]

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=agent._call_llm,
    )

    review = out["case_state"]["diagnosis_independent_review"]
    assert len(provider.calls) == 1
    assert review["main_provider_calls"] == 1
    assert review["repair_provider_calls"] == 0
    assert review["provider_calls"] == 1
    assert review["budget_before"] == 0
    assert review["budget_after"] == 1


def test_review_provider_exception_records_actual_consumed_calls() -> None:
    agent = _agent()
    provider = _ReviewProvider([ValueError("invalid")])
    agent.llm = provider  # type: ignore[assignment]

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=agent._call_llm,
    )

    review = out["case_state"]["diagnosis_independent_review"]
    assert len(provider.calls) == 1
    assert review["main_provider_calls"] == 1
    assert review["repair_provider_calls"] == 0
    assert review["provider_calls"] == 1
    assert review["budget_after"] == 1


def test_review_allows_one_budgeted_json_repair() -> None:
    agent = _agent()
    provider = _ReviewProvider(
        [
            "not json",
            '{"recommended_diagnosis":"湿疹","supporting_evidence_ids":["patient:a188ebedaf33f790"],"contradicting_evidence_ids":[],"confidence":"high"}',
        ]
    )
    agent.llm = provider  # type: ignore[assignment]

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=agent._call_llm,
    )

    review = out["case_state"]["diagnosis_independent_review"]
    assert out["diagnosis"] == "湿疹"
    assert len(provider.calls) == 2
    assert review["main_provider_calls"] == 1
    assert review["repair_provider_calls"] == 1
    assert review["provider_calls"] == 2
    assert review["budget_before"] == 0
    assert review["budget_after"] == 2


def test_review_repair_exception_records_both_provider_calls() -> None:
    agent = _agent()
    provider = _ReviewProvider(["not json", TimeoutError("repair timeout")])
    agent.llm = provider  # type: ignore[assignment]

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=agent._call_llm,
    )

    review = out["case_state"]["diagnosis_independent_review"]
    assert len(provider.calls) == 2
    assert review["main_provider_calls"] == 1
    assert review["repair_provider_calls"] == 1
    assert review["provider_calls"] == 2
    assert review["budget_after"] == 2


def test_review_rejects_repair_when_main_spends_last_budget_unit() -> None:
    agent = _agent()
    agent.llm_hard_cap = 1
    provider = _ReviewProvider(["not json"])
    agent.llm = provider  # type: ignore[assignment]

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=agent._call_llm,
    )

    review = out["case_state"]["diagnosis_independent_review"]
    assert out["diagnosis"] == "卡波西水痘样疹"
    assert len(provider.calls) == 1
    assert review["main_provider_calls"] == 1
    assert review["repair_provider_calls"] == 0
    assert review["provider_calls"] == 1
    assert review["budget_after"] == 1


def test_review_records_after_state_and_reason_codes() -> None:
    agent = _agent()

    async def llm(**kwargs: Any) -> Dict[str, Any]:
        return {
            "recommended_diagnosis": "湿疹",
            "supporting_evidence": ["非感染性皮损"],
            "contradicting_evidence": [],
            "confidence": "high",
        }

    out = _run_review(
        agent,
        report=_low_report(),
        candidates=[
            _candidate("卡波西水痘样疹", 20),
            _candidate(
                "湿疹",
                15,
                matched_evidence=["免疫抑制状态，出现成簇水疱伴发热。"],
            ),
        ],
        selected="卡波西水痘样疹",
        llm=llm,
    )
    review = out["case_state"]["diagnosis_independent_review"]
    assert set(review) >= {
        "triggered",
        "reason_codes",
        "provider_calls",
        "accepted",
        "before",
        "after",
        "skip_reasons",
    }
    assert "top2_margin_low" in review["reason_codes"]
    assert review["after"] in {"卡波西水痘样疹", "湿疹"}
