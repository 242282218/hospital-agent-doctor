from __future__ import annotations

import asyncio

from agent.clinical.final_submission import (
    FinalVerificationError as SubmissionFinalVerificationError,
)
from agent.diagnosis_consistency import enforce_candidate_pool_consistency
from agent.legacy_orchestrator import (
    FinalVerificationError,
    MyDoctorAgent,
    RemoteServiceCaseError,
    build_safe_escalation_plan,
    classify_isolatable_case_error,
    exam_candidates_for_intent,
    finalize_treatment_with_verified_fallback,
    has_active_upper_gi_bleed_pattern,
    has_immunosuppressed_acute_infection_pattern,
    has_pediatric_congenital_glaucoma_pattern,
    incomplete_case_result,
    load_knowledge_registry,
    prune_unsupported_disease_candidates,
    run_batch_isolated,
    run_case_isolated,
    select_diagnosis_axes,
    validate_safe_escalation_plan,
)
from agent.memory import VerifiedOnlyMemory


def test_active_upper_gi_bleed_is_red_flag_and_not_arrhythmia_only() -> None:
    text = "肝硬化、门静脉高压，突然呕血，之后反复黑便，头晕心悸。"
    assert has_active_upper_gi_bleed_pattern(text)
    axes = select_diagnosis_axes({"symptom_clusters": [{"label": "病例", "evidence": text}]})
    bleed = next(axis for axis in axes if axis["axis_id"] == "active_upper_gi_bleed"
    )
    assert bleed["priority"] == "red_flag"
    assert bleed["clinical_role"] == "current_problem"
    assert bleed["closure_requirement"] == "urgent_hemostasis_and_resuscitation"


def test_immunosuppressed_cough_infection_does_not_require_progressive_dyspnea() -> None:
    text = "HIV感染，肾移植后服用他克莫司和泼尼松，最近发热、鼻塞、咽痛、咳嗽。"
    assert has_immunosuppressed_acute_infection_pattern(text)
    axes = select_diagnosis_axes({"symptom_clusters": [{"label": "病例", "evidence": text}]})
    axis = next(axis for axis in axes if axis["axis_id"] == "immunosuppressed_acute_infection")
    assert axis["priority"] == "high"
    assert axis["candidate_official_names"] == ["肺炎"]


def test_pediatric_high_iop_promotes_congenital_glaucoma() -> None:
    text = "3岁儿童，右眼眼压32mmHg，角膜水肿混浊，畏光流泪，视力下降。"
    assert has_pediatric_congenital_glaucoma_pattern(text)
    axes = select_diagnosis_axes({"symptom_clusters": [{"label": "病例", "evidence": text}]})
    axis = next(axis for axis in axes if axis["axis_id"] == "pediatric_congenital_glaucoma")
    assert axis["candidate_official_names"] == ["先天性青光眼"]
    assert axis["priority"] == "high"


def test_high_axis_with_closure_requires_safe_escalation_without_candidate() -> None:
    axes = [{
        "axis_id": "active_upper_gi_bleed",
        "source": "rule",
        "status": "suspected",
        "clinical_role": "current_problem",
        "priority": "red_flag",
        "closure_requirement": "urgent_hemostasis_and_resuscitation",
        "evidence": ["呕血", "反复黑便"],
        "rule_candidate_official_names": [],
    }]
    result = enforce_candidate_pool_consistency(
        [{"disease": "心律失常", "score": 80}],
        diagnosis_axes=axes,
        disease_catalog={"消化科": ["肝硬化", "心律失常"]},
    )
    assert result.safe_escalation_required is True


def test_safe_escalation_preserves_emergency_closure() -> None:
    plan, reasoning = build_safe_escalation_plan(
        axis_id="active_upper_gi_bleed",
        closure_requirement="urgent_hemostasis_and_resuscitation",
        evidence=["呕血", "反复黑便"],
        existing_treatment="建立静脉通路并急诊会诊。",
    )
    assert "急诊" in plan
    assert "内镜" in plan
    assert "静脉通路" in plan
    assert "active_upper_gi_bleed" in reasoning
    assert validate_safe_escalation_plan(
        plan,
        axis_id="active_upper_gi_bleed",
        evidence=["呕血", "反复黑便"],
    ) is True


def test_safe_escalation_rejects_unsupported_specific_drug() -> None:
    assert validate_safe_escalation_plan(
        "立即使用某种未经证实的特异药物。",
        axis_id="active_upper_gi_bleed",
        evidence=["呕血", "反复黑便"],
    ) is False


def _07907_gi_bleed_case_features() -> dict:
    """07907-style active upper GI bleed red-flag features for finalize isolation."""
    return {
        "diagnosis_axes": [{
            "axis_id": "active_upper_gi_bleed",
            "source": "rule",
            "status": "suspected",
            "clinical_role": "current_problem",
            "priority": "red_flag",
            "closure_requirement": "urgent_hemostasis_and_resuscitation",
            "evidence": ["肝硬化门静脉高压", "突然呕血", "反复黑便"],
            "rule_candidate_official_names": [],
            "candidate_official_names": [],
        }],
        "diagnosis_candidate_records": [
            {"disease": "心律失常", "role": "current_problem", "score": 80},
        ],
        "candidate_diagnoses": ["心律失常"],
    }


def test_finalize_active_upper_gi_bleed_failed_verifier_safe_escalation_no_raise(
    monkeypatch,
) -> None:
    # Force re-verify stalemate so finalize must use red-flag safe_escalation (07907).
    monkeypatch.setattr(
        "agent.legacy_orchestrator.converge_verified_treatment",
        lambda **_kwargs: None,
    )
    case_features = _07907_gi_bleed_case_features()
    diagnosis, plan, reasoning, receipt = finalize_treatment_with_verified_fallback(
        diagnosis="心律失常",
        treatment_plan="仅观察心悸。",
        reasoning="误判为心律失常",
        verifier_result={
            "passed": False,
            "issues": [{
                "severity": "must_fix",
                "patchable": False,
                "problem": "未覆盖活动性上消化道出血急症处置",
            }],
            "patched_treatment": "",
        },
        examinations=["全血细胞计数（CBC）"],
        official_diseases=["心律失常", "肝硬化"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )
    # converge_verified_treatment is forced to stalemate here, so there is no
    # verifier proof to claim. The red-flag plan must still be submitted, but the
    # receipt must report the unverified tier instead of synthesizing passed=True.
    assert receipt.get("degraded") == "safe_escalation_unverified"
    assert receipt.get("passed") is False
    assert receipt.get("verification_status") == "axis_closure_only"
    assert any(
        issue.get("code") == "final_verifier_not_converged"
        for issue in receipt.get("issues") or []
    )
    assert "急诊" in plan or "住院" in plan
    assert "内镜" in plan
    assert "active_upper_gi_bleed" in reasoning or receipt.get("unresolved_axis_id") == (
        "active_upper_gi_bleed"
    )
    assert diagnosis  # label may be axis-aligned or preserved; must not raise


def test_batch_isolation_preserves_first_case_when_second_raises_final_verification() -> None:
    def ok_case() -> dict:
        return {"diagnosis": "A", "treatment_plan": "观察"}

    def boom_case() -> dict:
        raise FinalVerificationError("final treatment and conservative fallback failed verification")

    results = run_batch_isolated([
        ("case_a", ok_case),
        ("case_b", boom_case),
    ])
    assert len(results) == 2
    assert results[0]["status"] == "ok"
    assert results[0]["result"]["diagnosis"] == "A"
    assert results[1]["status"] == "incomplete"
    assert results[1]["error_type"] == "FinalVerificationError"
    assert results[1]["result"] is None
    # Single-case helper matches batch row shape.
    isolated = run_case_isolated("case_b", boom_case)
    assert isolated["status"] == "incomplete"


def test_incomplete_case_result_is_finished_false_for_sdk_batch() -> None:
    payload = incomplete_case_result(
        "Patient_07907",
        FinalVerificationError("final treatment and conservative fallback failed verification"),
    )
    assert payload["finished"] is False
    assert payload["status"] == "incomplete"
    assert payload["error_type"] == "FinalVerificationError"
    assert payload["diagnosis"] == []
    assert payload["treatment_plan"] == ""


def test_agent_test_isolates_final_verification_error_for_sdk_batch(monkeypatch) -> None:
    """SDK re-raises agent.test exceptions → HTTP 500; agent must return incomplete dict."""
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=VerifiedOnlyMemory())

    async def boom_online(**_kwargs):
        raise FinalVerificationError(
            "final treatment and conservative fallback failed verification"
        )

    async def sequential() -> list[dict]:
        results: list[dict] = []

        async def ok_run(*, patient_id: str, **_kwargs):
            return {
                "patient_id": patient_id,
                "finished": True,
                "diagnosis": ["上呼吸道感染"],
                "treatment_plan": "对症",
                "reasoning": "ok",
                "authority": "ClinicalOrchestrator",
            }

        monkeypatch.setattr(
            "agent.clinical.online_runtime.run_online_clinical_case",
            ok_run,
        )
        results.append(await agent.test(patient_id="case_a"))
        monkeypatch.setattr(
            "agent.clinical.online_runtime.run_online_clinical_case",
            boom_online,
        )
        results.append(await agent.test(patient_id="case_b"))
        monkeypatch.setattr(
            "agent.clinical.online_runtime.run_online_clinical_case",
            ok_run,
        )
        results.append(await agent.test(patient_id="case_c"))
        return results

    results = asyncio.run(sequential())
    assert len(results) == 3
    assert results[0]["finished"] is True
    assert results[1]["finished"] is False
    assert results[1]["status"] == "incomplete"
    assert results[1]["error_type"] == "FinalVerificationError"
    assert results[2]["finished"] is True
    # No exception escaped agent.test — batch loop can continue.


def test_classify_isolates_http_555_but_not_attribute_error() -> None:
    assert isinstance(
        classify_isolatable_case_error(RuntimeError("ModelScope output audit HTTP 555")),
        RemoteServiceCaseError,
    )
    assert isinstance(
        classify_isolatable_case_error(TimeoutError("read timed out")),
        RemoteServiceCaseError,
    )
    assert classify_isolatable_case_error(AttributeError("'NoneType' object has no attribute 'x'")) is None
    assert classify_isolatable_case_error(TypeError("bad arg")) is None
    assert isinstance(
        classify_isolatable_case_error(FinalVerificationError("x")),
        FinalVerificationError,
    )


def test_classify_isolates_runtime_submission_verification_error() -> None:
    error = SubmissionFinalVerificationError("final payload failed verification")
    assert classify_isolatable_case_error(error) is error


def test_agent_test_isolates_remote_http_555_for_sdk_batch(monkeypatch) -> None:
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=VerifiedOnlyMemory())

    async def boom_555(**_kwargs):
        raise RuntimeError("upstream patient reply blocked: HTTP 555 output audit")

    async def ok_run(*, patient_id: str, **_kwargs):
        return {
            "patient_id": patient_id,
            "finished": True,
            "diagnosis": ["上呼吸道感染"],
            "treatment_plan": "对症",
            "reasoning": "ok",
            "authority": "ClinicalOrchestrator",
        }

    async def sequential() -> list[dict]:
        results: list[dict] = []
        monkeypatch.setattr(
            "agent.clinical.online_runtime.run_online_clinical_case",
            ok_run,
        )
        results.append(await agent.test(patient_id="case_a"))
        monkeypatch.setattr(
            "agent.clinical.online_runtime.run_online_clinical_case",
            boom_555,
        )
        results.append(await agent.test(patient_id="case_b"))
        monkeypatch.setattr(
            "agent.clinical.online_runtime.run_online_clinical_case",
            ok_run,
        )
        results.append(await agent.test(patient_id="case_c"))
        return results

    results = asyncio.run(sequential())
    assert results[0]["finished"] is True
    assert results[1]["finished"] is False
    assert results[1]["status"] == "incomplete"
    assert results[1]["error_type"] == "RemoteServiceCaseError"
    assert results[2]["finished"] is True


def test_agent_test_still_raises_programming_bugs(monkeypatch) -> None:
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=VerifiedOnlyMemory())

    async def boom_bug(**_kwargs):
        raise AttributeError("local programming fault")

    monkeypatch.setattr(
        "agent.clinical.online_runtime.run_online_clinical_case",
        boom_bug,
    )
    try:
        asyncio.run(agent.test(patient_id="case_bug"))
        assert False, "programming bugs must not be swallowed"
    except AttributeError:
        pass


def test_historical_lymphoma_is_background_without_current_activity() -> None:
    case_state = {
        "chat_history": [{
            "from": "patient",
            "text": "以前得过非霍奇金淋巴瘤，已经治疗。现在肾移植后服用免疫抑制剂，最近发热咳嗽。",
        }],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }
    candidates = prune_unsupported_disease_candidates(
        [{"disease": "非霍奇金淋巴瘤", "score": 90}],
        case_state,
    )
    assert candidates[0]["role"] == "background_history"


def test_new_exam_intents_use_catalog_leaves() -> None:
    rules = load_knowledge_registry()["exam_intent_map"]
    bleed = exam_candidates_for_intent("活动性上消化道出血严重度与止血准备", rules)
    assert {item["name"] for item in bleed} == {"凝血功能全套", "血型鉴定及交叉配血", "内镜检查"}
    glaucoma = exam_candidates_for_intent("儿童高眼压病因与视神经损害评估", rules)
    assert {item["name"] for item in glaucoma} == {"房角镜检查", "眼底镜检查", "眼压测量（IOP）"}
