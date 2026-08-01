"""Canary 07188: adult xanthelasma + lipids must not close as rickets."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    axis_alignment_official_names,
    finalize_treatment_with_verified_fallback,
    has_hyperlipidemia_with_xanthelasma_pattern,
    preferred_safe_escalation_diagnosis,
    prune_unsupported_disease_candidates,
    required_differential_from_case,
    select_diagnosis_axes,
    select_disease_candidates,
)


def _07188_text() -> str:
    return (
        "今年42岁。过去半年，我上眼睑反复出现发黄、轻度隆起的斑块，熬夜或压力大时更明显。"
        "同时偶有右上腹隐痛，吃高脂餐后容易乏力腹胀。没有黄疸、痒疹或视力下降。"
        "没有药物过敏史，平时不吃处方药，偶尔吃抗生素和复合维生素。"
        "没有确诊过高血压或糖尿病，但之前查过有胰岛素抵抗，血脂也偶尔偏高。"
        "血脂检测总胆固醇268、LDL172、甘油三酯286、HDL34。"
    )


def test_patient_07188_pattern_opens_hyperlipidemia_axis() -> None:
    text = _07188_text()
    assert has_hyperlipidemia_with_xanthelasma_pattern(text)
    axes = select_diagnosis_axes({}, case_state={
        "chat_history": [{"from": "patient", "text": text}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {
            "血脂检测组合": {
                "status": "abnormal",
                "result": {
                    "总胆固醇": "268 mg/dL",
                    "LDL胆固醇": "172 mg/dL",
                    "甘油三酯": "286 mg/dL",
                    "HDL胆固醇": "34 mg/dL",
                },
            }
        },
        "exam_decision_trace": [],
    })
    axis = next(a for a in axes if a["axis_id"] == "hyperlipidemia_with_xanthelasma")
    assert "混合型高脂血症" in axis["candidate_official_names"]
    assert axis["priority"] == "high"


def test_rickets_only_pediatric_is_not_hyperlipidemia_pattern() -> None:
    text = "2岁幼儿方颅鸡胸，夜惊多汗，诊断考虑维生素D缺乏性佝偻病，无眼睑斑块。"
    assert has_hyperlipidemia_with_xanthelasma_pattern(text) is False


def test_empty_llm_hyperlipidemia_axis_defaults_official_label() -> None:
    axis = {
        "axis_id": "hyperlipidemia_with_xanthelasma",
        "source": "llm",
        "priority": "high",
        "clinical_role": "current_problem",
        "status": "suspected",
        "candidate_official_names": [],
        "rule_candidate_official_names": [],
        "promotable_candidate_official_names": [],
        "evidence": ["眼睑黄色斑块", "血脂升高", "脂肪肝相关症状"],
    }
    assert axis_alignment_official_names(axis) == ["混合型高脂血症"]
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="维生素D缺乏性佝偻病",
        case_features={"diagnosis_axes": [axis]},
        escalation_axis=axis,
        official_diseases=["维生素D缺乏性佝偻病", "混合型高脂血症"],
    )
    assert preferred == "混合型高脂血症"


def test_finalize_empty_hyperlipidemia_axis_does_not_keep_rickets() -> None:
    case_features = {
        "diagnosis_axes": [
            {
                "axis_id": "hyperlipidemia_with_xanthelasma",
                "source": "llm",
                "validated": True,
                "status": "suspected",
                "clinical_role": "current_problem",
                "priority": "high",
                "closure_requirement": "supported_official_diagnosis",
                "evidence": ["眼睑黄色斑块", "血脂升高", "脂肪肝相关症状"],
                "candidate_official_names": [],
                "rule_candidate_official_names": [],
                "promotable_candidate_official_names": [],
            }
        ],
        "diagnosis_candidate_records": [
            {"disease": "维生素D缺乏性佝偻病", "role": "current_problem", "score": 30},
            {
                "disease": "混合型高脂血症",
                "role": "current_problem",
                "score": 120,
                "axis_id": "hyperlipidemia_with_xanthelasma",
            },
        ],
        "candidate_diagnoses": ["维生素D缺乏性佝偻病", "混合型高脂血症"],
    }
    diagnosis, plan, reasoning, receipt = finalize_treatment_with_verified_fallback(
        diagnosis="维生素D缺乏性佝偻病",
        treatment_plan="补充维生素D。",
        reasoning="误选佝偻病",
        verifier_result={"passed": False, "issues": [], "patched_treatment": ""},
        examinations=["血脂检测组合"],
        official_diseases=["维生素D缺乏性佝偻病", "混合型高脂血症"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )
    assert diagnosis == "混合型高脂血症"
    assert receipt.get("aligned_diagnosis", diagnosis) == "混合型高脂血症"
    # Plan may mention 佝偻病 only as a forbidden alternative, not as the regimen.
    assert "他汀" in plan or "血脂" in plan or "生活方式" in plan or "饮食" in plan
    assert not (
        ("补充维生素D" in plan or plan.strip().startswith("佝偻"))
        and "他汀" not in plan
        and "血脂" not in plan
    )
    assert receipt.get("degraded") in {
        "safe_escalation",
        "safe_escalation_unverified",
        "axis_aligned_repair",
        "reverified",
        "conservative_fallback",
        "conservative_fallback_unverified",
    }
    if str(receipt.get("degraded")).startswith("safe_escalation"):
        assert "血脂" in plan or "他汀" in plan or "生活方式" in plan


def test_catalog_rickets_demoted_when_xanthoma_lipids_present() -> None:
    text = _07188_text()
    case_state = {
        "chat_history": [{"from": "patient", "text": text}],
        "ordered_examinations": ["血脂检测组合"],
        "invalid_examinations": [],
        "examination_results": {
            "血脂检测组合": {
                "status": "abnormal",
                "result": {"总胆固醇": "268", "LDL胆固醇": "172", "甘油三酯": "286"},
            }
        },
        "exam_decision_trace": [],
    }
    assert "混合型高脂血症" in required_differential_from_case(case_state)
    candidates = select_disease_candidates(
        case_state,
        {"内分泌科": ["混合型高脂血症"], "儿科": ["维生素D缺乏性佝偻病"]},
        limit=8,
    )
    by_name = {item["disease"]: item for item in candidates}
    assert "混合型高脂血症" in by_name
    if "维生素D缺乏性佝偻病" in by_name:
        assert by_name["维生素D缺乏性佝偻病"]["role"] == "background_history"
        assert int(by_name["维生素D缺乏性佝偻病"]["score"] or 0) <= 15
    pruned = prune_unsupported_disease_candidates(
        [
            {"disease": "维生素D缺乏性佝偻病", "score": 80, "source": "catalog_match"},
            {"disease": "混合型高脂血症", "score": 20, "source": "catalog_match"},
        ],
        case_state,
    )
    by_p = {item["disease"]: item for item in pruned}
    assert int(by_p["混合型高脂血症"]["score"] or 0) >= 120
    assert by_p["维生素D缺乏性佝偻病"]["role"] == "background_history"
