"""P2 shared-root-cause mechanism tests (no patient_id special-casing)."""
from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_treatment_safety,
    diagnosis_supportive_treatment_plan,
    find_anti_infective_evidence_gaps,
    reconcile_selected_diagnosis_plan,
)


def test_reselected_diagnosis_rebuilds_treatment_not_old_axis() -> None:
    plan, reasoning = reconcile_selected_diagnosis_plan(
        {"normalized_diagnosis": "维生素D缺乏性佝偻病", "raw_diagnosis": "佝偻病"},
        selected_diagnosis="急性淋巴细胞白血病",
        treatment_plan="仅建议补充维生素D观察。",
        reasoning="原方案按佝偻病处理。",
        default_reasoning="默认",
    )
    assert "急性淋巴细胞白血病" in plan or "诱导化疗" in plan or "中枢" in plan
    assert "佝偻病" not in plan or "不得" in plan
    assert "急性淋巴细胞白血病" in reasoning or "候选疾病约束" in reasoning


def test_supportive_plan_covers_emergency_drug_monitor_layers() -> None:
    all_plan = diagnosis_supportive_treatment_plan("急性淋巴细胞白血病")
    assert "化疗" in all_plan
    assert "中枢" in all_plan or "CNS" in all_plan.upper() or "感染" in all_plan
    assert "监测" in all_plan or "急诊" in all_plan or "住院" in all_plan

    rhd = diagnosis_supportive_treatment_plan("风湿性心脏病")
    assert "心脏" in rhd or "瓣膜" in rhd
    assert "监测" in rhd or "急诊" in rhd

    pneumonia = diagnosis_supportive_treatment_plan("肺炎")
    assert "抗感染" in pneumonia or "感染" in pneumonia
    assert "监测" in pneumonia or "急诊" in pneumonia


def test_anti_infective_gate_requires_sensitivity_or_empiric() -> None:
    bare = "立即静脉使用环丙沙星抗感染治疗。"
    result = find_anti_infective_evidence_gaps(bare, {})
    assert result["issues"], "bare named antibiotic must be flagged"
    assert result["patches"]

    with_sens_features = {
        "anti_infective_provenance": {
            "ast": [{"drug": "环丙沙星", "result": "S", "source": "urine"}]
        }
    }
    assert not find_anti_infective_evidence_gaps(bare, with_sens_features)["issues"]

    with_empiric = bare + " 经验用药，待药敏结果后调整。"
    with_empiric_features = {
        "anti_infective_provenance": {
            "empiric": {
                "allowed": True,
                "indication": "急性细菌性前列腺炎",
                "must_reassess_on_ast": True,
                "source": "exam_result",
                "evidence_ref": "exam:urinalysis",
            }
        }
    }
    assert not find_anti_infective_evidence_gaps(with_empiric, with_empiric_features)["issues"]

    # A source-less empiric object is an unverifiable claim and must stay blocked.
    unsourced_features = {
        "anti_infective_provenance": {
            "empiric": {
                "allowed": True,
                "indication": "急性细菌性前列腺炎",
                "must_reassess_on_ast": True,
            }
        }
    }
    assert find_anti_infective_evidence_gaps(with_empiric, unsourced_features)["issues"]


def test_anti_infective_gate_skips_non_infection_context() -> None:
    # No concrete antibiotic is named here (only generic follow-up language), so
    # the gate must not fire. A concrete named antibiotic ALWAYS enters the gate
    # regardless of infection keywords (covered by the nested-SDK P0 tests).
    # This case verifies the complementary non-named path stays silent.
    plan = "继续脱敏相关随访，无感染证据，暂无特殊药物。"
    features = {
        "positive_findings": ["皮疹"],
        "candidate_diagnoses": ["斑秃"],
        "case_text": "脱发半年，无发热无咳嗽",
    }
    result = find_anti_infective_evidence_gaps(plan, features)
    assert not result["issues"]


def test_apply_treatment_safety_wires_i3_patch() -> None:
    plan = "针对急性细菌性前列腺炎：静脉环丙沙星。"
    result = apply_treatment_safety(
        plan,
        diagnosis="急性细菌性前列腺炎",
        case_features={
            "positive_findings": ["发热", "会阴痛", "尿频"],
            "candidate_diagnoses": ["急性细菌性前列腺炎"],
            "case_text": "发热会阴痛尿频",
        },
        safety_profiles=[],
    )
    assert result["issues"] or "经验用药" in (result.get("treatment_plan") or "")
    assert "药敏" in (result.get("treatment_plan") or "") or result["issues"]


def test_personalization_layers_present_in_common_shells() -> None:
    vsd = diagnosis_supportive_treatment_plan("室间隔缺损")
    assert "氧" in vsd or "喂养" in vsd
    migraine = diagnosis_supportive_treatment_plan("偏头痛")
    assert "NSAIDs" in migraine or "对乙酰氨基酚" in migraine or "曲坦" in migraine
    assert "监测" in migraine or "评估" in migraine
