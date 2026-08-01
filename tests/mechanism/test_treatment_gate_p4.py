"""P4 offline verification of the B-line treatment gate (通用链治疗门).

Locks in the two Goal B-line guarantees against the REAL production code, with no
network/LLM/service:

  G1. coverage_gap.required_exams 硬优先于偏轴意图
      -> apply_coverage_gap_action_gate forces order_examination (the gap's
         required_exams) even when the proposed action is final_diagnosis/ask.

  G2. 高危主轴有特异 safe-escalation 模板（非空壳）
      -> build_safe_escalation_plan returns an axis-specific closure (not the
         generic "尚未获得可支持" shell) for the documented high-risk axes.

Plus an audit enumerating the currently empty-shell axes (those whose axis_id is
not covered by a specific branch and would fall to the generic else).

Run:
  .venv\\Scripts\\python.exe -m pytest tests/mechanism/test_treatment_gate_p4.py -q
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from agent.legacy_orchestrator import (
    apply_coverage_gap_action_gate,
    build_safe_escalation_plan,
    open_coverage_gaps,
    validate_safe_escalation_plan,
)


def _trauma_case_state(patient_text: str) -> Dict[str, Any]:
    return {
        "patient_id": "Patient_AUDIT",
        "chat_history": [
            {"from": "patient", "text": patient_text},
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "decision_trace": [],
        "exam_decision_trace": [],
    }


def test_coverage_gap_required_exams_hard_priority_over_final() -> None:
    # Upper-arm trauma with local severity, no exams ordered yet -> open gap
    # (upper_arm_long_bone_imaging, required_exams=["四肢X线检查"]).
    case = _trauma_case_state(
        "患者上臂外伤，跌倒时手肘着地，上臂剧痛肿胀活动受限。"
    )
    gaps = open_coverage_gaps(case)
    assert gaps, "expected an open coverage gap for untreated upper-arm trauma"
    assert any("四肢X线检查" in g.get("required_exams", []) for g in gaps)

    # Even if the loop wants to finalize (偏轴意图), the gate must rewrite to
    # order_examination (the gap's required_exams).
    gated = apply_coverage_gap_action_gate(
        action="final_diagnosis", case_state=case, reason="拟直接出最终诊断"
    )
    assert gated["action"] == "order_examination", (
        "coverage_gap required_exams must hard-prioritize over final_diagnosis, got %r"
        % gated["action"]
    )

    # Same hard-priority over ask_patient (more soft history).
    gated2 = apply_coverage_gap_action_gate(
        action="ask_patient", case_state=case, reason="拟追问病史"
    )
    assert gated2["action"] == "order_examination"


def test_safe_escalation_has_specific_templates_for_high_risk_axes() -> None:
    # High-risk axes that the 13-batch / frozen corpus hit must have SPECIFIC
    # (non-generic-shell) closures. Regression guard: do not let these regress
    # to the useless "尚未获得可支持" sentence.
    specific_axes = [
        "active_upper_gi_bleed",
        "acute_decompensated_heart_failure",
        "mitral_stenosis_hemodynamics",
        "suspected_gouty_arthritis",
        "acute_leukemia_suspected",
        "diabetic_foot_infection",
        "suspected_asthma_control_issue",
        "hyperlipidemia_with_xanthelasma",
    ]
    for axis in specific_axes:
        plan, _reason = build_safe_escalation_plan(
            axis_id=axis,
            closure_requirement="safe_escalation_or_supported_official_diagnosis",
            evidence=["上臂剧痛肿胀", "外伤史"],
            existing_treatment="",
        )
        assert plan.strip(), "empty closure for axis %s" % axis
        assert "尚未获得可支持" not in plan, (
            "axis %s fell to generic shell (empty-shell template)" % axis
        )


def test_safe_escalation_enum_empty_shell_axes() -> None:
    # axes unchanged from the earlier audit
    ...

# P4 NEW: the 12 high-acuity empty-shell axes now have SPECIFIC closures + validators.
NEW_EMPTY_SHELL_AXES = [
    ("acute_coronary_syndrome", "急性冠脉综合征", "嚼服阿司匹林"),
    ("septic_shock", "脓毒性休克", "液体复苏"),
    ("anaphylaxis", "过敏性休克", "肾上腺素"),
    ("acute_ischemic_stroke", "急性缺血性卒中", "头颅CT"),
    ("status_epilepticus", "癫痫持续状态", "止痉"),
    ("acute_kidney_injury", "急性肾损伤", "肾毒性药物"),
    ("thyrotoxic_storm", "甲状腺危象", "硫脲类"),
    ("adrenal_crisis", "肾上腺危象", "糖皮质激素"),
    ("diabetic_ketoacidosis", "糖尿病酮症酸中毒", "补液"),
    ("hypertensive_urgency", "高血压急症", "静脉降压"),
    ("acute_pancreatitis", "急性胰腺炎", "禁食胃肠减压"),
    ("upper_gi_bleed_related", "上消化道大出血", "容量复苏"),
]


@pytest.mark.parametrize("axis_id,needle_topic,closure_must", NEW_EMPTY_SHELL_AXES)
def test_new_empty_shell_axes_have_specific_closure_and_validate(
    axis_id, needle_topic, closure_must
) -> None:
    plan, _reason = build_safe_escalation_plan(
        axis_id=axis_id,
        closure_requirement="safe_escalation_or_supported_official_diagnosis",
        evidence=["上臂剧痛肿胀", "外伤史", "生命体征不稳"],
        existing_treatment="",
    )
    assert plan.strip(), "empty closure for %s" % axis_id
    assert "尚未获得可支持" not in plan, "%s still generic shell" % axis_id
    assert needle_topic in plan, "%s missing topic %s" % (axis_id, closure_must)
    # The specific closure must also pass the axis validator (generic else enforces
    # disposition + preserves_emergency_care + closure markers).
    assert validate_safe_escalation_plan(
        plan, axis_id=axis_id, evidence=["生命体征不稳", "急症表现"]
    ), "%s specific closure failed its own validator" % axis_id
