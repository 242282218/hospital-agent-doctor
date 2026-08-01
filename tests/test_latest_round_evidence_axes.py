"""Offline regressions for generalized evidence-closure gaps from the latest batch."""

from __future__ import annotations

import unittest

from agent.legacy_orchestrator import (
    apply_treatment_specificity_gate,
    build_name_map,
    extract_intake_facts,
    flatten_examination_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    open_coverage_gaps,
    select_diagnosis_axes,
    select_exam_plan,
)


class LatestRoundEvidenceAxesTest(unittest.TestCase):
    def setUp(self):
        self.catalog = load_examination_catalog()
        self.item_map = build_name_map(flatten_examination_catalog(self.catalog))
        self.intent_rules = load_knowledge_registry()["exam_intent_map"]

    def _plan(self, case_state: dict) -> dict:
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        return select_exam_plan(
            case_state=case_state,
            disease_candidates=[],
            diagnosis_axes=axes,
            examination_catalog=self.catalog,
            item_name_map=self.item_map,
            diagnosis_exam_profiles=[],
            exam_intent_rules=self.intent_rules,
        )

    def test_recently_born_infant_with_colloquial_fast_breathing_requires_echo(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "宝宝刚出生不久，今天呼吸突然变快，喂奶时更严重、容易出汗、吃奶少，嘴唇偶尔发青。",
                }
            ],
            "ordered_examinations": ["体格检查", "生命体征", "脉搏血氧饱和度监测（SpO2）"],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plan = self._plan(case_state)

        self.assertIn(
            "infant_congenital_structural_heart_disease",
            {axis["axis_id"] for axis in axes},
        )
        self.assertIn(
            "infant_chd_echocardiography",
            {gap["gap_id"] for gap in open_coverage_gaps(case_state)},
        )
        self.assertIn("超声心动图", plan["examinations"])

    def test_infant_without_cyanosis_does_not_open_structural_heart_axis(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "宝宝刚出生不久，吃奶时有些累但没有嘴唇发青，也没有呼吸变快。",
                }
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertNotIn(
            "infant_congenital_structural_heart_disease",
            {axis["axis_id"] for axis in axes},
        )

    def test_immunosuppressed_child_with_persistent_fever_and_purulent_cough_requires_chest_and_blood_workup(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孩子反复发热咳嗽六周，近几天咳黄绿色痰、喘得明显，正在服用免疫抑制药。",
                }
            ],
            "ordered_examinations": ["生命体征", "脉搏血氧饱和度监测（SpO2）"],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plan = self._plan(case_state)
        gap_ids = {gap["gap_id"] for gap in open_coverage_gaps(case_state)}

        self.assertIn(
            "high_risk_pediatric_lower_respiratory_infection",
            {axis["axis_id"] for axis in axes},
        )
        self.assertIn("高危儿科下呼吸道感染影像与病原覆盖", gap_ids)
        self.assertIn("胸部X线检查（CXR）", plan["examinations"])
        self.assertIn("血培养", plan["examinations"])

    def test_low_risk_short_cough_does_not_open_high_risk_lower_respiratory_axis(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "孩子咳嗽两天，没有发热，没有喘，也没有使用免疫抑制药。"}
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertNotIn(
            "high_risk_pediatric_lower_respiratory_infection",
            {axis["axis_id"] for axis in axes},
        )

    def test_high_risk_lower_respiratory_plan_cannot_stop_at_home_observation(self):
        case_features = {
            "case_text": "孩子反复发热咳嗽六周，近几天咳黄绿色痰、喘得明显，正在服用免疫抑制药。",
            "diagnosis_axes": [
                {"axis_id": "high_risk_pediatric_lower_respiratory_infection"}
            ],
        }

        result = apply_treatment_specificity_gate(
            treatment_plan="居家多饮水和雾化，48小时后无改善再复诊。",
            diagnosis="急性支气管炎",
            examinations=["生命体征", "脉搏血氧饱和度监测（SpO2）"],
            case_features=case_features,
        )

        self.assertIn(
            "high_risk_pediatric_lower_respiratory_escalation",
            {issue["code"] for issue in result["issues"]},
        )
        self.assertTrue(
            any(marker in " ".join(result["patches"]) for marker in ["急诊", "住院", "紧急评估"])
        )


if __name__ == "__main__":
    unittest.main()
