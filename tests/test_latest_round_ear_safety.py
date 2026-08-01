"""Offline regressions for acute ear pain after instrumentation."""

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


class LatestRoundEarSafetyTest(unittest.TestCase):
    def setUp(self):
        self.catalog = load_examination_catalog()
        self.item_map = build_name_map(flatten_examination_catalog(self.catalog))
        self.intent_rules = load_knowledge_registry()["exam_intent_map"]

    def test_acute_ear_pain_after_cleaning_requires_otoscopy_before_final(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孩子耳朵疼、像堵住了还有耳鸣，掏耳朵后三天更疼，晚上疼得厉害。",
                }
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[],
            diagnosis_axes=axes,
            examination_catalog=self.catalog,
            item_name_map=self.item_map,
            diagnosis_exam_profiles=[],
            exam_intent_rules=self.intent_rules,
        )

        self.assertIn(
            "acute_ear_pain_after_instrumentation",
            {axis["axis_id"] for axis in axes},
        )
        self.assertIn(
            "acute_ear_pain_otoscopy",
            {gap["gap_id"] for gap in open_coverage_gaps(case_state)},
        )
        self.assertIn("耳镜检查", plan["examinations"])

    def test_routine_ear_cleaning_without_pain_does_not_open_otoscopy_gap(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "平时洗澡后会轻轻清洁外耳，没有耳痛、耳闷或耳鸣。"}
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertNotIn(
            "acute_ear_pain_after_instrumentation",
            {axis["axis_id"] for axis in axes},
        )
        self.assertNotIn(
            "acute_ear_pain_otoscopy",
            {gap["gap_id"] for gap in open_coverage_gaps(case_state)},
        )

    def test_ear_irrigation_is_removed_when_otoscopy_is_not_completed(self):
        case_features = {
            "case_text": "孩子耳朵疼、像堵住了还有耳鸣，掏耳朵后三天更疼。",
            "diagnosis_axes": [{"axis_id": "acute_ear_pain_after_instrumentation"}],
        }

        result = apply_treatment_specificity_gate(
            treatment_plan="考虑耵聍栓塞，建议直接冲洗耳道后观察。",
            diagnosis="耵聍栓塞",
            examinations=[],
            case_features=case_features,
        )

        self.assertIn(
            "ear_irrigation_before_otoscopy",
            {issue["code"] for issue in result["issues"]},
        )
        self.assertNotIn("冲洗", result["treatment_plan"])
        self.assertIn("耳镜", " ".join(result["patches"]))


if __name__ == "__main__":
    unittest.main()
