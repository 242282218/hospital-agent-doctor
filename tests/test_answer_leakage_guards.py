import unittest
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.legacy_orchestrator import (
    build_name_map,
    flatten_examination_catalog,
    select_exam_plan,
)
from agent.memory import MarkdownMemory
from agent.prompt import (
    DIAGNOSTIC_AXIS_CONSULT_PROMPT,
    EVALUATION_REFLECTION_PROMPT,
    NEXT_ACTION_PROMPT,
    TREATMENT_REVIEW_PROMPT,
)


class AnswerLeakageGuardTest(unittest.TestCase):
    def test_exam_plan_requires_explainable_knowledge_rules_not_inline_answer_pack(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "孩子高热，湿疹处出现成簇水泡和结痂，精神差。"},
            ],
            "ordered_examinations": [],
        }
        disease_candidates = [
            {"department": "皮肤科", "disease": "卡波西水痘样疹", "score": 30},
        ]
        examination_catalog = {
            "体格检查": ["体格检查"],
            "实验室检查-血液": ["全血细胞计数（CBC）"],
            "微生物学检查": ["病毒核酸检测（Viral NAT）", "细菌培养及鉴定"],
            "病理检查": ["细胞学检查"],
        }

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=[],
            exam_intent_rules=[],
        )

        self.assertEqual([], plan["examinations"])

    def test_memory_reflection_strips_patient_id_and_answer_like_fields_when_verified(self):
        with TemporaryDirectory() as temp_dir:
            memory = MarkdownMemory(Path(temp_dir) / "memory.md", max_notes=3)
            memory.append_case_reflection(
                patient_id="Patient_02654",
                status="verified",
                evaluation_reflection={
                    "reflection": {
                        "profile": "Patient_02654 患儿高热伴水泡。",
                        "examination_reflection": "遗漏 expected: 体格检查、全血细胞计数（CBC）、病毒核酸检测（Viral NAT）。",
                        "treatment_reflection": "不要复制 reference 治疗原文。",
                        "future_strategy": "以后遇到类似病例不要背标准答案。",
                    }
                },
            )

            notes = "\n".join(memory.load_notes())

        self.assertNotIn("Patient_02654", notes)
        self.assertNotIn("expected", notes.lower())
        self.assertNotIn("reference", notes.lower())

    def test_reflection_prompt_forbids_expected_and_reference_copying(self):
        self.assertIn("不得复制完整 expected 检查列表", EVALUATION_REFLECTION_PROMPT)
        self.assertIn("不得复制 reference", EVALUATION_REFLECTION_PROMPT)

    def test_next_action_prompt_requires_medication_safety_intake(self):
        self.assertIn("药物过敏", NEXT_ACTION_PROMPT)
        self.assertIn("禁忌药", NEXT_ACTION_PROMPT)

    def test_next_action_prompt_collects_treatment_relevant_personal_context(self):
        self.assertIn("基础病或合并症", NEXT_ACTION_PROMPT)
        self.assertIn("职业或日常活动需求", NEXT_ACTION_PROMPT)
        self.assertIn("依从性", NEXT_ACTION_PROMPT)
        self.assertIn("治疗偏好", NEXT_ACTION_PROMPT)

    def test_axis_consult_limits_exam_intents_to_decision_changing_gaps(self):
        self.assertIn("最多 1-3 个", DIAGNOSTIC_AXIS_CONSULT_PROMPT)
        self.assertIn("改变主诊断或治疗安全", DIAGNOSTIC_AXIS_CONSULT_PROMPT)

    def test_treatment_review_requires_evidence_bound_safety_and_personalization(self):
        self.assertIn("不得编造", TREATMENT_REVIEW_PROMPT)
        self.assertIn("明确删除", TREATMENT_REVIEW_PROMPT)
        self.assertIn("基础病或合并症", TREATMENT_REVIEW_PROMPT)
        self.assertIn("职业或日常活动需求", TREATMENT_REVIEW_PROMPT)

    def test_production_knowledge_sources_do_not_contain_patient_ids(self):
        knowledge_dir = Path(__file__).resolve().parents[1] / "agent" / "knowledge"
        leaked = []
        for path in knowledge_dir.glob("*.json"):
            if re.search(r"Patient_(?:Comorbid-)?\d+", path.read_text(encoding="utf-8")):
                leaked.append(path.name)

        self.assertEqual([], leaked)


if __name__ == "__main__":
    unittest.main()
