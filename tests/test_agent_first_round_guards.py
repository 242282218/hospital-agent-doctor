import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from agent.legacy_orchestrator import (
    build_name_map,
    flatten_examination_catalog,
    load_knowledge_registry,
    should_force_examination,
    should_stop_patient_questions,
    should_stop_examinations,
    select_exam_plan,
    select_disease_candidates,
)
from agent.memory import MarkdownMemory


class FirstRoundGuardTest(unittest.TestCase):
    def test_select_disease_candidates_recalls_cross_department_skin_infection(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孩子两天前突然高热，脸和脖子长出很多小水泡，原来有湿疹的地方更严重。",
                },
                {
                    "from": "patient",
                    "text": "以前有特应性皮炎，现在皮疹结痂，很痛很痒，精神很差。",
                },
            ]
        }
        disease_catalog = {
            "儿科": ["流行性腮腺炎"],
            "皮肤科": ["卡波西水痘样疹", "疱疹样皮炎"],
            "感染科": ["水痘"],
        }

        candidates = select_disease_candidates(case_state, disease_catalog, limit=4)

        self.assertIn("卡波西水痘样疹", [item["disease"] for item in candidates])

    def test_select_disease_candidates_recalls_acute_bacterial_prostatitis(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "两天前开始发冷高烧，尿频尿急尿痛，排尿困难，差点尿不出来。",
                },
            ],
            "examination_results": {
                "尿液分析（UA）": {
                    "result": {"白细胞酯酶": "阳性", "亚硝酸盐": "阳性"}
                },
                "尿培养": {
                    "result": {"尿培养致病微生物": "大肠埃希菌：150,000 CFU/mL"}
                },
            },
        }
        disease_catalog = {
            "妇产科": ["卵睾性别发育异常（Ovotesticular DSD）"],
            "感染科": ["细菌感染"],
            "泌尿外科": ["急性细菌性前列腺炎"],
        }

        candidates = select_disease_candidates(case_state, disease_catalog, limit=4)

        self.assertEqual("急性细菌性前列腺炎", candidates[0]["disease"])


    def test_should_force_examination_after_two_patient_replies_without_exams(self):
        case_state = {
            "chat_history": [
                {"from": "doctor", "text": "哪里不舒服？"},
                {"from": "patient", "text": "发热和皮疹。"},
                {"from": "doctor", "text": "还有什么？"},
                {"from": "patient", "text": "精神很差。"},
            ],
            "ordered_examinations": [],
        }

        self.assertTrue(should_force_examination(case_state, min_patient_replies=2))

    def test_should_force_examination_when_question_repeats_answered_chronic_history(self):
        case_state = {
            "chat_history": [
                {"from": "doctor", "text": "是否有高血压、糖尿病，目前在用什么药？"},
                {"from": "patient", "text": "没有高血压，也没有糖尿病。没有长期处方药，偶尔吃布洛芬。"},
            ],
            "ordered_examinations": [],
        }

        self.assertTrue(
            should_force_examination(
                case_state,
                min_patient_replies=10,
                proposed_question="请再说一下您是否有高血压、糖尿病，以及现在用药情况。",
            )
        )

    def test_should_not_force_examination_for_new_unanswered_question_before_threshold(self):
        case_state = {
            "chat_history": [
                {"from": "doctor", "text": "是否有高血压、糖尿病，目前在用什么药？"},
                {"from": "patient", "text": "没有高血压，也没有糖尿病。没有长期处方药，偶尔吃布洛芬。"},
            ],
            "ordered_examinations": [],
        }

        self.assertFalse(
            should_force_examination(
                case_state,
                min_patient_replies=10,
                proposed_question="阴道脱出物平卧休息后是否能自行回纳？",
            )
        )

    def test_default_threshold_allows_third_non_repeated_high_yield_question(self):
        case_state = {
            "chat_history": [
                {"from": "doctor", "text": "请描述主要症状和病程。"},
                {"from": "patient", "text": "感冒后咳嗽、清嗓和鼻塞逐渐加重。"},
                {"from": "doctor", "text": "目前用药和药物过敏情况如何？"},
                {"from": "patient", "text": "偶尔服抗组胺药，没有药物过敏。"},
            ],
            "ordered_examinations": [],
        }

        self.assertFalse(
            should_force_examination(
                case_state,
                proposed_question="是否有发热、咳痰、呼吸困难或反酸烧心？",
            )
        )

    def test_should_not_treat_medication_allergy_question_as_repeated_history(self):
        case_state = {
            "chat_history": [
                {"from": "doctor", "text": "目前在用什么药？"},
                {"from": "patient", "text": "没有长期处方药，偶尔吃布洛芬。"},
            ],
            "ordered_examinations": [],
        }

        self.assertFalse(
            should_force_examination(
                case_state,
                min_patient_replies=10,
                proposed_question="是否有药物过敏或用药后出现过不良反应？",
            )
        )

    def test_should_force_examination_when_medication_allergy_was_explicitly_answered(self):
        case_state = {
            "chat_history": [
                {"from": "doctor", "text": "是否有药物过敏？"},
                {"from": "patient", "text": "没有药物过敏，也没有用药后不良反应。"},
            ],
            "ordered_examinations": [],
        }

        self.assertTrue(
            should_force_examination(
                case_state,
                min_patient_replies=10,
                proposed_question="请再确认是否有药物过敏或用药后不良反应。",
            )
        )

    def test_should_stop_examinations_after_action_limit(self):
        case_state = {
            "exam_decision_trace": [
                {"examinations": ["体格检查"]},
                {"examinations": ["病毒核酸检测（Viral NAT）"]},
                {"examinations": ["全血细胞计数（CBC）"]},
            ],
            "ordered_examinations": [
                "体格检查",
                "病毒核酸检测（Viral NAT）",
                "全血细胞计数（CBC）",
            ],
        }

        self.assertTrue(should_stop_examinations(case_state, max_exam_actions=3))

    def test_patient_question_hard_cap_applies_after_examinations(self):
        case_state = {
            "chat_history": [
                item
                for index in range(5)
                for item in (
                    {"from": "doctor", "text": f"问题{index}"},
                    {"from": "patient", "text": f"回答{index}"},
                )
            ],
            "ordered_examinations": ["体格检查"],
        }

        self.assertTrue(should_stop_patient_questions(case_state, max_patient_replies=5))

    def test_patient_question_hard_cap_counts_empty_patient_responses(self):
        case_state = {
            "chat_history": [
                item
                for index in range(5)
                for item in (
                    {"from": "doctor", "text": f"问题{index}"},
                    {"from": "patient", "text": ""},
                )
            ],
            "ordered_examinations": ["体格检查"],
        }

        self.assertTrue(should_stop_patient_questions(case_state, max_patient_replies=5))

    def test_select_exam_plan_uses_leaf_standard_names_for_eczema_herpeticum(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孩子高热，湿疹处出现成簇水泡和结痂，精神差，还在使用激素。",
                },
            ],
            "ordered_examinations": [],
        }
        disease_candidates = [
            {"department": "皮肤科", "disease": "卡波西水痘样疹", "score": 30},
        ]
        examination_catalog = {
            "体格检查": ["体格检查", "皮肤镜检查"],
            "实验室检查-血液": ["全血细胞计数（CBC）"],
            "微生物学检查": ["病毒核酸检测（Viral NAT）", "细菌培养及鉴定"],
            "病理检查": ["细胞学检查"],
        }

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
        )

        self.assertEqual(
            {
                "体格检查",
                "全血细胞计数（CBC）",
                "病毒核酸检测（Viral NAT）",
                "细胞学检查",
                "细菌培养及鉴定",
            },
            set(plan["examinations"]),
        )

    def test_select_exam_plan_excludes_already_ordered_exams(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "高热，特应性皮炎基础，皮肤有疱疹样水泡。"},
            ],
            "ordered_examinations": ["体格检查"],
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
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
        )

        self.assertNotIn("体格检查", plan["examinations"])
        self.assertIn("病毒核酸检测（Viral NAT）", plan["examinations"])

    def test_disease_candidates_do_not_use_memory_as_case_evidence(self):
        case_state = {
            "memory_notes": [
                "上一例：患儿湿疹基础，高热，簇集水泡，诊断卡波西水痘样疹。"
            ],
            "chat_history": [],
            "examination_results": {},
        }
        disease_catalog = {
            "皮肤科": ["卡波西水痘样疹"],
            "感染科": ["细菌感染"],
        }

        candidates = select_disease_candidates(case_state, disease_catalog, limit=4)

        self.assertEqual([], candidates)

    def test_exam_plan_does_not_use_memory_as_profile_evidence(self):
        case_state = {
            "memory_notes": [
                "上一例：湿疹基础，高热，疱疹样水泡，应完善病毒核酸和细胞学检查。"
            ],
            "chat_history": [],
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
        )

        self.assertEqual([], plan["examinations"])

    def test_memory_notes_exclude_current_patient_reflection(self):
        with TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.md"
            memory = MarkdownMemory(memory_path, max_notes=3)
            memory.append_case_reflection(
                patient_id="Patient_001",
                evaluation_reflection={"reflection": {"future_strategy": "same patient answer"}},
                status="verified",
            )
            memory.append_case_reflection(
                patient_id="Patient_002",
                evaluation_reflection={"reflection": {"future_strategy": "other patient rule"}},
                status="verified",
            )

            notes = memory.load_notes(exclude_patient_id="Patient_001")

        joined = "\n".join(notes)
        self.assertNotIn("same patient answer", joined)
        self.assertIn("other patient rule", joined)

    def test_new_reflection_is_candidate_and_not_loaded_by_default(self):
        with TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.md"
            memory = MarkdownMemory(memory_path, max_notes=3)
            memory.append_case_reflection(
                patient_id="Patient_001",
                evaluation_reflection={"reflection": {"future_strategy": "candidate rule"}},
            )

            default_notes = memory.load_notes()
            candidate_notes = memory.load_notes(include_candidates=True)

        self.assertEqual([], default_notes)
        self.assertIn("candidate rule", "\n".join(candidate_notes))


if __name__ == "__main__":
    unittest.main()
