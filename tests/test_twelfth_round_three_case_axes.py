"""Generalized offline axes from the twelfth-round 3-case e2e batch.

No patient IDs, expected exam packs, or reference treatment texts are hard-coded
into agent rules.
"""

from __future__ import annotations

import unittest

from agent.legacy_orchestrator import (
    build_name_map,
    extract_intake_facts,
    final_verifier,
    flatten_examination_catalog,
    inject_required_differentials,
    load_disease_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    open_coverage_gaps,
    required_differential_from_case,
    select_diagnosis_axes,
    select_disease_candidates,
    select_exam_plan,
    should_block_final_for_coverage_gaps,
)


class TwelfthRoundThreeCaseAxesTest(unittest.TestCase):
    def setUp(self):
        self.disease_catalog = load_disease_catalog()
        self.examination_catalog = load_examination_catalog()
        self.item_name_map = build_name_map(flatten_examination_catalog(self.examination_catalog))
        knowledge = load_knowledge_registry()
        self.exam_profiles = knowledge["diagnosis_exam_profiles"]
        self.exam_intent_rules = knowledge["exam_intent_map"]
        self.official_diseases = [
            disease for diseases in self.disease_catalog.values() for disease in diseases
        ]

    def _plan(self, case_state, candidates=None, axes=None, max_items=8):
        if candidates is None:
            candidates = inject_required_differentials(
                select_disease_candidates(case_state, self.disease_catalog, limit=12),
                case_state=case_state,
                disease_catalog=self.disease_catalog,
            )
        if axes is None:
            axes = select_diagnosis_axes(extract_intake_facts(case_state))
        return (
            select_exam_plan(
                case_state=case_state,
                disease_candidates=candidates,
                diagnosis_axes=axes,
                examination_catalog=self.examination_catalog,
                item_name_map=self.item_name_map,
                diagnosis_exam_profiles=self.exam_profiles,
                exam_intent_rules=self.exam_intent_rules,
                max_items=max_items,
            ),
            candidates,
            axes,
        )

    def test_chronic_alcohol_liver_opens_axis_and_requires_labs_and_ultrasound(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "天天喝酒二十多年，最近半年乏力、腹胀、食欲差，喝酒后更明显。",
                }
            ],
            "ordered_examinations": [],
        }
        plan, candidates, axes = self._plan(case_state)
        axis_ids = {item["axis_id"] for item in axes}
        names = {item["disease"] for item in candidates}
        exams = set(plan["examinations"])

        self.assertIn("chronic_alcohol_liver_injury", axis_ids)
        self.assertIn("酒精性肝病", names)
        self.assertTrue(any("肝功能" in exam or "LFT" in exam for exam in exams))
        self.assertTrue(any("腹部超声" in exam for exam in exams))
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))

    def test_physical_exam_only_does_not_close_alcohol_liver_coverage(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "长期大量饮酒，乏力腹胀食欲减退。",
                }
            ],
            "ordered_examinations": ["腹部检查", "皮肤及巩膜检查"],
            "examination_results": {
                "腹部检查": {"status": "normal", "result": {"腹水": "未见"}},
                "皮肤及巩膜检查": {"status": "normal", "result": {"黄疸": "未见"}},
            },
        }
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}
        self.assertIn("alcohol_liver_lab_injury", gap_ids)
        self.assertIn("alcohol_liver_structure", gap_ids)
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))

    def test_alcohol_liver_lifestyle_only_plan_is_patched(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "天天喝酒，乏力腹胀食欲差。",
                }
            ],
            "ordered_examinations": ["肝功能检查（LFTs）", "腹部超声"],
            "examination_results": {
                "肝功能检查（LFTs）": {"status": "abnormal", "result": {"ALT": "升高"}},
                "腹部超声": {"status": "abnormal", "result": {"肝脏": "回声增强"}},
            },
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = final_verifier(
            diagnosis="酒精性肝病",
            examinations=["肝功能检查（LFTs）", "腹部超声"],
            treatment_plan="建议戒酒，控制饮食并减重，定期观察。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": case_state["chat_history"][0]["text"],
                "patient_text": case_state["chat_history"][0]["text"],
                "positive_findings": ["长期大量饮酒", "肝病相关症状"],
                "candidate_diagnoses": ["酒精性肝病"],
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )
        self.assertIn("undertreated_alcohol_liver_without_monitoring_path", {item["code"] for item in result["issues"]})
        self.assertTrue(
            any(marker in result["patched_treatment"] for marker in ["肝功能", "超声", "并发症", "戒酒"])
        )

    def test_pleuritic_pain_prioritizes_chest_path_not_apa_echo_first(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "右侧胸口刀割样疼痛三个月，深呼吸咳嗽翻身时加重，有干咳胸闷，以前有肺梗死。",
                }
            ],
            "ordered_examinations": [],
        }
        plan, candidates, axes = self._plan(case_state)
        axis_ids = {item["axis_id"] for item in axes}
        exams = plan["examinations"]
        names = {item["disease"] for item in candidates}

        self.assertIn("pleuritic_pain_infection_embolism_effusion", axis_ids)
        self.assertTrue(names & {"胸膜炎", "肺炎"})
        self.assertTrue(any(any(marker in exam for marker in ["胸部X线", "胸部CT", "CXR"]) for exam in exams))
        self.assertNotEqual(
            set(exams),
            {"抗磷脂抗体（APA）组合检测", "超声心动图"},
        )
        self.assertIn("胸膜炎", required_differential_from_case(case_state))

    def test_apa_echo_do_not_close_pleuritic_chest_coverage(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "胸痛呈刀割样，深呼吸和咳嗽时加重，伴干咳，有肺梗死史。",
                }
            ],
            "ordered_examinations": ["抗磷脂抗体（APA）组合检测", "超声心动图"],
            "examination_results": {
                "抗磷脂抗体（APA）组合检测": {"status": "normal", "result": {"结论": "阴性"}},
                "超声心动图": {"status": "normal", "result": {"心包积液": "无"}},
            },
        }
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}
        self.assertIn("pleuritic_chest_imaging", gap_ids)
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))

    def test_high_energy_hindfoot_trauma_keeps_calcaneus_fracture_axis(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "车祸后左脚跟剧痛肿胀，有瘀斑，形状好像变了，完全不敢踩地。",
                }
            ],
            "ordered_examinations": [],
        }
        plan, candidates, axes = self._plan(case_state)
        axis_ids = {item["axis_id"] for item in axes}
        names = {item["disease"] for item in candidates}

        self.assertIn("high_energy_hindfoot_trauma", axis_ids)
        self.assertIn("跟骨骨折", names)
        self.assertIn("四肢X线检查", plan["examinations"])
        self.assertIn("跟骨骨折", required_differential_from_case(case_state))

    def test_high_energy_hindfoot_sprain_only_plan_is_blocked(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "车祸导致左脚跟剧痛肿胀瘀斑，不敢负重。",
                }
            ],
            "ordered_examinations": ["四肢X线检查"],
            "examination_results": {
                "四肢X线检查": {"status": "normal", "result": {"骨折": "未见明确骨折"}},
            },
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = final_verifier(
            diagnosis="踝关节扭伤",
            examinations=["四肢X线检查"],
            treatment_plan="X线未见骨折，已排除骨折，按踝关节扭伤予 RICE 制动冰敷抬高即可。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": case_state["chat_history"][0]["text"],
                "patient_text": case_state["chat_history"][0]["text"],
                "positive_findings": ["高能量足跟创伤"],
                "candidate_diagnoses": ["踝关节扭伤", "跟骨骨折"],
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )
        codes = {item["code"] for item in result["issues"]}
        self.assertTrue(
            codes
            & {
                "high_energy_hindfoot_overcalled_as_sprain",
                "overclaim_rule_out_without_coverage",
            }
        )
        patched = result["patched_treatment"]
        self.assertTrue(any(marker in patched for marker in ["骨科", "跟骨", "进一步", "骨创伤"]))
        self.assertNotIn("已排除骨折", patched)

    def test_simple_ankle_sprain_without_high_energy_is_not_forced_to_calcaneus(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "走路时不小心崴脚，外踝轻度肿痛，还能慢慢走路。",
                }
            ],
            "ordered_examinations": [],
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        self.assertNotIn("high_energy_hindfoot_trauma", {item["axis_id"] for item in axes})
        self.assertNotIn("跟骨骨折", required_differential_from_case(case_state))


if __name__ == "__main__":
    unittest.main()
