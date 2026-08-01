"""Generalized offline regressions from the twelfth random batch.

No patient IDs, expected exam packs, or reference treatment texts are hard-coded
into agent rules. Assertions only encode explainable clinical policies.
"""

from __future__ import annotations

import unittest

from agent import legacy_orchestrator as agent_module
from agent.legacy_orchestrator import (
    apply_axis_risk_gate,
    extract_intake_facts,
    final_verifier,
    open_coverage_gaps,
    select_diagnosis_axes,
    supported_axis_risks,
)


CATALOG = {
    "体格检查": ["生命体征", "鼻咽部检查", "口咽部检查", "体格检查"],
    "功能检查": ["脉搏血氧饱和度监测（SpO2）"],
    "实验室检查": ["细菌培养及鉴定", "尿液分析（UA）", "空腹血糖（FBG）", "口服葡萄糖耐量试验（OGTT）"],
}


def gdm_like_case() -> dict:
    return {
        "chat_history": [
            {
                "from": "patient",
                "text": "怀孕二十周，最近口渴尿多乏力，产检发现血糖高。没有头痛，也没有偏头痛史。",
            }
        ],
        "ordered_examinations": ["空腹血糖（FBG）", "口服葡萄糖耐量试验（OGTT）"],
        "examination_results": {
            "空腹血糖（FBG）": {"status": "abnormal", "result": {"空腹血糖": "6.4 mmol/L"}},
            "口服葡萄糖耐量试验（OGTT）": {
                "status": "abnormal",
                "result": {"空腹": "5.6", "1小时": "11.2", "2小时": "9.1"},
            },
        },
        "diagnosis_axes": [
            {
                "axis_id": "llm_noisy_axis",
                "evidence": ["口渴", "尿多"],
                "candidate_official_names": ["妊娠期糖尿病（GDM）"],
                "treatment_risks": ["pregnancy_screening_before_migraine_drugs"],
            }
        ],
        "treatment_risks": ["pregnancy_screening_before_migraine_drugs"],
        "positive_findings": ["口渴", "尿多", "乏力"],
        "candidate_diagnoses": ["妊娠期糖尿病（GDM）"],
    }


def postmenopausal_negative_culture_case() -> dict:
    return {
        "chat_history": [
            {
                "from": "patient",
                "text": "绝经后出现排尿烧灼、尿频尿急，偶有血丝，没有发热和腰痛。最近便秘，排尿会用力。",
            }
        ],
        "ordered_examinations": ["细菌培养及鉴定"],
        "examination_results": {
            "细菌培养及鉴定": {
                "status": "normal",
                "result": {"培养结果": "无细菌生长", "结论": "阴性"},
            }
        },
        "positive_findings": ["尿路刺激征", "绝经后"],
        "candidate_diagnoses": ["急性膀胱炎"],
    }


def hypoxic_respiratory_case() -> dict:
    return {
        "chat_history": [
            {
                "from": "patient",
                "text": "孩子流涕鼻塞四天，咳嗽加重，呼吸加快，夜间咳吐，没有哮喘史。",
            }
        ],
        "ordered_examinations": ["生命体征", "鼻咽部检查", "口咽部检查"],
        "examination_results": {
            "生命体征": {
                "status": "abnormal",
                "result": {
                    "呼吸频率": "36次/分",
                    "血氧饱和度": "92%",
                    "SpO2": "92%",
                    "体温": "36.8℃",
                },
            },
            "鼻咽部检查": {"status": "normal", "result": {"分泌物": "清亮"}},
            "口咽部检查": {"status": "normal", "result": {"扁桃体": "无脓苔"}},
        },
        "positive_findings": ["咳嗽", "呼吸加快"],
        "candidate_diagnoses": ["急性支气管炎"],
    }


def migraine_true_positive_case() -> dict:
    return {
        "chat_history": [
            {
                "from": "patient",
                "text": "育龄女性，偏头痛反复发作，坐飞机和旅行时加重，月经前也明显，伴恶心头晕。",
            }
        ],
        "ordered_examinations": [],
        "examination_results": {},
        "positive_findings": ["育龄女性", "偏头痛伴恶心头晕", "旅行或视觉运动诱发"],
        "candidate_diagnoses": ["偏头痛"],
        "diagnosis_axes": [
            {
                "axis_id": "migraine_reproductive_travel_trigger",
                "evidence": ["偏头痛伴恶心头晕", "育龄女性", "旅行或视觉运动诱发"],
                "candidate_official_names": ["偏头痛"],
                "treatment_risks": ["pregnancy_screening_before_migraine_drugs"],
            }
        ],
    }


class TwelfthRoundOfflineFailuresTest(unittest.TestCase):
    def test_migraine_risk_tag_is_dropped_without_migraine_context(self):
        risks = supported_axis_risks(
            ["pregnancy_screening_before_migraine_drugs"],
            axis_id="llm_noisy_axis",
            evidence=["口渴", "尿多"],
            candidates=["妊娠期糖尿病（GDM）"],
            normalized_case_text="怀孕二十周口渴尿多血糖高",
            diagnosis="妊娠期糖尿病（GDM）",
        )
        self.assertNotIn("pregnancy_screening_before_migraine_drugs", risks)

    def test_gdm_plan_is_not_polluted_by_migraine_pregnancy_patch(self):
        plan = (
            "制定医学营养治疗和血糖监测，必要时胰岛素治疗，加强产检与胎儿生长监测。"
        )
        result = apply_axis_risk_gate(
            plan,
            gdm_like_case(),
            diagnosis="妊娠期糖尿病（GDM）",
        )
        self.assertNotIn("pregnancy_screening_before_migraine_drugs", {item["code"] for item in result["issues"]})
        self.assertNotIn("偏头痛", result["treatment_plan"])
        self.assertEqual(plan, result["treatment_plan"])

    def test_true_migraine_still_requires_pregnancy_safety_when_missing(self):
        result = apply_axis_risk_gate(
            "给予布洛芬或曲普坦止痛，避免旅行诱因，记录发作频率。",
            migraine_true_positive_case(),
            diagnosis="偏头痛",
        )
        self.assertIn("pregnancy_screening_before_migraine_drugs", {item["code"] for item in result["issues"]})
        self.assertTrue(any("妊娠" in patch for patch in result["patches"]))

    def test_nausea_alone_does_not_open_migraine_axis(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "孕期恶心乏力，没有头痛，也没有偏头痛。"}
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        self.assertNotIn(
            "migraine_reproductive_travel_trigger",
            {item["axis_id"] for item in axes},
        )

    def test_postmenopausal_negative_culture_opens_noninfection_axis(self):
        axes = select_diagnosis_axes(extract_intake_facts(postmenopausal_negative_culture_case()))
        axis_ids = {item["axis_id"] for item in axes}
        self.assertIn("postmenopausal_urogenital_irritation_vs_infection", axis_ids)
        axis = next(
            item for item in axes if item["axis_id"] == "postmenopausal_urogenital_irritation_vs_infection"
        )
        candidates = set(axis["candidate_official_names"])
        self.assertIn("老年性阴道炎", candidates)
        self.assertIn("急性膀胱炎", candidates)

    def test_postmenopausal_negative_culture_blocks_empiric_fluoroquinolone_only_plan(self):
        case_state = postmenopausal_negative_culture_case()
        features = {
            "case_text": case_state["chat_history"][0]["text"],
            "patient_text": case_state["chat_history"][0]["text"],
            "positive_findings": case_state["positive_findings"],
            "candidate_diagnoses": case_state["candidate_diagnoses"],
            "examination_results": case_state["examination_results"],
            "diagnosis_axes": select_diagnosis_axes(extract_intake_facts(case_state)),
        }
        result = final_verifier(
            diagnosis="急性膀胱炎",
            examinations=["细菌培养及鉴定"],
            treatment_plan="经验性使用左氧氟沙星口服3到5天，多饮水，复查尿培养。",
            case_features=features,
            official_diseases=["急性膀胱炎", "老年性阴道炎", "尿道肉阜"],
            examination_catalog=CATALOG,
            exam_plan_trace=[],
            safety_profiles=[],
        )
        codes = {item["code"] for item in result["issues"]}
        self.assertIn("empiric_antibiotic_without_atrophy_path_after_negative_culture", codes)
        patched = result["patched_treatment"]
        self.assertTrue(any(marker in patched for marker in ["局部雌激素", "老年性", "萎缩", "外阴"]))
        self.assertFalse(
            "左氧氟沙星" in patched and "局部" not in patched and "雌激素" not in patched
        )

    def test_postmenopausal_plan_with_atrophy_path_is_accepted(self):
        case_state = postmenopausal_negative_culture_case()
        features = {
            "case_text": case_state["chat_history"][0]["text"],
            "patient_text": case_state["chat_history"][0]["text"],
            "positive_findings": case_state["positive_findings"],
            "candidate_diagnoses": ["老年性阴道炎", "急性膀胱炎"],
            "examination_results": case_state["examination_results"],
            "diagnosis_axes": select_diagnosis_axes(extract_intake_facts(case_state)),
        }
        plan = (
            "以绝经后泌尿生殖道萎缩评估为主，考虑局部雌激素制剂，处理便秘诱因，"
            "暂不经验性全身喹诺酮；若出现发热腰痛或复检提示感染再启动抗菌。"
        )
        result = final_verifier(
            diagnosis="老年性阴道炎",
            examinations=["细菌培养及鉴定"],
            treatment_plan=plan,
            case_features=features,
            official_diseases=["急性膀胱炎", "老年性阴道炎", "尿道肉阜"],
            examination_catalog=CATALOG,
            exam_plan_trace=[],
            safety_profiles=[],
        )
        self.assertNotIn(
            "empiric_antibiotic_without_atrophy_path_after_negative_culture",
            {item["code"] for item in result["issues"]},
        )

    def test_generic_bacterial_culture_counts_for_urinary_context(self):
        status_fn = getattr(agent_module, "urine_culture_evidence_status", None)
        self.assertTrue(callable(status_fn))
        features = {
            "case_text": "绝经后尿频尿急烧灼",
            "examination_results": {
                "细菌培养及鉴定": {"status": "normal", "result": {"培养结果": "无细菌生长"}}
            },
        }
        self.assertEqual("negative", status_fn(features))

    def test_respiratory_symptoms_without_vitals_open_coverage_gap(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "咳嗽四天加重，呼吸加快，夜间咳吐，流涕鼻塞。"}
            ],
            "ordered_examinations": ["鼻咽部检查"],
            "examination_results": {
                "鼻咽部检查": {"status": "normal", "result": {"分泌物": "清亮"}},
            },
        }
        gaps = open_coverage_gaps(case_state)
        self.assertIn("respiratory_oxygenation_vitals", {item["gap_id"] for item in gaps})
        required = []
        for item in gaps:
            if item["gap_id"] == "respiratory_oxygenation_vitals":
                required.extend(item.get("required_exams") or [])
        self.assertTrue(any(marker in exam for exam in required for marker in ["生命体征", "SpO2", "血氧"]))

    def test_hypoxia_requires_oxygen_therapy_goal(self):
        case_state = hypoxic_respiratory_case()
        features = {
            "case_text": case_state["chat_history"][0]["text"],
            "patient_text": case_state["chat_history"][0]["text"],
            "positive_findings": case_state["positive_findings"],
            "candidate_diagnoses": case_state["candidate_diagnoses"],
            "examination_results": case_state["examination_results"],
            "diagnosis_axes": select_diagnosis_axes(extract_intake_facts(case_state)),
        }
        result = final_verifier(
            diagnosis="急性支气管炎",
            examinations=["生命体征", "鼻咽部检查", "口咽部检查"],
            treatment_plan="多饮水，生理盐水滴鼻，必要时雾化，密切观察，暂不常规抗生素。",
            case_features=features,
            official_diseases=["急性支气管炎", "社区获得性肺炎"],
            examination_catalog=CATALOG,
            exam_plan_trace=[],
            safety_profiles=[],
        )
        self.assertIn("hypoxia_missing_oxygen_goal", {item["code"] for item in result["issues"]})
        self.assertTrue(any(marker in result["patched_treatment"] for marker in ["吸氧", "氧疗", "氧饱和"]))

    def test_hypoxia_with_oxygen_goal_is_accepted(self):
        case_state = hypoxic_respiratory_case()
        features = {
            "case_text": case_state["chat_history"][0]["text"],
            "patient_text": case_state["chat_history"][0]["text"],
            "positive_findings": case_state["positive_findings"],
            "candidate_diagnoses": case_state["candidate_diagnoses"],
            "examination_results": case_state["examination_results"],
            "diagnosis_axes": select_diagnosis_axes(extract_intake_facts(case_state)),
        }
        plan = "立即湿化吸氧维持 SpO2>94%，胸部物理治疗与雾化，监测呼吸与精神状态，免疫抑制背景需密切复诊。"
        result = final_verifier(
            diagnosis="急性支气管炎",
            examinations=["生命体征", "鼻咽部检查", "口咽部检查"],
            treatment_plan=plan,
            case_features=features,
            official_diseases=["急性支气管炎"],
            examination_catalog=CATALOG,
            exam_plan_trace=[],
            safety_profiles=[],
        )
        self.assertNotIn("hypoxia_missing_oxygen_goal", {item["code"] for item in result["issues"]})

    def test_urinary_case_can_require_menopause_status_intake(self):
        selector = getattr(agent_module, "select_required_intake_question", None)
        self.assertTrue(callable(selector))
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "中年女性，排尿烧灼尿频两天，没有发热腰痛。",
                }
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }
        question = selector(case_state)
        self.assertTrue(any(marker in question for marker in ["绝经", "更年期", "外阴"]))

    def test_answered_menopause_status_is_not_reasked(self):
        selector = getattr(agent_module, "select_required_intake_question", None)
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "中年女性，排尿烧灼尿频两天。"},
                {"from": "doctor", "text": "是否已绝经？有无外阴干涩？"},
                {"from": "patient", "text": "已绝经两年，有外阴干涩。"},
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }
        self.assertEqual("", selector(case_state))


if __name__ == "__main__":
    unittest.main()
