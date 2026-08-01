"""Generalized offline regressions from the thirteenth random 3-case batch."""

from __future__ import annotations

import unittest

from agent.legacy_orchestrator import (
    apply_treatment_specificity_gate,
    extract_intake_facts,
    final_verifier,
    load_disease_catalog,
    load_examination_catalog,
    open_coverage_gaps,
    prune_unsupported_disease_candidates,
    required_differential_from_case,
    select_diagnosis_axes,
    should_block_final_for_coverage_gaps,
)


class ThirteenthRoundOfflineFailuresTest(unittest.TestCase):
    def setUp(self):
        self.disease_catalog = load_disease_catalog()
        self.examination_catalog = load_examination_catalog()
        self.official_diseases = [
            disease for diseases in self.disease_catalog.values() for disease in diseases
        ]

    def febrile_polyuria_case(self, with_glucose: bool = False) -> dict:
        case = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "突然高烧39度，特别口渴，尿特别多，人都脱水了，头晕，血压也偏低。",
                }
            ],
            "ordered_examinations": ["血清电解质"],
            "examination_results": {
                "血清电解质": {
                    "status": "abnormal",
                    "result": {"血钠": "131 mEq/L", "血钾": "正常"},
                }
            },
        }
        if with_glucose:
            case["ordered_examinations"].extend(["血糖检测", "动脉血气（ABG）"])
            case["examination_results"]["血糖检测"] = {
                "status": "abnormal",
                "result": {"血糖": "17.3 mmol/L"},
            }
            case["examination_results"]["动脉血气（ABG）"] = {
                "status": "abnormal",
                "result": {"pH": "7.25", "碳酸氢根": "降低"},
            }
        return case

    def test_febrile_polyuria_opens_hyperglycemic_crisis_axis(self):
        axes = select_diagnosis_axes(extract_intake_facts(self.febrile_polyuria_case()))
        self.assertIn(
            "febrile_polyuria_dehydration_hyperglycemic_crisis",
            {item["axis_id"] for item in axes},
        )
        required = set(required_differential_from_case(self.febrile_polyuria_case()))
        self.assertTrue(required & {"1型糖尿病", "2型糖尿病（T2DM）", "尿崩症"})

    def test_electrolytes_alone_do_not_close_hyperglycemic_crisis_coverage(self):
        case_state = self.febrile_polyuria_case(with_glucose=False)
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}
        self.assertIn("hyperglycemic_crisis_glucose", gap_ids)
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))

    def test_diabetes_insipidus_hypotonic_path_blocked_before_crisis_exclusion(self):
        case_state = self.febrile_polyuria_case(with_glucose=False)
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = final_verifier(
            diagnosis="尿崩症",
            examinations=["血清电解质"],
            treatment_plan="按尿崩症处理，补充低渗盐水，稳定后使用去氨加压素。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": case_state["chat_history"][0]["text"],
                "patient_text": case_state["chat_history"][0]["text"],
                "positive_findings": ["高热", "多尿烦渴脱水"],
                "candidate_diagnoses": ["尿崩症", "2型糖尿病（T2DM）"],
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )
        codes = {item["code"] for item in result["issues"]}
        self.assertIn("di_path_before_hyperglycemic_crisis_exclusion", codes)
        patched = result["patched_treatment"]
        self.assertTrue(any(marker in patched for marker in ["血糖", "酮", "等渗", "高血糖危象", "DKA", "HHS"]))
        self.assertFalse("低渗盐水" in patched and "血糖" not in patched)

    def test_hyperglycemic_crisis_path_accepted_after_glucose_evidence(self):
        case_state = self.febrile_polyuria_case(with_glucose=True)
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plan = (
            "按疑似高血糖危象处理：等渗液体复苏，监测血糖酮体与血气，排查感染，"
            "并在专科指导下启动胰岛素路径，暂不按尿崩给予低渗液或去氨加压素。"
        )
        result = final_verifier(
            diagnosis="2型糖尿病（T2DM）",
            examinations=["血清电解质", "血糖检测", "动脉血气（ABG）"],
            treatment_plan=plan,
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": case_state["chat_history"][0]["text"],
                "patient_text": case_state["chat_history"][0]["text"],
                "positive_findings": ["高热", "多尿烦渴脱水"],
                "candidate_diagnoses": ["2型糖尿病（T2DM）", "尿崩症"],
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )
        self.assertNotIn(
            "di_path_before_hyperglycemic_crisis_exclusion",
            {item["code"] for item in result["issues"]},
        )

    def test_past_hay_fever_not_preferred_over_acute_gastroenteritis_complaint(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "暴食后上腹不适、呕吐腹泻，没有鼻子打喷嚏流涕加重。以前有季节性过敏性鼻炎。",
                }
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }
        candidates = prune_unsupported_disease_candidates(
            [
                {"disease": "花粉症（季节性过敏性鼻炎；花粉热）", "score": 80},
                {"disease": "急性单纯性胃炎", "score": 40},
                {"disease": "慢性胃炎", "score": 30},
            ],
            case_state,
        )
        names = [item["disease"] for item in candidates]
        self.assertIn("急性单纯性胃炎", names)
        self.assertTrue(names[0] != "花粉症（季节性过敏性鼻炎；花粉热）" or candidates[0]["score"] < 40)
        top = names[0]
        self.assertNotEqual(top, "花粉症（季节性过敏性鼻炎；花粉热）")

    def test_normal_electrolytes_do_not_force_magnesium_patch(self):
        case_features = {
            "case_text": "暴食后呕吐腹泻，上腹不适。",
            "patient_text": "暴食后呕吐腹泻，上腹不适。",
            "positive_findings": ["腹泻"],
            "examination_results": {
                "血清电解质": {
                    "status": "normal",
                    "result": {"血钾": "4.0 mmol/L [参考值：3.5-5.1]", "血钠": "140"},
                }
            },
        }
        result = apply_treatment_specificity_gate(
            treatment_plan="暂时禁食流质，口服补液，必要时止吐，暂不需静脉补钾。",
            diagnosis="急性单纯性胃炎",
            examinations=["血清电解质", "心电图（ECG）"],
            case_features=case_features,
        )
        self.assertNotIn("potassium_only_without_magnesium", {item["code"] for item in result["issues"]})
        self.assertNotIn("补镁", result["treatment_plan"])
        self.assertNotIn("门冬氨酸钾镁", result["treatment_plan"])

    def test_true_hypokalemia_with_diarrhea_still_requires_magnesium_when_replacing_k(self):
        case_features = {
            "case_text": "慢性腹泻，手抽筋，浑身没劲。",
            "patient_text": "慢性腹泻，手抽筋，浑身没劲。",
            "positive_findings": ["腹泻", "手抽筋"],
            "examination_results": {
                "血清电解质": {
                    "status": "abnormal",
                    "result": {"血钾": "2.8 mmol/L [参考值：3.5-5.1]"},
                }
            },
        }
        result = apply_treatment_specificity_gate(
            treatment_plan="口服氯化钾补钾治疗。",
            diagnosis="低钾血症",
            examinations=["血清电解质"],
            case_features=case_features,
        )
        self.assertIn("potassium_only_without_magnesium", {item["code"] for item in result["issues"]})


if __name__ == "__main__":
    unittest.main()
