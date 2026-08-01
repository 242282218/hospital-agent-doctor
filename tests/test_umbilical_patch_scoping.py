"""Neonatal umbilical patches must not pollute pediatric respiratory cases."""

from __future__ import annotations

import unittest

from agent.legacy_orchestrator import (
    final_verifier,
    load_disease_catalog,
    load_examination_catalog,
    supported_axis_risks,
)


class TestUmbilicalPatchScoping(unittest.TestCase):
    def setUp(self):
        self.official = [d for ds in load_disease_catalog().values() for d in ds]
        self.exam_catalog = load_examination_catalog()

    def test_unsupported_umbilical_risk_dropped(self):
        risks = supported_axis_risks(
            ["avoid_no_further_care_for_bleeding_mass"],
            axis_id="generic_llm_axis",
            evidence=["喘息"],
            candidates=["急性支气管炎"],
            normalized_case_text="十岁儿童感冒后喘息黄绿色鼻涕",
            diagnosis="急性支气管炎",
        )
        self.assertNotIn("avoid_no_further_care_for_bleeding_mass", risks)

    def test_pediatric_bronchitis_not_polluted_by_umbilical_patches(self):
        result = final_verifier(
            diagnosis="急性支气管炎",
            examinations=["生命体征", "体格检查", "前鼻镜检查"],
            treatment_plan=(
                "吸入沙丁胺醇缓解喘息，生理盐水洗鼻，必要时鼻用激素；"
                "观察呼吸与血氧，警惕肺炎。"
            ),
            official_diseases=self.official,
            examination_catalog=self.exam_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": "10岁儿童感冒后喘息、黄绿色鼻涕、过敏性鼻炎史",
                "patient_text": "10岁儿童感冒后喘息、黄绿色鼻涕、过敏性鼻炎史",
                "positive_findings": ["新生儿脐部病变", "湿润易出血肿块"],  # LLM pollution
                "candidate_diagnoses": ["急性支气管炎", "新生儿脐炎"],
                "treatment_risks": ["avoid_no_further_care_for_bleeding_mass"],
                "diagnosis_axes": [
                    {
                        "axis_id": "fake_llm_axis",
                        "treatment_risks": ["avoid_no_further_care_for_bleeding_mass"],
                        "evidence": ["喘息"],
                        "candidate_official_names": ["急性支气管炎"],
                    }
                ],
                "examination_results": {},
            },
            safety_profiles=[],
        )
        patched = result["patched_treatment"]
        self.assertNotIn("新生儿脐部", patched)
        self.assertNotIn("硝酸银", patched)
        codes = {item["code"] for item in result["issues"]}
        self.assertNotIn("undertreated_umbilical_granulation_bleeding_mass", codes)
        self.assertNotIn("avoid_no_further_care_for_bleeding_mass", codes)

    def test_true_neonatal_umbilical_still_gated(self):
        text = "新生儿脐部有一鲜红湿润小肿块，轻触易出血"
        result = final_verifier(
            diagnosis="化脓性肉芽肿",
            examinations=["体格检查"],
            treatment_plan="通常自愈，无需进一步处置，观察即可。",
            official_diseases=self.official,
            examination_catalog=self.exam_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": text,
                "patient_text": text,
                "positive_findings": ["新生儿脐部病变", "湿润易出血肿块"],
                "candidate_diagnoses": ["化脓性肉芽肿"],
                "diagnosis_axes": [
                    {
                        "axis_id": "umbilical_granulation_or_vascular_lesion",
                        "treatment_risks": ["avoid_no_further_care_for_bleeding_mass"],
                        "evidence": ["新生儿", "脐部", "湿润易出血肿块"],
                        "candidate_official_names": ["化脓性肉芽肿"],
                    }
                ],
                "examination_results": {},
            },
            safety_profiles=[],
        )
        codes = {item["code"] for item in result["issues"]}
        self.assertTrue(
            codes
            & {
                "undertreated_umbilical_granulation_bleeding_mass",
                "avoid_no_further_care_for_bleeding_mass",
            }
        )
        self.assertIn("局部专科", result["patched_treatment"])


if __name__ == "__main__":
    unittest.main()
