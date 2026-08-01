"""Risk patches must not pollute unrelated diagnoses.

Mechanism only — no patient IDs or reference treatments.
"""

from __future__ import annotations

import unittest

from agent.legacy_orchestrator import (
    final_verifier,
    hypoxia_evidence_status,
    load_disease_catalog,
    load_examination_catalog,
    supported_axis_risks,
)


class TestRiskPatchScoping(unittest.TestCase):
    def setUp(self):
        self.official = [d for ds in load_disease_catalog().values() for d in ds]
        self.exam_catalog = load_examination_catalog()

    def test_vitals_blood_pressure_not_parsed_as_hypoxia(self):
        status = hypoxia_evidence_status(
            {
                "case_text": "咯血与呼吸困难",
                "patient_text": "咯血与呼吸困难",
                "examination_results": {
                    "生命体征": {
                        "status": "normal",
                        "result": {
                            "血压": "120/80 mmHg",
                            "心率": "60-100 bpm",
                            "呼吸频率": "12-20 次/分",
                            "体温": "36.5",
                        },
                    },
                    "脉搏血氧饱和度监测（SpO2）": {
                        "status": "normal",
                        "result": {"SpO2": "98%"},
                    },
                },
            }
        )
        self.assertNotEqual(status, "low")

    def test_true_low_spo2_still_detected(self):
        status = hypoxia_evidence_status(
            {
                "case_text": "呼吸困难",
                "examination_results": {
                    "脉搏血氧饱和度监测（SpO2）": {
                        "status": "abnormal",
                        "result": {"SpO2": "88%"},
                    }
                },
            }
        )
        self.assertEqual(status, "low")

    def test_unsupported_infection_steroid_risk_from_llm_is_dropped(self):
        risks = supported_axis_risks(
            ["infection_before_steroid"],
            axis_id="generic_llm_axis",
            evidence=["鼻塞"],
            candidates=["血管运动性鼻炎"],
            normalized_case_text="打喷嚏流涕眼痒",
            diagnosis="血管运动性鼻炎",
        )
        self.assertNotIn("infection_before_steroid", risks)

    def test_corneal_infection_axis_keeps_infection_steroid_risk(self):
        risks = supported_axis_risks(
            ["infection_before_steroid"],
            axis_id="corneal_infection_with_target_rash",
            evidence=["眼红畏光", "靶形皮疹"],
            candidates=["角膜炎"],
            normalized_case_text="眼红畏光角膜病变伴靶形皮疹",
            diagnosis="角膜炎",
        )
        self.assertIn("infection_before_steroid", risks)

    def test_allergic_rhinitis_plan_not_polluted_by_infection_steroid_patch(self):
        result = final_verifier(
            diagnosis="血管运动性鼻炎",
            examinations=["前鼻镜检查"],
            treatment_plan="鼻用糖皮质激素喷雾每日一次，口服抗组胺药，生理盐水冲洗。",
            official_diseases=self.official,
            examination_catalog=self.exam_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": "户外后打喷嚏水样涕眼痒",
                "patient_text": "户外后打喷嚏水样涕眼痒",
                "positive_findings": ["鼻黏膜苍白水肿"],
                "treatment_risks": ["infection_before_steroid"],
                "diagnosis_axes": [
                    {
                        "axis_id": "generic_allergic_rhinitis",
                        "treatment_risks": ["infection_before_steroid"],
                        "evidence": ["喷嚏", "水样涕"],
                        "candidate_official_names": ["血管运动性鼻炎"],
                    }
                ],
                "examination_results": {},
            },
            safety_profiles=[],
        )
        patched = result["patched_treatment"]
        self.assertNotIn("感染证据未闭合时不应常规使用局部或全身糖皮质激素", patched)
        self.assertNotIn("眼科/专科评估", patched)

    def test_mpa_plan_not_polluted_by_false_hypoxia_from_vitals(self):
        result = final_verifier(
            diagnosis="显微镜下多血管炎",
            examinations=["生命体征", "脉搏血氧饱和度监测（SpO2）", "抗中性粒细胞胞质抗体（ANCA）谱"],
            treatment_plan="甲泼尼龙联合环磷酰胺诱导缓解，规避磺胺，监测血常规肾功能。",
            official_diseases=self.official,
            examination_catalog=self.exam_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": "咯血尿色变黑乏力",
                "patient_text": "咯血尿色变黑乏力",
                "positive_findings": ["MPO-ANCA升高"],
                "treatment_risks": [],
                "diagnosis_axes": [],
                "examination_results": {
                    "生命体征": {
                        "status": "normal",
                        "result": {
                            "血压": "120/80 mmHg",
                            "心率": "72 bpm",
                            "呼吸频率": "16 次/分",
                        },
                    },
                    "脉搏血氧饱和度监测（SpO2）": {
                        "status": "normal",
                        "result": {"SpO2": "97%"},
                    },
                },
            },
            safety_profiles=[],
        )
        patched = result["patched_treatment"]
        self.assertNotIn("存在客观低氧", patched)
        codes = {item["code"] for item in result["issues"]}
        self.assertNotIn("hypoxia_missing_oxygen_goal", codes)


if __name__ == "__main__":
    unittest.main()
