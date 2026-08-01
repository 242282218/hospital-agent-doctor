"""Generalized offline axes from the fourteenth random 3-case batch."""

from __future__ import annotations

import unittest

from agent.legacy_orchestrator import (
    extract_intake_facts,
    final_verifier,
    load_disease_catalog,
    load_examination_catalog,
    open_coverage_gaps,
    select_diagnosis_axes,
    select_exam_plan,
    build_name_map,
    flatten_examination_catalog,
    load_knowledge_registry,
    inject_required_differentials,
    select_disease_candidates,
    should_block_final_for_coverage_gaps,
)


class FourteenthRoundOfflineAxesTest(unittest.TestCase):
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

    def _plan(self, case_state, max_items=8):
        candidates = inject_required_differentials(
            select_disease_candidates(case_state, self.disease_catalog, limit=12),
            case_state=case_state,
            disease_catalog=self.disease_catalog,
        )
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=candidates,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            diagnosis_exam_profiles=self.exam_profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=max_items,
        )
        return plan, candidates, axes

    def neonatal_chd_case(self, with_echo: bool = False) -> dict:
        case = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孩子出生后6小时呼吸急促，嘴唇发青，吸氧改善不明显，喂奶时加重、易出汗、吃奶少、尿少。",
                }
            ],
            "ordered_examinations": ["体格检查", "生命体征", "脉搏血氧饱和度监测（SpO2）"],
            "examination_results": {
                "体格检查": {"status": "abnormal", "result": {"心率": "182次/分"}},
                "生命体征": {"status": "abnormal", "result": {"呼吸": "快"}},
                "脉搏血氧饱和度监测（SpO2）": {"status": "normal", "result": {"SpO2": "96%"}},
            },
        }
        if with_echo:
            case["ordered_examinations"].append("超声心动图")
            case["examination_results"]["超声心动图"] = {
                "status": "abnormal",
                "result": {"结构": "先天性结构异常待分型"},
            }
        return case

    def test_neonatal_chd_opens_axis_and_requires_echo(self):
        case_state = self.neonatal_chd_case(with_echo=False)
        plan, candidates, axes = self._plan(case_state)
        self.assertIn(
            "infant_congenital_structural_heart_disease",
            {item["axis_id"] for item in axes},
        )
        self.assertIn("先天性心脏病", {item["disease"] for item in candidates})
        self.assertTrue(any("超声心动图" in exam for exam in plan["examinations"]))
        self.assertIn("infant_chd_echocardiography", {item["gap_id"] for item in open_coverage_gaps(case_state)})
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))

    def test_vitals_and_spo2_do_not_close_infant_chd_coverage(self):
        case_state = self.neonatal_chd_case(with_echo=False)
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))

    def test_cyanotic_infant_blocks_default_pda_closure_path(self):
        case_state = self.neonatal_chd_case(with_echo=True)
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = final_verifier(
            diagnosis="先天性心脏病",
            examinations=case_state["ordered_examinations"],
            treatment_plan="按左向右分流心衰处理，使用呋塞米；若动脉导管未闭给予吲哚美辛关闭导管。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": case_state["chat_history"][0]["text"],
                "patient_text": case_state["chat_history"][0]["text"],
                "positive_findings": ["新生儿", "发绀", "呼吸急促"],
                "candidate_diagnoses": ["先天性心脏病"],
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )
        self.assertIn("duct_dependent_cyanosis_pda_closure_risk", {item["code"] for item in result["issues"]})
        self.assertNotIn("给予吲哚美辛关闭导管", result["patched_treatment"])
        self.assertNotIn("关闭导管。", result["patched_treatment"])
        self.assertTrue(
            any(marker in result["patched_treatment"] for marker in ["导管开放", "前列地尔", "专科", "维持导管"])
        )

        verified = final_verifier(
            diagnosis="先天性心脏病",
            examinations=case_state["ordered_examinations"],
            treatment_plan=result["patched_treatment"],
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": case_state["chat_history"][0]["text"],
                "patient_text": case_state["chat_history"][0]["text"],
                "positive_findings": ["新生儿", "发绀", "呼吸急促"],
                "candidate_diagnoses": ["先天性心脏病"],
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )

        self.assertNotIn(
            "duct_dependent_cyanosis_pda_closure_risk",
            {item["code"] for item in verified["issues"]},
        )

    def test_cyanotic_infant_pda_gate_is_independent_of_final_diagnosis_label(self):
        case_state = self.neonatal_chd_case(with_echo=True)
        case_features = {
            "case_text": case_state["chat_history"][0]["text"],
            "patient_text": case_state["chat_history"][0]["text"],
            "positive_findings": ["新生儿", "发绀", "呼吸急促"],
            "candidate_diagnoses": ["先天性风疹综合征", "动脉导管未闭"],
            "diagnosis_axes": [{
                "axis_id": "congenital_rubella",
                "evidence": ["宫内病毒暴露", "风疹IgM阳性"],
                "candidate_official_names": ["先天性风疹综合征"],
            }],
            "examination_results": case_state["examination_results"],
        }
        treatment = (
            "在严密监测下，可考虑使用前列腺素E1抑制剂（如吲哚美辛）"
            "促进动脉导管闭合，但需评估肾功能及出血风险。"
        )

        result = final_verifier(
            diagnosis="先天性风疹综合征",
            examinations=case_state["ordered_examinations"],
            treatment_plan=treatment,
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features=case_features,
            safety_profiles=[],
        )

        self.assertIn(
            "duct_dependent_cyanosis_pda_closure_risk",
            {item["code"] for item in result["issues"]},
        )
        self.assertNotIn("吲哚美辛", result["patched_treatment"])
        verified = final_verifier(
            diagnosis="先天性风疹综合征",
            examinations=case_state["ordered_examinations"],
            treatment_plan=result["patched_treatment"],
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features=case_features,
            safety_profiles=[],
        )
        self.assertNotIn(
            "duct_dependent_cyanosis_pda_closure_risk",
            {item["code"] for item in verified["issues"]},
        )

    def test_postop_chylothorax_negative_ultrasound_not_closed(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "两岁孩子一周前做了胸腔和纵隔淋巴结清扫手术，现在突然呼吸费力、活动减少。",
                }
            ],
            "ordered_examinations": ["胸部超声"],
            "examination_results": {
                "胸部超声": {"status": "normal", "result": {"胸腔积液": "双侧未见"}},
            },
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        self.assertIn("postop_chylothorax_or_pleural_effusion", {item["axis_id"] for item in axes})
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}
        self.assertIn("postop_chest_effusion_further_imaging", gap_ids)
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))

    def test_postop_chylothorax_needs_drainage_or_specialty_path(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "纵隔淋巴结清扫术后出现呼吸费力。",
                }
            ],
            "ordered_examinations": ["胸部超声", "胸部X线检查（CXR）"],
            "examination_results": {
                "胸部超声": {"status": "normal", "result": {"胸腔积液": "未见"}},
                "胸部X线检查（CXR）": {"status": "abnormal", "result": {"积液": "可疑"}},
            },
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = final_verifier(
            diagnosis="乳糜胸",
            examinations=["胸部超声", "胸部X线检查（CXR）"],
            treatment_plan="超声未见明显积液，继续观察，暂不处理。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": case_state["chat_history"][0]["text"],
                "patient_text": case_state["chat_history"][0]["text"],
                "positive_findings": ["术后呼吸费力"],
                "candidate_diagnoses": ["乳糜胸"],
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )
        self.assertIn("postop_chylothorax_observation_only", {item["code"] for item in result["issues"]})
        self.assertTrue(
            any(marker in result["patched_treatment"] for marker in ["引流", "胸外科", "禁食", "专科"])
        )


if __name__ == "__main__":
    unittest.main()
