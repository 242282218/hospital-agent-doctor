"""Acute pulmonary-renal coverage and glucocorticoid induction regressions.

Clues only (no Patient ID / expected packs / reference treatment text):
- acute cough with bloody sputum progressing to massive hemoptysis
- dark urine, oliguria, ankle swelling, systemic symptoms
- MPA diagnosis with CYC/rituximab but no glucocorticoid induction
"""

from __future__ import annotations

import unittest

from agent.legacy_orchestrator import (
    apply_coverage_gap_action_gate,
    extract_intake_facts,
    final_verifier,
    has_pulmonary_renal_vasculitis_pattern,
    inject_required_differentials,
    intake_facts_text,
    load_disease_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    normalize_name,
    open_coverage_gaps,
    select_diagnosis_axes,
    select_disease_candidates,
    select_exam_plan,
    should_block_final_for_coverage_gaps,
    build_name_map,
    flatten_examination_catalog,
)


ACUTE_PULMONARY_RENAL = (
    "3天前乏力和低热，接着咳嗽带血丝，很快变成大量咯血，活动后呼吸困难加重。"
    "同时尿色变黑、尿量减少，脚踝肿了，全身无力、肌肉和关节都痛。"
)


class TestPulmonaryRenalCoverageAndInduction(unittest.TestCase):
    def setUp(self):
        self.disease_catalog = load_disease_catalog()
        self.examination_catalog = load_examination_catalog()
        self.item_name_map = build_name_map(flatten_examination_catalog(self.examination_catalog))
        knowledge = load_knowledge_registry()
        self.exam_profiles = knowledge["diagnosis_exam_profiles"]
        self.exam_intent_rules = knowledge["exam_intent_map"]
        self.official = [d for ds in self.disease_catalog.values() for d in ds]

    def test_acute_colloquial_opens_pulmonary_renal_axis_and_chest_gap(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": ACUTE_PULMONARY_RENAL}],
            "ordered_examinations": [
                "体格检查",
                "生命体征",
                "脉搏血氧饱和度监测（SpO2）",
                "抗中性粒细胞胞质抗体（ANCA）谱",
                "尿液分析（UA）",
            ],
        }
        facts = extract_intake_facts(case_state)
        facts_text = normalize_name(intake_facts_text(facts))
        self.assertTrue(has_pulmonary_renal_vasculitis_pattern(facts_text))
        # Sputum blood-streak must not be misread as hematuria via bare 血丝 alone.
        labels = {item["label"] for item in facts.get("symptom_clusters", [])}
        if "血尿" in labels:
            hematuria = next(item for item in facts["symptom_clusters"] if item["label"] == "血尿")
            self.assertNotEqual(hematuria.get("evidence"), "血丝")

        axes = select_diagnosis_axes(facts)
        self.assertIn(
            "pulmonary_renal_vasculitis_vs_infection",
            {item["axis_id"] for item in axes},
        )
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}
        self.assertIn("chest_imaging", gap_ids)
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))
        gated = apply_coverage_gap_action_gate(action="final_diagnosis", case_state=case_state)
        self.assertEqual(gated["action"], "order_examination")

        candidates = inject_required_differentials(
            select_disease_candidates(case_state, self.disease_catalog, limit=12),
            case_state=case_state,
            disease_catalog=self.disease_catalog,
        )
        plan = select_exam_plan(
            case_state={**case_state, "ordered_examinations": []},
            disease_candidates=candidates,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            diagnosis_exam_profiles=self.exam_profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=8,
        )
        self.assertIn("胸部X线检查（CXR）", plan["examinations"])
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", plan["examinations"])

    def test_mpa_induction_without_glucocorticoid_is_patched(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": ACUTE_PULMONARY_RENAL}],
            "ordered_examinations": [
                "抗中性粒细胞胞质抗体（ANCA）谱",
                "尿液分析（UA）",
                "胸部X线检查（CXR）",
            ],
            "examination_results": {
                "抗中性粒细胞胞质抗体（ANCA）谱": {
                    "status": "abnormal",
                    "result": {"MPO-ANCA": "128 U/mL", "PR3-ANCA": "阴性"},
                }
            },
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = final_verifier(
            diagnosis="显微镜下多血管炎",
            examinations=case_state["ordered_examinations"],
            treatment_plan=(
                "联合环磷酰胺静脉滴注或口服，或选用利妥昔单抗。"
                "绝对卧床监测血红蛋白；磺胺过敏禁用复方新诺明，改用阿托伐醌。"
            ),
            official_diseases=self.official,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": ACUTE_PULMONARY_RENAL,
                "patient_text": ACUTE_PULMONARY_RENAL,
                "positive_findings": ["MPO-ANCA升高", "咯血", "尿色变黑"],
                "candidate_diagnoses": ["显微镜下多血管炎"],
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )
        codes = {item["code"] for item in result["issues"]}
        self.assertIn("anca_vasculitis_missing_glucocorticoid_induction", codes)
        patched = result["patched_treatment"]
        self.assertTrue(
            any(
                marker in patched
                for marker in ["甲泼尼龙", "激素冲击", "大剂量糖皮质", "糖皮质激素"]
            )
        )

    def test_mpa_plan_with_pulse_steroid_is_not_flagged(self):
        result = final_verifier(
            diagnosis="显微镜下多血管炎",
            examinations=["抗中性粒细胞胞质抗体（ANCA）谱", "胸部X线检查（CXR）", "尿液分析（UA）"],
            treatment_plan=(
                "立即大剂量甲泼尼龙冲击联合环磷酰胺诱导缓解，监测感染与血象；"
                "磺胺过敏者避免复方新诺明，改用阿托伐醌预防PCP。"
            ),
            official_diseases=self.official,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": ACUTE_PULMONARY_RENAL,
                "patient_text": ACUTE_PULMONARY_RENAL,
                "positive_findings": ["咯血", "MPO-ANCA"],
                "diagnosis_axes": select_diagnosis_axes(
                    extract_intake_facts(
                        {"chat_history": [{"from": "patient", "text": ACUTE_PULMONARY_RENAL}]}
                    )
                ),
                "examination_results": {},
            },
            safety_profiles=[],
        )
        codes = {item["code"] for item in result["issues"]}
        self.assertNotIn("anca_vasculitis_missing_glucocorticoid_induction", codes)

    def test_simple_cough_without_renal_systemic_not_forced(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "感冒后咳嗽两天，偶尔嗓子干，没有咯血，也没有腿肿。"}
            ],
            "ordered_examinations": [],
        }
        facts_text = normalize_name(intake_facts_text(extract_intake_facts(case_state)))
        self.assertFalse(has_pulmonary_renal_vasculitis_pattern(facts_text))
        self.assertNotIn(
            "pulmonary_renal_vasculitis_vs_infection",
            {item["axis_id"] for item in select_diagnosis_axes(extract_intake_facts(case_state))},
        )


if __name__ == "__main__":
    unittest.main()
