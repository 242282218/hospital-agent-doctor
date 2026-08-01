import asyncio
import unittest
from unittest.mock import AsyncMock

from agent import legacy_orchestrator

from agent.legacy_orchestrator import (
    apply_treatment_safety,
    build_name_map,
    extract_case_features,
    final_verifier,
    flatten_examination_catalog,
    normalize_diagnosis,
    validate_treatment_review,
)


class AgentOptimizationModulesTest(unittest.TestCase):
    def test_extract_case_features_keeps_only_planning_relevant_fields(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "孩子高热，湿疹处出现成簇水泡，精神很差。"},
                {"from": "patient", "text": "最近一直在吃泼尼松。"},
            ]
        }
        disease_candidates = [{"disease": "卡波西水痘样疹", "score": 30}]

        features = extract_case_features(case_state, disease_candidates)

        self.assertIn("高热", features["positive_findings"])
        self.assertIn("湿疹", features["positive_findings"])
        self.assertIn("泼尼松", features["medications"])
        self.assertIn("全身激素", features["immunosuppression"])
        self.assertIn("精神差", features["red_flags"])
        self.assertEqual(["卡波西水痘样疹"], features["candidate_diagnoses"])

    def test_normalize_diagnosis_uses_verified_alias_only(self):
        official_diseases = ["卡波西水痘样疹", "水痘"]
        alias_rules = [
            {
                "status": "verified",
                "input": ["Kaposi varicelliform eruption", "eczema herpeticum"],
                "output": "卡波西水痘样疹",
            },
            {
                "status": "candidate",
                "input": ["疱疹性湿疹"],
                "output": "卡波西水痘样疹",
            },
        ]

        normalized = normalize_diagnosis(
            "Kaposi varicelliform eruption",
            official_diseases=official_diseases,
            alias_rules=alias_rules,
            disease_candidates=[{"disease": "水痘"}],
        )
        candidate_only = normalize_diagnosis(
            "疱疹性湿疹",
            official_diseases=official_diseases,
            alias_rules=alias_rules,
            disease_candidates=[{"disease": "水痘"}],
        )

        self.assertEqual("卡波西水痘样疹", normalized["normalized_diagnosis"])
        self.assertEqual("alias_map", normalized["source"])
        self.assertEqual("", candidate_only["normalized_diagnosis"])
        self.assertEqual(["水痘"], candidate_only["fallback_candidates"])

    def test_normalize_diagnosis_refines_generic_infection_when_specific_candidate_exists(self):
        normalized = normalize_diagnosis(
            "细菌感染",
            official_diseases=["细菌感染", "急性细菌性前列腺炎"],
            alias_rules=[],
            disease_candidates=[
                {"disease": "细菌感染", "score": 36},
                {"disease": "急性细菌性前列腺炎", "score": 20},
            ],
        )

        self.assertEqual("急性细菌性前列腺炎", normalized["normalized_diagnosis"])
        self.assertEqual("specific_candidate", normalized["source"])

    def test_normalize_diagnosis_does_not_refine_generic_to_unrelated_candidate(self):
        normalized = normalize_diagnosis(
            "细菌感染",
            official_diseases=["细菌感染", "卵睾性别发育异常（Ovotesticular DSD）"],
            alias_rules=[],
            disease_candidates=[
                {"disease": "卵睾性别发育异常（Ovotesticular DSD）", "score": 10},
                {"disease": "细菌感染", "score": 2},
            ],
        )

        self.assertEqual("细菌感染", normalized["normalized_diagnosis"])
        self.assertEqual("official_catalog", normalized["source"])

    def test_treatment_safety_patches_missing_high_risk_infection_goals(self):
        treatment_plan = "建议口服阿昔洛韦，并外用抗生素软膏，居家观察。"
        case_features = {
            "positive_findings": ["高热", "湿疹", "疱疹样皮损"],
            "immunosuppression": ["全身激素"],
            "red_flags": ["精神差"],
        }
        safety_profiles = [
            {
                "status": "verified",
                "id": "severe_hsv_immunosuppressed",
                "risk_factors": ["全身激素", "高热", "疱疹样皮损"],
                "treatment_goals": ["住院", "静脉阿昔洛韦", "静脉抗生素", "调整免疫抑制"],
                "patch_templates": {
                    "hospitalization": "存在免疫抑制和全身症状，应住院或急诊/专科监测。",
                    "iv_acyclovir": "应优先使用静脉阿昔洛韦抗病毒治疗。",
                    "iv_antibiotic": "疑似继发细菌感染时应给予静脉抗生素覆盖，并根据培养结果调整。",
                    "immunosuppression": "应与开方医生协作暂停或调整全身糖皮质激素/免疫抑制药物。",
                },
            }
        ]

        result = apply_treatment_safety(
            treatment_plan,
            diagnosis="卡波西水痘样疹",
            case_features=case_features,
            safety_profiles=safety_profiles,
        )

        self.assertTrue(result["patched"])
        self.assertIn("静脉阿昔洛韦", result["treatment_plan"])
        self.assertIn("静脉抗生素", result["treatment_plan"])
        self.assertIn("暂停或调整", result["treatment_plan"])

    def test_final_verifier_blocks_non_leaf_exam_and_records_treatment_gap(self):
        examination_catalog = {
            "体格检查": ["体格检查", "皮肤检查"],
            "实验室检查-血液": ["全血细胞计数（CBC）"],
        }
        case_features = {
            "positive_findings": ["高热", "疱疹样皮损"],
            "immunosuppression": ["全身激素"],
            "red_flags": ["精神差"],
        }
        safety_profiles = [
            {
                "status": "verified",
                "risk_factors": ["全身激素", "高热", "疱疹样皮损"],
                "treatment_goals": ["静脉阿昔洛韦"],
                "patch_templates": {"iv_acyclovir": "应优先使用静脉阿昔洛韦抗病毒治疗。"},
            }
        ]

        result = final_verifier(
            diagnosis="卡波西水痘样疹",
            examinations=["实验室检查-血液"],
            treatment_plan="建议口服阿昔洛韦。",
            official_diseases=["卡波西水痘样疹"],
            examination_catalog=examination_catalog,
            exam_plan_trace=[],
            case_features=case_features,
            safety_profiles=safety_profiles,
        )

        self.assertFalse(result["passed"])
        self.assertIn("invalid_exam_name", [issue["code"] for issue in result["issues"]])
        self.assertIn("missing_treatment_goal", [issue["code"] for issue in result["issues"]])
        self.assertIn("静脉阿昔洛韦", result["patched_treatment"])

    def test_final_verifier_flags_generic_diagnosis_when_specific_candidate_exists(self):
        result = final_verifier(
            diagnosis="细菌感染",
            examinations=["尿液分析（UA）"],
            treatment_plan="给予抗感染治疗。",
            official_diseases=["细菌感染", "急性细菌性前列腺炎"],
            examination_catalog={"实验室检查-尿液": ["尿液分析（UA）"]},
            exam_plan_trace=[],
            case_features={"candidate_diagnoses": ["细菌感染", "急性细菌性前列腺炎"]},
            safety_profiles=[],
        )

        self.assertFalse(result["passed"])
        self.assertIn("underspecified_diagnosis", [issue["code"] for issue in result["issues"]])

    def test_treatment_review_rejects_invented_evidence(self):
        original = "建议对症治疗并根据复诊结果调整。"
        review = {
            "treatment_plan": "患者已怀孕且存在肾衰竭，应按高危妊娠调整全部用药。",
            "evidence_refs": ["心悸"],
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state={"chat_history": [{"from": "patient", "text": "偶发心悸。"}]},
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(original, result)

    def test_treatment_review_rejects_unrelated_ref_for_invented_comorbidity(self):
        original = "建议控制心室率并根据复诊结果调整。"
        review = {
            "treatment_plan": "患者合并糖尿病和高血压，应按基础病调整全部治疗。",
            "evidence_refs": ["活动后心悸"],
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state={"chat_history": [{"from": "patient", "text": "仅有活动后心悸。"}]},
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(original, result)

    def test_treatment_review_rejects_invented_current_medication_and_stop_action(self):
        original = "建议控制心室率并根据复诊结果调整。"
        review = {
            "treatment_plan": "患者正在服用华法林，应立即停药并调整抗凝方案。",
            "evidence_refs": ["活动后心悸"],
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state={
                "chat_history": [
                    {
                        "from": "patient",
                        "text": "仅有活动后心悸，父亲长期服用华法林。",
                    }
                ]
            },
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(original, result)

    def test_treatment_review_rejects_unsupported_medication_discontinuation(self):
        original = "建议控制心室率并根据复诊结果调整。"
        review = {
            "treatment_plan": "立即停用华法林并调整抗凝方案。",
            "evidence_refs": ["活动后心悸"],
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state={"chat_history": [{"from": "patient", "text": "仅有活动后心悸。"}]},
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(original, result)

    def test_treatment_review_rejects_patient_medication_without_temporal_adverb(self):
        original = "建议控制心室率并根据复诊结果调整。"
        review = {
            "treatment_plan": "患者服用华法林并建议增加剂量。",
            "evidence_refs": ["活动后心悸"],
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state={"chat_history": [{"from": "patient", "text": "仅有活动后心悸。"}]},
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(original, result)

    def test_treatment_review_rejects_drug_first_discontinuation(self):
        original = "建议控制心室率并根据复诊结果调整。"
        review = {
            "treatment_plan": "华法林应立即停用并调整抗凝方案。",
            "evidence_refs": ["活动后心悸"],
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state={"chat_history": [{"from": "patient", "text": "仅有活动后心悸。"}]},
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(original, result)

    def test_treatment_review_rejects_unsupported_continuation_and_dose_change(self):
        original = "建议控制心室率并根据复诊结果调整。"
        plans = [
            "继续华法林抗凝并将剂量加倍。",
            "建议增加华法林剂量。",
        ]

        for plan in plans:
            with self.subTest(plan=plan):
                result = validate_treatment_review(
                    {"treatment_plan": plan, "evidence_refs": ["活动后心悸"]},
                    original_treatment_plan=original,
                    case_state={"chat_history": [{"from": "patient", "text": "仅有活动后心悸。"}]},
                    diagnosis="心房颤动",
                    verifier_issues=[],
                )

                self.assertEqual(original, result)

    def test_treatment_review_allows_supported_medication_continuation(self):
        original = "建议控制心室率并根据复诊结果调整。"
        review = {
            "treatment_plan": "继续华法林抗凝，剂量由专科复核后调整。",
            "evidence_refs": ["目前服用华法林"],
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state={"chat_history": [{"from": "patient", "text": "我目前服用华法林。"}]},
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(review["treatment_plan"], result)

    def test_treatment_review_allows_removing_drug_present_in_original_plan(self):
        original = "建议使用利伐沙班抗凝并密切复诊。"
        review = {
            "treatment_plan": "鉴于反复鼻出血，应停用利伐沙班并由专科评估抗凝方案。",
            "evidence_refs": ["反复鼻出血"],
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state={"chat_history": [{"from": "patient", "text": "近期反复鼻出血。"}]},
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(review["treatment_plan"], result)

    def test_treatment_review_accepts_directly_supported_sensitive_facts(self):
        original = "建议控制心室率并根据复诊结果调整。"
        review = {
            "treatment_plan": "患者合并糖尿病且正在服用华法林，应结合现用药复核抗凝方案。",
            "evidence_refs": ["既往确诊糖尿病", "正在服用华法林"],
        }
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "我既往确诊糖尿病，目前正在服用华法林。",
                }
            ]
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(review["treatment_plan"], result)

    def test_treatment_review_accepts_refs_grounded_in_exam_evidence(self):
        original = "建议控制心室率并密切监测。"
        review = {
            "treatment_plan": "LVEF明显降低，应选择适用于心衰的心室率控制方案。",
            "evidence_refs": ["LVEF 28%"],
        }
        case_state = {
            "chat_history": [{"from": "patient", "text": "活动后气短伴心悸。"}],
            "examination_results": {
                "超声心动图": {"result": {"左心室射血分数": "LVEF 28%"}}
            },
        }

        result = validate_treatment_review(
            review,
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="心房颤动",
            verifier_issues=[],
        )

        self.assertEqual(review["treatment_plan"], result)

    def test_treatment_review_decision_accepts_supported_revision_with_hashes(self):
        original = "建议控制心室率并根据复诊结果调整。"
        revised = "继续华法林抗凝，剂量由专科复核后调整。"
        case_state = {
            "chat_history": [{"from": "patient", "text": "我目前服用华法林。"}]
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="心房颤动",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        decision = legacy_orchestrator.decide_treatment_review(
            {"treatment_plan": revised, "evidence_refs": [catalog[0]["id"]]},
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="心房颤动",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("accepted", decision["status"])
        self.assertTrue(decision["accepted"])
        self.assertEqual(revised, decision["treatment_plan"])
        self.assertEqual([], decision["failed_refs"])
        self.assertRegex(decision["before_hash"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(decision["after_hash"], r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(decision["before_hash"], decision["after_hash"])

    def test_treatment_review_decision_records_unknown_reference_rejection(self):
        original = "建议控制心室率并根据复诊结果调整。"
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "treatment_plan": "患者已怀孕，应调整全部用药。",
                "evidence_refs": ["unknown:fact"],
            },
            original_treatment_plan=original,
            case_state={"chat_history": [{"from": "patient", "text": "偶发心悸。"}]},
            diagnosis="心房颤动",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        self.assertEqual("rejected", decision["status"])
        self.assertFalse(decision["accepted"])
        self.assertEqual(original, decision["treatment_plan"])
        self.assertEqual(["unknown:fact"], decision["failed_refs"])
        self.assertIn("unknown_evidence_ref", decision["reason_codes"])
        self.assertEqual(decision["before_hash"], decision["after_hash"])

    def test_treatment_review_decision_records_unchanged_plan(self):
        original = "建议控制心室率并根据复诊结果调整。"
        decision = legacy_orchestrator.decide_treatment_review(
            {"treatment_plan": original, "evidence_refs": []},
            original_treatment_plan=original,
            case_state={},
            diagnosis="心房颤动",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        self.assertEqual("unchanged", decision["status"])
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["before_hash"], decision["after_hash"])

    def test_treatment_review_accepts_negative_exam_evidence_to_remove_eye_drug(self):
        original = "建议使用奥洛他定滴眼液治疗。密切观察眼红和分泌物。"
        case_state = {
            "chat_history": [{"from": "patient", "text": "孩子频繁眨眼并清嗓子。"}],
            "examination_results": {
                "外眼检查": {
                    "status": "normal",
                    "result": {"结膜": "结膜清晰，无充血或分泌物"},
                }
            },
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="结膜炎",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        negative_exam = next(
            item
            for item in catalog
            if item["source"] == "exam" and item["polarity"] == "negative"
        )
        review = {
            "edits": [
                {
                    "edit_id": "remove_unsupported_eye_drug",
                    "operation": "delete",
                    "target": "建议使用奥洛他定滴眼液治疗。",
                    "replacement": "",
                    "evidence_refs": [negative_exam["id"]],
                }
            ]
        }

        decision = legacy_orchestrator.decide_treatment_review(
            review,
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="结膜炎",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("accepted", decision["status"])
        self.assertNotIn("奥洛他定", decision["treatment_plan"])
        self.assertEqual(["remove_unsupported_eye_drug"], decision["accepted_edit_ids"])

    def test_treatment_review_splits_mixed_patient_fact_polarity(self):
        original = "建议对症处理并复诊。"
        revised = "继续按需服用奥美拉唑，并由专科复核疗程。"
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "没有药物过敏；目前按需服用奥美拉唑。",
                }
            ]
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="胃食管反流病",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        patient_entries = [item for item in catalog if item["source"] == "patient"]
        self.assertEqual(["negative", "positive"], [item["polarity"] for item in patient_entries])
        medication_evidence = next(
            item for item in patient_entries if "奥美拉唑" in item["text"]
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "treatment_plan": revised,
                "evidence_refs": [medication_evidence["id"]],
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="胃食管反流病",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("accepted", decision["status"])
        self.assertEqual(revised, decision["treatment_plan"])

    def test_treatment_review_atomic_edits_accept_only_supported_edit(self):
        original = "建议使用奥洛他定滴眼液治疗。密切观察眼红和分泌物。"
        case_state = {
            "chat_history": [{"from": "patient", "text": "孩子频繁眨眼并清嗓子。"}],
            "examination_results": {
                "外眼检查": {
                    "status": "normal",
                    "result": {"结膜": "结膜清晰，无充血或分泌物"},
                }
            },
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="结膜炎",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        negative_exam = next(item for item in catalog if item["source"] == "exam")
        review = {
            "edits": [
                {
                    "edit_id": "remove_eye_drug",
                    "operation": "delete",
                    "target": "建议使用奥洛他定滴眼液治疗。",
                    "replacement": "",
                    "evidence_refs": [negative_exam["id"]],
                },
                {
                    "edit_id": "invent_pregnancy",
                    "operation": "append",
                    "target": "",
                    "replacement": "患者已怀孕，应调整全部用药。",
                    "evidence_refs": ["unknown:pregnancy"],
                },
            ]
        }

        decision = legacy_orchestrator.decide_treatment_review(
            review,
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="结膜炎",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("partial", decision["status"])
        self.assertNotIn("奥洛他定", decision["treatment_plan"])
        self.assertNotIn("怀孕", decision["treatment_plan"])
        self.assertEqual(["remove_eye_drug"], decision["accepted_edit_ids"])
        self.assertEqual("invent_pregnancy", decision["rejected_edits"][0]["edit_id"])
        self.assertEqual(["unknown:pregnancy"], decision["failed_refs"])

    def test_atomic_treatment_review_requires_exact_evidence_ids(self):
        original = "建议对症处理并复诊。"
        case_state = {"chat_history": [{"from": "patient", "text": "症状持续。"}]}
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "free_text_ref",
                        "operation": "replace",
                        "target": original,
                        "replacement": "建议支持治疗并复诊。",
                        "evidence_refs": ["症状持续"],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="病毒感染",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("unknown_evidence_ref", decision["reason_codes"])

    def test_atomic_sensitive_claim_uses_only_edit_referenced_evidence(self):
        original = "建议对症处理并复诊。"
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "我已怀孕。"},
                {"from": "patient", "text": "没有药物过敏。"},
            ]
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        negative_allergy = next(
            item for item in catalog if item["polarity"] == "negative"
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "wrong_bound_fact",
                        "operation": "append",
                        "target": "",
                        "replacement": "患者已怀孕，应调整全部用药。",
                        "evidence_refs": [negative_allergy["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("unsupported_pregnancy", decision["reason_codes"])

    def test_negative_evidence_cannot_support_additive_treatment_edit(self):
        original = "建议观察并复诊。"
        case_state = {
            "examination_results": {
                "外眼检查": {
                    "status": "normal",
                    "result": {"结膜": "结膜清晰，无充血或分泌物"},
                }
            }
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="结膜炎",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        negative_exam = next(item for item in catalog if item["source"] == "exam")
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "add_eye_drug_from_negative",
                        "operation": "append",
                        "target": "",
                        "replacement": "使用奥洛他定滴眼液治疗。",
                        "evidence_refs": [negative_exam["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="结膜炎",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("negative_evidence_requires_delete", decision["reason_codes"])

    def test_missing_axis_evidence_can_only_add_evidence_closure(self):
        original = "建议对症处理并复诊。"
        axes = [
            {
                "axis_id": "lung_mass_axis",
                "source": "llm",
                "validated": True,
                "evidence": ["右上叶肿块", "血丝痰"],
                "missing_evidence": ["胸部CT和病理活检"],
            }
        ]
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state={},
            diagnosis="肺癌",
            diagnosis_axes=axes,
            verifier_issues=[],
        )
        missing = next(item for item in catalog if item["polarity"] == "missing")
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "close_missing_evidence",
                        "operation": "append",
                        "target": "",
                        "replacement": "尽快完善胸部CT检查和病理活检。",
                        "evidence_refs": [missing["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state={},
            diagnosis="肺癌",
            diagnosis_axes=axes,
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("accepted", decision["status"])

    def test_missing_axis_evidence_cannot_support_treatment_action(self):
        original = "建议对症处理并复诊。"
        axes = [
            {
                "axis_id": "lung_mass_axis",
                "source": "llm",
                "validated": True,
                "missing_evidence": ["胸部CT和病理活检"],
            }
        ]
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state={},
            diagnosis="肺癌",
            diagnosis_axes=axes,
            verifier_issues=[],
        )
        missing = next(item for item in catalog if item["polarity"] == "missing")
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "premature_treatment",
                        "operation": "append",
                        "target": "",
                        "replacement": "立即开始含铂化疗。",
                        "evidence_refs": [missing["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state={},
            diagnosis="肺癌",
            diagnosis_axes=axes,
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("missing_evidence_requires_closure", decision["reason_codes"])

    def test_final_diagnosis_alone_cannot_assert_sensitive_organ_fact(self):
        original = "建议由专科评估用药。"
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "renal_dose_adjustment",
                        "operation": "append",
                        "target": "",
                        "replacement": "患者存在肾功能不全，应按肾功能调整剂量。",
                        "evidence_refs": ["diagnosis:final"],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state={},
            diagnosis="肾功能不全",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("diagnosis_context_only", decision["reason_codes"])

    def test_validated_axis_can_support_sensitive_treatment_fact(self):
        original = "建议由专科评估用药。"
        axes = [
            {
                "axis_id": "renal_impairment_axis",
                "source": "llm",
                "validated": True,
                "evidence": ["患者存在肾功能不全"],
            }
        ]
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state={},
            diagnosis="病毒感染",
            diagnosis_axes=axes,
            verifier_issues=[],
        )
        axis_evidence = next(item for item in catalog if item["source"] == "axis")
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "renal_dose_adjustment",
                        "operation": "append",
                        "target": "",
                        "replacement": "患者存在肾功能不全，应按肾功能调整剂量。",
                        "evidence_refs": [axis_evidence["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state={},
            diagnosis="病毒感染",
            diagnosis_axes=axes,
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("accepted", decision["status"])

    def test_atomic_review_rejects_new_patient_facts_from_unrelated_evidence(self):
        cases = [
            ("患者对头孢过敏，应避免头孢类药。", "unsupported_drug_allergy"),
            ("患者合并冠心病，应调整治疗。", "unsupported_comorbidity"),
            ("患者35岁，应按年龄调整剂量。", "unsupported_age"),
            ("患者六岁，应按儿童剂量。", "unsupported_age"),
            (
                "患者的常规用药包括华法林，应立即停药。",
                "unsupported_current_medication",
            ),
        ]
        original = "建议对症处理并复诊。"
        case_state = {"chat_history": [{"from": "patient", "text": "头痛两天。"}]}
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        for replacement, reason in cases:
            with self.subTest(reason=reason):
                decision = legacy_orchestrator.decide_treatment_review(
                    {
                        "edits": [
                            {
                                "edit_id": reason,
                                "operation": "append",
                                "target": "",
                                "replacement": replacement,
                                "evidence_refs": [catalog[0]["id"]],
                            }
                        ]
                    },
                    original_treatment_plan=original,
                    case_state=case_state,
                    diagnosis="偏头痛",
                    diagnosis_axes=[],
                    verifier_issues=[],
                    evidence_catalog=catalog,
                )

                self.assertEqual("rejected", decision["status"])
                self.assertIn(reason, decision["reason_codes"])

    def test_family_fact_cannot_be_rebound_to_patient(self):
        original = "建议对症处理并复诊。"
        case_state = {
            "chat_history": [{"from": "patient", "text": "姐姐目前怀孕。"}]
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "rebind_family_pregnancy",
                        "operation": "append",
                        "target": "",
                        "replacement": "患者已怀孕，应调整全部用药。",
                        "evidence_refs": [catalog[0]["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("unsupported_pregnancy", decision["reason_codes"])

    def test_unlisted_relative_facts_cannot_be_rebound_to_patient(self):
        cases = [
            ("我姨妈已怀孕。", "患者已怀孕，应调整全部用药。", "unsupported_pregnancy"),
            ("我姨妈确诊冠心病。", "患者合并冠心病，应调整治疗。", "unsupported_comorbidity"),
            ("我姨妈65岁。", "患者65岁，应按年龄调整剂量。", "unsupported_age"),
        ]
        original = "建议对症处理并复诊。"
        for evidence_text, replacement, reason in cases:
            with self.subTest(reason=reason):
                case_state = {
                    "chat_history": [{"from": "patient", "text": evidence_text}]
                }
                catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
                    case_state=case_state,
                    diagnosis="偏头痛",
                    diagnosis_axes=[],
                    verifier_issues=[],
                )
                decision = legacy_orchestrator.decide_treatment_review(
                    {
                        "edits": [
                            {
                                "edit_id": reason,
                                "operation": "append",
                                "target": "",
                                "replacement": replacement,
                                "evidence_refs": [catalog[0]["id"]],
                            }
                        ]
                    },
                    original_treatment_plan=original,
                    case_state=case_state,
                    diagnosis="偏头痛",
                    diagnosis_axes=[],
                    verifier_issues=[],
                    evidence_catalog=catalog,
                )

                self.assertEqual("rejected", decision["status"])
                self.assertIn(reason, decision["reason_codes"])

    def test_screening_purpose_cannot_support_positive_allergy_fact(self):
        original = "建议对症处理并复诊。"
        case_state = {
            "examination_results": {
                "用药风险评估": {
                    "status": "abnormal",
                    "result": {"目的": "筛查药物过敏"},
                }
            }
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        exam_evidence = next(item for item in catalog if item["source"] == "exam")
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "invent_allergy",
                        "operation": "append",
                        "target": "",
                        "replacement": "患者有药物过敏，应调整用药。",
                        "evidence_refs": [exam_evidence["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertNotEqual("positive", exam_evidence["polarity"])
        self.assertEqual("rejected", decision["status"])
        self.assertIn("unsupported_drug_allergy", decision["reason_codes"])

    def test_patient_symptom_ref_cannot_support_new_antibiotic(self):
        original = "建议对症处理并复诊。"
        case_state = {"chat_history": [{"from": "patient", "text": "头痛两天。"}]}
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "unsupported_antibiotic",
                        "operation": "append",
                        "target": "",
                        "replacement": "经验性使用阿莫西林治疗。",
                        "evidence_refs": [catalog[0]["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("unsupported_treatment_indication", decision["reason_codes"])

    def test_diagnosis_alone_cannot_authorize_new_medication(self):
        original = "建议对症处理并复诊。"
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "unsupported_antibiotic",
                        "operation": "append",
                        "target": "",
                        "replacement": "经验性使用阿莫西林治疗。",
                        "evidence_refs": ["diagnosis:final"],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state={},
            diagnosis="病毒感染",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("unsupported_treatment_indication", decision["reason_codes"])

    def test_allergy_evidence_cannot_authorize_same_medication(self):
        original = "建议对症处理并复诊。"
        case_state = {
            "chat_history": [{"from": "patient", "text": "我对阿莫西林过敏。"}]
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="细菌感染",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "unsafe_allergen",
                        "operation": "append",
                        "target": "",
                        "replacement": "经验性使用阿莫西林治疗。",
                        "evidence_refs": [catalog[0]["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="细菌感染",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("unsupported_treatment_indication", decision["reason_codes"])

    def test_diagnosis_and_verifier_can_authorize_explicit_new_medication(self):
        original = "建议对症处理并复诊。"
        verifier_issues = [
            {
                "problem": "缺少针对细菌性肺炎的抗感染治疗",
                "edit": "给予阿莫西林治疗",
            }
        ]
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "supported_antibiotic",
                        "operation": "append",
                        "target": "",
                        "replacement": "给予阿莫西林治疗。",
                        "evidence_refs": ["diagnosis:final", "verifier:1"],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state={},
            diagnosis="细菌性肺炎",
            diagnosis_axes=[],
            verifier_issues=verifier_issues,
        )

        self.assertEqual("accepted", decision["status"])

    def test_current_medication_header_can_support_discontinuation(self):
        original = "建议由专科复核用药。"
        case_state = {
            "chat_history": [{"from": "patient", "text": "我的现用药包括华法林。"}]
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="药物不良反应",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "stop_current_medication",
                        "operation": "append",
                        "target": "",
                        "replacement": "立即停用华法林并由专科调整方案。",
                        "evidence_refs": [catalog[0]["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="药物不良反应",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("accepted", decision["status"])

    def test_atomic_same_text_replace_is_unchanged(self):
        original = "建议对症处理并复诊。"
        case_state = {"chat_history": [{"from": "patient", "text": "头痛两天。"}]}
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "same_text",
                        "operation": "replace",
                        "target": original,
                        "replacement": original,
                        "evidence_refs": [catalog[0]["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("unchanged", decision["status"])
        self.assertEqual(decision["before_hash"], decision["after_hash"])

    def test_conditional_safety_language_cannot_be_rewritten_as_patient_fact(self):
        cases = [
            ("孕妇禁用本药。", "患者妊娠12周，应调整全部用药。", "unsupported_pregnancy"),
            ("肾功能不全者慎用。", "患者肾功能不全，应减少剂量。", "unsupported_renal_impairment"),
            ("青霉素过敏者禁用。", "患者青霉素过敏，应禁用本药。", "unsupported_drug_allergy"),
        ]
        for original, replacement, reason in cases:
            with self.subTest(reason=reason):
                case_state = {"chat_history": [{"from": "patient", "text": "症状持续。"}]}
                catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
                    case_state=case_state,
                    diagnosis="病毒感染",
                    diagnosis_axes=[],
                    verifier_issues=[],
                )
                decision = legacy_orchestrator.decide_treatment_review(
                    {
                        "edits": [
                            {
                                "edit_id": reason,
                                "operation": "replace",
                                "target": original,
                                "replacement": replacement,
                                "evidence_refs": [catalog[0]["id"]],
                            }
                        ]
                    },
                    original_treatment_plan=original,
                    case_state=case_state,
                    diagnosis="病毒感染",
                    diagnosis_axes=[],
                    verifier_issues=[],
                    evidence_catalog=catalog,
                )

                self.assertEqual("rejected", decision["status"])
                self.assertIn(reason, decision["reason_codes"])

    def test_atomic_review_rejects_empty_result(self):
        original = "立即停用不安全药物。"
        case_state = {"chat_history": [{"from": "patient", "text": "出现严重不良反应。"}]}
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="药物不良反应",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "delete_all",
                        "operation": "delete",
                        "target": original,
                        "replacement": "",
                        "evidence_refs": [catalog[0]["id"]],
                    }
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="药物不良反应",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("empty_treatment_plan", decision["reason_codes"])
        self.assertEqual(original, decision["treatment_plan"])
        self.assertEqual(decision["before_hash"], decision["after_hash"])

    def test_empty_edits_cannot_fall_through_to_legacy_whole_plan(self):
        original = "建议对症处理并复诊。"
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [],
                "treatment_plan": "患者已怀孕，应调整全部用药。",
                "evidence_refs": ["diagnosis:final"],
            },
            original_treatment_plan=original,
            case_state={},
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        self.assertEqual("unchanged", decision["status"])
        self.assertEqual(original, decision["treatment_plan"])

    def test_duplicate_edit_ids_are_rejected(self):
        original = "建议对症处理并复诊。"
        case_state = {"chat_history": [{"from": "patient", "text": "症状持续。"}]}
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="病毒感染",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "edits": [
                    {
                        "edit_id": "duplicate",
                        "operation": "append",
                        "target": "",
                        "replacement": "监测体温。",
                        "evidence_refs": [catalog[0]["id"]],
                    },
                    {
                        "edit_id": "duplicate",
                        "operation": "append",
                        "target": "",
                        "replacement": "记录症状变化。",
                        "evidence_refs": [catalog[0]["id"]],
                    },
                ]
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="病毒感染",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("duplicate_edit_id", decision["reason_codes"])

    def test_missing_and_ambiguous_atomic_targets_are_rejected(self):
        case_state = {"chat_history": [{"from": "patient", "text": "症状持续。"}]}
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="病毒感染",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        cases = [
            ("建议对症处理并复诊。", "不存在的原文", "missing_edit_target"),
            ("复诊。复诊。", "复诊。", "ambiguous_edit_target"),
        ]
        for original, target, reason in cases:
            with self.subTest(reason=reason):
                decision = legacy_orchestrator.decide_treatment_review(
                    {
                        "edits": [
                            {
                                "edit_id": reason,
                                "operation": "delete",
                                "target": target,
                                "replacement": "",
                                "evidence_refs": [catalog[0]["id"]],
                            }
                        ]
                    },
                    original_treatment_plan=original,
                    case_state=case_state,
                    diagnosis="病毒感染",
                    diagnosis_axes=[],
                    verifier_issues=[],
                    evidence_catalog=catalog,
                )

                self.assertEqual("rejected", decision["status"])
                self.assertIn(reason, decision["reason_codes"])
                self.assertEqual(original, decision["treatment_plan"])
                self.assertEqual(decision["before_hash"], decision["after_hash"])

    def test_negative_patient_answer_and_doctor_question_cannot_support_pregnancy_claim(self):
        original = "建议对症处理并复诊。"
        case_state = {
            "chat_history": [
                {"from": "doctor", "text": "请问是否怀孕？"},
                {"from": "patient", "text": "没有怀孕。"},
            ]
        }
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )

        self.assertNotIn("请问是否怀孕？", [item["text"] for item in catalog])
        negative_patient = next(item for item in catalog if item["source"] == "patient")
        self.assertEqual("negative", negative_patient["polarity"])
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "treatment_plan": "患者已怀孕，应按妊娠期调整全部用药。",
                "evidence_refs": [negative_patient["id"]],
            },
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("unsupported_pregnancy", decision["reason_codes"])

    def test_unvalidated_axis_evidence_id_is_rejected(self):
        original = "建议对症处理并复诊。"
        axes = [
            {
                "axis_id": "unvalidated_axis",
                "source": "llm",
                "validated": False,
                "evidence": ["患者已怀孕"],
            }
        ]
        decision = legacy_orchestrator.decide_treatment_review(
            {
                "treatment_plan": "患者已怀孕，应调整全部用药。",
                "evidence_refs": ["axis:unvalidated_axis:evidence:1"],
            },
            original_treatment_plan=original,
            case_state={},
            diagnosis="偏头痛",
            diagnosis_axes=axes,
            verifier_issues=[],
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("axis:unvalidated_axis:evidence:1", decision["failed_refs"])

    def test_overlapping_atomic_edit_targets_are_rejected(self):
        original = "给予对症治疗并密切复诊。"
        case_state = {"chat_history": [{"from": "patient", "text": "症状持续。"}]}
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="病毒感染",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        review = {
            "edits": [
                {
                    "edit_id": "replace_one",
                    "operation": "replace",
                    "target": "对症治疗",
                    "replacement": "支持治疗",
                    "evidence_refs": [catalog[0]["id"]],
                },
                {
                    "edit_id": "replace_two",
                    "operation": "delete",
                    "target": "对症治疗并密切复诊",
                    "replacement": "",
                    "evidence_refs": [catalog[0]["id"]],
                },
            ]
        }

        decision = legacy_orchestrator.decide_treatment_review(
            review,
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="病毒感染",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("overlapping_edit_target", decision["reason_codes"])
        self.assertEqual(original, decision["treatment_plan"])

    def test_atomic_edits_cannot_split_a_sensitive_claim_across_appends(self):
        original = "建议对症处理并复诊。"
        case_state = {"chat_history": [{"from": "patient", "text": "近期反复头痛。"}]}
        catalog = legacy_orchestrator.build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
        )
        review = {
            "edits": [
                {
                    "edit_id": "split_sensitive_prefix",
                    "operation": "append",
                    "target": "",
                    "replacement": "患者已",
                    "evidence_refs": [catalog[0]["id"]],
                },
                {
                    "edit_id": "split_sensitive_suffix",
                    "operation": "append",
                    "target": "",
                    "replacement": "怀孕，应调整全部用药。",
                    "evidence_refs": [catalog[0]["id"]],
                },
            ]
        }

        decision = legacy_orchestrator.decide_treatment_review(
            review,
            original_treatment_plan=original,
            case_state=case_state,
            diagnosis="偏头痛",
            diagnosis_axes=[],
            verifier_issues=[],
            evidence_catalog=catalog,
        )

        self.assertEqual("rejected", decision["status"])
        self.assertIn("unsupported_pregnancy", decision["reason_codes"])
        self.assertEqual(original, decision["treatment_plan"])

    def test_review_treatment_plan_records_structured_decision(self):
        agent = object.__new__(legacy_orchestrator.MyDoctorAgent)
        agent._call_llm = AsyncMock(
            return_value={
                "edits": [
                    {
                        "edit_id": "replace_plan",
                        "operation": "replace",
                        "target": "建议控制心室率并根据复诊结果调整。",
                        "replacement": "继续华法林抗凝，剂量由专科复核后调整。",
                        "evidence_refs": ["patient:1"],
                    }
                ]
            }
        )
        case_state = {
            "chat_history": [
                {"from": "doctor", "text": "是否怀孕？"},
                {"from": "patient", "text": "我目前服用华法林。"},
            ]
        }

        result = asyncio.run(
            agent._review_treatment_plan(
                case_state=case_state,
                diagnosis="心房颤动",
                diagnosis_axes=[],
                treatment_plan="建议控制心室率并根据复诊结果调整。",
                # S4 conditional review: provide an actionable issue so the LLM path runs.
                verifier_issues=[
                    {
                        "code": "missing_treatment_goal",
                        "severity": "must_fix",
                        "patchable": True,
                    }
                ],
                patient_id="offline-review",
            )
        )

        self.assertEqual("继续华法林抗凝，剂量由专科复核后调整。", result)
        self.assertEqual("accepted", case_state["treatment_review_decision"]["status"])
        self.assertNotEqual(
            case_state["treatment_review_decision"]["before_hash"],
            case_state["treatment_review_decision"]["after_hash"],
        )
        rendered_prompt = agent._call_llm.await_args.kwargs["prompt"]
        self.assertIn("patient:1", rendered_prompt)
        self.assertNotIn("是否怀孕", rendered_prompt)
        self.assertNotIn("$review_evidence_catalog", rendered_prompt)


if __name__ == "__main__":
    unittest.main()
