"""Offline regression for eighth-round generalization axes (no answer leakage)."""

import unittest

from agent.legacy_orchestrator import (
    apply_coverage_gap_action_gate,
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
    should_force_exam_for_open_coverage,
)


class EighthRoundOfflineAxesTest(unittest.TestCase):
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
            facts = extract_intake_facts(case_state)
            axes = select_diagnosis_axes(facts)
        return select_exam_plan(
            case_state=case_state,
            disease_candidates=candidates,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            diagnosis_exam_profiles=self.exam_profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=max_items,
        ), candidates, axes

    # --- Topic 1: upper-arm trauma anatomy + negative scope ---

    def test_upper_arm_trauma_requires_humeral_shaft_and_limb_xray(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "跌倒后手肘着地，左上臂剧痛、肿胀、活动受限，不敢抬起来。",
                }
            ],
            "ordered_examinations": [],
        }
        plan, candidates, axes = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        axis_ids = [axis["axis_id"] for axis in axes]

        self.assertIn("肱骨干骨折", names)
        self.assertIn("upper_arm_trauma_fracture", axis_ids)
        self.assertIn("四肢X线检查", plan["examinations"])
        self.assertNotEqual(
            {"肩部X线检查", "手部X线检查"},
            set(plan["examinations"]),
        )

    def test_upper_arm_trauma_shoulder_hand_films_do_not_cover_humeral_shaft(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "跌倒后左上臂剧痛肿胀活动受限。",
                }
            ],
            "ordered_examinations": ["肩部X线检查", "手部X线检查"],
        }
        required = required_differential_from_case(case_state)
        self.assertIn("肱骨干骨折", required)

        result = final_verifier(
            diagnosis="肩袖损伤",
            examinations=["肩部X线检查", "手部X线检查"],
            treatment_plan="肩部和手部X线未见骨折，已排除所有骨折，按肩袖损伤康复。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "chief_complaint": "跌倒后左上臂剧痛肿胀活动受限",
                "positive_findings": ["上臂外伤", "剧痛肿胀活动受限"],
                "candidate_diagnoses": ["肩袖损伤", "肱骨干骨折"],
            },
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("overclaim_rule_out_without_coverage", codes)
        self.assertNotIn("已排除所有骨折", result["patched_treatment"])

    def test_negative_true_shoulder_pain_does_not_force_humeral_shaft(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "抬肩外展时肩部疼痛半年，没有上臂肿胀，也没有跌倒外伤。",
                }
            ],
            "ordered_examinations": [],
        }
        required = required_differential_from_case(case_state)
        self.assertNotIn("肱骨干骨折", required)
        facts = extract_intake_facts(case_state)
        axis_ids = [axis["axis_id"] for axis in select_diagnosis_axes(facts)]
        self.assertNotIn("upper_arm_trauma_fracture", axis_ids)

    # --- Topic 2: arrhythmia / precipitant vs ACS ---

    def test_palpitation_prioritizes_arrhythmia_ecg_and_electrolytes(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "突发心跳很快、胸闷气短，以前有过心脏病和心功能下降，最近腹泻吃得少。",
                }
            ],
            "ordered_examinations": [],
        }
        plan, candidates, axes = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        axis_ids = [axis["axis_id"] for axis in axes]

        self.assertTrue({"心律失常", "心动过速"} & set(names))
        self.assertIn("arrhythmia_electrolyte_precipitant", axis_ids)
        self.assertIn("心电图（ECG）", plan["examinations"])
        self.assertIn("血清电解质", plan["examinations"])

    def test_acs_intensive_therapy_blocked_without_acs_evidence(self):
        result = final_verifier(
            diagnosis="心律失常",
            examinations=["超声心动图"],
            treatment_plan="按急性冠脉综合征启动强化抗栓，并安排急诊PCI。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "chief_complaint": "突发心跳很快胸闷气短，既往心脏病",
                "positive_findings": ["突发心悸", "既往心脏病"],
                "candidate_diagnoses": ["心律失常", "冠状动脉粥样硬化性心脏病（冠状动脉疾病，CAD）"],
            },
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("acs_therapy_without_acs_evidence", codes)
        patched = result["patched_treatment"]
        # Original intensive pathway claims must be removed; advisory patch may mention them.
        self.assertNotIn("按急性冠脉综合征启动强化抗栓", patched)
        self.assertNotIn("并安排急诊PCI", patched)
        self.assertIn("不应直接强化抗栓", patched)

    def test_true_acs_path_not_blocked(self):
        result = final_verifier(
            diagnosis="冠状动脉粥样硬化性心脏病（冠状动脉疾病，CAD）",
            examinations=["心电图（ECG）", "肌钙蛋白"],
            treatment_plan="进行性压榨胸痛伴肌钙蛋白升高，启动强化抗栓并评估急诊PCI。",
            official_diseases=self.official_diseases + ["肌钙蛋白"],
            examination_catalog={
                **self.examination_catalog,
                "实验室检查-心脏": list(
                    dict.fromkeys(
                        self.examination_catalog.get("实验室检查-心脏", []) + ["肌钙蛋白"]
                    )
                ),
            },
            exam_plan_trace=[],
            case_features={
                "chief_complaint": "进行性压榨性胸痛，冷汗，肌钙蛋白升高",
                "positive_findings": ["压榨性胸痛", "肌钙蛋白升高"],
            },
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertNotIn("acs_therapy_without_acs_evidence", codes)

    # --- Topic 3: elbow overuse / CBC suppression ---

    def test_elbow_overuse_recalls_epicondylitis_and_suppresses_cbc(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "三个月来局限肘部疼痛，重复抓握和腕部活动后加重，没有发热红肿。",
                }
            ],
            "ordered_examinations": [],
        }
        plan, candidates, axes = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        axis_ids = [axis["axis_id"] for axis in axes]

        self.assertTrue({"肱骨内上髁炎", "网球肘"} & set(names))
        self.assertIn("elbow_overuse_enthesopathy", axis_ids)
        self.assertNotIn("全血细胞计数（CBC）", plan["examinations"])

    def test_inflammatory_elbow_allows_cbc(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "肘部红肿发热三天，体温39度，活动疼痛，怀疑感染。",
                }
            ],
            "ordered_examinations": [],
        }
        facts = extract_intake_facts(case_state)
        axis_ids = [axis["axis_id"] for axis in select_diagnosis_axes(facts)]
        self.assertNotIn("elbow_overuse_enthesopathy", axis_ids)

    # --- Topic 4: hypersplenism vs MDS ---

    def test_hepato_splenic_cytopenia_keeps_hypersplenism_axis(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "有慢性肝病用药史，最近左上腹饱胀、容易瘀青，化验三系减少。",
                }
            ],
            "examination_results": {
                "全血细胞计数（CBC）": {
                    "result": {"血红蛋白": "下降", "白细胞": "下降", "血小板": "下降"}
                }
            },
            "ordered_examinations": ["全血细胞计数（CBC）"],
        }
        plan, candidates, axes = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        axis_ids = [axis["axis_id"] for axis in axes]

        self.assertIn("脾功能亢进", names)
        self.assertIn("hypersplenism_vs_primary_marrow", axis_ids)
        self.assertIn("腹部超声", plan["examinations"])

    def test_mds_specific_therapy_blocked_without_marrow_when_hypersplenism_open(self):
        result = final_verifier(
            diagnosis="骨髓增生异常综合征",
            examinations=["全血细胞计数（CBC）"],
            treatment_plan="按MDS启动去甲基化治疗。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "chief_complaint": "慢性肝病，左上腹饱胀，三系减少",
                "positive_findings": ["慢性肝病", "左上腹饱胀", "三系减少"],
                "candidate_diagnoses": ["脾功能亢进", "骨髓增生异常综合征"],
                "diagnosis_axes": [
                    {
                        "axis_id": "hypersplenism_vs_primary_marrow",
                        "treatment_risks": ["mds_specific_without_marrow"],
                        "candidate_official_names": ["脾功能亢进", "骨髓增生异常综合征"],
                    }
                ],
            },
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("mds_therapy_without_marrow_evidence", codes)
        patched = result["patched_treatment"]
        self.assertNotIn("按MDS启动去甲基化治疗", patched)
        self.assertIn("不应启动去甲基化", patched)

    def test_isolated_cytopenia_without_liver_does_not_force_hypersplenism_only(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "体检发现三系减少，没有肝病史，也没有左上腹饱胀。",
                }
            ],
            "ordered_examinations": [],
        }
        required = required_differential_from_case(case_state)
        self.assertNotIn("脾功能亢进", required)

    # --- Stage 2: coverage gaps force exam / block premature final ---

    def test_upper_arm_partial_imaging_blocks_final_and_requires_limb_xray(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "跌倒后手肘着地，左上臂剧痛、肿胀、活动受限。",
                }
            ],
            "ordered_examinations": ["肩部X线检查", "手部X线检查"],
            "exam_decision_trace": [{"examinations": ["肩部X线检查", "手部X线检查"]}],
        }
        gap_ids = [item["gap_id"] for item in open_coverage_gaps(case_state)]
        self.assertIn("upper_arm_long_bone_imaging", gap_ids)
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))
        self.assertTrue(should_force_exam_for_open_coverage(case_state))

        gated = apply_coverage_gap_action_gate(
            action="final_diagnosis",
            case_state=case_state,
            reason="信息足够，进入 final。",
        )
        self.assertEqual(gated["action"], "order_examination")
        self.assertIn("长骨", gated["reason"])

        plan, _, _ = self._plan(case_state)
        self.assertIn("四肢X线检查", plan["examinations"])

    def test_upper_arm_covered_imaging_allows_final(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "跌倒后左上臂剧痛肿胀活动受限。",
                }
            ],
            "ordered_examinations": ["四肢X线检查"],
            "examination_results": {
                "四肢X线检查": {
                    "status": "normal",
                    "result": {"所见": "未见骨折"},
                }
            },
            "exam_decision_trace": [{"examinations": ["四肢X线检查"]}],
        }
        self.assertFalse(open_coverage_gaps(case_state))
        self.assertFalse(should_block_final_for_coverage_gaps(case_state))
        gated = apply_coverage_gap_action_gate(
            action="final_diagnosis",
            case_state=case_state,
            reason="覆盖已闭合。",
        )
        self.assertEqual(gated["action"], "final_diagnosis")

    def test_palpitation_echo_only_blocks_final_until_ecg_and_electrolytes(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "突发心跳很快、胸闷气短，以前有过心脏病和心功能下降，最近腹泻吃得少。",
                }
            ],
            "ordered_examinations": ["超声心动图"],
            "exam_decision_trace": [{"examinations": ["超声心动图"]}],
        }
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}
        self.assertIn("rhythm_ecg", gap_ids)
        self.assertIn("electrolyte_precipitant", gap_ids)
        gated = apply_coverage_gap_action_gate(
            action="final_diagnosis",
            case_state=case_state,
        )
        self.assertEqual(gated["action"], "order_examination")

        plan, candidates, _ = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        self.assertTrue({"心律失常", "心动过速"} & set(names))
        self.assertIn("心电图（ECG）", plan["examinations"])
        self.assertIn("血清电解质", plan["examinations"])

        covered = {
            **case_state,
            "ordered_examinations": ["超声心动图", "心电图（ECG）", "血清电解质"],
            "examination_results": {
                "超声心动图": {
                    "status": "normal",
                    "result": {"所见": "结构和功能已记录"},
                },
                "心电图（ECG）": {
                    "status": "normal",
                    "result": {"所见": "窦性心律"},
                },
                "血清电解质": {
                    "status": "normal",
                    "result": {"血钾": "正常"},
                },
            },
            "exam_decision_trace": [
                {"examinations": ["超声心动图"]},
                {"examinations": ["心电图（ECG）", "血清电解质"]},
            ],
        }
        self.assertFalse(open_coverage_gaps(covered))
        self.assertEqual(
            apply_coverage_gap_action_gate(action="final_diagnosis", case_state=covered)["action"],
            "final_diagnosis",
        )

    def test_elbow_overuse_empty_exams_force_local_assessment_not_cbc(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "三个月来局限肘部疼痛，重复抓握和腕部活动后加重，没有发热红肿。",
                }
            ],
            "ordered_examinations": [],
            "exam_decision_trace": [],
        }
        gap_ids = [item["gap_id"] for item in open_coverage_gaps(case_state)]
        self.assertIn("local_enthesopathy_exam", gap_ids)
        gated = apply_coverage_gap_action_gate(action="final_diagnosis", case_state=case_state)
        self.assertEqual(gated["action"], "order_examination")

        plan, candidates, _ = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        self.assertTrue({"肱骨内上髁炎", "网球肘"} & set(names))
        self.assertNotIn("全血细胞计数（CBC）", plan["examinations"])
        self.assertIn("体格检查", plan["examinations"])

        after_local = {
            **case_state,
            "ordered_examinations": ["体格检查"],
            "exam_decision_trace": [{"examinations": ["体格检查"]}],
        }
        self.assertNotIn(
            "local_enthesopathy_exam",
            [item["gap_id"] for item in open_coverage_gaps(after_local)],
        )

    def test_hepato_splenic_cbc_only_blocks_final_until_abdominal_ultrasound(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "有慢性肝病用药史，最近左上腹饱胀、容易瘀青，化验三系减少。",
                }
            ],
            "ordered_examinations": ["全血细胞计数（CBC）"],
            "exam_decision_trace": [{"examinations": ["全血细胞计数（CBC）"]}],
            "examination_results": {
                "全血细胞计数（CBC）": {
                    "result": {"血红蛋白": "下降", "白细胞": "下降", "血小板": "下降"}
                }
            },
        }
        gap_ids = [item["gap_id"] for item in open_coverage_gaps(case_state)]
        self.assertIn("hepato_splenic_structure", gap_ids)
        gated = apply_coverage_gap_action_gate(action="final_diagnosis", case_state=case_state)
        self.assertEqual(gated["action"], "order_examination")

        plan, candidates, axes = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        axis_ids = [axis["axis_id"] for axis in axes]
        self.assertIn("脾功能亢进", names)
        self.assertIn("骨髓增生异常综合征", names)
        self.assertIn("hypersplenism_vs_primary_marrow", axis_ids)
        self.assertIn("腹部超声", plan["examinations"])

    def test_simple_case_without_axis_patterns_does_not_block_final(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "最近眼睛干涩疲劳，看近处模糊，没有外伤，没有心悸，没有肝病。",
                }
            ],
            "ordered_examinations": ["视力检查"],
            "exam_decision_trace": [{"examinations": ["视力检查"]}],
        }
        self.assertEqual(open_coverage_gaps(case_state), [])
        self.assertFalse(should_block_final_for_coverage_gaps(case_state))
        gated = apply_coverage_gap_action_gate(action="final_diagnosis", case_state=case_state)
        self.assertEqual(gated["action"], "final_diagnosis")

    def test_exhausted_exam_budget_does_not_deadlock_on_open_gaps(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "跌倒后左上臂剧痛肿胀活动受限。",
                }
            ],
            "ordered_examinations": ["肩部X线检查"],
            # Simulate exam action budget exhausted while gap remains.
            "exam_decision_trace": [{"examinations": ["肩部X线检查"]}] * 3,
        }
        self.assertTrue(open_coverage_gaps(case_state))
        self.assertFalse(should_block_final_for_coverage_gaps(case_state, max_exam_actions=3))
        gated = apply_coverage_gap_action_gate(
            action="final_diagnosis",
            case_state=case_state,
            max_exam_actions=3,
        )
        self.assertEqual(gated["action"], "final_diagnosis")


if __name__ == "__main__":
    unittest.main()
