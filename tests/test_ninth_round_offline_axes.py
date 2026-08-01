"""Offline regression for ninth-round generalization axes (no answer leakage)."""

import unittest

from agent.legacy_orchestrator import (
    apply_coverage_gap_action_gate,
    build_name_map,
    extract_case_features,
    extract_intake_facts,
    final_verifier,
    flatten_examination_catalog,
    inject_required_differentials,
    load_disease_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    merge_axis_disease_candidates,
    merge_diagnosis_axes,
    open_coverage_gaps,
    required_differential_from_case,
    select_diagnosis_axes,
    select_disease_candidates,
    select_exam_plan,
    select_prefinal_axis_exam_plan,
    should_block_final_for_coverage_gaps,
)


class NinthRoundOfflineAxesTest(unittest.TestCase):
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

    # --- Pulmonary-renal vasculitis vs infection/TB ---

    def test_pulmonary_renal_pattern_keeps_vasculitis_and_core_exams(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "咳嗽三个月，痰中带血，最近乏力，脚踝水肿，以前肾不好。",
                }
            ],
            "ordered_examinations": [],
        }
        plan, candidates, axes = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        axis_ids = [axis["axis_id"] for axis in axes]
        by_name = {item["disease"]: item for item in candidates}

        self.assertIn("显微镜下多血管炎", names)
        self.assertIn("肺结核", names)
        self.assertIn("pulmonary_renal_vasculitis_vs_infection", axis_ids)
        # Required inject must NOT flatten TB to the same top score as vasculitis
        # (otherwise default_candidate=candidates[0] defaults to short-name 肺结核).
        vasculitis_names = ["显微镜下多血管炎", "多血管炎性肉芽肿"]
        best_vasculitis = max(int(by_name[name]["score"]) for name in vasculitis_names if name in by_name)
        tb_score = int(by_name["肺结核"]["score"])
        self.assertGreater(best_vasculitis, tb_score)
        first = names[0]
        self.assertIn(first, vasculitis_names)
        self.assertNotEqual(first, "肺结核")
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", plan["examinations"])
        self.assertTrue(
            {"尿液分析（UA）", "肾功能检查（RFTs）"} & set(plan["examinations"])
        )
        self.assertIn("胸部X线检查（CXR）", plan["examinations"])

    def test_pulmonary_renal_select_disease_candidates_ranks_vasculitis_above_tb(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "长期咳嗽伴痰中带血三个月，乏力，脚踝水肿加重。",
                }
            ],
            "ordered_examinations": [],
        }
        candidates = select_disease_candidates(case_state, self.disease_catalog, limit=12)
        names = [item["disease"] for item in candidates]
        self.assertIn("肺结核", names)
        vasculitis_positions = [
            names.index(name)
            for name in ["显微镜下多血管炎", "多血管炎性肉芽肿"]
            if name in names
        ]
        self.assertTrue(vasculitis_positions, "vasculitis candidates missing")
        self.assertLess(min(vasculitis_positions), names.index("肺结核"))
        self.assertGreater(
            max(int(item["score"]) for item in candidates if item["disease"] in {"显微镜下多血管炎", "多血管炎性肉芽肿"}),
            int(next(item["score"] for item in candidates if item["disease"] == "肺结核")),
        )

    def test_apa_only_blocks_final_until_pulmonary_renal_coverage(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "长期咳嗽伴痰中带血三个月，乏力，脚踝水肿加重。",
                }
            ],
            "ordered_examinations": ["抗磷脂抗体（APA）组合检测"],
            "exam_decision_trace": [{"examinations": ["抗磷脂抗体（APA）组合检测"]}],
        }
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}
        self.assertIn("anca_vasculitis_screen", gap_ids)
        self.assertIn("renal_urine_workup", gap_ids)
        self.assertIn("chest_imaging", gap_ids)
        self.assertTrue(should_block_final_for_coverage_gaps(case_state))
        gated = apply_coverage_gap_action_gate(action="final_diagnosis", case_state=case_state)
        self.assertEqual(gated["action"], "order_examination")

        plan, _, _ = self._plan(case_state)
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", plan["examinations"])
        self.assertNotEqual(plan["examinations"], ["抗磷脂抗体（APA）组合检测"])

    def test_anti_tb_blocked_without_closed_infection_when_pulmonary_renal_open(self):
        result = final_verifier(
            diagnosis="肺结核",
            examinations=["抗磷脂抗体（APA）组合检测"],
            treatment_plan="立即启动标准抗结核治疗方案（异烟肼、利福平、吡嗪酰胺、乙胺丁醇），疗程6-9个月。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "chief_complaint": "咳嗽三个月痰中带血乏力脚踝水肿",
                "positive_findings": ["慢性咳嗽咯血", "脚踝水肿", "乏力"],
                "candidate_diagnoses": ["显微镜下多血管炎", "肺结核"],
            },
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("anti_tb_without_closed_infection_evidence", codes)
        patched = result["patched_treatment"]
        self.assertNotIn("立即启动标准抗结核治疗方案", patched)
        self.assertIn("ANCA", patched)

    def test_true_tb_with_pathogen_evidence_not_blocked(self):
        result = final_verifier(
            diagnosis="肺结核",
            examinations=["痰培养及染色", "胸部X线检查（CXR）"],
            treatment_plan="痰涂片抗酸杆菌阳性，启动标准抗结核治疗（异烟肼、利福平、吡嗪酰胺、乙胺丁醇）。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "chief_complaint": "咳嗽低热盗汗，痰涂片抗酸杆菌阳性",
                "positive_findings": ["咳嗽", "盗汗", "结核病原闭合证据"],
            },
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertNotIn("anti_tb_without_closed_infection_evidence", codes)

    def test_isolated_cough_without_renal_systemic_not_force_vasculitis(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "最近有点干咳，没有咯血，没有水肿，也没有血尿。",
                }
            ],
            "ordered_examinations": [],
        }
        required = required_differential_from_case(case_state)
        self.assertNotIn("显微镜下多血管炎", required)
        facts = extract_intake_facts(case_state)
        axis_ids = [axis["axis_id"] for axis in select_diagnosis_axes(facts)]
        self.assertNotIn("pulmonary_renal_vasculitis_vs_infection", axis_ids)

    # --- Multi-electrolyte / hypomagnesemia ---

    def test_hypokalemia_malabsorption_recalls_hypomagnesemia_and_panel(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "腹泻好几天，浑身没劲，手抽筋，心跳快，有胰腺功能不全在吃胰酶。",
                }
            ],
            "ordered_examinations": [],
            "examination_results": {
                "血清电解质": {"result": {"血钾": "3.1 mEq/L", "血钙": "7.6 mg/dL"}}
            },
        }
        plan, candidates, axes = self._plan(case_state)
        names = [item["disease"] for item in candidates]
        axis_ids = [axis["axis_id"] for axis in axes]

        self.assertIn("低镁血症", names)
        self.assertIn("低钾血症", names)
        self.assertIn("multi_electrolyte_hypomagnesemia", axis_ids)
        self.assertIn("血清电解质", plan["examinations"])
        self.assertIn("心电图（ECG）", plan["examinations"])

    def test_potassium_only_therapy_blocked_without_magnesium(self):
        result = final_verifier(
            diagnosis="低钾血症",
            examinations=["血清电解质", "心电图（ECG）"],
            treatment_plan="立即开始补钾，首选口服氯化钾缓释片，必要时静脉滴注氯化钾并监测心电图。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "chief_complaint": "腹泻后手抽筋心跳快低钾",
                "positive_findings": ["症状性低钾", "腹泻吸收不良", "手抽筋"],
                "candidate_diagnoses": ["低镁血症", "低钾血症"],
            },
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("potassium_only_without_magnesium", codes)
        patched = result["patched_treatment"]
        self.assertTrue("补充镁" in patched or "硫酸镁" in patched or "补镁" in patched)
        self.assertIn("不是只补钾", patched)

    def test_k_and_mg_repletion_not_blocked(self):
        result = final_verifier(
            diagnosis="低镁血症",
            examinations=["血清电解质", "心电图（ECG）"],
            treatment_plan="静脉补充硫酸镁纠正低镁，同时补钾，监测心电图与复查电解质。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "chief_complaint": "腹泻后抽筋低钾低镁",
                "positive_findings": ["腹泻吸收不良", "症状性低钾"],
            },
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertNotIn("potassium_only_without_magnesium", codes)

    def test_mild_fatigue_without_gi_loss_not_force_hypomagnesemia(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "最近有点没劲，没有腹泻，没有抽筋，饮食正常。",
                }
            ],
            "ordered_examinations": [],
        }
        required = required_differential_from_case(case_state)
        self.assertNotIn("低镁血症", required)
        self.assertEqual(
            [item["gap_id"] for item in open_coverage_gaps(case_state) if "electrolyte" in item["gap_id"] or "multi" in item["gap_id"]],
            [],
        )

    def test_arrhythmia_dyspnea_axis_maps_structural_function_intent_to_echo(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "过去数月活动后气短逐渐加重，并伴有反复心悸。",
                }
            ],
            "ordered_examinations": [],
        }
        axes = [
            {
                "axis_id": "cardiac_arrhythmia_dyspnea_axis",
                "status": "suspected",
                "evidence": ["活动后气短逐渐加重", "反复心悸"],
                "exam_intents": [
                    "明确是否存在心律失常及其类型",
                    "评估心脏结构与功能状态",
                ],
            }
        ]

        plan, _, _ = self._plan(case_state, axes=axes)

        self.assertIn("心电图（ECG）", plan["examinations"])
        self.assertIn("超声心动图", plan["examinations"])

    def test_reduced_lvef_blocks_non_dihydropyridine_ccb_even_when_caveated(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "活动后气短并伴心悸，既往有心功能下降。",
                }
            ],
            "examination_results": {
                "超声心动图": {
                    "result": {
                        "超声心动图左心室射血分数": "LVEF 28%［参考：55-70%］",
                    },
                    "status": "abnormal",
                }
            },
        }
        features = extract_case_features(case_state, [{"disease": "心房颤动"}])

        result = final_verifier(
            diagnosis="心房颤动",
            examinations=["心电图（ECG）", "超声心动图"],
            treatment_plan=(
                "首选β受体阻滞剂，或使用非二氢吡啶类钙通道阻滞剂"
                "（如地尔硫卓），但后者需谨慎。"
            ),
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("contraindicated_drug_recommended", codes)
        self.assertNotIn("或使用非二氢吡啶类钙通道阻滞剂", result["patched_treatment"])
        self.assertIn("应避免非二氢吡啶类钙通道阻滞剂", result["patched_treatment"])

    def test_reduced_lvef_removes_long_compound_backup_drug_clause(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "活动后气短伴心悸。"}],
            "examination_results": {
                "超声心动图": {
                    "result": {"左心室射血分数": "LVEF 28%"},
                    "status": "abnormal",
                }
            },
        }
        features = extract_case_features(case_state, [{"disease": "心房颤动"}])

        result = final_verifier(
            diagnosis="心房颤动",
            examinations=["心电图（ECG）", "超声心动图"],
            treatment_plan=(
                "首选静脉或口服β受体阻滞剂（如美托洛尔）"
                "或非二氢吡啶类钙通道阻滞剂（如地尔硫卓），"
                "但使用非二氢吡啶类钙通道阻滞剂需谨慎。"
            ),
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        self.assertNotIn("或非二氢吡啶类钙通道阻滞剂", result["patched_treatment"])
        self.assertIn("应避免非二氢吡啶类钙通道阻滞剂", result["patched_treatment"])

    def test_unrelated_drug_avoidance_does_not_mask_contraindicated_ccb(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "活动后气短伴心悸。"}],
            "examination_results": {
                "超声心动图": {
                    "result": {"左心室射血分数": "LVEF 28%"},
                    "status": "abnormal",
                }
            },
        }
        features = extract_case_features(case_state, [{"disease": "心房颤动"}])

        result = final_verifier(
            diagnosis="心房颤动",
            examinations=["心电图（ECG）", "超声心动图"],
            treatment_plan="首选地尔硫卓控制心室率，同时避免使用胺碘酮。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        self.assertIn(
            "contraindicated_drug_recommended",
            [issue["code"] for issue in result["issues"]],
        )
        self.assertNotIn("首选地尔硫卓", result["patched_treatment"])

    def test_non_negated_contraindicated_ccb_mentions_are_removed(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "活动后气短伴心悸。"}],
            "examination_results": {
                "超声心动图": {
                    "result": {"左心室射血分数": "LVEF 28%"},
                    "status": "abnormal",
                }
            },
        }
        features = extract_case_features(case_state, [{"disease": "心房颤动"}])
        unsafe_plans = [
            "口服地尔硫卓控制心室率。",
            "选用维拉帕米控制心室率。",
            "首选地尔硫卓，应避免心率过低。",
            "地尔硫卓无过敏史，可口服控制心室率。",
            "地尔硫卓无明确禁忌，可使用。",
        ]

        for treatment_plan in unsafe_plans:
            with self.subTest(treatment_plan=treatment_plan):
                result = final_verifier(
                    diagnosis="心房颤动",
                    examinations=["心电图（ECG）", "超声心动图"],
                    treatment_plan=treatment_plan,
                    official_diseases=self.official_diseases,
                    examination_catalog=self.examination_catalog,
                    exam_plan_trace=[],
                    case_features=features,
                    safety_profiles=[],
                )

                self.assertIn(
                    "contraindicated_drug_recommended",
                    [issue["code"] for issue in result["issues"]],
                )
                self.assertNotIn("地尔硫卓控制", result["patched_treatment"])
                self.assertNotIn("维拉帕米控制", result["patched_treatment"])

    def test_normal_lvef_with_negated_low_ef_text_does_not_create_ccb_contraindication(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "偶发心悸。"}],
            "examination_results": {
                "超声心动图": {
                    "result": {
                        "左心室射血分数": "未见射血分数明显降低，LVEF 60%",
                    },
                    "status": "normal",
                }
            },
        }

        features = extract_case_features(case_state, [{"disease": "心房颤动"}])

        self.assertEqual([], features["medication_risk"])

    def test_reduced_lvef_parser_handles_reference_range_before_measured_value(self):
        report_values = [
            "LVEF参考范围55-70%，实测28%",
            "左心室射血分数（正常55-70%）：28%",
        ]

        for report_value in report_values:
            with self.subTest(report_value=report_value):
                case_state = {
                    "chat_history": [{"from": "patient", "text": "活动后气短伴心悸。"}],
                    "examination_results": {
                        "超声心动图": {
                            "result": {"左心室射血分数": report_value},
                            "status": "abnormal",
                        }
                    },
                }

                features = extract_case_features(case_state, [{"disease": "心房颤动"}])

                self.assertEqual(
                    ["非二氢吡啶类钙通道阻滞剂禁忌"],
                    features["medication_risk"],
                )

    def test_normal_lvef_does_not_use_unrelated_low_percentage(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "偶发心悸。"}],
            "examination_results": {
                "超声心动图": {
                    "result": {
                        "左心室射血分数": "LVEF 60%，实际血氧饱和度28%",
                    },
                    "status": "normal",
                }
            },
        }

        features = extract_case_features(case_state, [{"disease": "心房颤动"}])

        self.assertEqual([], features["medication_risk"])

    def test_generic_wound_pathogen_intent_does_not_recall_urine_culture(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "足部伤口红肿流脓并伴发热，没有尿频、尿急或尿痛。",
                }
            ],
            "ordered_examinations": [],
        }
        axes = [
            {
                "axis_id": "deep_wound_infection_axis",
                "status": "suspected",
                "evidence": ["足部伤口红肿流脓", "发热"],
                "missing_evidence": ["伤口病原学结果"],
                "exam_intents": ["明确致病菌种类以指导抗生素使用"],
            }
        ]

        plan, _, _ = self._plan(case_state, candidates=[], axes=axes)

        self.assertIn("细菌培养及鉴定", plan["examinations"])
        self.assertNotIn("尿培养", plan["examinations"])

    def test_urinary_pathogen_intent_uses_urine_culture_not_generic_culture(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "发热伴尿频、尿急、尿痛。"}
            ],
            "ordered_examinations": [],
        }
        axes = [
            {
                "axis_id": "urinary_infection_axis",
                "status": "suspected",
                "evidence": ["发热", "尿频、尿急、尿痛"],
                "missing_evidence": ["尿液病原和药敏"],
                "exam_intents": ["尿培养和药敏"],
            }
        ]

        plan, _, _ = self._plan(case_state, candidates=[], axes=axes)

        self.assertIn("尿培养", plan["examinations"])
        self.assertIn("抗菌药物敏感性试验（AST）", plan["examinations"])
        self.assertNotIn("细菌培养及鉴定", plan["examinations"])

    def test_prefinal_axis_review_reopens_only_unclosed_mapped_exam_gap(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "过去数月活动后气短逐渐加重，并伴有反复心悸。",
                }
            ],
            "ordered_examinations": ["心电图（ECG）", "血清电解质"],
            "examination_results": {
                "心电图（ECG）": {"result": {"心律": "心房颤动"}, "status": "abnormal"},
                "血清电解质": {"result": {"血钾": "3.2 mmol/L"}, "status": "abnormal"},
            },
        }
        axes = [
            {
                "axis_id": "cardiac_arrhythmia_dyspnea_axis",
                "status": "suspected",
                "evidence": ["活动后气短逐渐加重", "反复心悸"],
                "missing_evidence": ["心脏结构和收缩功能证据"],
                "exam_intents": ["评估心脏结构与功能状态"],
            }
        ]

        plan = select_prefinal_axis_exam_plan(
            case_state=case_state,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            exam_intent_rules=self.exam_intent_rules,
        )

        self.assertEqual(["超声心动图"], plan["examinations"])

    def test_prefinal_axis_review_rejects_intent_unrelated_to_missing_gap(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "过去数月活动后气短逐渐加重，并伴有反复心悸。",
                }
            ],
            "ordered_examinations": ["心电图（ECG）", "血清电解质"],
        }
        axes = [
            {
                "axis_id": "cardiac_arrhythmia_dyspnea_axis",
                "source": "llm",
                "status": "suspected",
                "evidence": ["活动后气短逐渐加重", "反复心悸"],
                "missing_evidence": ["心脏结构和收缩功能证据"],
                "exam_intents": ["ANCA血管炎筛查"],
            }
        ]

        plan = select_prefinal_axis_exam_plan(
            case_state=case_state,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            exam_intent_rules=self.exam_intent_rules,
        )

        self.assertEqual([], plan["examinations"])

    def test_prefinal_axis_review_rejects_wrong_subgoal_in_same_infection_domain(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "足部伤口红肿流脓并伴发热。"}
            ],
            "ordered_examinations": [],
        }
        axes = [
            {
                "axis_id": "wound_infection_axis",
                "source": "llm",
                "status": "suspected",
                "evidence": ["伤口红肿流脓", "发热"],
                "missing_evidence": ["感染病原学证据"],
                "exam_intents": ["感染严重度"],
            }
        ]

        plan = select_prefinal_axis_exam_plan(
            case_state=case_state,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            exam_intent_rules=self.exam_intent_rules,
        )

        self.assertEqual([], plan["examinations"])

    def test_prefinal_axis_review_accepts_cardiac_function_synonyms(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "活动后气短逐渐加重，并伴反复心悸。"}
            ],
            "ordered_examinations": ["心电图（ECG）", "血清电解质"],
        }
        axes = [
            {
                "axis_id": "cardiac_function_axis",
                "source": "llm",
                "status": "suspected",
                "evidence": ["活动后气短", "心悸"],
                "missing_evidence": ["左室收缩力"],
                "exam_intents": ["超声心动图评估射血分数"],
            }
        ]

        plan = select_prefinal_axis_exam_plan(
            case_state=case_state,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            exam_intent_rules=self.exam_intent_rules,
        )

        self.assertEqual(["超声心动图"], plan["examinations"])

    def test_prefinal_axis_review_limits_each_round_to_one_exam(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "活动后气短伴心悸和乏力。"}
            ],
            "ordered_examinations": [],
        }
        axes = [
            {
                "axis_id": "cardiac_multi_gap_axis",
                "status": "suspected",
                "evidence": ["活动后气短", "心悸"],
                "missing_evidence": ["心律、结构和电解质证据"],
                "exam_intents": [
                    "心律评估",
                    "评估心脏结构与功能状态",
                    "电解质评估",
                ],
            }
        ]

        plan = select_prefinal_axis_exam_plan(
            case_state=case_state,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            exam_intent_rules=self.exam_intent_rules,
        )

        self.assertEqual(1, len(plan["examinations"]))

    def test_prefinal_axis_review_does_not_directly_order_invasive_biopsy(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "有光敏皮疹和关节痛，没有血尿或泡沫尿。"}
            ],
            "ordered_examinations": [],
        }
        axes = [
            {
                "axis_id": "autoimmune_rash_axis",
                "status": "suspected",
                "evidence": ["光敏皮疹", "关节痛"],
                "missing_evidence": ["狼疮肾炎病理分型"],
                "exam_intents": ["明确狼疮性肾炎病理分型"],
            }
        ]

        plan = select_prefinal_axis_exam_plan(
            case_state=case_state,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            exam_intent_rules=self.exam_intent_rules,
        )

        self.assertEqual([], plan["examinations"])

    def test_deep_wound_bone_involvement_intent_maps_staging_exams(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "足部深部伤口持续红肿流脓并伴发热。",
                }
            ],
            "ordered_examinations": [],
        }
        axes = [
            {
                "axis_id": "deep_wound_infection_axis",
                "status": "suspected",
                "evidence": ["足部深部伤口", "红肿流脓", "发热"],
                "missing_evidence": ["感染严重程度和骨受累证据"],
                "exam_intents": ["评估感染严重程度及是否累及骨骼"],
            }
        ]

        plan, _, _ = self._plan(case_state, candidates=[], axes=axes)

        self.assertIn("全血细胞计数（CBC）", plan["examinations"])
        self.assertIn("C反应蛋白（CRP）", plan["examinations"])
        self.assertIn("四肢X线检查", plan["examinations"])

    def test_upper_airway_mucosal_intent_maps_nasal_endoscopy(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "感冒后持续鼻塞、清嗓和咳嗽，症状逐渐加重。",
                }
            ],
            "ordered_examinations": [],
        }
        axes = [
            {
                "axis_id": "persistent_upper_airway_syndrome",
                "status": "suspected",
                "evidence": ["持续鼻塞", "清嗓和咳嗽"],
                "missing_evidence": ["鼻腔黏膜和分泌物来源"],
                "exam_intents": ["评估鼻腔及鼻窦黏膜炎症情况"],
            }
        ]

        plan, _, _ = self._plan(case_state, candidates=[], axes=axes)

        self.assertEqual(["鼻内镜检查"], plan["examinations"])

    def test_validated_axis_candidates_feed_final_candidate_list_without_dominating_strong_evidence(self):
        base_candidates = [
            {
                "department": "心内科",
                "disease": "心房颤动",
                "score": 54,
                "source": "official_catalog",
            }
        ]
        axes = [
            {
                "axis_id": "arrhythmia_with_structural_heart_disease",
                "source": "rule",
                "status": "suspected",
                "evidence": ["进行性活动后气短", "左心室射血分数降低"],
                "candidate_official_names": ["心力衰竭"],
            }
        ]

        merged = merge_axis_disease_candidates(
            base_candidates,
            diagnosis_axes=axes,
            disease_catalog=self.disease_catalog,
            limit=8,
        )

        self.assertEqual("心房颤动", merged[0]["disease"])
        heart_failure = next(item for item in merged if item["disease"] == "心力衰竭")
        self.assertEqual("diagnosis_axis", heart_failure["source"])
        self.assertLess(heart_failure["score"], merged[0]["score"])

    def test_unrelated_llm_axis_candidate_is_not_injected(self):
        base_candidates = [
            {
                "department": "呼吸内科",
                "disease": "流行性感冒",
                "score": 54,
                "source": "official_catalog",
            }
        ]
        axes = [
            {
                "axis_id": "generic_fever_axis",
                "source": "llm",
                "status": "suspected",
                "evidence": ["发热", "咳嗽"],
                "candidate_official_names": ["骨髓炎"],
            }
        ]

        merged = merge_axis_disease_candidates(
            base_candidates,
            diagnosis_axes=axes,
            disease_catalog=self.disease_catalog,
            limit=8,
        )

        self.assertEqual(["流行性感冒"], [item["disease"] for item in merged])

    def test_llm_candidate_cannot_hitchhike_on_matching_rule_axis_id(self):
        rule_axis = {
            "axis_id": "arrhythmia_structural_axis",
            "source": "rule",
            "status": "suspected",
            "evidence": ["活动后气短", "心悸"],
            "candidate_official_names": ["心力衰竭"],
            "rule_candidate_official_names": ["心力衰竭"],
        }
        llm_axis = {
            "axis_id": "arrhythmia_structural_axis",
            "source": "llm",
            "status": "suspected",
            "evidence": ["活动后气短", "心悸"],
            "candidate_official_names": ["骨髓炎"],
            "llm_candidate_official_names": ["骨髓炎"],
        }
        merged_axes = merge_diagnosis_axes([llm_axis], [rule_axis])

        candidates = merge_axis_disease_candidates(
            [],
            diagnosis_axes=merged_axes,
            disease_catalog=self.disease_catalog,
            limit=8,
        )

        names = [item["disease"] for item in candidates]
        self.assertIn("心力衰竭", names)
        self.assertNotIn("骨髓炎", names)


if __name__ == "__main__":
    unittest.main()
