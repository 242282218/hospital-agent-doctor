"""Offline regressions from the tenth-round random batch."""

import unittest

from agent.legacy_orchestrator import (
    apply_negative_evidence_scope_gate,
    apply_treatment_safety,
    build_name_map,
    extract_intake_facts,
    flatten_examination_catalog,
    load_knowledge_registry,
    open_coverage_gaps,
    select_diagnosis_axes,
    select_exam_plan,
    select_prefinal_axis_exam_plan,
)


class TenthRoundOfflineFailuresTest(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "功能检查": ["新斯的明试验", "听力测定"],
            "影像学检查-超声": ["泌尿道超声", "肾脏超声"],
            "实验室检查-尿液": ["尿液分析（UA）", "尿培养"],
        }
        self.item_map = build_name_map(flatten_examination_catalog(self.catalog))
        self.intent_rules = load_knowledge_registry()["exam_intent_map"]

    def _prefinal_plan(self, case_state, axis):
        return select_prefinal_axis_exam_plan(
            case_state=case_state,
            diagnosis_axes=[axis],
            examination_catalog=self.catalog,
            item_name_map=self.item_map,
            exam_intent_rules=self.intent_rules,
        )

    def test_prefinal_neuromuscular_gap_maps_single_function_test(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "单侧上眼睑下垂，傍晚加重，休息后缓解。"}
            ],
            "ordered_examinations": ["脑和眼眶MRI"],
        }
        axis = {
            "axis_id": "ptosis_etiology_investigation",
            "source": "llm",
            "status": "missing_evidence",
            "evidence": ["上眼睑下垂", "疲劳后加重"],
            "missing_evidence": ["新斯的明试验结果"],
            "exam_intents": ["鉴别重症肌无力与神经源性上眼睑下垂"],
        }

        plan = self._prefinal_plan(case_state, axis)

        self.assertEqual(["新斯的明试验"], plan["examinations"])

    def test_prefinal_hearing_gap_maps_quantitative_hearing_test(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "首次突发旋转性眩晕，伴单侧耳鸣、耳闷和听力发闷。"}
            ],
            "ordered_examinations": ["眼震电图（ENG）", "前庭诱发肌源性电位（VEMP）"],
        }
        axis = {
            "axis_id": "acute_audio_vestibular_syndrome",
            "source": "llm",
            "status": "missing_evidence",
            "evidence": ["旋转性眩晕", "单侧听力发闷"],
            "missing_evidence": ["纯音测听结果以确认听力损失类型和程度"],
            "exam_intents": ["明确听力损失性质及程度"],
        }

        plan = self._prefinal_plan(case_state, axis)

        self.assertEqual(["听力测定"], plan["examinations"])

    def test_prefinal_urinary_stone_gap_maps_noninvasive_imaging(self):
        case_state = urinary_stone_infection_case()
        axis = {
            "axis_id": "urinary_stone_with_inflammation",
            "source": "llm",
            "status": "missing_evidence",
            "evidence": ["单侧腰痛伴血尿", "尿路刺激征"],
            "missing_evidence": ["泌尿系超声或CT以明确结石位置和大小"],
            "exam_intents": ["通过影像学检查确认是否存在泌尿系结石"],
        }

        plan = self._prefinal_plan(case_state, axis)

        self.assertEqual(["泌尿道超声"], plan["examinations"])

    def test_urinary_stone_infection_clues_keep_imaging_coverage_open(self):
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(urinary_stone_infection_case())}

        self.assertIn("urinary_stone_infection_imaging", gap_ids)

    def test_urinary_rule_axis_keeps_infection_and_stone_in_differential(self):
        axes = select_diagnosis_axes(extract_intake_facts(urinary_stone_infection_case()))
        axis = next(item for item in axes if item["axis_id"] == "symptomatic_urinary_infection_vs_stone")

        self.assertIn("急性膀胱炎", axis["candidate_official_names"])
        self.assertIn("输尿管结石", axis["candidate_official_names"])

    def test_completed_urinary_imaging_closes_coverage_gap(self):
        case_state = urinary_stone_infection_case()
        case_state["ordered_examinations"].append("泌尿道超声")
        case_state["examination_results"]["泌尿道超声"] = {
            "status": "normal",
            "result": {"泌尿系统": "未见结石或梗阻"},
        }

        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}

        self.assertNotIn("urinary_stone_infection_imaging", gap_ids)

    def test_urinary_imaging_order_without_result_does_not_close_gap(self):
        case_state = urinary_stone_infection_case()
        case_state["ordered_examinations"].append("泌尿道超声")

        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}

        self.assertIn("urinary_stone_infection_imaging", gap_ids)

    def test_invalid_urinary_imaging_does_not_close_coverage_gap(self):
        case_state = urinary_stone_infection_case()
        case_state["ordered_examinations"].append("泌尿道超声")
        case_state["invalid_examinations"] = ["泌尿道超声"]

        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}

        self.assertIn("urinary_stone_infection_imaging", gap_ids)

    def test_existing_equivalent_urinary_imaging_is_not_reordered(self):
        case_state = urinary_stone_infection_case()
        case_state["ordered_examinations"].append("肾脏超声")
        case_state["examination_results"]["肾脏超声"] = {
            "status": "abnormal",
            "result": {"右侧输尿管": "可疑强回声伴声影"},
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[],
            diagnosis_axes=axes,
            examination_catalog=self.catalog,
            item_name_map=self.item_map,
            diagnosis_exam_profiles=[],
            exam_intent_rules=self.intent_rules,
        )

        self.assertNotIn("泌尿道超声", plan["examinations"])

    def test_distributed_urinary_negation_does_not_open_axis_or_gap(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "无尿频、尿急、尿痛或腰痛，尿红细胞25/HPF。"}
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}

        self.assertNotIn("symptomatic_urinary_infection_vs_stone", {item["axis_id"] for item in axes})
        self.assertNotIn("urinary_stone_infection_imaging", gap_ids)

    def test_current_urinary_symptoms_override_past_negation(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "既往无尿痛，现在突发尿痛、右侧腰痛和肉眼血尿。",
                }
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}

        self.assertIn("symptomatic_urinary_infection_vs_stone", {item["axis_id"] for item in axes})
        self.assertIn("urinary_stone_infection_imaging", gap_ids)

    def test_unpunctuated_current_symptoms_override_past_negation(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "既往否认尿痛现突发尿痛伴右侧腰痛和肉眼血尿。"}
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}

        self.assertIn("symptomatic_urinary_infection_vs_stone", {item["axis_id"] for item in axes})
        self.assertIn("urinary_stone_infection_imaging", gap_ids)

    def test_renal_percussion_pain_opens_urinary_imaging_gap(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "右肾区叩击痛，伴尿频、尿急、尿痛和镜下血尿。",
                }
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }

        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}

        self.assertIn("urinary_stone_infection_imaging", gap_ids)

    def test_negative_culture_does_not_rule_out_symptomatic_pyuria(self):
        treatment = (
            "鉴于患者有抗生素过敏史且尿培养阴性，暂不经验性使用抗生素。"
            "尿培养阴性已排除尿路感染，仅按结石处理。"
        )
        result = apply_negative_evidence_scope_gate(
            treatment,
            examinations=["尿液分析（UA）", "尿培养"],
            case_features={"case_text": "有抗生素过敏史。" + urinary_case_text()},
        )

        issue_codes = {item["code"] for item in result["issues"]}
        self.assertIn("negative_culture_overrules_symptomatic_pyuria", issue_codes)
        self.assertNotIn("已排除尿路感染", result["treatment_plan"])
        self.assertIn("暂不经验性使用抗生素", result["treatment_plan"])
        self.assertIn("不能单独排除感染", " ".join(result["patches"]))
        self.assertIn("过敏", " ".join(result["patches"]))

    def test_asymptomatic_negative_culture_does_not_trigger_urinary_gate(self):
        result = apply_negative_evidence_scope_gate(
            "尿培养阴性，已排除尿路感染。",
            examinations=["尿液分析（UA）", "尿培养"],
            case_features={
                "case_text": "体检发现尿白细胞升高，无尿频、尿急、尿痛或腰痛，尿培养无生长。"
            },
        )

        self.assertEqual([], result["issues"])

    def test_normal_urine_white_cells_do_not_count_as_pyuria(self):
        result = apply_negative_evidence_scope_gate(
            "尿培养阴性，已排除尿路感染。",
            examinations=["尿液分析（UA）", "尿培养"],
            case_features={
                "case_text": "尿频尿急尿痛，尿白细胞0-2/HPF，尿培养无生长。",
                "examination_results": {
                    "尿液分析（UA）": {
                        "status": "normal",
                        "result": {"尿白细胞": "0-2/HPF", "白细胞酯酶": "阴性"},
                    },
                    "尿培养": {"status": "normal", "result": {"生长情况": "无生长"}},
                },
            },
        )

        self.assertEqual([], result["issues"])

    def test_positive_culture_reference_range_does_not_count_as_negative(self):
        result = apply_negative_evidence_scope_gate(
            "尿培养阴性，已排除尿路感染。",
            examinations=["尿液分析（UA）", "尿培养"],
            case_features={
                "case_text": "尿频尿急尿痛，尿白细胞18/HPF；尿培养大肠埃希菌生长，参考范围无生长。",
                "examination_results": {
                    "尿液分析（UA）": {
                        "status": "abnormal",
                        "result": {"尿白细胞": "18/HPF"},
                    },
                    "尿培养": {
                        "status": "abnormal",
                        "result": {
                            "培养结果": "大肠埃希菌 150,000 CFU/mL［参考范围：无生长或 <10,000］"
                        },
                    },
                },
            },
        )

        self.assertEqual([], result["issues"])

    def test_cannot_exclude_urinary_infection_is_not_treated_as_exclusion(self):
        result = apply_negative_evidence_scope_gate(
            "单次尿培养阴性不能排除尿路感染，患者稳定时可暂缓经验性抗菌并复核病因。",
            examinations=["尿液分析（UA）", "尿培养"],
            case_features={"case_text": urinary_case_text()},
        )

        self.assertEqual([], result["issues"])
        self.assertIn("不能排除尿路感染", result["treatment_plan"])
        self.assertIn("暂缓经验性抗菌", result["treatment_plan"])

    def test_insufficient_to_exclude_urinary_infection_is_preserved(self):
        result = apply_negative_evidence_scope_gate(
            "单次尿培养阴性不足以排除尿路感染，应结合影像和临床复评。",
            examinations=["尿液分析（UA）", "尿培养"],
            case_features={"case_text": urinary_case_text()},
        )

        self.assertEqual([], result["issues"])
        self.assertIn("不足以排除尿路感染", result["treatment_plan"])

    def test_stable_conditional_antibiotic_deferral_is_preserved(self):
        plans = [
            "生命体征稳定，尿培养阴性，暂不使用抗生素，24小时内复评。",
            "目前无尿路感染全身征象，继续影像与临床复评。",
        ]

        for plan in plans:
            with self.subTest(plan=plan):
                result = apply_negative_evidence_scope_gate(
                    plan,
                    examinations=["尿液分析（UA）", "尿培养"],
                    case_features={"case_text": urinary_case_text()},
                )
                self.assertEqual([], result["issues"])
                self.assertEqual(plan, result["treatment_plan"])

    def test_negative_culture_cannot_stop_infection_branch(self):
        result = apply_negative_evidence_scope_gate(
            "尿培养阴性，故不考虑尿路感染并停止抗菌治疗。",
            examinations=["尿液分析（UA）", "尿培养"],
            case_features={"case_text": urinary_case_text()},
        )

        self.assertIn(
            "negative_culture_overrules_symptomatic_pyuria",
            {item["code"] for item in result["issues"]},
        )
        self.assertNotIn("不考虑尿路感染", result["treatment_plan"])
        self.assertNotIn("停止抗菌治疗", result["treatment_plan"])
        self.assertFalse(result["treatment_plan"].rstrip().endswith(("，并", ",并")))

    def test_negative_culture_exclusion_variants_are_removed(self):
        variants = [
            "尿培养结果阴性，已排除尿路感染。",
            "尿培养未见细菌生长，故排除尿路感染。",
            "尿培养阴性，排除了尿路感染。",
            "尿培养阴性，尿路感染已排除。",
            "尿培养阴性，无需抗生素。",
            "尿培养阴性，明确无需使用抗生素。",
        ]

        for plan in variants:
            with self.subTest(plan=plan):
                result = apply_negative_evidence_scope_gate(
                    plan,
                    examinations=["尿液分析（UA）", "尿培养"],
                    case_features={"case_text": urinary_case_text()},
                )
                self.assertIn(
                    "negative_culture_overrules_symptomatic_pyuria",
                    {item["code"] for item in result["issues"]},
                )
                self.assertNotIn("排除尿路感染", result["treatment_plan"])
                self.assertNotIn("无需抗生素", result["treatment_plan"])

    def test_negative_culture_patch_mentions_allergy_only_when_supported(self):
        result = apply_negative_evidence_scope_gate(
            "尿培养阴性，已排除尿路感染。",
            examinations=["尿液分析（UA）", "尿培养"],
            case_features={"case_text": urinary_case_text()},
        )

        self.assertNotIn("过敏", " ".join(result["patches"]))

    def test_nsaid_is_removed_when_current_analgesic_and_risks_are_unresolved(self):
        plan = (
            "继续核对坦索罗辛的剂量和适应证；"
            "联合非甾体抗炎药（如布洛芬或双氯芬酸，若无禁忌）缓解疼痛。"
        )
        result = apply_treatment_safety(
            plan,
            diagnosis="输尿管结石",
            case_features={
                "case_text": "当前只说偶尔吃止痛药，具体药名、年龄、肾功能和出血风险尚未核对。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": [],
            },
            safety_profiles=[],
        )

        issue_codes = {item["code"] for item in result["issues"]}
        self.assertIn("nsaid_safety_facts_unresolved", issue_codes)
        self.assertNotIn("布洛芬", result["treatment_plan"])
        self.assertNotIn("双氯芬酸", result["treatment_plan"])
        self.assertNotIn("若无禁忌）", result["treatment_plan"])
        self.assertIn("坦索罗辛", result["treatment_plan"])

    def test_nsaid_is_removed_for_advanced_age_with_bruising_risk(self):
        result = apply_treatment_safety(
            "可使用双氯芬酸缓解疼痛。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "84岁，平时容易出现瘀斑，肾功能尚未检查。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": [],
            },
            safety_profiles=[],
        )

        self.assertIn("nsaid_high_risk_present", {item["code"] for item in result["issues"]})
        self.assertNotIn("双氯芬酸", result["treatment_plan"])

    def test_nsaid_is_removed_when_anticoagulant_follows_denied_bleeding(self):
        result = apply_treatment_safety(
            "确诊肾绞痛后可短期使用布洛芬镇痛。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "72岁，肾功能正常，正在使用抗凝药且否认消化道出血和瘀斑。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": [],
            },
            safety_profiles=[],
        )

        self.assertIn("nsaid_high_risk_present", {item["code"] for item in result["issues"]})
        self.assertNotIn("布洛芬", result["treatment_plan"])

    def test_nsaid_is_removed_when_anticoagulant_follows_conjunctive_denial(self):
        result = apply_treatment_safety(
            "确诊肾绞痛后可短期使用布洛芬镇痛。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "40岁，肾功能正常，无消化道出血且正在服用华法林。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": ["华法林"],
            },
            safety_profiles=[],
        )

        self.assertIn("nsaid_high_risk_present", {item["code"] for item in result["issues"]})
        self.assertNotIn("布洛芬", result["treatment_plan"])

    def test_unreviewed_nsaid_risks_are_not_treated_as_denied(self):
        result = apply_treatment_safety(
            "确诊肾绞痛后可短期使用布洛芬镇痛。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "72岁，肾功能正常，未核对胃溃疡、消化道出血、瘀斑、抗凝和抗血小板用药。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": [],
            },
            safety_profiles=[],
        )

        self.assertIn("nsaid_safety_facts_unresolved", {item["code"] for item in result["issues"]})
        self.assertNotIn("布洛芬", result["treatment_plan"])

    def test_not_stopped_anticoagulant_remains_positive_risk(self):
        result = apply_treatment_safety(
            "确诊肾绞痛后可短期使用布洛芬镇痛。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "72岁，肾功能正常，否认消化道出血和瘀斑，未停用抗凝药。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": ["抗凝药"],
            },
            safety_profiles=[],
        )

        self.assertIn("nsaid_high_risk_present", {item["code"] for item in result["issues"]})
        self.assertNotIn("布洛芬", result["treatment_plan"])

    def test_second_systemic_nsaid_is_removed(self):
        result = apply_treatment_safety(
            "加用双氯芬酸缓解疼痛。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "45岁，当前正在服用布洛芬。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": ["布洛芬"],
            },
            safety_profiles=[],
        )

        self.assertIn("duplicate_systemic_nsaid", {item["code"] for item in result["issues"]})
        self.assertNotIn("双氯芬酸", result["treatment_plan"])

    def test_nsaid_sanitizer_preserves_other_treatment_in_same_sentence(self):
        result = apply_treatment_safety(
            "继续坦索罗辛，同时加用布洛芬镇痛并监测尿量。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "84岁，容易瘀斑，肾功能未查。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": ["坦索罗辛"],
            },
            safety_profiles=[],
        )

        self.assertNotIn("布洛芬", result["treatment_plan"])
        self.assertIn("坦索罗辛", result["treatment_plan"])
        self.assertIn("监测尿量", result["treatment_plan"])

    def test_nsaid_is_removed_for_active_peptic_ulcer(self):
        result = apply_treatment_safety(
            "继续布洛芬缓解疼痛。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "正在服用布洛芬且患活动性胃溃疡。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": ["布洛芬"],
            },
            safety_profiles=[],
        )

        self.assertIn("nsaid_high_risk_present", {item["code"] for item in result["issues"]})
        self.assertNotIn("布洛芬", result["treatment_plan"])

    def test_patient_prefixed_advanced_age_is_detected(self):
        result = apply_treatment_safety(
            "确诊肾绞痛后可短期使用布洛芬镇痛。",
            diagnosis="输尿管结石",
            case_features={
                "case_text": "患者84岁，肾功能和出血用药风险尚未核对。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": [],
            },
            safety_profiles=[],
        )

        self.assertIn("nsaid_safety_facts_unresolved", {item["code"] for item in result["issues"]})
        self.assertNotIn("布洛芬", result["treatment_plan"])

    def test_short_nsaid_course_remains_when_specific_risks_are_denied(self):
        plan = "确诊肾绞痛后可短期使用布洛芬镇痛。"
        result = apply_treatment_safety(
            plan,
            diagnosis="输尿管结石",
            case_features={
                "case_text": "72岁，肾功能正常，否认胃溃疡、消化道出血、易瘀斑、抗凝和抗血小板用药。",
                "positive_findings": [],
                "immunosuppression": [],
                "red_flags": [],
                "medications": [],
            },
            safety_profiles=[],
        )

        self.assertNotIn("nsaid_safety_facts_unresolved", {item["code"] for item in result["issues"]})
        self.assertIn("布洛芬", result["treatment_plan"])


def urinary_stone_infection_case():
    return {
        "chat_history": [{"from": "patient", "text": "右侧腰痛，尿频尿急尿痛并有肉眼血尿。"}],
        "ordered_examinations": ["尿液分析（UA）", "尿培养"],
        "examination_results": {
            "尿液分析（UA）": {
                "result": {"白细胞酯酶": "阳性", "尿白细胞": "18/HPF", "尿红细胞": "25/HPF"}
            },
            "尿培养": {"result": {"生长情况": "无生长", "致病微生物": "未分离出"}},
        },
    }


def urinary_case_text():
    case_state = urinary_stone_infection_case()
    return " ".join(
        [case_state["chat_history"][0]["text"], str(case_state["examination_results"])]
    )


if __name__ == "__main__":
    unittest.main()
