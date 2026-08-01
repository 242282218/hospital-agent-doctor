"""Generalized offline regressions from the eleventh random batch."""

import unittest

from agent import legacy_orchestrator as agent_module
from agent.legacy_orchestrator import (
    apply_axis_risk_gate,
    build_name_map,
    extract_intake_facts,
    flatten_examination_catalog,
    load_knowledge_registry,
    normalize_diagnosis,
    open_coverage_gaps,
    select_allowed_candidate_diagnosis,
    select_diagnosis_axes,
    select_exam_plan,
    validate_axis_consult,
)


class EleventhRoundOfflineFailuresTest(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "实验室检查 - 血液": [
                "全血细胞计数（CBC）",
                "外周血涂片",
                "C反应蛋白（CRP）",
                "血清蛋白电泳（SPEP）",
            ],
            "实验室检查 - 免疫学": ["血清学抗体检测", "丙型肝炎病毒（HCV）抗体检测", "血清免疫电泳"],
            "实验室检查 - 微生物学": ["血培养"],
            "影像学检查 - CT": ["耳部CT扫描（Ear CT）"],
            "功能检查": ["听力测定", "鼓室压图及声导抗检查"],
        }
        self.item_map = build_name_map(flatten_examination_catalog(self.catalog))
        self.intent_rules = load_knowledge_registry()["exam_intent_map"]

    def test_relapsing_fever_bleeding_requires_exposure_followup(self):
        selector = getattr(agent_module, "select_required_intake_question", None)
        self.assertTrue(callable(selector), "required intake selector is missing")

        question = selector(relapsing_fever_bleeding_case(with_exams=False))

        self.assertTrue(any(marker in question for marker in ["蜱", "虫咬", "户外", "旅行"]))

    def test_answered_exposure_followup_is_not_repeated(self):
        selector = getattr(agent_module, "select_required_intake_question", None)
        self.assertTrue(callable(selector), "required intake selector is missing")
        case_state = relapsing_fever_bleeding_case(with_exams=False)
        case_state["chat_history"].append(
            {"from": "patient", "text": "近期没有旅行，也没去野外，没发现蜱虫或其他虫咬。"}
        )

        self.assertEqual("", selector(case_state))
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        axis = next(item for item in axes if item["axis_id"] == "systemic_infection_vs_primary_hematologic")
        self.assertNotIn("回归热", axis["candidate_official_names"])

    def test_generic_negative_answer_to_exposure_question_is_not_reasked(self):
        selector = getattr(agent_module, "select_required_intake_question", None)
        case_state = relapsing_fever_bleeding_case(with_exams=False)
        question = selector(case_state)
        case_state["chat_history"].extend(
            [
                {"from": "doctor", "text": question},
                {"from": "patient", "text": "您刚才问的这些情况都没有。"},
            ]
        )

        self.assertEqual("", selector(case_state))

    def test_empty_response_to_required_exposure_question_is_not_reasked(self):
        selector = getattr(agent_module, "select_required_intake_question", None)
        case_state = relapsing_fever_bleeding_case(with_exams=False)
        question = selector(case_state)
        case_state["chat_history"].extend(
            [
                {"from": "doctor", "text": question},
                {"from": "patient", "text": ""},
            ]
        )

        self.assertEqual("", selector(case_state))

    def test_postposed_negative_or_uncertain_vector_history_is_not_positive_exposure(self):
        for answer in ["蜱虫没见过，虫咬也没有。", "蜱虫叮咬不确定，记不清了。"]:
            with self.subTest(answer=answer):
                case_state = relapsing_fever_bleeding_case(with_exams=False)
                case_state["chat_history"].append({"from": "patient", "text": answer})
                axes = select_diagnosis_axes(extract_intake_facts(case_state))
                axis = next(
                    item for item in axes
                    if item["axis_id"] == "systemic_infection_vs_primary_hematologic"
                )

                self.assertNotIn("回归热", axis["candidate_official_names"])
                self.assertNotIn("媒介或暴露相关病原评估", axis["exam_intents"])

    def test_relapsing_fever_bleeding_keeps_infection_and_blood_axes(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        axis = next(item for item in axes if item["axis_id"] == "systemic_infection_vs_primary_hematologic")

        self.assertIn("细菌感染", axis["candidate_official_names"])
        self.assertIn("白血病", axis["candidate_official_names"])
        self.assertIn("特发性血小板减少性紫癜", axis["candidate_official_names"])
        self.assertNotIn("回归热", axis["candidate_official_names"])
        self.assertIn("infection_before_steroid", axis["treatment_risks"])

    def test_vector_exposure_allows_relapsing_fever_differential(self):
        case_state = relapsing_fever_bleeding_case()
        case_state["chat_history"].append(
            {"from": "patient", "text": "发病前去野外露营，被蜱虫叮咬过。"}
        )

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        axis = next(item for item in axes if item["axis_id"] == "systemic_infection_vs_primary_hematologic")

        self.assertIn("回归热", axis["candidate_official_names"])

    def test_vector_exposure_uses_directed_serology_without_generic_blood_culture(self):
        case_state = relapsing_fever_bleeding_case()
        case_state["chat_history"].append(
            {"from": "patient", "text": "发病前去野外露营，被蜱虫叮咬过。"}
        )
        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[],
            diagnosis_axes=axes,
            examination_catalog=self.catalog,
            item_name_map=self.item_map,
            diagnosis_exam_profiles=[],
            exam_intent_rules=self.intent_rules,
            max_items=3,
        )

        self.assertIn("外周血涂片", plan["examinations"])
        self.assertIn("血清学抗体检测", plan["examinations"])
        self.assertNotIn("血培养", plan["examinations"])

    def test_systemic_infection_evidence_gap_blocks_primary_blood_closure(self):
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(relapsing_fever_bleeding_case())}

        self.assertIn("systemic_infection_vs_primary_hematologic", gap_ids)

    def test_systemic_infection_gap_closes_after_first_line_coverage_results(self):
        case_state = relapsing_fever_bleeding_case()
        case_state["ordered_examinations"].extend(["外周血涂片", "血培养", "C反应蛋白（CRP）"])
        case_state["examination_results"].update(
            {
                "外周血涂片": {"status": "normal", "result": {"形态": "未见原始细胞"}},
                "血培养": {"status": "normal", "result": {"培养": "无生长"}},
                "C反应蛋白（CRP）": {"status": "normal", "result": {"CRP": "2 mg/L"}},
            }
        )

        self.assertNotIn(
            "systemic_infection_vs_primary_hematologic",
            {item["gap_id"] for item in open_coverage_gaps(case_state)},
        )

    def test_systemic_infection_axis_maps_pathogen_and_smear_exams(self):
        case_state = relapsing_fever_bleeding_case()
        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[],
            diagnosis_axes=axes,
            examination_catalog=self.catalog,
            item_name_map=self.item_map,
            diagnosis_exam_profiles=[],
            exam_intent_rules=self.intent_rules,
            max_items=3,
        )

        self.assertIn("外周血涂片", plan["examinations"])
        self.assertTrue(
            {"血清学抗体检测", "血培养"}.intersection(plan["examinations"])
        )

    def test_systemic_bleeding_axis_without_prior_exams_includes_cbc(self):
        case_state = relapsing_fever_bleeding_case(with_exams=False)
        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[],
            diagnosis_axes=axes,
            examination_catalog=self.catalog,
            item_name_map=self.item_map,
            diagnosis_exam_profiles=[],
            exam_intent_rules=self.intent_rules,
            max_items=3,
        )

        self.assertIn("全血细胞计数（CBC）", plan["examinations"])
        self.assertIn("外周血涂片", plan["examinations"])
        self.assertTrue({"血清学抗体检测", "血培养"}.intersection(plan["examinations"]))

    def test_unclosed_systemic_infection_removes_steroid(self):
        case_state = relapsing_fever_bleeding_case()
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = apply_axis_risk_gate(
            "立即给予泼尼松治疗血小板减少。",
            {
                "case_text": case_text(case_state),
                "positive_findings": ["反复高热", "多部位出血"],
                "red_flags": ["高热"],
                "organ_risk": [],
                "diagnosis_axes": axes,
                "treatment_risks": [],
            },
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
        self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_unclosed_systemic_infection_removes_common_systemic_steroid_aliases(self):
        case_state = relapsing_fever_bleeding_case()
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        for drug in [
            "甲泼尼龙",
            "泼尼松龙",
            "氢化可的松",
            "氟米龙",
            "氯替泼诺",
            "倍他米松",
            "prednisolone",
            "methylprednisolone",
            "dexamethasone",
            "betamethasone",
            "fluorometholone",
            "loteprednol",
            "prednisone",
            "强的松",
            "hydrocortisone",
        ]:
            with self.subTest(drug=drug):
                result = apply_axis_risk_gate(
                    "立即给予%s治疗。" % drug,
                    {
                        "case_text": case_text(case_state),
                        "positive_findings": ["反复高热", "多部位出血"],
                        "red_flags": ["高热"],
                        "organ_risk": [],
                        "diagnosis_axes": axes,
                        "treatment_risks": [],
                    },
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
                self.assertNotIn(drug, result["treatment_plan"])

    def test_negated_steroid_or_estrogen_mention_does_not_trigger_steroid_gate(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        plans = [
            "避免使用氟米龙滴眼液，改用人工泪液。",
            "停用 prednisolone acetate eye drops，继续局部护理。",
            "既往使用雌激素避孕药，本次不涉及糖皮质激素治疗。",
        ]
        for plan in plans:
            with self.subTest(plan=plan):
                result = apply_axis_risk_gate(
                    plan,
                    {
                        "case_text": case_text(relapsing_fever_bleeding_case()),
                        "diagnosis_axes": axes,
                        "treatment_risks": [],
                    },
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertNotIn("infection_before_steroid", {item["code"] for item in result["issues"]})
                self.assertEqual(plan, result["treatment_plan"])

    def test_double_negative_steroid_continuation_is_still_blocked(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        plans = [
            "无需停用泼尼松，继续原剂量。",
            "不建议停用 methylprednisolone，继续治疗。",
            "未停用强的松，仍在继续治疗。",
        ]
        for plan in plans:
            with self.subTest(plan=plan):
                result = apply_axis_risk_gate(
                    plan,
                    {
                        "case_text": case_text(relapsing_fever_bleeding_case()),
                        "diagnosis_axes": axes,
                        "treatment_risks": [],
                    },
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})

    def test_life_threatening_bleeding_allows_monitored_steroid_bridge(self):
        case_state = relapsing_fever_bleeding_case()
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = apply_axis_risk_gate(
            "因危及生命的颅内出血，在血液科与感染科监护下同步完成病原评估和抗感染处置，给予泼尼松作为桥接抢救。",
            {
                "case_text": case_text(case_state) + " 已发生颅内出血和血流动力学不稳定。",
                "positive_findings": ["反复高热", "颅内出血"],
                "red_flags": ["血流动力学不稳定"],
                "organ_risk": [],
                "diagnosis_axes": axes,
                "treatment_risks": [],
            },
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertIn("泼尼松", result["treatment_plan"])
        self.assertNotIn("infection_before_steroid", {item["code"] for item in result["issues"]})

    def test_negated_life_threatening_bleeding_does_not_allow_steroid_bridge(self):
        case_state = relapsing_fever_bleeding_case()
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = apply_axis_risk_gate(
            "在血液科与感染科监护下同步完成病原评估和抗感染处置，给予泼尼松桥接抢救。",
            {
                "case_text": case_text(case_state) + " 头颅CT未见颅内出血，血流动力学稳定。",
                "positive_findings": ["反复高热"],
                "red_flags": [],
                "organ_risk": [],
                "diagnosis_axes": axes,
                "treatment_risks": [],
            },
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
        self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_negated_monitoring_does_not_allow_steroid_bridge(self):
        case_state = relapsing_fever_bleeding_case()
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        result = apply_axis_risk_gate(
            "无需感染科监护，暂不同步病原评估和抗感染处置，仍给予泼尼松桥接抢救。",
            {
                "case_text": case_text(case_state) + " 已发生颅内出血。",
                "positive_findings": ["颅内出血"],
                "red_flags": [],
                "organ_risk": [],
                "diagnosis_axes": axes,
                "treatment_risks": [],
            },
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
        self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_unavailable_or_consult_only_specialists_do_not_allow_steroid_bridge(self):
        case_state = relapsing_fever_bleeding_case()
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plans = [
            "血液科无法监护，感染科无法参与，但同步完成病原评估和抗感染处置，仍给予泼尼松桥接抢救。",
            "联系血液科和感染科咨询意见，同步完成病原评估和抗感染处置，仍给予泼尼松桥接抢救。",
        ]
        for plan in plans:
            with self.subTest(plan=plan):
                result = apply_axis_risk_gate(
                    plan,
                    {
                        "case_text": case_text(case_state) + " 已发生颅内出血。",
                        "positive_findings": ["颅内出血"],
                        "red_flags": [],
                        "organ_risk": [],
                        "diagnosis_axes": axes,
                        "treatment_risks": [],
                    },
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
                self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_resolved_life_threatening_findings_do_not_allow_steroid_bridge(self):
        case_state = relapsing_fever_bleeding_case()
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        resolved_findings = [
            "颅内出血已排除。",
            "血流动力学不稳定已纠正，目前稳定。",
        ]
        plan = "在血液科与感染科共同监护下同步完成病原评估和抗感染处置，给予泼尼松桥接抢救。"
        for finding in resolved_findings:
            with self.subTest(finding=finding):
                result = apply_axis_risk_gate(
                    plan,
                    {
                        "case_text": case_text(case_state) + finding,
                        "positive_findings": [],
                        "red_flags": [],
                        "organ_risk": [],
                        "diagnosis_axes": axes,
                        "treatment_risks": [],
                    },
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
                self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_historical_or_family_bleeding_does_not_allow_steroid_bridge(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        histories = [
            "父亲曾发生颅内出血，我本人没有活动性大出血。",
            "三年前曾有颅内出血，现已痊愈，本次没有出血。",
            "既往有失血性休克，本次生命体征平稳。",
        ]
        plan = "在血液科与感染科共同监护下同步完成病原评估和抗感染处置，给予泼尼松桥接抢救。"
        for patient_text in histories:
            with self.subTest(patient_text=patient_text):
                result = apply_axis_risk_gate(
                    plan,
                    {
                        "patient_text": patient_text,
                        "diagnosis_axes": axes,
                        "treatment_risks": [],
                        "examination_results": {},
                    },
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})

    def test_resolved_other_field_does_not_hide_active_intracranial_bleeding(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        result = apply_axis_risk_gate(
            "在血液科与感染科共同监护下同步完成病原评估和抗感染处置，给予泼尼松桥接抢救。",
            {
                "patient_text": "患者当前发生危及生命的活动性出血。",
                "diagnosis_axes": axes,
                "treatment_risks": [],
                "examination_results": {
                    "头颅CT": {
                        "status": "abnormal",
                        "result": {"颅内出血": "明确", "其他出血": "已缓解"},
                    }
                },
            },
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertNotIn("infection_before_steroid", {item["code"] for item in result["issues"]})

    def test_doctor_life_threatening_question_is_not_positive_evidence(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        for exam_value in ["未见", "无", "无异常", "无证据", "阴性", "排除"]:
            with self.subTest(exam_value=exam_value):
                result = apply_axis_risk_gate(
                    "在血液科与感染科监护下同步完成病原评估和抗感染处置，给予泼尼松桥接抢救。",
                    {
                        "case_text": "医生：是否发生颅内出血或血流动力学不稳定？患者：没有。",
                        "patient_text": "没有颅内出血，血流动力学稳定。",
                        "positive_findings": [],
                        "red_flags": [],
                        "organ_risk": [],
                        "diagnosis_axes": axes,
                        "treatment_risks": [],
                        "examination_results": {
                            "头颅CT": {"status": "normal", "result": {"颅内出血": exam_value}},
                        },
                    },
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
                self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_corneal_infection_axis_cannot_use_hematologic_steroid_bridge(self):
        result = apply_axis_risk_gate(
            "在血液科与感染科监护下同步完成病原评估和抗感染处置，给予泼尼松桥接抢救。",
            {
                "case_text": "角膜感染伴颅内出血。",
                "positive_findings": ["角膜感染", "颅内出血"],
                "red_flags": ["颅内出血"],
                "organ_risk": [],
                "diagnosis_axes": [
                    {
                        "axis_id": "corneal_infection_with_target_rash",
                        "evidence": ["角膜感染", "靶形皮疹"],
                        "candidate_official_names": ["角膜炎"],
                        "treatment_risks": ["infection_before_steroid"],
                    }
                ],
                "treatment_risks": [],
            },
            diagnosis="角膜炎",
        )

        self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
        self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_single_negative_blood_culture_does_not_resolve_steroid_risk(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        result = apply_axis_risk_gate(
            "确诊后给予泼尼松治疗。",
            systemic_infection_safety_features(axes, include_serology=False, include_activity=False),
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
        self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_resolved_infection_workup_with_isolated_thrombocytopenia_allows_steroid(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        result = apply_axis_risk_gate(
            "由血液科确诊后给予泼尼松治疗。",
            systemic_infection_safety_features(axes),
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertNotIn("infection_before_steroid", {item["code"] for item in result["issues"]})
        self.assertIn("泼尼松", result["treatment_plan"])

    def test_positive_pathogen_result_still_blocks_steroid(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        result = apply_axis_risk_gate(
            "确诊后给予泼尼松治疗。",
            systemic_infection_safety_features(axes, pathogen_positive=True),
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
        self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_missing_or_abnormal_smear_does_not_resolve_steroid_risk(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        for kwargs in [{"include_smear": False}, {"smear_abnormal": True}]:
            with self.subTest(**kwargs):
                result = apply_axis_risk_gate(
                    "确诊后给予泼尼松治疗。",
                    systemic_infection_safety_features(axes, **kwargs),
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})
                self.assertNotIn("泼尼松", result["treatment_plan"])

    def test_mixed_positive_pathogen_payload_does_not_resolve_steroid_risk(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        features = systemic_infection_safety_features(axes)
        features["examination_results"]["血清学抗体检测"] = {
            "status": "abnormal",
            "result": {"阴性对照": "正常", "病原抗体": "阳性"},
        }

        result = apply_axis_risk_gate(
            "确诊后给予泼尼松治疗。",
            features,
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})

    def test_pending_pathogen_result_does_not_resolve_steroid_risk(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        features = systemic_infection_safety_features(axes)
        features["examination_results"]["血培养复查"] = {
            "status": "pending",
            "result": {"培养结果": "培养阴性，待复核"},
        }

        result = apply_axis_risk_gate(
            "确诊后给予泼尼松治疗。",
            features,
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})

    def test_pending_infection_closure_components_do_not_resolve_steroid_risk(self):
        axes = select_diagnosis_axes(extract_intake_facts(relapsing_fever_bleeding_case()))
        pending_payloads = [
            (
                "外周血涂片",
                {"status": "pending", "result": {"细胞形态": "未见原始细胞或异常细胞，待复核"}},
            ),
            (
                "全血细胞计数（CBC）",
                {
                    "status": "pending",
                    "result": {
                        "血白细胞计数": "7.2 x10^9/L [参考值：4.0-10.0]",
                        "血红蛋白": "132 g/L [参考值：110-155]",
                        "血小板计数": "62 x10^9/L [参考值：150-450]",
                    },
                },
            ),
            (
                "C反应蛋白（CRP）",
                {"status": "normal", "result": {"C反应蛋白": "待报告"}},
            ),
        ]
        for exam_name, payload in pending_payloads:
            with self.subTest(exam_name=exam_name):
                features = systemic_infection_safety_features(axes)
                features["examination_results"][exam_name] = payload

                result = apply_axis_risk_gate(
                    "确诊后给予泼尼松治疗。",
                    features,
                    diagnosis="特发性血小板减少性紫癜",
                )

                self.assertIn("infection_before_steroid", {item["code"] for item in result["issues"]})

    def test_vector_path_uses_negative_directed_serology_for_resolution(self):
        case_state = relapsing_fever_bleeding_case()
        case_state["chat_history"].append(
            {"from": "patient", "text": "发病前去野外露营，被蜱虫叮咬过。"}
        )
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        features = systemic_infection_safety_features(axes, include_culture=False)

        result = apply_axis_risk_gate(
            "由血液科确诊后给予泼尼松治疗。",
            features,
            diagnosis="特发性血小板减少性紫癜",
        )

        self.assertNotIn("infection_before_steroid", {item["code"] for item in result["issues"]})
        self.assertIn("泼尼松", result["treatment_plan"])

    def test_focal_ear_pain_with_conductive_loss_keeps_local_axis(self):
        axes = select_diagnosis_axes(extract_intake_facts(focal_ear_case()))
        axis = next(item for item in axes if item["axis_id"] == "focal_ear_pain_conductive_loss")

        candidates = set(axis["candidate_official_names"])
        self.assertTrue(candidates)
        self.assertTrue(candidates.issubset({"中耳炎", "外耳脓肿", "急性鼓膜炎", "乳突炎", "外耳道胆脂瘤"}))
        self.assertNotIn("外耳脓肿", candidates)
        self.assertNotIn("贫血", candidates)

    def test_focal_ear_pain_needs_structural_localization_after_normal_otoscopy(self):
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(focal_ear_case())}

        self.assertIn("focal_ear_pain_structural_localization", gap_ids)

    def test_abnormal_middle_ear_mechanism_result_closes_ear_localization_gap(self):
        case_state = focal_ear_case()
        case_state["ordered_examinations"].append("鼓室压图及声导抗检查")
        case_state["examination_results"]["鼓室压图及声导抗检查"] = {
            "status": "abnormal",
            "result": {"鼓室图": "B型，提示中耳传导机制异常"},
        }

        self.assertNotIn(
            "focal_ear_pain_structural_localization",
            {item["gap_id"] for item in open_coverage_gaps(case_state)},
        )

    def test_completed_abnormal_tympanometry_result_does_not_advance_to_ct(self):
        case_state = focal_ear_case()
        case_state["ordered_examinations"].append("鼓室压图及声导抗检查")
        case_state["examination_results"]["鼓室压图及声导抗检查"] = {
            "status": "completed",
            "result": {"鼓室图": "B型，提示中耳传导机制异常"},
        }

        gaps = open_coverage_gaps(case_state)

        self.assertNotIn("focal_ear_pain_structural_localization", {item["gap_id"] for item in gaps})
        self.assertNotIn("耳部CT扫描（Ear CT）", {exam for item in gaps for exam in item["required_exams"]})

    def test_normal_middle_ear_mechanism_advances_to_deep_ear_localization(self):
        case_state = focal_ear_case()
        case_state["ordered_examinations"].append("鼓室压图及声导抗检查")
        case_state["examination_results"]["鼓室压图及声导抗检查"] = {
            "status": "normal",
            "result": {"鼓室图": "A型，未解释传导性听力下降"},
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
            max_items=3,
        )

        self.assertIn("耳部CT扫描（Ear CT）", plan["examinations"])
        gap = next(item for item in open_coverage_gaps(case_state) if item["gap_id"] == "focal_ear_pain_structural_localization")
        self.assertNotIn("持续", gap["reason"])

    def test_pending_middle_ear_result_does_not_advance_to_ct(self):
        case_state = focal_ear_case()
        case_state["ordered_examinations"].append("鼓室压图及声导抗检查")
        case_state["examination_results"]["鼓室压图及声导抗检查"] = {
            "status": "pending",
            "result": {"鼓室图": "待报告"},
        }

        gaps = open_coverage_gaps(case_state)

        self.assertIn("focal_ear_pain_structural_localization", {item["gap_id"] for item in gaps})
        self.assertNotIn("耳部深层结构定位", {intent for gap in gaps for intent in gap["exam_intents"]})

    def test_unconfirmed_anemia_is_demoted_and_ruled_out_cerumen_is_pruned(self):
        pruner = getattr(agent_module, "prune_unsupported_disease_candidates", None)
        self.assertTrue(callable(pruner), "candidate evidence pruner is missing")
        candidates = [
            {"disease": "贫血", "score": 50},
            {"disease": "耵聍栓塞", "score": 49},
            {"disease": "外耳脓肿", "score": 48, "source": "diagnosis_axis"},
        ]

        result = pruner(candidates, focal_ear_case())
        by_name = {item["disease"]: item for item in result}

        self.assertIn("贫血", by_name)
        self.assertEqual("secondary", by_name["贫血"]["role"])
        self.assertLess(by_name["贫血"]["score"], by_name["外耳脓肿"]["score"])
        self.assertNotIn("耵聍栓塞", by_name)
        self.assertIn("外耳脓肿", by_name)

    def test_objective_low_hemoglobin_keeps_anemia_candidate(self):
        pruner = getattr(agent_module, "prune_unsupported_disease_candidates", None)
        self.assertTrue(callable(pruner), "candidate evidence pruner is missing")
        case_state = focal_ear_case()
        case_state["ordered_examinations"].append("全血细胞计数（CBC）")
        case_state["examination_results"]["全血细胞计数（CBC）"] = {
            "status": "abnormal",
            "result": {"血红蛋白": "82 g/L [参考值：110-155]"},
        }

        result = pruner([{"disease": "贫血", "score": 50}], case_state)

        self.assertEqual(["贫血"], [item["disease"] for item in result])

    def test_normal_hemoglobin_prunes_anemia_candidate(self):
        pruner = getattr(agent_module, "prune_unsupported_disease_candidates", None)
        self.assertTrue(callable(pruner), "candidate evidence pruner is missing")
        case_state = focal_ear_case()
        case_state["ordered_examinations"].append("全血细胞计数（CBC）")
        case_state["examination_results"]["全血细胞计数（CBC）"] = {
            "status": "normal",
            "result": {"血红蛋白": "132 g/L [参考值：110-155]"},
        }

        result = pruner([{"disease": "贫血", "score": 50}], case_state)

        self.assertEqual([], result)

    def test_final_diagnosis_cannot_bypass_pruned_candidate_pool(self):
        diagnosis = select_allowed_candidate_diagnosis(
            {"normalized_diagnosis": "贫血", "source": "official_catalog"},
            [{"disease": "中耳炎", "score": 48}],
            default_diagnosis="中耳炎",
        )

        self.assertEqual("中耳炎", diagnosis)

    def test_candidate_fallback_discards_rejected_diagnosis_plan_and_reasoning(self):
        reconciler = getattr(agent_module, "reconcile_selected_diagnosis_plan", None)
        self.assertTrue(callable(reconciler), "diagnosis-plan reconciler is missing")

        treatment, reasoning = reconciler(
            {"normalized_diagnosis": "贫血"},
            selected_diagnosis="中耳炎",
            treatment_plan="口服铁剂并复查血红蛋白。",
            reasoning="贫血可解释全部症状。",
            default_reasoning="默认推理。",
        )

        self.assertNotIn("铁", treatment)
        self.assertNotIn("贫血", reasoning)
        self.assertIn("中耳炎", reasoning)

        original = reconciler(
            {"normalized_diagnosis": "中耳炎"},
            selected_diagnosis="中耳炎",
            treatment_plan="按中耳炎制定治疗。",
            reasoning="现有证据支持中耳炎。",
            default_reasoning="默认推理。",
        )
        self.assertEqual(("按中耳炎制定治疗。", "现有证据支持中耳炎。"), original)

    def test_hearing_loss_without_ear_swelling_uses_accurate_ear_fact_label(self):
        case_state = focal_ear_case()
        case_state["chat_history"] = [
            {"from": "patient", "text": "单侧耳内针扎样剧痛，听力明显下降，但没有耳部肿胀。"}
        ]

        facts = extract_intake_facts(case_state)
        labels = {item["label"] for item in facts["symptom_clusters"]}

        self.assertIn("局灶重度耳痛伴局部功能受损", labels)
        self.assertNotIn("局灶重度耳痛肿胀", labels)

    def test_objective_conductive_loss_can_supply_local_ear_function_loss(self):
        case_state = focal_ear_case()
        case_state["chat_history"] = [
            {"from": "patient", "text": "右耳内针扎样剧痛，但自觉听力没有下降，也没有耳部肿胀。"}
        ]

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertIn("focal_ear_pain_conductive_loss", {item["axis_id"] for item in axes})
        self.assertIn("focal_ear_pain_structural_localization", {item["gap_id"] for item in open_coverage_gaps(case_state)})

    def test_negative_structured_hearing_result_does_not_open_conductive_axis(self):
        result_formats = [
            {"传导性听力损失": "未见"},
            {"气骨导差": "无"},
        ]
        for result in result_formats:
            with self.subTest(result=result):
                case_state = focal_ear_case()
                case_state["examination_results"]["听力测定"] = {
                    "status": "normal",
                    "result": result,
                }

                axes = select_diagnosis_axes(extract_intake_facts(case_state))

                self.assertNotIn("focal_ear_pain_conductive_loss", {item["axis_id"] for item in axes})

    def test_common_severe_ear_pain_phrases_open_objective_conductive_axis(self):
        descriptions = [
            "右耳剧痛，听力下降。",
            "左耳疼痛8/10，听力下降。",
            "单侧耳痛明显，听力下降。",
        ]
        for description in descriptions:
            with self.subTest(description=description):
                case_state = focal_ear_case()
                case_state["chat_history"] = [{"from": "patient", "text": description}]

                axes = select_diagnosis_axes(extract_intake_facts(case_state))

                self.assertIn("focal_ear_pain_conductive_loss", {item["axis_id"] for item in axes})

    def test_unrelated_swelling_does_not_open_focal_ear_axis(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "单侧耳内针扎样剧痛，脚踝肿胀，但听力没有下降。"}],
            "ordered_examinations": [],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertNotIn("focal_ear_pain_conductive_loss", {item["axis_id"] for item in axes})

    def test_low_mch_does_not_override_normal_hemoglobin(self):
        pruner = getattr(agent_module, "prune_unsupported_disease_candidates", None)
        case_state = focal_ear_case()
        case_state["ordered_examinations"].append("全血细胞计数（CBC）")
        case_state["examination_results"]["全血细胞计数（CBC）"] = {
            "status": "abnormal",
            "result": {
                "血红蛋白": "132 g/L [参考值：110-155]",
                "平均红细胞血红蛋白含量（MCH）": "25 pg [参考值：27-34]",
            },
        }

        result = pruner([{"disease": "贫血", "score": 50}], case_state)

        self.assertEqual([], result)

    def test_normal_cbc_range_format_prunes_anemia_candidate(self):
        pruner = getattr(agent_module, "prune_unsupported_disease_candidates", None)
        case_state = focal_ear_case()
        case_state["ordered_examinations"].append("全血细胞计数（CBC）")
        case_state["examination_results"]["全血细胞计数（CBC）"] = {
            "status": "normal",
            "result": {"血红蛋白": "120-160 g/L"},
        }

        result = pruner([{"disease": "贫血", "score": 50}], case_state)

        self.assertEqual([], result)

    def test_cryoglobulinemia_keeps_secondary_cause_axis(self):
        axes = select_diagnosis_axes(extract_intake_facts(cryoglobulinemia_case()))
        axis = next(item for item in axes if item["axis_id"] == "cryoglobulinemia_secondary_cause")

        self.assertIn("冷球蛋白血症", axis["candidate_official_names"])
        self.assertNotIn("多发性骨髓瘤", axis["candidate_official_names"])
        self.assertNotIn("急性丙型肝炎", axis["candidate_official_names"])
        self.assertIn("单克隆蛋白或浆细胞病评估", axis["exam_intents"])
        self.assertIn("HCV病因评估", axis["exam_intents"])

    def test_confirmed_cryoglobulin_with_neuropathy_opens_secondary_cause_axis(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "近期出现双足麻木和感觉减退。"}],
            "ordered_examinations": ["冷球蛋白检测"],
            "examination_results": {
                "冷球蛋白检测": {"status": "abnormal", "result": {"结论": "阳性"}},
            },
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertIn("cryoglobulinemia_secondary_cause", {item["axis_id"] for item in axes})
        self.assertIn("cryoglobulinemia_secondary_cause", {item["gap_id"] for item in open_coverage_gaps(case_state)})

    def test_confirmed_cryoglobulin_accepts_common_compatible_manifestations(self):
        manifestations = [
            "多发性单神经炎伴足下垂。",
            "双下肢网状青斑。",
            "手足烧灼痛和针刺感。",
            "反复关节炎。",
        ]
        for manifestation in manifestations:
            with self.subTest(manifestation=manifestation):
                case_state = {
                    "chat_history": [{"from": "patient", "text": manifestation}],
                    "ordered_examinations": ["冷球蛋白检测"],
                    "examination_results": {
                        "冷球蛋白检测": {"status": "abnormal", "result": {"结论": "阳性"}},
                    },
                }

                axes = select_diagnosis_axes(extract_intake_facts(case_state))

                self.assertIn("cryoglobulinemia_secondary_cause", {item["axis_id"] for item in axes})

    def test_cryoglobulin_embedded_in_cbc_payload_opens_secondary_cause_axis(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "双足麻木并有可触及紫癜。"}],
            "ordered_examinations": ["全血细胞计数（CBC）"],
            "examination_results": {
                "全血细胞计数（CBC）": {
                    "status": "abnormal",
                    "result": {
                        "血红蛋白": "正常",
                        "冷球蛋白": "620 ug/mL",
                        "温度特性": "4C沉淀，加温后溶解",
                    },
                },
            },
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertIn("cryoglobulinemia_secondary_cause", {item["axis_id"] for item in axes})
        self.assertIn("cryoglobulinemia_secondary_cause", {item["gap_id"] for item in open_coverage_gaps(case_state)})

    def test_embedded_cryoglobulin_uses_local_polarity_not_cbc_status(self):
        result_formats = [
            ("abnormal", {"血小板": "20 x10^9/L", "补充描述": "冷球蛋白未检出"}, False),
            ("abnormal", {"血小板": "20 x10^9/L", "补充描述": "冷球蛋白初筛阳性，待复核"}, False),
            ("completed", {"补充描述": "冷球蛋白阳性"}, True),
        ]
        for status, result, expected in result_formats:
            with self.subTest(status=status, result=result):
                case_state = {
                    "chat_history": [{"from": "patient", "text": "双下肢可触及紫癜。"}],
                    "ordered_examinations": ["全血细胞计数（CBC）"],
                    "examination_results": {
                        "全血细胞计数（CBC）": {"status": status, "result": result},
                    },
                }

                axes = select_diagnosis_axes(extract_intake_facts(case_state))

                self.assertEqual(
                    expected,
                    "cryoglobulinemia_secondary_cause" in {item["axis_id"] for item in axes},
                )

    def test_standard_cryoglobulin_abnormal_formats_open_secondary_cause_axis(self):
        result_formats = [
            ("abnormal", {"定量": "620 ug/mL"}),
            ("completed", {"结论": "阳性"}),
            ("completed", {"结论": "阳性（参考值：阴性）"}),
            ("completed", {"结论": "阳性", "参考范围": "阴性"}),
            ("completed", {"结论": "阳性，超过临界值"}),
            ("completed", {"定量": "120 mg/L（参考范围 0-20 mg/L）"}),
            ("completed", {"cryocrit": "8%（参考范围 0-1%）"}),
            ("completed", {"cryocrit": "8%（参考值<1%）"}),
        ]
        for status, result in result_formats:
            with self.subTest(status=status, result=result):
                case_state = {
                    "chat_history": [{"from": "patient", "text": "双下肢可触及紫癜。"}],
                    "ordered_examinations": ["冷球蛋白检测"],
                    "examination_results": {
                        "冷球蛋白检测": {"status": status, "result": result},
                    },
                }

                axes = select_diagnosis_axes(extract_intake_facts(case_state))

                self.assertIn("cryoglobulinemia_secondary_cause", {item["axis_id"] for item in axes})

    def test_pending_or_negative_cryoglobulin_does_not_open_secondary_cause_axis(self):
        result_formats = [
            ("pending", {"结论": "阳性，待复核"}),
            ("normal", {"结论": "阴性"}),
            ("completed", {"结论": "阴性"}),
            ("abnormal", {"结论": "未见冷球蛋白"}),
            ("completed", {"冷球蛋白": "未检出"}),
            ("completed", {"结论": "未见"}),
            ("completed", {"结论": "不升高"}),
            ("completed", {"标本温度": "40 ℃（参考范围 2-8 ℃）"}),
            ("completed", {"结论": "阳性，待复核"}),
            ("completed", {"结论": "未达到阳性阈值"}),
            ("completed", {"结论": "阴性（阳性判定阈值≥20）"}),
            ("abnormal", {"结论": "阳性，待复核"}),
        ]
        for status, result in result_formats:
            with self.subTest(status=status, result=result):
                case_state = {
                    "chat_history": [{"from": "patient", "text": "双下肢可触及紫癜。"}],
                    "ordered_examinations": ["冷球蛋白检测"],
                    "examination_results": {
                        "冷球蛋白检测": {"status": status, "result": result},
                    },
                }

                axes = select_diagnosis_axes(extract_intake_facts(case_state))

                self.assertNotIn("cryoglobulinemia_secondary_cause", {item["axis_id"] for item in axes})
                self.assertNotIn(
                    "cryoglobulinemia_secondary_cause",
                    {item["gap_id"] for item in open_coverage_gaps(case_state)},
                )

    def test_confirmed_cryoglobulin_without_compatible_manifestation_does_not_open_axis(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "紫癜没有，雷诺也没有，关节痛也没有，双足麻木也没有，皮肤溃疡也没有。",
                }
            ],
            "ordered_examinations": ["冷球蛋白检测"],
            "examination_results": {
                "冷球蛋白检测": {"status": "abnormal", "result": {"结论": "阳性"}},
            },
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertNotIn("cryoglobulinemia_secondary_cause", {item["axis_id"] for item in axes})
        self.assertNotIn("cryoglobulinemia_secondary_cause", {item["gap_id"] for item in open_coverage_gaps(case_state)})

    def test_llm_axis_cannot_override_pending_cryoglobulin_result(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "双下肢可触及紫癜。"}],
            "ordered_examinations": ["冷球蛋白检测"],
            "examination_results": {
                "冷球蛋白检测": {"status": "pending", "result": {"冷球蛋白": "待报告"}},
            },
        }
        axis_ids = [
            "cryoglobulinemia_secondary_cause",
            "cryoglobulinemia_secondary_etiology",
            "secondary_cryoglobulinemia_workup",
        ]
        for axis_id in axis_ids:
            with self.subTest(axis_id=axis_id):
                consult = validate_axis_consult(
                    {
                        "diagnosis_axes": [
                            {
                                "axis_id": axis_id,
                                "evidence": ["紫癜", "冷球蛋白"],
                                "candidate_official_names": ["冷球蛋白血症"],
                                "exam_intents": ["HCV病因评估"],
                            }
                        ]
                    },
                    case_state=case_state,
                    official_diseases=["冷球蛋白血症"],
                )

                self.assertNotIn(axis_id, {item["axis_id"] for item in consult["diagnosis_axes"]})

        generic_consult = validate_axis_consult(
            {
                "diagnosis_axes": [
                    {
                        "axis_id": "secondary_vasculitis_etiology",
                        "evidence": ["紫癜", "冷球蛋白"],
                        "candidate_official_names": [],
                        "exam_intents": ["HCV病因评估"],
                    }
                ]
            },
            case_state=case_state,
            official_diseases=["冷球蛋白血症"],
        )
        self.assertNotIn(
            "secondary_vasculitis_etiology",
            {item["axis_id"] for item in generic_consult["diagnosis_axes"]},
        )

    def test_cryoglobulinemia_secondary_cause_gap_maps_spep_and_hcv(self):
        case_state = cryoglobulinemia_case()
        gap_ids = {item["gap_id"] for item in open_coverage_gaps(case_state)}
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[],
            diagnosis_axes=axes,
            examination_catalog=self.catalog,
            item_name_map=self.item_map,
            diagnosis_exam_profiles=[],
            exam_intent_rules=self.intent_rules,
            max_items=3,
        )

        self.assertIn("cryoglobulinemia_secondary_cause", gap_ids)
        self.assertIn("血清蛋白电泳（SPEP）", plan["examinations"])
        self.assertIn("丙型肝炎病毒（HCV）抗体检测", plan["examinations"])

    def test_cryoglobulinemia_etiology_gap_closes_after_first_line_results(self):
        case_state = cryoglobulinemia_case()
        case_state["ordered_examinations"].extend(
            ["血清蛋白电泳（SPEP）", "丙型肝炎病毒（HCV）抗体检测"]
        )
        case_state["examination_results"].update(
            {
                "血清蛋白电泳（SPEP）": {"status": "normal", "result": {"单克隆峰": "未见"}},
                "丙型肝炎病毒（HCV）抗体检测": {"status": "normal", "result": {"HCV抗体": "阴性"}},
            }
        )

        self.assertNotIn(
            "cryoglobulinemia_secondary_cause",
            {item["gap_id"] for item in open_coverage_gaps(case_state)},
        )

    def test_pending_cryoglobulinemia_etiology_results_do_not_close_gap(self):
        case_state = cryoglobulinemia_case()
        case_state["ordered_examinations"].extend(
            ["血清蛋白电泳（SPEP）", "丙型肝炎病毒（HCV）抗体检测"]
        )
        case_state["examination_results"].update(
            {
                "血清蛋白电泳（SPEP）": {"status": "pending", "result": {"单克隆峰": "待报告"}},
                "丙型肝炎病毒（HCV）抗体检测": {"status": "pending", "result": {"HCV抗体": "待报告"}},
            }
        )

        self.assertIn(
            "cryoglobulinemia_secondary_cause",
            {item["gap_id"] for item in open_coverage_gaps(case_state)},
        )

    def test_raynaud_without_confirmed_cryoglobulin_does_not_open_etiology_axis(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "遇冷手指变白变青，偶有紫癜。"}],
            "ordered_examinations": ["尿液分析（UA）"],
            "examination_results": {
                "尿液分析（UA）": {"status": "normal", "result": {"尿蛋白": "阴性", "尿红细胞": "0/HPF"}}
            },
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertNotIn("cryoglobulinemia_secondary_cause", {item["axis_id"] for item in axes})
        self.assertNotIn("cryoglobulinemia_secondary_cause", {item["gap_id"] for item in open_coverage_gaps(case_state)})

    def test_nonconductive_or_mild_ear_symptoms_do_not_force_structural_imaging(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "耳朵轻微不适，没有肿胀。"}],
            "ordered_examinations": ["耳镜检查", "听力测定"],
            "examination_results": {
                "耳镜检查": {"status": "normal", "result": {"外耳道": "通畅", "鼓膜": "完整"}},
                "听力测定": {"status": "normal", "result": {"补充描述": "双耳听力正常"}},
            },
        }

        self.assertNotIn(
            "focal_ear_pain_structural_localization",
            {item["gap_id"] for item in open_coverage_gaps(case_state)},
        )

    def test_axis_validator_accepts_supported_rephrasing_but_rejects_hallucination(self):
        supported = validate_axis_consult(
            {
                "diagnosis_axes": [
                    {
                        "axis_id": "conductive_hearing_localization",
                        "evidence": [
                            "听力测定显示右耳听阈下降，呈传导性模式",
                            "耳镜检查示外耳道通畅，鼓膜完整",
                        ],
                        "candidate_official_names": ["中耳炎"],
                    }
                ]
            },
            case_state=focal_ear_case(),
            official_diseases=["中耳炎"],
        )
        hallucinated = validate_axis_consult(
            {
                "diagnosis_axes": [
                    {
                        "axis_id": "unsupported_deep_ear_destruction",
                        "evidence": ["颞骨骨质破坏", "胆脂瘤侵蚀听骨链"],
                        "candidate_official_names": ["中耳炎"],
                    }
                ]
            },
            case_state=focal_ear_case(),
            official_diseases=["中耳炎"],
        )

        self.assertIn("conductive_hearing_localization", {item["axis_id"] for item in supported["diagnosis_axes"]})
        self.assertNotIn("unsupported_deep_ear_destruction", {item["axis_id"] for item in hallucinated["diagnosis_axes"]})

    def test_axis_validator_does_not_treat_negated_symptoms_as_support(self):
        case_state = {
            "chat_history": [
                {"from": "patient", "text": "没有耳痛，也没有听力下降。"},
            ],
            "ordered_examinations": [],
            "examination_results": {},
        }
        consult = validate_axis_consult(
            {
                "diagnosis_axes": [
                    {
                        "axis_id": "unsupported_ear_axis",
                        "evidence": ["耳痛", "听力下降"],
                        "candidate_official_names": ["中耳炎"],
                    }
                ]
            },
            case_state=case_state,
            official_diseases=["中耳炎"],
        )

        self.assertNotIn("unsupported_ear_axis", {item["axis_id"] for item in consult["diagnosis_axes"]})

    def test_llm_mislabeled_intake_facts_cannot_trigger_rule_axis(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "只是轻微耳闷。"}],
            "ordered_examinations": [],
            "examination_results": {},
        }
        consult = validate_axis_consult(
            {
                "intake_facts": {
                    "symptom_clusters": [
                        {"label": "反复高热", "evidence": "耳闷", "confidence": "high"},
                        {"label": "头痛肌肉关节痛", "evidence": "耳闷", "confidence": "high"},
                        {"label": "多部位黏膜皮肤出血", "evidence": "耳闷", "confidence": "high"},
                    ]
                },
                "diagnosis_axes": [],
            },
            case_state=case_state,
            official_diseases=["细菌感染", "白血病", "特发性血小板减少性紫癜"],
        )

        self.assertNotIn(
            "systemic_infection_vs_primary_hematologic",
            {item["axis_id"] for item in consult["diagnosis_axes"]},
        )

    def test_axis_validator_rejects_empty_evidence_axis(self):
        consult = validate_axis_consult(
            {
                "diagnosis_axes": [
                    {
                        "axis_id": "unsupported_empty_axis",
                        "evidence": [],
                        "candidate_official_names": ["中耳炎"],
                        "exam_intents": ["耳部深层结构定位"],
                        "treatment_risks": ["infection_before_steroid"],
                    }
                ]
            },
            case_state={
                "chat_history": [{"from": "patient", "text": "只是轻微耳闷。"}],
                "ordered_examinations": [],
                "examination_results": {},
            },
            official_diseases=["中耳炎"],
        )

        self.assertNotIn("unsupported_empty_axis", {item["axis_id"] for item in consult["diagnosis_axes"]})

    def test_early_negation_does_not_hide_later_supported_axis_evidence(self):
        case_state = focal_ear_case()
        case_state["chat_history"].insert(
            0,
            {"from": "patient", "text": "没有药物过敏。之后右耳剧烈针扎样疼痛。"},
        )
        consult = validate_axis_consult(
            {
                "diagnosis_axes": [
                    {
                        "axis_id": "supported_later_ear_axis",
                        "evidence": ["右耳剧烈针扎样疼痛", "呈传导性模式"],
                        "candidate_official_names": ["中耳炎"],
                    }
                ]
            },
            case_state=case_state,
            official_diseases=["中耳炎"],
        )

        self.assertIn("supported_later_ear_axis", {item["axis_id"] for item in consult["diagnosis_axes"]})

    def test_tympanic_umbo_does_not_create_abdominal_umbilical_fact(self):
        case_state = focal_ear_case()
        case_state["examination_results"]["耳镜检查"]["result"]["鼓膜"] = "完整，脐部结构清晰"

        facts = extract_intake_facts(case_state)

        self.assertNotIn("脐部", {item["label"] for item in facts["anatomic_sites"]})

    def test_acute_leukemia_alias_falls_back_to_official_broad_name(self):
        normalized = normalize_diagnosis(
            "急性白血病",
            official_diseases=["白血病"],
            alias_rules=load_knowledge_registry()["alias_map"],
        )

        self.assertEqual("白血病", normalized["normalized_diagnosis"])


def relapsing_fever_bleeding_case(with_exams=True):
    state = {
        "chat_history": [
            {
                "from": "patient",
                "text": "三周来反复高烧，中间好转后又发热，伴剧烈头痛、全身肌肉关节酸痛、鼻出血、牙龈出血和皮肤瘀斑。",
            }
        ],
        "ordered_examinations": [],
        "examination_results": {},
    }
    if with_exams:
        state["ordered_examinations"] = ["凝血功能全套", "全血细胞计数（CBC）"]
        state["examination_results"] = {
            "凝血功能全套": {"status": "normal", "result": {"凝血酶原时间": "正常"}},
            "全血细胞计数（CBC）": {
                "status": "abnormal",
                "result": {
                    "血白细胞计数": "18.2 x10^9/L [参考值：5.0-15.5]",
                    "血红蛋白": "98 g/L [参考值：110-155]",
                    "血小板计数": "62 x10^9/L [参考值：150-450]",
                },
            },
        }
    return state


def focal_ear_case():
    return {
        "chat_history": [
            {"from": "patient", "text": "单侧耳朵里面针扎样剧痛并肿胀，听力明显下降；另有疲劳头晕。"}
        ],
        "ordered_examinations": ["耳镜检查", "听力测定"],
        "examination_results": {
            "耳镜检查": {
                "status": "normal",
                "result": {"外耳道": "通畅，无阻塞", "耵聍": "极少或无", "感染征象": "无感染"},
            },
            "听力测定": {
                "status": "abnormal",
                "result": {"补充描述": "呈传导性模式（气骨导差）"},
            },
        },
    }


def cryoglobulinemia_case():
    return {
        "chat_history": [
            {"from": "patient", "text": "八个月反复腰背痛、乏力、紫癜，遇冷手指变白变青。"}
        ],
        "ordered_examinations": ["冷球蛋白检测", "尿液分析（UA）"],
        "examination_results": {
            "冷球蛋白检测": {
                "status": "abnormal",
                "result": {"定量": "620 ug/mL", "温度特性": "4C沉淀，加温后溶解"},
            },
            "尿液分析（UA）": {
                "status": "abnormal",
                "result": {"尿蛋白": "3+", "尿红细胞": "8/HPF", "管型": "可见颗粒管型"},
            },
        },
    }


def systemic_infection_safety_features(
    axes,
    *,
    include_serology=True,
    include_culture=True,
    include_activity=True,
    pathogen_positive=False,
    include_smear=True,
    smear_abnormal=False,
):
    results = {
        "全血细胞计数（CBC）": {
            "status": "abnormal",
            "result": {
                "血白细胞计数": "7.2 x10^9/L [参考值：4.0-10.0]",
                "血红蛋白": "132 g/L [参考值：110-155]",
                "血小板计数": "42 x10^9/L [参考值：150-450]",
            },
        },
    }
    if include_activity:
        results["C反应蛋白（CRP）"] = {
            "status": "normal",
            "result": {"CRP": "2 mg/L [参考值：0-8]"},
        }
    if include_culture:
        results["血培养"] = {
            "status": "abnormal" if pathogen_positive else "normal",
            "result": {"培养结果": "检出致病菌" if pathogen_positive else "无生长"},
        }
    if include_serology:
        results["血清学抗体检测"] = {
            "status": "normal",
            "result": {"病原抗体": "阴性"},
        }
    if include_smear:
        results["外周血涂片"] = {
            "status": "abnormal" if smear_abnormal else "normal",
            "result": {"形态": "可见原始细胞" if smear_abnormal else "未见原始细胞或异常细胞"},
        }
    return {
        "case_text": "反复高热伴出血，已完成感染与血液学评估。",
        "positive_findings": ["反复高热", "多部位出血"],
        "red_flags": [],
        "organ_risk": [],
        "diagnosis_axes": axes,
        "treatment_risks": [],
        "examination_results": results,
    }


def case_text(case_state):
    chunks = [item["text"] for item in case_state["chat_history"]]
    chunks.append(str(case_state.get("examination_results", {})))
    return " ".join(chunks)


if __name__ == "__main__":
    unittest.main()
