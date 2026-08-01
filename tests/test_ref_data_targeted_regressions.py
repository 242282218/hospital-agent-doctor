"""Targeted offline regressions derived from official catalog semantics."""

import unittest

from agent.legacy_orchestrator import (
    apply_evidence_backed_diagnosis_guard,
    build_name_map,
    extract_case_features,
    extract_intake_facts,
    final_verifier,
    flatten_examination_catalog,
    load_disease_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    select_diagnosis_axes,
    select_disease_candidates,
    select_exam_plan,
)
from agent.knowledge.typed_rule_engine import parse_compiled_rule_pack
from tests.typed_rule_test_data import (
    active_congenital_differential_pack_payload,
    active_diagnosis_priority_pack_payload,
)


class RefDataTargetedRegressionTest(unittest.TestCase):
    def setUp(self):
        self.disease_catalog = load_disease_catalog()
        self.examination_catalog = load_examination_catalog()
        knowledge = load_knowledge_registry()
        self.exam_item_map = build_name_map(flatten_examination_catalog(self.examination_catalog))
        self.exam_intent_rules = knowledge["exam_intent_map"]
        self.profiles = knowledge["diagnosis_exam_profiles"]

    def test_confirmed_vsd_outranks_generic_congenital_heart_disease(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "新生儿出生后吃奶累、发绀、出汗，呼吸越来越快。",
            }],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {
                        "先天性心脏缺陷": "大型室间隔缺损",
                        "补充描述": "伴右向左分流和肺动脉高压",
                    },
                }
            },
        }
        candidates = select_disease_candidates(case_state, self.disease_catalog, limit=24)
        names = [item["disease"] for item in candidates]
        self.assertLess(names.index("室间隔缺损（VSD）"), names.index("先天性心脏病"))
        self.assertEqual(
            "室间隔缺损（VSD）",
            apply_evidence_backed_diagnosis_guard("先天性心脏病", case_state, candidates),
        )

    def test_cyanotic_vsd_does_not_jump_to_direct_radical_repair(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "新生儿出生后吃奶累、发绀、出汗，呼吸越来越快。",
            }],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {
                        "先天性心脏缺陷": "大型室间隔缺损",
                        "补充描述": "伴右向左分流和肺动脉高压",
                    },
                }
            },
        }
        features = extract_case_features(case_state, [{"disease": "室间隔缺损（VSD）"}])
        result = final_verifier(
            diagnosis="室间隔缺损（VSD）",
            examinations=["超声心动图"],
            treatment_plan="尽快进行VSD修补术根治疾病，并给予吸氧。",
            official_diseases=["室间隔缺损（VSD）"],
            examination_catalog={"影像学检查 - 超声": ["超声心动图"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )
        codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("cyanotic_vsd_direct_repair_without_staging", codes)
        self.assertNotIn("VSD修补术根治", result["patched_treatment"])
        self.assertTrue(
            "姑息" in result["patched_treatment"] or "分期" in result["patched_treatment"]
        )

    def test_water_exposure_and_calf_pain_recalls_leptospirosis(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "洪水泥水接触后突发高热，腓肠肌剧痛、结膜充血、尿色变深并少尿。",
            }],
            "examination_results": {},
        }
        catalog = {
            "感染科": ["钩端螺旋体病", "细菌感染"],
            "骨科": ["关节炎"],
        }
        candidates = select_disease_candidates(case_state, catalog, limit=3)
        self.assertEqual("钩端螺旋体病", candidates[0]["disease"])
        self.assertEqual(
            "钩端螺旋体病",
            apply_evidence_backed_diagnosis_guard("关节炎", case_state, candidates),
        )
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=candidates,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.exam_item_map,
            diagnosis_exam_profiles=self.profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=5,
        )
        self.assertIn("双份血清抗体检测", plan["examinations"])
        self.assertIn("肝肾功能检查（LFTs/RFTs）", plan["examinations"])

    def test_diuretic_metabolic_alkalosis_recalls_hypokalemia(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "自行服用利尿剂后多饮多尿、乏力、腿部抽筋。",
            }],
            "examination_results": {
                "血清电解质": {
                    "status": "abnormal",
                    "result": {
                        "血钾": "2.6 mmol/L",
                        "血钠": "149 mmol/L",
                        "碳酸氢盐": "31 mmol/L，代谢性碱中毒",
                    },
                },
                "肾功能检查（RFTs）": {
                    "status": "normal",
                    "result": {"肌酐": "正常"},
                },
            },
        }
        catalog = {
            "内分泌科": ["垂体前叶功能减退", "低钾血症"],
        }
        candidates = select_disease_candidates(case_state, catalog, limit=2)
        self.assertEqual("低钾血症", candidates[0]["disease"])
        self.assertEqual(
            "低钾血症",
            apply_evidence_backed_diagnosis_guard("垂体前叶功能减退", case_state, candidates),
        )
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=candidates,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.exam_item_map,
            diagnosis_exam_profiles=self.profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=5,
        )
        self.assertIn("心电图（ECG）", plan["examinations"])

    def test_multisystem_autoimmune_serositis_recalls_sle_axis(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "脱发、眉毛睫毛脱落，面部红斑日晒后加重，关节疼痛、眼睛干，最近胸痛。",
            }],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"心包积液": "少量环周性心包积液伴心包增厚"},
                }
            },
        }
        facts = extract_intake_facts(case_state)
        axes = select_diagnosis_axes(facts)
        self.assertIn("sle_organ_thrombosis_reproductive_risk", [axis["axis_id"] for axis in axes])
        candidates = select_disease_candidates(case_state, self.disease_catalog, limit=24)
        names = [item["disease"] for item in candidates]
        self.assertLess(names.index("系统性红斑狼疮"), names.index("慢性缩窄性心包炎"))
        self.assertEqual(
            "系统性红斑狼疮",
            apply_evidence_backed_diagnosis_guard("慢性缩窄性心包炎", case_state, candidates),
        )
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=candidates,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.exam_item_map,
            diagnosis_exam_profiles=self.profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=5,
        )
        self.assertIn("抗核抗体（ANA）谱", plan["examinations"])
        self.assertIn("尿液分析（UA）", plan["examinations"])

    def test_chest_trauma_uses_chest_imaging_not_limb_xray(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "被撞击跌倒后左前胸刺痛、压痛、肿胀和瘀斑，深呼吸、咳嗽或抬臂时加重。",
            }],
            "ordered_examinations": [],
            "examination_results": {},
        }
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[],
            diagnosis_axes=select_diagnosis_axes(extract_intake_facts(case_state)),
            examination_catalog=self.examination_catalog,
            item_name_map=self.exam_item_map,
            diagnosis_exam_profiles=self.profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=3,
        )
        self.assertIn("胸部X线检查（CXR）", plan["examinations"])
        self.assertNotIn("四肢X线检查", plan["examinations"])

    def test_chest_trauma_treatment_preserves_deep_breathing(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "撞击后前胸疼痛，深呼吸和咳嗽时加重，胸壁有瘀斑。",
            }],
            "ordered_examinations": ["胸部X线检查（CXR）"],
            "examination_results": {},
        }
        features = extract_case_features(case_state, [{"disease": "肌肉拉伤"}])
        result = final_verifier(
            diagnosis="肌肉拉伤",
            examinations=["胸部X线检查（CXR）"],
            treatment_plan="休息并减少深呼吸，疼痛缓解后复诊。",
            official_diseases=["肌肉拉伤"],
            examination_catalog={"影像学检查 - X线": ["胸部X线检查（CXR）"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )
        self.assertIn("深呼吸练习", result["patched_treatment"])
        self.assertNotIn("减少深呼吸", result["patched_treatment"])

    def test_congenital_viral_exposure_recalls_rubella_and_cmv_axes(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "宝宝出生即吃奶差、呼吸费力，生后第二天出现黄疸和红疹，最近嗜睡、少尿；孕早期有宫内病毒暴露。",
            }],
            "ordered_examinations": [],
            "examination_results": {},
        }

        facts = extract_intake_facts(case_state)
        axes = select_diagnosis_axes(facts)
        candidates = select_disease_candidates(case_state, self.disease_catalog, limit=24)
        names = [item["disease"] for item in candidates]
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=candidates,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.exam_item_map,
            diagnosis_exam_profiles=self.profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=6,
        )

        self.assertIn("congenital_infection_differential", [axis["axis_id"] for axis in axes])
        self.assertIn("先天性风疹综合征", names)
        self.assertIn("巨细胞病毒感染", names)
        self.assertIn("巨细胞病毒（CMV）抗体检测", plan["examinations"])
        self.assertIn("风疹抗体检测", plan["examinations"])

    def test_actual_congenital_wording_survives_cross_sentence_negation(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "宝宝出生那天起就吃奶不好，呼吸快，喂奶时出汗、嘴唇发青。"
                    "从第2天开始皮肤和眼睛发黄，身上有红疹。对声音反应好像变弱了。",
                },
                {
                    "from": "patient",
                    "text": "宝宝刚出生，0岁。没有药物过敏，也没吃过什么药。"
                    "之前检查发现孕早期有宫内病毒暴露，其他基础病不清楚。",
                },
            ],
            "examination_results": {},
        }

        axes = select_diagnosis_axes(extract_intake_facts(case_state))

        self.assertIn(
            "congenital_infection_differential",
            [axis["axis_id"] for axis in axes],
        )

    def test_typed_congenital_adapter_receives_trusted_facts_and_existing_axes(self):
        from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "宝宝出生后吃奶差，第2天出现黄疸和红疹。"
                    "没有药物过敏。孕早期有宫内病毒暴露。",
                }
            ],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"先天性心脏缺陷": "动脉导管未闭，存在左向右分流"},
                }
            },
        }
        case_state["diagnosis_axes"] = select_diagnosis_axes(
            extract_intake_facts(case_state)
        )
        candidates = [
            {"disease": "先天性风疹综合征", "score": 65, "source": "diagnosis_axis"},
            {"disease": "巨细胞病毒感染", "score": 65, "source": "diagnosis_axis"},
        ]
        pack = parse_compiled_rule_pack(
            active_congenital_differential_pack_payload()
        )

        _ordered, result = apply_diagnosis_candidate_rules(
            candidates,
            case_state=case_state,
            official_diseases=["先天性风疹综合征", "巨细胞病毒感染"],
            rule_pack=pack,
        )

        self.assertEqual(
            ("congenital_rubella", "congenital_cmv", "congenital_infection_differential"),
            result.output_context.diagnostic_axis_ids,
        )
        self.assertTrue(
            {
                "neonate",
                "intrauterine_viral_exposure",
                "congenital_jaundice",
                "congenital_rash",
                "patent_ductus_arteriosus",
            }.issubset(result.output_context.fact_codes)
        )
        self.assertEqual(
            "congenital_infection_axes_expanded",
            result.decisions[0].reason_code,
        )

    def test_hearing_symptoms_outscore_hypertension_history(self):
        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "我80岁，持续耳鸣伴双侧高频听力下降，和高血压病史无关的耳部症状越来越明显。",
            }],
            "examination_results": {
                "听力测定": {
                    "status": "abnormal",
                    "result": {"高频听阈": "4kHz 75dB，8kHz 85dB"},
                }
            },
        }

        candidates = select_disease_candidates(case_state, self.disease_catalog, limit=24)
        names = [item["disease"] for item in candidates]

        self.assertLess(names.index("耳鸣"), names.index("原发性高血压"))
        self.assertEqual(
            "耳鸣",
            apply_evidence_backed_diagnosis_guard("原发性高血压", case_state, candidates),
        )

    def test_colloquial_tinnitus_with_abnormal_audiometry_enters_candidates(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "耳朵里一直有高音调的蝉鸣声和电流嗡嗡声，"
                    "现在几乎天天都有；我有高血压和听力下降。",
                }
            ],
            "examination_results": {
                "听力测定": {
                    "status": "abnormal",
                    "result": {"高频听阈": "双侧4kHz 75dB，8kHz 85dB"},
                }
            },
        }

        candidates = select_disease_candidates(
            case_state,
            self.disease_catalog,
            limit=24,
        )

        self.assertIn("耳鸣", [item["disease"] for item in candidates])

    def test_unrelated_buzzing_does_not_become_tinnitus(self):
        from agent.legacy_orchestrator import has_hearing_symptom_pattern

        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "这两天耳朵有点疼。家里的电器一直发出嗡嗡声。",
            }],
            "examination_results": {
                "听力测定": {
                    "status": "abnormal",
                    "result": {"高频听阈": "双侧4kHz 75dB"},
                }
            },
        }

        self.assertFalse(has_hearing_symptom_pattern(case_state))

    def test_hearing_report_cannot_supply_the_subjective_tinnitus_symptom(self):
        from agent.legacy_orchestrator import has_hearing_symptom_pattern

        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "近三个月只有双侧高频听力下降。",
            }],
            "examination_results": {
                "听力测定": {
                    "status": "abnormal",
                    "result": {"高频听阈": "75dB", "备注": "耳鸣情况需另行询问"},
                }
            },
        }

        self.assertFalse(has_hearing_symptom_pattern(case_state))

    def test_typed_congenital_axes_materialize_in_engine_order(self):
        from agent.legacy_orchestrator import (
            apply_diagnosis_candidate_rules,
            materialize_diagnosis_rule_axes,
        )

        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "新生儿出生后出现黄疸和红疹，孕早期有宫内病毒暴露。",
                }
            ],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"先天性心脏缺陷": "动脉导管未闭"},
                }
            },
            "diagnosis_axes": [
                {
                    "axis_id": "congenital_infection_differential",
                    "source": "rule",
                    "evidence": ["新生儿或婴儿", "宫内病毒暴露"],
                    "candidate_official_names": ["先天性风疹综合征", "巨细胞病毒感染"],
                    "rule_candidate_official_names": ["先天性风疹综合征", "巨细胞病毒感染"],
                },
                {
                    "axis_id": "other_axis",
                    "source": "rule",
                    "evidence": ["other evidence", "other objective evidence"],
                    "candidate_official_names": [],
                    "rule_candidate_official_names": [],
                },
            ],
        }
        pack = parse_compiled_rule_pack(
            active_congenital_differential_pack_payload()
        )
        _ordered, result = apply_diagnosis_candidate_rules(
            [
                {"disease": "先天性风疹综合征"},
                {"disease": "巨细胞病毒感染"},
            ],
            case_state=case_state,
            official_diseases=["先天性风疹综合征", "巨细胞病毒感染"],
            rule_pack=pack,
        )

        axes = materialize_diagnosis_rule_axes(
            case_state["diagnosis_axes"],
            result,
        )

        self.assertEqual(
            ["congenital_rubella", "congenital_cmv", "other_axis"],
            [axis["axis_id"] for axis in axes],
        )
        self.assertEqual(
            ["先天性风疹综合征"],
            axes[0]["candidate_official_names"],
        )
        self.assertEqual(
            ["巨细胞病毒感染"],
            axes[1]["candidate_official_names"],
        )

    def test_diagnosis_rule_facts_do_not_promote_patient_claims_to_objective_evidence(self):
        from agent.legacy_orchestrator import diagnosis_rule_fact_codes

        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": (
                    "宝宝出生后黄疸、红疹，家属说血小板减少、脑室周围钙化，"
                    "动脉导管未闭，CMV唾液PCR阳性。"
                ),
            }],
            "examination_results": {},
        }

        facts = set(diagnosis_rule_fact_codes(case_state))

        self.assertNotIn("thrombocytopenia", facts)
        self.assertNotIn("congenital_neuroimaging_abnormality", facts)
        self.assertNotIn("patent_ductus_arteriosus", facts)
        self.assertNotIn("cmv_saliva_or_urine_pcr_positive_within_21_days", facts)

    def test_diagnosis_rule_facts_require_usable_abnormal_exam_payload(self):
        from agent.legacy_orchestrator import diagnosis_rule_fact_codes

        base = {
            "chat_history": [{
                "from": "patient",
                "text": "宝宝出生后黄疸、红疹，孕早期有宫内病毒暴露。",
            }],
            "examination_results": {},
        }
        pending = {
            **base,
            "examination_results": {
                "超声心动图": {
                    "status": "pending",
                    "result": {"结构": "动脉导管未闭，待报告"},
                }
            },
        }
        abnormal = {
            **base,
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结构": "动脉导管未闭"},
                },
                "巨细胞病毒（CMV）核酸检测": {
                    "status": "abnormal",
                    "result": {"标本": "出生后14天尿液", "PCR": "阳性"},
                },
            },
        }

        self.assertNotIn(
            "patent_ductus_arteriosus",
            set(diagnosis_rule_fact_codes(pending)),
        )
        abnormal_facts = set(diagnosis_rule_fact_codes(abnormal))
        self.assertIn("patent_ductus_arteriosus", abnormal_facts)
        self.assertIn(
            "cmv_saliva_or_urine_pcr_positive_within_21_days",
            abnormal_facts,
        )

    def test_diagnosis_rule_facts_do_not_treat_high_platelets_as_thrombocytopenia(self):
        from agent.legacy_orchestrator import diagnosis_rule_fact_codes

        facts = diagnosis_rule_fact_codes({
            "chat_history": [{"from": "patient", "text": "宝宝出生后出现黄疸。"}],
            "examination_results": {
                "全血细胞计数（CBC）": {
                    "status": "abnormal",
                    "result": {"血小板计数": "560 x 10^9/L，明显升高，血小板异常增多"},
                }
            },
        })

        self.assertNotIn("thrombocytopenia", facts)

    def test_pathogen_positive_control_does_not_confirm_cmv_or_rubella(self):
        from agent.legacy_orchestrator import diagnosis_rule_fact_codes

        facts = diagnosis_rule_fact_codes({
            "chat_history": [{"from": "patient", "text": "新生儿出生后黄疸、红疹。"}],
            "examination_results": {
                "CMV PCR与风疹IgM联合检测": {
                    "status": "abnormal",
                    "result": {
                        "阳性对照": "阳性",
                        "CMV PCR": "未检出",
                        "风疹IgM": "阴性",
                    },
                }
            },
        })

        self.assertNotIn("cmv_pcr_positive", facts)
        self.assertNotIn("rubella_igm_positive_in_infant", facts)

    def test_cmv_followup_window_is_not_a_birth_sample_age(self):
        from agent.legacy_orchestrator import diagnosis_rule_fact_codes

        facts = diagnosis_rule_fact_codes({
            "chat_history": [{"from": "patient", "text": "新生儿出生后黄疸、红疹。"}],
            "examination_results": {
                "巨细胞病毒（CMV）核酸检测": {
                    "status": "abnormal",
                    "result": {
                        "标本": "尿液，建议14天内复查",
                        "CMV PCR": "阳性",
                    },
                }
            },
        })

        self.assertIn("cmv_pcr_positive", facts)
        self.assertNotIn("cmv_saliva_or_urine_pcr_positive_within_21_days", facts)

    def test_typed_priority_adapter_promotes_objective_current_problem(self):
        from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "持续耳鸣伴双侧高频听力下降；既往有高血压，但与此次耳部症状无关。",
            }],
            "examination_results": {
                "听力测定": {
                    "status": "abnormal",
                    "result": {"高频听阈": "4kHz 75dB，8kHz 85dB"},
                }
            },
        }
        candidates = [
            {
                "department": "心内科",
                "disease": "原发性高血压",
                "score": 120,
                "source": "catalog_match",
                "role": "background_condition",
                "complaint_relation": "unrelated",
            },
            {
                "department": "耳鼻咽喉科",
                "disease": "耳鸣",
                "score": 90,
                "source": "required_differential",
                "role": "current_problem",
                "complaint_relation": "unrelated",
                "support_level": "objective",
                "evidence_codes": ["audiometry_abnormal"],
            },
        ]
        pack = parse_compiled_rule_pack(active_diagnosis_priority_pack_payload())

        ordered, result = apply_diagnosis_candidate_rules(
            candidates,
            case_state=case_state,
            official_diseases=[
                name for names in self.disease_catalog.values() for name in names
            ],
            rule_pack=pack,
        )

        self.assertEqual(["耳鸣", "原发性高血压"], [item["disease"] for item in ordered])
        self.assertEqual("耳鸣", result.output_context.preferred_diagnosis)
        self.assertEqual([pack.rules[0].rule_id], list(result.applied_rule_ids))

    def test_typed_priority_adapter_derives_hearing_evidence_and_background_role(self):
        from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "持续耳鸣伴高频听力下降；高血压只是既往病史，与此次耳部症状无关。",
            }],
            "examination_results": {
                "听力测定": {
                    "status": "abnormal",
                    "result": {"高频听阈": "4kHz 75dB，8kHz 85dB"},
                }
            },
        }
        candidates = [
            {"department": "心内科", "disease": "原发性高血压", "score": 120},
            {"department": "耳鼻咽喉科", "disease": "耳鸣", "score": 90},
        ]
        pack = parse_compiled_rule_pack(active_diagnosis_priority_pack_payload())

        ordered, result = apply_diagnosis_candidate_rules(
            candidates,
            case_state=case_state,
            official_diseases=[
                name for names in self.disease_catalog.values() for name in names
            ],
            rule_pack=pack,
        )

        self.assertEqual(["耳鸣", "原发性高血压"], [item["disease"] for item in ordered])
        self.assertEqual("耳鸣", result.output_context.preferred_diagnosis)

    def test_typed_priority_adapter_normalizes_legacy_candidate_roles(self):
        from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

        role_cases = [
            ("缺铁性贫血", "secondary", "differential"),
            ("花粉症", "background_history", "background_condition"),
            ("神经囊虫病", "etiology", "differential"),
            ("低钾血症", "consequence", "differential"),
            ("视网膜母细胞瘤", "must_exclude_etiology", "differential"),
            ("斜视", "symptom_or_secondary", "differential"),
            ("偏头痛", "unsafe_symptom_closure", "differential"),
            ("耳鸣", "current_problem", "current_problem"),
            ("巨细胞病毒感染", "future_role", "differential"),
        ]
        candidates = [
            {
                "disease": disease,
                "score": 100 - index,
                "role": legacy_role,
                "matched_evidence": ["LLM-generated narrative evidence"],
            }
            for index, (disease, legacy_role, _) in enumerate(role_cases)
        ]
        pack = parse_compiled_rule_pack(active_diagnosis_priority_pack_payload())

        ordered, result = apply_diagnosis_candidate_rules(
            candidates,
            case_state={"chat_history": [], "examination_results": {}},
            official_diseases=[disease for disease, _, _ in role_cases],
            rule_pack=pack,
        )

        self.assertEqual(
            [disease for disease, _, _ in role_cases],
            [item["disease"] for item in ordered],
        )
        self.assertEqual(
            [typed_role for _, _, typed_role in role_cases],
            [candidate.role for candidate in result.output_context.diagnosis_candidates],
        )
        self.assertTrue(
            all(
                candidate.support_level == "none" and candidate.evidence_codes == ()
                for candidate in result.output_context.diagnosis_candidates
            )
        )

    def test_typed_priority_adapter_normalizes_malformed_candidate_enums(self):
        from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

        malformed_cases = [
            ("support_level", "llm_confident", "none"),
            ("complaint_relation", "possibly_related", "unknown"),
            ("urgency", "critical", "routine"),
        ]
        pack = parse_compiled_rule_pack(active_diagnosis_priority_pack_payload())

        for field, malformed_value, fallback in malformed_cases:
            with self.subTest(field=field):
                candidate = {
                    "disease": "偏头痛",
                    "score": 90,
                    "role": "current_problem",
                    "support_level": "objective",
                    "complaint_relation": "explains",
                    "urgency": "emergency",
                }
                candidate[field] = malformed_value

                ordered, result = apply_diagnosis_candidate_rules(
                    [candidate],
                    case_state={"chat_history": [], "examination_results": {}},
                    official_diseases=["偏头痛"],
                    rule_pack=pack,
                )

                self.assertEqual(["偏头痛"], [item["disease"] for item in ordered])
                typed = result.output_context.diagnosis_candidates[0]
                self.assertEqual(fallback, getattr(typed, field))
                for valid_field in {"support_level", "complaint_relation", "urgency"} - {field}:
                    self.assertEqual(candidate[valid_field], getattr(typed, valid_field))

    def test_typed_priority_adapter_requires_abnormal_hearing_payload(self):
        from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

        payload_cases = {
            "not_examined": {},
            "pending": {
                "听力测定": {
                    "status": "pending",
                    "result": {"高频听阈": "待报告"},
                }
            },
            "unknown": {
                "听力测定": {
                    "status": "unknown",
                    "result": {"高频听阈": "4kHz 75dB"},
                    "abnormal_indicators": ["高频听阈升高"],
                }
            },
            "normal": {
                "听力测定": {
                    "status": "normal",
                    "result": {"高频听阈": "双耳正常"},
                    "abnormal_indicators": ["legacy false positive"],
                }
            },
        }
        candidates = [
            {
                "disease": "原发性高血压",
                "score": 120,
                "role": "background_condition",
                "complaint_relation": "unrelated",
            },
            {"disease": "耳鸣", "score": 90},
        ]
        pack = parse_compiled_rule_pack(active_diagnosis_priority_pack_payload())

        for label, examination_results in payload_cases.items():
            with self.subTest(label=label):
                ordered, result = apply_diagnosis_candidate_rules(
                    candidates,
                    case_state={
                        "chat_history": [{
                            "from": "patient",
                            "text": "持续耳鸣，患者只说已经做过听力测定。",
                        }],
                        "examination_results": examination_results,
                    },
                    official_diseases=["原发性高血压", "耳鸣"],
                    rule_pack=pack,
                )

                self.assertEqual(
                    ["原发性高血压", "耳鸣"],
                    [item["disease"] for item in ordered],
                )
                self.assertIsNone(result.output_context.preferred_diagnosis)
                tinnitus = result.output_context.diagnosis_candidates[1]
                self.assertEqual("differential", tinnitus.role)
                self.assertEqual("none", tinnitus.support_level)

    def test_typed_priority_adapter_accepts_structured_abnormal_hearing_indicator(self):
        from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "持续耳鸣；高血压只是既往病史，与此次症状无关。",
            }],
            "examination_results": {
                "听力测定": {
                    "result": {"右耳": "听阈异常"},
                    "abnormal_indicators": ["右耳听阈异常"],
                }
            },
        }
        candidates = [
            {"disease": "原发性高血压", "score": 120},
            {"disease": "耳鸣", "score": 90},
        ]
        pack = parse_compiled_rule_pack(active_diagnosis_priority_pack_payload())

        ordered, result = apply_diagnosis_candidate_rules(
            candidates,
            case_state=case_state,
            official_diseases=["原发性高血压", "耳鸣"],
            rule_pack=pack,
        )

        self.assertEqual(["耳鸣", "原发性高血压"], [item["disease"] for item in ordered])
        self.assertEqual("耳鸣", result.output_context.preferred_diagnosis)

    def test_typed_priority_adapter_does_not_infer_tinnitus_from_hearing_loss(self):
        from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "近三个月双侧高频听力下降；高血压只是既往病史，与此次听力问题无关。",
            }],
            "examination_results": {
                "听力测定": {
                    "status": "abnormal",
                    "result": {"高频听阈": "4kHz 75dB，8kHz 85dB"},
                }
            },
        }
        candidates = [
            {"disease": "原发性高血压", "score": 120},
            {"disease": "耳鸣", "score": 90},
        ]
        pack = parse_compiled_rule_pack(active_diagnosis_priority_pack_payload())

        ordered, result = apply_diagnosis_candidate_rules(
            candidates,
            case_state=case_state,
            official_diseases=["原发性高血压", "耳鸣"],
            rule_pack=pack,
        )

        self.assertEqual(["原发性高血压", "耳鸣"], [item["disease"] for item in ordered])
        self.assertIsNone(result.output_context.preferred_diagnosis)
        tinnitus = result.output_context.diagnosis_candidates[1]
        self.assertEqual("differential", tinnitus.role)
        self.assertEqual("none", tinnitus.support_level)

    def test_typed_priority_result_overrides_llm_background_selection(self):
        from agent.legacy_orchestrator import (
            apply_diagnosis_candidate_rules,
            select_rule_preferred_diagnosis,
        )

        case_state = {
            "chat_history": [{
                "from": "patient",
                "text": "持续耳鸣伴高频听力下降；高血压只是既往病史，与此次耳部症状无关。",
            }],
            "examination_results": {
                "听力测定": {
                    "status": "abnormal",
                    "result": {"高频听阈": "4kHz 75dB"},
                }
            },
        }
        candidates = [
            {"department": "心内科", "disease": "原发性高血压", "score": 120},
            {"department": "耳鼻咽喉科", "disease": "耳鸣", "score": 90},
        ]
        pack = parse_compiled_rule_pack(active_diagnosis_priority_pack_payload())
        ordered, result = apply_diagnosis_candidate_rules(
            candidates,
            case_state=case_state,
            official_diseases=[
                name for names in self.disease_catalog.values() for name in names
            ],
            rule_pack=pack,
        )

        diagnosis = select_rule_preferred_diagnosis(
            "原发性高血压",
            candidates=ordered,
            rule_result=result,
        )

        self.assertEqual("耳鸣", diagnosis)


if __name__ == "__main__":
    unittest.main()
