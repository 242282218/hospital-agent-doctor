import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.legacy_orchestrator import (
    build_name_map,
    extract_case_features,
    flatten_examination_catalog,
    final_verifier,
    extract_intake_facts,
    load_disease_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    normalize_candidates_from_diagnostic_context,
    normalize_diagnosis,
    select_disease_candidates,
    select_diagnosis_axes,
    select_exam_plan,
    validate_axis_consult,
)
from agent.memory import MarkdownMemory


class OfflineRegressionTopicsTest(unittest.TestCase):
    def test_offline_topic_eczema_herpeticum_exam_profile_positive(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孩子高热，脸和颈部出现成簇水泡和脓疱，中央有脐凹，原本有特应性皮炎，皮损疼痛结痂。",
                }
            ],
            "ordered_examinations": [],
        }
        disease_candidates = [{"department": "皮肤科", "disease": "卡波西水痘样疹", "score": 30}]
        examination_catalog = load_examination_catalog()

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
        )

        self.assertEqual(
            {
                "体格检查",
                "全血细胞计数（CBC）",
                "病毒核酸检测（Viral NAT）",
                "细胞学检查",
                "细菌培养及鉴定",
            },
            set(plan["examinations"]),
        )

    def test_offline_topic_eczema_herpeticum_exam_profile_negative_without_skin_evidence(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": "孩子发热两天，咽痛，没有皮疹或水泡。"}],
            "ordered_examinations": [],
        }
        disease_candidates = [{"department": "皮肤科", "disease": "卡波西水痘样疹", "score": 30}]
        examination_catalog = load_examination_catalog()

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
        )

        self.assertEqual([], plan["examinations"])

    def test_offline_topic_sle_memory_candidate_not_loaded(self):
        with TemporaryDirectory() as temp_dir:
            memory = MarkdownMemory(Path(temp_dir) / "memory.md", max_notes=3)
            memory.append_case_reflection(
                patient_id="Patient_02654",
                evaluation_reflection={"reflection": {"future_strategy": "卡波西检查经验仅候选"}},
            )

            default_notes = memory.load_notes()
            candidate_notes = memory.load_notes(include_candidates=True)

        self.assertEqual([], default_notes)
        self.assertIn("卡波西检查经验仅候选", "\n".join(candidate_notes))

    def test_offline_topic_sle_exam_profile_recall(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "日晒后脸上红斑皮疹，手腕和手指关节肿痛晨僵，反复口腔溃疡、脱发、低热。",
                }
            ],
            "examination_results": {
                "抗核抗体（ANA）谱": {"result": {"ANA": "1:1280 阳性", "抗Sm抗体": "阳性"}},
                "补体成分分析": {"result": {"C3": "降低", "C4": "降低"}},
                "尿液分析（UA）": {"result": {"尿蛋白": "++"}},
            },
            "ordered_examinations": [],
        }
        disease_candidates = select_disease_candidates(case_state, load_disease_catalog(), limit=8)
        examination_catalog = load_examination_catalog()

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
            max_items=8,
        )

        self.assertIn("系统性红斑狼疮", [item["disease"] for item in disease_candidates])
        self.assertTrue(
            {
                "全血细胞计数（CBC）",
                "尿液分析（UA）",
                "补体成分分析",
                "抗核抗体（ANA）谱",
                "抗磷脂抗体（APA）组合检测",
            }.issubset(set(plan["examinations"]))
        )

    def test_offline_topic_prostatitis_candidate_recall(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "男性，两天前发热寒战，尿频尿急尿痛，排尿困难，差点尿不出来，会阴部胀痛。",
                }
            ],
            "examination_results": {
                "尿液分析（UA）": {"result": {"白细胞酯酶": "阳性", "亚硝酸盐": "阳性"}},
                "尿培养": {"result": {"致病微生物": "大肠埃希菌 150000 CFU/mL"}},
            },
            "ordered_examinations": [],
        }
        disease_candidates = select_disease_candidates(case_state, load_disease_catalog(), limit=8)
        examination_catalog = load_examination_catalog()

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
            max_items=8,
        )

        self.assertEqual("急性细菌性前列腺炎", disease_candidates[0]["disease"])
        self.assertTrue(
            {
                "直肠指检（DRE）",
                "全血细胞计数（CBC）",
                "尿液分析（UA）",
                "尿培养",
                "抗菌药物敏感性试验（AST）",
                "前列腺超声",
            }.issubset(set(plan["examinations"]))
        )

    def test_offline_topic_generic_infection_not_refined_to_unrelated_candidate(self):
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

    def test_offline_topic_thiamine_deficiency_axis_recall_and_exams(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "我还在哺乳期，最近脚底像火烧一样发麻，晚上更严重，爬楼气短，心慌胸闷，脚踝水肿，睡觉要垫高枕头。",
                }
            ],
            "examination_results": {
                "尿液分析（UA）": {"result": {"尿蛋白": "阴性", "隐血": "阴性"}},
                "24小时尿蛋白检测": {"result": {"尿蛋白": "<150 mg/24小时"}},
            },
            "ordered_examinations": [],
        }

        disease_candidates = select_disease_candidates(case_state, load_disease_catalog(), limit=8)
        examination_catalog = load_examination_catalog()
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
            max_items=8,
        )

        self.assertIn("脚气病", [item["disease"] for item in disease_candidates])
        self.assertTrue(
            {"24小时尿硫胺素检测", "硫胺素负荷试验"}.intersection(set(plan["examinations"]))
        )

    def test_offline_topic_osteopetrosis_alias_and_candidate_ranking(self):
        official_disease_map = build_name_map(
            disease for diseases in load_disease_catalog().values() for disease in diseases
        )
        normalized = normalize_diagnosis(
            "骨硬化症",
            official_diseases=official_disease_map.values(),
            alias_rules=load_knowledge_registry()["alias_map"],
            disease_candidates=[],
        )
        self.assertEqual("大理石骨病", normalized["normalized_diagnosis"])

        diagnostic_context = {
            "case_features": {
                "demographics": [{"label": "儿童患者", "evidence": "孩子", "confidence": "high"}],
                "symptom_clusters": [
                    {"label": "反复低能量骨折", "evidence": "轻微碰撞后反复骨折", "confidence": "high"},
                    {"label": "生长发育迟缓", "evidence": "身高增长慢", "confidence": "high"},
                ],
                "exam_evidence": [
                    {"label": "家族骨硬化症史", "evidence": "哥哥诊断骨硬化症", "confidence": "high"}
                ],
                "organ_risk": [
                    {"label": "骨髓受累风险", "evidence": "面色苍白、腹部鼓胀", "confidence": "medium"}
                ],
            },
            "differential": [
                {"raw_name": "骨硬化症", "rank": 1, "reason": "家族史和反复低能量骨折"},
                {"raw_name": "白血病", "rank": 3, "reason": "贫血貌和骨痛需排除"},
            ],
            "normalization_suggestions": [
                {
                    "raw_name": "骨硬化症",
                    "suggested_official_name": "大理石骨病",
                    "confidence": "high",
                    "supporting_feature_labels": ["儿童患者", "反复低能量骨折", "家族骨硬化症史"],
                },
                {
                    "raw_name": "白血病",
                    "suggested_official_name": "急性淋巴细胞白血病",
                    "confidence": "medium",
                    "supporting_feature_labels": ["儿童患者", "骨髓受累风险"],
                },
            ],
        }

        candidates = normalize_candidates_from_diagnostic_context(
            diagnostic_context,
            literal_candidates=[],
            disease_catalog=load_disease_catalog(),
            official_disease_map=official_disease_map,
            alias_rules=load_knowledge_registry()["alias_map"],
            limit=6,
        )
        candidate_names = [item["disease"] for item in candidates]

        self.assertEqual("大理石骨病", candidate_names[0])
        self.assertLess(candidate_names.index("大理石骨病"), candidate_names.index("急性淋巴细胞白血病"))

    def test_offline_topic_osteopetrosis_alias_from_non_official_suggestion(self):
        official_disease_map = build_name_map(
            disease for diseases in load_disease_catalog().values() for disease in diseases
        )
        diagnostic_context = {
            "case_features": {
                "symptom_clusters": [
                    {"label": "反复骨折史", "evidence": "轻微碰撞后反复骨折", "confidence": "high"},
                    {"label": "生长发育迟缓", "evidence": "身高增长慢", "confidence": "high"},
                ],
                "exam_evidence": [
                    {"label": "家族遗传史阳性", "evidence": "哥哥诊断骨硬化症，父母近亲", "confidence": "high"}
                ],
                "organ_risk": [
                    {"label": "骨髓功能异常风险", "evidence": "面色苍白、腹部鼓胀", "confidence": "medium"}
                ],
            },
            "differential": [
                {"raw_name": "骨硬化症", "rank": 1, "reason": "家族史和反复骨折"},
                {"raw_name": "白血病", "rank": 3, "reason": "贫血貌和骨痛需排除"},
            ],
            "normalization_suggestions": [
                {
                    "raw_name": "骨硬化症",
                    "suggested_official_name": "骨硬化症",
                    "confidence": "medium",
                    "supporting_feature_labels": ["家族遗传史阳性", "反复骨折史", "生长发育迟缓"],
                },
                {
                    "raw_name": "白血病",
                    "suggested_official_name": "急性淋巴细胞白血病",
                    "confidence": "medium",
                    "supporting_feature_labels": ["骨髓功能异常风险", "反复骨折史"],
                },
            ],
        }

        candidates = normalize_candidates_from_diagnostic_context(
            diagnostic_context,
            literal_candidates=[],
            disease_catalog=load_disease_catalog(),
            official_disease_map=official_disease_map,
            alias_rules=load_knowledge_registry()["alias_map"],
            limit=6,
        )

        self.assertEqual("大理石骨病", candidates[0]["disease"])

    def test_offline_topic_osteopetrosis_axis_recall_and_exams(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孩子反复轻微碰撞就骨折，腿疼不愿走路，身高增长慢。哥哥诊断过骨硬化症，父母是近亲，最近脸色苍白、肚子鼓。",
                }
            ],
            "ordered_examinations": [],
        }
        disease_candidates = select_disease_candidates(case_state, load_disease_catalog(), limit=8)
        examination_catalog = load_examination_catalog()
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
            max_items=8,
        )

        self.assertIn("大理石骨病", [item["disease"] for item in disease_candidates])
        self.assertTrue({"基因检测", "骨髓穿刺和活检（BMAB）"}.intersection(set(plan["examinations"])))

    def test_offline_topic_penicillin_contraindication_blocks_first_line_treatment(self):
        result = final_verifier(
            diagnosis="钩端螺旋体病",
            examinations=["血培养"],
            treatment_plan="立即启动抗生素治疗，首选青霉素G静脉滴注，若患者对青霉素过敏，可改用头孢曲松。",
            official_diseases=["钩端螺旋体病"],
            examination_catalog={"实验室检查-微生物": ["血培养"]},
            exam_plan_trace=[],
            case_features={
                "drug_allergies": ["Penicillin", "青霉素"],
                "contraindicated_drugs": ["Penicillin"],
            },
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("contraindicated_drug_recommended", issue_codes)
        self.assertFalse(result["passed"])
        self.assertNotIn("首选青霉素G", result["patched_treatment"])
        self.assertIn("头孢曲松", result["patched_treatment"])

    def test_sixth_round_llm_axis_validator_keeps_supported_axis_only(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "新生儿脐部有鲜红湿润的小肿块，轻轻摩擦就出血，其他皮肤没有黑痣样改变。",
                }
            ],
            "ordered_examinations": [],
        }
        raw_consult = {
            "intake_facts": {
                "demographics": [{"label": "新生儿", "evidence": "新生儿", "confidence": "high"}],
                "anatomic_sites": [{"label": "脐部", "evidence": "脐部", "confidence": "high"}],
            },
            "diagnosis_axes": [
                {
                    "axis_id": "umbilical_granulation_or_vascular_lesion",
                    "status": "suspected",
                    "evidence": ["新生儿", "脐部", "鲜红湿润", "易出血"],
                    "missing_evidence": ["病理或局部专科检查"],
                    "candidate_official_names": ["化脓性肉芽肿", "黑色素细胞痣", "不存在的病名"],
                    "exam_intents": ["局部皮肤病变评估", "出血病变病理评估"],
                    "treatment_risks": ["avoid_no_further_care_for_bleeding_mass"],
                },
                {
                    "axis_id": "unsupported_external_genital_axis",
                    "status": "suspected",
                    "evidence": ["外生殖器"],
                    "candidate_official_names": ["脐尿管囊肿"],
                    "exam_intents": ["外生殖器检查"],
                },
            ],
        }

        consult = validate_axis_consult(
            raw_consult,
            case_state=case_state,
            official_diseases=[disease for diseases in load_disease_catalog().values() for disease in diseases],
        )

        axes = consult["diagnosis_axes"]
        self.assertEqual(["umbilical_granulation_or_vascular_lesion"], [axis["axis_id"] for axis in axes])
        self.assertIn("化脓性肉芽肿", axes[0]["candidate_official_names"])
        self.assertNotIn("不存在的病名", axes[0]["candidate_official_names"])
        self.assertNotIn("黑色素细胞痣", axes[0]["candidate_official_names"])

    def test_offline_topic_umbilical_granulation_candidate_and_features(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "新生儿脐部有鲜红湿润的小肿块，轻轻摩擦就出血，其他皮肤没有黑痣样改变。",
                }
            ],
            "ordered_examinations": [],
        }

        disease_candidates = select_disease_candidates(case_state, load_disease_catalog(), limit=10)
        candidate_names = [item["disease"] for item in disease_candidates]
        features = extract_case_features(case_state, disease_candidates)

        self.assertIn("化脓性肉芽肿", candidate_names)
        if "黑色素细胞痣" in candidate_names:
            self.assertLess(candidate_names.index("化脓性肉芽肿"), candidate_names.index("黑色素细胞痣"))
        self.assertIn("新生儿脐部病变", features["positive_findings"])
        self.assertIn("湿润易出血肿块", features["positive_findings"])
        self.assertIn("无黑痣样改变", features["positive_findings"])

    def test_offline_topic_final_verifier_patches_umbilical_bleeding_mass_no_care(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "新生儿脐部有鲜红湿润的小肿块，轻轻摩擦就出血，其他皮肤没有黑痣样改变。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "化脓性肉芽肿"}])

        result = final_verifier(
            diagnosis="化脓性肉芽肿",
            examinations=["皮肤检查"],
            treatment_plan="考虑局部良性病变，通常自愈，无需进一步处置。",
            official_diseases=["化脓性肉芽肿", "新生儿脐炎", "黑色素细胞痣"],
            examination_catalog={"体格检查": ["皮肤检查"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("undertreated_umbilical_granulation_bleeding_mass", issue_codes)
        self.assertFalse(result["passed"])
        self.assertIn("局部专科评估", result["patched_treatment"])
        self.assertIn("止血", result["patched_treatment"])
        self.assertIn("病理", result["patched_treatment"])

    def test_sixth_round_axis_first_exam_plan_survives_wrong_candidate_diagnosis(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "眼痛、眼红、畏光、视物模糊，医生说角膜有浸润；皮肤有对称靶形丘疹和水疱。",
                }
            ],
            "ordered_examinations": [],
        }
        diagnosis_axes = [
            {
                "axis_id": "corneal_infection_with_target_rash",
                "status": "suspected",
                "evidence": ["眼痛", "畏光", "角膜", "靶形丘疹", "水疱"],
                "candidate_official_names": ["角膜炎", "角膜溃疡"],
                "exam_intents": ["角膜感染评估", "眼部病原培养", "皮肤黏膜反应评估"],
                "treatment_risks": ["infection_before_steroid"],
            }
        ]
        examination_catalog = load_examination_catalog()

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[{"department": "皮肤科", "disease": "囊肿性痤疮", "score": 20}],
            diagnosis_axes=diagnosis_axes,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
            max_items=6,
        )

        self.assertEqual("diagnosis_axis_exam_intent", plan["category"])
        self.assertTrue({"角膜检查", "眼部分泌物培养", "角膜刮片分析"}.intersection(plan["examinations"]))

    def test_sixth_round_rule_axes_positive_and_negative_boundaries(self):
        positive_state = {
            "chat_history": [
                {"from": "patient", "text": "育龄女性，偏头痛反复发作，坐飞机和旅行时加重，月经前也明显，伴恶心头晕。"}
            ],
            "ordered_examinations": [],
        }
        negative_state = {
            "chat_history": [{"from": "patient", "text": "男性，普通紧张性头痛，睡眠差，无旅行诱发和月经相关。"}],
            "ordered_examinations": [],
        }

        positive_axes = select_diagnosis_axes(extract_intake_facts(positive_state))
        negative_axes = select_diagnosis_axes(extract_intake_facts(negative_state))

        self.assertIn("migraine_reproductive_travel_trigger", [axis["axis_id"] for axis in positive_axes])
        self.assertNotIn("migraine_reproductive_travel_trigger", [axis["axis_id"] for axis in negative_axes])

    def test_sixth_round_final_gate_blocks_infection_steroid_and_unsupported_exclusion(self):
        result = final_verifier(
            diagnosis="角膜炎",
            examinations=["角膜检查"],
            treatment_plan="诊断角膜炎，常规使用糖皮质激素滴眼液控制炎症，暂无感染风险。",
            official_diseases=["角膜炎"],
            examination_catalog={"眼科检查": ["角膜检查"]},
            exam_plan_trace=[],
            case_features={
                "positive_findings": ["角膜感染", "靶形皮疹"],
                "diagnosis_axes": [
                    {
                        "axis_id": "corneal_infection_with_target_rash",
                        "status": "suspected",
                        "treatment_risks": ["infection_before_steroid", "unsupported_no_infection_risk"],
                    }
                ],
            },
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("infection_before_steroid", issue_codes)
        self.assertIn("unsupported_no_infection_risk", issue_codes)
        self.assertNotIn("常规使用糖皮质激素", result["patched_treatment"])
        self.assertIn("感染控制", result["patched_treatment"])

    def test_axis_gate_removes_non_negated_oral_steroid(self):
        for treatment_plan in ["口服泼尼松控制炎症。", "给予激素滴眼液控制炎症。"]:
            with self.subTest(treatment_plan=treatment_plan):
                result = final_verifier(
                    diagnosis="角膜炎",
                    examinations=["角膜检查"],
                    treatment_plan=treatment_plan,
                    official_diseases=["角膜炎"],
                    examination_catalog={"眼科检查": ["角膜检查"]},
                    exam_plan_trace=[],
                    case_features={
                        "positive_findings": ["角膜感染"],
                        "diagnosis_axes": [
                            {
                                "axis_id": "corneal_infection_axis",
                                "status": "suspected",
                                "treatment_risks": ["infection_before_steroid"],
                            }
                        ],
                    },
                    safety_profiles=[],
                )

                self.assertNotIn("口服泼尼松", result["patched_treatment"])
                self.assertNotIn("给予激素滴眼液", result["patched_treatment"])
                self.assertIn("感染证据未闭合", result["patched_treatment"])

    def test_sixth_round_final_gate_adds_sle_and_bleeding_risk_goals(self):
        result = final_verifier(
            diagnosis="系统性红斑狼疮",
            examinations=["抗核抗体（ANA）谱"],
            treatment_plan="考虑系统性红斑狼疮，给予羟氯喹和小剂量激素，暂无肾损害证据。",
            official_diseases=["系统性红斑狼疮"],
            examination_catalog={"免疫学检查": ["抗核抗体（ANA）谱"]},
            exam_plan_trace=[],
            case_features={
                "positive_findings": ["光敏皮疹", "关节痛", "补体降低"],
                "diagnosis_axes": [
                    {
                        "axis_id": "sle_organ_thrombosis_reproductive_risk",
                        "status": "suspected",
                        "treatment_risks": ["sle_renal_thrombosis_unclosed", "unsupported_no_renal_damage"],
                    }
                ],
            },
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("sle_renal_thrombosis_unclosed", issue_codes)
        self.assertIn("unsupported_no_renal_damage", issue_codes)
        self.assertIn("尿常规", result["patched_treatment"])

    def test_sixth_round_final_gate_blocks_estrogen_contraception_with_sle_thrombosis_risk(self):
        result = final_verifier(
            diagnosis="系统性红斑狼疮",
            examinations=["抗核抗体（ANA）谱", "补体成分分析"],
            treatment_plan="考虑系统性红斑狼疮，给予羟氯喹；可使用含雌激素避孕药进行避孕。",
            official_diseases=["系统性红斑狼疮"],
            examination_catalog={"免疫学检查": ["抗核抗体（ANA）谱", "补体成分分析"]},
            exam_plan_trace=[],
            case_features={
                "positive_findings": ["光敏皮疹", "关节痛", "补体降低"],
                "diagnosis_axes": [
                    {
                        "axis_id": "sle_organ_thrombosis_reproductive_risk",
                        "status": "suspected",
                        "evidence": ["抗核抗体", "补体降低", "光敏皮疹"],
                        "candidate_official_names": ["系统性红斑狼疮"],
                        "treatment_risks": ["sle_renal_thrombosis_unclosed"],
                    }
                ],
            },
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("estrogen_contraception_with_sle_thrombosis_risk", issue_codes)
        self.assertFalse(result["passed"])
        self.assertNotIn("可使用含雌激素避孕药", result["patched_treatment"])
        self.assertIn("避免含雌激素避孕", result["patched_treatment"])

    def test_axis_gate_removes_non_negated_estrogen_contraception_phrase(self):
        result = final_verifier(
            diagnosis="系统性红斑狼疮",
            examinations=["抗核抗体（ANA）谱"],
            treatment_plan="采用含雌激素避孕方案。",
            official_diseases=["系统性红斑狼疮"],
            examination_catalog={"免疫学检查": ["抗核抗体（ANA）谱"]},
            exam_plan_trace=[],
            case_features={
                "positive_findings": ["光敏皮疹", "关节痛"],
                "diagnosis_axes": [
                    {
                        "axis_id": "sle_organ_thrombosis_reproductive_risk",
                        "status": "suspected",
                        "evidence": ["光敏皮疹", "关节痛"],
                        "candidate_official_names": ["系统性红斑狼疮"],
                        "treatment_risks": ["sle_renal_thrombosis_unclosed"],
                    }
                ],
            },
            safety_profiles=[],
        )

        self.assertNotIn("采用含雌激素避孕", result["patched_treatment"])
        self.assertIn("避免含雌激素避孕", result["patched_treatment"])

    def test_seventh_round_validator_drops_sle_risk_without_sle_evidence(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "48岁女性，搬重物后阴道有下坠和脱出感，排尿困难，尿线变细。没有皮疹、关节痛或口腔溃疡。",
                }
            ],
            "ordered_examinations": [],
        }
        raw_consult = {
            "diagnosis_axes": [
                {
                    "axis_id": "pelvic_floor_or_urinary_outlet_problem",
                    "status": "suspected",
                    "evidence": ["女性", "阴道下坠", "排尿困难"],
                    "candidate_official_names": ["压力性尿失禁"],
                    "exam_intents": ["盆底和泌尿评估"],
                    "treatment_risks": ["sle_renal_thrombosis_unclosed", "unsupported_no_renal_damage"],
                }
            ]
        }

        consult = validate_axis_consult(
            raw_consult,
            case_state=case_state,
            official_diseases=[disease for diseases in load_disease_catalog().values() for disease in diseases],
        )

        self.assertEqual([], consult["treatment_risks"])
        self.assertEqual([], consult["diagnosis_axes"][0]["treatment_risks"])

    def test_seventh_round_final_gate_does_not_add_sle_patch_to_non_sle_case(self):
        result = final_verifier(
            diagnosis="压力性尿失禁",
            examinations=["盆腔检查"],
            treatment_plan="考虑盆底功能障碍，建议盆底康复训练并评估排尿困难。",
            official_diseases=["压力性尿失禁", "系统性红斑狼疮"],
            examination_catalog={"体格检查": ["盆腔检查"]},
            exam_plan_trace=[],
            case_features={
                "positive_findings": ["女性", "阴道脱出感", "排尿困难"],
                "treatment_risks": ["sle_renal_thrombosis_unclosed"],
                "diagnosis_axes": [
                    {
                        "axis_id": "pelvic_floor_or_urinary_outlet_problem",
                        "status": "suspected",
                        "evidence": ["女性", "阴道脱出感", "排尿困难"],
                        "treatment_risks": ["unsupported_no_renal_damage"],
                    }
                ],
            },
            safety_profiles=[],
        )

        self.assertNotIn("SLE", result["patched_treatment"])
        self.assertNotIn("抗磷脂", result["patched_treatment"])

    def test_seventh_round_exam_plan_filters_female_prostate_exams(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "48岁女性，阴道有下坠和脱出感，伴排尿困难、尿线变细，偶尔尿痛。",
                }
            ],
            "ordered_examinations": [],
        }
        examination_catalog = load_examination_catalog()

        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=[{"department": "泌尿外科", "disease": "急性细菌性前列腺炎", "score": 30}],
            diagnosis_axes=[
                {
                    "axis_id": "wrong_prostate_axis",
                    "status": "suspected",
                    "exam_intents": ["评估前列腺压痛和急性炎症体征", "前列腺感染定位"],
                }
            ],
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
            max_items=8,
        )

        self.assertNotIn("前列腺超声", plan["examinations"])
        self.assertNotIn("直肠指检（DRE）", plan["examinations"])
        self.assertFalse(any("前列腺" in exam for exam in plan["examinations"]))

    def test_seventh_round_extracts_family_history_separately_from_personal_history(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孕晚期胎儿估重偏大，之前糖耐量异常。父母都有高血压，但我本人血压一直正常，没有妊娠期高血压。",
                }
            ],
            "ordered_examinations": [],
        }

        features = extract_case_features(case_state, [{"disease": "巨大儿"}])

        self.assertIn("家族高血压", features["family_history"])
        self.assertIn("无妊娠期高血压", features["personal_history"])
        self.assertIn("糖耐量异常", features["personal_history"])
        self.assertNotIn("妊娠期高血压", features["personal_history"])

    def test_seventh_round_final_verifier_blocks_family_history_as_personal_hypertension(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孕晚期胎儿估重偏大，之前糖耐量异常。父母都有高血压，但我本人血压一直正常，没有妊娠期高血压。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "巨大儿"}])

        result = final_verifier(
            diagnosis="巨大儿",
            examinations=["产科超声"],
            treatment_plan="考虑巨大儿。患者有妊娠期高血压病史，需要调整降压药；继续监测胎儿大小和分娩风险。",
            official_diseases=["巨大儿"],
            examination_catalog={"影像学检查": ["产科超声"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("unsupported_personal_history_claim", issue_codes)
        self.assertIn("unsupported_antihypertensive_adjustment", issue_codes)
        self.assertFalse(result["passed"])
        self.assertNotIn("患者有妊娠期高血压病史", result["patched_treatment"])
        self.assertNotIn("调整降压药", result["patched_treatment"])
        self.assertIn("家族高血压仅作为风险因素", result["patched_treatment"])

    def test_seventh_round_extracts_symptomatic_large_renal_cyst_features(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "左侧腰部钝痛长期存在且最近加重，伴尿频。肾脏超声提示较大薄壁囊肿并压迫邻近结构。",
                }
            ],
            "ordered_examinations": [],
        }

        features = extract_case_features(case_state, [{"disease": "肾囊肿"}])

        self.assertIn("腰痛加重", features["positive_findings"])
        self.assertIn("较大肾囊肿", features["positive_findings"])
        self.assertIn("压迫邻近结构", features["positive_findings"])

    def test_seventh_round_final_verifier_patches_undertreated_symptomatic_renal_cyst(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "左侧腰部钝痛长期存在且最近加重，伴尿频。肾脏超声提示较大薄壁囊肿并压迫邻近结构。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "肾囊肿"}])

        result = final_verifier(
            diagnosis="肾囊肿",
            examinations=["肾脏超声"],
            treatment_plan="考虑肾囊肿，建议观察随访，必要时止痛。",
            official_diseases=["肾囊肿"],
            examination_catalog={"影像学检查": ["肾脏超声"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("undertreated_symptomatic_renal_cyst", issue_codes)
        self.assertFalse(result["passed"])
        self.assertIn("泌尿外科", result["patched_treatment"])
        self.assertIn("介入", result["patched_treatment"])
        self.assertIn("血尿", result["patched_treatment"])
        self.assertIn("感染", result["patched_treatment"])

    def test_sixth_round_extracts_recurrent_epistaxis_deviated_septum_context(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "我是学生，长期鼻塞，运动时容易鼻出血，最近反复鼻出血和额头痛。以前鼻部受过外伤，鼻内镜提示鼻中隔偏曲和锐利骨嵴接触下鼻甲。",
                }
            ],
            "ordered_examinations": [],
        }

        features = extract_case_features(case_state, [{"disease": "鼻中隔偏曲"}])

        self.assertIn("反复鼻出血", features["positive_findings"])
        self.assertIn("鼻中隔偏曲", features["positive_findings"])
        self.assertIn("学生", features["personal_context"])
        self.assertIn("运动诱发鼻出血", features["personal_context"])

    def test_sixth_round_final_verifier_patches_deviated_septum_recurrent_epistaxis_care(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "我是学生，长期鼻塞，运动时容易鼻出血，最近反复鼻出血和额头痛。以前鼻部受过外伤，鼻内镜提示鼻中隔偏曲和锐利骨嵴接触下鼻甲。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "鼻中隔偏曲"}])

        result = final_verifier(
            diagnosis="鼻中隔偏曲",
            examinations=["鼻内镜检查"],
            treatment_plan="考虑鼻中隔偏曲，可使用鼻用激素，必要时评估手术。",
            official_diseases=["鼻中隔偏曲"],
            examination_catalog={"内镜检查": ["鼻内镜检查"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("undertreated_deviated_septum_recurrent_epistaxis", issue_codes)
        self.assertFalse(result["passed"])
        self.assertIn("生理盐水", result["patched_treatment"])
        self.assertIn("凝血", result["patched_treatment"])
        self.assertIn("运动防护", result["patched_treatment"])

    def test_final_verifier_does_not_invent_student_or_exercise_context(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "68岁，长期鼻塞并反复鼻出血，鼻内镜提示鼻中隔偏曲。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "鼻中隔偏曲"}])

        result = final_verifier(
            diagnosis="鼻中隔偏曲",
            examinations=["鼻内镜检查"],
            treatment_plan="考虑鼻中隔偏曲，可使用鼻用激素，必要时评估手术。",
            official_diseases=["鼻中隔偏曲"],
            examination_catalog={"内镜检查": ["鼻内镜检查"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        self.assertNotIn("学生", result["patched_treatment"])
        self.assertNotIn("运动防护", result["patched_treatment"])
        self.assertIn("局部保湿", result["patched_treatment"])
        self.assertIn("凝血", result["patched_treatment"])

    def test_offline_topic_extracts_acute_prostatitis_complication_context(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "男性，高热寒战，尿频尿急尿痛，排尿困难差点尿不出来，会阴部胀痛。尿培养提示大肠埃希菌，药敏结果待回。",
                }
            ],
            "ordered_examinations": [],
        }

        features = extract_case_features(case_state, [{"disease": "急性细菌性前列腺炎"}])

        self.assertIn("发热性尿路感染", features["positive_findings"])
        self.assertIn("尿潴留风险", features["positive_findings"])
        self.assertIn("会阴痛", features["positive_findings"])
        self.assertIn("尿培养阳性", features["positive_findings"])

    def test_offline_topic_final_verifier_patches_acute_prostatitis_without_ast_or_retention_plan(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "男性，高热寒战，尿频尿急尿痛，排尿困难差点尿不出来，会阴部胀痛。尿培养提示大肠埃希菌，药敏结果待回。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "急性细菌性前列腺炎"}])

        result = final_verifier(
            diagnosis="急性细菌性前列腺炎",
            examinations=["尿培养"],
            treatment_plan="考虑急性细菌性前列腺炎，给予口服抗生素，多饮水，门诊随访。",
            official_diseases=["急性细菌性前列腺炎"],
            examination_catalog={"实验室检查-微生物": ["尿培养"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("undertreated_acute_bacterial_prostatitis_complication_risk", issue_codes)
        self.assertFalse(result["passed"])
        self.assertIn("药敏", result["patched_treatment"])
        self.assertIn("尿潴留", result["patched_treatment"])
        self.assertIn("脓肿", result["patched_treatment"])

    def test_offline_topic_final_verifier_removes_steroid_continuation_in_high_risk_hsv(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "孩子高热，湿疹处出现成簇水疱和结痂，精神差，近期一直在用全身糖皮质激素。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "卡波西水痘样疹"}])

        result = final_verifier(
            diagnosis="卡波西水痘样疹",
            examinations=["皮肤检查"],
            treatment_plan="考虑卡波西水痘样疹，继续原全身糖皮质激素控制皮炎，口服阿昔洛韦，居家观察。",
            official_diseases=["卡波西水痘样疹"],
            examination_catalog={"体格检查": ["皮肤检查"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=load_knowledge_registry()["treatment_safety_profiles"],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("contraindicated_immunosuppression_continuation_in_severe_hsv", issue_codes)
        self.assertFalse(result["passed"])
        self.assertNotIn("继续原全身糖皮质激素", result["patched_treatment"])
        self.assertIn("暂停或调整全身糖皮质激素", result["patched_treatment"])
        self.assertIn("住院", result["patched_treatment"])
        self.assertIn("静脉阿昔洛韦", result["patched_treatment"])

    def test_offline_topic_final_verifier_blocks_routine_antibiotics_for_adenoviral_conjunctivitis(self):
        result = final_verifier(
            diagnosis="腺病毒性结膜炎",
            examinations=["裂隙灯检查"],
            treatment_plan="考虑腺病毒性结膜炎，预防性使用局部抗生素滴眼液，每日四次，并注意休息。",
            official_diseases=["腺病毒性结膜炎", "结膜炎"],
            examination_catalog={"眼科检查": ["裂隙灯检查"]},
            exam_plan_trace=[],
            case_features={
                "positive_findings": ["眼红", "黏液性分泌物", "近期上感"],
                "candidate_diagnoses": ["腺病毒性结膜炎"],
            },
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("unnecessary_routine_antibiotics_for_viral_conjunctivitis", issue_codes)
        self.assertFalse(result["passed"])
        self.assertNotIn("预防性使用局部抗生素", result["patched_treatment"])
        self.assertIn("人工泪液", result["patched_treatment"])
        self.assertIn("继发细菌感染", result["patched_treatment"])

    def test_final_verifier_preserves_unrelated_eye_drop_frequency(self):
        result = final_verifier(
            diagnosis="腺病毒性结膜炎",
            examinations=["裂隙灯检查"],
            treatment_plan="预防性使用局部抗生素滴眼液，每日四次；人工泪液每日六次。",
            official_diseases=["腺病毒性结膜炎"],
            examination_catalog={"眼科检查": ["裂隙灯检查"]},
            exam_plan_trace=[],
            case_features={
                "positive_findings": ["眼红", "近期上感"],
                "candidate_diagnoses": ["腺病毒性结膜炎"],
            },
            safety_profiles=[],
        )

        self.assertIn("人工泪液每日六次", result["patched_treatment"])

    def test_offline_topic_post_traumatic_brain_injury_not_swallowed_by_migraine(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "4个月前头部磕碰后开始反复头痛和头晕，最近注意力不集中、短期记忆差，怕光怕吵，站立不稳，快速转头更晕。",
                }
            ],
            "ordered_examinations": [],
        }
        disease_candidates = select_disease_candidates(case_state, load_disease_catalog(), limit=10)
        candidate_names = [item["disease"] for item in disease_candidates]

        self.assertIn("创伤后脑损伤综合征", candidate_names)
        self.assertLess(candidate_names.index("创伤后脑损伤综合征"), candidate_names.index("偏头痛"))

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        self.assertIn("post_traumatic_headache_cognitive_vestibular", [axis["axis_id"] for axis in axes])

        examination_catalog = load_examination_catalog()
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            diagnosis_axes=axes,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
            max_items=8,
        )

        self.assertTrue({"神经系统检查", "前庭功能检查", "神经心理评估"}.issubset(set(plan["examinations"])))

    def test_offline_topic_final_verifier_patches_post_traumatic_brain_injury_migraine_only_plan(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "4个月前头部磕碰后开始反复头痛和头晕，最近注意力不集中、短期记忆差，站立不稳，快速转头更晕。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "创伤后脑损伤综合征"}])

        result = final_verifier(
            diagnosis="创伤后脑损伤综合征",
            examinations=["神经系统检查", "前庭功能检查", "神经心理评估"],
            treatment_plan="考虑创伤后头痛，按偏头痛处理，必要时服用布洛芬止痛，注意休息。",
            official_diseases=["创伤后脑损伤综合征", "偏头痛"],
            examination_catalog={"体格检查": ["神经系统检查"], "功能评估": ["前庭功能检查", "神经心理评估"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("undertreated_post_traumatic_brain_injury_cognitive_vestibular", issue_codes)
        self.assertFalse(result["passed"])
        self.assertIn("止痛药", result["patched_treatment"])
        self.assertIn("药物过度使用性头痛", result["patched_treatment"])
        self.assertIn("前庭康复", result["patched_treatment"])
        self.assertIn("认知支持", result["patched_treatment"])

    def test_offline_topic_infant_congenital_heart_disease_axis_and_exams(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "婴儿出生后就呼吸急促，吃奶困难，喂养时明显出汗，哭闹时口唇发绀。查体心率快、肋下凹陷，P2亢进。",
                }
            ],
            "ordered_examinations": [],
        }
        disease_candidates = select_disease_candidates(case_state, load_disease_catalog(), limit=10)
        candidate_names = [item["disease"] for item in disease_candidates]

        self.assertIn("先天性心脏病", candidate_names)
        self.assertNotIn("三房心", candidate_names)

        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        self.assertIn("infant_congenital_structural_heart_disease", [axis["axis_id"] for axis in axes])

        examination_catalog = load_examination_catalog()
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=disease_candidates,
            diagnosis_axes=axes,
            examination_catalog=examination_catalog,
            item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
            diagnosis_exam_profiles=load_knowledge_registry()["diagnosis_exam_profiles"],
            exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
            max_items=8,
        )

        self.assertIn("超声心动图", set(plan["examinations"]))
        self.assertNotIn("心电图（ECG）", set(plan["examinations"]))
        self.assertNotIn("胸部X线检查（CXR）", set(plan["examinations"]))

    def test_offline_topic_final_verifier_patches_undertreated_infant_congenital_heart_disease(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "婴儿出生后就呼吸急促，吃奶困难，喂养时明显出汗，哭闹时口唇发绀。查体心率快、肋下凹陷，P2亢进。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "三房心"}])

        self.assertIn("婴儿早发心肺症状", features["positive_findings"])
        self.assertIn("喂养困难出汗", features["positive_findings"])
        self.assertIn("发绀或肺高压体征", features["positive_findings"])

        result = final_verifier(
            diagnosis="三房心",
            examinations=["超声心动图", "心电图（ECG）", "胸部X线检查（CXR）"],
            treatment_plan="考虑三房心，给予吸氧观察，按呼吸道症状对症处理。",
            official_diseases=["三房心", "先天性心脏病"],
            examination_catalog={"心血管检查": ["超声心动图", "心电图（ECG）"], "影像学检查": ["胸部X线检查（CXR）"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("undertreated_infant_congenital_structural_heart_disease", issue_codes)
        self.assertFalse(result["passed"])
        self.assertTrue("心脏专科" in result["patched_treatment"] or "心外科" in result["patched_treatment"])
        self.assertTrue("手术" in result["patched_treatment"] or "介入" in result["patched_treatment"])
        self.assertIn("肺高压", result["patched_treatment"])
        self.assertIn("心衰", result["patched_treatment"])
        self.assertIn("喂养", result["patched_treatment"])

    def test_offline_topic_final_verifier_patches_generic_migraine_plan_for_reproductive_travel_trigger(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "育龄女性，偏头痛反复发作，坐飞机和旅行时加重，月经前也明显，伴恶心头晕。",
                }
            ],
            "ordered_examinations": [],
        }
        features = extract_case_features(case_state, [{"disease": "偏头痛"}])

        self.assertIn("育龄女性", features["positive_findings"])
        self.assertIn("偏头痛伴恶心头晕", features["positive_findings"])
        self.assertIn("旅行或视觉运动诱发", features["positive_findings"])
        self.assertIn("月经相关", features["positive_findings"])

        result = final_verifier(
            diagnosis="偏头痛",
            examinations=["头颅MRI"],
            treatment_plan="考虑偏头痛，给予布洛芬或曲普坦止痛，避免诱因，注意休息。",
            official_diseases=["偏头痛"],
            examination_catalog={"影像学检查": ["头颅MRI"]},
            exam_plan_trace=[],
            case_features=features,
            safety_profiles=[],
        )

        issue_codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("undertreated_migraine_reproductive_travel_trigger", issue_codes)
        self.assertFalse(result["passed"])
        self.assertIn("妊娠", result["patched_treatment"])
        self.assertTrue("旅行" in result["patched_treatment"] or "晕动" in result["patched_treatment"])
        self.assertIn("发作频率", result["patched_treatment"])
        self.assertIn("预防用药", result["patched_treatment"])


if __name__ == "__main__":
    unittest.main()
