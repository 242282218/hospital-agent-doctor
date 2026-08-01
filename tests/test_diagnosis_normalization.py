import unittest

from agent.legacy_orchestrator import (
    accept_normalization_suggestion,
    case_feature_label_set,
    differential_raw_name_matches,
    normalize_candidates_from_diagnostic_context,
    resolve_official_from_surface_name,
)
from agent.prompt import DIAGNOSTIC_CONTEXT_PROMPT


class DiagnosisNormalizationTest(unittest.TestCase):
    def test_case_feature_label_set_reads_object_arrays(self):
        features = {
            "demographics": [
                {"label": "成年男性", "evidence": "患者为成年男性", "confidence": "high"}
            ],
            "symptom_clusters": [
                {"label": "发热性尿路感染", "evidence": "高热、尿痛", "confidence": "high"}
            ],
            "red_flags": [
                {"label": "排尿困难", "evidence": "患者诉排尿困难", "confidence": "medium"}
            ],
        }

        labels = case_feature_label_set(features)

        self.assertIn("成年男性", labels)
        self.assertIn("发热性尿路感染", labels)
        self.assertIn("排尿困难", labels)

    def test_case_feature_label_set_ignores_non_array_slots(self):
        features = {
            "demographics": {"label": "成年男性", "evidence": "患者为成年男性", "confidence": "high"},
            "symptom_clusters": "发热性尿路感染",
        }

        labels = case_feature_label_set(features)

        self.assertEqual(set(), labels)

    def test_accepts_context_supported_official_suggestion(self):
        features = {
            "demographics": [{"label": "成年男性", "evidence": "男性", "confidence": "high"}],
            "symptom_clusters": [{"label": "发热性尿路感染", "evidence": "高热尿痛", "confidence": "high"}],
            "exam_evidence": [{"label": "尿液感染证据", "evidence": "白细胞酯酶阳性", "confidence": "high"}],
            "organ_risk": [{"label": "前列腺受累风险", "evidence": "排尿困难", "confidence": "medium"}],
        }
        suggestion = {
            "raw_name": "前列腺炎",
            "suggested_official_name": "急性细菌性前列腺炎",
            "confidence": "medium",
            "supporting_feature_labels": ["成年男性", "发热性尿路感染", "前列腺受累风险"],
            "rationale": "男性发热性尿路感染伴排尿困难。",
        }

        result = accept_normalization_suggestion(
            suggestion,
            case_features=features,
            differential_raw_names=["前列腺炎", "急性膀胱炎"],
            official_disease_map={"急性细菌性前列腺炎": "急性细菌性前列腺炎"},
        )

        self.assertTrue(result["accepted"])
        self.assertEqual("急性细菌性前列腺炎", result["normalized_diagnosis"])
        self.assertEqual("context_suggestion", result["source"])

    def test_rejects_suggestion_with_missing_supporting_feature(self):
        features = {
            "demographics": [{"label": "成年男性", "evidence": "男性", "confidence": "high"}],
        }
        suggestion = {
            "raw_name": "前列腺炎",
            "suggested_official_name": "急性细菌性前列腺炎",
            "confidence": "medium",
            "supporting_feature_labels": ["成年男性", "尿液感染证据"],
        }

        result = accept_normalization_suggestion(
            suggestion,
            case_features=features,
            differential_raw_names=["前列腺炎"],
            official_disease_map={"急性细菌性前列腺炎": "急性细菌性前列腺炎"},
        )

        self.assertFalse(result["accepted"])
        self.assertEqual("missing_supporting_features", result["reason"])

    def test_rejects_suggestion_when_label_is_not_exact_match(self):
        features = {
            "demographics": [{"label": "成年 男性", "evidence": "男性", "confidence": "high"}],
        }
        suggestion = {
            "raw_name": "前列腺炎",
            "suggested_official_name": "急性细菌性前列腺炎",
            "confidence": "medium",
            "supporting_feature_labels": ["成年男性"],
        }

        result = accept_normalization_suggestion(
            suggestion,
            case_features=features,
            differential_raw_names=["前列腺炎"],
            official_disease_map={"急性细菌性前列腺炎": "急性细菌性前列腺炎"},
        )

        self.assertFalse(result["accepted"])
        self.assertEqual("missing_supporting_features", result["reason"])

    def test_rejects_suggestion_without_supporting_features(self):
        features = {
            "demographics": [{"label": "成年男性", "evidence": "男性", "confidence": "high"}],
        }
        suggestion = {
            "raw_name": "前列腺炎",
            "suggested_official_name": "急性细菌性前列腺炎",
            "confidence": "medium",
            "supporting_feature_labels": [],
        }

        result = accept_normalization_suggestion(
            suggestion,
            case_features=features,
            differential_raw_names=["前列腺炎"],
            official_disease_map={"急性细菌性前列腺炎": "急性细菌性前列腺炎"},
        )

        self.assertFalse(result["accepted"])
        self.assertEqual("empty_supporting_features", result["reason"])

    def test_rejects_non_whitelisted_confidence(self):
        features = {
            "demographics": [{"label": "成年男性", "evidence": "男性", "confidence": "high"}],
        }
        suggestion = {
            "raw_name": "前列腺炎",
            "suggested_official_name": "急性细菌性前列腺炎",
            "confidence": "不确定",
            "supporting_feature_labels": ["成年男性"],
        }

        result = accept_normalization_suggestion(
            suggestion,
            case_features=features,
            differential_raw_names=["前列腺炎"],
            official_disease_map={"急性细菌性前列腺炎": "急性细菌性前列腺炎"},
        )

        self.assertFalse(result["accepted"])
        self.assertEqual("low_confidence", result["reason"])

    def test_rejects_female_to_prostate_conflict(self):
        features = {
            "demographics": [{"label": "女性", "evidence": "患者为女性", "confidence": "high"}],
            "exam_evidence": [{"label": "尿液感染证据", "evidence": "尿检阳性", "confidence": "high"}],
        }
        suggestion = {
            "raw_name": "前列腺炎",
            "suggested_official_name": "急性细菌性前列腺炎",
            "confidence": "medium",
            "supporting_feature_labels": ["女性", "尿液感染证据"],
        }

        result = accept_normalization_suggestion(
            suggestion,
            case_features=features,
            differential_raw_names=["前列腺炎"],
            official_disease_map={"急性细菌性前列腺炎": "急性细菌性前列腺炎"},
        )

        self.assertFalse(result["accepted"])
        self.assertEqual("hard_conflict", result["reason"])

    def test_rejects_diabetic_neuropathy_without_diabetes_evidence(self):
        features = {
            "symptom_clusters": [
                {"label": "周围神经病变症状", "evidence": "脚底烧灼感和麻木", "confidence": "high"}
            ],
            "organ_risk": [
                {"label": "心脏功能受损风险", "evidence": "端坐呼吸和下肢水肿", "confidence": "high"}
            ],
        }
        suggestion = {
            "raw_name": "糖尿病周围神经病变",
            "suggested_official_name": "糖尿病周围神经病变",
            "confidence": "medium",
            "supporting_feature_labels": ["周围神经病变症状"],
        }

        result = accept_normalization_suggestion(
            suggestion,
            case_features=features,
            differential_raw_names=["糖尿病周围神经病变"],
            official_disease_map={"糖尿病周围神经病变": "糖尿病周围神经病变"},
        )

        self.assertFalse(result["accepted"])
        self.assertEqual("missing_required_axis_evidence", result["reason"])

    def test_context_suggestion_reranks_over_neighbor_diagnosis(self):
        diagnostic_context = {
            "case_features": {
                "demographics": [{"label": "成年男性", "evidence": "男性", "confidence": "high"}],
                "symptom_clusters": [{"label": "发热性尿路感染", "evidence": "高热尿痛", "confidence": "high"}],
                "exam_evidence": [{"label": "尿液感染证据", "evidence": "尿检阳性", "confidence": "high"}],
                "organ_risk": [{"label": "前列腺受累风险", "evidence": "排尿困难", "confidence": "medium"}],
            },
            "differential": [
                {"raw_name": "急性膀胱炎", "rank": 1, "reason": "尿频尿痛"},
                {"raw_name": "前列腺炎", "rank": 2, "reason": "男性发热尿路感染伴排尿困难"},
            ],
            "normalization_suggestions": [
                {
                    "raw_name": "前列腺炎",
                    "suggested_official_name": "急性细菌性前列腺炎",
                    "confidence": "medium",
                    "supporting_feature_labels": ["成年男性", "发热性尿路感染", "前列腺受累风险"],
                }
            ],
        }
        literal_candidates = [
            {"department": "泌尿外科", "disease": "急性膀胱炎", "score": 20},
            {"department": "泌尿外科", "disease": "急性细菌性前列腺炎", "score": 30},
        ]

        candidates = normalize_candidates_from_diagnostic_context(
            diagnostic_context,
            literal_candidates=literal_candidates,
            disease_catalog={"泌尿外科": ["急性膀胱炎", "急性细菌性前列腺炎"]},
            official_disease_map={
                "急性膀胱炎": "急性膀胱炎",
                "急性细菌性前列腺炎": "急性细菌性前列腺炎",
            },
            alias_rules=[],
            limit=4,
        )

        self.assertEqual("急性细菌性前列腺炎", candidates[0]["disease"])
        self.assertEqual("context_suggestion", candidates[0]["source"])

    def test_strong_literal_candidate_beats_medium_context_suggestion(self):
        diagnostic_context = {
            "case_features": {
                "demographics": [{"label": "成年男性", "evidence": "男性", "confidence": "high"}],
                "symptom_clusters": [{"label": "发热性尿路感染", "evidence": "高热尿痛", "confidence": "high"}],
                "organ_risk": [{"label": "前列腺受累风险", "evidence": "排尿困难", "confidence": "medium"}],
            },
            "differential": [
                {"raw_name": "前列腺炎", "rank": 1, "reason": "男性发热尿路感染伴排尿困难"},
            ],
            "normalization_suggestions": [
                {
                    "raw_name": "前列腺炎",
                    "suggested_official_name": "急性细菌性前列腺炎",
                    "confidence": "medium",
                    "supporting_feature_labels": ["成年男性", "发热性尿路感染", "前列腺受累风险"],
                }
            ],
        }
        literal_candidates = [
            {"department": "感染科", "disease": "肾盂肾炎", "score": 95},
            {"department": "泌尿外科", "disease": "急性细菌性前列腺炎", "score": 30},
        ]

        candidates = normalize_candidates_from_diagnostic_context(
            diagnostic_context,
            literal_candidates=literal_candidates,
            disease_catalog={"感染科": ["肾盂肾炎"], "泌尿外科": ["急性细菌性前列腺炎"]},
            official_disease_map={
                "肾盂肾炎": "肾盂肾炎",
                "急性细菌性前列腺炎": "急性细菌性前列腺炎",
            },
            alias_rules=[],
            limit=4,
        )

        self.assertEqual("肾盂肾炎", candidates[0]["disease"])
        self.assertEqual("literal_context", candidates[0]["source"])

    def test_generic_final_diagnosis_refines_to_context_ranked_candidate(self):
        from agent.legacy_orchestrator import normalize_diagnosis

        normalized = normalize_diagnosis(
            "细菌感染",
            official_diseases=["细菌感染", "急性膀胱炎", "急性细菌性前列腺炎"],
            alias_rules=[],
            disease_candidates=[
                {"disease": "急性细菌性前列腺炎", "score": 60, "source": "context_suggestion"},
                {"disease": "急性膀胱炎", "score": 50, "source": "official_catalog"},
            ],
        )

        self.assertEqual("急性细菌性前列腺炎", normalized["normalized_diagnosis"])
        self.assertEqual("specific_candidate", normalized["source"])

    def test_unconditional_alias_normalizes_sle(self):
        from agent.legacy_orchestrator import normalize_diagnosis

        normalized = normalize_diagnosis(
            "SLE",
            official_diseases=["系统性红斑狼疮"],
            alias_rules=[
                {
                    "status": "verified",
                    "input": ["SLE", "systemic lupus erythematosus"],
                    "output": "系统性红斑狼疮",
                }
            ],
            disease_candidates=[],
        )

        self.assertEqual("系统性红斑狼疮", normalized["normalized_diagnosis"])
        self.assertEqual("alias_map", normalized["source"])

    def test_prompt_requires_structured_diagnostic_context(self):
        self.assertIn("case_features", DIAGNOSTIC_CONTEXT_PROMPT)
        self.assertIn("normalization_suggestions", DIAGNOSTIC_CONTEXT_PROMPT)
        self.assertIn("supporting_feature_labels", DIAGNOSTIC_CONTEXT_PROMPT)
        self.assertNotIn("急性细菌性前列腺炎", DIAGNOSTIC_CONTEXT_PROMPT)




    def test_gout_suggestion_raw_matches_longer_differential_surface(self):
        features = {
            "symptom_clusters": [
                {"label": "夜间足趾剧痛", "evidence": "大脚趾夜间剧痛", "confidence": "high"}
            ],
            "exam_evidence": [
                {"label": "高尿酸", "evidence": "血尿酸 9.6", "confidence": "high"}
            ],
        }
        suggestion = {
            "raw_name": "痛风",
            "suggested_official_name": "痛风",
            "confidence": "high",
            "supporting_feature_labels": ["夜间足趾剧痛", "高尿酸"],
        }
        result = accept_normalization_suggestion(
            suggestion,
            case_features=features,
            differential_raw_names=["痛风性关节炎", "假性痛风", "感染性关节炎"],
            official_disease_map={"痛风": "痛风", "关节炎": "关节炎"},
        )
        self.assertTrue(result["accepted"], result)
        self.assertEqual("痛风", result["normalized_diagnosis"])

    def test_patient_04007_gout_surface_enters_candidates_top(self):
        diagnostic_context = {
            "case_features": {
                "symptom_clusters": [
                    {"label": "夜间足趾剧痛", "evidence": "大脚趾夜间剧痛", "confidence": "high"}
                ],
                "exam_evidence": [
                    {"label": "高尿酸", "evidence": "血尿酸 9.6", "confidence": "high"},
                    {"label": "痛风石", "evidence": "痛风石", "confidence": "high"},
                ],
            },
            "differential": [
                {"raw_name": "痛风性关节炎", "rank": 1, "reason": "夜间大脚趾剧痛+高尿酸"},
                {"raw_name": "假性痛风", "rank": 2, "reason": "鉴别"},
                {"raw_name": "感染性关节炎", "rank": 3, "reason": "鉴别"},
            ],
            "normalization_suggestions": [
                {
                    "raw_name": "痛风",
                    "suggested_official_name": "痛风",
                    "confidence": "high",
                    "supporting_feature_labels": ["夜间足趾剧痛", "高尿酸"],
                }
            ],
        }
        candidates = normalize_candidates_from_diagnostic_context(
            diagnostic_context,
            literal_candidates=[{"disease": "慢性肾脏病", "score": 40, "source": "catalog_match"}],
            disease_catalog={"风湿免疫科": ["痛风", "关节炎"], "肾内科": ["慢性肾脏病"]},
            official_disease_map={
                "痛风": "痛风",
                "关节炎": "关节炎",
                "慢性肾脏病": "慢性肾脏病",
            },
            alias_rules=[],
            limit=6,
            trusted_case_text="大脚趾夜间剧痛，有痛风石，血尿酸 9.6，肾功能轻度异常。",
        )
        names = [item["disease"] for item in candidates]
        self.assertIn("痛风", names, names)
        self.assertEqual("痛风", candidates[0]["disease"])

    def test_surface_containment_prefers_longer_official_name(self):
        official = {
            "感染性关节炎": "感染性关节炎",
            "关节炎": "关节炎",
            "痛风": "痛风",
        }
        self.assertEqual(
            "感染性关节炎",
            resolve_official_from_surface_name("感染性关节炎", official),
        )
        self.assertEqual("痛风", resolve_official_from_surface_name("痛风性关节炎", official))
        self.assertTrue(differential_raw_name_matches("痛风", ["痛风性关节炎"]))
        self.assertFalse(differential_raw_name_matches("蜂窝织炎", ["痛风性关节炎"]))


if __name__ == "__main__":
    unittest.main()
