"""Colloquial high-energy hindfoot trauma must not close as sprain.

Clues only (no Patient ID / expected exam pack / reference treatment):
- fall from height while working
- heel (脚后跟) severe pain, swelling, cannot bear weight, foot looks wider
- plain limb X-ray reported no fracture
"""

from __future__ import annotations

import unittest

from agent.legacy_orchestrator import (
    apply_high_energy_hindfoot_diagnosis_guard,
    build_name_map,
    extract_intake_facts,
    final_verifier,
    flatten_examination_catalog,
    has_high_energy_hindfoot_trauma_pattern,
    inject_required_differentials,
    load_disease_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    normalize_name,
    open_coverage_gaps,
    prune_unsupported_disease_candidates,
    required_differential_from_case,
    select_diagnosis_axes,
    select_disease_candidates,
    select_exam_plan,
)


COLLOQUIAL_HEEL = (
    "大概6小时前，我在高处干活不小心掉下来了。"
    "当时脚后跟就剧痛，现在肿得厉害，连地都踩不了，感觉脚变宽了。"
    "还恶心、有点头晕。"
)


class TestHighEnergyHindfootColloquial(unittest.TestCase):
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

    def test_colloquial_markers_match_high_energy_hindfoot_pattern(self):
        self.assertTrue(has_high_energy_hindfoot_trauma_pattern(normalize_name(COLLOQUIAL_HEEL)))

    def test_colloquial_case_opens_axis_and_calcaneus_differential(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": COLLOQUIAL_HEEL}],
            "ordered_examinations": [],
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        axis_ids = {item["axis_id"] for item in axes}
        required = set(required_differential_from_case(case_state))
        candidates = inject_required_differentials(
            select_disease_candidates(case_state, self.disease_catalog, limit=12),
            case_state=case_state,
            disease_catalog=self.disease_catalog,
        )
        names = {item["disease"] for item in candidates}
        plan = select_exam_plan(
            case_state=case_state,
            disease_candidates=candidates,
            diagnosis_axes=axes,
            examination_catalog=self.examination_catalog,
            item_name_map=self.item_name_map,
            diagnosis_exam_profiles=self.exam_profiles,
            exam_intent_rules=self.exam_intent_rules,
            max_items=8,
        )

        self.assertIn("high_energy_hindfoot_trauma", axis_ids)
        self.assertIn("跟骨骨折", required)
        self.assertIn("跟骨骨折", names)
        self.assertTrue(any("X线" in exam or "x线" in exam.lower() for exam in plan["examinations"]))

    def test_negative_plain_film_does_not_allow_sprain_closure(self):
        case_state = {
            "chat_history": [{"from": "patient", "text": COLLOQUIAL_HEEL}],
            "ordered_examinations": ["四肢X线检查"],
            "examination_results": {
                "四肢X线检查": {
                    "status": "normal",
                    "result": {"骨折/脱位/骨性病变": "未见骨折、脱位或骨性病变。"},
                }
            },
        }
        axes = select_diagnosis_axes(extract_intake_facts(case_state))
        candidates = inject_required_differentials(
            select_disease_candidates(case_state, self.disease_catalog, limit=12),
            case_state=case_state,
            disease_catalog=self.disease_catalog,
        )
        candidates = prune_unsupported_disease_candidates(candidates, case_state)
        names = [item["disease"] for item in candidates]
        self.assertIn("跟骨骨折", names)
        # Etiology fracture should rank above soft-tissue sprain when pattern matches.
        self.assertLess(names.index("跟骨骨折"), names.index("踝关节扭伤") if "踝关节扭伤" in names else 99)

        guarded = apply_high_energy_hindfoot_diagnosis_guard(
            "踝关节扭伤",
            case_state,
            candidates,
        )
        self.assertEqual(normalize_name(guarded), normalize_name("跟骨骨折"))

        result = final_verifier(
            diagnosis="踝关节扭伤",
            examinations=["四肢X线检查"],
            treatment_plan="四肢X线未见骨折，按踝关节扭伤予 RICE 制动冰敷抬高，逐步负重训练。",
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=[],
            case_features={
                "case_text": COLLOQUIAL_HEEL,
                "patient_text": COLLOQUIAL_HEEL,
                "positive_findings": [],
                "candidate_diagnoses": names,
                "diagnosis_axes": axes,
                "examination_results": case_state["examination_results"],
            },
            safety_profiles=[],
        )
        codes = {item["code"] for item in result["issues"]}
        self.assertIn("high_energy_hindfoot_overcalled_as_sprain", codes)
        patched = result["patched_treatment"]
        self.assertTrue(any(marker in patched for marker in ["骨科", "跟骨", "进一步", "骨创伤", "CT"]))

    def test_simple_twist_without_fall_is_not_forced(self):
        case_state = {
            "chat_history": [
                {
                    "from": "patient",
                    "text": "走路时不小心崴脚，外踝轻度肿痛，还能慢慢走路。",
                }
            ],
            "ordered_examinations": [],
        }
        self.assertFalse(
            has_high_energy_hindfoot_trauma_pattern(
                normalize_name(case_state["chat_history"][0]["text"])
            )
        )
        self.assertNotIn(
            "high_energy_hindfoot_trauma",
            {item["axis_id"] for item in select_diagnosis_axes(extract_intake_facts(case_state))},
        )
        self.assertNotIn("跟骨骨折", required_differential_from_case(case_state))


if __name__ == "__main__":
    unittest.main()
