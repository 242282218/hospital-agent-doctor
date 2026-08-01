"""P0-4 remaining migration: helper-fact parity for all leftover axes."""
from __future__ import annotations

import json
from pathlib import Path

from agent.knowledge.typed_rule_engine import RuleContext, apply_rules, parse_compiled_rule_pack
from agent.legacy_orchestrator import diagnosis_rule_fact_codes, normalize_name, select_diagnosis_axes
import agent.legacy_orchestrator as lo

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "agent" / "knowledge" / "clinical_pattern_rules.json"
MAP_PATH = ROOT / "agent" / "knowledge" / "diagnosis_axis_remaining_migration.json"


def _state(text: str) -> dict:
    return {
        "chat_history": [{"from": "patient", "text": text}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
        "diagnosis_axes": [],
    }


def _mapping() -> dict:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def _payload() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def test_remaining_mapping_covers_forty_helpers() -> None:
    mapping = _mapping()
    assert mapping["count"] == 40
    assert len(mapping["rules"]) == 40
    helpers = {item["helper"] for item in mapping["rules"]}
    assert len(helpers) == 40
    assert "reduced_ejection_fraction_heart_failure_axis" not in {
        item["rule_id"] for item in mapping["rules"]
    }


def test_no_has_pattern_branches_remain_in_select_diagnosis_axes() -> None:
    import inspect
    import re

    source = inspect.getsource(select_diagnosis_axes)
    # Only top-level axis-emitting branches are forbidden. Post-materialize
    # specialization may still read has_* helpers (e.g. vector exposure).
    assert re.findall(r"^\s+if has_.*:\n(?:\s+axes\.append)", source, re.M) == []


def test_remaining_rules_shadow_parity_on_controls() -> None:
    payload = _payload()
    pack = parse_compiled_rule_pack(payload)
    rules_by_id = {rule["rule_id"]: rule for rule in payload["rules"]}
    mismatches = []
    for item in _mapping()["rules"]:
        rule = rules_by_id[item["rule_id"]]
        helper = getattr(lo, item["helper"])
        for control in [*rule["positive_controls"], *rule["negative_controls"]]:
            text = " ".join(control["facts"])
            legacy = bool(helper(text)) or bool(helper(normalize_name(text)))
            facts = diagnosis_rule_fact_codes(_state(text))
            # For excluded helpers, include their fact if present so exclusion can fire.
            result = apply_rules(pack, "diagnosis_candidates", RuleContext(fact_codes=facts))
            decision = next(d for d in result.decisions if d.rule_id == item["rule_id"])
            typed = decision.outcome in {"applied", "matched_no_change"}
            # If excluded group present, typed should not apply even if primary fires.
            expected = control["kind"] == "positive"
            if item["excluded_codes"]:
                # Recompute expected with exclusions.
                excluded_hit = any(
                    code in facts for code in item["excluded_codes"]
                )
                if excluded_hit:
                    expected = False
            if legacy != (item["fact_code"] in facts) and control["kind"] == "positive":
                # Primary fact extraction must mirror helper for positives.
                mismatches.append(
                    {
                        "rule_id": item["rule_id"],
                        "control_id": control["control_id"],
                        "kind": "fact_extract",
                        "legacy": legacy,
                        "fact": item["fact_code"] in facts,
                        "text": text,
                    }
                )
                continue
            if typed != expected:
                mismatches.append(
                    {
                        "rule_id": item["rule_id"],
                        "control_id": control["control_id"],
                        "legacy": legacy,
                        "typed": typed,
                        "expected": expected,
                        "outcome": decision.outcome,
                        "text": text,
                    }
                )
    assert mismatches == []


def test_select_diagnosis_axes_emits_migrated_axis_from_patient_text() -> None:
    # Soft-tissue batch1 still works through typed pack.
    axes = select_diagnosis_axes(
        {},
        case_state=_state("小腿红肿热痛，伴发热寒战"),
    )
    assert any(
        axis.get("axis_id") == "acute_lower_extremity_soft_tissue_infection"
        for axis in axes
    )


def test_hfref_decompensation_emits_single_acute_heart_failure_axis() -> None:
    axes = select_diagnosis_axes(
        {},
        case_state=_state("LVEF 35%，端坐呼吸，双下肢水肿，活动后气短"),
    )

    heart_failure_axes = [
        axis for axis in axes if "心力衰竭" in axis.get("candidate_official_names", [])
    ]
    assert len(heart_failure_axes) == 1
    axis = heart_failure_axes[0]
    assert axis["axis_id"] == "heart_failure_decompensation"
    assert "急性心衰利钠肽与容量评估" in axis["exam_intents"]
    assert "do_not_delay_diuresis_for_labs_only" in axis["treatment_risks"]
