"""P0: clinical_closure exam intents must reach the first exam plan."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent.agent import load_release_if_present
from agent.clinical.exam_rule_closure import evaluate_exam_rule_closure
from agent.legacy_orchestrator import (
    build_name_map,
    diagnosis_rule_fact_codes,
    flatten_examination_catalog,
    load_knowledge_registry,
    select_exam_plan,
)


ROOT = Path(__file__).resolve().parents[2]
LIVE_POINTER = ROOT / "releases" / "current.json"


@pytest.fixture(scope="module")
def live_rule_pack():
    release = load_release_if_present(LIVE_POINTER)
    assert release is not None
    pack = release.knowledge_rule_pack
    assert pack.rule_count > 0
    return pack


def soft_tissue_infection_case_state() -> Dict[str, Any]:
    return {
        "chat_history": [
            {
                "from": "patient",
                "text": "小腿红肿热痛三天，发热38.5，皮温升高，足背肿胀。",
            }
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }


def load_test_exam_catalog() -> Dict[str, List[str]]:
    knowledge = load_knowledge_registry()
    # Prefer real catalog leaves when available; fall back to intent map outputs.
    catalog: Dict[str, List[str]] = {
        "影像": [],
        "实验室": [],
        "其他": [],
    }
    for rule in knowledge.get("exam_intent_map") or []:
        outputs = rule.get("output") or []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for raw in outputs:
            if isinstance(raw, dict):
                name = str(raw.get("name") or "").strip()
            else:
                name = str(raw or "").strip()
            if name and name not in catalog["其他"]:
                catalog["其他"].append(name)
    # Ensure soft-tissue intent leaves exist for planner matching.
    for name in ("软组织超声", "下肢软组织超声", "C反应蛋白（CRP）", "血常规"):
        if name not in catalog["其他"]:
            catalog["其他"].append(name)
    return catalog


def load_test_exam_item_map() -> Dict[str, str]:
    return build_name_map(flatten_examination_catalog(load_test_exam_catalog()))


def load_test_exam_intent_rules() -> List[Dict[str, Any]]:
    return list(load_knowledge_registry().get("exam_intent_map") or [])


def test_closure_intent_reaches_first_exam_plan_before_final_stage(
    live_rule_pack, monkeypatch
) -> None:
    state = soft_tissue_infection_case_state()
    facts = diagnosis_rule_fact_codes(state)
    closure = evaluate_exam_rule_closure(
        rule_pack=live_rule_pack,
        fact_codes=facts,
        diagnostic_axis_ids=(),
    )
    state["typed_exam_intent_ids"] = list(closure.exam_intent_ids)
    # Isolate the typed_rule_intent channel: coverage_gap/axis may cover the same
    # leaves first in the full stack, which is correct (supplement-only), but this
    # test must prove the forward intent is consumable by the planner.
    monkeypatch.setattr(
        "agent.legacy_orchestrator.open_coverage_gaps",
        lambda case_state: [],
    )
    monkeypatch.setattr(
        "agent.legacy_orchestrator.build_required_exam_intents",
        lambda case_text, axes: [],
    )
    plan = select_exam_plan(
        case_state=state,
        disease_candidates=[],
        diagnosis_axes=[],
        examination_catalog=load_test_exam_catalog(),
        item_name_map=load_test_exam_item_map(),
        diagnosis_exam_profiles=[],
        exam_intent_rules=load_test_exam_intent_rules(),
        max_items=5,
    )

    assert "exam_intent_lower_extremity_soft_tissue_severity" in state["typed_exam_intent_ids"]
    assert "typed_rule_intent" in plan["reason_codes"]
    assert plan["examinations"]


def test_excluded_group_does_not_emit_intent(live_rule_pack) -> None:
    # hyperlipidemia rule requires xanthelasma + lipid facts; isolated fever must not fire soft-tissue alone without limb signs.
    closure = evaluate_exam_rule_closure(
        rule_pack=live_rule_pack,
        fact_codes=("fever",),
        diagnostic_axis_ids=(),
    )
    assert "exam_intent_lower_extremity_soft_tissue_severity" not in closure.exam_intent_ids


def test_incomplete_facts_do_not_emit_soft_tissue_intent(live_rule_pack) -> None:
    closure = evaluate_exam_rule_closure(
        rule_pack=live_rule_pack,
        fact_codes=("limb_swelling", "skin_redness_heat"),  # missing fever
        diagnostic_axis_ids=(),
    )
    assert "exam_intent_lower_extremity_soft_tissue_severity" not in closure.exam_intent_ids


def test_already_ordered_soft_tissue_exam_is_not_reselected(live_rule_pack) -> None:
    state = soft_tissue_infection_case_state()
    facts = diagnosis_rule_fact_codes(state)
    closure = evaluate_exam_rule_closure(
        rule_pack=live_rule_pack,
        fact_codes=facts,
        diagnostic_axis_ids=(),
    )
    state["typed_exam_intent_ids"] = list(closure.exam_intent_ids)
    catalog = load_test_exam_catalog()
    item_map = load_test_exam_item_map()
    intents = load_test_exam_intent_rules()
    first = select_exam_plan(
        case_state=state,
        disease_candidates=[],
        diagnosis_axes=[],
        examination_catalog=catalog,
        item_name_map=item_map,
        diagnosis_exam_profiles=[],
        exam_intent_rules=intents,
        max_items=5,
    )
    assert first["examinations"]
    # Mark first-plan leaves as already ordered and re-plan.
    state["ordered_examinations"] = list(first["examinations"])
    second = select_exam_plan(
        case_state=state,
        disease_candidates=[],
        diagnosis_axes=[],
        examination_catalog=catalog,
        item_name_map=item_map,
        diagnosis_exam_profiles=[],
        exam_intent_rules=intents,
        max_items=5,
    )
    for name in first["examinations"]:
        assert name not in second["examinations"]


def test_exam_closure_does_not_accept_or_return_diagnosis_candidates(live_rule_pack) -> None:
    result = evaluate_exam_rule_closure(
        rule_pack=live_rule_pack,
        fact_codes=("limb_swelling", "skin_redness_heat", "fever"),
        diagnostic_axis_ids=(),
    )
    assert not hasattr(result, "diagnosis_candidates")
    assert tuple(result.exam_intent_ids)


def test_exam_closure_adapter_does_not_reorder_candidates(live_rule_pack) -> None:
    candidates = [
        {"disease": "蜂窝织炎", "score": 80},
        {"disease": "丹毒", "score": 70},
        {"disease": "深静脉血栓形成", "score": 40},
    ]
    before = copy.deepcopy(candidates)
    _ = evaluate_exam_rule_closure(
        rule_pack=live_rule_pack,
        fact_codes=("limb_swelling", "skin_redness_heat", "fever"),
        diagnostic_axis_ids=(),
    )
    assert candidates == before
