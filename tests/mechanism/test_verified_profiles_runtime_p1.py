"""T08: runtime may only read frozen verified profiles, never candidates.

The reader is the boundary between offline aggregation and the online path. It
accepts frozen registry assets only, refuses unknown schemas, and never lets a
profile leak into the generic note channel where it would be serialized into a
prompt verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent.knowledge.verified_profiles import (
    VerifiedProfileIndex,
    validate_verified_profile_asset,
)
from agent.memory import VerifiedOnlyMemory
from offline.ground_truth_profiles import GroundTruthRecord
from offline.profile_candidates import (
    aggregate_exam_profiles,
    aggregate_treatment_profiles,
)

_RECEIPT_HASH = "sha256:" + "c" * 64
_DIAGNOSIS = "卡波西水痘样疹"
_CATALOG_ORDER = ["体格检查", "全血细胞计数（CBC）", "病毒核酸检测（Viral NAT）"]


def _exam_profile() -> Dict[str, Any]:
    records = [
        GroundTruthRecord(
            patient_id="Patient_1%04d" % index,
            diagnosis_items=(_DIAGNOSIS,),
            exam_items=("体格检查", "全血细胞计数（CBC）"),
            treatment_text="静脉注射阿昔洛韦抗病毒；住院监测；补液支持。",
            contraindication_items=("糖皮质激素",),
            source_run="train_harvest_1",
            evaluation_hash="sha256:" + "a" * 64,
            partition="build",
        )
        for index in range(1, 6)
    ]
    return aggregate_exam_profiles(
        records,
        partition="build",
        exam_catalog_order=_CATALOG_ORDER,
        source_receipt_hash=_RECEIPT_HASH,
    )[0]


def _treatment_profile() -> Dict[str, Any]:
    records = [
        GroundTruthRecord(
            patient_id="Patient_1%04d" % index,
            diagnosis_items=(_DIAGNOSIS,),
            exam_items=("体格检查",),
            treatment_text="静脉注射阿昔洛韦抗病毒；住院监测；补液支持。",
            contraindication_items=("糖皮质激素",),
            source_run="train_harvest_1",
            evaluation_hash="sha256:" + "a" * 64,
            partition="build",
        )
        for index in range(1, 6)
    ]
    return aggregate_treatment_profiles(
        records,
        partition="build",
        source_receipt_hash=_RECEIPT_HASH,
    )["profiles"][0]


def _reflection_rule() -> Dict[str, Any]:
    return {
        "schema_version": "reflection-rule/v1",
        "trigger_codes": ["immunosuppressed_infection", "vesicular_rash"],
        "stages": ["diagnosis", "examination"],
        "note": "免疫抑制伴疱疹样皮损时，优先闭合病毒病原和继发细菌感染风险。",
        "source_refs": ["reflection_a", "reflection_b", "reflection_c"],
        "support_count": 3,
        "source_receipt_hash": _RECEIPT_HASH,
    }


def _asset(candidate_type: str, content: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": "%s__x" % candidate_type,
        "candidate_type": candidate_type,
        "content": content,
    }


def _registry(tmp_path: Path, assets: List[Dict[str, Any]]) -> Path:
    path = tmp_path / "verified_registry.json"
    payload = {"schema_version": "verified-registry/v1", "assets": assets}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_index_returns_exam_profiles_for_exact_diagnosis() -> None:
    index = VerifiedProfileIndex([_asset("disease_exam_profile", _exam_profile())])
    found = index.exam_profiles([_DIAGNOSIS])
    assert len(found) == 1
    assert found[0]["diagnosis_name"] == _DIAGNOSIS
    # Unknown or partially matching names must not resolve.
    assert index.exam_profiles(["卡波西"]) == []
    assert index.exam_profiles(["不存在的病"]) == []


def test_index_returns_deepcopies() -> None:
    index = VerifiedProfileIndex([_asset("disease_exam_profile", _exam_profile())])
    first = index.exam_profiles([_DIAGNOSIS])
    first[0]["exam_items"][0]["name"] = "被污染"
    second = index.exam_profiles([_DIAGNOSIS])
    assert second[0]["exam_items"][0]["name"] != "被污染"


def test_index_returns_treatment_profiles() -> None:
    index = VerifiedProfileIndex([_asset("disease_treatment_profile", _treatment_profile())])
    found = index.treatment_profiles([_DIAGNOSIS])
    assert len(found) == 1
    assert found[0]["goal_codes"]
    assert index.exam_profiles([_DIAGNOSIS]) == []


def test_index_retrieves_reflection_notes_by_trigger_and_stage() -> None:
    index = VerifiedProfileIndex([_asset("reflection_rule", _reflection_rule())])
    notes = index.reflection_notes(
        trigger_codes={"vesicular_rash"},
        stage="diagnosis",
    )
    assert len(notes) == 1
    # Stage must match.
    assert index.reflection_notes(trigger_codes={"vesicular_rash"}, stage="treatment") == []
    # Trigger must intersect.
    assert index.reflection_notes(trigger_codes={"unrelated_code"}, stage="diagnosis") == []
    # No trigger at all yields nothing.
    assert index.reflection_notes(trigger_codes=set(), stage="diagnosis") == []


def test_index_reflection_notes_respect_limit() -> None:
    rules = []
    for suffix in ("a", "b", "c", "d"):
        rule = _reflection_rule()
        rule["note"] = "反思要点%s：优先闭合病原学证据。" % suffix
        rules.append(_asset("reflection_rule", rule))
    index = VerifiedProfileIndex(rules)
    notes = index.reflection_notes(
        trigger_codes={"vesicular_rash"},
        stage="diagnosis",
        limit=2,
    )
    assert len(notes) == 2


@pytest.mark.parametrize(
    "candidate_type",
    ["disease_exam_profile", "disease_treatment_profile", "reflection_rule"],
)
def test_unknown_schema_fails_closed(candidate_type: str) -> None:
    with pytest.raises(ValueError):
        VerifiedProfileIndex([_asset(candidate_type, {"schema_version": "bogus/v9"})])


def test_validate_rejects_unknown_candidate_type() -> None:
    with pytest.raises(ValueError):
        validate_verified_profile_asset("mystery_type", {"schema_version": "x"})


def test_memory_routes_profiles_away_from_generic_notes(tmp_path: Path) -> None:
    """A profile must never be json-dumped into the generic prompt note channel."""
    registry = _registry(
        tmp_path,
        [
            _asset("disease_exam_profile", _exam_profile()),
            _asset("disease_treatment_profile", _treatment_profile()),
            _asset("reflection_rule", _reflection_rule()),
        ],
    )
    memory = VerifiedOnlyMemory(registry)
    notes = memory.load_notes()
    blob = " ".join(notes)
    for marker in ("disease-exam-profile", "disease-treatment-profile", "exam_items", "goal_codes"):
        assert marker not in blob
    assert notes == []


def test_memory_exposes_profiles_through_index(tmp_path: Path) -> None:
    registry = _registry(
        tmp_path,
        [
            _asset("disease_exam_profile", _exam_profile()),
            _asset("disease_treatment_profile", _treatment_profile()),
        ],
    )
    memory = VerifiedOnlyMemory(registry)
    assert memory.exam_profiles([_DIAGNOSIS])[0]["diagnosis_name"] == _DIAGNOSIS
    assert memory.treatment_profiles([_DIAGNOSIS])[0]["goal_codes"]


def test_memory_fails_closed_on_unknown_asset_type(tmp_path: Path) -> None:
    registry = _registry(tmp_path, [_asset("mystery_type", {"a": 1})])
    with pytest.raises(ValueError):
        VerifiedOnlyMemory(registry)


def test_memory_rejects_candidate_file(tmp_path: Path) -> None:
    """A raw candidate file is not a frozen registry and must fail closed."""
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "schema_version": "candidate/v1",
                "candidate_id": "x",
                "candidate_type": "disease_exam_profile",
                "proposed_effect": _exam_profile(),
                "status": "candidate",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        VerifiedOnlyMemory(candidate)


def test_offline_to_runtime_round_trip_preserves_effect(tmp_path: Path) -> None:
    """T05/T06 output must survive registry serialization byte-for-byte."""
    exam = _exam_profile()
    treatment = _treatment_profile()
    registry = _registry(
        tmp_path,
        [
            _asset("disease_exam_profile", exam),
            _asset("disease_treatment_profile", treatment),
        ],
    )
    memory = VerifiedOnlyMemory(registry)
    assert memory.exam_profiles([exam["diagnosis_name"]]) == [exam]
    assert memory.treatment_profiles([treatment["diagnosis_name"]]) == [treatment]


def test_load_notes_returns_reflection_only_with_trigger_and_stage(tmp_path: Path) -> None:
    registry = _registry(tmp_path, [_asset("reflection_rule", _reflection_rule())])
    memory = VerifiedOnlyMemory(registry)
    assert memory.load_notes() == []
    notes = memory.load_notes(
        trigger_codes={"vesicular_rash"},
        stage="diagnosis",
    )
    assert len(notes) == 1
    assert "免疫抑制" in notes[0]
    # include_candidates must never widen the source.
    assert memory.load_notes(include_candidates=True) == []


def test_select_exam_plan_uses_verified_exam_profile_after_prior() -> None:
    """Priority is exact prior > verified profile > heuristics."""
    from agent.legacy_orchestrator import MyDoctorAgent, select_exam_plan

    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=None)
    leaf = "超声心动图"
    case_state = {
        "patient_id": "Patient_01061",
        "mode": "test",
        "chat_history": [
            {"from": "doctor", "text": "哪里不适"},
            {"from": "patient", "text": "活动后气促伴心悸。"},
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "decision_trace": [],
        "exam_decision_trace": [],
    }
    profile = {
        "schema_version": "disease-exam-profile/v1",
        "diagnosis_name": "三房心",
        "exam_items": [
            {"name": leaf, "support_count": 9, "support_ratio": 0.9, "rank": 1},
        ],
        "support_case_count": 10,
        "source_receipt_hash": "sha256:" + "c" * 64,
    }
    plan = select_exam_plan(
        case_state=case_state,
        disease_candidates=[{"disease": "三房心", "score": 10, "source": "catalog_match"}],
        examination_catalog=agent.examination_catalog,
        item_name_map=agent.exam_item_map,
        verified_exam_profiles=[profile],
        max_items=4,
    )
    assert leaf in plan["examinations"]
    assert "verified_disease_profile" in plan["reason_codes"]


def test_verified_exam_profile_leaf_still_passes_catalog_and_dedup() -> None:
    from agent.legacy_orchestrator import MyDoctorAgent, select_exam_plan

    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=None)
    case_state = {
        "patient_id": "Patient_01061",
        "mode": "test",
        "chat_history": [{"from": "patient", "text": "活动后气促。"}],
        "ordered_examinations": ["超声心动图"],
        "invalid_examinations": [],
        "examination_results": {},
        "decision_trace": [],
        "exam_decision_trace": [],
    }
    profile = {
        "schema_version": "disease-exam-profile/v1",
        "diagnosis_name": "三房心",
        "exam_items": [
            {"name": "超声心动图", "support_count": 9, "support_ratio": 0.9, "rank": 1},
            {"name": "不在目录的检查", "support_count": 8, "support_ratio": 0.8, "rank": 2},
        ],
        "support_case_count": 10,
        "source_receipt_hash": "sha256:" + "c" * 64,
    }
    plan = select_exam_plan(
        case_state=case_state,
        disease_candidates=[{"disease": "三房心", "score": 10, "source": "catalog_match"}],
        examination_catalog=agent.examination_catalog,
        item_name_map=agent.exam_item_map,
        verified_exam_profiles=[profile],
        max_items=4,
    )
    # Already ordered leaf is not re-ordered, and a non-catalog name never enters.
    assert plan["examinations"].count("超声心动图") == 0
    assert "不在目录的检查" not in plan["examinations"]


def test_treatment_profile_goal_codes_become_missing_goal_evidence() -> None:
    from agent.legacy_orchestrator import build_treatment_review_evidence_catalog

    catalog = build_treatment_review_evidence_catalog(
        case_state={"chat_history": [], "examination_results": {}},
        diagnosis="卡波西水痘样疹",
        diagnosis_axes=[],
        verifier_issues=[],
        treatment_profiles=[
            {
                "schema_version": "disease-treatment-profile/v1",
                "diagnosis_name": "卡波西水痘样疹",
                "goal_codes": ["antiviral_therapy"],
                "risk_codes": ["sepsis_risk"],
                "contraindication_codes": ["systemic_corticosteroid"],
                "support_stats": {
                    "support_case_count": 12,
                    "goal_support_counts": {"antiviral_therapy": 11},
                    "risk_support_counts": {"sepsis_risk": 9},
                    "contraindication_support_counts": {"systemic_corticosteroid": 8},
                },
                "source_receipt_hash": "sha256:" + "c" * 64,
            }
        ],
        treatment_plan="仅对症退热处理。",
    )
    sources = {entry["source"] for entry in catalog}
    assert "profile_goal" in sources
    goal_entries = [entry for entry in catalog if entry["source"] == "profile_goal"]
    assert goal_entries and all(entry["polarity"] == "missing" for entry in goal_entries)
    # Codes only; no free-text prescription or dose may be introduced.
    blob = json.dumps(catalog, ensure_ascii=False)
    assert "阿昔洛韦" not in blob
    assert "mg" not in blob


def test_treatment_profile_goal_already_covered_is_not_missing() -> None:
    from agent.legacy_orchestrator import build_treatment_review_evidence_catalog

    catalog = build_treatment_review_evidence_catalog(
        case_state={"chat_history": [], "examination_results": {}},
        diagnosis="卡波西水痘样疹",
        diagnosis_axes=[],
        verifier_issues=[],
        treatment_profiles=[
            {
                "schema_version": "disease-treatment-profile/v1",
                "diagnosis_name": "卡波西水痘样疹",
                "goal_codes": ["antiviral_therapy"],
                "risk_codes": [],
                "contraindication_codes": [],
                "support_stats": {
                    "support_case_count": 12,
                    "goal_support_counts": {"antiviral_therapy": 11},
                    "risk_support_counts": {},
                    "contraindication_support_counts": {},
                },
                "source_receipt_hash": "sha256:" + "c" * 64,
            }
        ],
        treatment_plan="静脉注射阿昔洛韦抗病毒治疗。",
    )
    goal_entries = [entry for entry in catalog if entry["source"] == "profile_goal"]
    assert goal_entries == []


def test_treatment_review_catalog_without_profiles_is_unchanged() -> None:
    from agent.legacy_orchestrator import build_treatment_review_evidence_catalog

    args = {
        "case_state": {"chat_history": [], "examination_results": {}},
        "diagnosis": "卡波西水痘样疹",
        "diagnosis_axes": [],
        "verifier_issues": [],
    }
    before = build_treatment_review_evidence_catalog(**args)
    after = build_treatment_review_evidence_catalog(
        **args, treatment_profiles=[], treatment_plan=""
    )
    assert before == after


_CONCRETE_DRUGS = (
    "阿昔洛韦",
    "头孢",
    "青霉素",
    "万古霉素",
    "布洛芬",
    "对乙酰氨基酚",
    "泼尼松",
    "华法林",
    "肝素",
)
_DOSE_MARKERS = ("mg", "ml", "每日", "bid", "tid", "q8h", "静脉注射")


def test_goal_and_risk_review_text_never_names_a_drug_or_dose() -> None:
    """A profile may state a goal, never a prescription."""
    from agent.knowledge.verified_profiles import (
        GOAL_CODE_REVIEW_TEXT,
        RISK_CODE_REVIEW_TEXT,
    )

    for book in (GOAL_CODE_REVIEW_TEXT, RISK_CODE_REVIEW_TEXT):
        for code, text in book.items():
            lowered = text.lower()
            for marker in _CONCRETE_DRUGS + _DOSE_MARKERS:
                assert marker.lower() not in lowered, (code, marker, text)


def test_all_profile_evidence_is_review_only() -> None:
    """Every profile-derived entry is a review target, never an applied edit."""
    from agent.knowledge.verified_profiles import verified_treatment_profile_evidence

    def factory(**kwargs: Any) -> Dict[str, Any]:
        return dict(kwargs)

    entries = verified_treatment_profile_evidence(
        [
            {
                "goal_codes": ["antibacterial_therapy", "inpatient_monitoring"],
                "risk_codes": ["sepsis_risk"],
                "contraindication_codes": ["systemic_corticosteroid"],
            }
        ],
        treatment_plan="",
        entry_factory=factory,
    )
    assert entries
    assert all(entry["polarity"] == "missing" for entry in entries)
    assert all(entry["source"].startswith("profile_") for entry in entries)
