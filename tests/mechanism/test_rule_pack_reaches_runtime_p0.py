"""P0 guardrails: typed rule packs must actually reach the live runtime loader."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from agent.agent import load_release_if_present
from agent.runtime.release_loader import load_current_release
from offline.artifacts import file_hash, write_immutable_json
from offline.release import build_candidate_pack, verify_release_pack


ROOT = Path(__file__).resolve().parents[2]
LIVE_POINTER = ROOT / "releases" / "current.json"
LIVE_RULES = ROOT / "agent" / "knowledge" / "clinical_pattern_rules.json"


def _path_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _minimal_rules_payload() -> dict:
    # Full live pack already carries rule_count/rules_hash; keep it intact.
    return json.loads(LIVE_RULES.read_text(encoding="utf-8"))


def test_live_release_pack_is_not_empty() -> None:
    """Prevent break #1 recurrence: production pointer must load a non-empty pack."""
    release = load_release_if_present(LIVE_POINTER)
    assert release is not None, "releases/current.json must exist"
    pack = release.knowledge_rule_pack
    assert pack.rule_count > 0, "live pack still empty"
    assert all(rule.runtime.status == "active" for rule in pack.rules)


def test_manifest_declares_knowledge_rules_hash() -> None:
    pointer = json.loads(LIVE_POINTER.read_text(encoding="utf-8"))
    release_dir = ROOT / str(pointer["release_dir"])
    manifest = json.loads(
        (release_dir / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert "knowledge_rules_hash" in manifest
    assert isinstance(manifest["knowledge_rules_hash"], str)
    assert len(manifest["knowledge_rules_hash"]) == 64


def test_rules_file_hash_matches_manifest(tmp_path: Path) -> None:
    release_dir = tmp_path / "release_with_rules"
    rules = _minimal_rules_payload()
    manifest = build_candidate_pack(
        release_dir=release_dir,
        code_commit="deadbeef",
        prompt_pack={"system": "doctor"},
        policy_pack={"simple_cap": 5},
        registry={"schema_version": "verified-registry/v1", "assets": []},
        knowledge_hashes={"alias_map.json": "a" * 64},
        catalog_hashes={"diseases_catalog.json": "b" * 64},
        knowledge_rule_pack=rules,
    )
    assert manifest["knowledge_rules_hash"] == file_hash(
        release_dir / "knowledge_rules.json"
    )
    verify_release_pack(release_dir)

    pointer_path = tmp_path / "current.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": str(release_dir),
                "pack_hash": _path_hash(release_dir / "release_manifest.json"),
                "runtime_schema_version": "clinical-runtime/v1",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_current_release(pointer_path)
    assert loaded.knowledge_rule_pack.rule_count == len(rules["rules"])
    assert loaded.knowledge_rule_pack.rule_count > 0

    # Tamper after publish: loader must fail closed on hash mismatch.
    rules_path = release_dir / "knowledge_rules.json"
    tampered = json.loads(rules_path.read_text(encoding="utf-8"))
    tampered["rules"][0]["priority"] = 99
    rules_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="knowledge rules hash mismatch"):
        load_current_release(pointer_path)


def test_build_candidate_pack_rejects_invalid_rule_pack(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_candidate_pack(
            release_dir=tmp_path / "bad_rules_release",
            code_commit="deadbeef",
            prompt_pack={},
            policy_pack={},
            registry={"schema_version": "verified-registry/v1", "assets": []},
            knowledge_hashes={},
            catalog_hashes={},
            knowledge_rule_pack={"schema_version": "nope", "rules": []},
        )
    assert not (tmp_path / "bad_rules_release" / "knowledge_rules.json").exists()


def test_build_candidate_pack_without_rules_stays_compatible(tmp_path: Path) -> None:
    manifest = build_candidate_pack(
        release_dir=tmp_path / "legacy_style",
        code_commit="deadbeef",
        prompt_pack={},
        policy_pack={},
        registry={"schema_version": "verified-registry/v1", "assets": []},
        knowledge_hashes={},
        catalog_hashes={},
    )
    assert "knowledge_rules_hash" not in manifest
    assert not (tmp_path / "legacy_style" / "knowledge_rules.json").exists()


def test_fact_group_effect_reaches_exam_intents() -> None:
    from agent.knowledge.typed_rule_engine import RuleContext, apply_rules, parse_compiled_rule_pack

    pack = parse_compiled_rule_pack(json.loads(LIVE_RULES.read_text(encoding="utf-8")))
    result = apply_rules(
        pack,
        "clinical_closure",
        RuleContext(fact_codes=("limb_swelling", "skin_redness_heat", "fever")),
    )
    assert "acute_limb_soft_tissue_infection" in result.output_context.fact_codes
    assert "exam_intent_lower_extremity_soft_tissue_severity" in (
        result.output_context.exam_intent_ids
    )


def test_excluded_groups_actually_fire_from_extractor() -> None:
    from agent.knowledge.typed_rule_engine import RuleContext, apply_rules, parse_compiled_rule_pack
    from agent.legacy_orchestrator import diagnosis_rule_fact_codes

    case_state = {
        "chat_history": [
            {
                "from": "patient",
                "text": "42岁上眼睑黄色斑块，总胆固醇268，LDL172，但为非感染性湿疹",
            }
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }
    facts = diagnosis_rule_fact_codes(case_state)
    assert "drug_allergy" not in facts
    assert "noninfectious_eczema" in facts
    pack = parse_compiled_rule_pack(json.loads(LIVE_RULES.read_text(encoding="utf-8")))
    result = apply_rules(pack, "clinical_closure", RuleContext(fact_codes=facts))
    decisions = {d.rule_id: d for d in result.decisions}
    assert decisions["hyperlipidemia_with_xanthelasma_pattern"].outcome == "excluded"

    allergy_state = {
        "chat_history": [{"from": "patient", "text": "青霉素药物过敏，偶发皮疹"}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }
    allergy_facts = diagnosis_rule_fact_codes(allergy_state)
    assert "drug_allergy" in allergy_facts


def test_all_declared_stages_are_executed_at_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every stage present on the live pack must be invoked by the diagnosis adapter."""
    from agent.knowledge import typed_rule_engine
    from agent.legacy_orchestrator import apply_diagnosis_candidate_rules

    release = load_release_if_present(LIVE_POINTER)
    assert release is not None
    pack = release.knowledge_rule_pack
    declared_stages = {
        rule.runtime.stage
        for rule in pack.rules
        if rule.runtime.status == "active" and rule.runtime.stage
    }
    assert declared_stages, "live pack must declare at least one active stage"

    seen_stages: list[str] = []
    real_apply = typed_rule_engine.apply_rules

    def tracking_apply(rule_pack, stage, context):
        seen_stages.append(stage)
        return real_apply(rule_pack, stage, context)

    monkeypatch.setattr(typed_rule_engine, "apply_rules", tracking_apply)
    # apply_diagnosis_candidate_rules binds apply_rules at import time on some paths;
    # patch the name used by the adapter module as well.
    import agent.legacy_orchestrator as legacy

    monkeypatch.setattr(legacy, "apply_rules", tracking_apply)

    apply_diagnosis_candidate_rules(
        [{"disease": "原发性高血压", "score": 10, "source": "unit"}],
        case_state={
            "chat_history": [{"from": "patient", "text": "偶发头痛"}],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {},
            "exam_decision_trace": [],
            "diagnosis_axes": [],
        },
        official_diseases=["原发性高血压"],
        rule_pack=pack,
    )
    for stage in declared_stages:
        assert stage in seen_stages, "stage never executed at runtime: %s" % stage
