"""P0: contract-bound examination evidence for the A4 cross-system review."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import agent.clinical.exam_axis_evidence_contract as contract_module
from agent.clinical.exam_axis_evidence_contract import (
    CONTRACT_VERSION,
    ExamAxisEvidenceContractUnavailable,
    has_specific_cross_system_conflict,
    parse_exam_axis_evidence_contract,
)


ASSET_PATH = (
    Path(__file__).resolve().parents[2]
    / "agent"
    / "knowledge"
    / "exam_axis_evidence_contract.json"
)
FINDING_CODE = "controlled_respiratory_finding_against_ocular_axis"
AXIS_ID = "pediatric_congenital_glaucoma"


def _asset() -> dict[str, Any]:
    return json.loads(ASSET_PATH.read_text(encoding="utf-8"))


def _finding(**changes: Any) -> dict[str, str]:
    finding = {
        "schema_version": CONTRACT_VERSION,
        "finding_code": FINDING_CODE,
        "polarity": "present",
        "target_system_id": "respiratory",
        "source_evidence_id": "sdk:exam:controlled:001",
    }
    finding.update(changes)
    return finding


def _results(*findings: Any, status: str = "abnormal") -> dict[str, Any]:
    return {
        "受控检查": {
            "status": status,
            "result": {"opaque": True},
            "structured_findings": list(findings),
        }
    }


def test_versioned_asset_is_a_valid_closed_contract() -> None:
    contract = parse_exam_axis_evidence_contract(_asset())
    assert contract is not None
    finding = contract.finding(FINDING_CODE)
    assert finding is not None
    assert finding.target_system_id == "respiratory"
    assert finding.contradictions[0].axis_id == AXIS_ID


def test_asset_rejects_unknown_top_level_field() -> None:
    raw = _asset()
    raw["unexpected"] = True
    assert parse_exam_axis_evidence_contract(raw) is None


def test_asset_rejects_same_system_axis_mapping() -> None:
    raw = _asset()
    raw["findings"][0]["contradictions"][0]["axis_system_id"] = "respiratory"
    assert parse_exam_axis_evidence_contract(raw) is None


def test_contract_fires_only_for_active_cross_system_axis() -> None:
    assert has_specific_cross_system_conflict(
        _results(_finding()),
        [{"axis_id": AXIS_ID}],
    )


def test_contract_rejects_legacy_opaque_result_and_forged_fields() -> None:
    results = {
        "受控检查": {
            "status": "abnormal",
            "result": {
                "cross_organ_conflict": True,
                "conflicting_axis_id": AXIS_ID,
                "structured_findings": [_finding()],
            },
        }
    }
    assert not has_specific_cross_system_conflict(results, [{"axis_id": AXIS_ID}])


def test_contract_rejects_unknown_finding_version_code_and_extra_field() -> None:
    unknown_version = _finding(schema_version="exam-axis-evidence-contract/v2")
    unknown_code = _finding(finding_code="unknown_finding")
    extra_field = _finding()
    extra_field["unexpected"] = "value"
    for finding in (unknown_version, unknown_code, extra_field):
        assert not has_specific_cross_system_conflict(
            _results(finding), [{"axis_id": AXIS_ID}]
        )


def test_contract_rejects_missing_source_same_system_and_non_active_axis() -> None:
    no_source = _finding(source_evidence_id="")
    same_system_asset = _asset()
    same_system_asset["findings"][0]["contradictions"][0]["axis_system_id"] = "respiratory"
    same_system_contract = parse_exam_axis_evidence_contract(same_system_asset)
    assert same_system_contract is None
    assert not has_specific_cross_system_conflict(
        _results(no_source), [{"axis_id": AXIS_ID}]
    )
    assert not has_specific_cross_system_conflict(
        _results(_finding()), [{"axis_id": "inactive_axis"}]
    )


def test_contract_rejects_invalid_status_and_unknown_row_fields() -> None:
    assert not has_specific_cross_system_conflict(
        _results(_finding(), status="invalid"), [{"axis_id": AXIS_ID}]
    )
    results = _results(_finding())
    row = deepcopy(results["受控检查"])
    row["unexpected"] = True
    results["受控检查"] = row
    assert has_specific_cross_system_conflict(results, [{"axis_id": AXIS_ID}])


def test_contract_load_failure_does_not_silently_clear_cross_system_review(
    monkeypatch,
) -> None:
    contract_module.load_exam_axis_evidence_contract.cache_clear()
    monkeypatch.setattr(
        contract_module,
        "load_exam_axis_evidence_contract",
        lambda: (_ for _ in ()).throw(
            ExamAxisEvidenceContractUnavailable("contract unavailable")
        ),
    )
    with pytest.raises(ExamAxisEvidenceContractUnavailable, match="contract unavailable"):
        has_specific_cross_system_conflict(
            _results(_finding()),
            [{"axis_id": AXIS_ID}],
        )
