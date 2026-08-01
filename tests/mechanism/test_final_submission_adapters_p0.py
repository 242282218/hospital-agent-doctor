"""P0 regression tests for trusted A2 final-submission adapters."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from agent.clinical.final_submission import (
    FinalAuthorizationRegistry,
    FinalPayload,
    LoadedRuntimeIdentity,
)
from agent.clinical import final_submission_adapters as adapters_module


RELEASE_ID = "adapter-revise-once-test"


def _payload() -> FinalPayload:
    return FinalPayload(
        diagnoses=("社区获得性肺炎",),
        treatment_plan="原始治疗方案",
        reasoning="临床依据",
    )


def _adapters() -> adapters_module._CaseAdapters:
    return adapters_module._CaseAdapters({"diagnoses": ["社区获得性肺炎"]})


def _report(*, patched_treatment: str, issues: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "passed": False,
        "patched_treatment": patched_treatment,
        "issues": issues,
    }


@pytest.mark.parametrize(
    ("report", "expected_plan"),
    [
        (_report(patched_treatment="修订后的治疗方案", issues=[{"code": "fixable"}]), "修订后的治疗方案"),
        (_report(patched_treatment="", issues=[{"code": "fixable"}]), "原始治疗方案"),
        (_report(patched_treatment="原始治疗方案", issues=[{"code": "fixable"}]), "原始治疗方案"),
        (
            _report(
                patched_treatment="修订后的治疗方案",
                issues=[{"code": "not_patchable", "patchable": False}],
            ),
            "原始治疗方案",
        ),
        (
            _report(
                patched_treatment="修订后的治疗方案",
                issues=[{"code": "submission_blocked", "blocks_submission": True}],
            ),
            "原始治疗方案",
        ),
    ],
    ids=(
        "applies-safe-different-patch",
        "ignores-empty-patch",
        "ignores-identical-patch",
        "rejects-unpatchable-issue",
        "rejects-submission-blocking-issue",
    ),
)
def test_revise_once_rechecks_closed_verifier_and_applies_only_safe_patch(
    monkeypatch: pytest.MonkeyPatch,
    report: Mapping[str, Any],
    expected_plan: str,
) -> None:
    calls: list[Mapping[str, Any]] = []

    def verifier(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(kwargs)
        return report

    monkeypatch.setattr(adapters_module, "final_verifier", verifier)

    result = _adapters().revise_once(_payload(), ["initial_issue"])

    assert result.treatment_plan == expected_plan
    assert len(calls) == 1
    assert calls[0]["diagnosis"] == "社区获得性肺炎"
    assert calls[0]["treatment_plan"] == "原始治疗方案"


@pytest.mark.parametrize(
    ("payload_diagnoses", "context_diagnoses"),
    [
        (("肺炎",), ()),
        (("肺炎",), ("支气管炎",)),
        (("肺炎", "哮喘"), ("哮喘", "肺炎")),
    ],
    ids=("missing-context", "different-diagnosis", "different-order"),
)
def test_verify_rejects_payload_diagnoses_not_bound_to_context(
    payload_diagnoses: tuple[str, ...],
    context_diagnoses: tuple[str, ...],
) -> None:
    coordinator = adapters_module.build_case_coordinator(
        registry=FinalAuthorizationRegistry(release_identity_hash=RELEASE_ID),
        runtime_identity=LoadedRuntimeIdentity(status="strict_verified", identity_hash=RELEASE_ID),
        clinical_context={"diagnoses": list(context_diagnoses)},
    )
    payload = FinalPayload(
        diagnoses=payload_diagnoses,
        treatment_plan="治疗方案",
        reasoning="临床依据",
    )

    with pytest.raises(ValueError, match="diagnoses do not match clinical context"):
        coordinator.verify(payload=payload, clinical_context={})


def test_mismatched_diagnoses_call_no_legacy_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def unexpected_authority(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        del args, kwargs
        calls.append("authority")
        return {"passed": True}

    monkeypatch.setattr(adapters_module, "apply_treatment_safety", unexpected_authority)
    monkeypatch.setattr(adapters_module, "final_verifier", unexpected_authority)
    monkeypatch.setattr(adapters_module, "converge_verified_treatment", unexpected_authority)
    monkeypatch.setattr(adapters_module, "enforce_five_dimension_gate", unexpected_authority)
    coordinator = adapters_module.build_case_coordinator(
        registry=FinalAuthorizationRegistry(release_identity_hash=RELEASE_ID),
        runtime_identity=LoadedRuntimeIdentity(status="strict_verified", identity_hash=RELEASE_ID),
        clinical_context={"diagnoses": ["肺炎"]},
    )
    payload = FinalPayload(
        diagnoses=("支气管炎",),
        treatment_plan="治疗方案",
        reasoning="临床依据",
    )

    with pytest.raises(ValueError, match="diagnoses do not match clinical context"):
        coordinator.verify(payload=payload, clinical_context={})

    assert calls == []


def test_build_case_coordinator_wires_adapter_revise_once() -> None:
    coordinator = adapters_module.build_case_coordinator(
        registry=FinalAuthorizationRegistry(release_identity_hash=RELEASE_ID),
        runtime_identity=LoadedRuntimeIdentity(status="strict_verified", identity_hash=RELEASE_ID),
        clinical_context={"diagnoses": ["社区获得性肺炎"]},
    )

    assert coordinator._revise_once is not None
    assert coordinator._revise_once.__self__.__class__ is adapters_module._CaseAdapters
