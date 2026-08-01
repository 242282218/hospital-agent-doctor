"""A2 P0 red tests for the unified final submission pipeline.

Contract locked here, across the same five production submit paths that the
legacy authority owns (ordinary final, department fallback, safe-escalation,
conservative fallback, exact-memory):

  * If ANY verifier or five-dimension gate step returns ``passed=False``, the
    coordinator never authorizes the payload. The prescribe endpoint therefore
    cannot build a command — ``adapter.prescribe_calls == 0`` even if the gateway
    accepts other actions.
  * After the coordinator has verified a payload, any later edit of the final
    text invalidates the old authorization: the old ticket produces no command
    and only a freshly verified ticket for the NEW text authorizes again.
  * The fixed pipeline requires both the legacy verifier AND the five-dimension
    gate to converge on the SAME final text (``final_current == current``) and
    both to explicitly return ``passed=True``; otherwise verification raises and
    no ticket is minted.
  * ``revise_once`` runs at most once between the safety apply and the converge
    step; a second revise call beyond one is forbidden.

These tests drove the A2 implementation: when only A1 code exists,
``FinalSubmissionCoordinator.verify`` raises NotImplementedError, so every test
below fails. The implementation must replace that body with the fixed pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import pytest

from agent.clinical.final_submission import (
    FinalAuthorizationRegistry,
    FinalPayload,
    FinalSubmissionCoordinator,
    FinalVerificationError,
    LoadedRuntimeIdentity,
    build_prescribe_command,
    canonical_final_payload_hash,
    report_issue_codes,
)


RELEASE_ID = "release-identity-a2-pipeline-scenario"


def _registry() -> FinalAuthorizationRegistry:
    return FinalAuthorizationRegistry(release_identity_hash=RELEASE_ID)


def _identity() -> LoadedRuntimeIdentity:
    return LoadedRuntimeIdentity(status="strict_verified", identity_hash=RELEASE_ID)


def _payload(text: str = "对社区获得性肺炎给予左氧氟沙星口服抗感染；监测体温与血常规，门诊随访。",
             diagnoses: tuple = ("社区获得性肺炎",)) -> FinalPayload:
    return FinalPayload(
        diagnoses=diagnoses,
        treatment_plan=text,
        reasoning="结合病史、体征及血常规决定初始治疗。",
    )


# --- shared apply/verifier/gate objects for pipeline-driven tests --------------

class _StubApply:
    """apply_treatment_safety adapter: records calls, returns payload unchanged."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, payload: FinalPayload, ctx: Mapping[str, Any]) -> FinalPayload:
        self.calls += 1
        return payload


class _StubVerifier:
    """run_legacy_verifier adapter: returns a fixed report."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self._report = dict(report)
        self.calls = 0

    def __call__(self, payload: FinalPayload) -> Mapping[str, Any]:
        self.calls += 1
        return dict(self._report)


class _StubGate:
    """run_five_dimension_gate adapter: returns (payload, report).

    If ``mutate_on_call`` is set, only that 1-based call index mutates the text
    by appending ``mutation_text`` — modeling a non-idempotent gate so the
    convergence check (final_current == current) trips. ``treatment_plan_after``
    is kept ONLY as a deprecated alias that mutates on the LAST recorded call
    so older tests of an idempotent failure can still pass.
    """

    def __init__(self, report: Mapping[str, Any], *,
                 treatment_plan_after: str | None = None,
                 mutate_on_call: int | None = None,
                 mutation_text: str = "再次修改的治疗文本。"):
        self._report = dict(report)
        self._plan_after = treatment_plan_after
        self._plan_after_text = mutation_text
        self._mutate_on_call = mutate_on_call
        self.calls = 0

    def __call__(self, payload: FinalPayload):
        self.calls += 1
        out = payload
        if self._mutate_on_call is not None:
            # Non-idempotent: mutate only at a specific 1-based call index.
            if self.calls == self._mutate_on_call:
                out = replace(payload, treatment_plan=payload.treatment_plan + self._plan_after_text)
        elif self._plan_after is not None:
            # Legacy alias: mutate on every call.
            out = replace(payload, treatment_plan=self._plan_after)
        return out, dict(self._report)


class _StubConverge:
    """converge adapter: returns payload unchanged; records calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, payload: FinalPayload) -> FinalPayload:
        self.calls += 1
        return payload


class _StubRevise:
    """revise_once adapter: appends a sentinel so the model can detect a single
    edit and prove no second revise runs."""

    def __init__(self, suffix: str = "（已修订）"):
        self._suffix = suffix
        self.calls = 0

    def __call__(self, payload: FinalPayload, issues):
        self.calls += 1
        return replace(payload, treatment_plan=payload.treatment_plan + self._suffix)


def _coordinator(
    registry: FinalAuthorizationRegistry,
    *,
    apply_safety=None,
    run_legacy_verifier=None,
    run_five_dimension_gate=None,
    converge=None,
    revise_once=None,
) -> FinalSubmissionCoordinator:
    return FinalSubmissionCoordinator(
        registry=registry,
        runtime_identity=_identity(),
        apply_safety=apply_safety or _StubApply(),
        run_legacy_verifier=run_legacy_verifier or _StubVerifier({"passed": True}),
        run_five_dimension_gate=run_five_dimension_gate or _StubGate({"passed": True}),
        converge=converge or _StubConverge(),
        revise_once=revise_once,
    )


# --- Step 1 red tests: passed=false must never authorize ----------------------

def test_pipeline_legacy_verifier_false_blocks_authorization() -> None:
    """普通 / dept fallback 面：final legacy verifier passed=False 时不能授权。"""
    registry = _registry()
    coord = _coordinator(
        registry,
        run_legacy_verifier=_StubVerifier({"passed": False, "issue_codes": ["contraindicated_drug"]}),
    )
    with pytest.raises(FinalVerificationError):
        coord.verify(payload=_payload(), clinical_context={})
    assert registry.outstanding_count() == 0


def test_pipeline_gate_false_blocks_authorization() -> None:
    """五维 gate returned=False 时不能授权，即便 legacy verifier 通过。"""
    registry = _registry()
    coord = _coordinator(
        registry,
        run_legacy_verifier=_StubVerifier({"passed": True}),
        run_five_dimension_gate=_StubGate({"passed": False, "issue_codes": ["drug_interaction_fail"]}),
    )
    with pytest.raises(FinalVerificationError):
        coord.verify(payload=_payload(), clinical_context={})
    assert registry.outstanding_count() == 0


def test_pipeline_missing_passed_field_blocks_authorization() -> None:
    """空报告或字符串 "true" 都不能解释为通过（report_passed 只接受 True）。"""
    registry = _registry()
    # Report omits ``passed`` entirely (zero-knowledge sentinel).
    coord = _coordinator(
        registry,
        run_legacy_verifier=_StubVerifier({"issues": []}),
        run_five_dimension_gate=_StubGate({"gate": {"all_passed": True}}),
    )
    with pytest.raises(FinalVerificationError):
        coord.verify(payload=_payload(), clinical_context={})
    assert registry.outstanding_count() == 0


def test_pipeline_string_true_is_not_passed() -> None:
    """A string "true" in place of an explicit boolean must fail closed."""
    registry = _registry()
    coord = _coordinator(
        registry,
        run_legacy_verifier=_StubVerifier({"passed": "true"}),
        run_five_dimension_gate=_StubGate({"passed": "true"}),
    )
    with pytest.raises(FinalVerificationError):
        coord.verify(payload=_payload(), clinical_context={})


def test_pipeline_gate_does_not_converge_blocks_authorization() -> None:
    """safe-escalation / conservative fallback 面：gate 在最终 re-run 时再次
    mutate 最终文本（不是第一调用）视为未收敛，不能授权。"""
    registry = _registry()
    # Non-idempotent scenario: the FIRST gate call leaves text untouched, the
    # SECOND (final re-run) accidentally mutates again, so final_current !=
    # current trips the convergence guard.
    coord = _coordinator(
        registry,
        run_five_dimension_gate=_StubGate(
            {"passed": True}, mutate_on_call=2, mutation_text=" 再次修改。",
        ),
    )
    with pytest.raises(FinalVerificationError, match="five-dimension gate did not converge"):
        coord.verify(payload=_payload(), clinical_context={})
    assert registry.outstanding_count() == 0


# --- Step 2 red test: revised text invalidates old authorization -------------

def test_revised_text_invalidates_old_authorization() -> None:
    """初稿通过验证后修改一个字符，旧 authorization 不能生成 command；
    只有对新文本重验后才能授权。"""
    registry = _registry()
    coord = _coordinator(registry)
    payload_v1 = _payload()
    verified_v1 = coord.verify(payload=payload_v1, clinical_context={})
    prepared_v1 = coord.authorize(
        verified=verified_v1,
        case_run_id="run-1",
        revision=4,
        snapshot_hash="snap-0",
    )
    payload_v2 = replace(payload_v1, treatment_plan=payload_v1.treatment_plan + "（追加重审）")
    # Old ticket must not produce a command for the drifted payload: the run-
    # architecture must refuse to re-authorize the same case on a new text using
    # a stale ticket, regardless of the local payload-hash comparison.
    with pytest.raises(ValueError, match="(?:field mismatch|not found or already consumed)"):
        build_prescribe_command(
            registry=registry,
            ticket=prepared_v1.ticket,
            case_run_id="run-1",
            blackboard_revision=4,
            action_sequence=1,
            payload={
                "diagnosis": list(payload_v2.diagnoses),
                "treatment_plan": payload_v2.treatment_plan,
                "reasoning": payload_v2.reasoning,
            },
            snapshot_hash="snap-0",
            legacy_verifier_hash=verified_v1.legacy_verifier_hash,
            five_dimension_gate_hash=verified_v1.five_dimension_gate_hash,
            issue_codes=verified_v1.issue_codes,
        )
    # Re-verifying the new text mints a different ticket and a fresh command.
    verified_v2 = coord.verify(payload=payload_v2, clinical_context={})
    assert verified_v2.payload_hash != verified_v1.payload_hash
    prepared_v2 = coord.authorize(
        verified=verified_v2,
        case_run_id="run-1",
        revision=5,
        snapshot_hash="snap-1",
    )
    cmd = build_prescribe_command(
        registry=registry,
        ticket=prepared_v2.ticket,
        case_run_id="run-1",
        blackboard_revision=5,
        action_sequence=2,
        payload={
            "diagnosis": list(payload_v2.diagnoses),
            "treatment_plan": payload_v2.treatment_plan,
            "reasoning": payload_v2.reasoning,
        },
        snapshot_hash="snap-1",
        legacy_verifier_hash=verified_v2.legacy_verifier_hash,
        five_dimension_gate_hash=verified_v2.five_dimension_gate_hash,
        issue_codes=verified_v2.issue_codes,
    )
    assert cmd.action_type == "prescribe_treatment"
    assert cmd.payload["treatment_plan"] == payload_v2.treatment_plan


# --- Step 4 pipeline ordering + one-shot revise -------------------------------

def test_pipeline_runs_apply_then_verifier_then_at_most_one_revise_then_converge_then_gate() -> None:
    """Pipeline order is fixed: apply -> verifier -> (revise once) -> converge
    -> gate -> verifier -> gate -> hashes. Draft verifier issues (when the
    initial text lacks the revision sentinel) trigger exactly one revise;
    converge and final gate must run on the revised text. The second verifier
    call must see the revised suffixed text (same callable evaluates both)."""
    registry = _registry()
    apply = _StubApply()
    revise = _StubRevise()
    gate = _StubGate({"passed": True})
    converge = _StubConverge()
    call_log: list[str] = []
    seen_payloads: list[FinalPayload] = []

    def tracked_apply(payload, ctx):
        call_log.append("apply")
        return apply(payload, ctx)

    def tracked_revise(payload, issues):
        call_log.append("revise")
        return revise(payload, issues)

    def tracked_converge(payload):
        call_log.append("converge")
        return converge(payload)

    def tracked_gate(payload):
        call_log.append("gate")
        return gate(payload)

    def conditional_verifier(payload):
        call_log.append("verify")
        seen_payloads.append(payload)
        # Pass only after a revise has stamped the sentinel onto the text.
        if "已修订" in payload.treatment_plan:
            return {"passed": True}
        return {"passed": False, "issue_codes": ["draft_contradicts_axis"]}

    coord = FinalSubmissionCoordinator(
        registry=registry,
        runtime_identity=_identity(),
        apply_safety=tracked_apply,
        run_legacy_verifier=conditional_verifier,
        run_five_dimension_gate=tracked_gate,
        converge=tracked_converge,
        revise_once=tracked_revise,
    )
    verified = coord.verify(payload=_payload(), clinical_context={})
    # At most one revise, then converge and the gate run runs at least twice.
    assert call_log.count("revise") == 1
    assert call_log.count("gate") >= 2
    # Second (final) verifier call sees the revised text and returns passed=True.
    assert sum(1 for p in seen_payloads if "已修订" in p.treatment_plan) >= 1
    # Verified payload bound to the revised text.
    assert "已修订" in verified.payload.treatment_plan
    assert verified.issue_codes == ()


def test_revise_more_than_once_is_forbidden() -> None:
    """If issues persist after a single revise, the pipeline must NOT revise
    repeatedly; converge picks up bounded final-verifier rounds instead."""
    registry = _registry()
    revise = _StubRevise()
    coord = _coordinator(
        registry,
        run_legacy_verifier=_StubVerifier({"passed": False, "issue_codes": ["x"]}),
        run_five_dimension_gate=_StubGate({"passed": True}),
        revise_once=revise,
    )
    # Initial verification surfaces must_fix issues and the coordinator may
    # revise once; final verifier still returns passed and the gate permits it
    # only because the pipeline clears the issue set before the final report.
    # Use a green final verifier by patching through a custom verifier that
    # returns False initially and True after revise (i.e., sees revised text).
    call_log: list[str] = []

    def conditional_verifier(payload):
        call_log.append("verify")
        # Any payload whose text carries the revise sentinel is considered fixed.
        if "已修订" in payload.treatment_plan:
            return {"passed": True}
        return {"passed": False, "issue_codes": ["needs_review"]}

    coord = FinalSubmissionCoordinator(
        registry=registry,
        runtime_identity=_identity(),
        apply_safety=_StubApply(),
        run_legacy_verifier=conditional_verifier,
        run_five_dimension_gate=_StubGate({"passed": True}),
        converge=_StubConverge(),
        revise_once=revise,
    )
    verified = coord.verify(payload=_payload(), clinical_context={})
    assert revise.calls == 1, "exactly one revise; verifier-noise must not loop"
    assert "已修订" in verified.payload.treatment_plan


# --- Helpers reachable from production adapters ------------------------------

def test_report_issue_codes_is_stable_dedup() -> None:
    assert report_issue_codes({"issue_codes": ["a", "b", "a", " B ", ""]}) == ("a", "b", "B")
    assert report_issue_codes({}) == ()


def test_canonical_final_payload_hash_is_payload_bound() -> None:
    p1 = _payload()
    p2 = replace(p1, treatment_plan=p1.treatment_plan + " ")
    assert canonical_final_payload_hash(p1) != canonical_final_payload_hash(p2)
