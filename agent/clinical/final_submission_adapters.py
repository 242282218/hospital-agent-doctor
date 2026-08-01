"""A2 adapters: bridge the legacy clinical authorities into the trusted
FinalSubmissionCoordinator pipeline.

The coordinator's pipeline signature is fixed (plan §A2 §4.1):
  * ``apply_safety(payload, clinical_context) -> FinalPayload``
  * ``run_legacy_verifier(payload) -> Mapping``
  * ``run_five_dimension_gate(payload) -> (FinalPayload, Mapping)``
  * ``converge(payload) -> FinalPayload``

The legacy authorities (`legacy_orchestrator.final_verifier`,
`enforce_five_dimension_gate`, `converge_verified_treatment`, the case-engine
`apply_treatment_safety`) require per-diagnosis, per-case state that is only
available at the prescribe call site (case_features, examinations,
safety_profiles, official_diseases, examination_catalog, exam_plan_trace). The
coordinator's ``verify()`` only re-receives that context through
``apply_safety``; the other adapters cannot. We therefore build the per-case
adapter package at the call site where ``clinical_context`` is in scope, close
the authoritative context into each adapter, and hand the package to the
coordinator constructor.

Security note: the business side (legacy_orchestrator / online_runtime) cannot
inject ``apply_safety`` or ``run_legacy_verifier`` per call. The adapters are
constructed from trusted legacy callables only, so the coordinator still
fires-closed when their report contains no explicit boolean ``passed=True``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from agent.clinical.final_submission import (
    FinalAuthorizationRegistry,
    FinalPayload,
    FinalSubmissionCoordinator,
    LoadedRuntimeIdentity,
    report_issue_codes,
    report_passed,
)
from agent.legacy_orchestrator import (
    apply_treatment_safety,
    converge_verified_treatment,
    enforce_five_dimension_gate,
    final_verifier,
)
from agent.observability.runtime_events import canonical_hash


# Clinical context keys anchored at each prescribe call site.
CONTEXT_KEYS: Tuple[str, ...] = (
    "diagnoses",
    "examinations",
    "official_diseases",
    "examination_catalog",
    "exam_plan_trace",
    "case_features",
    "safety_profiles",
    "clinical_basis",
    "safety_facts",
)


def make_clinical_context(**values: Any) -> Dict[str, Any]:
    """Build the per-call-site ``clinical_context`` dict.

    The dict is closed over by the per-case adapter package so the trusted
    pipeline can re-run legacy verifier/gate/converge against the SAME case
    evidence used to draft the prescription.
    """
    ctx: Dict[str, Any] = {}
    for key in CONTEXT_KEYS:
        if key in values:
            ctx[key] = values[key]
    if "diagnosis" in values and "diagnoses" not in ctx:
        ctx["diagnosis"] = values["diagnosis"]
    return ctx


def _diagnoses_from_context(ctx: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = ctx.get("diagnoses")
    if raw is None:
        single = ctx.get("diagnosis")
        if isinstance(single, str) and single.strip():
            return (single,)
        return ()
    items: List[str] = []
    if isinstance(raw, str):
        items = [raw.strip()] if raw.strip() else []
    else:
        items = [str(x).strip() for x in raw if str(x).strip()]
    return tuple(items)


def _examinations_from_context(ctx: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = ctx.get("examinations") or ()
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    return tuple(str(x).strip() for x in list(raw) if str(x).strip())


class _CaseAdapters:
    """Per-case trusted adapters that close over the clinical context."""

    def __init__(self, ctx: Mapping[str, Any]) -> None:
        self._ctx = deepcopy(dict(ctx))
        self._diagnoses = _diagnoses_from_context(self._ctx)
        self._examinations = _examinations_from_context(self._ctx)
        self._official_diseases = list(self._ctx.get("official_diseases") or ())
        self._catalog = dict(self._ctx.get("examination_catalog") or {})
        self._trace = list(self._ctx.get("exam_plan_trace") or [])
        self._features = deepcopy(dict(self._ctx.get("case_features") or {}))
        self._features["safety_facts"] = deepcopy(self._ctx.get("safety_facts") or [])
        self._profiles = list(self._ctx.get("safety_profiles") or [])
        self._basis = list(self._ctx.get("clinical_basis") or [])

    def _require_payload_diagnoses(self, payload: FinalPayload) -> Tuple[str, ...]:
        diagnoses = tuple(str(item).strip() for item in payload.diagnoses if str(item).strip())
        if not diagnoses:
            raise ValueError("final payload diagnoses must not be empty")
        if diagnoses != self._diagnoses:
            raise ValueError("final payload diagnoses do not match clinical context")
        return diagnoses

    # --- apply_safety: merges legacy apply_treatment_safety patches --------

    def apply_safety(self, payload: FinalPayload, ctx: Mapping[str, Any]) -> FinalPayload:
        del ctx
        diagnoses = self._require_payload_diagnoses(payload)
        plan = payload.treatment_plan
        for diagnosis in diagnoses:
            if not diagnosis:
                continue
            result = apply_treatment_safety(
                plan,
                diagnosis=diagnosis,
                case_features=self._features,
                safety_profiles=self._profiles,
            )
            patched = str(result.get("treatment_plan") or "").strip()
            if patched and patched != plan:
                plan = patched
        if plan == payload.treatment_plan:
            return payload
        return FinalPayload(
            diagnoses=payload.diagnoses,
            treatment_plan=plan,
            reasoning=payload.reasoning,
        )

    # --- run_legacy_verifier: all diagnoses must pass -----------------

    def run_legacy_verifier(self, payload: FinalPayload) -> Mapping[str, Any]:
        diagnoses = self._require_payload_diagnoses(payload)
        all_issues: List[Dict[str, Any]] = []
        any_failed = False
        patched_text = payload.treatment_plan
        for diagnosis in diagnoses:
            if not diagnosis:
                # Empty diagnosis is itself a blocking must_fix at the verifier
                any_failed = True
                all_issues.append(
                    {
                        "field": "diagnosis",
                        "code": "empty_diagnosis",
                        "severity": "must_fix",
                        "patchable": False,
                    }
                )
                continue
            report = final_verifier(
                diagnosis=diagnosis,
                examinations=list(self._examinations),
                treatment_plan=patched_text,
                official_diseases=self._official_diseases,
                examination_catalog=self._catalog,
                exam_plan_trace=self._trace,
                case_features=self._features,
                safety_profiles=self._profiles,
            )
            for issue in report.get("issues") or []:
                if isinstance(issue, dict):
                    all_issues.append(issue)
            if not report_passed(report):
                any_failed = True
            maybe_patched = str(report.get("patched_treatment") or "").strip()
            if maybe_patched and maybe_patched != patched_text:
                patched_text = maybe_patched
        overall_passed = (not any_failed) and bool(diagnoses)
        issue_codes = tuple(report_issue_codes({"issue_codes": _collect_issue_codes(all_issues)}))
        return {
            "passed": overall_passed,
            "issues": all_issues,
            "issue_codes": list(issue_codes),
            "patched_treatment": patched_text,
            "treatment_hash": canonical_hash(patched_text),
        }

    def revise_once(self, payload: FinalPayload, issues: Sequence[str]) -> FinalPayload:
        """Apply one trusted legacy-verifier patch when every issue permits it."""
        del issues
        report = self.run_legacy_verifier(payload)
        report_issues = report.get("issues") or []
        for issue in report_issues:
            if not isinstance(issue, Mapping):
                continue
            if issue.get("patchable") is False or issue.get("blocks_submission") is True:
                return payload
        patched = str(report.get("patched_treatment") or "").strip()
        if not patched or patched == payload.treatment_plan:
            return payload
        return FinalPayload(
            diagnoses=payload.diagnoses,
            treatment_plan=patched,
            reasoning=payload.reasoning,
        )

    # --- run_five_dimension_gate: returns sanitized FinalPayload + report --

    def run_five_dimension_gate(self, payload: FinalPayload) -> Tuple[FinalPayload, Mapping[str, Any]]:
        diagnoses = self._require_payload_diagnoses(payload)
        result = enforce_five_dimension_gate(
            diagnoses=list(diagnoses),
            treatment_plan=payload.treatment_plan,
            clinical_basis=list(self._basis),
            case_features=self._features,
            examinations=list(self._examinations),
        )
        gate = dict(result.get("gate") or {})
        sanitized = str(result.get("treatment_plan") or payload.treatment_plan)
        # A ``pass`` requires the aggregated gate to be unblocked AND all_passed.
        passed = bool(gate.get("all_passed") is True and gate.get("blocked") is False)
        issue_codes: List[str] = []
        for finding in (gate.get("blocking_findings") or gate.get("review_findings") or []):
            if isinstance(finding, Mapping):
                code = str(finding.get("dimension") or "")
                if code:
                    issue_codes.append("five_dim_" + code)
        out_payload = payload
        if sanitized != payload.treatment_plan:
            out_payload = FinalPayload(
                diagnoses=payload.diagnoses,
                treatment_plan=sanitized,
                reasoning=payload.reasoning,
            )
        report = {
            "passed": passed,
            "issue_codes": list(issue_codes),
            "blocked": bool(gate.get("blocked")),
            "status": str(gate.get("status") or "review"),
            "all_passed": bool(gate.get("all_passed")),
        }
        return out_payload, report

    # --- converge: bounded legacy converge_verified_treatment per case ---

    def converge(self, payload: FinalPayload) -> FinalPayload:
        diagnoses = self._require_payload_diagnoses(payload)
        # converge_verified_treatment is per-diagnosis; take the converged text
        # of the first diagnosis as the canonical final (matches upstream order).
        converged_text = payload.treatment_plan
        for diagnosis in diagnoses:
            if not diagnosis:
                continue
            report = converge_verified_treatment(
                diagnosis=diagnosis,
                examinations=list(self._examinations),
                treatment_plan=converged_text,
                official_diseases=self._official_diseases,
                examination_catalog=self._catalog,
                exam_plan_trace=self._trace,
                case_features=self._features,
                safety_profiles=self._profiles,
            )
            if report is None:
                continue
            maybe = str(report.get("patched_treatment") or "").strip()
            if maybe:
                converged_text = maybe
        if converged_text == payload.treatment_plan:
            return payload
        return FinalPayload(
            diagnoses=payload.diagnoses,
            treatment_plan=converged_text,
            reasoning=payload.reasoning,
        )


def _collect_issue_codes(issues: Iterable[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("code") or "").strip()
        if code and code not in out:
            out.append(code)
    return out


def build_case_coordinator(
    *,
    registry: FinalAuthorizationRegistry,
    runtime_identity: LoadedRuntimeIdentity,
    clinical_context: Mapping[str, Any],
) -> FinalSubmissionCoordinator:
    """Construct the per-case trusted coordinator bound to the case adapters.

    The adapters close over the clinical context captured at the prescribe call
    site. The A2 fixed pipeline is apply -> legacy verifier -> one-shot revise
    -> converge -> gate -> re-run verifier -> re-run gate.
    """
    adapters = _CaseAdapters(clinical_context)
    return FinalSubmissionCoordinator(
        registry=registry,
        runtime_identity=runtime_identity,
        apply_safety=adapters.apply_safety,
        run_legacy_verifier=adapters.run_legacy_verifier,
        run_five_dimension_gate=adapters.run_five_dimension_gate,
        converge=adapters.converge,
        revise_once=adapters.revise_once,
    )


# --- Identity fabrication (release instrumentation) -------------------------

def build_asset_identity_hash(*, pack_hash: str, manifest_hash: str, asset_hashes: Mapping[str, str]) -> str:
    """Compute a release-asset identity hash for A2.

    The identity binds the loaded pointer pack_hash, the manifest content hash
    and all signed asset file hashes (prompt_pack, policy_pack, registry,
    knowledge_rules, catalog and knowledge hashes declared in the manifest). It
    never includes runtime code hash — that lands in A6's strict_verified
    upgrade. Returns the same digest for the same inputs regardless of order.
    """
    parts = {
        "pack_hash": str(pack_hash or ""),
        "manifest_hash": str(manifest_hash or ""),
        "asset_hashes": {str(k): str(v) for k, v in sorted((asset_hashes or {}).items())},
    }
    return canonical_hash(parts)


__all__ = [
    "CONTEXT_KEYS",
    "build_asset_identity_hash",
    "build_case_coordinator",
    "make_clinical_context",
]
