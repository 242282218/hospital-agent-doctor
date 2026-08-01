from __future__ import annotations

from agent.clinical.model import EvidenceItem, SkillOperation, SkillProposal
from agent.clinical.orchestrator import ClinicalOrchestrator
from agent.validators.final_verifier import FinalVerifier, TreatmentDraftSanitizer


def test_sanitizer_removes_unsafe_phrases_before_final_verification() -> None:
    orchestrator = ClinicalOrchestrator()
    evidence = EvidenceItem(
        evidence_id="ev-1",
        concept="symptom",
        value="发热",
        kind="patient_statement",
        source_ref="message://patient/1",
    )
    orchestrator.apply_proposal(
        SkillProposal(
            proposal_id="evidence",
            skill_name="IntakeExtractor",
            input_revision=0,
            purpose="test",
            operations=(SkillOperation("add_evidence", {"item": evidence}),),
        )
    )
    orchestrator.apply_proposal(
        SkillProposal(
            proposal_id="hypothesis",
            skill_name="HypothesisBuilder",
            input_revision=orchestrator.snapshot.revision,
            purpose="test",
            operations=(
                SkillOperation(
                    "add_or_update_hypothesis",
                    {
                        "item": {
                            "hypothesis_id": "h1",
                            "official_disease_name": "肺炎",
                            "supporting_evidence_ids": ["ev-1"],
                            "status": "selected",
                        }
                    },
                ),
            ),
        )
    )
    orchestrator.apply_proposal(
        SkillProposal(
            proposal_id="treatment",
            skill_name="TreatmentPlanner",
            input_revision=orchestrator.snapshot.revision,
            purpose="test",
            operations=(
                SkillOperation(
                    "update_treatment_draft",
                    {
                        "urgency_and_disposition": "outpatient",
                        "treatment_items": (),
                        "draft_text": "抗感染。立即大剂量激素。无需随访。",
                    },
                ),
            ),
        )
    )

    verifier = FinalVerifier()
    assert verifier.verify(orchestrator.snapshot).p0_count >= 1

    sanitizer = TreatmentDraftSanitizer()
    orchestrator.apply_proposal(sanitizer.propose(orchestrator.snapshot))
    cleaned = orchestrator.snapshot.treatment_state.draft_text
    assert "立即大剂量激素" not in cleaned
    assert "无需随访" not in cleaned

    orchestrator.apply_proposal(verifier.propose_issue_replacement(orchestrator.snapshot))
    assert verifier.verify(orchestrator.snapshot).p0_count == 0
