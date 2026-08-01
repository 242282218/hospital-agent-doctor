from __future__ import annotations

from agent.observability.runtime_events import (
    SequencedEventSink,
    canonical_hash,
    safe_diagnosis_state_event,
    safe_exam_plan_event,
)


def test_sink_redacts_clinical_fields_from_safe_event_constructors() -> None:
    written: list[dict[str, object]] = []
    sink = SequencedEventSink(append=written.append, case_run_id="case-1")

    sink(
        safe_exam_plan_event(
            {
                "examinations": ["chest CT"],
                "reason_codes": ["persistent cough with hemoptysis"],
                "accepted": [{"source": "clinical", "semantic_key": "lung_mass"}],
                "open_gap_ids": ["lung_findings"],
            }
        )
    )
    sink(
        safe_diagnosis_state_event(
            axis_ids=["anatomic_site"],
            candidate_names=["lung cancer"],
            consistency_issue_codes=["malignancy evidence insufficient"],
        )
    )

    exam_event, diagnosis_event = written
    assert exam_event["examinations_hash"] == canonical_hash(["chest CT"])
    assert exam_event["examination_count"] == 1
    assert "examinations" not in exam_event
    assert "persistent cough with hemoptysis" not in str(exam_event)
    assert diagnosis_event["axis_ids_hash"] == canonical_hash(["anatomic_site"])
    assert diagnosis_event["axis_count"] == 1
    assert "axis_ids" not in diagnosis_event
    assert "anatomic_site" not in str(diagnosis_event)
    assert diagnosis_event["candidate_names_hash"] == canonical_hash(["lung cancer"])
    assert diagnosis_event["candidate_count"] == 1
    assert "candidate_names" not in diagnosis_event
    assert "lung cancer" not in str(diagnosis_event)
    assert "malignancy evidence insufficient" not in str(diagnosis_event)


def test_sink_projects_exam_value_record_without_clinical_plaintext() -> None:
    written: list[dict[str, object]] = []
    sink = SequencedEventSink(append=written.append, case_run_id="case-exam-value")
    candidate_hash_before = "sha256:" + "a" * 64

    sink(
        safe_exam_plan_event(
            {
                "examinations": ["UNIQUE_EXAM_TOKEN"],
                "reason_codes": ["axis_intent"],
                "value": {
                    "gap_ids": ["UNIQUE_GAP_TOKEN"],
                    "intent_ids": ["UNIQUE_INTENT_TOKEN"],
                    "semantic_keys": ["UNIQUE_SEMANTIC_TOKEN"],
                    "candidate_hash_before": candidate_hash_before,
                    "candidate_hash_after": "unknown",
                    "treatment_changed": "unknown",
                    "urgency_changed": "unknown",
                    "cost": None,
                    "duration_ms": None,
                    "raw_result": "UNIQUE_CLINICAL_TEXT_TOKEN",
                },
            }
        )
    )

    event = written[0]
    assert event["value_gap_count"] == 1
    assert event["value_intent_count"] == 1
    assert event["candidate_hash_before"] == candidate_hash_before
    assert event["candidate_hash_after"] == "unknown"
    assert event["treatment_changed"] == "unknown"
    assert event["urgency_changed"] == "unknown"
    assert event["cost"] is None
    assert event["duration_ms"] is None
    for token in ("UNIQUE_GAP_TOKEN", "UNIQUE_INTENT_TOKEN", "UNIQUE_SEMANTIC_TOKEN", "UNIQUE_CLINICAL_TEXT_TOKEN"):
        assert token not in str(event)


def test_sink_projects_direct_dict_emit_before_enveloping() -> None:
    written: list[dict[str, object]] = []
    sink = SequencedEventSink(append=written.append, case_run_id="case-2")

    sink(
        {
            "type": "runtime_decision",
            "action": "ask_patient",
            "reason": "patient has chest pain and dyspnea",
            "question": "How long has the chest pain lasted?",
            "diagnosis": "acute coronary syndrome",
            "examinations": ["ECG"],
            "candidates": ["myocardial infarction"],
            "issues": ["missing troponin result"],
            "unexpected_text": "must not enter trace",
        }
    )

    event = written[0]
    assert event == {
        "schema_version": "clinical-runtime-event/v1",
        "case_run_id": "case-2",
        "type": "runtime_decision",
        "action": "ask_patient",
        "reason_hash": canonical_hash("patient has chest pain and dyspnea"),
        "question_hash": "",
        "ordered_exam_count": 0,
        "sequence": 1,
    }


def test_sink_normalizes_external_observation_statuses() -> None:
    written: list[dict[str, object]] = []
    sink = SequencedEventSink(append=written.append, case_run_id="case-observation")
    clinical_text = "UNIQUE_DIAGNOSIS_TOKEN UNIQUE_TREATMENT_TOKEN UNIQUE_REASON_TOKEN"

    sink(
        {
            "type": "action_observation",
            "command_id": "command-1",
            "action_sequence": 1,
            "dispatch_status": clinical_text,
            "observation_status": clinical_text,
            "item_status_counts": {clinical_text: 2, "succeeded": 1},
            "raw_result": {"results": {"exam": {"status": clinical_text}}},
        }
    )

    event = written[0]
    assert event["dispatch_status"] == "unknown"
    assert event["observation_status"] == "unknown"
    assert event["item_status_counts"] == {"unknown": 2, "succeeded": 1}
    assert event["result_hash"] == canonical_hash(
        {"results": {"exam": {"status": clinical_text}}}
    )
    assert clinical_text not in str(event)


def test_sink_hashes_raw_action_command_payload() -> None:
    written: list[dict[str, object]] = []
    sink = SequencedEventSink(append=written.append, case_run_id="case-raw-command")
    payload = {
        "diagnosis": "UNIQUE_DIAGNOSIS_TOKEN",
        "treatment_plan": "UNIQUE_TREATMENT_TOKEN",
        "reasoning": "UNIQUE_REASON_TOKEN",
    }

    sink(
        {
            "type": "action_command",
            "command_id": "command-1",
            "action_sequence": 1,
            "action_type": "prescribe_treatment",
            "payload": payload,
            "exam_items": ["UNIQUE_EXAM_TOKEN"],
        }
    )

    event = written[0]
    assert event["payload_hash"] == canonical_hash(payload)
    assert event["exam_items_hash"] == canonical_hash(["UNIQUE_EXAM_TOKEN"])
    for token in payload.values():
        assert token not in str(event)
    assert "UNIQUE_EXAM_TOKEN" not in str(event)
    assert "payload" not in event
    assert "exam_items" not in event


def test_sink_reduces_unrecognized_direct_event_to_hash_and_count() -> None:
    written: list[dict[str, object]] = []
    sink = SequencedEventSink(append=written.append, case_run_id="case-3")
    raw_event = {"type": "diagnosis_detail", "diagnosis": "pneumonia", "reason": "fever and cough"}

    sink(raw_event)

    assert written[0] == {
        "schema_version": "clinical-runtime-event/v1",
        "case_run_id": "case-3",
        "type": "unrecognized_runtime_event",
        "event_type_hash": canonical_hash("diagnosis_detail"),
        "event_hash": canonical_hash(raw_event),
        "field_count": 3,
        "sequence": 1,
    }
