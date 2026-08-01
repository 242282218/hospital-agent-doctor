from __future__ import annotations

import json
from pathlib import Path

from agent.legacy_orchestrator import MyDoctorAgent
from agent.memory import VerifiedOnlyMemory
from agent.runtime.sdk_event_logger import RedactingEventLogger, install_sdk_event_logger


TOKENS = {
    "PATIENT_SECRET_8421",
    "QUESTION_SECRET_8421",
    "ANSWER_SECRET_8421",
    "EXAM_RESULT_SECRET_8421",
    "TREATMENT_SECRET_8421",
    "REPORT_SECRET_8421",
    "https://internal-secret.invalid",
}


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sdk_event_logger_redacts_operational_and_evaluation_logs(tmp_path: Path) -> None:
    logger = RedactingEventLogger(tmp_path)
    logger.write_event(
        "SEND_MESSAGE",
        "team-secret",
        "PATIENT_SECRET_8421",
        payload={
            "patient_id": "PATIENT_SECRET_8421",
            "target": "PATIENT_SECRET_8421",
            "input": {"question": "QUESTION_SECRET_8421"},
            "service_base_url": "https://internal-secret.invalid",
        },
    )
    logger.write_event(
        "PATIENT_REPLY",
        "PATIENT_SECRET_8421",
        "PATIENT_SECRET_8421",
        payload={"content": "ANSWER_SECRET_8421", "message_type": "patient_to_doctor"},
    )
    logger.write_event(
        "DO_EXAMINATION",
        "team-secret",
        "PATIENT_SECRET_8421",
        payload={
            "items": ["血常规"],
            "reason": "QUESTION_SECRET_8421",
            "results": {"血常规": {"result": "EXAM_RESULT_SECRET_8421"}},
            "results_count": 1,
        },
    )
    logger.write_evaluation_result(
        "PATIENT_SECRET_8421", {"report": "REPORT_SECRET_8421"}
    )

    serialized = "\n".join(
        [
            logger.events_path.read_text(encoding="utf-8"),
            logger.evaluation_results_path.read_text(encoding="utf-8"),
        ]
    )
    for token in TOKENS:
        assert token not in serialized
    event = _read_lines(logger.events_path)[0]
    assert event["patient_ref"].startswith("sha256:")
    assert event["payload"]["payload_hash"].startswith("sha256:")
    assert event["payload"]["field_count"] == 4
    evaluation = _read_lines(logger.evaluation_results_path)[0]
    assert evaluation["report_hash"].startswith("sha256:")


def test_final_result_contract_and_latest_lookup_remain_functional(tmp_path: Path) -> None:
    logger = RedactingEventLogger(tmp_path)
    final_result = {
        "patient_id": "PATIENT_SECRET_8421",
        "diagnosis": ["肺炎"],
        "treatment_plan": "TREATMENT_SECRET_8421",
        "reasoning": "REPORT_SECRET_8421",
        "finished": True,
    }
    logger.write_final_result(final_result)
    assert _read_lines(logger.final_results_path) == [final_result]
    assert logger.latest_final_result("PATIENT_SECRET_8421") == final_result




def test_real_sdk_actions_use_redacted_logger(tmp_path: Path) -> None:
    from hospital_agent_sdk.actions import DoctorActions

    class PatientClient:
        async def invoke(self, *, patient_id: str, input_data: dict) -> str:
            return "ANSWER_SECRET_8421"

    class ExamClient:
        async def get_results(self, *, patient_id: str, items: list[str]) -> dict:
            return {"results": {items[0]: {"status": "abnormal", "result": "EXAM_RESULT_SECRET_8421"}}}

    class EvaluateClient:
        async def evaluate_case(self, final_result: dict) -> dict:
            return {"report": "REPORT_SECRET_8421"}

        async def evaluate(self, output_log_path: str | Path) -> dict:
            return {"report": "REPORT_SECRET_8421"}

    logger = RedactingEventLogger(tmp_path)
    actions = DoctorActions(
        patient_client=PatientClient(),
        exam_client=ExamClient(),
        evaluate_client=EvaluateClient(),
        event_logger=logger,
        team_id="team-secret",
    )

    async def scenario() -> None:
        await actions.ask_patient(
            "PATIENT_SECRET_8421", {"question": "QUESTION_SECRET_8421"}
        )
        await actions.order_examination(
            "PATIENT_SECRET_8421", ["血常规"], reason="QUESTION_SECRET_8421"
        )
        final_result = await actions.prescribe_treatment(
            "PATIENT_SECRET_8421",
            ["肺炎"],
            "TREATMENT_SECRET_8421",
            reasoning="REPORT_SECRET_8421",
        )
        await actions.evaluation("PATIENT_SECRET_8421", final_result)

    import asyncio

    asyncio.run(scenario())
    operational = "\n".join(
        [
            logger.events_path.read_text(encoding="utf-8"),
            logger.evaluation_results_path.read_text(encoding="utf-8"),
        ]
    )
    for token in TOKENS:
        assert token not in operational
    assert "PATIENT_SECRET_8421" in logger.final_results_path.read_text(encoding="utf-8")
    install_sdk_event_logger()
    install_sdk_event_logger()
    import hospital_agent_sdk.runtime as sdk_runtime

    assert sdk_runtime.EventLogger is RedactingEventLogger
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=VerifiedOnlyMemory())
    agent.runtime_config({})
    assert sdk_runtime.EventLogger is RedactingEventLogger
