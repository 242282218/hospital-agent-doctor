from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.legacy_orchestrator import MyDoctorAgent, load_knowledge_rules
from agent.memory import VerifiedOnlyMemory


class _Logger:
    output_dir: Path


def test_knowledge_loader_fails_closed_for_broken_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_knowledge_rules(missing)

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="failed to load knowledge file"):
        load_knowledge_rules(broken)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"rules": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no verified rules"):
        load_knowledge_rules(empty)


def test_request_service_url_override_is_loopback_only() -> None:
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=VerifiedOnlyMemory())
    with pytest.raises(ValueError, match="override is disabled"):
        agent.runtime_config({"service_base_url": "https://attacker.invalid"})
    with pytest.raises(ValueError, match="loopback"):
        agent.runtime_config(
            {"local_test": True, "service_base_url": "http://169.254.169.254"}
        )
    config = agent.runtime_config(
        {
            "local_test": True,
            "service_base_url": "http://127.0.0.1:8001",
            "service_token": "test-token",
            "model_api_key": "test-key",
        }
    )
    assert config.service_base_url == "http://127.0.0.1:8001"


def test_prompt_debug_log_contains_metadata_only(tmp_path: Path) -> None:
    agent = MyDoctorAgent(config={"log_llm_prompts": True}, memory=VerifiedOnlyMemory())
    logger = _Logger()
    logger.output_dir = tmp_path
    agent.logger = logger
    agent._write_prompt_log(
        prompt_name="diagnosis",
        patient_id="Patient_sensitive_123",
        system_prompt="secret system prompt",
        user_prompt="secret patient history",
        response="secret treatment response",
    )

    files = list((tmp_path / "llm_prompts").glob("*.txt"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "secret system prompt" not in content
    assert "secret patient history" not in content
    assert "secret treatment response" not in content
    assert "Patient_sensitive_123" not in content
    assert "system_prompt_sha256:" in content
    assert "response_chars:" in content


def test_incomplete_case_result_does_not_echo_exception_text() -> None:
    from agent.legacy_orchestrator import incomplete_case_result

    result = incomplete_case_result(
        "Patient_secret_1",
        RuntimeError("https://secret.invalid token=abc patient raw body"),
    )

    assert result["finished"] is False
    assert result["error_code"] == "case_incomplete"
    assert "secret.invalid" not in json.dumps(result)
    assert "token=abc" not in json.dumps(result)
    assert "raw body" not in json.dumps(result)


def test_agent_rejects_concurrent_case_execution() -> None:
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=VerifiedOnlyMemory())

    async def slow_run(*, patient_id: str, mode: str) -> dict:
        await asyncio.sleep(0.02)
        return {"patient_id": patient_id, "mode": mode, "finished": True}

    agent._run_agent = slow_run  # type: ignore[method-assign]

    async def scenario() -> None:
        first = asyncio.create_task(
            agent._run_agent_with_isolation("case-a", "test")
        )
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="does not support concurrent"):
            await agent._run_agent_with_isolation("case-b", "test")
        await first

    asyncio.run(scenario())
