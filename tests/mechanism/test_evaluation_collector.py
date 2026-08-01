from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from agent.runtime.evaluation_attempt_store import EvaluationAttemptStore
from agent.runtime.evaluation_collector import EvaluationCollector


class FakeEvaluationAdapter:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.failure: Optional[BaseException] = None

    async def collect_evaluation(self, final_result: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(final_result)
        if self.failure is not None:
            raise self.failure
        return {"diagnosisAccuracy": 1.0}


def test_train_collector_returns_attachment_and_attempts_once() -> None:
    async def scenario() -> tuple[FakeEvaluationAdapter, Any]:
        adapter = FakeEvaluationAdapter()
        collector = EvaluationCollector(adapter=adapter, mode="train")
        attachment = await collector.collect({"patient_id": "transport-only", "finished": True})
        with pytest.raises(RuntimeError, match="already attempted"):
            await collector.collect({"patient_id": "transport-only", "finished": True})
        return adapter, attachment

    adapter, attachment = asyncio.run(scenario())
    assert len(adapter.calls) == 1
    assert attachment.report == {"diagnosisAccuracy": 1.0}
    assert len(attachment.clinical_result_hash) == 64


def test_test_mode_never_calls_evaluation() -> None:
    async def scenario() -> FakeEvaluationAdapter:
        adapter = FakeEvaluationAdapter()
        collector = EvaluationCollector(adapter=adapter, mode="test")
        with pytest.raises(RuntimeError, match="train mode"):
            await collector.collect({"finished": True})
        return adapter

    adapter = asyncio.run(scenario())
    assert adapter.calls == []


def test_unfinished_result_is_rejected_before_attempt() -> None:
    async def scenario() -> FakeEvaluationAdapter:
        adapter = FakeEvaluationAdapter()
        collector = EvaluationCollector(adapter=adapter, mode="train")
        with pytest.raises(ValueError, match="finished=true"):
            await collector.collect({"finished": False})
        return adapter

    adapter = asyncio.run(scenario())
    assert adapter.calls == []


def test_failed_evaluation_cannot_be_retried() -> None:
    async def scenario() -> tuple[FakeEvaluationAdapter, BaseException]:
        adapter = FakeEvaluationAdapter()
        failure = TimeoutError("evaluation timeout")
        adapter.failure = failure
        collector = EvaluationCollector(adapter=adapter, mode="train")
        with pytest.raises(TimeoutError) as first:
            await collector.collect({"finished": True})
        assert first.value is failure
        with pytest.raises(RuntimeError, match="already attempted"):
            await collector.collect({"finished": True})
        return adapter, failure

    adapter, _ = asyncio.run(scenario())
    assert len(adapter.calls) == 1


def test_attempt_store_blocks_second_collector_instance(tmp_path) -> None:
    async def scenario() -> FakeEvaluationAdapter:
        adapter = FakeEvaluationAdapter()
        store = EvaluationAttemptStore(tmp_path / "attempts")
        first = EvaluationCollector(
            adapter=adapter,
            mode="train",
            case_run_id="case-a",
            store=store,
        )
        await first.collect({"finished": True, "diagnosis": ["x"]})
        second = EvaluationCollector(
            adapter=adapter,
            mode="train",
            case_run_id="case-a",
            store=store,
        )
        with pytest.raises(RuntimeError, match="already attempted"):
            await second.collect({"finished": True, "diagnosis": ["x"]})
        return adapter

    adapter = asyncio.run(scenario())
    assert len(adapter.calls) == 1


def test_failed_evaluation_still_persists_attempt(tmp_path) -> None:
    async def scenario() -> FakeEvaluationAdapter:
        adapter = FakeEvaluationAdapter()
        adapter.failure = TimeoutError("boom")
        store = EvaluationAttemptStore(tmp_path / "attempts")
        first = EvaluationCollector(
            adapter=adapter,
            mode="train",
            case_run_id="case-b",
            store=store,
        )
        with pytest.raises(TimeoutError):
            await first.collect({"finished": True})
        second = EvaluationCollector(
            adapter=adapter,
            mode="train",
            case_run_id="case-b",
            store=store,
        )
        with pytest.raises(RuntimeError, match="already attempted"):
            await second.collect({"finished": True})
        return adapter

    adapter = asyncio.run(scenario())
    assert len(adapter.calls) == 1
