from __future__ import annotations

from pathlib import Path

from hospital_agent_sdk.server import create_agent_server

import agent.agent as entrypoint


def test_health_smoke_does_not_call_case_handler(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []

    def forbidden_handler(**_: object) -> dict[str, object]:
        calls.append(object())
        raise AssertionError("health must not invoke /test handler")

    monkeypatch.chdir(tmp_path)
    app = create_agent_server(test_handler=forbidden_handler)
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert calls == []


def test_build_agent_and_health_smoke_use_local_release_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    agent = entrypoint.build_agent(release_pointer="releases/release_C_runtime_final-pointer.json")
    app = create_agent_server(test_handler=agent.test)
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
