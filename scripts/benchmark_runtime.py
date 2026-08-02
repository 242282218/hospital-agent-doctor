"""Repeatable local startup and health-request benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parents[1]


def _percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def _phase(phase: str) -> int:
    sys.path.insert(0, str(BASE_DIR))
    if phase == "import":
        started = time.perf_counter()
        importlib.import_module("agent.agent")
    elif phase == "build":
        started = time.perf_counter()
        from agent.agent import build_agent

        build_agent(release_pointer="releases/current.json")
    else:
        raise ValueError("unsupported benchmark phase: %s" % phase)
    print(json.dumps({"phase": phase, "seconds": time.perf_counter() - started}))
    return 0


def _sample_phase(phase: str, samples: int) -> List[float]:
    values: List[float] = []
    for _ in range(samples):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--phase", phase],
            cwd=str(BASE_DIR),
            capture_output=True,
            check=True,
            text=True,
        )
        values.append(float(json.loads(result.stdout.strip().splitlines()[-1])["seconds"]))
    return values


def _health_samples(requests: int) -> Dict[str, Any]:
    sys.path.insert(0, str(BASE_DIR))
    from agent.agent import build_agent
    from hospital_agent_sdk.server import create_agent_server

    agent = build_agent(release_pointer="releases/current.json")
    app = create_agent_server(test_handler=agent.test)
    values: List[float] = []
    for _ in range(requests):
        started = time.perf_counter()
        response = app.test_client().get("/health")
        values.append(time.perf_counter() - started)
        if response.status_code != 200 or response.get_json() != {"status": "ok"}:
            raise RuntimeError("health smoke failed: %s %s" % (response.status_code, response.data))
    return {
        "count": len(values),
        "first_seconds": values[0],
        "median_seconds": statistics.median(values),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": max(values),
    }


def _summary(values: List[float]) -> Dict[str, Any]:
    return {
        "samples": len(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": max(values),
        "runs_seconds": values,
    }


def run_benchmark(samples: int, health_requests: int) -> Dict[str, Any]:
    if samples < 1 or health_requests < 1:
        raise ValueError("samples and health_requests must be positive")
    return {
        "schema_version": "runtime-speed-benchmark/v1",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "import": _summary(_sample_phase("import", samples)),
        "build": _summary(_sample_phase("build", samples)),
        "health": _health_samples(health_requests),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("import", "build"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--health-requests", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.phase:
        return _phase(args.phase)
    report = run_benchmark(args.samples, args.health_requests)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
