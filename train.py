"""Local training entrypoint for the baseline agent."""

from __future__ import annotations

import json

from agent.agent import build_agent


def main() -> None:
    agent = build_agent()
    summary = agent.run_train()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
