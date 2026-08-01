"""SDK composition root for the doctor agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from hospital_agent_sdk import AgentBuilder, load_config

from .clinical.final_submission import LoadedRuntimeIdentity
from .legacy_orchestrator import MyDoctorAgent
from .memory import build_memory
from .runtime.release_loader import LoadedRelease, load_current_release


CONFIG_PATH = Path("config.yaml")
RELEASE_POINTER = Path("releases/current.json")


def load_release_if_present(
    pointer_path: Union[str, Path] = RELEASE_POINTER,
) -> Optional[LoadedRelease]:
    path = Path(pointer_path)
    if not path.exists():
        return None
    return load_current_release(path)


def build_agent(
    config_path: Union[str, Path] = CONFIG_PATH,
    *,
    release_pointer: Union[str, Path] = RELEASE_POINTER,
) -> MyDoctorAgent:
    config = load_config(config_path)
    release = load_release_if_present(release_pointer)
    memory = build_memory(config, loaded_release=release)
    runtime_identity = (
        getattr(release, "runtime_identity", None) if release is not None else None
    )
    agent = MyDoctorAgent(
        config=config,
        memory=memory,
        rule_pack=release.knowledge_rule_pack if release is not None else None,
        runtime_identity=runtime_identity,
        prompt_pack=release.prompt_pack if release is not None else None,
    )
    if release is not None:
        # Mutate the same config object so composition preserves identity.
        target = agent.config if isinstance(agent.config, dict) else {}
        target["release_pack"] = {
            "pack_hash": release.pointer.get("pack_hash"),
            "pointer_path": str(Path(release_pointer).resolve()),
            "schema_version": release.manifest.get("schema_version"),
            "policy": dict(release.policy_pack),
            "runtime_code_hash": release.manifest.get("runtime_code_hash") or "",
            "prompt_pack_hash": release.manifest.get("prompt_pack_hash") or "",
            "authority_policy": release.manifest.get("authority_policy") or "",
            "authority_policy_hash": release.manifest.get("authority_policy_hash") or "",
            "prompt_pack_keys": sorted(release.prompt_pack.keys()),
            "registry_asset_count": len(list(release.registry.get("assets") or [])),
            "typed_rule_count": release.knowledge_rule_pack.rule_count,
            "typed_rules_active": any(
                rule.runtime.status == "active"
                for rule in release.knowledge_rule_pack.rules
            ),
            "runtime_identity_status": str(
                getattr(runtime_identity, "status", "legacy_unverified")
            ),
            "runtime_identity_hash": str(
                getattr(runtime_identity, "identity_hash", "")
            ),
        }
        agent.config = target
    return agent


def main() -> None:
    AgentBuilder(build_agent()).start()


__all__ = ["MyDoctorAgent", "build_agent", "load_release_if_present", "main"]


if __name__ == "__main__":
    main()
