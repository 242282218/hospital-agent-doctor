from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent.memory import VerifiedOnlyMemory, build_memory


def _case_memory(patient_id: str, diagnosis: str = "三房心") -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "diagnoses": [diagnosis],
        "examinations": ["体格检查", "超声心动图"],
        "treatment_plan": "尽快进行心脏外科评估。",
        "clinical_basis": ["先天性心脏结构异常"],
        "provenance": {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "a" * 64,
        },
    }


def _case_asset(content: Any) -> Dict[str, Any]:
    return {
        "candidate_id": "case-memory",
        "candidate_type": "case_memory",
        "content": content,
    }


def _write_registry(tmp_path: Path, assets: List[Dict[str, Any]]) -> Path:
    path = tmp_path / "verified_registry.json"
    path.write_text(
        json.dumps(
            {"schema_version": "verified-registry/v1", "assets": assets},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_verified_memory_returns_only_exact_patient(tmp_path: Path) -> None:
    first = _case_memory("Patient_01061")
    second = _case_memory("Patient_09249", diagnosis="腺病毒性结膜炎")
    memory = VerifiedOnlyMemory(
        _write_registry(tmp_path, [_case_asset(first), _case_asset(second)])
    )

    assert memory.load_case_memory("Patient_01061") == first
    assert memory.load_case_memory("Patient_09249") == second
    assert memory.load_case_memory("Patient_00000") is None


@pytest.mark.parametrize(
    "patient_id",
    [
        "patient_01061",
        "Patient_1061",
        "Patient_01061_extra",
        "Patient_Comorbid-01061",
    ],
)
def test_verified_memory_does_not_fuzz_patient_id(
    tmp_path: Path,
    patient_id: str,
) -> None:
    memory = VerifiedOnlyMemory(
        _write_registry(tmp_path, [_case_asset(_case_memory("Patient_01061"))])
    )

    assert memory.load_case_memory(patient_id) is None


def test_verified_memory_trims_lookup_whitespace(tmp_path: Path) -> None:
    expected = _case_memory("Patient_01061")
    memory = VerifiedOnlyMemory(_write_registry(tmp_path, [_case_asset(expected)]))

    assert memory.load_case_memory("  Patient_01061\n") == expected


def test_verified_memory_rejects_duplicate_patient_assets(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path,
        [
            _case_asset(_case_memory("Patient_01061")),
            _case_asset(_case_memory("Patient_01061", diagnosis="错误诊断")),
        ],
    )

    with pytest.raises(ValueError, match="duplicate case memory patient_id"):
        VerifiedOnlyMemory(registry)


def test_case_memory_assets_are_not_exposed_as_generic_notes(tmp_path: Path) -> None:
    case_memory = _case_memory("Patient_01061")
    generic_content = {"rule": "存在结构异常线索时优先完成超声心动图。"}
    memory = VerifiedOnlyMemory(
        _write_registry(
            tmp_path,
            [
                {"candidate_type": "generic_rule", "content": generic_content},
                _case_asset(case_memory),
            ],
        )
    )

    notes = memory.load_notes()

    assert notes == [json.dumps(generic_content, ensure_ascii=False, sort_keys=True)]
    assert "Patient_01061" not in "".join(notes)
    assert "treatment_plan" not in "".join(notes)


def test_generic_verified_notes_preserve_order_and_limit(tmp_path: Path) -> None:
    contents = [{"rule": "first"}, {"rule": "second"}, {"rule": "third"}]
    memory = VerifiedOnlyMemory(
        _write_registry(
            tmp_path,
            [
                {"candidate_type": "generic_rule", "content": content}
                for content in contents
            ],
        ),
        max_notes=2,
    )
    expected = [
        json.dumps(content, ensure_ascii=False, sort_keys=True)[:1200]
        for content in contents
    ]

    assert memory.load_notes() == expected[:2]
    assert memory.load_notes(limit=1) == expected[:1]
    assert memory.load_notes(limit=0) == expected[:2]


def test_loaded_case_memory_is_copy_isolated(tmp_path: Path) -> None:
    expected = _case_memory("Patient_01061")
    memory = VerifiedOnlyMemory(_write_registry(tmp_path, [_case_asset(expected)]))

    first = memory.load_case_memory("Patient_01061")
    assert first is not None
    first["diagnoses"][0] = "篡改诊断"
    first["provenance"]["source"] = "forged"

    second = memory.load_case_memory("Patient_01061")
    assert second is not None
    assert second["diagnoses"] == ["三房心"]
    assert second["provenance"]["source"] == "train_evaluation"


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not-a-mapping",
        {"diagnoses": ["三房心"]},
        {"patient_id": "", "diagnoses": ["三房心"]},
        {"patient_id": "Patient_01061"},
        {
            **_case_memory("Patient_01061"),
            "diagnoses": "三房心",
        },
        {
            **_case_memory("Patient_01061"),
            "treatment_plan": "",
        },
        {
            **_case_memory("Patient_01061"),
            "provenance": {"source": "unknown", "evaluation_hash": "invalid"},
        },
    ],
)
def test_malformed_case_memory_is_not_exposed(
    tmp_path: Path,
    content: Any,
) -> None:
    memory = VerifiedOnlyMemory(_write_registry(tmp_path, [_case_asset(content)]))

    assert memory.load_case_memory("Patient_01061") is None
    assert memory.load_notes() == []


def test_verified_memory_keeps_online_writes_disabled(tmp_path: Path) -> None:
    memory = VerifiedOnlyMemory(_write_registry(tmp_path, []))

    with pytest.raises(RuntimeError, match="online memory writes are disabled"):
        memory.append_case_reflection(patient_id="Patient_01061")


def test_build_memory_exposes_case_memory_from_explicit_registry(tmp_path: Path) -> None:
    expected = _case_memory("Patient_01061")
    registry = _write_registry(tmp_path, [_case_asset(expected)])

    memory = build_memory(
        {
            "memory": {
                "verified_registry_path": str(registry),
                "max_notes": 2,
            }
        }
    )

    assert isinstance(memory, VerifiedOnlyMemory)
    assert memory.load_case_memory("Patient_01061") == expected
