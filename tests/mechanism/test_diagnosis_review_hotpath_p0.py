"""A4 hot-path review integration contracts."""
from __future__ import annotations

import ast
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[2] / "agent" / "legacy_orchestrator.py"


def _function_source(name: str) -> str:
    text = _SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError("missing function: %s" % name)


def test_primary_candidate_path_runs_review_before_treatment_safety() -> None:
    source = _function_source("run_full_clinical_loop")
    primary_start = source.index("if disease_candidates:")
    fallback_start = source.index("# 诊断步骤 1")
    primary = source[primary_start:fallback_start]
    assert primary.index("_run_independent_diagnosis_review(") < primary.index("apply_treatment_safety(")
    assert "diagnosis_review_rebuild_required" in primary


def test_department_fallback_runs_review_before_treatment_safety() -> None:
    source = _function_source("run_full_clinical_loop")
    fallback = source[source.index("# 诊断步骤 1") :]
    assert fallback.index("_run_independent_diagnosis_review(") < fallback.index("apply_treatment_safety(")
    assert "diagnosis_review_rebuild_required" in fallback


def test_reviewed_diagnosis_discards_first_treatment_and_reasoning() -> None:
    source = _function_source("run_full_clinical_loop")
    assert source.count('final_plan["treatment_plan"] = ""') == 1
    assert source.count('final_plan["reasoning"] = ""') == 1
    assert 'treatment_plan = ""' in source
    assert 'reasoning = ""' in source


def test_review_replacement_rechecks_candidate_and_dominant_axis_guards() -> None:
    source = _function_source("_run_independent_diagnosis_review")
    assert "enforce_selected_diagnosis_consistency(" in source
    assert "preferred_safe_escalation_diagnosis(" in source
    assert "supporting_ids" in source
    assert "evidence_id not in evidence_ids" in source
