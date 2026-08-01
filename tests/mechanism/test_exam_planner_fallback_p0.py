"""P0: A5 planner admits only structured, useful catalog leaves."""

from __future__ import annotations

from agent.clinical.exam_planner import exam_value_record, plan_examinations


CATALOG = ("血常规", "C反应蛋白（CRP）", "胸部CT")


def _key(name: str) -> str:
    return name.strip().lower().replace("（crp）", "")


def _plan(*, examinations: list[str], accepted: list[dict]) -> dict:
    return {"examinations": examinations, "accepted": accepted}


def test_empty_plan_has_no_supported_exam_without_fallback() -> None:
    result = plan_examinations(
        raw_plan=_plan(examinations=[], accepted=[]),
        open_gaps=[],
        ordered_examinations=[],
        allowed_catalog_leaves=CATALOG,
        semantic_key_fn=_key,
    )

    assert result.status == "no_supported_exam"
    assert result.examinations == ()
    assert result.fallback_kind == "none"


def test_gap_plan_only_emits_its_accepted_catalog_leaf() -> None:
    result = plan_examinations(
        raw_plan=_plan(
            examinations=["血常规", "胸部CT"],
            accepted=[
                {
                    "name": "血常规",
                    "source": "coverage_gap_required",
                    "axis_id": "infection_gap",
                    "intent": "infection_workup",
                }
            ],
        ),
        open_gaps=[
            {"gap_id": "infection_gap", "required_exams": ["血常规"]},
            {"gap_id": "other_gap", "required_exams": ["胸部CT"]},
        ],
        ordered_examinations=[],
        allowed_catalog_leaves=CATALOG,
        semantic_key_fn=_key,
    )

    assert result.status == "ready"
    assert result.examinations == ("血常规",)
    assert result.gap_ids == ("infection_gap",)
    assert result.intent_ids == ("infection_workup",)


def test_unmapped_gap_cannot_select_catalog_default() -> None:
    result = plan_examinations(
        raw_plan=_plan(examinations=["血常规"], accepted=[]),
        open_gaps=[{"gap_id": "unmapped", "required_exams": []}],
        ordered_examinations=[],
        allowed_catalog_leaves=CATALOG,
        semantic_key_fn=_key,
    )

    assert result.status == "no_supported_exam"
    assert result.examinations == ()


def test_free_text_reason_is_not_structured_authorization() -> None:
    result = plan_examinations(
        raw_plan={
            "examinations": ["血常规"],
            "accepted": [{"name": "血常规", "source": "llm_exam_item"}],
            "reason_codes": ["coverage_gap_required:infection_gap"],
        },
        open_gaps=[{"gap_id": "infection_gap", "required_exams": ["血常规"]}],
        ordered_examinations=[],
        allowed_catalog_leaves=CATALOG,
        semantic_key_fn=_key,
    )

    assert result.status == "no_supported_exam"


def test_duplicate_or_semantic_duplicate_is_not_executable() -> None:
    raw_plan = _plan(
        examinations=["C反应蛋白（CRP）"],
        accepted=[
            {
                "name": "C反应蛋白（CRP）",
                "source": "typed_rule_intent",
                "intent": "inflammation",
            }
        ],
    )
    result = plan_examinations(
        raw_plan=raw_plan,
        open_gaps=[],
        ordered_examinations=["C反应蛋白"],
        allowed_catalog_leaves=CATALOG,
        semantic_key_fn=_key,
    )

    assert result.status == "no_supported_exam"
    assert result.examinations == ()


def test_value_record_marks_unobserved_outcomes_unknown() -> None:
    result = plan_examinations(
        raw_plan=_plan(
            examinations=["血常规"],
            accepted=[
                {
                    "name": "血常规",
                    "source": "required_intent",
                    "intent": "infection_workup",
                }
            ],
        ),
        open_gaps=[],
        ordered_examinations=[],
        allowed_catalog_leaves=CATALOG,
        semantic_key_fn=_key,
    )

    record = exam_value_record(
        result,
        candidates=[{"disease": "肺炎", "score": 10}],
        semantic_key_fn=_key,
    )
    assert record["semantic_keys"] == ["血常规"]
    assert record["candidate_hash_before"].startswith("sha256:")
    assert record["candidate_hash_after"] == "unknown"
    assert record["treatment_changed"] == "unknown"
    assert record["urgency_changed"] == "unknown"
    assert record["cost"] is None
    assert record["duration_ms"] is None
