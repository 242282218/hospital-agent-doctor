from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from agent.knowledge.catalog_index import (
    CatalogIndex,
    DiseaseCatalogEntry,
    ExamCatalogEntry,
    load_catalog_index,
)
from agent.knowledge.verified_reader import VerifiedKnowledgeReader


ROOT = Path(__file__).resolve().parents[1]
REF_DATA_DIR = ROOT / "data" / "ref_data"
KNOWLEDGE_DIR = ROOT / "agent" / "knowledge"


def load_index():
    reader = VerifiedKnowledgeReader(KNOWLEDGE_DIR)
    return load_catalog_index(
        REF_DATA_DIR,
        exam_overrides=reader.read_verified_rules("catalog_exam_overrides.json"),
        disease_alias_rules=reader.read_verified_rules("alias_map.json"),
    )


def write_catalog(
    root: Path,
    *,
    departments: object | None = None,
    diseases: object | None = None,
    examinations: object | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "departments.json").write_text(
        json.dumps({"departments": ["测试科"] if departments is None else departments}, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "diseases_catalog.json").write_text(
        json.dumps({"diseases": {"测试科": ["测试病"]} if diseases is None else diseases}, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "examinations_catalog.json").write_text(
        json.dumps(
            {
                "examinations": {
                    "测试类别": [{"name": "测试检查", "description": "测试描述"}]
                }
                if examinations is None
                else examinations
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def test_catalog_index_preserves_complete_official_catalogs() -> None:
    index = load_index()
    departments_payload = json.loads((REF_DATA_DIR / "departments.json").read_text(encoding="utf-8"))
    diseases_payload = json.loads((REF_DATA_DIR / "diseases_catalog.json").read_text(encoding="utf-8"))
    examinations_payload = json.loads(
        (REF_DATA_DIR / "examinations_catalog.json").read_text(encoding="utf-8")
    )
    expected_diseases = {
        (official_name, department)
        for department, names in diseases_payload["diseases"].items()
        for official_name in names
    }
    expected_examinations = {
        (item["name"], category, item["description"])
        for category, items in examinations_payload["examinations"].items()
        for item in items
    }

    assert len(index.departments) == 19
    assert len(index.diseases) == 584
    assert len(index.examinations) == 843
    assert len(index.exam_categories) == 18
    assert index.departments == tuple(departments_payload["departments"])
    assert index.exam_categories == tuple(examinations_payload["examinations"])
    assert {(entry.official_name, entry.department) for entry in index.diseases} == expected_diseases
    assert {
        (entry.official_name, entry.category, entry.description)
        for entry in index.examinations
    } == expected_examinations
    assert [entry.official_name for entry in index.find_diseases(" ＳＬＥ ")] == [
        "系统性红斑狼疮"
    ]
    assert [entry.official_name for entry in index.find_examinations(" ＩＯＰ ")] == [
        "眼压测量（IOP）"
    ]


def test_alias_and_category_lookup_return_official_entries() -> None:
    index = load_index()

    disease = index.find_diseases("CAD")
    exam = index.find_examinations("IOP")
    eye_exams = index.examinations_for_category("功能检查")

    assert [entry.official_name for entry in disease] == [
        "冠状动脉粥样硬化性心脏病（冠状动脉疾病，CAD）"
    ]
    assert [entry.official_name for entry in exam] == ["眼压测量（IOP）"]
    assert "眼压测量（IOP）" in {entry.official_name for entry in eye_exams}


def test_disease_alias_rules_are_explicit_dependencies() -> None:
    index = load_catalog_index(REF_DATA_DIR, exam_overrides=())

    assert index.find_diseases("SLE") == ()


def test_near_duplicates_remain_officially_distinct_and_deduplicate_by_priority() -> None:
    index = load_index()

    groups = {
        frozenset(entry.official_name for entry in group)
        for group in index.near_duplicate_examination_groups()
    }

    assert groups == {
        frozenset({"斜视评估", "斜视评估（Hirschberg试验）"}),
        frozenset({"眼压测量", "眼压测量（IOP）"}),
        frozenset({"外斐试验", "外斐试验（立克次体凝集）"}),
        frozenset({"钡灌肠检查（BE）", "钡灌肠检查"}),
    }
    assert [entry.official_name for entry in index.find_examinations("眼压测量")] == [
        "眼压测量"
    ]
    assert index.deduplicate_examination_names(
        ["眼压测量", "眼压测量（IOP）"],
        preferred_official_names=["眼压测量（IOP）"],
    ) == ("眼压测量（IOP）",)


def test_unapproved_parenthetical_variants_are_not_deduplicated() -> None:
    base = ExamCatalogEntry("自定义检查", "类别", "说明", "自定义检查")
    variant = ExamCatalogEntry("自定义检查（变体）", "类别", "说明", "自定义检查(变体)")
    index = CatalogIndex(
        departments=(),
        diseases=(),
        examinations=(base, variant),
        exam_categories=("类别",),
    )

    assert index.near_duplicate_examination_groups() == ()
    assert index.deduplicate_examination_names([base.official_name, variant.official_name]) == (
        base.official_name,
        variant.official_name,
    )


def test_official_exact_lookup_does_not_mix_alias_matches() -> None:
    official_disease = DiseaseCatalogEntry("Exact Disease", "科室", "exact disease")
    alias_disease = DiseaseCatalogEntry(
        "Alias Disease", "科室", "alias disease", aliases=("EXACT DISEASE",)
    )
    official_exam = ExamCatalogEntry("Exact Exam", "类别", "说明", "exact exam")
    alias_exam = ExamCatalogEntry(
        "Alias Exam", "类别", "说明", "alias exam", aliases=("EXACT EXAM",)
    )
    index = CatalogIndex(
        departments=("科室",),
        diseases=(official_disease, alias_disease),
        examinations=(official_exam, alias_exam),
        exam_categories=("类别",),
    )

    assert index.find_diseases("EXACT DISEASE") == (official_disease,)
    assert index.find_examinations("EXACT EXAM") == (official_exam,)


def test_reader_filtered_candidate_overrides_do_not_reach_catalog_index(tmp_path: Path) -> None:
    catalog_dir = write_catalog(tmp_path / "catalog")
    (tmp_path / "catalog_exam_overrides.json").write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "verified",
                        "status": "verified",
                        "official_name": "测试检查",
                        "procedure_kind": "therapeutic_like",
                        "risk_level": "high",
                    },
                    {
                        "id": "candidate",
                        "status": "candidate",
                        "official_name": "测试检查",
                        "procedure_kind": "diagnostic",
                        "risk_level": "low",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reader = VerifiedKnowledgeReader(tmp_path)

    assert [rule["id"] for rule in reader.read_verified_rules("catalog_exam_overrides.json")] == [
        "verified"
    ]
    entry = load_catalog_index(
        catalog_dir,
        exam_overrides=reader.read_verified_rules("catalog_exam_overrides.json"),
    ).find_examinations("测试检查")[0]
    assert (entry.procedure_kind, entry.risk_level) == ("therapeutic_like", "high")


@pytest.mark.parametrize(
    "payload",
    [[], {}, {"rules": {}}, {"rules": [None]}],
    ids=["non-object", "missing-rules", "rules-not-list", "non-object-rule"],
)
def test_verified_reader_rejects_malformed_rule_payloads(tmp_path: Path, payload: object) -> None:
    (tmp_path / "rules.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="rules"):
        VerifiedKnowledgeReader(tmp_path).read_verified_rules("rules.json")


def test_direct_unverified_overrides_do_not_apply_and_invalid_verified_overrides_fail(
    tmp_path: Path,
) -> None:
    catalog_dir = write_catalog(tmp_path / "catalog")
    unverified = {
        "status": "candidate",
        "official_name": "测试检查",
        "procedure_kind": "therapeutic_like",
        "risk_level": "high",
    }
    index = load_catalog_index(catalog_dir, exam_overrides=[unverified])

    assert index.find_examinations("测试检查")[0].procedure_kind == "diagnostic"
    assert index.find_examinations("测试检查")[0].risk_level == "low"
    missing_status = {
        "official_name": "测试检查",
        "procedure_kind": "therapeutic_like",
        "risk_level": "high",
    }
    index = load_catalog_index(catalog_dir, exam_overrides=[missing_status])
    assert index.find_examinations("测试检查")[0].procedure_kind == "diagnostic"

    with pytest.raises(ValueError, match="official_name"):
        load_catalog_index(
            catalog_dir,
            exam_overrides=[
                {
                    "status": "verified",
                    "official_name": "未知检查",
                    "procedure_kind": "diagnostic",
                    "risk_level": "low",
                }
            ],
        )
    with pytest.raises(ValueError, match="procedure_kind"):
        load_catalog_index(
            catalog_dir,
            exam_overrides=[
                {
                    "status": "verified",
                    "official_name": "测试检查",
                    "procedure_kind": "unsupported",
                    "risk_level": "low",
                }
            ],
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            [
                {
                    "status": "verified",
                    "official_name": "测试检查",
                    "procedure_kind": "diagnostic",
                    "risk_level": "invalid",
                }
            ],
            "risk_level",
        ),
        (
            [
                {
                    "status": "verified",
                    "official_name": "测试检查",
                    "procedure_kind": "diagnostic",
                    "risk_level": "low",
                },
                {
                    "status": "verified",
                    "official_name": "测试检查",
                    "procedure_kind": "therapeutic_like",
                    "risk_level": "high",
                },
            ],
            "duplicate verified override",
        ),
    ],
)
def test_catalog_index_fails_closed_for_invalid_verified_overrides(
    tmp_path: Path, overrides: list[dict[str, str]], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        load_catalog_index(
            write_catalog(tmp_path / "catalog"),
            exam_overrides=overrides,
        )


def test_catalog_index_rejects_verified_alias_rule_with_unknown_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output"):
        load_catalog_index(
            write_catalog(tmp_path / "catalog"),
            exam_overrides=(),
            disease_alias_rules=[
                {
                    "status": "verified",
                    "input": ["测试别名"],
                    "output": "未知疾病",
                }
            ],
        )


def test_catalog_index_rejects_duplicate_normalized_official_names(tmp_path: Path) -> None:
    duplicate_diseases = write_catalog(
        tmp_path / "duplicate-disease",
        diseases={"测试科": ["测试病", " 测试病 "]},
    )
    with pytest.raises(ValueError, match="duplicate normalized disease"):
        load_catalog_index(duplicate_diseases, exam_overrides=())

    duplicate_examinations = write_catalog(
        tmp_path / "duplicate-examination",
        examinations={
            "测试类别": [
                {"name": "测试检查", "description": "说明一"},
                {"name": " 测试检查 ", "description": "说明二"},
            ]
        },
    )
    with pytest.raises(ValueError, match="duplicate normalized examination"):
        load_catalog_index(duplicate_examinations, exam_overrides=())


def test_catalog_index_rejects_empty_or_invalid_catalog_fields(tmp_path: Path) -> None:
    invalid_catalogs = (
        write_catalog(tmp_path / "empty-department", departments=[""] ),
        write_catalog(tmp_path / "empty-category", examinations={"": []}),
        write_catalog(
            tmp_path / "empty-name",
            examinations={"测试类别": [{"name": "", "description": "说明"}]},
        ),
        write_catalog(
            tmp_path / "empty-description",
            examinations={"测试类别": [{"name": "测试检查", "description": ""}]},
        ),
        write_catalog(tmp_path / "invalid-diseases-schema", diseases=[]),
    )

    for catalog_dir in invalid_catalogs:
        with pytest.raises(ValueError):
            load_catalog_index(catalog_dir, exam_overrides=())


def test_loading_catalog_index_does_not_modify_official_json() -> None:
    catalog_paths = tuple(sorted(REF_DATA_DIR.glob("*.json")))
    before = {path: sha256(path.read_bytes()).hexdigest() for path in catalog_paths}

    load_index()

    after = {path: sha256(path.read_bytes()).hexdigest() for path in catalog_paths}
    assert after == before
