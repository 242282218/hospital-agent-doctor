from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar


_PARENTHETICAL_CONTENT = re.compile(r"[（(]([^（）()]*)[）)]")
_ALIAS_PART_SEPARATOR = re.compile(r"[,，;；、/]")
_ALLOWED_PROCEDURE_KINDS = frozenset(
    {"diagnostic", "diagnostic_interventional", "therapeutic_like"}
)
_ALLOWED_RISK_LEVELS = frozenset({"low", "medium", "high"})
_APPROVED_NEAR_DUPLICATE_NAMES = (
    ("斜视评估", "斜视评估（Hirschberg试验）"),
    ("眼压测量", "眼压测量（IOP）"),
    ("外斐试验", "外斐试验（立克次体凝集）"),
    ("钡灌肠检查（BE）", "钡灌肠检查"),
)


def _strict_lookup_key(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized.strip()).casefold()


_APPROVED_NEAR_DUPLICATE_KEYS = tuple(
    tuple(_strict_lookup_key(name) for name in group)
    for group in _APPROVED_NEAR_DUPLICATE_NAMES
)
_APPROVED_NEAR_DUPLICATE_BY_NAME = {
    name: group_index
    for group_index, group in enumerate(_APPROVED_NEAR_DUPLICATE_KEYS)
    for name in group
}


def _unique_names(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = _strict_lookup_key(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _parenthetical_aliases(name: str) -> tuple[str, ...]:
    aliases: list[str] = []
    for content in _PARENTHETICAL_CONTENT.findall(name):
        content = content.strip()
        if content:
            aliases.append(content)
        aliases.extend(
            part.strip() for part in _ALIAS_PART_SEPARATOR.split(content) if part.strip()
        )
    return _unique_names(aliases)


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not _strict_lookup_key(value):
        raise ValueError("%s must be a non-empty string" % field)
    return value


@dataclass(frozen=True)
class DiseaseCatalogEntry:
    official_name: str
    department: str
    normalized_name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExamCatalogEntry:
    official_name: str
    category: str
    description: str
    normalized_name: str
    aliases: tuple[str, ...] = ()
    semantic_group: str = ""
    procedure_kind: str = "diagnostic"
    risk_level: str = "low"


_CatalogEntry = TypeVar("_CatalogEntry", DiseaseCatalogEntry, ExamCatalogEntry)


class CatalogIndex:
    def __init__(
        self,
        *,
        departments: Sequence[str],
        diseases: Sequence[DiseaseCatalogEntry],
        examinations: Sequence[ExamCatalogEntry],
        exam_categories: Sequence[str],
    ) -> None:
        self.departments = tuple(departments)
        self.diseases = tuple(diseases)
        self.examinations = tuple(examinations)
        self.exam_categories = tuple(exam_categories)
        self._disease_official_lookup = self._build_official_lookup(self.diseases)
        self._disease_alias_lookup = self._build_alias_lookup(self.diseases)
        self._exam_official_lookup = self._build_official_lookup(self.examinations)
        self._exam_alias_lookup = self._build_alias_lookup(self.examinations)
        self._exams_by_category = self._build_category_lookup()
        self._near_duplicate_groups = self._build_near_duplicate_groups()

    @staticmethod
    def _append_entry(
        grouped: dict[str, list[_CatalogEntry]], key: str, entry: _CatalogEntry
    ) -> None:
        entries = grouped.setdefault(key, [])
        if entry not in entries:
            entries.append(entry)

    @classmethod
    def _build_official_lookup(
        cls, entries: Sequence[_CatalogEntry]
    ) -> dict[str, tuple[_CatalogEntry, ...]]:
        grouped: dict[str, list[_CatalogEntry]] = {}
        for entry in entries:
            cls._append_entry(grouped, _strict_lookup_key(entry.official_name), entry)
        return {key: tuple(matches) for key, matches in grouped.items()}

    @classmethod
    def _build_alias_lookup(
        cls, entries: Sequence[_CatalogEntry]
    ) -> dict[str, tuple[_CatalogEntry, ...]]:
        grouped: dict[str, list[_CatalogEntry]] = {}
        for entry in entries:
            official_key = _strict_lookup_key(entry.official_name)
            for alias in entry.aliases:
                alias_key = _strict_lookup_key(alias)
                if alias_key and alias_key != official_key:
                    cls._append_entry(grouped, alias_key, entry)
        return {key: tuple(matches) for key, matches in grouped.items()}

    def _build_category_lookup(self) -> dict[str, tuple[ExamCatalogEntry, ...]]:
        grouped: dict[str, list[ExamCatalogEntry]] = {}
        for examination in self.examinations:
            key = _strict_lookup_key(examination.category)
            grouped.setdefault(key, []).append(examination)
        return {key: tuple(matches) for key, matches in grouped.items()}

    def _build_near_duplicate_groups(self) -> tuple[tuple[ExamCatalogEntry, ...], ...]:
        by_official_name = {
            _strict_lookup_key(examination.official_name): examination
            for examination in self.examinations
        }
        groups: list[tuple[ExamCatalogEntry, ...]] = []
        for group_keys in _APPROVED_NEAR_DUPLICATE_KEYS:
            if all(name in by_official_name for name in group_keys):
                groups.append(tuple(by_official_name[name] for name in group_keys))
        return tuple(groups)

    def find_diseases(self, value: object) -> tuple[DiseaseCatalogEntry, ...]:
        key = _strict_lookup_key(value)
        official_matches = self._disease_official_lookup.get(key)
        if official_matches is not None:
            return official_matches
        return self._disease_alias_lookup.get(key, ())

    def find_examinations(self, value: object) -> tuple[ExamCatalogEntry, ...]:
        key = _strict_lookup_key(value)
        official_matches = self._exam_official_lookup.get(key)
        if official_matches is not None:
            return official_matches
        return self._exam_alias_lookup.get(key, ())

    def examinations_for_category(self, category: object) -> tuple[ExamCatalogEntry, ...]:
        return self._exams_by_category.get(_strict_lookup_key(category), ())

    def near_duplicate_examination_groups(self) -> tuple[tuple[ExamCatalogEntry, ...], ...]:
        return self._near_duplicate_groups

    def deduplicate_examination_names(
        self,
        names: Sequence[object],
        preferred_official_names: Sequence[object] = (),
    ) -> tuple[str, ...]:
        preferred_keys = tuple(
            key for name in preferred_official_names if (key := _strict_lookup_key(name))
        )
        grouped: dict[tuple[str, object], list[ExamCatalogEntry]] = {}
        for name in names:
            for examination in self.find_examinations(name):
                official_key = _strict_lookup_key(examination.official_name)
                approved_group = _APPROVED_NEAR_DUPLICATE_BY_NAME.get(official_key)
                group_key = (
                    ("approved", approved_group)
                    if approved_group is not None
                    else ("official", official_key)
                )
                candidates = grouped.setdefault(group_key, [])
                if examination not in candidates:
                    candidates.append(examination)

        selected: list[str] = []
        for candidates in grouped.values():
            choice = next(
                (
                    candidate
                    for preferred_key in preferred_keys
                    for candidate in candidates
                    if _strict_lookup_key(candidate.official_name) == preferred_key
                ),
                candidates[0],
            )
            selected.append(choice.official_name)
        return tuple(selected)


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("catalog payload must be a JSON object: %s" % path)
    return payload


def _validated_disease_aliases(
    disease_alias_rules: Sequence[Mapping[str, object]], official_disease_keys: set[str]
) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, list[str]] = {}
    for rule in disease_alias_rules:
        if not isinstance(rule, Mapping) or rule.get("status") != "verified":
            continue
        output = _required_text(rule.get("output"), "disease alias output")
        output_key = _strict_lookup_key(output)
        if output_key not in official_disease_keys:
            raise ValueError("disease alias output is not an official disease: %s" % output)
        inputs = rule.get("input")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("disease alias input must be a non-empty list")
        values = aliases.setdefault(output_key, [])
        for value in inputs:
            values.append(_required_text(value, "disease alias input"))
    return {key: _unique_names(values) for key, values in aliases.items()}


def _validated_exam_overrides(
    exam_overrides: Sequence[Mapping[str, object]], official_exam_keys: set[str]
) -> dict[str, Mapping[str, object]]:
    overrides: dict[str, Mapping[str, object]] = {}
    for rule in exam_overrides:
        if not isinstance(rule, Mapping) or rule.get("status") != "verified":
            continue
        official_name = _required_text(rule.get("official_name"), "override official_name")
        official_key = _strict_lookup_key(official_name)
        if official_key not in official_exam_keys:
            raise ValueError(
                "override official_name is not an official examination leaf: %s"
                % official_name
            )
        procedure_kind = rule.get("procedure_kind")
        if procedure_kind not in _ALLOWED_PROCEDURE_KINDS:
            raise ValueError("override procedure_kind is invalid: %s" % procedure_kind)
        risk_level = rule.get("risk_level")
        if risk_level not in _ALLOWED_RISK_LEVELS:
            raise ValueError("override risk_level is invalid: %s" % risk_level)
        if official_key in overrides:
            raise ValueError("duplicate verified override official_name: %s" % official_name)
        overrides[official_key] = rule
    return overrides


def load_catalog_index(
    ref_data_dir: Path,
    *,
    exam_overrides: Sequence[Mapping[str, object]],
    disease_alias_rules: Sequence[Mapping[str, object]] = (),
) -> CatalogIndex:
    ref_data_dir = Path(ref_data_dir)
    departments_payload = _load_json(ref_data_dir / "departments.json")
    diseases_payload = _load_json(ref_data_dir / "diseases_catalog.json")
    examinations_payload = _load_json(ref_data_dir / "examinations_catalog.json")

    department_values = departments_payload.get("departments")
    disease_groups = diseases_payload.get("diseases")
    examination_groups = examinations_payload.get("examinations")
    if not isinstance(department_values, list):
        raise ValueError("departments must be a list")
    if not isinstance(disease_groups, Mapping):
        raise ValueError("diseases must be an object")
    if not isinstance(examination_groups, Mapping):
        raise ValueError("examinations must be an object")

    departments = tuple(_required_text(value, "department") for value in department_values)
    department_keys = tuple(_strict_lookup_key(department) for department in departments)
    if len(department_keys) != len(set(department_keys)):
        raise ValueError("duplicate normalized department")
    declared_departments = set(department_keys)

    disease_group_keys: set[str] = set()
    disease_names: set[str] = set()
    raw_diseases: list[tuple[str, str, str]] = []
    for department, names in disease_groups.items():
        department = _required_text(department, "disease department")
        department_key = _strict_lookup_key(department)
        if department_key not in declared_departments:
            raise ValueError("disease department is not declared: %s" % department)
        if department_key in disease_group_keys:
            raise ValueError("duplicate normalized disease department: %s" % department)
        disease_group_keys.add(department_key)
        if not isinstance(names, list):
            raise ValueError("disease names must be a list: %s" % department)
        for official_name in names:
            official_name = _required_text(official_name, "disease official_name")
            normalized_name = _strict_lookup_key(official_name)
            if normalized_name in disease_names:
                raise ValueError("duplicate normalized disease official name: %s" % official_name)
            disease_names.add(normalized_name)
            raw_diseases.append((official_name, department, normalized_name))

    verified_aliases = _validated_disease_aliases(disease_alias_rules, disease_names)
    diseases = [
        DiseaseCatalogEntry(
            official_name=official_name,
            department=department,
            normalized_name=normalized_name,
            aliases=_unique_names(
                (*_parenthetical_aliases(official_name), *verified_aliases.get(normalized_name, ()))
            ),
        )
        for official_name, department, normalized_name in raw_diseases
    ]

    category_keys: set[str] = set()
    examination_names: set[str] = set()
    raw_examinations: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    exam_categories: list[str] = []
    for category, items in examination_groups.items():
        category = _required_text(category, "examination category")
        category_key = _strict_lookup_key(category)
        if category_key in category_keys:
            raise ValueError("duplicate normalized examination category: %s" % category)
        category_keys.add(category_key)
        if not isinstance(items, list):
            raise ValueError("examination items must be a list: %s" % category)
        exam_categories.append(category)
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("examination item must be an object: %s" % category)
            official_name = _required_text(item.get("name"), "examination official_name")
            description = _required_text(item.get("description"), "examination description")
            normalized_name = _strict_lookup_key(official_name)
            if normalized_name in examination_names:
                raise ValueError(
                    "duplicate normalized examination official name: %s" % official_name
                )
            examination_names.add(normalized_name)
            raw_examinations.append(
                (
                    official_name,
                    category,
                    description,
                    normalized_name,
                    _parenthetical_aliases(official_name),
                )
            )

    overrides = _validated_exam_overrides(exam_overrides, examination_names)
    examinations: list[ExamCatalogEntry] = []
    for official_name, category, description, normalized_name, aliases in raw_examinations:
        override = overrides.get(normalized_name)
        approved_group = _APPROVED_NEAR_DUPLICATE_BY_NAME.get(normalized_name)
        examinations.append(
            ExamCatalogEntry(
                official_name=official_name,
                category=category,
                description=description,
                normalized_name=normalized_name,
                aliases=aliases,
                semantic_group=(
                    "approved_near_duplicate_%d" % approved_group
                    if approved_group is not None
                    else ""
                ),
                procedure_kind=(
                    str(override["procedure_kind"]) if override else "diagnostic"
                ),
                risk_level=str(override["risk_level"]) if override else "low",
            )
        )

    return CatalogIndex(
        departments=departments,
        diseases=diseases,
        examinations=examinations,
        exam_categories=exam_categories,
    )
