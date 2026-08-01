"""Strict runtime reader for frozen disease profiles and reflection rules.

The reader only ever sees assets that a frozen verified registry already
contains: it never touches a candidate file, a harvest run or any other offline
artifact. Every asset is re-validated against the exact offline schema, so a
release that somehow carries a malformed or unknown-schema profile fails closed
instead of leaking answer-shaped content into prompts.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Collection, Mapping, Sequence

from agent.knowledge.profile_schema import (
    EXAM_PROFILE_TYPE,
    PROFILE_ASSET_TYPES,
    PROFILE_EFFECT_VALIDATORS,
    REFLECTION_RULE_TYPE,
    TREATMENT_PROFILE_TYPE,
)

__all__ = [
    "EXAM_PROFILE_TYPE",
    "TREATMENT_PROFILE_TYPE",
    "REFLECTION_RULE_TYPE",
    "PROFILE_ASSET_TYPES",
    "VerifiedProfileIndex",
    "validate_verified_profile_asset",
    "verified_treatment_profile_evidence",
]


def validate_verified_profile_asset(candidate_type: Any, content: Any) -> dict[str, Any]:
    """Re-validate a frozen asset; unknown types and bad content fail closed."""
    validator = PROFILE_EFFECT_VALIDATORS.get(candidate_type)
    if validator is None:
        raise ValueError("unsupported verified profile type: %r" % (candidate_type,))
    if not validator(content):
        raise ValueError("invalid %s content in frozen registry" % candidate_type)
    return deepcopy(dict(content))


class VerifiedProfileIndex:
    """Read-only index over frozen profile assets, keyed by exact diagnosis name."""

    def __init__(self, assets: Sequence[Mapping[str, Any]] = ()) -> None:
        self._exam: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._treatment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._reflections: list[dict[str, Any]] = []
        for asset in assets or ():
            if not isinstance(asset, Mapping):
                raise ValueError("verified profile asset must be an object")
            candidate_type = asset.get("candidate_type")
            content = validate_verified_profile_asset(candidate_type, asset.get("content"))
            if candidate_type == EXAM_PROFILE_TYPE:
                self._exam[content["diagnosis_name"]].append(content)
            elif candidate_type == TREATMENT_PROFILE_TYPE:
                self._treatment[content["diagnosis_name"]].append(content)
            else:
                self._reflections.append(content)

    def __bool__(self) -> bool:
        return bool(self._exam or self._treatment or self._reflections)

    @property
    def exam_profile_count(self) -> int:
        return sum(len(items) for items in self._exam.values())

    @property
    def treatment_profile_count(self) -> int:
        return sum(len(items) for items in self._treatment.values())

    @property
    def reflection_rule_count(self) -> int:
        return len(self._reflections)

    def exam_profiles(self, diagnoses: Sequence[str]) -> list[dict[str, Any]]:
        """Exact-name lookup only; no fuzzy or partial diagnosis matching."""
        return deepcopy(
            [
                profile
                for diagnosis in diagnoses or ()
                for profile in self._exam.get(diagnosis, ())
            ]
        )

    def treatment_profiles(self, diagnoses: Sequence[str]) -> list[dict[str, Any]]:
        return deepcopy(
            [
                profile
                for diagnosis in diagnoses or ()
                for profile in self._treatment.get(diagnosis, ())
            ]
        )

    def profile_examination_names(
        self,
        diagnoses: Sequence[str],
        *,
        limit: int = 12,
    ) -> list[str]:
        """Ranked catalog leaf names, deduplicated across the given diagnoses."""
        ordered: list[str] = []
        seen: set[str] = set()
        for profile in self.exam_profiles(diagnoses):
            for item in profile.get("exam_items") or ():
                name = str(item.get("name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return ordered[: max(0, int(limit))]

    def treatment_goal_codes(self, diagnoses: Sequence[str]) -> list[str]:
        codes: list[str] = []
        for profile in self.treatment_profiles(diagnoses):
            for code in profile.get("goal_codes") or ():
                if code not in codes:
                    codes.append(str(code))
        return codes

    def treatment_risk_codes(self, diagnoses: Sequence[str]) -> list[str]:
        codes: list[str] = []
        for profile in self.treatment_profiles(diagnoses):
            for code in profile.get("risk_codes") or ():
                if code not in codes:
                    codes.append(str(code))
        return codes

    def treatment_contraindication_codes(self, diagnoses: Sequence[str]) -> list[str]:
        codes: list[str] = []
        for profile in self.treatment_profiles(diagnoses):
            for code in profile.get("contraindication_codes") or ():
                if code not in codes:
                    codes.append(str(code))
        return codes

    def reflection_notes(
        self,
        *,
        trigger_codes: Collection[str],
        stage: str,
        limit: int = 3,
    ) -> list[str]:
        """Reflections need both a stage match and a trigger intersection."""
        trigger_set = {str(code) for code in trigger_codes or ()}
        stage_name = str(stage or "")
        if not trigger_set or not stage_name:
            return []
        notes: list[str] = []
        for rule in self._reflections:
            if stage_name not in (rule.get("stages") or ()):
                continue
            if not trigger_set.intersection(rule.get("trigger_codes") or ()):
                continue
            note = str(rule.get("note") or "").strip()
            if note and note not in notes:
                notes.append(note)
        return notes[: max(0, int(limit))]


# Human-readable review targets for each closed code. The text is a review
# instruction, never a prescription: no drug, dose or route appears here.
GOAL_CODE_REVIEW_TEXT = {
    "antiviral_therapy": "需确认是否已闭合抗病毒治疗目标",
    "antibacterial_therapy": "需确认是否已闭合抗细菌感染治疗目标",
    "analgesia_antipyresis": "需确认是否已闭合镇痛退热支持目标",
    "fluid_and_nutrition": "需确认是否已闭合补液与营养支持目标",
    "inpatient_monitoring": "需确认是否已闭合住院观察与监测目标",
    "surgical_evaluation": "需确认是否已闭合外科评估目标",
    "specialist_referral": "需确认是否已闭合专科会诊目标",
    "immunosuppressant_adjustment": "需确认是否已闭合免疫抑制方案调整目标",
    "wound_and_skin_care": "需确认是否已闭合创面与皮肤护理目标",
    "anticoagulation": "需确认是否已闭合抗凝管理目标",
}

RISK_CODE_REVIEW_TEXT = {
    "sepsis_risk": "需复核脓毒症风险相关监测与升级条件",
    "neurologic_involvement": "需复核神经系统受累的评估与随访",
    "secondary_bacterial_infection": "需复核继发细菌感染的防治与复评",
    "immunosuppression": "需复核免疫抑制状态下的用药安全",
    "bleeding_risk": "需复核出血风险相关监测",
    "airway_compromise": "需复核气道与呼吸支持条件",
}

CONTRAINDICATION_CODE_REVIEW_TEXT = {
    "systemic_corticosteroid": "需复核全身糖皮质激素的使用禁忌",
    "live_attenuated_vaccine": "需复核减毒活疫苗接种禁忌",
    "aspirin_in_children": "需复核儿童阿司匹林使用禁忌",
    "tetracycline_class": "需复核四环素类使用禁忌",
    "codeine_or_tramadol": "需复核可待因/曲马多使用禁忌",
    "nsaid": "需复核非甾体抗炎药使用禁忌",
    "methotrexate": "需复核甲氨蝶呤使用禁忌",
    "nephrotoxic_drug": "需复核肾毒性药物使用禁忌",
}

# Markers proving a goal is already addressed by the current plan text.
_GOAL_COVERAGE_MARKERS = {
    "antiviral_therapy": ("抗病毒", "阿昔洛韦", "伐昔洛韦"),
    "antibacterial_therapy": ("抗菌", "抗生素", "抗感染", "头孢", "青霉素"),
    "analgesia_antipyresis": ("退热", "镇痛", "对乙酰氨基酚", "布洛芬"),
    "fluid_and_nutrition": ("补液", "水分", "营养", "饮食"),
    "inpatient_monitoring": ("住院", "监测", "观察", "监护"),
    "surgical_evaluation": ("手术", "外科", "矫治", "切除"),
    "specialist_referral": ("专科", "会诊", "转诊"),
    "immunosuppressant_adjustment": ("免疫抑制", "泼尼松", "激素"),
    "wound_and_skin_care": ("皮肤", "创面", "换药"),
    "anticoagulation": ("抗凝", "华法林", "肝素"),
}


def _goal_is_covered(code: str, normalized_plan: str) -> bool:
    markers = _GOAL_COVERAGE_MARKERS.get(code, ())
    return any(marker in normalized_plan for marker in markers)


def verified_treatment_profile_evidence(
    profiles: Sequence[Mapping[str, Any]],
    *,
    treatment_plan: str = "",
    entry_factory: Any,
) -> list[dict[str, Any]]:
    """Turn closed profile codes into review targets, never into prescriptions."""
    normalized_plan = str(treatment_plan or "")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    index = 0
    for profile in profiles or ():
        if not isinstance(profile, Mapping):
            continue
        for code in profile.get("goal_codes") or ():
            text = GOAL_CODE_REVIEW_TEXT.get(str(code))
            if not text or code in seen or _goal_is_covered(str(code), normalized_plan):
                continue
            seen.add(code)
            index += 1
            entries.append(
                entry_factory(
                    evidence_id="profile_goal:%d" % index,
                    source="profile_goal",
                    text=text,
                    polarity="missing",
                )
            )
        for code in profile.get("risk_codes") or ():
            text = RISK_CODE_REVIEW_TEXT.get(str(code))
            if not text or code in seen:
                continue
            seen.add(code)
            index += 1
            entries.append(
                entry_factory(
                    evidence_id="profile_risk:%d" % index,
                    source="profile_risk",
                    text=text,
                    polarity="missing",
                )
            )
        for code in profile.get("contraindication_codes") or ():
            text = CONTRAINDICATION_CODE_REVIEW_TEXT.get(str(code))
            if not text or code in seen:
                continue
            seen.add(code)
            index += 1
            entries.append(
                entry_factory(
                    evidence_id="profile_contraindication:%d" % index,
                    source="profile_contraindication",
                    text=text,
                    polarity="missing",
                )
            )
    return entries


def verified_profile_risk_codes(
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Risk codes usable as reflection triggers, deduplicated and ordered."""
    codes: list[str] = []
    for profile in profiles or ():
        if not isinstance(profile, Mapping):
            continue
        for code in profile.get("risk_codes") or ():
            text = str(code).strip()
            if text and text not in codes:
                codes.append(text)
    return tuple(codes)
