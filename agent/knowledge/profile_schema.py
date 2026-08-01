"""Single source of truth for frozen profile / reflection asset schemas.

The runtime must be able to re-validate every asset a frozen release carries
without importing anything from ``offline`` (the architecture boundary only
allows ``offline -> agent``). The schema therefore lives here, and the offline
builders import it, so a candidate written offline and an asset read online can
never drift apart.

Nothing in this module reads the filesystem or aggregates data: it is pure
structural validation over closed vocabularies.
"""
from __future__ import annotations

import unicodedata
from types import MappingProxyType
from typing import Any, Mapping

EXAM_PROFILE_TYPE = "disease_exam_profile"
TREATMENT_PROFILE_TYPE = "disease_treatment_profile"
REFLECTION_RULE_TYPE = "reflection_rule"

PROFILE_ASSET_TYPES = frozenset(
    {EXAM_PROFILE_TYPE, TREATMENT_PROFILE_TYPE, REFLECTION_RULE_TYPE}
)

EXAM_PROFILE_SCHEMA = "disease-exam-profile/v1"
TREATMENT_PROFILE_SCHEMA = "disease-treatment-profile/v1"
REFLECTION_RULE_SCHEMA = "reflection-rule/v1"

EXAM_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "diagnosis_name",
        "exam_items",
        "support_case_count",
        "source_receipt_hash",
    }
)
EXAM_ITEM_FIELDS = frozenset({"name", "support_count", "support_ratio", "rank"})

TREATMENT_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "diagnosis_name",
        "goal_codes",
        "risk_codes",
        "contraindication_codes",
        "support_stats",
        "source_receipt_hash",
    }
)
SUPPORT_STATS_FIELDS = frozenset(
    {
        "support_case_count",
        "goal_support_counts",
        "risk_support_counts",
        "contraindication_support_counts",
    }
)

REFLECTION_RULE_FIELDS = frozenset(
    {
        "schema_version",
        "trigger_codes",
        "stages",
        "note",
        "source_refs",
        "support_count",
        "source_receipt_hash",
    }
)

# Closed codebooks. Free-text treatment sentences never become codes, so an
# unmappable phrase is quarantined offline instead of widening the vocabulary.
GOAL_CODEBOOK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "antiviral_therapy": ("阿昔洛韦", "伐昔洛韦", "抗病毒"),
        "antibacterial_therapy": ("头孢", "青霉素", "抗菌", "抗生素", "万古霉素"),
        "analgesia_antipyresis": ("对乙酰氨基酚", "退热", "镇痛", "布洛芬"),
        "fluid_and_nutrition": ("补液", "水分", "营养", "饮食"),
        "inpatient_monitoring": ("住院", "监测", "观察", "监护"),
        "surgical_evaluation": ("手术", "外科", "矫治", "切除"),
        "specialist_referral": ("专科", "会诊", "转诊"),
        "immunosuppressant_adjustment": ("免疫抑制", "泼尼松", "激素调整"),
        "wound_and_skin_care": ("皮肤清洁", "创面", "换药"),
        "anticoagulation": ("抗凝", "华法林", "肝素"),
    }
)

RISK_CODEBOOK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "sepsis_risk": ("脓毒", "败血", "菌血"),
        "neurologic_involvement": ("神经", "中枢", "脑"),
        "secondary_bacterial_infection": ("继发感染", "继发细菌", "细菌播散"),
        "immunosuppression": ("免疫抑制", "免疫功能低下"),
        "bleeding_risk": ("出血", "凝血"),
        "airway_compromise": ("呼吸衰竭", "气道", "窒息"),
    }
)

CONTRAINDICATION_CODEBOOK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "systemic_corticosteroid": ("糖皮质激素", "泼尼松", "地塞米松", "激素"),
        "live_attenuated_vaccine": ("减毒活疫苗", "活疫苗", "mmr", "bcg"),
        "aspirin_in_children": ("阿司匹林",),
        "tetracycline_class": ("四环素",),
        "codeine_or_tramadol": ("可待因", "曲马多"),
        "nsaid": ("非甾体", "布洛芬", "双氯芬酸"),
        "methotrexate": ("甲氨蝶呤",),
        "nephrotoxic_drug": ("氨基糖苷", "肾毒"),
    }
)

ALLOWED_STAGES = frozenset({"diagnosis", "examination", "treatment"})

ALLOWED_TRIGGER_CODES = frozenset(
    {
        "immunosuppressed_infection",
        "vesicular_rash",
        "fever",
        "noninfectious_eczema",
        "isolated_vesicle_without_systemic_risk",
        "neonate",
        "infant",
        "intrauterine_viral_exposure",
        "acute_limb_soft_tissue_infection",
        "hyperlipidemia_with_xanthelasma",
        "severe_pneumonia_aerosol_exposure",
        "pediatric_congenital_glaucoma",
        "high_energy_hindfoot_trauma",
        "suspected_sepsis",
        "bleeding_tendency",
        "confirmed_resistance",
        "drug_allergy",
    }
)

NOTE_MAX_UNICODE_CHARS = 160
NOTE_MAX_CJK_CHARS = 80
REFLECTION_MIN_SUPPORT_COUNT = 3


def valid_exam_profile_effect(effect: Any) -> bool:
    """Exact field whitelist so no answer-shaped key can ride along."""
    if not isinstance(effect, Mapping) or set(effect) != EXAM_PROFILE_FIELDS:
        return False
    if effect.get("schema_version") != EXAM_PROFILE_SCHEMA:
        return False
    if not isinstance(effect.get("diagnosis_name"), str) or not effect["diagnosis_name"].strip():
        return False
    if not isinstance(effect.get("source_receipt_hash"), str):
        return False
    support = effect.get("support_case_count")
    if not isinstance(support, int) or isinstance(support, bool) or support <= 0:
        return False
    items = effect.get("exam_items")
    if not isinstance(items, list) or not items:
        return False
    for index, item in enumerate(items):
        if not isinstance(item, Mapping) or set(item) != EXAM_ITEM_FIELDS:
            return False
        if not isinstance(item.get("name"), str) or not item["name"].strip():
            return False
        count = item.get("support_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            return False
        ratio = item.get("support_ratio")
        if not isinstance(ratio, float) or not 0.0 < ratio <= 1.0:
            return False
        if item.get("rank") != index + 1:
            return False
    return True


def _valid_code_list(value: Any, codebook: Mapping[str, tuple[str, ...]]) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item in codebook for item in value)
        and list(value) == sorted(value)
        and len(set(value)) == len(value)
    )


def _valid_support_counts(value: Any, codes: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(codes)
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in value.values()
        )
    )


def valid_treatment_profile_effect(effect: Any) -> bool:
    """Exact field whitelist plus closed codebook membership."""
    if not isinstance(effect, Mapping) or set(effect) != TREATMENT_PROFILE_FIELDS:
        return False
    if effect.get("schema_version") != TREATMENT_PROFILE_SCHEMA:
        return False
    diagnosis = effect.get("diagnosis_name")
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        return False
    goal_codes = effect.get("goal_codes")
    risk_codes = effect.get("risk_codes")
    contraindication_codes = effect.get("contraindication_codes")
    if not _valid_code_list(goal_codes, GOAL_CODEBOOK):
        return False
    if not _valid_code_list(risk_codes, RISK_CODEBOOK):
        return False
    if not _valid_code_list(contraindication_codes, CONTRAINDICATION_CODEBOOK):
        return False
    if not goal_codes and not contraindication_codes:
        return False
    if not isinstance(effect.get("source_receipt_hash"), str):
        return False
    stats = effect.get("support_stats")
    if not isinstance(stats, Mapping) or set(stats) != SUPPORT_STATS_FIELDS:
        return False
    support = stats.get("support_case_count")
    if not isinstance(support, int) or isinstance(support, bool) or support <= 0:
        return False
    if not _valid_support_counts(stats.get("goal_support_counts"), goal_codes):
        return False
    if not _valid_support_counts(stats.get("risk_support_counts"), risk_codes):
        return False
    if not _valid_support_counts(
        stats.get("contraindication_support_counts"), contraindication_codes
    ):
        return False
    return True


def clean_code_tuple(value: Any, allowed: frozenset[str]) -> tuple[str, ...] | None:
    """Sorted, deduplicated codes drawn only from a closed vocabulary."""
    if not isinstance(value, list) or not value:
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        text = item.strip()
        if not text or text not in allowed:
            return None
        items.append(text)
    if len(set(items)) != len(items):
        return None
    return tuple(sorted(items))


def valid_reflection_note(value: Any) -> bool:
    """Notes stay short so they cannot smuggle a full case answer."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text != value:
        return False
    if len(text) > NOTE_MAX_UNICODE_CHARS:
        return False
    cjk = sum(1 for char in text if unicodedata.east_asian_width(char) in {"W", "F"})
    return cjk <= NOTE_MAX_CJK_CHARS


def valid_reflection_rule_effect(effect: Any) -> bool:
    """Exact field whitelist plus closed trigger/stage vocabulary."""
    if not isinstance(effect, Mapping) or set(effect) != REFLECTION_RULE_FIELDS:
        return False
    if effect.get("schema_version") != REFLECTION_RULE_SCHEMA:
        return False
    if clean_code_tuple(effect.get("trigger_codes"), ALLOWED_TRIGGER_CODES) is None:
        return False
    if clean_code_tuple(effect.get("stages"), ALLOWED_STAGES) is None:
        return False
    if not valid_reflection_note(effect.get("note")):
        return False
    if not isinstance(effect.get("source_receipt_hash"), str):
        return False
    source_refs = effect.get("source_refs")
    if not isinstance(source_refs, list) or not source_refs:
        return False
    if not all(isinstance(item, str) and item.strip() for item in source_refs):
        return False
    if list(source_refs) != sorted(source_refs) or len(set(source_refs)) != len(source_refs):
        return False
    support = effect.get("support_count")
    if not isinstance(support, int) or isinstance(support, bool):
        return False
    return support == len(source_refs) and support >= REFLECTION_MIN_SUPPORT_COUNT


PROFILE_EFFECT_VALIDATORS: Mapping[str, Any] = MappingProxyType(
    {
        EXAM_PROFILE_TYPE: valid_exam_profile_effect,
        TREATMENT_PROFILE_TYPE: valid_treatment_profile_effect,
        REFLECTION_RULE_TYPE: valid_reflection_rule_effect,
    }
)


def registered_profile_code_values() -> frozenset[str]:
    """Every value a closed codebook or vocabulary may legitimately emit."""
    return frozenset(
        set(GOAL_CODEBOOK)
        | set(RISK_CODEBOOK)
        | set(CONTRAINDICATION_CODEBOOK)
        | set(ALLOWED_TRIGGER_CODES)
        | set(ALLOWED_STAGES)
    )
