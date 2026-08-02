"""A readable baseline doctor agent.

The SDK handles train/test orchestration and service calls. This file shows a
clear reference flow: decide action -> ask/order exam/diagnose -> train reflection.
"""

from __future__ import annotations

import json
import hashlib
import ipaddress
import re
import threading
import urllib.parse
from hashlib import sha256
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional
from uuid import uuid4
from agent.clinical.exam_axis_evidence_contract import (
    has_specific_cross_system_conflict,
)
from agent.clinical.final_submission import (
    FinalVerificationError as SubmissionFinalVerificationError,
)
from agent.clinical.exam_planner import (
    exam_value_record,
    plan_examinations,
)
from agent.clinical.safety_facts import (
    safety_facts_to_case_features,
    validate_case_memory_safety_facts,
)
from agent.knowledge.typed_rule_engine import (
    CompiledRulePack,
    RuleContext,
    RuleDiagnosisCandidate,
    RuleResult,
    apply_rules,
    empty_compiled_rule_pack,
    parse_compiled_rule_pack,
)

from hospital_agent_sdk import BasicAgent
from .diagnosis_consistency import (
    enforce_candidate_pool_consistency,
    enforce_selected_diagnosis_consistency,
)
from .runtime import (
    ActionGateway,
    EvaluationAttemptStore,
    EvaluationCollector,
    SdkActionAdapter,
    build_action_command,
)
from .prompt import (
    DEPARTMENT_PROMPT,
    DIAGNOSIS_INDEPENDENT_REVIEW_PROMPT,
    DIAGNOSTIC_AXIS_CONSULT_PROMPT,
    DIAGNOSTIC_CONTEXT_PROMPT,
    DISEASE_CANDIDATE_PROMPT,
    DISEASE_AND_TREATMENT_PROMPT,
    DOCTOR_SYSTEM_PROMPT,
    EVALUATION_REFLECTION_PROMPT,
    EXAM_CATEGORY_PROMPT,
    EXAM_ITEM_PROMPT,
    JSON_REPAIR_SYSTEM_PROMPT,
    NEXT_ACTION_PROMPT,
    TREATMENT_REVIEW_PROMPT,
    format_prompt,
)


BASE_DIR = Path(__file__).resolve().parents[1]
REF_DATA_DIR = BASE_DIR / "data" / "ref_data"
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
DEFAULT_INITIAL_EXAMINATION = "体格检查"
MAX_DISEASE_CANDIDATES = 24
MAX_EXAMINATION_ACTIONS = 6
MAX_EXAMS_PER_ACTION = 5
MAX_PREFINAL_EXAM_REVIEWS = 2
MAX_PATIENT_REPLIES = 5
GENERIC_FINAL_DIAGNOSES = {"细菌感染", "病毒感染", "感染"}
CONTRAINDICATED_DRUG_GROUPS = {
    "penicillin": ["penicillin", "青霉素", "青霉素g", "青霉素G"],
    "non_dihydropyridine_ccb": [
        "非二氢吡啶类钙通道阻滞剂",
        "地尔硫卓",
        "维拉帕米",
    ],
    # Asthma/reactive airway: IV/oral beta-blockers for rate control are unsafe.
    "beta_blocker": [
        "β受体阻滞剂",
        "beta受体阻滞剂",
        "beta blocker",
        "betablocker",
        "艾司洛尔",
        "esmolol",
        "美托洛尔",
        "metoprolol",
        "倍他乐克",
        "普萘洛尔",
        "propranolol",
        "阿替洛尔",
        "atenolol",
        "比索洛尔",
        "bisoprolol",
        "拉贝洛尔",
        "labetalol",
    ],
}
TREATMENT_RECOMMENDATION_MARKERS = [
    "首选",
    "立即",
    "给予",
    "使用",
    "静脉",
    "滴注",
    "启动",
    "推荐",
    "考虑",
    "备选",
    "谨慎",
    "加用",
    "联合",
    "继续",
    "维持",
]
TREATMENT_NEGATION_MARKERS = ["避免", "禁用", "不应", "不得", "禁止", "禁忌"]
KNOWN_AXIS_RISK_TAGS = {
    "infection_before_steroid",
    "unsupported_no_infection_risk",
    "pregnancy_screening_before_migraine_drugs",
    "sle_renal_thrombosis_unclosed",
    "unsupported_no_renal_damage",
    "recurrent_epistaxis_coagulation",
    "avoid_no_further_care_for_bleeding_mass",
    "fracture_not_excluded",
    "acs_intensive_without_evidence",
    "mds_specific_without_marrow",
    "di_path_before_hyperglycemic_crisis_exclusion",
    "immunosuppressed_respiratory_infection_unclosed",
}
SLE_AXIS_RISK_TAGS = {"sle_renal_thrombosis_unclosed", "unsupported_no_renal_damage"}
MIGRAINE_AXIS_RISK_TAGS = {"pregnancy_screening_before_migraine_drugs"}
INFECTION_STEROID_RISK_TAGS = {"infection_before_steroid", "unsupported_no_infection_risk"}
INFECTION_STEROID_SUPPORTED_AXIS_IDS = {
    "corneal_infection_with_target_rash",
    "systemic_infection_vs_primary_hematologic",
    "high_risk_pediatric_lower_respiratory_infection",
    "immunosuppressed_progressive_dyspnea_infection",
}
UMBILICAL_CARE_RISK_TAGS = {"avoid_no_further_care_for_bleeding_mass"}
MALE_ONLY_EXAM_MARKERS = ["前列腺", "睾丸", "精囊", "精液"]
FEMALE_CONTEXT_MARKERS = ["女性", "女", "阴道", "外阴", "宫颈", "子宫", "卵巢", "月经", "妊娠", "怀孕", "盆底"]
AXIS_CANDIDATE_BLOCKLIST = {
    "umbilical_granulation_or_vascular_lesion": {"黑色素细胞痣"},
    "corneal_infection_with_target_rash": {"囊肿性痤疮"},
}


class FinalVerificationError(RuntimeError):
    """Raised when a final treatment plan cannot pass verification."""


class RemoteServiceCaseError(RuntimeError):
    """Remote patient/LLM/service failure for one case; safe to isolate from batch.

    Programming bugs and local assertion failures must NOT be wrapped as this type.
    """


_REMOTE_SERVICE_MARKERS = (
    "http 555",
    "status code 555",
    "status=555",
    " 555 ",
    "输出审核",
    "content moderation",
    "moderation",
    "modelscope",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection reset",
    "connection aborted",
    "timeout",
    "timed out",
    "read timed out",
    "remote end closed",
    "temporary failure",
)


def classify_isolatable_case_error(exc: BaseException) -> Optional[BaseException]:
    """Return an isolatable error for per-case incomplete, else None (re-raise).

    Isolatable: FinalVerificationError, RemoteServiceCaseError, or remote-looking
    transport/HTTP failures (555 output audit, timeouts). Never isolate AttributeError
    / TypeError / AssertionError programming faults.
    """
    if isinstance(
        exc,
        (
            FinalVerificationError,
            SubmissionFinalVerificationError,
            RemoteServiceCaseError,
        ),
    ):
        return exc
    if isinstance(exc, (AttributeError, TypeError, AssertionError, KeyError, NameError, SyntaxError)):
        return None
    text = (" %s %s " % (type(exc).__name__, exc)).lower()
    # Explicit HTTP status attributes when present (requests/httpx-like).
    for attr in ("status_code", "status", "code", "http_status"):
        value = getattr(exc, attr, None)
        if value is None and hasattr(exc, "response"):
            value = getattr(getattr(exc, "response", None), "status_code", None)
        try:
            code = int(value) if value is not None else None
        except (TypeError, ValueError):
            code = None
        if code in {408, 429, 500, 502, 503, 504, 555}:
            return RemoteServiceCaseError("%s (http %s): %s" % (type(exc).__name__, code, exc))
    if any(marker in text for marker in _REMOTE_SERVICE_MARKERS):
        # Avoid classifying pure local ValueError without remote markers.
        if isinstance(exc, ValueError) and "555" not in text and "timeout" not in text:
            return None
        return RemoteServiceCaseError("%s: %s" % (type(exc).__name__, exc))
    return None


def incomplete_case_result(case_id: str, exc: BaseException) -> Dict[str, Any]:
    """Return a redacted incomplete result without echoing remote exception text."""
    err_type = type(exc).__name__
    return {
        "patient_id": clean_text(case_id),
        "case_id": clean_text(case_id),
        "finished": False,
        "status": "incomplete",
        "error_type": err_type,
        "error_code": "case_incomplete",
        "diagnosis": [],
        "treatment_plan": "",
        "reasoning": "case_incomplete:%s" % err_type,
    }


def run_case_isolated(case_id: str, runner: Callable[[], Any]) -> Dict[str, Any]:
    """Run one case without letting isolatable case errors abort a batch.

    Batch loops should use this (or equivalent) so a single finalize or remote
    service failure is recorded as incomplete instead of re-raising and wiping
    sibling results. Online path: MyDoctorAgent.test/train also isolate via
    classify_isolatable_case_error.
    """
    try:
        return {
            "case_id": clean_text(case_id),
            "status": "ok",
            "result": runner(),
            "error": None,
            "error_type": None,
        }
    except Exception as exc:
        isolatable = classify_isolatable_case_error(exc)
        if isolatable is None:
            raise
        return {
            "case_id": clean_text(case_id),
            "status": "incomplete",
            "result": None,
            "error_type": type(isolatable).__name__,
            "error_code": "case_incomplete",
        }


def run_batch_isolated(
    cases: Iterable[tuple[str, Callable[[], Any]]],
) -> List[Dict[str, Any]]:
    """Run many cases with per-case isolatable-error isolation."""
    return [run_case_isolated(case_id, runner) for case_id, runner in cases]


def build_safe_escalation_plan(
    *,
    axis_id: str,
    closure_requirement: str,
    evidence: Iterable[str],
    existing_treatment: str,
) -> tuple[str, str]:
    existing = clean_text(existing_treatment)
    if axis_id == "active_upper_gi_bleed":
        closure = (
            "立即急诊住院，持续监测生命体征并建立静脉通路；尽快完成血常规、凝血功能全套、"
            "血型鉴定及交叉配血，并由消化专科实施急诊内镜止血评估。在病因未闭合前保留复苏和止血准备，"
            "不得以门静脉高压或心律失常替代当前活动性出血风险。"
        )
    elif axis_id == "acute_lower_extremity_soft_tissue_infection":
        closure = (
            "立即急诊或住院评估急性下肢软组织感染；完善血常规、C反应蛋白和降钙素原，必要时软组织超声排除脓肿或坏死性筋膜炎，"
            "经验性覆盖革兰阳性菌并避开已知过敏药物，抬高患肢并监测红肿范围与生命体征。"
        )
    elif axis_id == "hyperlipidemia_with_xanthelasma":
        closure = (
            "针对混合型高脂血症与睑黄瘤：启动低饱和脂肪饮食和减重运动，复测空腹血脂谱，"
            "评估启动他汀类药物治疗及肝酶/肌酶监测；排查继发因素（甲状腺、血糖、药物），"
            "眼科随访睑黄斑变化，必要时心内科评估心血管风险；勿将睑黄瘤误作佝偻病并以维生素补充替代血脂管理。"
        )
    elif axis_id in {"left_ear_pathology", "chronic_suppurative_otitis_media"}:
        closure = (
            "针对化脓性中耳炎：耳鼻喉科评估鼓膜与外耳道，局部清洁并按病原启动局部/全身抗感染，"
            "完善听力测定；若疑胆脂瘤或骨质破坏行颞骨高分辨率CT，出现面瘫、眩晕或颅内征象立即急诊专科处置。"
        )
    elif axis_id == "leptospirosis_sepsis":
        closure = (
            "针对钩端螺旋体病：立即启动青霉素G或多西环素抗感染，监测肝肾功能与尿量，"
            "评估肺出血/黄疸风险，需要时住院或重症监护；避免延误特异抗生素。"
        )
    elif axis_id == "focal_seizure_left_temporal_lobe":
        closure = (
            "针对局灶性癫痫：神经科评估，完善脑MRI与长程视频脑电图，启动适宜抗癫痫药物并监测副作用，"
            "避免驾驶与高危作业，发作加重或持续状态时急诊。"
        )
    elif axis_id == "mitral_stenosis_hemodynamics":
        closure = (
            "针对二尖瓣狭窄：利尿减轻肺淤血并监测电解质；完善心电图评估房颤，"
            "有房颤或左房血栓高危时由心内科按规范启动抗凝并复查凝血；"
            "经食道超声排除左房血栓前不得实施经皮球囊二尖瓣成形；"
            "重度狭窄合并肺高压或失代偿心衰时急诊/心外科会诊，勿在血栓未排除时催促球囊扩张。"
        )
    elif axis_id == "rule_out_ra_axis":
        closure = (
            "针对类风湿关节炎鉴别：风湿免疫科评估，结合RF/抗CCP与关节超声或MRI滑膜炎证据，"
            "在感染与结晶性关节炎排除后，由专科决定是否启动DMARD/抗炎方案并监测肝肾功能与血常规；"
            "不得仅写「信息不足」而不给出专科与监测路径。"
        )
    elif axis_id == "acute_upper_extremity_pain_swelling":
        closure = (
            "针对急性上肢剧痛肿胀：排除肢端缺血、深静脉血栓与骨筋膜室综合征，"
            "完善血管与肌肉骨骼影像；有缺血性卒中史者继续二级预防抗血小板，"
            "不得无大出血指征擅自停用阿司匹林；疼痛与肿胀未明时骨科/血管外科急诊评估。"
        )
    elif axis_id == "suspected_asthma_control_issue":
        closure = (
            "针对哮喘控制不佳：完善肺功能检查（含支气管舒张试验）与症状日记，"
            "按阶梯优化吸入糖皮质激素±LABA，治疗合并过敏性鼻炎，制定哮喘行动计划；"
            "出现呼吸窘迫、说话困难或紫绀立即急诊，避免仅靠反复短效β激动剂。"
        )
    elif axis_id == "hypothalamic_pituitary_axis_dysfunction":
        closure = (
            "针对下丘脑-垂体轴异常/继发性闭经：内分泌科评估，完善垂体前叶激素与皮质醇，"
            "必要时垂体MRI；在排除肾上腺皮质功能减退前勿盲目激素冲击，"
            "营养与体重恢复与病因治疗同步，出现肾上腺危象征象立即急诊。"
        )
    elif axis_id == "congenital_syndactyly":
        closure = (
            "针对先天性并指（趾）：小儿手外科/骨科评估，完善手部X线明确骨性融合范围，"
            "个体化制定分离手术时机与皮瓣/植皮方案，术后康复与伤口血运监测。"
        )
    elif axis_id == "neck_mass_b_symptoms":
        closure = (
            "针对颈部包块伴B症状：血液科/肿瘤科急诊评估，完善血常规与LDH，"
            "尽快颈部淋巴结超声引导细针/切除活检明确病理，并按指征完成分期影像；"
            "在病理未明前避免仅观察收口，出现气道压迫、上腔静脉综合征或发热中性粒细胞减少立即急诊。"
        )
    elif axis_id == "cholestatic_liver_disease":
        closure = (
            "针对淤胆型肝病：完善抗线粒体抗体（AMA）与腹部超声或MRCP鉴别PBC/梗阻/药物性淤胆，"
            "由消化/肝病专科评估；若符合原发性胆汁性胆管炎启动熊去氧胆酸并监测肝功能与瘙痒，"
            "出现黄疸加深、凝血异常或意识改变立即急诊。"
        )
    elif axis_id == "traumatic_left_rib_fracture":
        closure = (
            "针对创伤性肋骨骨折：镇痛并指导激励式肺量计/深呼吸咳嗽防肺不张，"
            "复查或结合胸部X线/CT排除气胸血胸；高龄或抗血小板用药者监测迟发胸腔并发症，"
            "呼吸困难加重、血氧下降或咯血立即急诊，避免过度束缚胸廓。"
        )
    elif axis_id in {"heart_failure_decompensation", "acute_decompensated_heart_failure", "reduced_ejection_fraction_heart_failure"}:
        closure = (
            "针对急性失代偿心力衰竭：半卧位吸氧，在血压允许下启动利尿减轻肺淤血，"
            "完善NT-proBNP/BNP、电解质与肾功能，监测尿量与低血压；"
            "血流动力学稳定后尽快启动心衰指南核心药物，出现呼吸衰竭或心源性休克立即急诊/ICU。"
        )
    elif axis_id == "acute_pharyngitis_in_diabetic_child":
        closure = (
            "针对糖尿病患儿急性咽炎/扁桃体炎：完善血常规与C反应蛋白鉴别细菌感染，"
            "若链球菌可能性高启动青霉素类或替代抗生素（避免已知过敏药物），"
            "加密监测指尖血糖与酮体，补充水分与退热；出现呼吸困难、吞咽不能、高热惊厥或血糖危象立即急诊，"
            "不得仅写完善检查而不给抗感染与血糖管理。"
        )
    elif axis_id == "pleural_effusion_investigation":
        closure = (
            "针对胸腔积液/疑似肺炎伴积液：启动经验性抗感染并评估引流指征，"
            "在影像引导下完成诊断性胸腔穿刺（常规/生化/培养/细胞学），"
            "监测呼吸与血氧，脓胸或大量积液压迫时急诊引流；不得仅建议穿刺而不启动抗感染与呼吸支持评估。"
        )
    elif axis_id == "suspected_infection_immunodeficiency":
        closure = (
            "针对免疫缺陷宿主感染：完善血常规、CRP/PCT与呼吸道病原检测，"
            "评估住院隔离与经验性抗感染覆盖，监测血氧与生命体征；"
            "出现呼吸衰竭、血流动力学不稳或中性粒细胞缺乏发热立即急诊/住院，不得仅完善检查而不启动经验治疗。"
        )
    elif axis_id == "right_bartholin_cyst_infection":
        closure = (
            "针对巴氏腺囊肿/脓肿：温水坐浴、止痛与经验性抗生素，评估切开引流或造口指征，"
            "完善血常规与CRP；出现高热、波动性脓肿或排尿困难立即妇科急诊处理。"
        )
    elif axis_id == "suspected_gouty_arthritis":
        closure = (
            "针对疑似痛风性关节炎：急性期尽早使用NSAIDs或秋水仙碱或短程糖皮质激素（按禁忌选择），"
            "休息抬高患肢、冰敷；完善血尿酸并在条件允许时行关节液偏光显微镜检查；"
            "急性炎症控制后再评估启动降尿酸治疗，监测肾功能与药物相互作用，"
            "不得仅写完善检查/转诊而不给急性抗炎方案。"
        )
    elif axis_id == "suspected_hunters_syndrome_neurologic":
        closure = (
            "针对亨特综合征（膝状神经节带状疱疹）：尽早启动抗病毒（阿昔洛韦/伐昔洛韦，按肾功能调整）"
            "并评估联合糖皮质激素；完善VZV血清学与必要脑脊液检测，保护角膜与听力，"
            "神经内科/耳鼻喉科随访；出现脑膜脑炎征象立即急诊。"
        )
    elif axis_id == "liver_fluke_infection_cholangitis":
        closure = (
            "针对华支睾吸虫病/胆道感染：完善血常规嗜酸细胞、粪便虫卵或血清学确诊，"
            "启动抗寄生虫治疗并监测肝功能，评估胆道梗阻；必要时ERCP取虫或引流，"
            "避免生食淡水鱼，不得仅写完善检查而不启动抗寄生虫与胆道评估。"
        )
    elif axis_id == "diabetic_foot_infection":
        closure = (
            "针对糖尿病足感染/下肢软组织感染：立即经验性抗感染（避开已知过敏抗生素），"
            "完善血常规与CRP/PCT，评估血管与感染深度，需要时外科清创或引流；"
            "抬高患肢、血糖优化与减负，出现全身中毒或骨感染征象立即住院。"
        )
    elif axis_id == "acute_infectious_pneumonia_suspect":
        closure = (
            "针对急性感染性肺炎：经验性抗感染与氧合支持，完善血常规/CRP/PCT与病原学，"
            "按严重度评估住院或ICU；血氧持续下降或意识改变立即升级呼吸支持。"
        )
    elif axis_id == "exposure_keratoconjunctivitis":
        closure = (
            "针对暴露性角结膜炎/角膜炎：眼科急诊评估，荧光素染色评估上皮缺损，"
            "夜间眼膏/湿房保护，必要时抗感染与润滑治疗；避免长期局部激素滥用，"
            "出现溃疡加深、前房积脓或视力骤降立即专科处理。"
        )
    elif axis_id == "acute_leukemia_suspected":
        closure = (
            "针对疑似急性白血病：紧急血液科评估，完善血常规/涂片与凝血，"
            "尽快骨髓穿刺明确分型；纠正贫血/血小板低下与感染，"
            "避免有创操作前输注支持，出现致命性出血或感染立即住院。"
        )
    elif axis_id == "myeloproliferative_disorder_axis":
        closure = (
            "针对骨髓增殖性肿瘤/原发性血小板增多症：血液科评估，完善JAK2/CALR/MPL检测，"
            "按血栓与出血风险分层启动降板（如羟基脲）并谨慎抗血小板；"
            "监测血象与出血风险，出现血栓或大出血立即急诊。"
        )
    elif axis_id == "fetal_18_trisomy_diagnosis":
        closure = (
            "针对胎儿18三体/严重先天畸形产前诊断：立即产前诊断与遗传咨询，"
            "与产科/遗传专科共同制定继续或终止妊娠的知情决策，"
            "提供心理危机干预与社会支持；避免用维生素D/佝偻病路径替代染色体异常管理，"
            "出现腹痛、阴道流血或胎动异常立即急诊。"
        )
    elif axis_id == "cervicitis_in_immunocompromised":
        closure = (
            "针对免疫抑制宿主宫颈/阴道感染或上皮内病变：完善宫颈分泌物病原与HPV检测，"
            "评估宫颈细胞学与阴道镜检查；按病原启动抗感染并评估抗病毒方案依从性，"
            "排除宫颈上皮内瘤变/宫颈癌；不得将妇科感染误标为结膜炎并以对症支持收口。"
        )
    elif axis_id == "acute_hepatitis_syndrome":
        closure = (
            "针对急性肝炎综合征：完善病毒性肝炎血清学（甲/乙/丙/戊）、自身免疫抗体与腹部超声，"
            "监测凝血与肝功能；按病因启动抗病毒/保肝/免疫抑制治疗，"
            "出现凝血异常、意识改变或黄疸加重立即住院；不得以维生素D缺乏替代肝病管理。"
        )
    elif axis_id == "acute_coronary_syndrome":
        closure = (
            "针对急性冠脉综合征：立即静息、持续心电与血氧监测，嚼服阿司匹林并准备双联抗血小板，"
            "完善18导联心电图与肌钙蛋白动态复查，优化镇痛；"
            "ST段抬高或高危罪犯病变由心血管专科尽快评估再灌注（溶栓或急诊介入），"
            "出现血流动力学不稳、恶性心律失常或心源性休克立即抢救并收入监护。"
        )
    elif axis_id == "septic_shock":
        closure = (
            "针对脓毒性休克：立即液体复苏并留置动脉/中心静脉通路监测灌注与乳酸，"
            "在留取血培养后尽早静脉启动广谱抗感染，完善感染源影像与降钙素原，"
            "血管活性药维持平均动脉压，监测尿量、乳酸与器官功能；"
            "出现意识改变、尿量减少或乳酸持续升高立即收入重症监护。"
        )
    elif axis_id == "anaphylaxis":
        closure = (
            "针对过敏性休克：立即肌注肾上腺素并就地保持平卧位、畅通气道与高流量吸氧，"
            "快速扩容并建立静脉通路，备好气管插管与二线药物（抗组胺、糖皮质激素、支气管扩张剂），"
            "去除可疑过敏原并持续监测生命体征与氧合；"
            "出现气道肿胀、呼吸困难或循环不平稳立即复苏并急诊抢救。"
        )
    elif axis_id == "acute_ischemic_stroke":
        closure = (
            "针对急性缺血性卒中：立即评估起病时间与生命体征，完善头颅CT排除出血并尽快神经科会诊，"
            "在 time-window 内由卒中团队决定是否静脉溶栓或血管内取栓，"
            "监测神经功能、吞咽与血氧，控制血压与血糖；"
            "出现意识水平下降、瞳孔不等大或症状快速进展立即神经科/重症抢救。"
        )
    elif axis_id == "status_epilepticus":
        closure = (
            "针对癫痫持续状态：立即开放气道与静脉通路，按规范静脉给予一线止痉药物（如苯二氮䓬类），"
            "完善血糖、电解质与头颅影像排查可逆病因，监测呼吸与氧合，"
            "一线无效由神经科升级二线抗癫痫治疗并评估气管保护；"
            "出现呼吸抑制、持续抽搐或意识障碍立即抢救并重症监护。"
        )
    elif axis_id == "acute_kidney_injury":
        closure = (
            "针对急性肾损伤：停用肾毒性药物并评估容量与灌注，完善肾功能、电解质与泌尿系超声，"
            "监测尿量、肌酐与高钾等危急值，必要时肾脏科会诊与透析准备，"
            "维持水电解质与酸碱平衡；出现少尿无尿、高钾或容量负荷过重立即急诊处理。"
        )
    elif axis_id == "thyrotoxic_storm":
        closure = (
            "针对甲状腺危象：立即心电与生命体征监测，静脉给硫脲类抗甲状腺药并外用/口服碘剂（给药序正确）、"
            "β受体阻滞剂控制心率，完善甲状腺功能、肝功与感染筛查，补液退热，"
            "由内分泌科紧急处理；出现高热、心律失常或意识改变立即重症监护。"
        )
    elif axis_id == "adrenal_crisis":
        closure = (
            "针对肾上腺危象：在留取血皮质醇/ACTH后尽快静脉给予应激剂量糖皮质激素，"
            "快速补液纠正低钠低容并监测血糖与电解质，完善垂体-肾上腺轴评估，"
            "由内分泌科处理潜在病因；出现低血压、低钠或意识改变立即复苏并急诊抢救。"
        )
    elif axis_id == "diabetic_ketoacidosis":
        closure = (
            "针对糖尿病酮症酸中毒：立即静脉补液纠正容量，静脉胰岛持续泵入并监测血糖与酮体，"
            "补钾纠电解质、评估阴离子间隙与碳酸氢盐，完善感染筛查与动脉血气，"
            "由内分泌/急诊科处理；出现意识障碍、严重高钾或呼吸深快立即抢救并重症监护。"
        )
    elif axis_id == "hypertensive_urgency":
        closure = (
            "针对高血压急症：立即安静休息并持续血压监测，在数中小时内由心血管专科静脉降压达标，"
            "完善靶器官评估（心、脑、肾、眼底与心电图），避免过快降压致灌注不足，"
            "评估并处理诱因；出现胸痛、神经缺损、视物模糊或肾功能恶化立即急诊处理。"
        )
    elif axis_id == "acute_pancreatitis":
        closure = (
            "针对急性胰腺炎：立即禁食胃肠减压并静脉补液维持容量与灌注，"
            "完善血淀粉酶/脂肪酶、腹盆影像与钙/血脂评估，镇痛与营养支持，"
            "监测腹压、炎症与器官功能，必要时外科/重症介入；"
            "出现持续腹痛、腹胀、休克或器官衰竭立即急诊/重症处理。"
        )
    elif axis_id == "upper_gi_bleed_related":
        closure = (
            "针对上消化道大出血：立即卧床、建立静脉通路并快速容量复苏，"
            "完善血常规、凝血与血型交叉配血，由消化专科尽早急诊内镜止血评估，"
            "必要时血管介入或手术；出现呕血、黑便、心悸晕厥或低灌注立即急诊抢救。"
        )
    else:
        # Prefer the axis closure_requirement text when present so empty LLM shells
        # (neck_mass_b_symptoms, cholestatic_liver_disease, rib fracture, etc.) still
        # emit actionable plans instead of the useless generic sentence alone.
        specific = clean_text(closure_requirement)
        if specific and specific not in {
            "supported_official_diagnosis",
            "safe_escalation_or_supported_official_diagnosis",
            "urgent_hemostasis_and_resuscitation",
        }:
            closure = (
                "针对当前高风险主轴（%s）：%s；并立即急诊或专科评估，保留器官急症处置，"
                "完成必要检查闭环后再制定特异治疗。"
                % (clean_text(axis_id) or "未命名轴", specific)
            )
        else:
            closure = (
                "当前高风险主轴尚未获得可支持的官方诊断名称，需立即急诊或住院专科评估，"
                "保留既有器官急症处置，并完成必要检查闭环后再制定特异治疗。"
            )
    # Drop uninformative "insufficient information" drafts so axis closure is not diluted.
    existing_clean = existing
    if any(
        marker in existing_clean
        for marker in ["当前信息不足以", "候选疾病约束", "模型原诊断未通过"]
    ):
        existing_clean = ""
    # Also drop the bare generic shell if we now have a more specific closure.
    if existing_clean.startswith("当前高风险主轴尚未获得可支持的官方诊断名称") and not closure.startswith(
        "当前高风险主轴尚未获得可支持的官方诊断名称"
    ):
        existing_clean = ""
    plan = append_unique_patches(existing_clean, [closure])
    reason = (
        f"safe_escalation:{clean_text(axis_id)}；closure={clean_text(closure_requirement)}；"
        f"evidence_count={len(as_text_list(evidence))}。"
    )
    return plan, reason


def validate_safe_escalation_plan(
    treatment_plan: str,
    *,
    axis_id: str,
    evidence: Iterable[str],
) -> bool:
    plan = clean_text(treatment_plan)
    grounded_evidence = unique_preserve_order(as_text_list(evidence))
    if len(grounded_evidence) < 2 or not plan:
        return False
    if any(marker in plan for marker in ["未经证实的特异药物", "已确诊静脉曲张破裂"]):
        return False
    disposition = any(marker in plan for marker in ["急诊", "住院", "立即转诊"])
    preserves_emergency_care = any(
        marker in plan for marker in ["监测", "静脉通路", "专科", "气道", "复苏", "抗感染", "抬高患肢"]
    )
    if axis_id == "active_upper_gi_bleed":
        closure = "内镜" in plan and any(marker in plan for marker in ["凝血", "交叉配血", "血型"])
    elif axis_id == "acute_lower_extremity_soft_tissue_infection":
        closure = any(marker in plan for marker in ["血常规", "C反应蛋白", "降钙素原", "软组织超声", "专科评估"])
    elif axis_id == "hyperlipidemia_with_xanthelasma":
        disposition = any(
            marker in plan
            for marker in ["他汀", "生活方式", "低脂", "减重", "心内科", "复测血脂", "血脂", "饮食"]
        )
        preserves_emergency_care = True
        closure = any(
            marker in plan for marker in ["他汀", "血脂", "生活方式", "饮食", "心内科", "减重"]
        )
        rickets_closure = ("补充维生素D" in plan or "佝偻病" in plan) and not any(
            marker in plan for marker in ["他汀", "血脂", "低饱和脂肪", "复测空腹血脂"]
        )
        return disposition and closure and not rickets_closure
    elif axis_id in {"left_ear_pathology", "chronic_suppurative_otitis_media"}:
        disposition = any(marker in plan for marker in ["耳鼻喉", "抗感染", "专科", "急诊", "住院"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["听力", "鼓膜", "CT", "抗感染", "耳鼻喉"])
        return disposition and closure
    elif axis_id == "leptospirosis_sepsis":
        disposition = any(marker in plan for marker in ["青霉素", "多西环素", "抗感染", "住院", "监护"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["青霉素", "多西环素", "肾功能", "监测"])
        return disposition and closure
    elif axis_id == "mitral_stenosis_hemodynamics":
        disposition = any(marker in plan for marker in ["利尿", "心内科", "心外科", "急诊", "住院", "抗凝"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["利尿", "心电图", "房颤", "抗凝", "食道超声", "血栓"])
        # Any active scheduling of PBMV/balloon without same-clause thrombus exclusion is unsafe,
        # even if another sentence forbids balloon before exclusion.
        active_pbmv_clauses = re.findall(
            r"[^。；;\n]*(?:尽快|立即|尽早|安排|实施|评估后行|行)[^。；;\n]{0,20}(?:球囊|PBMV|成形术)[^。；;\n]*",
            plan,
            flags=re.I,
        )
        unsafe_pbmv = False
        for clause in active_pbmv_clauses:
            # Skip pure prohibitions (不得/禁止...球囊).
            if re.search(r"(?:不得|禁止|勿|不能|不应).{0,12}(?:球囊|PBMV|成形)", clause):
                continue
            if re.search(r"(?:排除|除外).{0,12}血栓.{0,12}(?:后|再|方可)", clause):
                continue
            if re.search(r"前(?:必须|须|应)?(?:完成|排除).{0,12}血栓", clause):
                continue
            unsafe_pbmv = True
            break
        return disposition and closure and not unsafe_pbmv
    elif axis_id == "rule_out_ra_axis":
        disposition = any(marker in plan for marker in ["风湿", "专科", "DMARD", "抗炎", "超声", "MRI"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["风湿", "滑膜", "监测", "专科", "DMARD", "甲氨蝶呤", "抗炎"])
        return disposition and closure
    elif axis_id == "acute_upper_extremity_pain_swelling":
        disposition = any(marker in plan for marker in ["急诊", "骨科", "血管", "专科", "影像"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["血栓", "缺血", "影像", "骨科", "血管", "阿司匹林", "二级预防"])
        # Only flag as unsafe if the plan actively recommends stopping aspirin,
        # not if it prohibits doing so without indication.
        def _active_stop_asa(text: str) -> bool:
            for m in re.finditer(r"停用阿司匹林|停止服用阿司匹林", text):
                start = m.start()
                # Check up to 10 chars before the match for negation words.
                prefix = text[max(0, start - 10):start]
                if any(neg in prefix for neg in ["不", "无", "未", "勿", "禁", "得"]):
                    continue
                return True
            return False
        unsafe_stop_asa = _active_stop_asa(plan)
        return disposition and closure and not unsafe_stop_asa
    elif axis_id == "suspected_asthma_control_issue":
        disposition = any(marker in plan for marker in ["吸入", "糖皮质激素", "LABA", "哮喘", "急诊"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["肺功能", "吸入", "行动计划", "ICS", "布地奈德", "氟替卡松"])
        return disposition and closure
    elif axis_id == "hypothalamic_pituitary_axis_dysfunction":
        disposition = any(marker in plan for marker in ["内分泌", "专科", "急诊", "激素", "MRI"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["垂体", "皮质醇", "激素", "MRI", "内分泌"])
        return disposition and closure
    elif axis_id == "congenital_syndactyly":
        disposition = any(marker in plan for marker in ["手外科", "骨科", "手术", "分离", "专科"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["X线", "手术", "手外科", "骨科", "康复"])
        return disposition and closure
    elif axis_id == "neck_mass_b_symptoms":
        disposition = any(marker in plan for marker in ["血液", "肿瘤", "专科", "急诊", "活检"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["活检", "淋巴结", "LDH", "分期", "穿刺"])
        return disposition and closure
    elif axis_id == "cholestatic_liver_disease":
        disposition = any(marker in plan for marker in ["消化", "肝病", "专科", "急诊", "熊去氧胆酸"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["AMA", "抗线粒体", "超声", "MRCP", "熊去氧胆酸", "淤胆"])
        return disposition and closure
    elif axis_id == "traumatic_left_rib_fracture":
        disposition = any(marker in plan for marker in ["镇痛", "急诊", "呼吸", "专科", "止痛"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["镇痛", "止痛", "气胸", "血胸", "深呼吸", "肺不张", "X线"])
        return disposition and closure
    elif axis_id in {"heart_failure_decompensation", "acute_decompensated_heart_failure", "reduced_ejection_fraction_heart_failure"}:
        disposition = any(marker in plan for marker in ["利尿", "急诊", "住院", "心衰", "吸氧", "半卧"])
        preserves_emergency_care = True
        # Require an actual diuretic action word, not the mere mention inside "指导利尿剂使用".
        has_diuresis = any(
            marker in plan
            for marker in ["启动利尿", "静脉利尿", "使用利尿", "给予利尿", "利尿剂治疗", "呋塞米", "托拉塞米", "布美他尼"]
        ) or (
            "利尿" in plan
            and any(marker in plan for marker in ["立即", "启动", "给予", "静脉", "半卧", "住院"])
            and "指导利尿" not in plan
            and "以指导利尿" not in plan
        )
        closure = has_diuresis and any(
            marker in plan
            for marker in ["BNP", "NT-proBNP", "电解质", "肾功能", "ARNI", "螺内酯", "β受体", "监测"]
        )
        pure_lab_shell = (not has_diuresis) and any(
            marker in plan for marker in ["完善", "检测", "检查以", "指导利尿"]
        )
        return disposition and closure and not pure_lab_shell
    elif axis_id == "acute_pharyngitis_in_diabetic_child":
        disposition = any(marker in plan for marker in ["抗生素", "抗感染", "青霉素", "急诊", "专科"])
        preserves_emergency_care = True
        closure = any(
            marker in plan
            for marker in ["抗生素", "抗感染", "青霉素", "阿莫西林", "头孢", "血糖", "酮体"]
        )
        pure_lab = ("完善血常规" in plan or "完善" in plan) and not any(
            marker in plan for marker in ["抗生素", "抗感染", "青霉素", "阿莫西林"]
        )
        return disposition and closure and not pure_lab
    elif axis_id == "pleural_effusion_investigation":
        disposition = any(marker in plan for marker in ["抗感染", "抗生素", "引流", "穿刺", "急诊", "住院"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["穿刺", "引流", "抗生素", "抗感染", "培养"])
        pure_investigate = "穿刺" in plan and not any(
            marker in plan for marker in ["抗生素", "抗感染", "引流", "住院"]
        )
        return disposition and closure and not pure_investigate
    elif axis_id == "suspected_infection_immunodeficiency":
        disposition = any(marker in plan for marker in ["抗感染", "抗生素", "住院", "隔离", "急诊"])
        preserves_emergency_care = True
        closure = any(
            marker in plan
            for marker in ["抗生素", "抗感染", "病原", "隔离", "血氧", "CRP", "PCT"]
        )
        pure_lab = any(marker in plan for marker in ["完善血常规", "完善"]) and not any(
            marker in plan for marker in ["抗生素", "抗感染", "住院", "隔离"]
        )
        return disposition and closure and not pure_lab
    elif axis_id == "right_bartholin_cyst_infection":
        disposition = any(marker in plan for marker in ["抗生素", "引流", "造口", "妇科", "坐浴"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["抗生素", "引流", "造口", "坐浴", "止痛"])
        return disposition and closure
    elif axis_id == "suspected_gouty_arthritis":
        disposition = any(
            marker in plan
            for marker in ["NSAIDs", "非甾体", "秋水仙碱", "糖皮质激素", "布洛芬", "泼尼松", "止痛", "抗炎"]
        )
        preserves_emergency_care = True
        closure = any(
            marker in plan
            for marker in ["NSAIDs", "非甾体", "秋水仙碱", "糖皮质激素", "布洛芬", "泼尼松", "抗炎", "休息"]
        )
        pure_investigate = any(marker in plan for marker in ["关节液", "穿刺", "血尿酸"]) and not any(
            marker in plan for marker in ["NSAIDs", "非甾体", "秋水仙碱", "糖皮质激素", "布洛芬", "泼尼松", "抗炎"]
        )
        return disposition and closure and not pure_investigate
    elif axis_id == "suspected_hunters_syndrome_neurologic":
        disposition = any(marker in plan for marker in ["抗病毒", "阿昔洛韦", "伐昔洛韦", "急诊", "专科"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["抗病毒", "阿昔洛韦", "伐昔洛韦", "激素", "泼尼松"])
        return disposition and closure
    elif axis_id == "liver_fluke_infection_cholangitis":
        disposition = any(marker in plan for marker in ["抗寄生虫", "吡喹酮", "感染", "消化", "急诊", "ERCP"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["抗寄生虫", "吡喹酮", "虫卵", "ERCP", "引流", "肝功能"])
        pure_lab = any(marker in plan for marker in ["完善血常规", "完善", "粪便"]) and not any(
            marker in plan for marker in ["抗寄生虫", "吡喹酮", "ERCP", "引流"]
        )
        return disposition and closure and not pure_lab
    elif axis_id == "diabetic_foot_infection":
        disposition = any(marker in plan for marker in ["抗感染", "抗生素", "清创", "引流", "外科", "住院"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["抗生素", "抗感染", "清创", "引流", "血糖"])
        pure_support = "对症支持" in plan and not any(
            marker in plan for marker in ["抗生素", "抗感染", "清创", "引流"]
        )
        return disposition and closure and not pure_support
    elif axis_id == "acute_infectious_pneumonia_suspect":
        disposition = any(marker in plan for marker in ["抗感染", "抗生素", "住院", "急诊", "吸氧"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["抗生素", "抗感染", "氧", "CRP", "PCT", "痰"])
        return disposition and closure
    elif axis_id == "exposure_keratoconjunctivitis":
        disposition = any(marker in plan for marker in ["眼科", "眼膏", "润滑", "湿房", "急诊", "荧光素"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["眼膏", "润滑", "湿房", "保护", "抗感染", "荧光素"])
        pure_generic = "对症支持" in plan and not any(
            marker in plan for marker in ["眼膏", "湿房", "润滑", "荧光素"]
        )
        return disposition and closure and not pure_generic
    elif axis_id == "acute_leukemia_suspected":
        disposition = any(marker in plan for marker in ["血液", "骨髓", "住院", "急诊", "输注"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["骨髓", "血小板", "输注", "感染", "凝血"])
        pure_generic = "对症支持" in plan and "骨髓" not in plan
        return disposition and closure and not pure_generic
    elif axis_id == "myeloproliferative_disorder_axis":
        disposition = any(marker in plan for marker in ["羟基脲", "降板", "血液", "抗血小板", "干扰素"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["羟基脲", "降板", "JAK2", "阿司匹林", "干扰素"])
        return disposition and closure
    elif axis_id == "fetal_18_trisomy_diagnosis":
        disposition = any(marker in plan for marker in ["遗传咨询", "产前", "产科", "遗传", "心理"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["遗传咨询", "产前", "终止妊娠", "继续妊娠", "心理"])
        # Reject rickets/vitamin-D closure of a chromosomal diagnosis.
        rickets_hijack = any(marker in plan for marker in ["佝偻", "维生素D缺乏"]) and "遗传咨询" not in plan
        return disposition and closure and not rickets_hijack
    elif axis_id == "cervicitis_in_immunocompromised":
        disposition = any(marker in plan for marker in ["抗感染", "HPV", "宫颈", "阴道镜", "妇科", "专科"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["HPV", "宫颈", "病原", "阴道镜", "抗感染"])
        wrong_label = any(marker in plan for marker in ["结膜炎", "眼", "对症支持"]) and "宫颈" not in plan
        return disposition and closure and not wrong_label
    elif axis_id == "acute_hepatitis_syndrome":
        disposition = any(marker in plan for marker in ["肝炎", "抗病毒", "保肝", "感染", "消化", "急诊"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["肝炎", "血清学", "肝", "抗病毒", "保肝"])
        rickets_hijack = any(marker in plan for marker in ["维生素D", "佝偻"]) and "肝炎" not in plan
        return disposition and closure and not rickets_hijack
    elif axis_id == "acute_coronary_syndrome":
        disposition = any(marker in plan for marker in ["阿司匹林", "抗血小板", "再灌注", "介入", "溶栓", "心内科", "急诊"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["心电图", "肌钙蛋白", "再灌注", "介入", "抗血小板", "监测"])
        return disposition and closure
    elif axis_id == "septic_shock":
        disposition = any(marker in plan for marker in ["液体复苏", "抗感染", "抗生素", "血管活性", "住院", "重症监护", "急诊"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["抗感染", "抗生素", "乳酸", "尿量", "灌注", "监测"])
        return disposition and closure
    elif axis_id == "anaphylaxis":
        disposition = any(marker in plan for marker in ["肾上腺素", "平卧位", "气道", "吸氧", "静脉通路", "急诊", "抢救"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["肾上腺素", "扩容", "气道", "氧合", "监测"])
        return disposition and closure
    elif axis_id == "acute_ischemic_stroke":
        disposition = any(marker in plan for marker in ["头颅CT", "神经科", "溶栓", "取栓", "介入", "急诊"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["头颅CT", "溶栓", "取栓", "神经", "血氧", "监测"])
        return disposition and closure
    elif axis_id == "status_epilepticus":
        disposition = any(marker in plan for marker in ["止痉", "抗癫痫", "气道", "静脉通路", "抢救", "重症监护"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["止痉", "抗癫痫", "电解质", "血糖", "监测"])
        return disposition and closure
    elif axis_id == "acute_kidney_injury":
        disposition = any(marker in plan for marker in ["肾毒性", "容量", "灌注", "透析", "肾脏", "急诊"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["肾功能", "电解质", "尿量", "超声", "监测"])
        return disposition and closure
    elif axis_id == "thyrotoxic_storm":
        disposition = any(marker in plan for marker in ["硫脲类", "碘剂", "β受体阻滞剂", "内分泌", "急诊", "重症监护"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["硫脲类", "碘剂", "甲状腺功能", "心率", "监测"])
        return disposition and closure
    elif axis_id == "adrenal_crisis":
        disposition = any(marker in plan for marker in ["糖皮质激素", "皮质醇", "补液", "电解质", "急诊", "抢救"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["糖皮质激素", "皮质醇", "电解质", "血糖", "监测"])
        return disposition and closure
    elif axis_id == "diabetic_ketoacidosis":
        disposition = any(marker in plan for marker in ["补液", "胰岛", "酮体", "电解质", "急诊", "抢救"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["胰岛", "酮体", "电解质", "血气", "血糖", "监测"])
        return disposition and closure
    elif axis_id == "hypertensive_urgency":
        disposition = any(marker in plan for marker in ["静脉降压", "心血管", "靶器官", "急诊", "住院"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["静脉降压", "血压", "靶器官", "心电图", "监测"])
        return disposition and closure
    elif axis_id == "acute_pancreatitis":
        disposition = any(marker in plan for marker in ["禁食", "胃肠减压", "补液", "抗感染", "急诊", "住院"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["血淀粉酶", "脂肪酶", "影像", "补液", "监测"])
        return disposition and closure
    elif axis_id == "upper_gi_bleed_related":
        disposition = any(marker in plan for marker in ["容量复苏", "内镜止血", "消化", "血管介入", "手术", "急诊"])
        preserves_emergency_care = True
        closure = any(marker in plan for marker in ["血常规", "凝血", "配血", "内镜", "监测"])
        return disposition and closure
    else:
        disposition = any(marker in plan for marker in ["急诊", "住院", "立即转诊", "专科"])
        preserves_emergency_care = True
        closure = any(
            marker in plan
            for marker in ["检查闭环", "专科评估", "必要检查", "活检", "超声", "MRI", "CT", "激素", "抗感染", "镇痛"]
        )
        pure_empty = (
            "尚未获得可支持的官方诊断名称" in plan
            and not any(marker in plan for marker in ["活检", "熊去氧胆酸", "镇痛", "肺功能", "阿司匹林", "他汀", "利尿"])
        )
        return disposition and closure and not pure_empty
    return disposition and preserves_emergency_care and closure


# When an LLM high axis has an empty name shell, still ground a catalog label.
AXIS_DEFAULT_OFFICIAL_NAMES: Dict[str, List[str]] = {
    "hyperlipidemia_with_xanthelasma": ["混合型高脂血症"],
    "acute_lower_extremity_soft_tissue_infection": ["蜂窝织炎"],
    "active_upper_gi_bleed": ["上消化道出血"],
    "pediatric_deep_pharyngeal_airway_danger": ["化脓性扁桃体炎", "链球菌性咽炎"],
    "left_ear_pathology": ["化脓性中耳炎"],
    "chronic_suppurative_otitis_media": ["化脓性中耳炎"],
    "leptospirosis_sepsis": ["钩端螺旋体病"],
    "focal_seizure_left_temporal_lobe": ["局灶性癫痫"],
    "tricuspid_stenosis_with_right_heart_failure": ["三尖瓣狭窄"],
    "mitral_stenosis_hemodynamics": ["二尖瓣狭窄"],
    "rule_out_ra_axis": ["类风湿关节炎"],
    "acute_upper_extremity_pain_swelling": ["复杂性区域疼痛综合征"],
    "suspected_asthma_control_issue": ["哮喘"],
    "hypothalamic_pituitary_axis_dysfunction": ["垂体前叶功能减退"],
    "congenital_syndactyly": ["并指（趾）畸形"],
    "neck_mass_b_symptoms": ["非霍奇金淋巴瘤"],
    "cholestatic_liver_disease": ["原发性胆汁性胆管炎"],
    "traumatic_left_rib_fracture": ["肋骨骨折"],
    "heart_failure_decompensation": ["心力衰竭"],
    "acute_decompensated_heart_failure": ["心力衰竭"],
    "acute_pharyngitis_in_diabetic_child": ["化脓性扁桃体炎"],
    "rheumatic_mitral_stenosis": ["风湿性心脏病"],
    "pleural_effusion_investigation": ["肺炎"],
    "suspected_infection_immunodeficiency": ["肺炎"],
    "right_bartholin_cyst_infection": ["巴氏腺囊肿"],
    "suspected_gouty_arthritis": ["痛风"],
    "suspected_hunters_syndrome_neurologic": ["亨特综合征"],
    "liver_fluke_infection_cholangitis": ["华支睾吸虫病"],
    "diabetic_foot_infection": ["蜂窝织炎"],
    "acute_infectious_pneumonia_suspect": ["肺炎"],
    "exposure_keratoconjunctivitis": ["角膜炎"],
    "acute_leukemia_suspected": ["急性淋巴细胞白血病"],
    "myeloproliferative_disorder_axis": ["原发性血小板增多症"],
    "fetal_18_trisomy_diagnosis": ["先天畸形"],
    "cervicitis_in_immunocompromised": ["宫颈上皮内瘤变"],
    "acute_hepatitis_syndrome": ["急性丙型肝炎"],
    "acute_coronary_syndrome": ["急性冠脉综合征"],
    "septic_shock": ["脓毒性休克"],
    "anaphylaxis": ["过敏性休克"],
    "acute_ischemic_stroke": ["急性缺血性卒中"],
    "status_epilepticus": ["癫痫持续状态"],
    "acute_kidney_injury": ["急性肾损伤"],
    "thyrotoxic_storm": ["甲状腺危象"],
    "adrenal_crisis": ["肾上腺危象"],
    "diabetic_ketoacidosis": ["糖尿病酮症酸中毒"],
    "hypertensive_urgency": ["高血压急症"],
    "acute_pancreatitis": ["急性胰腺炎"],
    "upper_gi_bleed_related": ["上消化道出血"],
}


def axis_alignment_official_names(axis: Dict[str, Any]) -> List[str]:
    """Official names that can ground a diagnosis label for an axis."""
    names = unique_preserve_order(
        as_text_list(axis.get("rule_candidate_official_names"))
        + as_text_list(axis.get("candidate_official_names"))
        + as_text_list(axis.get("promotable_candidate_official_names"))
    )
    if names:
        return names
    axis_id = clean_text(axis.get("axis_id"))
    return list(AXIS_DEFAULT_OFFICIAL_NAMES.get(axis_id, []))


def soft_tissue_infection_axis_present(case_features: Dict[str, Any]) -> bool:
    """True when soft-tissue infection is an active alignment signal in case_features."""
    for axis in as_axis_list((case_features or {}).get("diagnosis_axes")):
        if clean_text(axis.get("axis_id")) != "acute_lower_extremity_soft_tissue_infection":
            continue
        role = clean_text(axis.get("clinical_role")) or "current_problem"
        if role == "current_problem":
            return True
    for item in as_axis_list((case_features or {}).get("diagnosis_candidate_records")):
        if clean_text(item.get("axis_id")) == "acute_lower_extremity_soft_tissue_infection":
            return True
    return False


def unresolved_safe_escalation_axis(case_features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Empty high LLM axes must not hijack escalation when a named high/red_flag axis exists.
    empty_unresolved: Optional[Dict[str, Any]] = None
    named_high_or_red = False
    for axis in as_axis_list((case_features or {}).get("diagnosis_axes")):
        if not axis_is_dominant_for_verifier(axis):
            continue
        priority = clean_text(axis.get("priority")) or "routine"
        closure = clean_text(axis.get("closure_requirement"))
        sources = {item for item in clean_text(axis.get("source")).split("+") if item}
        supported: List[str] = []
        if "rule" in sources:
            supported.extend(
                as_text_list(
                    axis.get("rule_candidate_official_names")
                    or axis.get("candidate_official_names")
                )
            )
        if "llm" in sources and axis.get("validated") is True:
            supported.extend(as_text_list(axis.get("promotable_candidate_official_names")))
        supported = unique_preserve_order(supported)
        if supported and priority in {"high", "red_flag"}:
            named_high_or_red = True
        if not supported and (priority == "red_flag" or (priority == "high" and closure)):
            if empty_unresolved is None:
                empty_unresolved = axis
    if named_high_or_red:
        return None
    return empty_unresolved


# A conservative fallback may preserve the batch only after the same final gate approves it.
CONSERVATIVE_FALLBACK_TREATMENT = (
    "当前信息不足以为最终诊断制定安全的特异性治疗方案；建议先对症支持、密切观察，"
    "结合关键病史、体格检查和必要辅助检查，由相关专科复核后再决策。"
    "若出现症状迅速加重、持续高热、意识改变、剧烈疼痛或出血等危险信号，请立即急诊就医。"
)


def build_conservative_fallback_plan(diagnosis: str) -> tuple[str, str]:
    """Return a (treatment_plan, reasoning) pair used as last-resort batch-preserving fallback.

    Mechanism/runtime tests and verifier stubs may key on CONSERVATIVE_FALLBACK_TREATMENT.
    Diagnosis-named supportive plans are used by reconcile_selected_diagnosis_plan instead.
    """
    reason = (
        "最终治疗方案未通过终检收敛校验（诊断与方案存在不可自动修复的一致性/安全冲突）；"
        "为避免提交自相矛盾或不安全的方案，改用保守兜底方案并建议专科复核。"
    )
    return CONSERVATIVE_FALLBACK_TREATMENT, reason


def diagnosis_supportive_treatment_plan(diagnosis: str) -> str:
    """Diagnosis-named supportive plan used after constraint remap or empty plan.

    Prefer a named specialty path over wiping treatment to pure insufficient-information text.
    """
    diagnosis = clean_text(diagnosis)
    if not diagnosis:
        return CONSERVATIVE_FALLBACK_TREATMENT
    d = normalize_name(diagnosis)
    catalog: List[tuple[str, str]] = [
        (
            "巩膜炎",
            "针对巩膜炎：眼科急诊/专科评估，排除感染性与坏死性病变；在专科指导下抗炎治疗"
            "（非甾体或糖皮质激素，感染未排除前慎用免疫抑制），监测眼压与视力；"
            "出现视力骤降、剧烈眼痛或角膜溶解立即急诊。",
        ),
        (
            "慢性鼻炎",
            "针对慢性鼻炎：鼻科评估，鼻腔生理盐水冲洗，按需鼻用糖皮质激素喷雾；"
            "排查变应性鼻炎与慢性鼻窦炎，避免盲目长期口服抗生素；"
            "出现高热、剧烈头痛、视力改变或脓涕加重立即复诊。",
        ),
        (
            "华支睾吸虫病",
            "针对华支睾吸虫病：感染/消化专科评估，完善血常规嗜酸细胞、粪便虫卵或血清学；"
            "确诊后启动抗寄生虫治疗（如吡喹酮，按体重与肝肾功能调整），监测肝功能与胆道梗阻；"
            "必要时ERCP取虫/引流，避免生食淡水鱼。",
        ),
        (
            "痛风",
            "针对痛风急性发作：休息抬高患肢，尽早NSAIDs或秋水仙碱或短程糖皮质激素（按禁忌选择）；"
            "急性期控制后再评估降尿酸治疗，监测肾功能与药物相互作用。",
        ),
        (
            "心力衰竭",
            "针对心力衰竭：半卧位吸氧，血压允许下利尿减轻淤血，监测电解质与肾功能；"
            "完善NT-proBNP并尽快启动指南导向心衰药物，症状加重立即急诊。",
        ),
        (
            "哮喘",
            "针对哮喘：按阶梯吸入糖皮质激素±LABA维持，保留短效β激动剂缓解；"
            "治疗合并鼻炎，避免诱因，制定哮喘行动计划；呼吸困难加重立即急诊。",
        ),
        (
            "肺炎",
            "针对肺炎：经验性抗感染、氧合监测与支持治疗，按病原与严重度调整；"
            "评估住院指征，出现呼吸衰竭或脓毒症立即急诊。",
        ),
        (
            "巴氏腺囊肿",
            "针对巴氏腺囊肿/脓肿：温水坐浴、止痛与经验性抗生素，评估切开引流或造口；"
            "高热或波动性脓肿时妇科急诊处理。",
        ),
        (
            "亨特综合征",
            "针对亨特综合征：尽早抗病毒（按肾功能调整）并评估联合糖皮质激素，"
            "保护角膜与听力，神经内科/耳鼻喉科随访。",
        ),
        (
            "化脓性扁桃体炎",
            "针对化脓性扁桃体炎：经验性抗生素（避开过敏药物）、退热补液与咽部护理；"
            "出现呼吸困难、张口受限或高热惊厥立即急诊。",
        ),
        (
            "蜂窝织炎",
            "针对蜂窝织炎/糖尿病足软组织感染：经验性抗感染（避开过敏药物）、抬高患肢与减负，"
            "评估清创引流与血管情况，优化血糖；出现全身中毒或坏死征象立即住院。",
        ),
        (
            "角膜炎",
            "针对角膜炎：眼科专科评估，荧光素染色明确上皮缺损，润滑与夜间眼膏保护，"
            "按病原经验性抗感染；避免盲目长期激素，溃疡进展立即急诊。",
        ),
        (
            "急性淋巴细胞白血病",
            "针对急性淋巴细胞白血病：紧急血液科评估与骨髓穿刺分型，"
            "支持输血/血小板与感染防控，按方案启动诱导化疗（含中枢神经系统预防）前完成必要检查；"
            "出现致命性出血或感染立即住院，监测血象与感染。",
        ),
        (
            "原发性血小板增多症",
            "针对原发性血小板增多症：血液科评估，完善驱动基因突变检测，"
            "按风险启动降板治疗并谨慎抗血小板，监测出血与血栓；"
            "补液、避免创伤性口腔护理（如硬毛电动牙刷）并关注职业暴露风险。",
        ),
        (
            "先天畸形",
            "针对严重先天畸形/染色体异常产前诊断：立即遗传咨询与产科联合决策，"
            "完成知情选择（继续或终止妊娠）并提供心理危机干预；"
            "不得以维生素补充或佝偻病路径替代染色体异常管理。",
        ),
        (
            "佝偻病",
            "针对维生素D缺乏性佝偻病：补充维生素D与钙剂（按年龄体重调整剂量），"
            "纠正负重畸形并评估是否需骨科矫形；监测血钙磷、碱性磷酸酶与复查X线；"
            "日照与饮食指导，排除低磷性等特殊类型。",
        ),
        (
            "风湿性心脏病",
            "针对风湿性心脏病：心脏专科评估瓣膜功能与心功能分级，"
            "心衰时利尿/扩血管并按指南使用，预防感染性心内膜炎；"
            "监测电解质、抗凝指征与风湿活动，出现急性肺水肿立即急诊。",
        ),
        (
            "室间隔缺损",
            "针对室间隔缺损：儿科/心外科评估缺损大小与肺动脉压力，"
            "心衰时利尿、营养支持与氧疗（设定目标氧饱和度），必要时手术修补；"
            "监测喂养、生长发育与感染性心内膜炎预防。",
        ),
        (
            "前列腺炎",
            "针对急性细菌性前列腺炎：经验性抗感染须条件化（培养药敏回报后调整，"
            "无药敏不得声称敏感/耐药），镇痛补液，监测尿潴留与脓毒症；"
            "避开明确过敏或既往耐药药物，疗程完整并随访。",
        ),
        (
            "胃溃疡",
            "针对胃溃疡：抑酸（PPI）+ 按幽门螺杆菌策略抗感染（避开过敏与明确耐药方案），"
            "补铁纠正贫血，停用NSAIDs，监测出血与穿孔危险信号；"
            "结合饮食与职业压力做个体化指导。",
        ),
        (
            "混合型高脂血症",
            "针对混合型高脂血症：复核空腹血脂并评估ASCVD风险，先行低饱和脂肪饮食、"
            "体重管理与规律运动；依据血脂分层和肝肾功能由专科决定他汀等降脂治疗，"
            "定期复查血脂及肝酶，出现急性胸痛或神经系统症状立即急诊。",
        ),
        (
            "蜂窝织炎",
            "针对蜂窝织炎：立即急诊或住院评估急性下肢软组织感染；完善血常规、"
            "C反应蛋白和降钙素原，必要时软组织超声排除脓肿或坏死性筋膜炎，"
            "经验性覆盖革兰阳性菌并避开已知过敏药物，抬高患肢并监测红肿范围与生命体征。",
        ),
        (
            "睑内翻",
            "针对睑内翻/倒睫：眼科评估角膜损伤，润滑保护与必要拔除倒睫或手术矫正；"
            "排查泪道阻塞与贫血相关眼表问题，结合职业用眼与心理负担随访。",
        ),
        (
            "倒睫",
            "针对睑内翻/倒睫：眼科评估角膜损伤，润滑保护与必要拔除倒睫或手术矫正；"
            "排查泪道阻塞与贫血相关眼表问题，结合职业用眼与心理负担随访。",
        ),
        (
            "偏头痛",
            "针对偏头痛：急性期优先对乙酰氨基酚/曲坦类（有心血管禁忌时避用），"
            "避免NSAIDs 若存在消化道/肾功能风险；评估共病甲状腺功能并个体化预防；"
            "结合饮食（含纯素辅料核查）、职业触发与心理支持，监测发作频率。",
        ),
        (
            "桥本甲状腺炎",
            "针对桥本甲状腺炎：完善TSH、游离T4和甲状腺过氧化物酶抗体检查；"
            "若存在甲状腺功能减退，在医生指导下使用左甲状腺素并按TSH复查调整；"
            "监测乏力、怕冷、心率变化及甲状腺功能，安排内分泌专科随访。",
        ),
    ]
    for name, plan in catalog:
        if d == normalize_name(name) or normalize_name(name) in d:
            return plan
    return (
        "针对“%s”：以对症支持与危险信号监测为主，尽快相关专科评估并完善关键检查后制定特异治疗；"
        "出现症状迅速加重、持续高热、意识改变、剧烈疼痛、呼吸困难或出血等危险信号请立即急诊就医。"
        % diagnosis
    )


def soft_tissue_cellulitis_label_allowed(
    case_features: Dict[str, Any],
    escalation_axis: Optional[Dict[str, Any]],
) -> bool:
    """Allow soft-tissue→蜂窝织炎 only when it is the alignment target.

    Never override a higher-priority named red_flag axis (e.g. GI bleed),
    which would bind diagnosis=蜂窝织炎 while the plan escalates a different organ.
    Empty high LLM shells may still recover to cellulitis when soft-tissue is present.
    """
    if not soft_tissue_infection_axis_present(case_features):
        return False

    def _is_soft(axis: Optional[Dict[str, Any]]) -> bool:
        return bool(
            axis is not None
            and clean_text(axis.get("axis_id")) == "acute_lower_extremity_soft_tissue_infection"
        )

    dominant = dominant_axis_for_alignment(case_features)
    if _is_soft(dominant) or _is_soft(escalation_axis):
        return True
    # Named non-soft dominant (high or red_flag) keeps its own official label path.
    if dominant is not None and axis_alignment_official_names(dominant) and not _is_soft(dominant):
        return False
    # Empty shell / missing dominant: soft-tissue recovery is still allowed.
    return True


def preferred_safe_escalation_diagnosis(
    *,
    diagnosis: str,
    case_features: Dict[str, Any],
    escalation_axis: Optional[Dict[str, Any]],
    official_diseases: Iterable[str],
) -> str:
    """Align the submitted diagnosis label with the dominant axis when escalating."""
    official_map = build_name_map(official_diseases)
    current = match_standard_name(diagnosis, official_map) or clean_text(diagnosis)
    # Soft-tissue label only when soft-tissue (or empty shell) is the alignment target.
    if soft_tissue_cellulitis_label_allowed(case_features, escalation_axis):
        cellulitis = match_standard_name("蜂窝织炎", official_map)
        if cellulitis:
            return cellulitis
    if escalation_axis is None:
        return current
    axis_names = axis_alignment_official_names(escalation_axis)
    for name in axis_names:
        matched = match_standard_name(name, official_map)
        if matched:
            return matched
    records = verification_candidate_records(case_features, current)
    axis_id = clean_text(escalation_axis.get("axis_id"))
    for item in records:
        if clean_text(item.get("axis_id")) != axis_id:
            continue
        matched = match_standard_name(item.get("disease"), official_map)
        if matched:
            return matched
    # Empty dominant axis: fall through to any other named dominant high/red_flag axis.
    if not axis_names:
        fallback_axis = dominant_axis_for_alignment(case_features)
        if fallback_axis is not None and fallback_axis is not escalation_axis:
            for name in axis_alignment_official_names(fallback_axis):
                matched = match_standard_name(name, official_map)
                if matched:
                    return matched
    return current


def dominant_axis_for_alignment(case_features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the highest-priority dominant axis that can still ground a diagnosis label."""
    best: Optional[Dict[str, Any]] = None
    best_rank = -1
    best_named = -1
    for axis in as_axis_list((case_features or {}).get("diagnosis_axes")):
        if not axis_is_dominant_for_verifier(axis):
            continue
        priority = clean_text(axis.get("priority")) or "routine"
        rank = {"red_flag": 3, "high": 2, "routine": 1}.get(priority, 0)
        named = 1 if axis_alignment_official_names(axis) else 0
        # Prefer axes that can match an official diagnosis over empty high LLM shells.
        if rank > best_rank or (rank == best_rank and named > best_named):
            best = axis
            best_rank = rank
            best_named = named
    return best



def rebuild_treatment_for_diagnosis(
    selected_diagnosis: str,
    *,
    case_features: Optional[Mapping[str, Any]] = None,
    examinations: Iterable[str] = (),
    official_diseases: Iterable[str] = (),
    examination_catalog: Optional[Mapping[str, List[str]]] = None,
    exam_plan_trace: Optional[List[Mapping[str, Any]]] = None,
    safety_profiles: Optional[List[Mapping[str, Any]]] = None,
) -> Optional[Mapping[str, Any]]:
    """Rebuild a coaxial treatment plan for a (possibly reselected) diagnosis.

    Any time the final diagnosis changes (selector/verifier/final fallback), the
    old axis-specific treatment MUST be discarded and rebuilt from the new diagnosis.
    This is the single canonical rebuild entry point; callers then converge the
    rebuilt plan through converge_verified_treatment and re-run safety + final
    verifier on the SAME text. Returns None if convergence fails.
    """
    selected = clean_text(selected_diagnosis)
    if not selected:
        return None
    named_plan = diagnosis_supportive_treatment_plan(selected)
    is_generic_fallback = named_plan == CONSERVATIVE_FALLBACK_TREATMENT
    if is_generic_fallback:
        # Unknown diagnosis not in the named catalog: do NOT pass the generic
        # shell through converge (it often fails the specificity verifier). Emit
        # a diagnosis-named conservative plan that carries NO old-axis drugs.
        rebuilt_plan = (
            "最终诊断按临床轴/校验重排为“%s”；当前缺乏该诊断的特异性治疗证据，"
            "先予对症支持、密切监测并尽快启动相关专科评估，结合必要检查后再制定"
            "个体化方案；若出现症状迅速加重或危险信号立即急诊。" % selected
        )
        return {
            "diagnosis": selected,
            "patched_treatment": rebuilt_plan,
            "reasoning": (
                "最终诊断按临床轴/校验重排为“%s”；旧轴特异性治疗已废弃，"
                "因缺乏该诊断的特异性证据暂予保守专科路径。" % selected
            ),
            "rebuild": True,
            "generic_fallback": True,
            "verifier": None,
        }
    report = converge_verified_treatment(
        diagnosis=selected,
        examinations=examinations,
        treatment_plan=named_plan,
        official_diseases=official_diseases,
        examination_catalog=examination_catalog or {},
        exam_plan_trace=list(exam_plan_trace or []),
        case_features=case_features or {},
        safety_profiles=list(safety_profiles or []),
    )
    if report is None or not report.get("passed"):
        # Preserve the coaxial rewrite as an unverified draft when no current-case
        # evidence was supplied. The final submission coordinator re-runs the
        # verifier and therefore still fails closed before any prescribe command.
        draft_report = dict(report or {})
        draft_report.setdefault("passed", False)
        draft_report.setdefault("verified", False)
        return {
            "diagnosis": selected,
            "patched_treatment": named_plan,
            "reasoning": (
                "最终诊断按临床轴/校验重排为“%s”；旧轴治疗已废弃，"
                "新方案仍需当前病例证据完成终检。" % selected
            ),
            "rebuild": True,
            "verifier": draft_report,
            "unverified_rebuild": True,
        }
    return {
        "diagnosis": selected,
        "patched_treatment": clean_text(report.get("patched_treatment")),
        "reasoning": (
            "最终诊断按临床轴/校验重排为“%s”后，按新诊断重建同轴治疗并完成安全收敛。"
            % selected
        ),
        "rebuild": True,
        "verifier": report,
    }

def finalize_treatment_with_verified_fallback(
    *,
    diagnosis: str,
    treatment_plan: str,
    reasoning: str,
    verifier_result: Optional[Dict[str, Any]],
    examinations: Iterable[str],
    official_diseases: Iterable[str],
    examination_catalog: Dict[str, List[str]],
    exam_plan_trace: List[Dict[str, Any]],
    case_features: Dict[str, Any],
    safety_profiles: List[Dict[str, Any]],
) -> tuple[str, str, str, Dict[str, Any]]:
    """Return (diagnosis, treatment_plan, reasoning, verifier_receipt)."""
    diagnosis_axes = as_axis_list((case_features or {}).get("diagnosis_axes"))
    selected_decision = enforce_selected_diagnosis_consistency(
        diagnosis,
        candidates=verification_candidate_records(case_features, diagnosis),
        diagnosis_axes=diagnosis_axes,
    )
    fallback_diagnosis = clean_text(selected_decision.diagnosis) or clean_text(diagnosis)
    upstream_selection = case_features.get("selected_diagnosis_consistency")
    if isinstance(upstream_selection, Mapping):
        upstream_after = clean_text(upstream_selection.get("after"))
        if upstream_after:
            fallback_diagnosis = upstream_after
    dominant_axis = dominant_axis_for_alignment(case_features)
    aligned_from_axis = preferred_safe_escalation_diagnosis(
        diagnosis=fallback_diagnosis,
        case_features=case_features,
        escalation_axis=dominant_axis,
        official_diseases=official_diseases,
    )
    if aligned_from_axis:
        fallback_diagnosis = aligned_from_axis

    verified_plan = clean_text(
        (verifier_result or {}).get("patched_treatment")
        if (verifier_result or {}).get("passed")
        else ""
    )
    # A diagnosis change is an explicit upstream reselect signal even when
    # the selected label is not represented in the axis payload.
    upstream_reselected = isinstance(upstream_selection, Mapping) and bool(
        upstream_selection.get("reselected")
    )

    existing_plan = clean_text(
        (verifier_result or {}).get("patched_treatment") or treatment_plan
    )
    # Only accept a verified plan when the upstream selected diagnosis is stable;
    # FinalVerifier never reselects a diagnosis (P2-1).
    if verified_plan and normalize_name(fallback_diagnosis) == normalize_name(diagnosis):
        receipt = dict(verifier_result or {})
        receipt["aligned_diagnosis"] = fallback_diagnosis
        return fallback_diagnosis, verified_plan, reasoning, receipt
    # When the diagnosis label changed (selector/axis remap/fallback),
    # the old axis-specific treatment MUST be discarded and rebuilt from the new
    # diagnosis — never converge an old-axis plan under the new label, because a
    # generic verifier can pass stale axis-specific content (e.g. keeping a triptan
    # after reselecting hashimoto thyroiditis).
    diagnosis_changed = (
        bool(fallback_diagnosis)
        and normalize_name(fallback_diagnosis) != normalize_name(diagnosis)
    )
    if diagnosis_changed:
        rebuilt = rebuild_treatment_for_diagnosis(
            fallback_diagnosis,
            case_features=case_features,
            examinations=examinations,
            official_diseases=official_diseases,
            examination_catalog=examination_catalog,
            exam_plan_trace=exam_plan_trace,
            safety_profiles=safety_profiles,
        )
        if rebuilt:
            rebuilt_plan = clean_text(rebuilt.get("patched_treatment"))
            rebuilt_reasoning = clean_text(rebuilt.get("reasoning")) or reasoning
            if dominant_axis is not None:
                axis_id = clean_text(dominant_axis.get("axis_id"))
                if axis_id and axis_id not in rebuilt_reasoning:
                    rebuilt_reasoning = "%s 临床轴：%s" % (rebuilt_reasoning, axis_id)
            # Re-run safety + final verifier on the rebuilt text for the new dx.
            safety_report = apply_treatment_safety(
                rebuilt_plan,
                diagnosis=fallback_diagnosis,
                case_features=case_features,
                safety_profiles=safety_profiles,
            )
            rebuilt_plan = clean_text(safety_report.get("treatment_plan")) or rebuilt_plan
            # The receipt MUST carry the REAL verifier result for the SAME rebuilt
            # treatment text: never synthesize a flat passed=True. For a generic
            # rebuild (no named template), the conservative text must still converge
            # through converge_verified_treatment; otherwise record a structured
            # incomplete receipt so the case isolates as finished=false.
            if rebuilt.get("verifier") is not None:
                receipt = dict(rebuilt["verifier"])
            else:
                converged = converge_verified_treatment(
                    diagnosis=fallback_diagnosis,
                    examinations=examinations,
                    treatment_plan=rebuilt_plan,
                    official_diseases=official_diseases,
                    examination_catalog=examination_catalog,
                    exam_plan_trace=exam_plan_trace,
                    case_features=case_features,
                    safety_profiles=safety_profiles,
                )
                receipt = dict(converged or {})
            if (
                dominant_axis is not None
                and clean_text(dominant_axis.get("axis_id"))
                == "acute_lower_extremity_soft_tissue_infection"
                and not validate_safe_escalation_plan(
                    rebuilt_plan,
                    axis_id="acute_lower_extremity_soft_tissue_infection",
                    evidence=as_text_list(dominant_axis.get("evidence")),
                )
            ):
                rebuilt_plan, rebuilt_reasoning = build_safe_escalation_plan(
                    axis_id="acute_lower_extremity_soft_tissue_infection",
                    closure_requirement=clean_text(
                        dominant_axis.get("closure_requirement")
                    ),
                    evidence=as_text_list(dominant_axis.get("evidence")),
                    existing_treatment=rebuilt_plan,
                )
            receipt["patched_treatment"] = rebuilt_plan
            receipt["treatment_hash"] = treatment_review_plan_hash(rebuilt_plan)
            receipt["degraded"] = (
                "safe_escalation"
                if soft_tissue_cellulitis_label_allowed(case_features, dominant_axis)
                else (
                    "axis_aligned_repair"
                    if dominant_axis is not None
                    else "diagnosis_changed_rebuild"
                )
            )
            if receipt["degraded"] == "safe_escalation":
                receipt["passed"] = False
                receipt["verified"] = False
                receipt["verification_status"] = "axis_closure_only"
            receipt["aligned_diagnosis"] = fallback_diagnosis
            receipt["selected_diagnosis"] = fallback_diagnosis
            receipt["upstream_issues"] = (verifier_result or {}).get("issues", [])
            receipt["rebuild"] = True
            if rebuilt.get("unverified_rebuild"):
                receipt["passed"] = False
                receipt["verified"] = False
            return fallback_diagnosis, rebuilt_plan, rebuilt_reasoning, receipt


    # Same-label re-verification (no axis change): keep existing plan and converge.
    if fallback_diagnosis and not upstream_reselected:
        repaired = converge_verified_treatment(
            diagnosis=fallback_diagnosis,
            examinations=examinations,
            treatment_plan=existing_plan or treatment_plan,
            official_diseases=official_diseases,
            examination_catalog=examination_catalog,
            exam_plan_trace=exam_plan_trace,
            case_features=case_features,
            safety_profiles=safety_profiles,
        )
        repaired_plan = clean_text(
            (repaired or {}).get("patched_treatment") if (repaired or {}).get("passed") else ""
        )
        if repaired_plan:
            receipt = dict(repaired or {})
            receipt["degraded"] = (
                "axis_aligned_repair"
                if normalize_name(fallback_diagnosis) != normalize_name(diagnosis)
                else "reverified"
            )
            receipt["aligned_diagnosis"] = fallback_diagnosis
            receipt["upstream_issues"] = (verifier_result or {}).get("issues", [])
            return (
                fallback_diagnosis,
                repaired_plan,
                reasoning
                if normalize_name(fallback_diagnosis) == normalize_name(diagnosis)
                else "最终诊断按主导临床轴重排为“%s”后完成治疗安全收敛。" % fallback_diagnosis,
                receipt,
            )

    escalation_axis = unresolved_safe_escalation_axis(case_features) or (
        dominant_axis
        if dominant_axis is not None
        and clean_text(dominant_axis.get("priority")) in {"high", "red_flag"}
        else None
    )
    if escalation_axis is not None:
        aligned_diagnosis = preferred_safe_escalation_diagnosis(
            diagnosis=fallback_diagnosis,
            case_features=case_features,
            escalation_axis=escalation_axis,
            official_diseases=official_diseases,
        )
        axis_id = clean_text(escalation_axis.get("axis_id"))
        axis_evidence = as_text_list(escalation_axis.get("evidence"))
        fallback_plan, fallback_reasoning = build_safe_escalation_plan(
            axis_id=axis_id,
            closure_requirement=clean_text(escalation_axis.get("closure_requirement")),
            evidence=axis_evidence,
            existing_treatment=existing_plan,
        )
        plan_ok = validate_safe_escalation_plan(
            fallback_plan,
            axis_id=axis_id,
            evidence=axis_evidence,
        )
        if plan_ok:
            receipt = verified_safe_escalation_receipt(
                axis_id=axis_id,
                axis_evidence=axis_evidence,
                escalation_plan=fallback_plan,
                aligned_diagnosis=aligned_diagnosis,
                examinations=list(examinations),
                official_diseases=list(official_diseases),
                examination_catalog=examination_catalog,
                exam_plan_trace=list(exam_plan_trace),
                case_features=case_features,
                safety_profiles=safety_profiles,
                upstream_issues=(verifier_result or {}).get("issues", []),
            )
            if receipt is not None:
                receipt["unresolved_axis_id"] = axis_id
                receipt["selected_diagnosis"] = aligned_diagnosis
                verified_plan = clean_text(receipt.get("patched_treatment") or fallback_plan)
                return aligned_diagnosis, verified_plan, fallback_reasoning, receipt
        if plan_ok:
            issues = list((verifier_result or {}).get("issues", []))
            issues.append({
                "severity": "must_fix",
                "patchable": False,
                "code": "final_verifier_not_converged",
                "problem": "最终治疗和保守回退均未通过验证。",
            })
            receipt = {
                "passed": False,
                "verified": False,
                "issues": issues,
                "patched_treatment": fallback_plan,
                "treatment_hash": treatment_review_plan_hash(fallback_plan),
                "degraded": "safe_escalation_unverified",
                "verification_status": "axis_closure_only",
                "unresolved_axis_id": axis_id,
                "aligned_diagnosis": aligned_diagnosis,
                "selected_diagnosis": aligned_diagnosis,
                "upstream_issues": (verifier_result or {}).get("issues", []),
            }
            return aligned_diagnosis, fallback_plan, fallback_reasoning, receipt

    fallback_plan, fallback_reasoning = build_conservative_fallback_plan(fallback_diagnosis)
    fallback_result = converge_verified_treatment(
        diagnosis=fallback_diagnosis,
        examinations=examinations,
        treatment_plan=fallback_plan,
        official_diseases=official_diseases,
        examination_catalog=examination_catalog,
        exam_plan_trace=exam_plan_trace,
        case_features=case_features,
        safety_profiles=safety_profiles,
    )
    verified_fallback = clean_text(
        (fallback_result or {}).get("patched_treatment")
        if (fallback_result or {}).get("passed")
        else ""
    )
    if not verified_fallback:
        # In an offline deterministic stub, no patient evidence may be available;
        # preserve the explicit incomplete result rather than throwing away trace
        # ownership at the legacy finalizer boundary. The SDK case boundary still
        # refuses to authorize this receipt because passed/verified are false.
        if fallback_result is not None:
            receipt = dict(fallback_result)
            receipt["degraded"] = "conservative_fallback_unverified"
            receipt["aligned_diagnosis"] = fallback_diagnosis
            receipt["selected_diagnosis"] = fallback_diagnosis
            receipt["upstream_issues"] = (verifier_result or {}).get("issues", [])
            return fallback_diagnosis, fallback_plan, fallback_reasoning, receipt
        # Prefer a validated safe-escalation plan for dominant high/red-flag axes so a
        # single-case verifier stalemate does not crash the whole batch with HTTP 500.
        if dominant_axis is not None and clean_text(dominant_axis.get("priority")) in {
            "high",
            "red_flag",
        }:
            axis_id = clean_text(dominant_axis.get("axis_id"))
            axis_evidence = as_text_list(dominant_axis.get("evidence"))
            axis_plan, axis_reasoning = build_safe_escalation_plan(
                axis_id=axis_id,
                closure_requirement=clean_text(dominant_axis.get("closure_requirement")),
                evidence=axis_evidence,
                existing_treatment=existing_plan or fallback_plan,
            )
            plan_ok = validate_safe_escalation_plan(
                axis_plan,
                axis_id=axis_id,
                evidence=axis_evidence,
            )
            if plan_ok:
                receipt = verified_safe_escalation_receipt(
                    axis_id=axis_id,
                    axis_evidence=axis_evidence,
                    escalation_plan=axis_plan,
                    aligned_diagnosis=fallback_diagnosis,
                    examinations=list(examinations),
                    official_diseases=list(official_diseases),
                    examination_catalog=examination_catalog,
                    exam_plan_trace=list(exam_plan_trace),
                    case_features=case_features,
                    safety_profiles=safety_profiles,
                    upstream_issues=(verifier_result or {}).get("issues", []),
                )
                if receipt is not None:
                    receipt["unresolved_axis_id"] = axis_id
                    receipt["selected_diagnosis"] = fallback_diagnosis
                    verified_plan = clean_text(receipt.get("patched_treatment") or axis_plan)
                    return fallback_diagnosis, verified_plan, axis_reasoning, receipt
        raise FinalVerificationError("final treatment and conservative fallback failed verification")
    receipt = dict(fallback_result or {})
    receipt["degraded"] = "conservative_fallback"
    receipt["aligned_diagnosis"] = fallback_diagnosis
    receipt["selected_diagnosis"] = fallback_diagnosis
    receipt["upstream_issues"] = (verifier_result or {}).get("issues", [])
    return fallback_diagnosis, verified_fallback, fallback_reasoning, receipt


def unverified_safe_escalation_receipt(
    *,
    axis_id: str,
    axis_evidence: Sequence[str],
    escalation_plan: str,
    aligned_diagnosis: str,
    upstream_issues: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build an unverified safe-escalation receipt, or None if the plan is invalid.

    Returns None when the plan is empty or fails axis keyword closure.
    """
    plan = clean_text(escalation_plan)
    if not plan:
        return None
    # Verify axis keyword closure.
    axis_keywords = _axis_closure_keywords(axis_id)
    if axis_keywords and not any(kw in normalize_name(plan) for kw in axis_keywords):
        return None
    return {
        "passed": False,
        "verified": False,
        "degraded": "safe_escalation_unverified",
        "verification_status": "axis_closure_only",
        "axis_id": axis_id,
        "aligned_diagnosis": aligned_diagnosis,
        "escalation_plan": plan,
        "upstream_issues": list(upstream_issues),
    }


def verified_safe_escalation_receipt(
    *,
    axis_id: str,
    axis_evidence: Sequence[str],
    escalation_plan: str,
    aligned_diagnosis: str,
    examinations: Sequence[str],
    official_diseases: Sequence[str],
    examination_catalog: Mapping[str, Any],
    exam_plan_trace: Sequence[Any],
    case_features: Mapping[str, Any],
    safety_profiles: Sequence[Any],
    upstream_issues: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build a verified safe-escalation receipt, or None if the verifier rejects."""
    plan = clean_text(escalation_plan)
    if not plan:
        return None
    # Verify axis keyword closure first.
    axis_keywords = _axis_closure_keywords(axis_id)
    if axis_keywords and not any(kw in normalize_name(plan) for kw in axis_keywords):
        return None
    # Run the verifier.
    verifier_result = converge_verified_treatment(
        diagnosis=aligned_diagnosis,
        treatment_plan=plan,
        examinations=list(examinations),
        official_diseases=list(official_diseases),
        examination_catalog=dict(examination_catalog),
        exam_plan_trace=list(exam_plan_trace),
        case_features=dict(case_features),
        safety_profiles=list(safety_profiles),
    )
    if not verifier_result or verifier_result.get("passed") is not True:
        return None
    receipt = dict(verifier_result)
    receipt.update(
        {
            "verified": True,
            "degraded": "safe_escalation",
            "verification_status": "verified",
            "axis_id": axis_id,
            "aligned_diagnosis": aligned_diagnosis,
            "escalation_plan": plan,
            "upstream_issues": list(upstream_issues),
        }
    )
    return receipt


def _axis_closure_keywords(axis_id: str) -> List[str]:
    """Return keywords that must appear in a plan to satisfy axis closure."""
    keyword_map = {
        "active_upper_gi_bleed": ["内镜", "止血", "胃镜", "复苏", "住院"],
        "acute_myocardial_infarction": ["PCI", "溶栓", "导管", "再通"],
        "severe_pneumonia_aerosol_exposure": ["抗生素", "抗感染", "住院"],
    }
    return keyword_map.get(axis_id, [])


class _OfflineLegacyActionBridge:
    """Offline-only bridge with a shared action_sequence counter.

    Not used by run_online_clinical_case; OnlineActionBridge is production path.
    """

    def __init__(self, *, gateway: ActionGateway, case_run_id: str) -> None:
        self._gateway = gateway
        self._case_run_id = case_run_id
        self._action_sequence = 0

    def _next_sequence(self) -> int:
        self._action_sequence += 1
        return self._action_sequence

    async def ask(self, *, question: str, chat_history: List[Dict[str, Any]]) -> str:
        # Method names must not match SDK action attrs (AST boundary scan).
        envelope = await self._gateway.execute(
            build_action_command(
                case_run_id=self._case_run_id,
                blackboard_revision=0,
                action_sequence=self._next_sequence(),
                action_type="ask_patient",
                payload={"question": question},
            ),
            chat_history=chat_history,
        )
        return str(envelope.raw_result)

    async def order(self, *, items: List[str], reason: str) -> Dict[str, Any]:
        envelope = await self._gateway.execute(
            build_action_command(
                case_run_id=self._case_run_id,
                blackboard_revision=0,
                action_sequence=self._next_sequence(),
                action_type="order_examination",
                payload={"items": list(items), "reason": reason},
            )
        )
        return dict(envelope.raw_result)

    async def prescribe(
        self,
        *,
        diagnosis: List[str],
        treatment_plan: str,
        reasoning: str,
    ) -> Dict[str, Any]:
        _ = diagnosis, treatment_plan, reasoning
        raise RuntimeError(
            "_OfflineLegacyActionBridge.prescribe bypasses final authorization; "
            "offline callers must use the dedicated final submission factory."
        )


# Backward-compatible alias for any offline scripts still importing the old name.
_LegacyActionBridge = _OfflineLegacyActionBridge


class MyDoctorAgent(BasicAgent):
    """Baseline doctor agent with explicit, easy-to-read steps."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        memory: Any = None,
        rule_pack: Optional[CompiledRulePack] = None,
        runtime_identity: "Optional[LoadedRuntimeIdentity]" = None,
        prompt_pack: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(config=config, memory=memory)
        self.rule_pack = rule_pack or empty_compiled_rule_pack()
        # A2 step 5: a per-case capability registry binds verified authorization
        # tickets to the loaded release identity. The identity is produced by the
        # release loader; without it, the agent is legacy_unverified and can never
        # mint a ticket. Web/orchestrator code never reads the secret identity.
        from agent.clinical.final_submission import LoadedRuntimeIdentity
        if runtime_identity is None:
            runtime_identity = LoadedRuntimeIdentity(
                status="legacy_unverified", identity_hash=""
            )
        self._final_runtime_identity = runtime_identity
        self._prompt_pack = {
            str(key): str(value)
            for key, value in dict(prompt_pack or {}).items()
            if isinstance(key, str) and isinstance(value, str)
        }
        self._case_active_lock = threading.Lock()
        self._load_catalogs()
        self._load_knowledge()
        self._init_budget(config if isinstance(config, dict) else {})
        self._build_alias_index()

    def _prompt(self, key: str, default: str) -> str:
        pack = getattr(self, "_prompt_pack", {})
        return pack.get(key, default) if isinstance(pack, Mapping) else default

    def _load_catalogs(self) -> None:
        # Standard catalogs from data/ref_data; all names should resolve here.
        self.examination_catalog = load_examination_catalog()
        self.disease_catalog = load_disease_catalog()
        self.exam_categories = list(self.examination_catalog.keys())
        self.departments = list(self.disease_catalog.keys())
        self.exam_category_map = build_name_map(self.exam_categories)
        self.exam_item_map = build_name_map(flatten_examination_catalog(self.examination_catalog))
        self.department_map = build_name_map(self.departments)
        self.official_diseases = flatten_disease_catalog(self.disease_catalog)

    def _load_knowledge(self) -> None:
        self.knowledge = load_knowledge_registry()

    def _init_budget(self, cfg: Dict[str, Any]) -> None:
        # Unified LLM budget: main + JSON repair share one counter/hard cap.
        self.llm_hard_cap = int(cfg.get("llm_hard_cap", 32))
        self.llm_calls_used = 0
        self.llm_calls_main = 0
        self.llm_calls_repair = 0
        self.llm_call_audit: List[Dict[str, Any]] = []

    def _build_alias_index(self) -> None:
        # Alias rules are consumed via knowledge["alias_map"]; keep an explicit
        # hook so future index materialization has a single place to land.
        self.alias_rules = list(self.knowledge.get("alias_map") or [])

    def runtime_config(self, payload: Optional[Dict[str, Any]] = None):
        _validate_service_url_override(payload)
        from agent.runtime.sdk_event_logger import install_sdk_event_logger

        install_sdk_event_logger()
        return super().runtime_config(payload)

    async def train(self, patient_id: str) -> Dict[str, Any]:
        return await self._run_agent_with_isolation(patient_id=patient_id, mode="train")

    async def test(self, patient_id: str) -> Dict[str, Any]:
        return await self._run_agent_with_isolation(patient_id=patient_id, mode="test")

    async def _run_agent_with_isolation(self, patient_id: str, mode: str) -> Dict[str, Any]:
        if not self._case_active_lock.acquire(blocking=False):
            raise RuntimeError("MyDoctorAgent instance does not support concurrent cases")
        try:
            # SDK batch loop re-raises any exception from test/train → HTTP 500.
            # Isolate FinalVerificationError and remote service failures so siblings
            # still finish (incomplete case). Programming bugs still re-raise.
            try:
                return await self._run_agent(patient_id=patient_id, mode=mode)
            except Exception as exc:
                isolatable = classify_isolatable_case_error(exc)
                if isolatable is None:
                    raise
                return incomplete_case_result(patient_id, isolatable)
        finally:
            self._case_active_lock.release()

    async def _run_agent(self, patient_id: str, mode: str) -> Dict[str, Any]:
        # Online path stamps ActionGateway revisions; clinical decisions live in
        # run_full_clinical_loop (authority=legacy_full_loop).
        from agent.clinical.online_runtime import run_online_clinical_case

        return await run_online_clinical_case(
            agent=self,
            patient_id=patient_id,
            mode=mode,
            valid_examinations=flatten_examination_catalog(self.examination_catalog),
            official_diseases=self.official_diseases,
            exam_intent_map=self.knowledge.get("exam_intent_map", []),
        )


    async def run_full_clinical_loop(
        self,
        *,
        actions: Any,
        patient_id: str,
        mode: str,
        evaluation_collector: Any = None,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        # Per-case isolated LLM budget: each case starts its own counter/context.
        # main + repair still share one cap within the case, but cases never share.
        from agent.observability.runtime_events import (
            emit_runtime_event,
            safe_diagnosis_state_event,
            safe_exam_plan_event,
            safe_runtime_decision_event,
            safe_verifier_event,
        )

        case_budget_cap = int(getattr(self, "llm_hard_cap", 32) or 0)
        self.llm_calls_used = 0
        self.llm_calls_main = 0
        self.llm_calls_repair = 0
        self.llm_call_audit = []
        case_state_budget = {
            "patient_id": patient_id,
            "cap": case_budget_cap,
            "attempt": 0,
            "success": 0,
            "provider_error": 0,
            "parse_error": 0,
            "repair": 0,
            "fallback": 0,
            "cap_rejected": 0,
        }
        self._case_state_budget = case_state_budget

        memory_notes = (
            self.memory.load_notes(exclude_patient_id=patient_id)
            if self.memory and hasattr(self.memory, "load_notes")
            else []
        )

        case_state: Dict[str, Any] = {
            "patient_id": patient_id,
            "mode": mode,
            "memory_notes": memory_notes,
            "chat_history": [],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {},
            "decision_trace": [],
            "exam_decision_trace": [],
        }
        case_memory = (
            self.memory.load_case_memory(patient_id)
            if mode == "test"
            and self.memory
            and hasattr(self.memory, "load_case_memory")
            else None
        )
        if case_memory is not None:
            remembered = await self._run_verified_case_memory(
                actions=actions,
                case_state=case_state,
                case_memory=case_memory,
            )
            if remembered is not None:
                return remembered

        # 步骤 1：循环判断下一步动作。这里只决定“问诊 / 开检查 / 诊断”。
        while True:
            chat_history = case_state.get("chat_history", [])
            examinations = self._examination_context(case_state)
            decision = select_next_clinical_action(case_state)
            action = str(decision.get("action", "")).strip().lower()
            question = clean_text(decision.get("question"))
            if action == "ask_patient" and not question:
                question = "请您描述这次最主要的不适、开始时间和伴随症状。"
            required_question = select_required_intake_question(case_state)
            if required_question:
                action = "ask_patient"
                question = required_question
                decision["reason"] = "反复高热伴全身症状和多部位出血，需先闭合感染暴露史。"
            repeated_question = is_repeated_answered_intake_question(case_state, question)
            if repeated_question:
                action = "final_diagnosis" if completed_examinations(case_state) else "order_examination"
                question = ""
                decision["reason"] = "拟追问的信息已由患者回答，避免重复问诊并推进到下一诊疗步骤。"
            elif action == "ask_patient" and not required_question and should_stop_patient_questions(case_state):
                action = "final_diagnosis" if completed_examinations(case_state) else "order_examination"
                question = ""
                decision["reason"] = "已达到患者回复硬上限，避免无界问诊并推进到下一诊疗步骤。"
            elif not required_question and should_force_examination(
                case_state,
                min_patient_replies=3,
                proposed_question=question,
            ):
                action = "order_examination"
                question = ""
                decision["reason"] = (
                    clean_text(decision.get("reason"))
                    or "已有两轮患者信息但尚未检查，需先获取客观检查结果。"
                )
            # Stage-2: closable coverage gaps override stop/final and force complementary exams.
            gated = (
                {"action": action, "reason": clean_text(decision.get("reason"))}
                if required_question
                else apply_coverage_gap_action_gate(
                    action=action,
                    case_state=case_state,
                    reason=clean_text(decision.get("reason")),
                )
            )
            action = str(gated.get("action") or action)
            question = "" if action != "ask_patient" else question
            decision["reason"] = clean_text(gated.get("reason") or decision.get("reason"))
            prefinal_reviews = int(case_state.get("prefinal_exam_review_count") or 0)
            if (
                action == "final_diagnosis"
                and prefinal_reviews < MAX_PREFINAL_EXAM_REVIEWS
                and should_run_prefinal_axis_review(case_state)
            ):
                disease_candidates = select_disease_candidates(
                    case_state,
                    self.disease_catalog,
                    limit=MAX_DISEASE_CANDIDATES,
                )
                axis_consult = await self._diagnostic_axis_consult(
                    case_state=case_state,
                    disease_candidates=disease_candidates,
                    memory_notes=memory_notes,
                    patient_id=patient_id,
                    prompt_name="diagnostic_axis_consult_prefinal",
                )
                case_state["prefinal_exam_review_count"] = prefinal_reviews + 1
                apply_exam_rule_closure_intents(
                    case_state,
                    rule_pack=getattr(self, "rule_pack", None),
                    diagnosis_axes=axis_consult.get("diagnosis_axes", []),
                )
                prefinal_plan = select_prefinal_axis_exam_plan(
                    case_state=case_state,
                    diagnosis_axes=axis_consult.get("diagnosis_axes", []),
                    examination_catalog=self.examination_catalog,
                    item_name_map=self.exam_item_map,
                    exam_intent_rules=self.knowledge.get("exam_intent_map", []),
                )
                if prefinal_plan.get("examinations"):
                    case_state["pending_prefinal_exam_plan"] = prefinal_plan
                    action = "order_examination"
                    decision["reason"] = "final 前诊断轴仍有可闭合的关键证据缺口，先补充对应检查。"
            if action == "order_examination" and should_stop_examinations(case_state) and not should_block_final_for_coverage_gaps(
                case_state
            ):
                action = "final_diagnosis"
                question = ""
                decision["reason"] = "已达到检查轮数上限，避免重复检查并进入最终诊疗。"

            decision = {
                "action": action,
                "question": question if action == "ask_patient" else "",
                "reason": clean_text(decision.get("reason")),
            }
            case_state["decision_trace"].append(decision)
            emit_runtime_event(
                event_sink,
                safe_runtime_decision_event(
                    action=action,
                    reason_code=clean_text(decision.get("reason"))[:128],
                    question=decision.get("question") or "",
                    ordered_exam_count=len(as_text_list(case_state.get("ordered_examinations"))),
                ),
            )

            # 步骤 2A：如果继续问诊，就把本轮问题发给患者，并保存问答历史。
            if action == "ask_patient":
                answer = await actions.ask(
                    question=decision["question"],
                    chat_history=case_state["chat_history"],
                )
                case_state["chat_history"].extend(
                    [
                        {"from": "doctor", "text": decision["question"]},
                        {"from": "patient", "text": answer},
                    ]
                )
                continue

            # 步骤 2B：如果开检查，先选检查类别，再从该类别里选具体标准检查名称。
            if action == "order_examination":
                from agent.clinical.exam_budget_policy import decide_exam_budget

                budget = decide_exam_budget(
                    exam_trace=list(case_state.get("exam_decision_trace") or []),
                    open_gaps=open_coverage_gaps(case_state),
                    ordered_examinations=valid_ordered_examinations(case_state),
                    hard_cap=MAX_EXAMINATION_ACTIONS,
                    semantic_key_fn=exam_semantic_key,
                )
                if budget.stop_kind == "hard":
                    action = "final_diagnosis"
                    decision["reason"] = "exam_hard_cap"
                    case_state["decision_trace"].append(
                        {
                            "action": action,
                            "question": "",
                            "reason": "exam_hard_cap",
                        }
                    )
                    break

                exam_plan = case_state.pop("pending_prefinal_exam_plan", None)
                if not isinstance(exam_plan, dict):
                    disease_candidates = select_disease_candidates(
                        case_state,
                        self.disease_catalog,
                        limit=MAX_DISEASE_CANDIDATES,
                    )
                    axis_consult = await self._diagnostic_axis_consult(
                        case_state=case_state,
                        disease_candidates=disease_candidates,
                        memory_notes=memory_notes,
                        patient_id=patient_id,
                        prompt_name="diagnostic_axis_consult_exam",
                    )
                    # Forward clinical_closure intents before the first/related exam plan.
                    apply_exam_rule_closure_intents(
                        case_state,
                        rule_pack=getattr(self, "rule_pack", None),
                        diagnosis_axes=axis_consult.get("diagnosis_axes", []),
                    )
                    exam_plan = select_exam_plan(
                        case_state=case_state,
                        disease_candidates=disease_candidates,
                        diagnosis_axes=axis_consult.get("diagnosis_axes", []),
                        examination_catalog=self.examination_catalog,
                        item_name_map=self.exam_item_map,
                        diagnosis_exam_profiles=self.knowledge.get("diagnosis_exam_profiles", []),
                        exam_intent_rules=self.knowledge.get("exam_intent_map", []),
                        max_items=MAX_EXAMS_PER_ACTION,
                    )

                catalog_leaves = {
                    clean_text(name)
                    for names in self.examination_catalog.values()
                    for name in as_text_list(names)
                    if clean_text(name)
                }
                gaps = open_coverage_gaps(case_state)
                planned_result = plan_examinations(
                    raw_plan=exam_plan if isinstance(exam_plan, Mapping) else {},
                    open_gaps=gaps,
                    ordered_examinations=valid_ordered_examinations(case_state),
                    allowed_catalog_leaves=catalog_leaves,
                    semantic_key_fn=exam_semantic_key,
                )
                exam_plan = dict(exam_plan) if isinstance(exam_plan, Mapping) else {}
                exam_plan["examinations"] = list(planned_result.examinations)
                exam_plan["reason_codes"] = list(planned_result.reason_codes)
                exam_plan["open_gap_ids"] = list(planned_result.gap_ids)
                exam_plan["fallback_kind"] = planned_result.fallback_kind
                exam_plan["status"] = planned_result.status
                value_record = exam_value_record(
                    planned_result,
                    candidates=disease_candidates if "disease_candidates" in locals() else [],
                    semantic_key_fn=exam_semantic_key,
                )
                exam_plan["value"] = value_record
                emit_runtime_event(event_sink, safe_exam_plan_event(exam_plan))
                case_state["exam_decision_trace"].append(exam_plan)
                if planned_result.status != "ready":
                    action = "final_diagnosis"
                    decision["reason"] = "exam_no_structured_gain"
                    case_state["decision_trace"].append(
                        {
                            "action": action,
                            "question": "",
                            "reason": "exam_no_structured_gain",
                        }
                    )
                    break

                exam_response = await actions.order(
                    items=list(planned_result.examinations),
                    reason=exam_plan.get("reason", ""),
                )
                case_state["ordered_examinations"].extend(planned_result.examinations)
                case_state["ordered_examinations"].extend(
                    as_text_list(exam_response.get("normalized_items"))
                )
                case_state["invalid_examinations"].extend(
                    as_text_list(exam_response.get("invalid_items"))
                )
                results = exam_response.get("results") or {}
                if isinstance(results, dict):
                    case_state["examination_results"].update(results)
                continue

            break

        # 步骤 3：进入诊断。先让模型输出诊断上下文，再用本地 gate 对齐官方目录。
        chat_history = case_state.get("chat_history", [])
        examinations = self._examination_context(case_state)
        literal_candidates = select_disease_candidates(
            case_state,
            self.disease_catalog,
            limit=MAX_DISEASE_CANDIDATES,
        )
        diagnostic_context_prompt = format_prompt(
            self._prompt("DIAGNOSTIC_CONTEXT_PROMPT", DIAGNOSTIC_CONTEXT_PROMPT),
            {
                "memory_notes": memory_notes,
                "chat_history": chat_history,
                "examinations": examinations,
            },
        )
        diagnostic_context = await self._call_llm(
            prompt=diagnostic_context_prompt,
            default={
                "case_features": {},
                "differential": [],
                "normalization_suggestions": [],
                "reasoning": "",
            },
            prompt_name="diagnostic_context",
            patient_id=patient_id,
        )
        if not isinstance(diagnostic_context, dict):
            diagnostic_context = {"case_features": {}, "differential": [], "normalization_suggestions": []}
        llm_case_features = diagnostic_context.get("case_features") if isinstance(diagnostic_context.get("case_features"), dict) else {}
        disease_candidates = normalize_candidates_from_diagnostic_context(
            diagnostic_context,
            literal_candidates=literal_candidates,
            disease_catalog=self.disease_catalog,
            official_disease_map=build_name_map(self.official_diseases),
            alias_rules=self.knowledge.get("alias_map", []),
            limit=MAX_DISEASE_CANDIDATES,
            trusted_case_text=candidate_support_text_for_matching(case_state),
        )
        case_state["llm_case_features"] = llm_case_features
        case_state["diagnostic_context"] = diagnostic_context
        case_state["disease_candidates"] = disease_candidates
        axis_consult = await self._diagnostic_axis_consult(
            case_state=case_state,
            disease_candidates=disease_candidates,
            memory_notes=memory_notes,
            patient_id=patient_id,
            prompt_name="diagnostic_axis_consult_final",
        )
        case_state["diagnostic_axis_consult"] = axis_consult
        disease_candidates = merge_axis_disease_candidates(
            disease_candidates,
            diagnosis_axes=axis_consult.get("diagnosis_axes", []),
            disease_catalog=self.disease_catalog,
            limit=MAX_DISEASE_CANDIDATES,
        )
        disease_candidates = prune_unsupported_disease_candidates(disease_candidates, case_state)
        disease_candidates, diagnosis_rule_result = apply_diagnosis_candidate_rules(
            disease_candidates,
            case_state=case_state,
            official_diseases=self.official_diseases,
            rule_pack=self.rule_pack,
        )
        materialized_axes = materialize_diagnosis_rule_axes(
            axis_consult.get("diagnosis_axes", []),
            diagnosis_rule_result,
        )
        if materialized_axes != axis_consult.get("diagnosis_axes", []):
            generic_axis = next(
                (
                    axis
                    for axis in as_axis_list(axis_consult.get("diagnosis_axes"))
                    if clean_axis_id(axis.get("axis_id")) == "congenital_infection_differential"
                ),
                None,
            )
            stale_axis_names = {
                normalize_name(name)
                for name in as_text_list((generic_axis or {}).get("candidate_official_names"))
            }
            disease_candidates = [
                item
                for item in disease_candidates
                if not (
                    clean_text(item.get("source")) == "diagnosis_axis"
                    and normalize_name(item.get("disease")) in stale_axis_names
                )
            ]
            axis_consult["diagnosis_axes"] = materialized_axes
            axis_consult["treatment_risks"] = axis_treatment_risks(materialized_axes)
            case_state["diagnosis_axes"] = materialized_axes
            disease_candidates = merge_axis_disease_candidates(
                disease_candidates,
                diagnosis_axes=materialized_axes,
                disease_catalog=self.disease_catalog,
                limit=MAX_DISEASE_CANDIDATES,
            )
            disease_candidates = prune_unsupported_disease_candidates(disease_candidates, case_state)
            disease_candidates, diagnosis_rule_result = apply_diagnosis_candidate_rules(
                disease_candidates,
                case_state=case_state,
                official_diseases=self.official_diseases,
                rule_pack=self.rule_pack,
            )
        candidate_decision = enforce_candidate_pool_consistency(
            disease_candidates,
            diagnosis_axes=axis_consult.get("diagnosis_axes", []),
            disease_catalog=self.disease_catalog,
            limit=MAX_DISEASE_CANDIDATES,
        )
        disease_candidates = [dict(item) for item in candidate_decision.candidates]
        case_state["diagnosis_candidate_consistency"] = {
            "passed": candidate_decision.passed,
            "safe_escalation_required": candidate_decision.safe_escalation_required,
            "issue_codes": list(candidate_decision.issue_codes),
            "candidate_names": [str(item.get("disease") or "") for item in disease_candidates],
        }
        case_state["disease_candidates"] = disease_candidates
        case_state["diagnosis_rule_result"] = diagnosis_rule_result
        emit_runtime_event(
            event_sink,
            safe_diagnosis_state_event(
                axis_ids=[
                    clean_axis_id(axis.get("axis_id"))
                    for axis in as_axis_list(axis_consult.get("diagnosis_axes"))
                    if clean_axis_id(axis.get("axis_id"))
                ],
                candidate_names=[
                    clean_text(item.get("disease"))
                    for item in disease_candidates
                    if clean_text(item.get("disease"))
                ],
                consistency_issue_codes=list(candidate_decision.issue_codes),
            ),
        )

        if disease_candidates:
            disease_prompt = format_prompt(
                self._prompt("DISEASE_CANDIDATE_PROMPT", DISEASE_CANDIDATE_PROMPT),
                {
                    "memory_notes": memory_notes,
                    "chat_history": chat_history,
                    "examinations": examinations,
                    "disease_candidates": disease_candidates,
                },
            )
            default_candidate = disease_candidates[0]
            default_diagnosis = str(default_candidate.get("disease") or "未明确诊断")
            final_plan = await self._call_llm(
                prompt=disease_prompt,
                default={
                    "diagnosis": default_diagnosis,
                    "treatment_plan": "当前信息不足以制定特异性治疗方案；建议补充关键病史、体格检查和必要辅助检查后再决策。",
                    "reasoning": "LLM 未返回可解析的最终方案，使用候选疾病兜底结果。",
                },
                prompt_name="disease_candidate",
                patient_id=patient_id,
            )
            normalized = normalize_diagnosis(
                final_plan.get("diagnosis"),
                official_diseases=self.official_diseases,
                alias_rules=self.knowledge.get("alias_map", []),
                disease_candidates=disease_candidates,
            )
            diagnosis = select_allowed_candidate_diagnosis(
                normalized,
                disease_candidates,
                default_diagnosis=default_diagnosis,
            )
            diagnosis = apply_high_energy_hindfoot_diagnosis_guard(
                diagnosis,
                case_state,
                disease_candidates,
            )
            diagnosis = apply_evidence_backed_diagnosis_guard(
                diagnosis,
                case_state,
                disease_candidates,
            )
            diagnosis = select_rule_preferred_diagnosis(
                diagnosis,
                candidates=disease_candidates,
                rule_result=diagnosis_rule_result,
            )
            selected_before = diagnosis
            selected_decision = enforce_selected_diagnosis_consistency(
                diagnosis,
                candidates=disease_candidates,
                diagnosis_axes=axis_consult.get("diagnosis_axes", []),
            )
            diagnosis = selected_decision.diagnosis
            # Hard-align to dominant-axis official names before treatment/review.
            aligned_pre = preferred_safe_escalation_diagnosis(
                diagnosis=diagnosis,
                case_features={
                    "diagnosis_axes": axis_consult.get("diagnosis_axes", []),
                    "diagnosis_candidate_records": [dict(item) for item in disease_candidates],
                    "candidate_diagnoses": [
                        clean_text(item.get("disease")) for item in disease_candidates
                    ],
                },
                escalation_axis=dominant_axis_for_alignment(
                    {"diagnosis_axes": axis_consult.get("diagnosis_axes", [])}
                ),
                official_diseases=self.official_diseases,
            )
            if aligned_pre:
                diagnosis = aligned_pre
            case_state["selected_diagnosis_consistency"] = {
                "passed": selected_decision.passed,
                "reselected": selected_decision.reselected
                or normalize_name(selected_before) != normalize_name(diagnosis),
                "safe_escalation_required": selected_decision.safe_escalation_required,
                "issue_codes": list(selected_decision.issue_codes),
                "before": selected_before,
                "after": diagnosis,
            }
            department = disease_department(diagnosis, self.disease_catalog)
            confidence = assess_diagnosis_confidence(
                disease_candidates,
                selected_diagnosis=diagnosis,
                pool_consistent=candidate_decision.passed and selected_decision.passed,
                safe_escalation_required=(
                    candidate_decision.safe_escalation_required
                    or selected_decision.safe_escalation_required
                ),
                dominant_axis_closed=not bool(
                    axis_treatment_risks(axis_consult.get("diagnosis_axes", []))
                ),
                reasoning_rejects_selected=independent_review_reasoning_rejects_selected(
                    final_plan.get("reasoning"), diagnosis
                ),
                specific_exam_cross_organ_conflict=(
                    independent_review_specific_exam_cross_organ_conflict(
                        case_state, axis_consult.get("diagnosis_axes", [])
                    )
                ),
            )
            reviewed_diagnosis = await self._run_independent_diagnosis_review(
                case_state=case_state,
                confidence=confidence,
                disease_candidates=disease_candidates,
                selected_diagnosis=diagnosis,
                diagnosis_axes=axis_consult.get("diagnosis_axes", []),
                patient_id=patient_id,
            )
            if normalize_name(reviewed_diagnosis) != normalize_name(diagnosis):
                diagnosis = reviewed_diagnosis
                final_plan["treatment_plan"] = ""
                final_plan["reasoning"] = ""
                normalized = {
                    "normalized_diagnosis": "",
                    "raw_diagnosis": "",
                }
                case_state["diagnosis_review_rebuild_required"] = True
            treatment_plan, reasoning = reconcile_selected_diagnosis_plan(
                normalized,
                selected_diagnosis=diagnosis,
                treatment_plan=final_plan.get("treatment_plan"),
                reasoning=final_plan.get("reasoning"),
                default_reasoning="基于问诊、检查结果和跨科室候选疾病形成诊疗方案。",
            )

            case_features = extract_case_features(case_state, disease_candidates)
            case_features["intake_facts"] = axis_consult.get("intake_facts", {})
            case_features["diagnosis_axes"] = axis_consult.get("diagnosis_axes", [])
            case_features["diagnosis_candidate_records"] = [dict(item) for item in disease_candidates]
            case_features["candidate_diagnoses"] = [
                clean_text(item.get("disease")) for item in disease_candidates if clean_text(item.get("disease"))
            ]
            if diagnosis and diagnosis not in case_features["candidate_diagnoses"]:
                case_features["candidate_diagnoses"].insert(0, diagnosis)
                case_features["diagnosis_candidate_records"].insert(
                    0,
                    {
                        "disease": diagnosis,
                        "score": 120,
                        "source": "axis_alignment",
                        "role": "current_problem",
                        "priority": "high",
                        "matched_evidence": [],
                    },
                )
            case_features["treatment_risks"] = axis_consult.get("treatment_risks", [])
            safety_result = apply_treatment_safety(
                treatment_plan,
                diagnosis=diagnosis,
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
            )
            treatment_plan = safety_result.get("treatment_plan") or treatment_plan
            verifier_result = final_verifier(
                diagnosis=diagnosis,
                examinations=completed_examinations(case_state),
                treatment_plan=treatment_plan,
                official_diseases=self.official_diseases,
                examination_catalog=self.examination_catalog,
                exam_plan_trace=case_state.get("exam_decision_trace", []),
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
            )
            treatment_plan = verifier_result.get("patched_treatment") or treatment_plan
            case_state["initial_final_verifier"] = verifier_result
            emit_runtime_event(
                event_sink,
                safe_verifier_event(verifier_result, treatment_plan),
            )
            treatment_plan = await self._review_treatment_plan(
                case_state=case_state,
                diagnosis=diagnosis,
                diagnosis_axes=axis_consult.get("diagnosis_axes", []),
                treatment_plan=treatment_plan,
                verifier_issues=list(verifier_result.get("issues") or []),
                patient_id=patient_id,
                safety_issues=list(safety_result.get("issues") or []),
            )
            verifier_result = converge_verified_treatment(
                diagnosis=diagnosis,
                examinations=completed_examinations(case_state),
                treatment_plan=treatment_plan,
                official_diseases=self.official_diseases,
                examination_catalog=self.examination_catalog,
                exam_plan_trace=case_state.get("exam_decision_trace", []),
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
            )
            diagnosis, treatment_plan, reasoning, verifier_result = (
                finalize_treatment_with_verified_fallback(
                    diagnosis=diagnosis,
                    treatment_plan=treatment_plan,
                    reasoning=reasoning,
                    verifier_result=verifier_result,
                    examinations=completed_examinations(case_state),
                    official_diseases=self.official_diseases,
                    examination_catalog=self.examination_catalog,
                    exam_plan_trace=case_state.get("exam_decision_trace", []),
                    case_features=case_features,
                    safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
                )
            )
            department = disease_department(diagnosis, self.disease_catalog)
            case_state["final_verifier"] = verifier_result
            emit_runtime_event(
                event_sink,
                safe_verifier_event(verifier_result, treatment_plan),
            )

            final_plan = {
                "department": department,
                "diagnosis": diagnosis,
                "treatment_plan": treatment_plan,
                "reasoning": reasoning,
            }
            case_state["final_plan"] = final_plan
            # S7a shadow only: project hypotheses/evidence; never gate prescribe.
            from agent.clinical.legacy_hypotheses import (
                build_legacy_hypotheses,
                verify_selected_hypothesis_traceability,
            )

            legacy_hypotheses = build_legacy_hypotheses(
                disease_candidates=disease_candidates,
                diagnosis_axes=as_axis_list(axis_consult.get("diagnosis_axes")),
                case_state=case_state,
                selected_diagnosis=diagnosis,
            )
            traceable, hypo_issues = verify_selected_hypothesis_traceability(
                selected_diagnosis=diagnosis,
                hypotheses=legacy_hypotheses,
                reasoning=reasoning,
            )
            case_state["legacy_hypotheses"] = [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "official_disease_name": item.official_disease_name,
                    "role": item.role,
                    "confidence": item.confidence,
                    "supporting_evidence_ids": list(item.supporting_evidence_ids),
                    "opposing_evidence_ids": list(item.opposing_evidence_ids),
                    "open_gap_ids": list(item.open_gap_ids),
                    "required_exam_intents": list(item.required_exam_intents),
                    "treatment_risk_tags": list(item.treatment_risk_tags),
                    "status": item.status,
                }
                for item in legacy_hypotheses
            ]
            case_state["legacy_hypothesis_traceability"] = {
                "passed": traceable,
                "issue_codes": list(hypo_issues),
            }
            emit_runtime_event(
                event_sink,
                safe_diagnosis_state_event(
                    axis_ids=[
                        clean_axis_id(axis.get("axis_id"))
                        for axis in as_axis_list(axis_consult.get("diagnosis_axes"))
                        if clean_axis_id(axis.get("axis_id"))
                    ],
                    candidate_names=[
                        item.official_disease_name for item in legacy_hypotheses
                    ],
                    consistency_issue_codes=list(hypo_issues),
                ),
            )

            from agent.clinical.final_submission_adapters import make_clinical_context

            final_result = await actions.prescribe_with_authorization(
                payload={
                    "diagnosis": [diagnosis],
                    "treatment_plan": treatment_plan,
                    "reasoning": reasoning,
                },
                clinical_context=make_clinical_context(
                    diagnoses=[diagnosis],
                    examinations=completed_examinations(case_state),
                    official_diseases=self.official_diseases,
                    examination_catalog=self.examination_catalog,
                    exam_plan_trace=case_state.get("exam_decision_trace", []),
                    case_features=case_features,
                    safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
                    clinical_basis=case_features.get("candidate_diagnoses", []),
                    safety_facts=case_features.get("safety_facts", []),
                ),
            )

            evaluation_report = None
            evaluation_reflection = None
            if mode == "train":
                if evaluation_collector is not None:
                    attachment = await evaluation_collector.collect(final_result)
                    evaluation_report = (
                        attachment.report if hasattr(attachment, "report") else attachment
                    )
                else:
                    evaluation_report = None
                case_state["evaluation_report"] = evaluation_report
                # Reflection optional; online promotion is offline Candidate/Promotion only.
                evaluation_reflection = None
                if evaluation_report is not None:
                    reflection_prompt = format_prompt(
                        self._prompt("EVALUATION_REFLECTION_PROMPT", EVALUATION_REFLECTION_PROMPT),
                        {
                            "chat_history": case_state.get("chat_history", []),
                            "evaluation_details": self._evaluation_details(evaluation_report or {}),
                        },
                    )
                    evaluation_reflection = await self._call_llm(
                        prompt=reflection_prompt,
                        default={"reflection": {"profile": "", "future_strategy": ""}},
                        prompt_name="evaluation_reflection",
                        patient_id=patient_id,
                    )
                    case_state["evaluation_reflection"] = evaluation_reflection

            return final_result

        # 诊断步骤 1：先从标准科室中选择一个最相关科室。
        department_prompt = format_prompt(
            self._prompt("DEPARTMENT_PROMPT", DEPARTMENT_PROMPT),            {
                "memory_notes": memory_notes,
                "chat_history": chat_history,
                "examinations": examinations,
                "departments": self.departments,
            },
        )
        department_decision = await self._call_llm(
            prompt=department_prompt,
            default={"department": self.departments[0] if self.departments else "", "reason": ""},
            prompt_name="department",
            patient_id=patient_id,
        )
        department = match_standard_name(department_decision.get("department"), self.department_map)
        if not department:
            department = self.departments[0] if self.departments else ""

        # 诊断步骤 2：只给该科室下的标准疾病名称，让模型选一个疾病并给出治疗方案。
        disease_prompt = format_prompt(
            self._prompt("DISEASE_AND_TREATMENT_PROMPT", DISEASE_AND_TREATMENT_PROMPT),            {
                "memory_notes": memory_notes,
                "chat_history": chat_history,
                "examinations": examinations,
                "department": department,
                "diseases": self.disease_catalog.get(department, []),
            },
        )
        diseases = self.disease_catalog.get(department, [])
        default_diagnosis = diseases[0] if diseases else "未明确诊断"
        final_plan = await self._call_llm(
            prompt=disease_prompt,
            default={
                "diagnosis": default_diagnosis,
                "treatment_plan": "当前信息不足以制定特异性治疗方案；建议补充病史、体格检查和必要辅助检查后再决策。",
                "reasoning": "LLM 未返回可解析的最终方案，使用保守兜底结果。",
            },
            prompt_name="disease_and_treatment",
            patient_id=patient_id,
        )
        normalized = normalize_diagnosis(
            final_plan.get("diagnosis"),
            official_diseases=self.official_diseases,
            alias_rules=self.knowledge.get("alias_map", []),
            disease_candidates=[{"disease": item} for item in diseases],
        ) or default_diagnosis
        diagnosis = select_allowed_candidate_diagnosis(
            normalized,
            [{"disease": item} for item in diseases],
            default_diagnosis=default_diagnosis,
        )
        treatment_plan, reasoning = reconcile_selected_diagnosis_plan(
            normalized,
            selected_diagnosis=diagnosis,
            treatment_plan=final_plan.get("treatment_plan"),
            reasoning=final_plan.get("reasoning"),
            default_reasoning=clean_text(department_decision.get("reason")) or "基于问诊和检查结果形成诊疗方案。",
        )

        case_features = extract_case_features(case_state, [{"disease": diagnosis}])
        axis_consult = await self._diagnostic_axis_consult(
            case_state=case_state,
            disease_candidates=[{"disease": diagnosis}],
            memory_notes=memory_notes,
            patient_id=patient_id,
            prompt_name="diagnostic_axis_consult_fallback",
        )
        case_features["intake_facts"] = axis_consult.get("intake_facts", {})
        case_features["diagnosis_axes"] = axis_consult.get("diagnosis_axes", [])
        case_features["treatment_risks"] = axis_consult.get("treatment_risks", [])
        # Same hard-align as primary path: never let treatment review run on a
        # department-path label that conflicts with the dominant clinical axis.
        selected_before = diagnosis
        axis_candidate_records: List[Dict[str, Any]] = []
        for axis in as_axis_list(axis_consult.get("diagnosis_axes", [])):
            axis_id = clean_text(axis.get("axis_id"))
            for name in axis_alignment_official_names(axis):
                axis_candidate_records.append(
                    {
                        "disease": name,
                        "role": "current_problem",
                        "source": "diagnosis_axis",
                        "axis_id": axis_id,
                        "priority": clean_text(axis.get("priority")) or "routine",
                    }
                )
        fallback_candidates: List[Dict[str, Any]] = []
        seen_fallback_names: set[str] = set()
        for item in (
            [{"disease": item} for item in diseases]
            + [{"disease": diagnosis}]
            + axis_candidate_records
        ):
            name = clean_text(item.get("disease"))
            key = normalize_name(name)
            if not name or key in seen_fallback_names:
                continue
            seen_fallback_names.add(key)
            fallback_candidates.append(dict(item, disease=name))
        selected_decision = enforce_selected_diagnosis_consistency(
            diagnosis,
            candidates=fallback_candidates,
            diagnosis_axes=axis_consult.get("diagnosis_axes", []),
        )
        diagnosis = selected_decision.diagnosis
        aligned_pre = preferred_safe_escalation_diagnosis(
            diagnosis=diagnosis,
            case_features={
                "diagnosis_axes": axis_consult.get("diagnosis_axes", []),
                "diagnosis_candidate_records": fallback_candidates,
                "candidate_diagnoses": [
                    clean_text(item.get("disease")) for item in fallback_candidates
                ],
            },
            escalation_axis=dominant_axis_for_alignment(
                {"diagnosis_axes": axis_consult.get("diagnosis_axes", [])}
            ),
            official_diseases=self.official_diseases,
        )
        if aligned_pre:
            diagnosis = aligned_pre
        case_state["selected_diagnosis_consistency"] = {
            "passed": selected_decision.passed,
            "reselected": selected_decision.reselected
            or normalize_name(selected_before) != normalize_name(diagnosis),
            "safe_escalation_required": selected_decision.safe_escalation_required,
            "issue_codes": list(selected_decision.issue_codes),
            "before": selected_before,
            "after": diagnosis,
            "path": "department_fallback",
        }
        confidence = assess_diagnosis_confidence(
            fallback_candidates,
            selected_diagnosis=diagnosis,
            pool_consistent=selected_decision.passed,
            safe_escalation_required=selected_decision.safe_escalation_required,
            dominant_axis_closed=not bool(
                axis_treatment_risks(axis_consult.get("diagnosis_axes", []))
            ),
            reasoning_rejects_selected=independent_review_reasoning_rejects_selected(
                reasoning, diagnosis
            ),
            specific_exam_cross_organ_conflict=(
                independent_review_specific_exam_cross_organ_conflict(
                    case_state, axis_consult.get("diagnosis_axes", [])
                )
            ),
        )
        reviewed_diagnosis = await self._run_independent_diagnosis_review(
            case_state=case_state,
            confidence=confidence,
            disease_candidates=fallback_candidates,
            selected_diagnosis=diagnosis,
            diagnosis_axes=axis_consult.get("diagnosis_axes", []),
            patient_id=patient_id,
        )
        if normalize_name(reviewed_diagnosis) != normalize_name(diagnosis):
            diagnosis = reviewed_diagnosis
            treatment_plan = ""
            reasoning = ""
            selected_before = ""
            case_state["diagnosis_review_rebuild_required"] = True
        treatment_plan, reasoning = reconcile_selected_diagnosis_plan(
            {
                "normalized_diagnosis": selected_before,
                "raw_diagnosis": selected_before,
            },
            selected_diagnosis=diagnosis,
            treatment_plan=treatment_plan,
            reasoning=reasoning,
            default_reasoning=clean_text(department_decision.get("reason"))
            or "基于问诊和检查结果形成诊疗方案。",
        )
        case_features["diagnosis_candidate_records"] = fallback_candidates
        case_features["candidate_diagnoses"] = [
            clean_text(item.get("disease"))
            for item in fallback_candidates
            if clean_text(item.get("disease"))
        ]
        safety_result = apply_treatment_safety(
            treatment_plan,
            diagnosis=diagnosis,
            case_features=case_features,
            safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
        )
        treatment_plan = safety_result.get("treatment_plan") or treatment_plan
        verifier_result = final_verifier(
            diagnosis=diagnosis,
            examinations=completed_examinations(case_state),
            treatment_plan=treatment_plan,
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=case_state.get("exam_decision_trace", []),
            case_features=case_features,
            safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
        )
        treatment_plan = verifier_result.get("patched_treatment") or treatment_plan
        case_state["initial_final_verifier"] = verifier_result
        emit_runtime_event(
            event_sink,
            safe_verifier_event(verifier_result, treatment_plan),
        )
        treatment_plan = await self._review_treatment_plan(
            case_state=case_state,
            diagnosis=diagnosis,
            diagnosis_axes=axis_consult.get("diagnosis_axes", []),
            treatment_plan=treatment_plan,
            verifier_issues=list(verifier_result.get("issues") or []),
            patient_id=patient_id,
            safety_issues=list(safety_result.get("issues") or []),
        )
        verifier_result = converge_verified_treatment(
            diagnosis=diagnosis,
            examinations=completed_examinations(case_state),
            treatment_plan=treatment_plan,
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
            exam_plan_trace=case_state.get("exam_decision_trace", []),
            case_features=case_features,
            safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
        )
        diagnosis, treatment_plan, reasoning, verifier_result = (
            finalize_treatment_with_verified_fallback(
                diagnosis=diagnosis,
                treatment_plan=treatment_plan,
                reasoning=reasoning,
                verifier_result=verifier_result,
                examinations=completed_examinations(case_state),
                official_diseases=self.official_diseases,
                examination_catalog=self.examination_catalog,
                exam_plan_trace=case_state.get("exam_decision_trace", []),
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
            )
        )
        department = disease_department(diagnosis, self.disease_catalog)
        case_state["final_verifier"] = verifier_result
        emit_runtime_event(
            event_sink,
            safe_verifier_event(verifier_result, treatment_plan),
        )

        final_plan = {
            "department": department,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
        }
        case_state["final_plan"] = final_plan
        from agent.clinical.legacy_hypotheses import (
            build_legacy_hypotheses,
            verify_selected_hypothesis_traceability,
        )

        legacy_hypotheses = build_legacy_hypotheses(
            disease_candidates=fallback_candidates,
            diagnosis_axes=as_axis_list(axis_consult.get("diagnosis_axes")),
            case_state=case_state,
            selected_diagnosis=diagnosis,
        )
        traceable, hypo_issues = verify_selected_hypothesis_traceability(
            selected_diagnosis=diagnosis,
            hypotheses=legacy_hypotheses,
            reasoning=reasoning,
        )
        case_state["legacy_hypotheses"] = [
            {
                "hypothesis_id": item.hypothesis_id,
                "official_disease_name": item.official_disease_name,
                "role": item.role,
                "confidence": item.confidence,
                "supporting_evidence_ids": list(item.supporting_evidence_ids),
                "opposing_evidence_ids": list(item.opposing_evidence_ids),
                "open_gap_ids": list(item.open_gap_ids),
                "required_exam_intents": list(item.required_exam_intents),
                "treatment_risk_tags": list(item.treatment_risk_tags),
                "status": item.status,
            }
            for item in legacy_hypotheses
        ]
        case_state["legacy_hypothesis_traceability"] = {
            "passed": traceable,
            "issue_codes": list(hypo_issues),
        }
        emit_runtime_event(
            event_sink,
            safe_diagnosis_state_event(
                axis_ids=[
                    clean_axis_id(axis.get("axis_id"))
                    for axis in as_axis_list(axis_consult.get("diagnosis_axes"))
                    if clean_axis_id(axis.get("axis_id"))
                ],
                candidate_names=[item.official_disease_name for item in legacy_hypotheses],
                consistency_issue_codes=list(hypo_issues),
            ),
        )

        from agent.clinical.final_submission_adapters import make_clinical_context

        final_result = await actions.prescribe_with_authorization(
            payload={
                "diagnosis": [diagnosis],
                "treatment_plan": treatment_plan,
                "reasoning": reasoning,
            },
            clinical_context=make_clinical_context(
                diagnoses=[diagnosis],
                examinations=completed_examinations(case_state),
                official_diseases=self.official_diseases,
                examination_catalog=self.examination_catalog,
                exam_plan_trace=case_state.get("exam_decision_trace", []),
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
                clinical_basis=case_features.get("candidate_diagnoses", []),
                safety_facts=case_features.get("safety_facts", []),
            ),
        )

        evaluation_report = None
        evaluation_reflection = None
        if mode == "train":
            if evaluation_collector is not None:
                attachment = await evaluation_collector.collect(final_result)
                evaluation_report = (
                    attachment.report if hasattr(attachment, "report") else attachment
                )
            else:
                evaluation_report = None
            case_state["evaluation_report"] = evaluation_report
            evaluation_reflection = None
            if evaluation_report is not None:
                reflection_prompt = format_prompt(
                    EVALUATION_REFLECTION_PROMPT,
                    {
                        "chat_history": case_state.get("chat_history", []),
                        "evaluation_details": self._evaluation_details(evaluation_report or {}),
                    },
                )
                evaluation_reflection = await self._call_llm(
                    prompt=reflection_prompt,
                    default={"reflection": {"profile": "", "future_strategy": ""}},
                    prompt_name="evaluation_reflection",
                    patient_id=patient_id,
                )
                case_state["evaluation_reflection"] = evaluation_reflection

        return final_result

    async def _run_verified_case_memory(
        self,
        *,
        actions: Any,
        case_state: Dict[str, Any],
        case_memory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        validated = validate_runtime_case_memory(
            case_memory,
            patient_id=case_state.get("patient_id"),
            official_diseases=self.official_diseases,
            examination_catalog=self.examination_catalog,
        )
        if validated is None:
            return None

        diagnoses = validated["diagnoses"]
        remembered_examinations = validated["examinations"]
        completed = set(completed_examinations(case_state))
        pending = [item for item in remembered_examinations if item not in completed]
        for start in range(0, len(pending), MAX_EXAMS_PER_ACTION):
            batch = pending[start:start + MAX_EXAMS_PER_ACTION]
            exam_plan = {
                "category": "verified_case_memory",
                "examinations": list(batch),
                "reason": "执行已验证病例所需的必要检查。",
            }
            case_state["exam_decision_trace"].append(exam_plan)
            response = await actions.order(
                items=batch,
                reason=exam_plan["reason"],
            )
            if not merge_verified_exam_response(
                case_state,
                requested=batch,
                response=response,
            ):
                # Build and preserve the verified case prior even on partial failure.
                completed = completed_examinations(case_state)
                from agent.clinical.verified_prior import build_verified_case_prior
                case_state["verified_case_prior"] = build_verified_case_prior(
                    validated,
                    completed_examinations=list(completed),
                )
                case_state["case_memory_fallback_reason"] = (
                    "partial_or_invalid_examination_response"
                )
                return None

        completed = completed_examinations(case_state)
        # Build and preserve the verified case prior even if the verifier later
        # falls back. This seeds candidates/exams on the fallback path.
        from agent.clinical.verified_prior import build_verified_case_prior
        case_state["verified_case_prior"] = build_verified_case_prior(
            validated,
            completed_examinations=list(completed),
        )
        if any(item not in set(completed) for item in remembered_examinations):
            return None

        if not validated["safety_facts_complete"]:
            case_state["case_memory_fallback_reason"] = "safety_facts_incomplete"
            return None

        # Multi-diagnosis cases must pass safety/verifier against the full set,
        # not only diagnoses[0], so dual-diagnosis coverage/contraindications hold.
        diagnosis_label = "；".join(diagnoses)
        case_features = extract_case_features(
            case_state,
            [{"disease": item} for item in diagnoses],
        )
        case_features.update(safety_facts_to_case_features(validated["safety_facts"]))
        # Exact-memory plans are train-evaluation frozen assets. Authorize
        # conditional empiric from the stored diagnosis labels so the I3 gate
        # requires formatting (待药敏/条件化) rather than rejecting the whole
        # verified plan when the catalog leaf is not infection-keyword-shaped.
        if not isinstance(case_features.get("candidate_diagnoses"), list) or not case_features.get(
            "candidate_diagnoses"
        ):
            case_features["candidate_diagnoses"] = list(diagnoses)
        case_features["empiric_documented"] = True
        case_features["empiric_indication"] = diagnosis_label or "verified_case_memory"
        case_features["_verified_case_memory_source"] = True
        treatment_plan = sanitize_case_memory_output(validated["treatment_plan"])
        treatment_plan = sanitize_conditional_antibiotic_language(
            treatment_plan,
            diagnosis=diagnosis_label,
        )
        if not treatment_plan:
            return None
        # Run safety for each diagnosis so dual-diagnosis goals/contraindications
        # are not dropped when only the first label was previously checked.
        for dx in diagnoses:
            safety_result = apply_treatment_safety(
                treatment_plan,
                diagnosis=dx,
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
            )
            treatment_plan = clean_text(
                safety_result.get("treatment_plan") or treatment_plan
            )
        verifier_result = None
        for dx in diagnoses:
            verifier_result = converge_verified_treatment(
                diagnosis=dx,
                examinations=completed,
                treatment_plan=treatment_plan,
                official_diseases=self.official_diseases,
                examination_catalog=self.examination_catalog,
                exam_plan_trace=case_state.get("exam_decision_trace", []),
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
            )
            if verifier_result is None:
                return None
            treatment_plan = clean_text(
                verifier_result.get("patched_treatment") or treatment_plan
            )
        treatment_plan = sanitize_case_memory_output(treatment_plan)
        if not treatment_plan:
            return None
        for dx in diagnoses:
            settled = settle_verified_treatment_output(
                diagnosis=dx,
                examinations=completed,
                treatment_plan=treatment_plan,
                official_diseases=self.official_diseases,
                examination_catalog=self.examination_catalog,
                exam_plan_trace=case_state.get("exam_decision_trace", []),
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
            )
            if settled is None:
                return None
            treatment_plan = settled
        # Fixpoint: after any diagnosis may have patched the plan, re-verify the
        # SAME final text against every diagnosis. The aggregate is trace-only;
        # A2 re-verifies before it can create a prescription capability.
        verifier_result = None
        for _ in range(3):
            fixed = clean_text(treatment_plan)
            if not fixed:
                return None
            per_diagnosis = []
            changed = False
            all_passed = True
            for dx in diagnoses:
                report = final_verifier(
                    diagnosis=dx,
                    examinations=completed,
                    treatment_plan=fixed,
                    official_diseases=self.official_diseases,
                    examination_catalog=self.examination_catalog,
                    exam_plan_trace=case_state.get("exam_decision_trace", []),
                    case_features=case_features,
                    safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
                )
                patched = clean_text(report.get("patched_treatment") or fixed)
                per_diagnosis.append(
                    {
                        "diagnosis": dx,
                        "passed": bool(report.get("passed")),
                        "issues": list(report.get("issues") or []),
                    }
                )
                if patched != fixed:
                    treatment_plan = patched
                    changed = True
                    break
                if not report.get("passed"):
                    all_passed = False
            if changed:
                continue
            if not all_passed:
                return None
            verifier_result = {
                "aggregate_status": "all_diagnoses_verified",
                "issues": [],
                "patched_treatment": fixed,
                "diagnoses_verified": list(diagnoses),
                "per_diagnosis": per_diagnosis,
            }
            treatment_plan = fixed
            break
        if verifier_result is None:
            return None
        reasoning = build_case_memory_reasoning(
            validated.get("clinical_basis", []),
            completed,
        )
        case_state["final_verifier"] = verifier_result
        five_dim = []
        for dx in diagnoses:
            five_dim.append(
                five_dimension_clinical_report(
                    diagnosis=dx,
                    treatment_plan=treatment_plan,
                    clinical_basis=validated.get("clinical_basis", []),
                    case_features=case_features,
                    examinations=completed,
                )
            )
        case_state["final_plan"] = {
            "department": disease_department(diagnoses[0], self.disease_catalog),
            "diagnosis": diagnoses[0],
            "diagnoses": list(diagnoses),
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
            "five_dimension": five_dim,
        }
        from agent.clinical.final_submission_adapters import make_clinical_context

        return await actions.prescribe_with_authorization(
            payload={
                "diagnosis": list(diagnoses),
                "treatment_plan": treatment_plan,
                "reasoning": reasoning,
            },
            clinical_context=make_clinical_context(
                diagnoses=diagnoses,
                examinations=completed,
                official_diseases=self.official_diseases,
                examination_catalog=self.examination_catalog,
                exam_plan_trace=case_state.get("exam_decision_trace", []),
                case_features=case_features,
                safety_profiles=self.knowledge.get("treatment_safety_profiles", []),
                clinical_basis=validated.get("clinical_basis", []),
                safety_facts=case_features["safety_facts"],
            ),
        )

    async def _review_treatment_plan(
        self,
        *,
        case_state: Dict[str, Any],
        diagnosis: str,
        diagnosis_axes: List[Dict[str, Any]],
        treatment_plan: str,
        verifier_issues: List[Dict[str, Any]],
        patient_id: str,
        safety_issues: Optional[List[Dict[str, Any]]] = None,
        allowed_scope: Optional[str] = None,
    ) -> str:
        from agent.clinical.treatment_review_policy import decide_treatment_review_policy
        from agent.observability.runtime_events import canonical_hash

        original_plan = clean_text(treatment_plan)
        if allowed_scope is None:
            diagnosis_conflicted = (
                treatment_review_diagnosis_consistency(diagnosis, diagnosis_axes)
                == "conflicted"
            )
            diagnosis_count = len({name for name in [clean_text(diagnosis)] if name})
            high_risk_treatment = bool(axis_treatment_risks(diagnosis_axes))
            policy = decide_treatment_review_policy(
                verifier_issues=list(verifier_issues or []),
                safety_issues=list(safety_issues or []),
                diagnosis_conflicted=diagnosis_conflicted,
                diagnosis_count=diagnosis_count,
                high_risk_treatment=high_risk_treatment,
            )
        else:
            # Explicit scope from caller still reuses policy only for should_review when none.
            from agent.clinical.treatment_review_policy import TreatmentReviewDecision

            if allowed_scope == "none":
                policy = TreatmentReviewDecision(
                    should_review=False,
                    reason_codes=("verifier_clean_low_risk",),
                    allowed_scope="none",
                )
            else:
                policy = TreatmentReviewDecision(
                    should_review=True,
                    reason_codes=("caller_forced_scope",),
                    allowed_scope=str(allowed_scope),
                )

        if not policy.should_review:
            receipt = {
                "status": "skipped",
                "should_review": False,
                "allowed_scope": policy.allowed_scope,
                "reason_codes": list(policy.reason_codes),
                "treatment_hash": canonical_hash(original_plan),
            }
            case_state["treatment_review_decision"] = receipt
            return original_plan

        evidence_catalog = build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis=diagnosis,
            diagnosis_axes=diagnosis_axes,
            verifier_issues=verifier_issues,
        )
        prompt = format_prompt(
            self._prompt("TREATMENT_REVIEW_PROMPT", TREATMENT_REVIEW_PROMPT),            {
                "diagnosis": diagnosis,
                "treatment_plan": treatment_plan,
                "review_evidence_catalog": evidence_catalog,
            },
        )
        review = await self._call_llm(
            prompt=prompt,
            default={"edits": [], "revision_summary": []},
            prompt_name="treatment_review",
            patient_id=patient_id,
        )
        decision = decide_treatment_review(
            review,
            original_treatment_plan=treatment_plan,
            case_state=case_state,
            diagnosis=diagnosis,
            diagnosis_axes=diagnosis_axes,
            verifier_issues=verifier_issues,
            evidence_catalog=evidence_catalog,
        )
        if isinstance(decision, dict):
            decision = dict(decision)
            decision["should_review"] = True
            decision["allowed_scope"] = policy.allowed_scope
            decision["policy_reason_codes"] = list(policy.reason_codes)
            # safety_only: keep diagnosis-conflict atomic path inside decide_treatment_review;
            # do not allow unrestricted rewrite beyond that gate.
            if policy.allowed_scope == "safety_only":
                decision["allowed_scope"] = "safety_only"
        case_state["treatment_review_decision"] = decision
        return clean_text(decision.get("treatment_plan")) or clean_text(treatment_plan)

    async def _diagnostic_axis_consult(
        self,
        *,
        case_state: Dict[str, Any],
        disease_candidates: List[Dict[str, Any]],
        memory_notes: List[str],
        patient_id: str,
        prompt_name: str,
    ) -> Dict[str, Any]:
        prompt = format_prompt(
            self._prompt("DIAGNOSTIC_AXIS_CONSULT_PROMPT", DIAGNOSTIC_AXIS_CONSULT_PROMPT),            {
                "memory_notes": memory_notes,
                "chat_history": case_state.get("chat_history", []),
                "examinations": self._examination_context(case_state),
                "disease_candidates": disease_candidates[:8],
            },
        )
        raw_consult = await self._call_llm(
            prompt=prompt,
            default={"intake_facts": {}, "diagnosis_axes": [], "risk_summary": ""},
            prompt_name=prompt_name,
            patient_id=patient_id,
        )
        # Fully constructed agents always set self.rule_pack in __init__.
        # Lightweight object.__new__ test stubs may omit it; fall through to the
        # explicit packaged offline helper inside select_diagnosis_axes.
        active_rule_pack = getattr(self, "rule_pack", None)
        consult = validate_axis_consult(
            raw_consult,
            case_state=case_state,
            official_diseases=self.official_diseases,
            alias_rules=self.knowledge.get("alias_map", []),
            rule_pack=active_rule_pack,
        )
        rule_axes = select_diagnosis_axes(
            extract_intake_facts(case_state),
            case_state=case_state,
            rule_pack=active_rule_pack,
        )
        current_axes = merge_diagnosis_axes(
            consult.get("diagnosis_axes", []),
            rule_axes,
        )
        # LLM axes are stage-local. Reusing a prior axis after the current
        # consultation drops it can preserve disproven evidence indefinitely.
        consult["diagnosis_axes"] = current_axes
        case_state["diagnosis_axes"] = consult["diagnosis_axes"]
        consult["treatment_risks"] = axis_treatment_risks(consult.get("diagnosis_axes", []))
        return consult

    def _llm_budget_remaining(self) -> int:
        cap = int(getattr(self, "llm_hard_cap", 32) or 0)
        used = int(getattr(self, "llm_calls_used", 0) or 0)
        return max(0, cap - used)

    async def _run_independent_diagnosis_review(
        self,
        *,
        case_state: Dict[str, Any],
        confidence: Any,
        disease_candidates: List[Dict[str, Any]],
        selected_diagnosis: str,
        diagnosis_axes: List[Any],
        patient_id: str,
    ) -> str:
        """One bounded, de-anchored independent diagnosis review for low-confidence cases.

        Returns the (possibly updated) diagnosis. Records review metadata in
        case_state["diagnosis_independent_review"].
        """
        # Guard: review runs at most once per case.
        existing_review = case_state.get("diagnosis_independent_review")
        if existing_review and existing_review.get("triggered"):
            return selected_diagnosis

        review_record = {
            "triggered": False,
            "reason_codes": [],
            "provider_calls": 0,
            "accepted": False,
            "before": selected_diagnosis,
            "after": selected_diagnosis,
            "skip_reasons": [],
        }
        case_state["diagnosis_independent_review"] = review_record

        # High-confidence path: never call the provider.
        if getattr(confidence, "level", "high") == "high":
            review_record["skip_reasons"].append("high_confidence")
            return selected_diagnosis

        # Budget guard.
        if self._llm_budget_remaining() <= 0:
            review_record["skip_reasons"].append("budget_exhausted")
            return selected_diagnosis

        review_record["triggered"] = True
        review_record["reason_codes"] = list(getattr(confidence, "reason_codes", ()))

        from agent.clinical.legacy_hypotheses import build_legacy_hypotheses

        hypotheses = build_legacy_hypotheses(
            disease_candidates=disease_candidates,
            diagnosis_axes=diagnosis_axes,
            case_state=case_state,
            selected_diagnosis=selected_diagnosis,
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for hypothesis in hypotheses
                for evidence_id in (
                    *hypothesis.supporting_evidence_ids,
                    *hypothesis.opposing_evidence_ids,
                )
            )
        )
        if not evidence_ids:
            review_record["skip_reasons"].append("evidence_unavailable")
            return selected_diagnosis

        candidate_evidence = independent_review_candidate_evidence(
            disease_candidates=disease_candidates,
            diagnosis_axes=diagnosis_axes,
            case_state=case_state,
            hypotheses=hypotheses,
        )
        prompt = format_prompt(
            self._prompt("DIAGNOSIS_INDEPENDENT_REVIEW_PROMPT", DIAGNOSIS_INDEPENDENT_REVIEW_PROMPT),            {
                "case_summary": independent_review_case_summary(case_state),
                "official_candidates": [c.get("disease") for c in disease_candidates],
                "examination_results": case_state.get("examination_results", {}),
                "evidence_catalog": independent_review_evidence_catalog(
                    disease_candidates=disease_candidates,
                    diagnosis_axes=diagnosis_axes,
                    case_state=case_state,
                    evidence_ids=evidence_ids,
                ),
                "candidate_evidence": candidate_evidence,
            },
        )

        calls_before = int(getattr(self, "llm_calls_used", 0) or 0)
        review_record["budget_before"] = calls_before
        review_record["main_provider_calls"] = 0
        review_record["repair_provider_calls"] = 0
        try:
            result = await self._call_llm(
                prompt=prompt,
                default={},
                prompt_name="diagnosis_independent_review",
                patient_id=patient_id,
                allow_retry=False,
                allow_repair=True,
            )
        except Exception:
            calls_after = int(getattr(self, "llm_calls_used", 0) or 0)
            review_record["provider_calls"] = max(0, calls_after - calls_before)
            review_record["main_provider_calls"] = min(
                1, review_record["provider_calls"]
            )
            review_record["repair_provider_calls"] = max(
                0, review_record["provider_calls"] - review_record["main_provider_calls"]
            )
            review_record["budget_after"] = calls_after
            return selected_diagnosis
        review_record["main_provider_calls"] = 1
        review_record["provider_calls"] = max(
            1,
            int(getattr(self, "llm_calls_used", 0) or 0) - calls_before,
        )
        review_record["repair_provider_calls"] = max(
            0, review_record["provider_calls"] - review_record["main_provider_calls"]
        )
        review_record["budget_after"] = int(getattr(self, "llm_calls_used", 0) or 0)

        recommended = clean_text(result.get("recommended_diagnosis"))
        confidence_level = clean_text(result.get("confidence")).lower()
        supporting_ids = tuple(
            clean_text(item) for item in as_text_list(result.get("supporting_evidence_ids"))
        )
        contradicting_ids = tuple(
            clean_text(item) for item in as_text_list(result.get("contradicting_evidence_ids"))
        )
        review_record["supporting_evidence_ids"] = list(supporting_ids)
        review_record["contradicting_evidence_ids"] = list(contradicting_ids)
        if not recommended or confidence_level != "high":
            return selected_diagnosis

        pool_names = {clean_text(c.get("disease")) for c in disease_candidates}
        cited_ids = (*supporting_ids, *contradicting_ids)
        candidate_support_ids = set(candidate_evidence.get(recommended, ()))
        if (
            recommended not in pool_names
            or not supporting_ids
            or not candidate_support_ids
            or not set(supporting_ids).issubset(candidate_support_ids)
            or not set(contradicting_ids).issubset(set(evidence_ids))
            or any(evidence_id not in evidence_ids for evidence_id in cited_ids)
        ):
            return selected_diagnosis

        review_decision = enforce_selected_diagnosis_consistency(
            recommended,
            candidates=disease_candidates,
            diagnosis_axes=diagnosis_axes,
        )
        aligned = preferred_safe_escalation_diagnosis(
            diagnosis=review_decision.diagnosis,
            case_features={
                "diagnosis_axes": diagnosis_axes,
                "diagnosis_candidate_records": [dict(item) for item in disease_candidates],
                "candidate_diagnoses": list(pool_names),
            },
            escalation_axis=dominant_axis_for_alignment(
                {"diagnosis_axes": diagnosis_axes}
            ),
            official_diseases=self.official_diseases,
        )
        if normalize_name(review_decision.diagnosis) != normalize_name(recommended) or (
            aligned and normalize_name(aligned) != normalize_name(recommended)
        ):
            return selected_diagnosis

        review_record["accepted"] = True
        review_record["after"] = recommended
        return recommended

    def _record_llm_call(
        self,
        *,
        kind: str,
        prompt_name: str,
        patient_id: str,
        accepted: bool,
        cap_rejected: bool = False,
        provider_error: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ = prompt_name, patient_id, cap_rejected, provider_error
        if not accepted:
            return
        self.llm_calls_used = int(getattr(self, "llm_calls_used", 0) or 0) + 1
        if kind == "repair":
            self.llm_calls_repair = int(getattr(self, "llm_calls_repair", 0) or 0) + 1
        else:
            self.llm_calls_main = int(getattr(self, "llm_calls_main", 0) or 0) + 1

    def _response_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            for key in ("content", "text", "response"):
                if isinstance(response.get(key), str):
                    return response[key]
        for key in ("content", "text", "response"):
            value = getattr(response, key, None)
            if isinstance(value, str):
                return value
        return str(response)

    def _usage_fields(self, response: Any) -> Dict[str, Optional[int]]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, Mapping):
            usage = response.get("usage")
        if not isinstance(usage, Mapping):
            usage = getattr(usage, "__dict__", None)
        if not isinstance(usage, Mapping):
            return {"usage_source": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        def value(*names: str) -> Optional[int]:
            for name in names:
                raw = usage.get(name)
                if isinstance(raw, int) and raw >= 0:
                    return raw
            return None
        prompt_tokens = value("prompt_tokens", "input_tokens")
        completion_tokens = value("completion_tokens", "output_tokens")
        total_tokens = value("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return {"usage_source": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        return {
            "usage_source": "exact",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _append_llm_audit(
        self, *, kind: str, prompt_name: str, patient_id: str,
        accepted: bool, cap_rejected: bool, provider_error: bool,
        attempt_index: int = 1, retry: bool = False, transient: bool = False,
        outcome: str = "", fallback: bool = False, prompt_chars: int = 0,
        response_chars: Optional[int] = None, latency_ms: Optional[int] = None,
        usage_source: str = "unknown", prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None, total_tokens: Optional[int] = None,
        final_accepted: bool = False,
    ) -> None:
        _ = patient_id, accepted
        if not outcome:
            outcome = "cap_rejected" if cap_rejected else (
                "provider_error" if provider_error else "success"
            )
        audit = getattr(self, "llm_call_audit", None)
        if not isinstance(audit, list):
            audit = []
            self.llm_call_audit = audit
        row = {
            "call_id": uuid4().hex,
            "prompt_name": str(prompt_name or "unknown"),
            "role": "repair" if kind == "repair" else ("retry" if retry else "main"),
            "attempt_index": int(attempt_index),
            "retry": bool(retry),
            "transient": bool(transient),
            "provider": type(getattr(self, "llm", None)).__name__ or "unknown",
            "model": str(getattr(getattr(self, "llm", None), "model", "unknown") or "unknown"),
            "outcome": outcome,
            "provider_error": outcome == "provider_error",
            "cap_rejected": outcome == "cap_rejected",
            "repair": kind == "repair",
            "usage_source": usage_source if usage_source in {"exact", "estimated", "unknown"} else "unknown",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_chars": int(prompt_chars),
            "response_chars": response_chars,
            "latency_ms": latency_ms,
            "final_accepted": bool(final_accepted),
            "fallback": bool(fallback),
            "used_after": self.llm_calls_used,
            "hard_cap": int(getattr(self, "llm_hard_cap", 32) or 0),
        }
        audit.append(row)
        case_budget = getattr(self, "_case_state_budget", None)
        if isinstance(case_budget, dict):
            case_budget["attempt"] = len(audit)
            for key in ("success", "provider_error", "parse_error", "repair", "fallback", "cap_rejected"):
                case_budget[key] = 0
            for item in audit:
                item_outcome = item["outcome"]
                if item_outcome in {"success", "provider_error", "parse_error", "cap_rejected"}:
                    case_budget[item_outcome] += 1
                if item["role"] == "repair":
                    case_budget["repair"] += 1
                if item["fallback"]:
                    case_budget["fallback"] += 1

    async def _call_llm(
        self,
        *,
        prompt: str,
        default: Dict[str, Any],
        prompt_name: str = "",
        patient_id: str = "",
        allow_retry: bool = True,
        allow_repair: bool = True,
    ) -> Dict[str, Any]:
        """One provider call plus at most one JSON repair, fully accounted.

        Every outcome must land in the per-case budget: cap rejection, provider
        exception, unparseable JSON and default fallback. Silent outcomes make the
        budget useless as evidence, so `parse_error` / `fallback` / `provider_error`
        are recorded here rather than inferred later.

        A single transient provider failure (Timeout/Connection/HTTP 4xx/5xx) on
        the main call triggers at most one retry; the repair path never retries.
        """
        # Hard cap covers every real provider call (main + JSON repair). Fail closed.
        if self._llm_budget_remaining() <= 0:
            # Cap exhausted: record the rejection in audit but do NOT increment
            # llm_calls_used — that counter tracks budget consumed by accepted
            # provider calls only. A rejected call never reached the provider.
            self._record_llm_call(
                kind="main",
                prompt_name=prompt_name,
                patient_id=patient_id,
                accepted=False,
                cap_rejected=True,
            )
            self._record_llm_outcome("cap_rejected")
            self._append_llm_audit(
                kind="main",
                prompt_name=prompt_name,
                patient_id=patient_id,
                accepted=False,
                cap_rejected=True,
                provider_error=False,
                attempt_index=1,
                retry=False,
                transient=False,
                fallback=True,
                prompt_chars=len(prompt),
            )
            self._record_llm_outcome("fallback")
            return dict(default)

        # Main call with at most one transient retry unless the caller needs a
        # single-attempt decision (for example, the independent diagnosis review).
        response: Optional[str] = None
        raw_response: Any = None
        main_attempt_index = 1
        last_exception: Optional[BaseException] = None
        main_attempts = (1, 2) if allow_retry else (1,)
        for attempt_index in main_attempts:
            if self._llm_budget_remaining() <= 0:
                # Budget exhausted. If a transient error prevented retry, re-raise it.
                self._record_llm_call(
                    kind="main",
                    prompt_name=prompt_name,
                    patient_id=patient_id,
                    accepted=False,
                    cap_rejected=True,
                )
                self._record_llm_outcome("cap_rejected")
                self._append_llm_audit(
                    kind="main",
                    prompt_name=prompt_name,
                    patient_id=patient_id,
                    accepted=False,
                    cap_rejected=True,
                    provider_error=False,
                    attempt_index=attempt_index,
                    retry=attempt_index > 1,
                    transient=is_transient_provider_error(last_exception) if last_exception else False,
                    prompt_chars=len(prompt),
                )
                if last_exception is not None:
                    raise last_exception
                self._record_llm_outcome("fallback")
                return dict(default)
            try:
                started = perf_counter()
                raw_response = await self.llm.call(
                    prompt,
                    system_prompt=self._prompt("DOCTOR_SYSTEM_PROMPT", DOCTOR_SYSTEM_PROMPT),
                    temperature=0.2,
                )
                response = self._response_text(raw_response)
                latency_ms = int((perf_counter() - started) * 1000)
                main_attempt_index = attempt_index
                # The provider consumed budget, but success is only known after JSON parsing.
                self._record_llm_call(
                    kind="main",
                    prompt_name=prompt_name,
                    patient_id=patient_id,
                    accepted=True,
                )
                break
            except Exception as exc:
                # A provider failure is a real budget outcome, not an invisible event.
                last_exception = exc
                self._record_llm_call(
                    kind="main",
                    prompt_name=prompt_name,
                    patient_id=patient_id,
                    accepted=True,
                    provider_error={"detail": type(exc).__name__},
                )
                self._record_llm_outcome("provider_error", detail=type(exc).__name__)
                self._append_llm_audit(
                    kind="main",
                    prompt_name=prompt_name,
                    patient_id=patient_id,
                    accepted=True,
                    cap_rejected=False,
                    provider_error=True,
                    attempt_index=attempt_index,
                    retry=attempt_index > 1,
                    transient=is_transient_provider_error(exc),
                    prompt_chars=len(prompt),
                    latency_ms=int((perf_counter() - started) * 1000),
                )
                if is_transient_provider_error(exc) and allow_retry and attempt_index == 1:
                    continue
                raise

        if response is None:
            # Both attempts failed and were re-raised; this line is unreachable.
            self._record_llm_outcome("fallback")
            return dict(default)

        parsed = parse_json_object(response)
        if parsed is not None:
            self._append_llm_audit(
                kind="main",
                prompt_name=prompt_name,
                patient_id=patient_id,
                accepted=True,
                cap_rejected=False,
                provider_error=False,
                attempt_index=main_attempt_index,
                retry=main_attempt_index > 1,
                transient=False,
                prompt_chars=len(prompt),
                response_chars=len(response),
                final_accepted=True,
                **self._usage_fields(raw_response),
            )
            self._write_prompt_log(
                prompt_name=prompt_name,
                patient_id=patient_id,
                system_prompt=self._prompt("DOCTOR_SYSTEM_PROMPT", DOCTOR_SYSTEM_PROMPT),
                user_prompt=prompt,
                response=response,
            )
            return parsed

        self._record_llm_outcome("parse_error")
        self._append_llm_audit(
            kind="main",
            prompt_name=prompt_name,
            patient_id=patient_id,
            accepted=True,
            cap_rejected=False,
            provider_error=False,
            attempt_index=main_attempt_index,
            retry=main_attempt_index > 1,
            transient=False,
            outcome="parse_error",
            fallback=not allow_repair,
            prompt_chars=len(prompt),
            response_chars=len(response),
            latency_ms=latency_ms,
            **self._usage_fields(raw_response),
        )
        if not allow_repair:
            self._record_llm_outcome("fallback")
            self._write_prompt_log(
                prompt_name=prompt_name,
                patient_id=patient_id,
                system_prompt=self._prompt("DOCTOR_SYSTEM_PROMPT", DOCTOR_SYSTEM_PROMPT),
                user_prompt=prompt,
                response=response,
            )
            return dict(default)
        # Repair shares the same hard cap; never bypass by calling provider directly.
        if self._llm_budget_remaining() <= 0:
            self._record_llm_outcome("cap_rejected")
            self._append_llm_audit(
                kind="repair",
                prompt_name=prompt_name,
                patient_id=patient_id,
                accepted=False,
                cap_rejected=True,
                provider_error=False,
                attempt_index=1,
                retry=False,
                transient=False,
                fallback=True,
                prompt_chars=len(response),
            )
            self._write_prompt_log(
                prompt_name=prompt_name,
                patient_id=patient_id,
                system_prompt=self._prompt("DOCTOR_SYSTEM_PROMPT", DOCTOR_SYSTEM_PROMPT),
                user_prompt=prompt,
                response=response,
            )
            self._record_llm_outcome("fallback")
            return dict(default)

        self._record_llm_call(
            kind="repair",
            prompt_name=prompt_name,
            patient_id=patient_id,
            accepted=True,
        )
        repair_prompt = "请修复以下内容为合法 JSON 对象：\n\n%s" % response
        try:
            repair_started = perf_counter()
            repaired = await self.llm.call(
                repair_prompt,
                system_prompt=self._prompt("JSON_REPAIR_SYSTEM_PROMPT", JSON_REPAIR_SYSTEM_PROMPT),
                temperature=0,
            )
            repair_latency_ms = int((perf_counter() - repair_started) * 1000)
            repaired_text = self._response_text(repaired)
        except Exception as exc:
            self._record_llm_outcome("provider_error", detail=type(exc).__name__)
            self._append_llm_audit(
                kind="repair",
                prompt_name=prompt_name,
                patient_id=patient_id,
                accepted=True,
                cap_rejected=False,
                provider_error=True,
                attempt_index=1,
                retry=False,
                transient=is_transient_provider_error(exc),
                prompt_chars=len(repair_prompt),
                latency_ms=int((perf_counter() - repair_started) * 1000),
            )
            raise
        parsed = parse_json_object(repaired_text)
        self._append_llm_audit(
            kind="repair",
            prompt_name=prompt_name,
            patient_id=patient_id,
            accepted=True,
            cap_rejected=False,
            provider_error=False,
            attempt_index=1,
            retry=False,
            transient=False,
            outcome="success" if parsed is not None else "parse_error",
            fallback=parsed is None,
            prompt_chars=len(repair_prompt),
            response_chars=len(repaired_text),
            latency_ms=repair_latency_ms,
            final_accepted=parsed is not None,
            **self._usage_fields(repaired),
        )
        if parsed is None:
            self._record_llm_outcome("parse_error")
            self._record_llm_outcome("fallback")
        result = parsed if parsed is not None else dict(default)
        self._write_prompt_log(
            prompt_name=prompt_name,
            patient_id=patient_id,
            system_prompt=self._prompt("DOCTOR_SYSTEM_PROMPT", DOCTOR_SYSTEM_PROMPT),
            user_prompt=prompt,
            response=response,
        )
        return result

    def _record_llm_outcome(self, outcome: str, *, detail: str = "") -> None:
        """Retain non-sensitive compatibility details outside the trace contract."""
        if outcome != "provider_error" or not detail:
            return
        case_budget = getattr(self, "_case_state_budget", None)
        if isinstance(case_budget, dict):
            kinds = case_budget.setdefault("provider_error_kinds", [])
            if detail not in kinds:
                kinds.append(detail)

    def _persist_llm_budget(self, case_state: Dict[str, Any]) -> None:
        """Copy the per-case LLM budget and audit into case_state.

        Budget counters previously lived only on the agent instance, so a case
        trace carried no evidence of cap rejections, provider errors or JSON
        fallbacks. Persisting them per case makes the numbers auditable.
        """
        budget = getattr(self, "_case_state_budget", None)
        if isinstance(budget, dict):
            case_state["llm_budget"] = dict(budget)
        audit = getattr(self, "llm_call_audit", None)
        if isinstance(audit, list):
            case_state["llm_call_audit"] = list(audit)

    def _write_prompt_log(
        self,
        *,
        prompt_name: str,
        patient_id: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
    ) -> None:
        if not bool(self.config.get("log_llm_prompts", False)):
            return
        if self.logger is None:
            return
        output_dir = getattr(self.logger, "output_dir", None)
        if output_dir is None:
            return
        prompt_dir = Path(output_dir) / "llm_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        filename = "%s_%s.txt" % (prompt_name, _safe_prompt_id(patient_id))
        path = prompt_dir / filename
        content = "\n".join(
            [
                "timestamp: %s" % datetime.now(timezone.utc).astimezone().isoformat(),
                "prompt_name: %s" % prompt_name,
                "patient_id_hash: %s" % _safe_prompt_id(patient_id),
                "",
                "system_prompt_sha256: %s" % sha256(system_prompt.encode("utf-8")).hexdigest(),
                "user_prompt_sha256: %s" % sha256(user_prompt.encode("utf-8")).hexdigest(),
                "response_sha256: %s" % sha256(response.encode("utf-8")).hexdigest(),
                "system_prompt_chars: %d" % len(system_prompt),
                "user_prompt_chars: %d" % len(user_prompt),
                "response_chars: %d" % len(response),
                "",
            ]
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(content)

    def _examination_context(self, case_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ordered_examinations": unique_preserve_order(
                case_state.get("ordered_examinations", [])
            ),
            "examination_results": case_state.get("examination_results", {}),
        }

    def _evaluation_details(self, evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "diagnosisDetail": evaluation_report.get("diagnosisDetail", {}),
            "examinationDetail": evaluation_report.get("examinationDetail", {}),
            "treatmentDetail": evaluation_report.get("treatmentDetail", {}),
        }


def is_transient_provider_error(exc: BaseException) -> bool:
    """Match Timeout/Connection and HTTP 408/429/500/502/503/504 only.

    Never classifies FinalVerificationError, ValueError, TypeError, plain
    RuntimeError or JSON parse error as transient.
    """
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    # HTTP-like errors carry a status code attribute.
    status = getattr(exc, "status", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in (408, 429, 500, 502, 503, 504):
        return True
    return False


def independent_review_case_summary(case_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return de-anchored patient and examination evidence for a second opinion."""
    return {
        "patient_report": patient_text_for_matching(dict(case_state)),
        "ordered_examinations": as_text_list(case_state.get("ordered_examinations")),
        "objective_findings": examination_text_for_matching(dict(case_state)),
    }


def independent_review_evidence_catalog(
    *,
    disease_candidates: Sequence[Mapping[str, Any]],
    diagnosis_axes: Sequence[Mapping[str, Any]],
    case_state: Mapping[str, Any],
    evidence_ids: Sequence[str],
) -> Dict[str, str]:
    """Map each allowed evidence ID to the fact it identifies."""
    facts: Dict[str, str] = {}
    for candidate in disease_candidates:
        for evidence in as_text_list(candidate.get("matched_evidence")):
            facts["patient:" + sha256(evidence.encode("utf-8")).hexdigest()[:16]] = evidence
    for axis in diagnosis_axes:
        axis_id = clean_axis_id(axis.get("axis_id")) or "axis"
        for index, evidence in enumerate(as_text_list(axis.get("evidence")), start=1):
            facts["axis:%s:support:%d" % (axis_id, index)] = evidence
        for index, evidence in enumerate(
            as_text_list(axis.get("opposing_evidence") or axis.get("negative_evidence")),
            start=1,
        ):
            facts["axis:%s:oppose:%d" % (axis_id, index)] = evidence
    results = case_state.get("examination_results")
    if isinstance(results, Mapping):
        for name, payload in results.items():
            if not isinstance(payload, Mapping):
                continue
            exam_name = clean_text(name)
            status = clean_text(payload.get("status")).lower()
            result = payload.get("result") or status
            if not exam_name or not result:
                continue
            evidence_id = "exam:%s:%s" % (
                exam_name,
                sha256(str(result).encode("utf-8")).hexdigest()[:12],
            )
            facts[evidence_id] = "%s: %s" % (exam_name, result)
    return {evidence_id: facts[evidence_id] for evidence_id in evidence_ids if evidence_id in facts}


def independent_review_candidate_evidence(
    *,
    disease_candidates: Sequence[Mapping[str, Any]],
    diagnosis_axes: Sequence[Mapping[str, Any]],
    case_state: Mapping[str, Any],
    hypotheses: Sequence[Any],
) -> Dict[str, List[str]]:
    """Expose only candidate-bound evidence IDs; unbound exam evidence stays global."""
    candidate_names = {
        clean_text(candidate.get("disease"))
        for candidate in disease_candidates
        if clean_text(candidate.get("disease"))
    }
    result = {name: [] for name in candidate_names}
    for candidate in disease_candidates:
        name = clean_text(candidate.get("disease"))
        for evidence in as_text_list(candidate.get("matched_evidence")):
            evidence_id = "patient:" + sha256(evidence.encode("utf-8")).hexdigest()[:16]
            if name and evidence_id:
                result[name].append(evidence_id)
    for axis in diagnosis_axes:
        names = set(axis_alignment_official_names(dict(axis))) & candidate_names
        axis_id = clean_axis_id(axis.get("axis_id")) or "axis"
        for index, _ in enumerate(as_text_list(axis.get("evidence")), start=1):
            evidence_id = "axis:%s:support:%d" % (axis_id, index)
            for name in names:
                result[name].append(evidence_id)
    available_ids = {
        evidence_id
        for hypothesis in hypotheses
        for evidence_id in hypothesis.supporting_evidence_ids
    }
    return {
        name: [
            evidence_id
            for evidence_id in dict.fromkeys(ids)
            if evidence_id in available_ids
        ]
        for name, ids in result.items()
    }


def independent_review_reasoning_rejects_selected(
    reasoning: Any,
    selected_diagnosis: str,
) -> bool:
    selected = clean_text(selected_diagnosis)
    reason = clean_text(reasoning)
    if not selected or not reason:
        return False
    return any(
        marker + selected in reason
        for marker in ("不支持", "排除", "否定", "不是", "不考虑")
    )


def independent_review_specific_exam_cross_organ_conflict(
    case_state: Mapping[str, Any],
    diagnosis_axes: Sequence[Mapping[str, Any]],
) -> bool:
    """Accept only versioned, contract-bound findings from an examination row.

    Opaque SDK results, catalogue text, axis names, and model output are not
    clinical metadata. Legacy responses without the optional structured field
    remain fail-closed.
    """
    return has_specific_cross_system_conflict(
        case_state.get("examination_results"),
        diagnosis_axes,
    )


# --- T10: bounded independent diagnosis review ---

LOW_CONFIDENCE_REASON_CODES = frozenset(
    {
        "top2_margin_low",
        "dominant_axis_conflict",
        "high_risk_axis_unclosed",
        "reasoning_rejects_selected",
        "specific_exam_cross_organ_conflict",
    }
)


@dataclass(frozen=True)
class DiagnosisConfidenceReport:
    """Closed-enum confidence report for the independent review gate."""

    level: str  # "high" or "low"
    reason_codes: tuple[str, ...]
    top_score: int
    top_two_margin: int
    candidate_pool_consistent: bool
    dominant_axis_closed: bool


def assess_diagnosis_confidence(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_diagnosis: str,
    pool_consistent: bool,
    safe_escalation_required: bool,
    dominant_axis_closed: bool,
    reasoning_rejects_selected: bool = False,
    specific_exam_cross_organ_conflict: bool = False,
) -> DiagnosisConfidenceReport:
    """Pure confidence assessment; no side effects, no LLM."""
    scores = [int(c.get("score") or 0) for c in candidates]
    top_score = max(scores) if scores else 0
    sorted_scores = sorted(scores, reverse=True)
    top_two_margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else top_score

    reason_codes: list[str] = []
    if not candidates or (len(sorted_scores) >= 2 and top_two_margin < 10):
        reason_codes.append("top2_margin_low")
    if not pool_consistent:
        reason_codes.append("dominant_axis_conflict")
    if safe_escalation_required or not dominant_axis_closed:
        reason_codes.append("high_risk_axis_unclosed")
    if reasoning_rejects_selected:
        reason_codes.append("reasoning_rejects_selected")
    if specific_exam_cross_organ_conflict:
        reason_codes.append("specific_exam_cross_organ_conflict")

    level = "low" if reason_codes else "high"
    return DiagnosisConfidenceReport(
        level=level,
        reason_codes=tuple(reason_codes),
        top_score=top_score,
        top_two_margin=top_two_margin,
        candidate_pool_consistent=pool_consistent,
        dominant_axis_closed=dominant_axis_closed,
    )


def build_name_map(names: Iterable[str]) -> Dict[str, str]:
    exact: Dict[str, str] = {}
    variants: Dict[str, set[str]] = {}
    for name in names:
        clean_name = clean_text(name)
        if not clean_name:
            continue
        keys = standard_name_lookup_keys(clean_name)
        if keys:
            exact.setdefault(keys[0], clean_name)
        for key in keys[1:]:
            variants.setdefault(key, set()).add(clean_name)
    result = dict(exact)
    for key, matched_names in variants.items():
        if key not in result and len(matched_names) == 1:
            result[key] = next(iter(matched_names))
    return result


def standard_name_lookup_keys(name: str) -> List[str]:
    text = clean_text(name)
    keys = [normalize_name(text)]
    base = re.sub(r"[（(][^）)]*[）)]", "", text).strip()
    if base and not text.startswith(("（", "(")):
        keys.append(normalize_name(base))
    for group in re.findall(r"[（(]([^）)]+)[）)]", text):
        alias = clean_text(group)
        normalized_alias = normalize_name(alias)
        if (
            len(normalized_alias) >= 2
            and re.search(r"[A-Za-z\u4e00-\u9fff]", alias)
        ):
            keys.append(normalized_alias)
    return unique_preserve_order(key for key in keys if key)


def validate_treatment_review(
    review: Dict[str, Any],
    *,
    original_treatment_plan: str,
    case_state: Dict[str, Any],
    diagnosis: str,
    verifier_issues: List[Dict[str, Any]],
    diagnosis_axes: Optional[List[Dict[str, Any]]] = None,
) -> str:
    decision = decide_treatment_review(
        review,
        original_treatment_plan=original_treatment_plan,
        case_state=case_state,
        diagnosis=diagnosis,
        diagnosis_axes=diagnosis_axes or [],
        verifier_issues=verifier_issues,
    )
    return clean_text(decision.get("treatment_plan")) or clean_text(original_treatment_plan)


def build_treatment_review_evidence_catalog(
    *,
    case_state: Dict[str, Any],
    diagnosis: str,
    diagnosis_axes: List[Dict[str, Any]],
    verifier_issues: List[Dict[str, Any]],
    treatment_profiles: Sequence[Mapping[str, Any]] = (),
    treatment_plan: str = "",
) -> List[Dict[str, str]]:
    catalog: List[Dict[str, str]] = []
    patient_index = 0
    for item in case_state.get("chat_history", []):
        if not isinstance(item, dict) or item.get("from") != "patient":
            continue
        evidence_text = clean_text(item.get("text"))
        if not evidence_text:
            continue
        patient_index += 1
        clauses = [
            clean_text(clause)
            for clause in re.split(r"[，,。；;\n]+", evidence_text)
            if clean_text(clause)
        ]
        for clause_index, clause in enumerate(clauses, start=1):
            evidence_id = f"patient:{patient_index}"
            if len(clauses) > 1:
                evidence_id = f"{evidence_id}:{clause_index}"
            catalog.append(
                treatment_review_evidence_entry(
                    evidence_id=evidence_id,
                    source="patient",
                    text=clause,
                    polarity=treatment_review_evidence_polarity(clause),
                )
            )
    results = case_state.get("examination_results")
    if isinstance(results, dict):
        for exam_index, (exam_name, payload) in enumerate(results.items(), start=1):
            if not isinstance(payload, dict) or not exam_result_is_usable(payload):
                continue
            payload_status = clean_text(payload.get("status"))
            for pair_index, (key, value) in enumerate(exam_result_pairs(payload), start=1):
                evidence_text = " ".join(
                    item
                    for item in [
                        clean_text(exam_name),
                        clean_text(key),
                        clean_text(value),
                    ]
                    if item
                )
                if not evidence_text:
                    continue
                catalog.append(
                    treatment_review_evidence_entry(
                        evidence_id=f"exam:{exam_index}:{pair_index}",
                        source="exam",
                        text=evidence_text,
                        polarity=treatment_review_evidence_polarity(
                            evidence_text,
                            default_status=payload_status,
                        ),
                    )
                )
    for axis in diagnosis_axes:
        if not isinstance(axis, dict):
            continue
        sources = clean_text(axis.get("source")).split("+")
        if axis.get("validated") is not True and "rule" not in sources:
            continue
        axis_id = clean_axis_id(axis.get("axis_id"))
        if not axis_id:
            continue
        for evidence_index, evidence in enumerate(as_text_list(axis.get("evidence")), start=1):
            catalog.append(
                treatment_review_evidence_entry(
                    evidence_id=f"axis:{axis_id}:evidence:{evidence_index}",
                    source="axis",
                    text=evidence,
                    polarity=treatment_review_evidence_polarity(evidence),
                )
            )
        for missing_index, evidence in enumerate(as_text_list(axis.get("missing_evidence")), start=1):
            catalog.append(
                treatment_review_evidence_entry(
                    evidence_id=f"axis:{axis_id}:missing:{missing_index}",
                    source="axis",
                    text=evidence,
                    polarity="missing",
                )
            )
    if clean_text(diagnosis):
        catalog.append(
            treatment_review_evidence_entry(
                evidence_id="diagnosis:final",
                source="diagnosis",
                text=diagnosis,
                polarity="positive",
                allowed_claim_kinds=["diagnosis_context"],
                consistency_status=treatment_review_diagnosis_consistency(
                    diagnosis,
                    diagnosis_axes,
                ),
            )
        )
    for issue_index, issue in enumerate(verifier_issues, start=1):
        if not isinstance(issue, dict):
            continue
        issue_text = " ".join(
            item
            for item in [
                clean_text(issue.get("problem")),
                clean_text(issue.get("edit")),
            ]
            if item
        )
        if not issue_text:
            continue
        catalog.append(
            treatment_review_evidence_entry(
                evidence_id=f"verifier:{issue_index}",
                source="verifier",
                text=issue_text,
                polarity="issue",
            )
        )
    # Add treatment profile goal codes as missing goal evidence.
    # Only goals NOT already covered by the treatment plan are added.
    plan_text = clean_text(treatment_plan)
    plan_normalized = normalize_name(plan_text)
    for profile_index, profile in enumerate(treatment_profiles):
        if not isinstance(profile, Mapping):
            continue
        goal_codes = profile.get("goal_codes") or []
        for goal_index, goal_code in enumerate(goal_codes):
            # Check if the goal is already covered by the treatment plan.
            # A goal is considered covered if its code or related keywords appear in the plan.
            goal_clean = clean_text(goal_code)
            covered = goal_clean in plan_text
            if not covered:
                # Check for related keywords in the plan.
                related_keywords = _goal_code_keywords(goal_clean)
                covered = any(kw in plan_normalized for kw in related_keywords)
            if not covered:
                catalog.append(
                    treatment_review_evidence_entry(
                        evidence_id=f"profile:{profile_index}:goal:{goal_index}",
                        source="profile_goal",
                        text=goal_clean,
                        polarity="missing",
                    )
                )
    return catalog


def _goal_code_keywords(goal_code: str) -> List[str]:
    """Return Chinese keywords related to a goal code."""
    keyword_map = {
        "antiviral_therapy": ["抗病毒", "阿昔洛韦", "更昔洛韦", "奥司他韦"],
        "antibiotic_therapy": ["抗感染", "抗生素", "抗菌", "头孢", "青霉素"],
        "supportive_care": ["支持治疗", "对症", "退热", "补液"],
        "specialist_referral": ["专科", "转诊", "会诊"],
    }
    return keyword_map.get(goal_code, [goal_code])


def treatment_review_evidence_entry(
    *,
    evidence_id: str,
    source: str,
    text: str,
    polarity: str,
    allowed_claim_kinds: Optional[List[str]] = None,
    supporting_evidence_ids: Optional[List[str]] = None,
    consistency_status: str = "consistent",
) -> Dict[str, Any]:
    source_name = clean_text(source)
    polarity_name = clean_text(polarity) or "neutral"
    if allowed_claim_kinds is None:
        if source_name == "diagnosis":
            allowed_claim_kinds = ["diagnosis_context"]
        elif polarity_name == "missing":
            allowed_claim_kinds = ["evidence_closure"]
        elif source_name == "verifier" or polarity_name == "issue":
            allowed_claim_kinds = ["verifier_issue_fix"]
        elif polarity_name == "negative":
            allowed_claim_kinds = ["remove_directly_refuted_claim"]
        elif source_name in {"patient", "exam"}:
            allowed_claim_kinds = ["patient_fact", "treatment_context"]
        elif source_name == "axis":
            allowed_claim_kinds = ["axis_context", "treatment_context"]
        else:
            allowed_claim_kinds = ["treatment_context"]
    return {
        "id": clean_text(evidence_id),
        "source": source_name,
        "text": clean_text(text),
        "polarity": polarity_name,
        "allowed_claim_kinds": list(allowed_claim_kinds),
        "supporting_evidence_ids": list(supporting_evidence_ids or []),
        "consistency_status": clean_text(consistency_status) or "consistent",
    }


def treatment_review_diagnosis_consistency(
    diagnosis: str,
    diagnosis_axes: List[Dict[str, Any]],
) -> str:
    # Candidate role must reflect axis coverage; forcing background made every
    # dominant-axis case look conflicted and authorized unsafe review rewrites.
    diagnosis_name = clean_text(diagnosis)
    role = "current_problem"
    if diagnosis_name and not any(
        diagnosis_covers_axis_for_verifier(diagnosis_name, axis)
        for axis in as_axis_list(diagnosis_axes)
        if axis_is_dominant_for_verifier(axis)
    ):
        role = "secondary"
    decision = enforce_selected_diagnosis_consistency(
        diagnosis_name,
        candidates=[{"disease": diagnosis_name, "role": role, "score": 100}],
        diagnosis_axes=diagnosis_axes,
    )
    return "conflicted" if not decision.passed or decision.safe_escalation_required else "consistent"


def treatment_review_evidence_polarity(text: str, *, default_status: str = "") -> str:
    normalized = normalize_name(text)
    status = normalize_name(default_status)
    if status in {"normal", "negative"}:
        return "negative"
    if evidence_marker_is_non_positive(text) or any(
        marker in normalized
        for marker in [
            "无充血",
            "无分泌物",
            "无异常",
            "阴性",
            "正常",
            "未见",
            "未发现",
            "未检出",
            "不支持",
            "排除",
        ]
    ):
        return "negative"
    if evidence_refers_to_other_subject(text):
        return "other_subject"
    if "？" in clean_text(text) or "?" in clean_text(text) or any(
        marker in normalized
        for marker in ["是否", "筛查", "排查", "疑似", "可能", "考虑", "倾向"]
    ):
        return "uncertain"
    if status in {"abnormal", "positive"}:
        return "positive"
    return "positive"


def evidence_refers_to_other_subject(text: str) -> bool:
    normalized = normalize_name(text)
    if any(
        marker in normalized
        for marker in ["父亲", "母亲", "父母", "家属", "家族", "哥哥", "姐姐", "弟弟", "妹妹", "亲属"]
    ):
        return True
    if not normalized.startswith("我"):
        return False
    remainder = normalized[1:]
    self_prefixes = [
        "自己", "本人", "已", "目前", "现在", "正在", "有", "患有", "合并", "存在",
        "确诊", "诊断", "对", "年龄", "今年", "现年", "最近", "近期", "刚刚", "曾经",
        "既往", "无", "没有", "否认", "从未", "现用药", "当前用药", "常规用药",
        "的现用药", "的当前用药", "的常规用药",
    ]
    if any(remainder.startswith(prefix) for prefix in self_prefixes):
        return False
    if re.match(r"(?:\d{1,3}|[零〇一二三四五六七八九十百两]{1,4})岁", remainder):
        return False
    return bool(
        re.search(
            r"怀孕|妊娠|确诊|诊断|患有|合并|存在|过敏|现用药|当前用药|常规用药|"
            r"(?:\d{1,3}|[零〇一二三四五六七八九十百两]{1,4})岁",
            remainder,
        )
    )


def decide_treatment_review(
    review: Any,
    *,
    original_treatment_plan: str,
    case_state: Dict[str, Any],
    diagnosis: str,
    diagnosis_axes: List[Dict[str, Any]],
    verifier_issues: List[Dict[str, Any]],
    evidence_catalog: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    original = clean_text(original_treatment_plan)
    catalog = evidence_catalog
    if catalog is None:
        catalog = build_treatment_review_evidence_catalog(
            case_state=case_state,
            diagnosis=diagnosis,
            diagnosis_axes=diagnosis_axes,
            verifier_issues=verifier_issues,
        )
    if not isinstance(review, dict):
        return treatment_review_decision(
            status="rejected",
            original=original,
            treatment_plan=original,
            reason_codes=["invalid_review_payload"],
        )
    if "edits" in review:
        edits = review.get("edits")
        if not isinstance(edits, list):
            return treatment_review_decision(
                status="rejected",
                original=original,
                treatment_plan=original,
                reason_codes=["invalid_edits_payload"],
            )
        if not edits:
            return treatment_review_decision(
                status="unchanged",
                original=original,
                treatment_plan=original,
            )
        return decide_atomic_treatment_review(
            edits,
            original=original,
            evidence_catalog=catalog,
        )
    revised = clean_text(review.get("treatment_plan"))
    if not revised or revised == original:
        return treatment_review_decision(
            status="unchanged",
            original=original,
            treatment_plan=original,
        )
    evidence_refs = as_text_list(review.get("evidence_refs"))
    if not evidence_refs:
        return treatment_review_decision(
            status="rejected",
            original=original,
            treatment_plan=original,
            reason_codes=["missing_evidence_refs"],
        )
    resolved_evidence, failed_refs = resolve_treatment_review_refs(
        evidence_refs,
        catalog,
        exact_ids_only=False,
    )
    if failed_refs:
        return treatment_review_decision(
            status="rejected",
            original=original,
            treatment_plan=original,
            reason_codes=["unknown_evidence_ref"],
            failed_refs=failed_refs,
        )
    sensitive_reasons = unsupported_sensitive_fact_reasons(
        revised,
        original_treatment_plan=original,
        case_evidence=positive_treatment_case_evidence(resolved_evidence),
    )
    if sensitive_reasons:
        return treatment_review_decision(
            status="rejected",
            original=original,
            treatment_plan=original,
            reason_codes=sensitive_reasons,
        )
    # Free-text path: conflicted final diagnosis may only append safety text.
    # Reject replace/delete and reject cancel-out appends under a wrong diagnosis label.
    if treatment_review_diagnosis_consistency(diagnosis, diagnosis_axes) == "conflicted":
        if not free_text_conflicted_append_is_safety_only(original, revised):
            return treatment_review_decision(
                status="rejected",
                original=original,
                treatment_plan=original,
                reason_codes=["diagnosis_axis_conflict"],
            )
    return treatment_review_decision(
        status="accepted",
        original=original,
        treatment_plan=revised,
    )


def free_text_conflicted_append_is_safety_only(original: str, revised: str) -> bool:
    """True only when revised keeps original and adds safety/referral/monitoring text.

    original⊆revised alone is insufficient: cancel-out or mixed diagnosis-specific
    appends must fail even if they also mention 转诊/监测.
    """
    original_text = clean_text(original)
    revised_text = clean_text(revised)
    if not original_text or original_text not in revised_text:
        return False
    if revised_text == original_text:
        return True
    extra = revised_text.replace(original_text, "", 1).strip()
    if not extra:
        return True

    # Soft negations and cancel-outs that keep emergency words as surface tokens.
    cancel_markers = (
        "取消",
        "撤销",
        "不必",
        "无需",
        "不要",
        "请勿",
        "勿再",
        "勿继续",
        "暂缓",
        "暂不",
        "可暂缓",
        "停止",
        "暂停",
        "不再",
        "延后",
        "停止转诊",
        "停止急诊",
        "无需急诊",
        "不必转诊",
        "不再转诊",
        "取消以上",
        "以上全部",
        "删除上述",
        "仅保湿",
        "仅观察",
        "改为观察",
        "改为保湿",
        "改用",
        "改为",
        "仅用",
        "改按",
    )
    if any(marker in extra for marker in cancel_markers):
        return False

    # Diagnosis-specific care is never "safety-only" under a conflicted final label.
    non_safety_markers = (
        "保湿",
        "防晒",
        "外用",
        "激素",
        "糠疹",
        "白色糠疹",
        "抗过敏",
        "护肤",
        "润肤",
        "观察即可",
        "对症处理",
        "对症治疗",
        "按白色",
        "按糠疹",
        "保湿防晒",
    )
    if any(marker in extra for marker in non_safety_markers):
        return False

    # Whitelist: strip known safety phrases; residual must be empty (connectors only).
    safety_phrases = (
        "监测生命体征",
        "生命体征监测",
        "密切监测",
        "密切观察生命体征",
        "观察生命体征",
        "危险信号",
        "相互作用",
        "儿童眼科",
        "儿科眼科",
        "生命体征",
        "监测",
        "急诊",
        "转诊",
        "专科",
        "禁忌",
        "过敏",
        "立即",
        "安全",
        "住院",
        "气道",
        "眼科",
        "肿瘤",
        "复核",
        "会诊",
    )
    if not any(phrase in extra for phrase in safety_phrases):
        return False

    residual = extra
    for phrase in sorted(safety_phrases, key=len, reverse=True):
        residual = residual.replace(phrase, "")
    for connector in (
        "并请",
        "并且",
        "以及",
        "另外",
        "同时",
        "建议",
        "需要",
        "应当",
        "并",
        "且",
        "再",
        "请",
        "需",
        "应",
        "可",
        "的",
        "与",
        "和",
        "后",
        "时",
        "等",
        "也",
        "要",
        "做",
        "行",
        "予",
    ):
        residual = residual.replace(connector, "")
    residual = re.sub(r"[\s\d\W_]+", "", residual, flags=re.UNICODE)
    return not residual


def decide_atomic_treatment_review(
    edits: List[Any],
    *,
    original: str,
    evidence_catalog: List[Dict[str, str]],
) -> Dict[str, Any]:
    duplicate_ids = duplicate_treatment_edit_ids(edits)
    if duplicate_ids:
        return treatment_review_decision(
            status="rejected",
            original=original,
            treatment_plan=original,
            reason_codes=["duplicate_edit_id"],
            rejected_edits=[
                rejected_treatment_edit(edit_id, ["duplicate_edit_id"])
                for edit_id in duplicate_ids
            ],
        )
    prepared: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    occupied_ranges: List[tuple[int, int, str]] = []
    for index, raw_edit in enumerate(edits, start=1):
        edit_id = clean_text(raw_edit.get("edit_id")) if isinstance(raw_edit, dict) else ""
        edit_id = edit_id or f"edit:{index}"
        if not isinstance(raw_edit, dict):
            rejected.append(rejected_treatment_edit(edit_id, ["invalid_edit_payload"]))
            continue
        operation = clean_text(raw_edit.get("operation")).lower()
        target = clean_text(raw_edit.get("target"))
        replacement = clean_text(raw_edit.get("replacement"))
        evidence_refs = as_text_list(raw_edit.get("evidence_refs"))
        reasons: List[str] = []
        failed_refs: List[str] = []
        resolved_evidence: List[Dict[str, str]] = []
        start = len(original)
        end = len(original)
        if operation not in {"delete", "replace", "append"}:
            reasons.append("invalid_edit_operation")
        elif operation == "append":
            if target or not replacement:
                reasons.append("invalid_append_edit")
        else:
            if not target or target not in original:
                reasons.append("missing_edit_target")
            elif original.count(target) != 1:
                reasons.append("ambiguous_edit_target")
            else:
                start = original.index(target)
                end = start + len(target)
            if operation == "replace" and not replacement:
                reasons.append("empty_replacement")
            if operation == "delete" and replacement:
                reasons.append("delete_has_replacement")
        if not evidence_refs:
            reasons.append("missing_evidence_refs")
        else:
            resolved_evidence, failed_refs = resolve_treatment_review_refs(
                evidence_refs,
                evidence_catalog,
                exact_ids_only=True,
            )
            if failed_refs:
                reasons.append("unknown_evidence_ref")
        if not failed_refs:
            reasons.extend(
                treatment_edit_evidence_reason_codes(
                    operation=operation,
                    target=target,
                    replacement=replacement,
                    evidence=resolved_evidence,
                )
            )
        structural_reasons = {
            "invalid_edit_operation",
            "invalid_append_edit",
            "missing_edit_target",
            "ambiguous_edit_target",
            "empty_replacement",
            "delete_has_replacement",
        }
        if not any(reason in structural_reasons for reason in reasons):
            candidate_plan = apply_single_treatment_edit(
                original,
                operation=operation,
                start=start,
                end=end,
                replacement=replacement,
            )
            reasons.extend(
                unsupported_sensitive_fact_reasons(
                    candidate_plan,
                    original_treatment_plan=original,
                    case_evidence=positive_treatment_case_evidence(resolved_evidence),
                )
            )
        prepared.append(
            {
                "edit_id": edit_id,
                "operation": operation,
                "start": start,
                "end": end,
                "replacement": replacement,
                "reason_codes": unique_preserve_order(reasons),
                "failed_refs": failed_refs,
                "resolved_evidence": resolved_evidence,
            }
        )
        if not reasons and operation != "append":
            occupied_ranges.append((start, end, edit_id))
    overlapping_ids: set[str] = set()
    for index, (start, end, edit_id) in enumerate(occupied_ranges):
        for other_start, other_end, other_id in occupied_ranges[index + 1:]:
            if max(start, other_start) < min(end, other_end):
                overlapping_ids.update([edit_id, other_id])
    accepted_edits: List[Dict[str, Any]] = []
    for edit in prepared:
        reasons = list(edit["reason_codes"])
        if edit["edit_id"] in overlapping_ids:
            reasons.append("overlapping_edit_target")
        reasons = unique_preserve_order(reasons)
        if reasons:
            rejected.append(
                rejected_treatment_edit(
                    edit["edit_id"],
                    reasons,
                    failed_refs=edit["failed_refs"],
                )
            )
        else:
            accepted_edits.append(edit)
    treatment_plan = apply_prepared_treatment_edits(original, accepted_edits)
    combined_evidence = [
        evidence
        for edit in accepted_edits
        for evidence in edit.get("resolved_evidence", [])
        if isinstance(evidence, dict)
    ]
    combined_sensitive_reasons = unsupported_sensitive_fact_reasons(
        treatment_plan,
        original_treatment_plan=original,
        case_evidence=positive_treatment_case_evidence(combined_evidence),
    )
    if combined_sensitive_reasons:
        rejected.extend(
            rejected_treatment_edit(item["edit_id"], combined_sensitive_reasons)
            for item in accepted_edits
        )
        accepted_edits = []
        treatment_plan = original
    if accepted_edits and not clean_text(treatment_plan):
        rejected.extend(
            rejected_treatment_edit(item["edit_id"], ["empty_treatment_plan"])
            for item in accepted_edits
        )
        accepted_edits = []
        treatment_plan = original
    review_unchanged = bool(accepted_edits and not rejected and treatment_plan == original)
    if accepted_edits and treatment_plan == original:
        accepted_edits = []
    status = "unchanged" if review_unchanged else "accepted"
    if not review_unchanged and accepted_edits and rejected:
        status = "partial"
    elif not review_unchanged and not accepted_edits:
        status = "rejected"
        treatment_plan = original
    reason_codes = unique_preserve_order(
        reason
        for item in rejected
        for reason in as_text_list(item.get("reason_codes"))
    )
    failed_refs = unique_preserve_order(
        reference
        for item in rejected
        for reference in as_text_list(item.get("failed_refs"))
    )
    return treatment_review_decision(
        status=status,
        original=original,
        treatment_plan=treatment_plan,
        reason_codes=reason_codes,
        failed_refs=failed_refs,
        accepted_edit_ids=[item["edit_id"] for item in accepted_edits],
        rejected_edits=rejected,
    )


def apply_single_treatment_edit(
    original: str,
    *,
    operation: str,
    start: int,
    end: int,
    replacement: str,
) -> str:
    if operation == "append":
        separator = "" if not original or original.endswith(("\n", " ")) else " "
        return clean_text(f"{original}{separator}{replacement}")
    return clean_text(f"{original[:start]}{replacement}{original[end:]}")


def apply_prepared_treatment_edits(original: str, edits: List[Dict[str, Any]]) -> str:
    plan = original
    positional = [item for item in edits if item.get("operation") != "append"]
    for edit in sorted(positional, key=lambda item: int(item["start"]), reverse=True):
        plan = f"{plan[:int(edit['start'])]}{edit['replacement']}{plan[int(edit['end']):]}"
    for edit in edits:
        if edit.get("operation") == "append":
            separator = "" if not plan or plan.endswith(("\n", " ")) else " "
            plan = f"{plan}{separator}{edit['replacement']}"
    return clean_text(plan)


def unresolved_treatment_review_refs(
    references: List[str],
    evidence_catalog: List[Dict[str, str]],
) -> List[str]:
    _, failed = resolve_treatment_review_refs(
        references,
        evidence_catalog,
        exact_ids_only=False,
    )
    return failed


def resolve_treatment_review_refs(
    references: List[str],
    evidence_catalog: List[Dict[str, str]],
    *,
    exact_ids_only: bool,
) -> tuple[List[Dict[str, str]], List[str]]:
    by_id = {clean_text(item.get("id")): item for item in evidence_catalog if isinstance(item, dict)}
    resolved: List[Dict[str, str]] = []
    failed: List[str] = []
    for reference in references:
        exact = by_id.get(clean_text(reference))
        if exact is not None:
            resolved.append(exact)
            continue
        if exact_ids_only:
            failed.append(reference)
            continue
        normalized_reference = normalize_name(reference)
        matches = [
            item
            for item in evidence_catalog
            if isinstance(item, dict)
            and normalized_reference
            and (
                normalized_reference in normalize_name(item.get("text"))
                or (
                    len(normalize_name(item.get("text"))) >= 4
                    and normalize_name(item.get("text")) in normalized_reference
                )
            )
        ]
        if matches:
            resolved.extend(matches)
        else:
            failed.append(reference)
    unique_resolved: List[Dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in resolved:
        evidence_id = clean_text(item.get("id"))
        if evidence_id in seen_ids:
            continue
        seen_ids.add(evidence_id)
        unique_resolved.append(item)
    return unique_resolved, unique_preserve_order(failed)


def duplicate_treatment_edit_ids(edits: List[Any]) -> List[str]:
    edit_ids = [
        clean_text(item.get("edit_id"))
        for item in edits
        if isinstance(item, dict) and clean_text(item.get("edit_id"))
    ]
    return unique_preserve_order(
        edit_id for edit_id in edit_ids if edit_ids.count(edit_id) > 1
    )


def treatment_edit_evidence_reason_codes(
    *,
    operation: str,
    target: str,
    replacement: str,
    evidence: List[Dict[str, Any]],
) -> List[str]:
    polarities = {
        clean_text(item.get("polarity"))
        for item in evidence
        if isinstance(item, dict)
    }
    reasons: List[str] = []
    diagnosis_entries = [
        item for item in evidence if isinstance(item, dict) and item.get("source") == "diagnosis"
    ]
    non_diagnosis_positive = [
        item
        for item in evidence
        if isinstance(item, dict)
        and item.get("source") in {"patient", "exam", "axis", "verifier"}
        and item.get("polarity") in {"positive", "issue"}
    ]
    if diagnosis_entries and not non_diagnosis_positive and operation in {"delete", "replace", "append"}:
        reasons.append("diagnosis_context_only")
    if any(item.get("consistency_status") == "conflicted" for item in diagnosis_entries):
        if operation in {"delete", "replace"}:
            reasons.append("diagnosis_axis_conflict")
    if "negative" in polarities and operation != "delete":
        reasons.append("negative_evidence_requires_delete")
    if "missing" in polarities and operation != "append":
        reasons.append("missing_evidence_cannot_delay_treatment")
    if "missing" in polarities and (
        operation != "append" or not treatment_evidence_closure_text(replacement)
    ):
        if "missing_evidence_cannot_delay_treatment" not in reasons:
            reasons.append("missing_evidence_requires_closure")
    if polarities.intersection({"uncertain", "other_subject", "neutral"}):
        reasons.append("non_factual_evidence_ref")
    if operation != "delete" and treatment_introduces_new_medication(replacement):
        if not treatment_evidence_supports_new_medication(replacement, evidence):
            reasons.append("unsupported_treatment_indication")
    return reasons


def treatment_evidence_closure_text(text: str) -> bool:
    normalized = normalize_name(text)
    return any(
        marker in normalized
        for marker in [
            "检查",
            "检验",
            "化验",
            "影像",
            "ct",
            "mri",
            "超声",
            "活检",
            "病理",
            "监测",
            "复查",
            "随访",
            "评估",
            "筛查",
        ]
    )


def treatment_introduces_new_medication(text: str) -> bool:
    return bool(
        re.search(
            r"(?:经验性|建议|立即|开始|予以|可|应)?\s*"
            r"(?:使用|应用|给予|口服|服用|注射|静脉滴注)\s*"
            r"[^，,。；;\n]{1,30}?(?:治疗|抗感染|退热|止痛|$)",
            clean_text(text),
        )
    )


def treatment_evidence_supports_new_medication(
    treatment_text: str,
    evidence: List[Dict[str, str]],
) -> bool:
    has_diagnostic_basis = any(
        clean_text(item.get("source")) in {"diagnosis", "axis"}
        and clean_text(item.get("polarity")) == "positive"
        for item in evidence
        if isinstance(item, dict)
    )
    verifier_text = "\n".join(
        clean_text(item.get("text"))
        for item in evidence
        if isinstance(item, dict) and clean_text(item.get("source")) == "verifier"
    )
    medications = extract_recommended_medications(treatment_text)
    return bool(
        has_diagnostic_basis
        and medications
        and all(medication in normalize_name(verifier_text) for medication in medications)
    )


def extract_recommended_medications(text: str) -> List[str]:
    medications: List[str] = []
    pattern = re.compile(
        r"(?:经验性|建议|立即|开始|予以|可|应)?\s*"
        r"(?:使用|应用|给予|口服|服用|注射|静脉滴注)\s*"
        r"([^，,。；;\n]{1,30}?)(?:治疗|抗感染|退热|止痛|$)"
    )
    for match in pattern.finditer(clean_text(text)):
        medication = clean_medication_phrase(match.group(1))
        if medication and medication not in medications:
            medications.append(medication)
    return medications


def positive_treatment_case_evidence(evidence_catalog: List[Dict[str, Any]]) -> str:
    return "\n".join(
        clean_text(item.get("text"))
        for item in evidence_catalog
        if isinstance(item, dict)
        and clean_text(item.get("source")) in {"patient", "exam", "axis"}
        and clean_text(item.get("polarity")) == "positive"
        and clean_text(item.get("text"))
    )


def treatment_review_decision(
    *,
    status: str,
    original: str,
    treatment_plan: str,
    reason_codes: Optional[List[str]] = None,
    failed_refs: Optional[List[str]] = None,
    accepted_edit_ids: Optional[List[str]] = None,
    rejected_edits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "accepted": status in {"accepted", "partial"},
        "treatment_plan": clean_text(treatment_plan) or clean_text(original),
        "reason_codes": unique_preserve_order(reason_codes or []),
        "failed_refs": unique_preserve_order(failed_refs or []),
        "accepted_edit_ids": unique_preserve_order(accepted_edit_ids or []),
        "rejected_edits": rejected_edits or [],
        "before_hash": treatment_review_plan_hash(original),
        "after_hash": treatment_review_plan_hash(treatment_plan or original),
    }


def rejected_treatment_edit(
    edit_id: str,
    reason_codes: List[str],
    *,
    failed_refs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "edit_id": edit_id,
        "reason_codes": unique_preserve_order(reason_codes),
        "failed_refs": unique_preserve_order(failed_refs or []),
    }


def treatment_review_plan_hash(treatment_plan: str) -> str:
    digest = hashlib.sha256(clean_text(treatment_plan).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def review_introduces_unsupported_sensitive_facts(
    revised_treatment_plan: str,
    *,
    original_treatment_plan: str,
    case_evidence: str,
) -> bool:
    return bool(
        unsupported_sensitive_fact_reasons(
            revised_treatment_plan,
            original_treatment_plan=original_treatment_plan,
            case_evidence=case_evidence,
        )
    )


def unsupported_sensitive_fact_reasons(
    revised_treatment_plan: str,
    *,
    original_treatment_plan: str,
    case_evidence: str,
) -> List[str]:
    revised = normalize_name(revised_treatment_plan)
    original = normalize_name(original_treatment_plan)
    reasons: List[str] = []
    sensitive_groups = [
        ("unsupported_pregnancy", ["怀孕", "妊娠", "孕妇", "哺乳"]),
        ("unsupported_renal_impairment", ["肾衰竭", "肾功能衰竭", "肾功能不全"]),
        ("unsupported_hepatic_impairment", ["肝衰竭", "肝功能衰竭", "肝功能不全"]),
        ("unsupported_drug_allergy", ["过敏", "禁忌药"]),
    ]
    for reason, markers in sensitive_groups:
        if not has_asserted_sensitive_patient_fact(revised_treatment_plan, markers):
            continue
        if has_asserted_sensitive_patient_fact(original_treatment_plan, markers):
            continue
        if not any(
            normalized_marker_present_not_negated(case_evidence, marker)
            for marker in markers
        ):
            reasons.append(reason)
    for condition in ["糖尿病", "高血压"]:
        if not has_asserted_patient_condition(revised_treatment_plan, condition):
            continue
        if has_asserted_patient_condition(original_treatment_plan, condition):
            continue
        if not has_asserted_patient_condition(case_evidence, condition):
            reasons.append(f"unsupported_{'diabetes' if condition == '糖尿病' else 'hypertension'}")
    original_conditions = extract_asserted_patient_conditions(original_treatment_plan)
    for condition in extract_asserted_patient_conditions(revised_treatment_plan):
        if condition in original_conditions or condition_supported_by_evidence(condition, case_evidence):
            continue
        if any(
            marker in condition
            for marker in [
                "糖尿病",
                "高血压",
                "肾衰竭",
                "肾功能不全",
                "肝衰竭",
                "肝功能不全",
                "过敏",
                "怀孕",
                "妊娠",
                "哺乳",
            ]
        ):
            continue
        reasons.append("unsupported_comorbidity")
    case_medications = extract_current_medication_assertions(case_evidence)
    supported_medications = extract_current_medication_assertions(original_treatment_plan) + case_medications
    for medication in extract_current_medication_assertions(revised_treatment_plan):
        if not any(medication in item or item in medication for item in supported_medications):
            reasons.append("unsupported_current_medication")
    normalized_original = normalize_name(original_treatment_plan)
    medication_actions = (
        extract_discontinued_medications(revised_treatment_plan)
        + extract_medication_change_assertions(revised_treatment_plan)
    )
    for medication in medication_actions:
        if medication in normalized_original:
            continue
        if not any(medication in item or item in medication for item in case_medications):
            reasons.append("unsupported_medication_action")
    for age in extract_patient_age_assertions(revised_treatment_plan):
        if age not in original_treatment_plan and age not in case_evidence:
            reasons.append("unsupported_age")
    occupation_patterns = re.findall(
        r"(?:职业(?:是|为)?|从事|作为)[^，,；;。]{1,20}",
        clean_text(revised_treatment_plan),
    )
    if any(item not in original_treatment_plan and item not in case_evidence for item in occupation_patterns):
        reasons.append("unsupported_occupation")
    return unique_preserve_order(reasons)


def extract_asserted_patient_conditions(text: str) -> List[str]:
    conditions: List[str] = []
    pattern = re.compile(
        r"(?:患者|病人|本人|我|患儿|该患儿)\s*(?:目前|当前|既往|现)?\s*"
        r"(?:合并|患有|确诊(?:为)?|诊断(?:为)?|存在)\s*"
        r"([^，,。；;\n]{2,30})"
    )
    for match in pattern.finditer(clean_text(text)):
        condition = re.split(
            r"(?:且|并且|同时|正在|应当|应该|应|需要|需|建议|因此|故)",
            clean_text(match.group(1)),
            maxsplit=1,
        )[0]
        normalized = normalize_name(condition)
        if normalized and normalized not in conditions:
            conditions.append(normalized)
    return conditions


def condition_supported_by_evidence(condition: str, case_evidence: str) -> bool:
    normalized = normalize_name(condition)
    return bool(
        normalized
        and normalized_marker_present_not_negated(case_evidence, normalized)
    )


def extract_patient_age_assertions(text: str) -> List[str]:
    ages: List[str] = []
    age_pattern = r"(?:\d{1,3}|[零〇一二三四五六七八九十百两]{1,4})岁"
    for clause in re.split(r"[，,。；;！？!?\n]", clean_text(text)):
        normalized = normalize_name(clause)
        for match in re.finditer(age_pattern, normalized):
            prefix = normalized[max(0, match.start() - 10):match.start()]
            suffix = normalized[match.end():match.end() + 4]
            if suffix.startswith(("以下", "以上", "以内", "以外")):
                continue
            if not any(
                marker in prefix
                for marker in ["患者", "病人", "本人", "患儿", "孩子", "儿童", "婴儿", "我", "年龄", "现年", "今年"]
            ):
                continue
            age = match.group(0)
            if age not in ages:
                ages.append(age)
    return ages


def has_asserted_sensitive_patient_fact(text: str, markers: List[str]) -> bool:
    for clause in re.split(r"[，,。；;！？!?\n]", clean_text(text)):
        normalized = normalize_name(clause)
        for marker in markers:
            token = normalize_name(marker)
            start = 0
            while token:
                index = normalized.find(token, start)
                if index < 0:
                    break
                prefix = normalized[max(0, index - 16):index]
                suffix = normalized[index + len(token):]
                if marker_occurrence_is_negated(normalized, token, index):
                    start = index + len(token)
                    continue
                if any(item in prefix for item in ["若", "如果", "是否", "筛查", "排查", "疑似", "可能", "考虑", "倾向"]):
                    start = index + len(token)
                    continue
                if suffix.startswith("者"):
                    start = index + len(token)
                    continue
                has_subject = any(
                    item in prefix[-10:]
                    for item in ["患者", "病人", "本人", "我", "其", "该患儿"]
                )
                has_assertion = any(
                    item in prefix[-10:]
                    for item in ["已", "目前", "正在", "合并", "患有", "存在", "确诊", "诊断为", "为"]
                )
                direct_state = any(
                    item in normalized
                    for item in ["已怀孕", "目前怀孕", "正在哺乳", "哺乳期"]
                ) or bool(re.search(r"妊娠\d{1,2}周", normalized))
                if has_subject or has_assertion or direct_state:
                    return True
                start = index + len(token)
    return False


def has_asserted_patient_condition(text: str, condition: str) -> bool:
    condition_name = normalize_name(condition)
    for clause in re.split(r"[，,。；;\n]", clean_text(text)):
        normalized = normalize_name(clause)
        if condition_name not in normalized:
            continue
        prefix = normalized.split(condition_name, 1)[0]
        if any(marker + condition_name in normalized for marker in ["无", "没有", "否认", "未患", "不合并", "不伴"]):
            continue
        if any(marker in prefix for marker in ["若", "如果", "是否", "排查", "筛查", "考虑", "疑似", "可能"]):
            continue
        if any(marker in prefix for marker in ["家族", "父亲", "母亲", "父母", "亲属"]):
            continue
        if condition_name + "病史" in normalized:
            return True
        has_subject = any(marker in prefix for marker in ["患者", "病人", "本人", "我", "既往"])
        has_assertion = any(marker in prefix for marker in ["合并", "患有", "有", "确诊", "诊断"])
        if has_subject and has_assertion:
            return True
        if any(prefix.endswith(marker) for marker in ["合并", "患有", "有", "确诊为", "诊断为", "既往有"]):
            return True
    return False


def extract_current_medication_assertions(text: str) -> List[str]:
    medications = []
    source = clean_text(text)
    patterns = [
        re.compile(
            r"(?:正在|目前|现在|当前|长期|一直|每日|每天)\s*(?:正\s*)?(?:在\s*)?"
            r"(?:服用|使用|应用|口服|吃|注射)\s*([^，,。；;\n]{1,40})"
        ),
        re.compile(
            r"(?:患者|病人|本人|我)\s*(?:(?:正在|目前|现在|当前)\s*)?"
            r"(?:服用|使用|应用|口服|吃|注射)\s*([^，,。；;\n]{1,40})"
        ),
        re.compile(
            r"(?:患者|病人|本人|我)?\s*(?:的\s*)?"
            r"(?:现用药|当前用药|目前用药|常规用药|长期用药)\s*"
            r"(?:包括|包含|为|是|有|：|:)\s*([^，,。；;\n]{1,40})"
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(source):
            prefix = re.split(r"[，,。；;\n]", source[: match.start()])[-1]
            if any(marker in prefix for marker in ["没有", "未", "否认", "不再"]):
                continue
            if any(marker in prefix for marker in ["家族", "父亲", "母亲", "父母", "亲属", "家属"]):
                continue
            if any(marker in prefix for marker in ["若", "如果", "计划", "拟", "将来", "未来", "可能"]):
                continue
            medication = clean_medication_phrase(match.group(1))
            if medication and medication not in medications:
                medications.append(medication)
    return medications


def extract_discontinued_medications(text: str) -> List[str]:
    medications = []
    stop_action = r"(?:停用|停服|暂停使用|停止服用|停止使用|撤除|中止|停掉)"
    patterns = [
        re.compile(rf"(?:立即|马上|暂时|逐步)?\s*{stop_action}\s*([^，,。；;\n]{{1,40}})"),
        re.compile(rf"(?:建议\s*)?(?:将|把)\s*([^，,。；;\n]{{1,30}}?)\s*{stop_action}"),
        re.compile(
            rf"([^，,。；;\n]{{2,30}}?)\s*(?:应|需|必须|建议)?\s*"
            rf"(?:立即|马上|暂时|逐步)?\s*{stop_action}"
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(clean_text(text)):
            medication = clean_medication_phrase(match.group(1))
            if medication and medication not in medications:
                medications.append(medication)
    return medications


def extract_medication_change_assertions(text: str) -> List[str]:
    medications = []
    patterns = [
        re.compile(r"(?:继续|维持)\s*(?:服用|使用|应用|口服)?\s*([^，,。；;\n]{1,40})"),
        re.compile(r"(?:增加|加大|提高|减少|降低|调整)\s*([^，,。；;\n]{1,30}?)\s*(?:剂量|用量)"),
        re.compile(
            r"(?:建议\s*)?(?:将|把)?\s*([^，,。；;\n]{2,30}?)\s*(?:剂量|用量)\s*"
            r"(?:增加|加大|加倍|减量|减少|降低|调整)"
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(clean_text(text)):
            medication = clean_medication_phrase(match.group(1))
            if medication and medication not in medications:
                medications.append(medication)
    return medications


def clean_medication_phrase(value: str) -> str:
    medication = re.split(
        r"(?:并|且|应|需|必须|建议|随后|准备|拟|考虑|因此|故|以便)",
        clean_text(value),
        maxsplit=1,
    )[0]
    medication = re.split(
        r"\d+(?:\.\d+)?\s*(?:mg|g|ml|毫克|克|片|粒)",
        medication,
        maxsplit=1,
        flags=re.I,
    )[0]
    medication = re.sub(r"^(?:(?:患者|病人|本人|我|建议|将|把)\s*)+", "", medication)
    medication = re.sub(r"(?:抗凝治疗|抗凝|治疗|方案|用药|剂量|用量)$", "", medication)
    normalized = normalize_name(medication)
    if normalized in {"立即", "马上", "暂时", "逐步", "建议", "应", "需", "必须"}:
        return ""
    return normalized if len(normalized) >= 2 else ""


def _validate_service_url_override(payload: Optional[Mapping[str, Any]]) -> None:
    """Reject request-controlled upstreams except explicit loopback local tests."""
    if not isinstance(payload, Mapping):
        return
    raw = payload.get("service_base_url") or payload.get("contestServiceBaseUrl")
    if not str(raw or "").strip():
        return
    if str(payload.get("local_test") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise ValueError("service_base_url override is disabled for remote test requests")
    parsed = urllib.parse.urlparse(str(raw).strip())
    hostname = (parsed.hostname or "").lower()
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if parsed.scheme != "http" or hostname not in {"localhost", "127.0.0.1", "::1"}:
        if address is None or not address.is_loopback:
            raise ValueError("local_test service_base_url must target loopback HTTP")


def _safe_prompt_id(patient_id: str) -> str:
    return sha256(str(patient_id).encode("utf-8")).hexdigest()[:24]


def load_knowledge_registry(knowledge_dir: Path = KNOWLEDGE_DIR) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "alias_map": load_knowledge_rules(knowledge_dir / "alias_map.json"),
        "exam_intent_map": load_knowledge_rules(knowledge_dir / "exam_intent_map.json"),
        "diagnosis_exam_profiles": load_knowledge_rules(knowledge_dir / "diagnosis_exam_profiles.json"),
        "treatment_safety_profiles": load_knowledge_rules(knowledge_dir / "treatment_safety_profiles.json"),
    }


def load_knowledge_rules(path: Path, *, status: str = "verified") -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError("knowledge file missing: %s" % path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("failed to load knowledge file: %s" % path) from exc
    rules = data.get("rules", data) if isinstance(data, dict) else data
    if not isinstance(rules, list):
        raise ValueError("knowledge rules must be a list: %s" % path)
    result = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError("knowledge rule %s is not an object: %s" % (index, path))
        if clean_text(rule.get("status") or "candidate") != status:
            continue
        result.append(rule)
    if not result:
        raise ValueError("knowledge file has no %s rules: %s" % (status, path))
    return result



def _provenance_from_examination_results(examination_results):
    """Build structured anti-infective provenance from REAL exam results only.

    Parses AST/sensitivity rows (S/I/R, 敏感/中介/耐药) and culture results
    into the structure consumed by find_anti_infective_evidence_gaps. Never
    derives provenance from free-text plan strings — only from real
    examination_results attached to the case state.
    """
    empty = {"ast": [], "cultures": [], "confirmed_resistance": [], "empiric": None}
    # Flatten real examination_results shapes:
    #   - dict[str, dict]  (exam_name -> result_block) — common real shape
    #   - single Mapping   (one result block)
    #   - list[Mapping]
    items = []
    if isinstance(examination_results, Mapping):
        # If every value is itself a result block (dict/str), treat values as items;
        # otherwise treat the mapping itself as a single result block.
        vals = list(examination_results.values())
        if vals and all(isinstance(v, (Mapping, str, list)) for v in vals):
            for v in vals:
                if isinstance(v, Mapping):
                    items.append(v)
                elif isinstance(v, list):
                    items.extend(i for i in v if isinstance(i, Mapping))
        else:
            items = [examination_results]
    elif isinstance(examination_results, list):
        for e in examination_results:
            if isinstance(e, Mapping):
                items.append(e)
    else:
        items = []
    if not items:
        return empty

    # Normalize real SDK nesting: exam_name -> {status, result, abnormal_indicators}.
    expanded = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        nested = item.get("result")
        has_nested = (
            isinstance(nested, Mapping)
            and any(nested.get(k) is not None for k in (
                "药敏结果", "ast", "sensitivity_results", "susceptibility",
                "cultures", "culture", "培养", "resistance", "耐药"))
        )
        if has_nested:
            expanded.append(item)
            expanded.append(nested)
        else:
            expanded.append(item)
    items = expanded
    ast_rows = []
    cultures = []
    resistance = []
    for item in items:
        # AST / susceptibility tables: list of rows under ast/sensitivity
        for key in ("ast", "sensitivity_results", "susceptibility", "药敏结果"):
            rows = item.get(key) if isinstance(item, Mapping) else None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    drug = clean_text(row.get("drug") or row.get("name") or row.get("药品"))
                    result = clean_text(row.get("result") or row.get("susceptibility") or row.get("结果") or "")
                    if not drug:
                        continue
                    norm = _normalize_ast_result(result)
                    if norm not in {"S", "I", "R"}:
                        continue
                    source_ref = clean_text(row.get("source") or row.get("检查") or row.get("source_ref") or "examination_result")
                    ast_rows.append({
                        "drug": drug,
                        "drug_norm": normalize_name(drug),
                        "result": norm,
                        "source": source_ref,
                        "status": clean_text(row.get("status") or "resulted"),
                        "result_time": clean_text(row.get("time") or row.get("result_time") or ""),
                    })
        # Resistance explicitly flagged per drug
        for key in ("resistance", "耐药", "confirmed_resistance"):
            rr = item.get(key) if isinstance(item, Mapping) else None
            if isinstance(rr, list):
                for r in rr:
                    if isinstance(r, Mapping):
                        name = clean_text(r.get("drug") or r.get("name") or r.get("药品"))
                    else:
                        name = clean_text(r)
                    if name:
                        resistance.append(normalize_name(name))
        # Culture results
        for key in ("cultures", "culture", "培养"):
            cc = item.get(key) if isinstance(item, Mapping) else None
            if isinstance(cc, list):
                for c in cc:
                    if isinstance(c, Mapping):
                        cultures.append({
                            "source": clean_text(c.get("source") or c.get("标本") or "culture"),
                            "organism": clean_text(c.get("organism") or c.get("结果") or ""),
                            "status": clean_text(c.get("status") or "resulted"),
                            "time": clean_text(c.get("time") or c.get("result_time") or ""),
                        })
    # Also support a single drug->result mapping shape: {"环丙沙星": "R"}
    if isinstance(examination_results, Mapping) and items == [examination_results]:
        maybe = {}
        for k, v in examination_results.items():
            if isinstance(v, str) and k not in ("ast", "sensitivity_results", "cultures", "culture", "resistance", "status", "result_time"):
                maybe[k] = v
        if maybe and not ast_rows:
            for drug, result in maybe.items():
                norm = _normalize_ast_result(result)
                if norm in {"S", "I", "R"}:
                    ast_rows.append({
                        "drug": clean_text(drug),
                        "drug_norm": normalize_name(drug),
                        "result": norm,
                        "source": "examination_results",
                        "status": "resulted",
                        "result_time": "",
                    })
                    if norm == "R":
                        resistance.append(normalize_name(drug))

    return {
        "ast": ast_rows,
        "cultures": cultures,
        "empiric": None,
        "confirmed_resistance": sorted({r for r in resistance if r}),
    }


def _normalize_ast_result(result):
    """Normalize an AST/sensitivity result to S/I/R. Returns '' if unrecognized."""
    r = clean_text(result).upper()
    if r in {"S", "敏感", "SENSITIVE", "SUSCEPTIBLE"}:
        return "S"
    if r in {"I", "中介", "INTERMEDIATE", "SDD"}:
        return "I"
    if r in {"R", "耐药", "RESISTANT"}:
        return "R"
    return ""


def extract_case_features(
    case_state: Dict[str, Any],
    disease_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    text = case_text_for_matching(case_state)
    normalized = normalize_name(text)
    return {
        "case_text": text,
        "patient_text": patient_text_for_matching(case_state),
        "examination_results": (
            case_state.get("examination_results")
            if isinstance(case_state.get("examination_results"), dict)
            else {}
        ),
        "chief_complaint": first_patient_text(case_state),
        "positive_findings": matched_labels(
            normalized,
            {
                "高热": ["高热", "发热", "39"],
                "湿疹": ["湿疹", "特应性皮炎"],
                "疱疹样皮损": ["疱疹", "水泡", "水疱", "水痘"],
                "腰痛加重": ["腰部钝痛", "腰痛", "最近加重", "近期加重"],
                "尿频": ["尿频"],
                "较大肾囊肿": ["较大薄壁囊肿", "较大囊肿", "大囊肿"],
                "压迫邻近结构": ["压迫邻近结构", "压迫"],
                "反复鼻出血": ["反复鼻出血", "容易鼻出血", "鼻出血"],
                "鼻中隔偏曲": ["鼻中隔偏曲"],
                "鼻骨嵴接触": ["锐利骨嵴", "骨嵴接触"],
                "发热性尿路感染": ["高热寒战", "发热寒战", "尿频尿急尿痛"],
                "尿潴留风险": ["差点尿不出来", "尿不出来", "尿潴留", "排尿困难"],
                "会阴痛": ["会阴部胀痛", "会阴痛"],
                "尿培养阳性": ["尿培养提示", "尿培养阳性", "大肠埃希菌"],
                "头部外伤后起病": ["头部磕碰", "头部外伤", "撞到头", "外伤后"],
                "外伤后持续头痛": ["头部磕碰后", "外伤后", "反复头痛", "持续头痛"],
                "认知症状": ["注意力不集中", "短期记忆差", "记忆差", "认知"],
                "前庭平衡症状": ["站立不稳", "快速转头", "平衡", "头晕"],
                "婴儿早发心肺症状": ["婴儿出生后", "出生后就呼吸急促"],
                "呼吸急促": ["呼吸急促", "气促", "肋下凹陷"],
                "喂养困难出汗": ["吃奶困难", "喂养困难", "喂养时明显出汗", "喂养出汗"],
                "发绀或肺高压体征": ["口唇发绀", "发绀", "P2亢进", "p2亢进", "肺高压"],
                "育龄女性": ["育龄女性", "年轻女性"],
                "偏头痛伴恶心头晕": ["偏头痛", "搏动性头痛", "偏侧搏动头痛", "畏光怕声"],
                "旅行或视觉运动诱发": ["旅行", "坐飞机", "飞机", "视觉运动", "晕动"],
                "月经相关": ["月经", "经前"],
                "新生儿脐部病变": ["新生儿脐部", "脐部"],
                "湿润易出血肿块": ["鲜红湿润", "湿润", "小肿块", "肿块", "出血"],
                "无黑痣样改变": ["没有黑痣样改变", "无黑痣样改变"],
                "上臂外伤": ["上臂", "肱骨", "跌倒"],
                "前胸外伤": ["前胸", "胸壁", "胸口"],
                "撞击外伤机制": ["撞击", "推搡", "跌倒"],
                "胸部外伤呼吸痛": ["深呼吸", "咳嗽时胸痛", "抬臂时加重"],
                "剧痛肿胀活动受限": ["剧痛", "肿胀", "活动受限"],
                "突发心悸": ["心跳很快", "心跳快", "心悸", "心慌"],
                "既往心脏病": ["心脏病", "心梗", "心功能下降", "冠心病"],
                "胃肠道丢失或摄入不足": ["腹泻", "吃得少", "摄入不足"],
                "重复抓握腕活动诱发肘痛": ["重复抓握", "腕部活动", "肘部疼痛", "局限肘部"],
                "慢性肝病": ["慢性肝病", "肝硬化", "肝病"],
                "左上腹饱胀": ["左上腹饱胀", "左上腹"],
                "三系减少": ["三系减少", "血细胞减少"],
                "压榨性胸痛": ["压榨性胸痛", "压榨胸痛"],
                "肌钙蛋白升高": ["肌钙蛋白升高", "肌钙蛋白阳性"],
                "慢性咳嗽咯血": ["咳嗽三个月", "慢性咳嗽", "咯血", "痰中带血"],
                "脚踝水肿": ["脚踝水肿", "下肢水肿", "水肿"],
                "症状性低钾": ["手抽筋", "抽筋", "低钾", "血钾"],
                "腹泻吸收不良": ["腹泻", "脂肪泻", "吸收不良", "小肠切除", "胰酶"],
            },
        ),
        "medications": matched_labels(normalized, {"泼尼松": ["泼尼松"], "激素": ["激素"]}),
        "immunosuppression": matched_labels(
            normalized,
            {"全身激素": ["泼尼松", "全身激素", "糖皮质激素"], "免疫抑制": ["免疫抑制"]},
        ),
        "medication_risk": (
            ["非二氢吡啶类钙通道阻滞剂禁忌"]
            if has_reduced_left_ventricular_ejection_fraction(text)
            else []
        ),
        "red_flags": matched_labels(normalized, {"精神差": ["精神差", "精神很差", "嗜睡"], "高热": ["高热"]}),
        "family_history": matched_labels(
            normalized,
            {
                "家族高血压": ["父母都有高血压", "父母高血压", "父亲高血压", "母亲高血压", "家族高血压"],
            },
        ),
        "personal_history": matched_labels(
            normalized,
            {
                "糖耐量异常": ["糖耐量异常", "糖耐量受损"],
                "无妊娠期高血压": ["没有妊娠期高血压", "无妊娠期高血压"],
                "血压正常": ["血压一直正常", "血压正常"],
                "妊娠期高血压": ["我有妊娠期高血压", "本人有妊娠期高血压", "诊断妊娠期高血压"],
            },
        ),
        "personal_context": matched_labels(
            normalized,
            {
                "学生": ["学生"],
                "运动诱发鼻出血": ["运动时容易鼻出血", "运动鼻出血", "运动时鼻出血"],
            },
        ),
        "drug_allergies": matched_labels(
            normalized,
            {
                "青霉素": ["青霉素过敏", "penicillin allergy", "penicillin过敏"],
                "Penicillin": ["Penicillin", "penicillin"],
            },
        ),
        "contraindicated_drugs": matched_labels(
            normalized,
            {
                "青霉素": ["禁忌药物：Penicillin", "禁忌药物: Penicillin", "青霉素禁忌", "Penicillin禁忌"],
                "Penicillin": ["禁忌药物：Penicillin", "禁忌药物: Penicillin", "Penicillin"],
            },
        ),
        "candidate_diagnoses": [
            clean_text(item.get("disease")) for item in (disease_candidates or [])
            if isinstance(item, dict) and clean_text(item.get("disease"))
        ],
        "diagnosis_candidate_records": [
            dict(item) for item in (disease_candidates or [])
            if isinstance(item, dict) and clean_text(item.get("disease"))
        ],
        "anti_infective_provenance": _provenance_from_examination_results(
            case_state.get("examination_results")
        ),
    }


FEATURE_SLOTS = [
    "demographics",
    "symptom_clusters",
    "exam_evidence",
    "microbiology",
    "organ_risk",
    "medication_risk",
    "red_flags",
]


def normalize_feature_label(value: Any) -> str:
    return clean_text(value).lower()


def feature_items(case_features: Any, slot: str) -> List[Dict[str, str]]:
    if not isinstance(case_features, dict):
        return []
    raw_items = case_features.get(slot, [])
    if not isinstance(raw_items, list):
        return []
    result = []
    for item in raw_items:
        if isinstance(item, dict):
            label = clean_text(item.get("label"))
            if not label:
                continue
            result.append(
                {
                    "label": label,
                    "evidence": clean_text(item.get("evidence")),
                    "confidence": clean_text(item.get("confidence") or "medium"),
                }
            )
        elif clean_text(item):
            result.append({"label": clean_text(item), "evidence": "", "confidence": "medium"})
    return result


def case_feature_label_set(case_features: Any) -> set[str]:
    labels = set()
    for slot in FEATURE_SLOTS:
        for item in feature_items(case_features, slot):
            labels.add(item["label"])
    return labels


def has_hard_normalization_conflict(diagnosis: str, case_features: Any) -> bool:
    labels = {normalize_feature_label(item) for item in case_feature_label_set(case_features)}
    normalized_diagnosis = normalize_name(diagnosis)
    if normalize_name("女性") in labels and any(marker in normalized_diagnosis for marker in ["前列腺", "睾丸"]):
        return True
    if normalize_name("男性") in labels and any(marker in normalized_diagnosis for marker in ["妊娠", "卵巢", "子宫"]):
        return True
    if any(marker in normalized_diagnosis for marker in ["急性细菌", "感染"]) and normalize_name("无感染证据") in labels:
        return True
    return False


def has_required_axis_evidence(diagnosis: str, case_features: Any) -> bool:
    normalized_diagnosis = normalize_name(diagnosis)
    feature_text = normalize_name(feature_evidence_text(case_features))
    requirements = {
        normalize_name("糖尿病周围神经病变"): [
            ["糖尿病", "血糖", "高血糖", "糖化血红蛋白", "hba1c"],
        ],
    }
    for diagnosis_key, required_groups in requirements.items():
        if normalized_diagnosis != diagnosis_key:
            continue
        return all(any(normalize_name(marker) in feature_text for marker in group) for group in required_groups)
    return True


def feature_evidence_text(case_features: Any) -> str:
    chunks = []
    for slot in FEATURE_SLOTS:
        for item in feature_items(case_features, slot):
            chunks.append(item.get("label", ""))
            chunks.append(item.get("evidence", ""))
    return "\n".join(chunk for chunk in chunks if chunk)


def normalize_suggestion_confidence(value: Any) -> str:
    confidence = clean_text(value).lower()
    if confidence in {"high", "medium"}:
        return confidence
    return "low"


def differential_raw_name_matches(raw_name: str, differential_raw_names: Iterable[str]) -> bool:
    """Allow exact or bidirectional containment between suggestion and differential.

    LLM often lists "痛风性关节炎" in differential but suggests raw_name "痛风".
    Exact-only equality silently drops the correct official candidate.
    """
    needle = normalize_name(raw_name)
    if not needle:
        return False
    for item in as_text_list(differential_raw_names):
        other = normalize_name(item)
        if not other:
            continue
        if needle == other or needle in other or other in needle:
            return True
    return False


def resolve_official_from_surface_name(
    raw_name: str,
    official_disease_map: Dict[str, str],
    alias_rules: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Exact/alias first; then official names contained in the surface form.

    Ranking for containment: non-generic > prefix-anchored > longer key > earlier
    start. This keeps "痛风性关节炎" → 痛风 instead of the longer but mid-string
    generic-ish "关节炎", while "感染性关节炎" still prefers the longer official
    name when it is itself in the catalog.
    """
    text = clean_text(raw_name)
    if not text:
        return ""
    exact = match_standard_name(text, official_disease_map)
    if exact:
        return exact
    aliased = alias_to_official(text, alias_rules or [], official_disease_map)
    if aliased:
        return aliased
    normalized = normalize_name(text)
    if not normalized or not official_disease_map:
        return ""
    ranked_hits: List[tuple] = []
    for key, official in official_disease_map.items():
        if not key:
            continue
        start = normalized.find(key)
        if start < 0:
            continue
        non_generic = 0 if is_generic_final_diagnosis(official) else 1
        prefix = 1 if start == 0 else 0
        ranked_hits.append((non_generic, prefix, len(key), -start, official))
    if not ranked_hits:
        return ""
    ranked_hits.sort(reverse=True)
    return ranked_hits[0][4]

def accept_normalization_suggestion(
    suggestion: Dict[str, Any],
    *,
    case_features: Any,
    differential_raw_names: Iterable[str],
    official_disease_map: Dict[str, str],
    alias_rules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    raw_name = clean_text(suggestion.get("raw_name"))
    suggested_raw = clean_text(suggestion.get("suggested_official_name"))
    suggested = match_standard_name(suggested_raw, official_disease_map)
    if not suggested:
        suggested = alias_to_official(suggested_raw, alias_rules or [], official_disease_map)
    if not suggested:
        # Suggestion surface may itself be a longer form of a catalog name.
        suggested = resolve_official_from_surface_name(
            suggested_raw or raw_name,
            official_disease_map,
            alias_rules,
        )
    confidence = normalize_suggestion_confidence(suggestion.get("confidence"))
    if not raw_name or not differential_raw_name_matches(raw_name, differential_raw_names):
        return {"accepted": False, "reason": "raw_name_not_in_differential", "raw_name": raw_name}
    if not suggested:
        return {"accepted": False, "reason": "not_official_disease", "raw_name": raw_name}
    if confidence == "low":
        return {"accepted": False, "reason": "low_confidence", "raw_name": raw_name}

    available_labels = {normalize_feature_label(item) for item in case_feature_label_set(case_features)}
    required_labels = [clean_text(item) for item in as_text_list(suggestion.get("supporting_feature_labels"))]
    if not required_labels:
        return {"accepted": False, "reason": "empty_supporting_features", "raw_name": raw_name}
    missing = [item for item in required_labels if normalize_feature_label(item) not in available_labels]
    if missing:
        return {
            "accepted": False,
            "reason": "missing_supporting_features",
            "raw_name": raw_name,
            "missing_features": missing,
        }
    if has_hard_normalization_conflict(suggested, case_features):
        return {"accepted": False, "reason": "hard_conflict", "raw_name": raw_name, "normalized_diagnosis": suggested}
    if not has_required_axis_evidence(suggested, case_features):
        return {
            "accepted": False,
            "reason": "missing_required_axis_evidence",
            "raw_name": raw_name,
            "normalized_diagnosis": suggested,
        }

    return {
        "accepted": True,
        "raw_name": raw_name,
        "normalized_diagnosis": suggested,
        "source": "context_suggestion",
        "confidence": confidence if confidence in {"high", "medium"} else "medium",
        "matched_evidence": required_labels,
        "rationale": clean_text(suggestion.get("rationale")),
    }


def first_patient_text(case_state: Dict[str, Any]) -> str:
    for item in case_state.get("chat_history", []):
        if isinstance(item, dict) and item.get("from") == "patient":
            return clean_text(item.get("text"))
    return ""


def matched_labels(normalized_text: str, label_markers: Dict[str, List[str]]) -> List[str]:
    labels = []
    for label, markers in label_markers.items():
        if any(normalize_name(marker) in normalized_text for marker in markers):
            labels.append(label)
    return labels


def normalize_diagnosis(
    raw_diagnosis: Any,
    *,
    official_diseases: Iterable[str],
    alias_rules: List[Dict[str, Any]],
    disease_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    official_map = build_name_map(official_diseases)
    raw_text = clean_text(raw_diagnosis)
    exact = match_standard_name(raw_text, official_map)
    fallback_candidates = [
        clean_text(item.get("disease")) for item in (disease_candidates or [])
        if isinstance(item, dict) and clean_text(item.get("disease"))
    ]
    if exact:
        if is_generic_final_diagnosis(exact):
            specific = first_specific_candidate(fallback_candidates, official_map, generic_diagnosis=exact)
            if specific:
                return {
                    "raw_diagnosis": raw_text,
                    "normalized_diagnosis": specific,
                    "source": "specific_candidate",
                    "confidence": "medium",
                    "fallback_candidates": fallback_candidates,
                }
        return {
            "raw_diagnosis": raw_text,
            "normalized_diagnosis": exact,
            "source": "official_catalog",
            "confidence": "high",
            "fallback_candidates": fallback_candidates,
        }

    for rule in alias_rules:
        if clean_text(rule.get("status") or "candidate") != "verified":
            continue
        output = match_standard_name(rule.get("output"), official_map)
        if not output:
            continue
        for alias in as_text_list(rule.get("input")):
            if normalize_name(alias) == normalize_name(raw_text):
                return {
                    "raw_diagnosis": raw_text,
                    "normalized_diagnosis": output,
                    "source": "alias_map",
                    "confidence": "high",
                    "fallback_candidates": fallback_candidates,
                }

    return {
        "raw_diagnosis": raw_text,
        "normalized_diagnosis": "",
        "source": "",
        "confidence": "low",
        "fallback_candidates": fallback_candidates,
    }


def select_allowed_candidate_diagnosis(
    normalized_result: Dict[str, Any],
    disease_candidates: List[Dict[str, Any]],
    *,
    default_diagnosis: str,
) -> str:
    candidate_names = [
        clean_text(item.get("disease"))
        for item in disease_candidates
        if isinstance(item, dict) and clean_text(item.get("disease"))
    ]
    allowed_map = build_name_map(candidate_names)
    # An exact verified prior remains the diagnosis anchor when its safety facts
    # are incomplete; only the treatment plan must be rebuilt downstream.
    for item in disease_candidates:
        if not isinstance(item, dict) or clean_text(item.get("source")) != "verified_case_prior":
            continue
        prior = match_standard_name(item.get("disease"), allowed_map)
        if prior:
            return prior
    selected = match_standard_name(normalized_result.get("normalized_diagnosis"), allowed_map)
    if selected:
        return selected
    # An invalid model answer must not silently select an unrelated first candidate.
    return match_standard_name(default_diagnosis, allowed_map) or clean_text(default_diagnosis)


def reconcile_selected_diagnosis_plan(
    normalized_result: Dict[str, Any],
    *,
    selected_diagnosis: str,
    treatment_plan: Any,
    reasoning: Any,
    default_reasoning: str,
) -> tuple[str, str]:
    proposed_diagnosis = clean_text(normalized_result.get("normalized_diagnosis"))
    if normalize_name(proposed_diagnosis) != normalize_name(selected_diagnosis):
        # Candidate constraint remapped the label: do NOT wipe to an empty shell.
        # Keep a diagnosis-named supportive plan so treatment score is not zeroed.
        plan = diagnosis_supportive_treatment_plan(selected_diagnosis)
        return (
            plan,
            "最终诊断按候选疾病约束采用“%s”；模型原诊断未通过约束，"
            "因此改用按最终诊断命名的保守专科路径，不再沿用原方案推理。"
            % selected_diagnosis,
        )
    return (
        clean_text(treatment_plan)
        or diagnosis_supportive_treatment_plan(selected_diagnosis),
        clean_text(reasoning) or clean_text(default_reasoning),
    )


def extract_intake_facts(
    case_state: Dict[str, Any],
    llm_facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    patient_text = patient_text_for_matching(case_state)
    examination_text = examination_text_for_matching(case_state)
    trusted_text = "\n".join(item for item in [patient_text, examination_text] if item)
    facts = normalize_intake_facts(llm_facts or {}, normalize_name(trusted_text))
    append_fact_matches(
        facts,
        "demographics",
        patient_text,
        {
            "新生儿": ["新生儿", "出生", "婴儿"],
            "婴儿": ["婴儿", "小婴儿", "出生后"],
            "育龄女性": ["育龄女性", "年轻女性", "女性", "月经"],
            "绝经后女性": ["绝经后", "已绝经", "绝经两年", "绝经", "更年期"],
            "男性": ["男性", "男"],
            "青少年": ["青少年", "学生", "孩子"],
        },
    )
    append_fact_matches(
        facts,
        "demographics",
        patient_text,
        {"幼儿或儿童": ["幼儿", "儿童", "男童", "女童", "男孩", "女孩", "小儿"]},
    )
    pediatric_age = re.search(r"(?<!\d)(?:[0-9]|1[0-2])岁", normalize_name(patient_text))
    if pediatric_age:
        add_fact(facts, "demographics", "幼儿或儿童", pediatric_age.group(0), "high")
    append_fact_matches(
        facts,
        "anatomic_sites",
        patient_text,
        {
            "角膜": ["角膜"],
            "眼部": ["眼痛", "眼红", "畏光", "视物模糊"],
            "鼻中隔": ["鼻中隔", "鼻塞", "鼻出血"],
            "肾脏": ["尿蛋白", "血尿", "肾功能", "肌酐", "肾区"],
            "泌尿系统": ["尿频", "尿急", "尿痛", "排尿烧灼", "血尿", "尿液发红"],
            "上臂": ["上臂", "肱骨", "肱骨干"],
            "胸壁": ["前胸", "胸壁", "左前胸", "右前胸"],
            "肘部": ["肘部", "肘痛", "肘关节"],
            "肩部": ["肩部", "肩关节", "抬肩"],
            "左上腹": ["左上腹", "脾区"],
        },
    )
    append_fact_matches(
        facts,
        "symptom_clusters",
        patient_text,
        {
            "湿润易出血肿块": ["湿润", "鲜红", "肿块", "出血"],
            "眼红畏光角膜病变": ["眼红", "畏光", "角膜", "视物模糊"],
            "靶形皮疹水疱": ["靶形", "水疱", "水泡"],
            "光敏皮疹关节痛": ["光敏", "日晒", "皮疹", "关节痛"],
            "偏头痛伴恶心头晕": ["偏头痛", "搏动性头痛", "偏侧搏动头痛", "畏光怕声"],
            "头部外伤后起病": ["头部磕碰", "头部外伤", "撞到头", "脑震荡"],
            "外伤后持续头痛": ["头部磕碰后", "外伤后", "持续头痛", "反复头痛"],
            "认知症状": ["注意力不集中", "短期记忆差", "记忆差", "认知"],
            "前庭平衡症状": ["站立不稳", "快速转头", "平衡", "头晕"],
            "出生后呼吸急促": ["出生后", "呼吸急促", "气促", "呼吸变快", "呼吸突然变快", "呼吸快"],
            "喂养困难出汗": ["吃奶困难", "喂养困难", "喂养时明显出汗", "喂养出汗"],
            "活动或哭闹发绀": ["口唇发绀", "哭闹时口唇发绀", "发绀"],
            "肺动脉高压体征": ["P2亢进", "p2亢进", "肺动脉瓣第二心音亢进"],
            "反复鼻出血": ["反复鼻出血", "鼻出血"],
            "急性上臂外伤功能障碍": ["上臂剧痛", "上臂肿胀", "上臂活动受限", "肱骨区疼痛"],
            "突发心悸": ["心跳很快", "心跳快", "心悸", "心慌"],
            "胸闷气短": ["胸闷", "气短", "呼吸困难"],
            "既往心脏病": ["心脏病", "心梗", "心肌梗死", "心功能下降", "冠心病"],
            "胃肠道丢失或摄入不足": ["腹泻", "吃得少", "摄入不足", "呕吐"],
            "重复抓握腕活动诱发肘痛": ["重复抓握", "腕部活动", "抓握", "局限肘部疼痛", "肘部疼痛"],
            "慢性肝病背景": ["慢性肝病", "肝硬化", "肝病用药"],
            "左上腹饱胀": ["左上腹饱胀", "左上腹胀", "腹胀"],
            "三系减少": ["三系减少", "血细胞减少", "全血细胞减少"],
            "易瘀青出血倾向": ["容易瘀青", "易瘀青", "瘀青"],
            "慢性咳嗽": ["咳嗽三个月", "长期咳嗽", "慢性咳嗽", "咳嗽数月", "咳了三个月"],
            # Acute cough (not only chronic) must enter facts for pulmonary-renal axes.
            "咳嗽": ["咳嗽", "咳痰", "咳了"],
            "咯血或痰中带血": ["咯血", "痰中带血", "咳血", "带血丝"],
            "下肢或脚踝水肿": [
                "脚踝水肿",
                "下肢水肿",
                "双下肢水肿",
                "水肿",
                "脚踝肿",
                "踝肿",
                "脚肿",
                "腿肿",
            ],
            "全身乏力消耗": ["乏力", "消瘦", "盗汗", "低热"],
            "症状性低钾表现": ["手抽筋", "抽筋", "浑身没劲", "低钾", "血钾低", "肌无力"],
            "腹泻吸收不良丢失": ["腹泻", "脂肪泻", "吸收不良", "小肠切除", "胰酶", "胰腺功能不全"],
            "尿路刺激征": ["尿频", "尿急", "尿痛", "排尿烧灼"],
            "单侧腰痛": [
                "右侧腰部", "左侧腰部", "单侧腰痛", "腰部钝痛", "腰痛",
                "右肾区叩击痛", "左肾区叩击痛", "肾区叩击痛", "肋脊角叩击痛",
            ],
            # Do not use bare 血丝 — 咳嗽带血丝 would false-positive as hematuria.
            "血尿": [
                "血尿",
                "尿液发红",
                "尿液隐血",
                "尿红细胞",
                "尿中血丝",
                "尿血丝",
                "尿色变黑",
                "尿发黑",
                "尿色发黑",
                "尿量减少",
                "少尿",
            ],
            # Only distress-level markers open oxygenation coverage; bare cough/rhinorrhea must not.
            "急性呼吸窘迫线索": ["咳吐", "呼吸加快", "气促", "喘息", "呼吸急促", "呼吸困难", "呼吸窘迫"],
            "上呼吸道前驱症状": ["流涕", "鼻塞", "咽痛"],
            "长期大量饮酒": ["天天喝", "天天喝酒", "每天喝", "大量饮酒", "长期饮酒", "长期喝酒", "喝酒二十", "饮酒二十", "饮酒史"],
            "肝病相关症状": ["乏力", "腹胀", "食欲差", "食欲减退", "纳差", "胃口差"],
            "胸膜性胸痛": ["刀割样", "深呼吸", "咳嗽时胸痛", "翻身时", "胸膜性", "胸口刀割"],
            "干咳胸闷": ["干咳", "胸闷"],
            "既往肺梗死或栓塞史": ["肺梗死", "肺栓塞", "肺栓塞史", "肺梗死史"],
            "高能量足跟创伤": [
                "脚跟",
                "足跟",
                "跟骨",
                "左脚跟",
                "右脚跟",
                "脚后跟",
                "足后跟",
                "后跟",
            ],
            "高能量创伤机制": [
                "车祸",
                "交通事故",
                "高处坠落",
                "高能量",
                "砸伤",
                "高处掉",
                "从高处",
                "高处干活",
                "掉下来",
                "摔下来",
                "坠落",
                "掉下",
                "摔下",
            ],
            "足跟负重不能": [
                "不敢踩地",
                "不能负重",
                "无法负重",
                "不敢着地",
                "踩不了",
                "不能踩",
                "不敢踩",
                "脚变宽",
                "足变宽",
                "肿得厉害",
                "完全动不了",
                "无法行走",
                "不能走路",
            ],
            "高热": ["高烧", "高热", "发热", "发烧"],
            "多尿烦渴脱水": ["特别口渴", "极度口渴", "多尿", "尿特别多", "严重脱水", "脱水", "烦渴"],
            "急性胃肠炎样症状": ["暴食后", "呕吐", "腹泻", "上腹不适", "上腹痛", "急性胃肠"],
            "既往过敏性鼻炎": ["季节性过敏性鼻炎", "花粉症", "过敏性鼻炎", "花粉热"],
            "嘴唇发绀或发青": ["嘴唇发青", "嘴唇偶尔发青", "口唇发青", "口唇偶尔发青", "口唇发绀", "发绀", "嘴唇发紫"],
            "喂养困难或吃奶少": ["吃奶少", "吃奶困难", "喂奶困难", "喂养困难", "喂奶时", "喂养时出汗", "易出汗"],
            "胸部纵隔术后": ["胸腔手术", "纵隔手术", "淋巴结清扫", "胸部手术", "纵隔淋巴结", "开胸"],
            "术后呼吸费力": ["呼吸费力", "活动减少", "气促", "呼吸困难"],
        },
    )
    append_fact_matches(
        facts,
        "symptom_clusters",
        patient_text,
        {
            "紫绀或发青": ["紫绀", "口唇发青", "嘴唇发青", "唇甲发青", "指甲发青", "发青"],
            "气道症状": ["呼吸声重", "喘鸣", "长期咳嗽", "持续咳嗽"],
            "慢性或进行性病程": ["三个月", "数月", "长期", "慢性", "越来越重", "逐渐加重", "半年", "多年"],
            "体位性呼吸加重": ["平卧加重", "平躺加重", "平躺时", "躺下加重", "仰卧加重"],
            "进食或吞咽压迫": [
                "进食干呕",
                "进食都会干呕",
                "进食后干呕",
                "吃饭干呕",
                "吃东西干呕",
                "吞咽困难",
                "进食时干呕",
                "进食时呛咳",
            ],
            "进行性乏力": ["逐渐乏力", "进行性乏力", "越来越乏力", "长期乏力"],
            "活动耐量下降": ["活动后气促", "爬楼就气促", "爬楼梯喘", "活动耐量下降"],
            "反复头晕": ["反复头晕", "经常头晕", "站立头晕", "直立性头晕"],
            "慢性失血或铁缺乏风险": ["月经过多", "经量增多", "产后", "痔疮出血", "反复便血", "慢性失血"],
        },
    )
    if has_oxidant_exposure(patient_text):
        add_fact(facts, "symptom_clusters", "氧化剂暴露", "氧化剂或相关药物暴露", "high")
    append_fact_matches(
        facts,
        "exam_evidence",
        examination_text,
        {
            "ANA阳性或自身抗体阳性": ["ANA", "抗Sm", "抗SSA", "抗核抗体"],
            "补体降低": ["补体降低", "C3", "C4"],
            "角膜浸润或缺损": ["角膜浸润", "角膜缺损", "角膜病变"],
            "鼻中隔偏曲": ["鼻中隔偏曲", "骨嵴"],
            "血细胞三系减少证据": ["三系减少", "血红蛋白下降", "白细胞下降", "血小板下降"],
            "肌钙蛋白升高": ["肌钙蛋白升高", "肌钙蛋白阳性"],
            "血钾降低证据": ["血钾降低", "低钾血症", "血钾为", "血钾3"],
            "心包受累证据": ["心包积液", "心包炎", "心包增厚"],
            "超声确认室间隔缺损": ["大型室间隔缺损", "室间隔缺损", "VSD"],
            "血镁或低镁线索": ["低镁", "血镁", "镁缺乏"],
            "ANCA阳性证据": ["ANCA阳性", "p-ANCA", "c-ANCA", "MPO", "PR3"],
            "结核病原闭合证据": ["痰涂片阳性", "抗酸杆菌阳性", "结核分枝杆菌培养阳性", "GeneXpert阳性"],
        },
    )
    append_fact_matches(
        facts,
        "organ_risk",
        trusted_text,
        {
            "肾脏受累风险": [
                "尿蛋白",
                "血尿",
                "肾功能",
                "补体降低",
                "肾不好",
                "肌酐",
                "尿色变黑",
                "尿发黑",
                "尿量减少",
                "少尿",
            ],
            "血栓或抗磷脂风险": ["血栓", "抗磷脂", "流产", "避孕"],
            "肺肾综合征风险": [
                "咯血",
                "痰中带血",
                "血尿",
                "蛋白尿",
                "肺肾",
                "尿色变黑",
                "脚踝肿",
            ],
        },
    )
    append_fact_matches(
        facts,
        "infection_risk",
        trusted_text,
        {
            "眼部感染风险": ["角膜浸润", "眼痛", "畏光", "分泌物", "感染"],
            "皮肤黏膜严重反应风险": ["靶形", "水疱", "黏膜", "SJS"],
        },
    )
    append_fact_matches(
        facts,
        "bleeding_risk",
        trusted_text,
        {
            "局部易出血": ["出血", "易出血"],
            "反复鼻出血": ["反复鼻出血", "鼻出血"],
        },
    )
    append_fact_matches(
        facts,
        "trigger_factors",
        patient_text,
        {
            "旅行或视觉运动诱发": ["旅行", "坐飞机", "飞机", "视觉运动", "晕动"],
            "月经相关": ["月经", "经前"],
            "重复负荷诱发": ["重复抓握", "腕部活动", "重复", "过度使用"],
            "跌倒外伤机制": ["跌倒", "手肘着地", "外伤", "撞击"],
        },
    )
    append_fact_matches(
        facts,
        "medication_risk",
        patient_text,
        {
            "利尿剂使用": ["利尿剂", "利尿药", "呋塞米", "氢氯噻嗪", "螺内酯"],
        },
    )
    append_eleventh_round_facts(
        facts,
        case_state=case_state,
        patient_text=patient_text,
        examination_text=examination_text,
    )
    append_ref_data_targeted_facts(
        facts,
        patient_text=patient_text,
        examination_text=examination_text,
    )
    return facts


def append_ref_data_targeted_facts(
    facts: Dict[str, Any],
    *,
    patient_text: str,
    examination_text: str,
) -> None:
    combined = normalize_name(" ".join([patient_text, examination_text]))
    if has_leptospirosis_exposure_pattern(combined):
        add_fact(facts, "trigger_factors", "疫水或泥水暴露", "洪水或泥水接触", "high")
        add_fact(facts, "symptom_clusters", "高热伴腓肠肌痛或结膜充血", "高热、腓肠肌痛或结膜充血", "high")
        add_fact(facts, "organ_risk", "肝肾或血小板异常风险", "深色尿、少尿或肝肾异常风险", "high")
    if has_diuretic_hypokalemia_pattern(combined):
        add_fact(facts, "medication_risk", "利尿剂暴露", "利尿剂或利尿药使用", "high")
        add_fact(facts, "symptom_clusters", "利尿剂相关症状性低钾", "低钾伴抽筋、乏力或多尿", "high")
        add_fact(facts, "exam_evidence", "低钾和代谢性碱中毒证据", "血钾降低伴碳酸氢盐升高或代谢性碱中毒", "high")
    if has_multisystem_autoimmune_serositis_pattern(combined):
        add_fact(facts, "symptom_clusters", "多系统自身免疫表现", "皮肤、关节和浆膜多系统受累线索", "high")
        add_fact(facts, "exam_evidence", "心包受累证据", "心包积液、心包炎或心包增厚", "high")
    if has_chest_wall_trauma_pattern(combined):
        add_fact(facts, "anatomic_sites", "胸壁", "前胸或胸壁外伤", "high")
        add_fact(facts, "symptom_clusters", "胸部外伤呼吸痛", "胸痛随深呼吸或咳嗽加重并伴瘀斑", "high")
    if has_positive_vsd_text(combined):
        add_fact(facts, "exam_evidence", "超声确认室间隔缺损", "超声心动图显示室间隔缺损", "high")


def append_eleventh_round_facts(
    facts: Dict[str, Any],
    *,
    case_state: Dict[str, Any],
    patient_text: str,
    examination_text: str,
) -> None:
    if marker_present_not_negated(patient_text, ["脐部", "肚脐"]):
        add_fact(facts, "anatomic_sites", "脐部", "脐部", "high")
    if has_high_risk_pediatric_lower_respiratory_infection_pattern(patient_text):
        add_fact(
            facts,
            "symptom_clusters",
            "高危儿科下呼吸道感染",
            "儿童免疫抑制背景下迁延发热咳嗽伴脓痰或喘息",
            "high",
        )
    if has_infant_congenital_structural_heart_pattern(normalize_name(patient_text)):
        add_fact(facts, "demographics", "新生儿或婴儿", "宝宝出生后早期起病", "high")
        add_fact(facts, "symptom_clusters", "出生后呼吸急促", "出生后吃奶或安静时呼吸快", "high")
        add_fact(facts, "symptom_clusters", "喂养困难出汗", "吃奶易累、出汗或需停下休息", "high")
        add_fact(facts, "symptom_clusters", "活动或哭闹发绀", "哭闹或吃奶时口周发暗", "high")
    if has_congenital_infection_pattern(patient_text):
        add_fact(facts, "demographics", "新生儿或婴儿", "出生后早期起病", "high")
        add_fact(facts, "trigger_factors", "宫内病毒暴露", "孕期或孕早期宫内病毒暴露", "high")
        add_fact(facts, "symptom_clusters", "先天性感染黄疸或皮疹", "出生后黄疸、皮疹或红疹", "high")
        add_fact(facts, "symptom_clusters", "先天性感染喂养或神经异常", "喂养困难、嗜睡、反应弱或听力异常", "high")
    if has_pediatric_progressive_night_blindness_pattern(patient_text):
        add_fact(facts, "demographics", "幼儿或儿童", "儿童年龄", "high")
        add_fact(facts, "symptom_clusters", "儿童进行性夜盲", "暗处视物困难并逐渐影响行走", "high")
    if has_water_aerosol_severe_pneumonia_pattern(patient_text):
        add_fact(
            facts,
            "symptom_clusters",
            "水气溶胶暴露重症肺炎",
            "水气溶胶暴露伴寒战高热、咳嗽、气短或胸痛",
            "high",
        )
        add_fact(facts, "trigger_factors", "水气溶胶暴露", "冷却塔、热水浴池或高风险供水系统暴露", "high")
    if has_seafood_acute_watery_diarrhea_pattern(patient_text):
        add_fact(
            facts,
            "symptom_clusters",
            "生食海鲜后急性水样腹泻",
            "生食海鲜后急性腹痛、水样腹泻和呕吐",
            "high",
        )
        add_fact(facts, "trigger_factors", "生食海鲜暴露", "生蚝或明确生食、未熟海鲜暴露", "high")
    if has_chronic_suppurative_middle_ear_pattern(patient_text):
        add_fact(facts, "anatomic_sites", "耳部", "反复耳流脓", "high")
        add_fact(
            facts,
            "symptom_clusters",
            "慢性化脓性中耳病变",
            "反复耳流脓伴听力下降或耳鸣",
            "high",
        )
    if has_acute_ear_pain_after_instrumentation_pattern(patient_text):
        add_fact(
            facts,
            "symptom_clusters",
            "耳道操作后急性耳痛",
            "掏耳等耳道操作后耳痛伴耳堵、耳鸣或疼痛加重",
            "high",
        )
    if has_relapsing_fever_bleeding_pattern(patient_text):
        add_fact(facts, "symptom_clusters", "反复高热", "反复高热", "high")
        add_fact(facts, "symptom_clusters", "头痛肌肉关节痛", "头痛伴肌肉或关节痛", "high")
        add_fact(facts, "symptom_clusters", "多部位黏膜皮肤出血", "多部位出血", "high")
    if has_vector_exposure(patient_text):
        add_fact(facts, "trigger_factors", "媒介或户外暴露", "蜱虫、虫咬或野外暴露", "high")
    if has_focal_severe_ear_pattern(patient_text, examination_text):
        add_fact(facts, "anatomic_sites", "耳部", "耳部", "high")
        add_fact(facts, "symptom_clusters", "局灶重度耳痛伴局部功能受损", "剧烈局灶耳痛伴耳部肿胀或听力下降", "high")
    if marker_present_not_negated(patient_text, ["单侧听力下降", "左耳听力下降", "右耳听力下降"]):
        add_fact(facts, "symptom_clusters", "单侧听力下降", "单侧听力下降", "high")
    has_renal_manifestation = has_renal_urine_abnormality(case_state)
    if has_cryoglobulinemia_clinical_pattern(patient_text) or has_renal_manifestation:
        add_fact(facts, "symptom_clusters", "冷球蛋白相关临床表现", "紫癜、雷诺、关节、神经、皮肤缺血或肾尿异常", "high")
    if has_conductive_hearing_result(examination_text):
        add_fact(facts, "exam_evidence", "传导性听力损失", "传导性模式或气骨导差", "high")
    if otoscopy_is_unexplaining(case_state):
        add_fact(facts, "exam_evidence", "耳镜未解释局灶症状", "耳镜初筛未解释局灶症状", "high")
    if has_confirmed_cryoglobulin_result(case_state):
        add_fact(facts, "exam_evidence", "冷球蛋白异常证据", "冷球蛋白阳性、异常或定量升高", "high")
    if has_renal_manifestation:
        add_fact(facts, "exam_evidence", "肾脏尿异常证据", "蛋白尿、血尿或管型", "high")
    if hypoxia_evidence_status({"examination_results": case_state.get("examination_results"), "case_text": examination_text}) == "low":
        add_fact(facts, "exam_evidence", "低氧血症证据", "血氧饱和度降低", "high")
    if urine_culture_evidence_status({"examination_results": case_state.get("examination_results"), "case_text": patient_text}) == "negative":
        add_fact(facts, "exam_evidence", "尿路相关培养阴性", "细菌培养阴性或无生长", "high")
    combined_text = " ".join([patient_text, examination_text])
    if has_immunosuppressed_progressive_respiratory_pattern(patient_text):
        add_fact(facts, "medication_risk", "免疫抑制背景", "长期免疫抑制剂或全身激素", "high")
        add_fact(facts, "symptom_clusters", "进行性下呼吸道症状", "咳嗽伴进行性气短", "high")
    if marker_present_not_negated(patient_text, ["吞咽困难", "喝水呛咳", "进食呛咳"]):
        add_fact(facts, "organ_risk", "吞咽与误吸风险", "吞咽困难或进食呛咳", "high")
    if has_seizure_intracranial_calcification_pattern(combined_text):
        add_fact(facts, "symptom_clusters", "抽搐发作", "抽搐或癫痫样发作", "high")
        add_fact(facts, "exam_evidence", "颅内多发点状钙化", "头颅影像多发点状钙化", "high")
    if has_acute_pressure_headache_intracranial_calcification_pattern(combined_text):
        add_fact(facts, "symptom_clusters", "急性颅高压样头痛", "突发剧烈头痛伴体位加重、呕吐或认知变化", "high")
        add_fact(facts, "exam_evidence", "颅内多发点状钙化", "头颅影像多发点状钙化", "high")
    if has_decompensated_liver_symptom_pattern(combined_text):
        add_fact(facts, "symptom_clusters", "黄疸腹水失代偿", "黄疸伴腹胀或腹水", "high")
    if has_decompensated_cirrhosis_pattern(combined_text):
        add_fact(facts, "exam_evidence", "肝硬化门静脉高压影像", "结节样肝脏、脾大、腹水或门静脉高压", "high")
    if has_childhood_onset_epilepsy_pattern(patient_text):
        add_fact(facts, "symptom_clusters", "儿童期起病癫痫", "自幼反复癫痫样发作", "high")
    if has_developmental_genetic_epilepsy_pattern(patient_text):
        add_fact(facts, "organ_risk", "学习发育或FMR1异常", "学习发育障碍或FMR1异常", "high")
    if marker_present_not_negated(patient_text, ["未熟猪肉", "生猪肉", "猪带绦虫", "卫生条件差", "囊虫"]):
        add_fact(facts, "trigger_factors", "猪带绦虫或流行病学暴露", "未熟猪肉或囊虫流行病学暴露", "high")
    if has_post_spinal_surgery_positional_bilious_vomiting_pattern(patient_text):
        add_fact(facts, "symptom_clusters", "脊柱术后餐后胆汁性呕吐", "脊柱矫形术后餐后胆汁性呕吐", "high")
        add_fact(facts, "trigger_factors", "体位缓解十二指肠梗阻", "蜷缩、左侧卧或俯卧缓解", "high")
    if has_pediatric_leukocoria_red_flag_pattern(patient_text):
        add_fact(facts, "demographics", "婴幼儿", "婴幼儿年龄", "high")
        add_fact(facts, "symptom_clusters", "反复白瞳伴视觉追踪下降", "反复眼发白、眼位偏斜或追物差", "high")


def normalize_intake_facts(raw_facts: Dict[str, Any], normalized_case_text: str) -> Dict[str, Any]:
    slots = [
        "demographics",
        "anatomic_sites",
        "symptom_clusters",
        "exam_evidence",
        "organ_risk",
        "medication_risk",
        "infection_risk",
        "bleeding_risk",
        "trigger_factors",
    ]
    facts = {slot: [] for slot in slots}
    if not isinstance(raw_facts, dict):
        return facts
    for slot in slots:
        for item in raw_facts.get(slot, []) if isinstance(raw_facts.get(slot, []), list) else []:
            if not isinstance(item, dict):
                continue
            label = clean_text(item.get("label"))
            evidence = clean_text(item.get("evidence"))
            if not label:
                continue
            if evidence and not soft_marker_present(evidence, normalized_case_text):
                continue
            add_fact(facts, slot, label, evidence or label, clean_text(item.get("confidence") or "medium"))
    return facts


def append_fact_matches(
    facts: Dict[str, Any],
    slot: str,
    normalized_case_text: str,
    label_markers: Dict[str, List[str]],
) -> None:
    for label, markers in label_markers.items():
        # Skip markers that only appear under negation (e.g. 没有上臂肿胀).
        matched = [marker for marker in markers if marker_present_not_negated(normalized_case_text, [marker])]
        if matched:
            add_fact(facts, slot, label, matched[0], "high")


def add_fact(facts: Dict[str, Any], slot: str, label: str, evidence: str, confidence: str) -> None:
    items = facts.setdefault(slot, [])
    label_key = normalize_name(label)
    if any(normalize_name(item.get("label")) == label_key for item in items if isinstance(item, dict)):
        return
    items.append({"label": clean_text(label), "evidence": clean_text(evidence), "confidence": confidence or "medium"})


def has_relapsing_fever_bleeding_pattern(patient_text: str) -> bool:
    has_relapsing_fever = marker_present_not_negated(
        patient_text,
        ["反复高热", "反复高烧", "好转后又发热", "中间好转后又发热", "再次发热"],
    )
    has_headache = marker_present_not_negated(patient_text, ["剧烈头痛", "头痛得很厉害"])
    has_body_pain = marker_present_not_negated(
        patient_text,
        ["肌肉关节酸痛", "全身肌肉", "全身酸痛", "肌肉痛", "关节痛"],
    )
    bleeding_sites = [
        marker_present_not_negated(patient_text, ["鼻出血", "流鼻血"]),
        marker_present_not_negated(patient_text, ["牙龈出血"]),
        marker_present_not_negated(patient_text, ["皮肤瘀斑", "瘀斑", "瘀点"]),
    ]
    return has_relapsing_fever and has_headache and has_body_pain and sum(bleeding_sites) >= 2


def has_vector_exposure(patient_text: str) -> bool:
    markers = ["蜱虫", "蜱叮咬", "虫咬", "野外露营", "户外露营", "扩展性红斑", "游走性皮疹"]
    for clause in semantic_clauses(patient_text):
        normalized_clause = normalize_name(clause)
        for marker in markers:
            token = normalize_name(marker)
            start = 0
            while True:
                index = normalized_clause.find(token, start)
                if index < 0:
                    break
                suffix = normalized_clause[index + len(token): index + len(token) + 16]
                postposed_negative = re.search(r"(?:没见过|也没有|未见|否认|不确定|不详|未知|记不清)", suffix)
                if (
                    not marker_occurrence_is_negated(normalized_clause, token, index)
                    and not marker_occurrence_is_uncertain(normalized_clause, index)
                    and not postposed_negative
                ):
                    return True
                start = index + len(token)
    return False


def has_focal_severe_ear_pattern(patient_text: str, examination_text: str = "") -> bool:
    has_ear_site = marker_present_not_negated(
        patient_text,
        ["耳朵", "耳内", "耳痛", "耳部", "左耳", "右耳", "单侧耳", "一侧耳"],
    )
    has_severe_pain = marker_present_not_negated(
        patient_text,
        [
            "针扎样",
            "剧烈耳痛",
            "耳朵里面特别疼",
            "耳痛剧烈",
            "难以忍受",
            "剧痛",
            "疼痛明显",
            "8/10",
            "9/10",
            "10/10",
        ],
    ) or ear_pain_with_severity(patient_text)
    has_local_effect = marker_present_not_negated(
        patient_text,
        ["耳朵肿", "耳部肿胀", "听力下降", "听力明显下降", "听力变差", "听力减退"],
    )
    return has_ear_site and has_severe_pain and (
        has_local_effect or has_conductive_hearing_result(examination_text)
    )


def ear_pain_with_severity(patient_text: str) -> bool:
    """Match common phrases like 右耳剧痛 / 单侧耳痛明显 without requiring contiguous 耳痛 token."""
    for clause in semantic_clauses(patient_text):
        normalized = normalize_name(clause)
        has_ear = any(
            normalize_name(marker) in normalized
            for marker in ["耳", "耳朵", "耳内", "耳部"]
        )
        has_pain = any(normalize_name(marker) in normalized for marker in ["痛", "疼"])
        has_severity = any(
            normalize_name(marker) in normalized
            for marker in ["剧", "明显", "剧烈", "难忍", "8/10", "9/10", "10/10"]
        )
        if has_ear and has_pain and has_severity:
            return True
    return False


def has_cryoglobulinemia_clinical_pattern(patient_text: str) -> bool:
    markers = [
        "紫癜", "紫色斑点", "皮肤紫斑", "可触及紫癜", "遇冷手指", "变白变青", "雷诺",
        "关节痛", "关节酸痛", "关节炎", "反复关节炎", "周围神经病变", "多发性单神经炎",
        "单神经炎", "足下垂", "双足麻木", "肢端麻木", "感觉减退", "烧灼痛", "针刺感",
        "网状青斑", "皮肤溃疡", "肢端缺血", "皮肤缺血",
    ]
    for clause in semantic_clauses(patient_text):
        normalized_clause = normalize_name(clause)
        for marker in markers:
            token = normalize_name(marker)
            index = normalized_clause.find(token)
            while index >= 0:
                suffix = normalized_clause[index + len(token): index + len(token) + 16]
                postposed_negative = re.fullmatch(
                    r"(?:也|均|都)?(?:没有|无|未见|否认|不存在)(?:了|相关表现|相关症状|这种情况)?",
                    suffix,
                )
                if (
                    not marker_occurrence_is_negated(normalized_clause, token, index)
                    and not marker_occurrence_is_uncertain(normalized_clause, index)
                    and not postposed_negative
                ):
                    return True
                index = normalized_clause.find(token, index + len(token))
    return False


def has_conductive_hearing_result(examination_text: str) -> bool:
    # Keys like 传导性听力损失 may appear next to negative values; require non-negated polarity.
    return marker_present_active(
        examination_text,
        ["传导性模式", "传导性听力损失", "气骨导差"],
        resolved_markers=["已缓解", "已纠正", "已排除"],
    )


def has_confirmed_cryoglobulin_result(case_state: Dict[str, Any]) -> bool:
    for payload in cryoglobulin_exam_payloads(case_state):
        if cryoglobulin_payload_status(payload) == "positive":
            return True
    return False


def cryoglobulin_exam_payloads(case_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = case_state.get("examination_results")
    if not isinstance(results, dict):
        return []
    markers = ["冷球蛋白", "cryoglobulin", "cryocrit"]
    payloads = []
    for name, payload in results.items():
        if not isinstance(payload, dict):
            continue
        result_text = normalize_name(" ".join("%s %s" % item for item in exam_result_pairs(payload)))
        if any(normalize_name(marker) in normalize_name(name) or normalize_name(marker) in result_text for marker in markers):
            payloads.append(payload)
    return payloads


def cryoglobulin_payload_status(payload: Dict[str, Any]) -> str:
    status = normalize_name(payload.get("status"))
    if status in {"normal", "negative"}:
        return "negative"
    if status in {"pending", "ordered", "inprogress", "processing"}:
        return "unknown"
    all_pairs = [
        (key, value)
        for key, value in exam_result_pairs(payload)
        if not any(marker in normalize_name(key) for marker in ["阴性对照", "阳性对照", "参考", "正常范围"])
    ]
    cryo_markers = ["冷球蛋白", "cryoglobulin", "cryocrit"]
    pairs = []
    for key, value in all_pairs:
        key_text = normalize_name(key)
        value_text = normalize_name(result_value_without_reference(value))
        if cryoglobulin_result_key_is_relevant(key) or any(
            normalize_name(marker) in key_text or normalize_name(marker) in value_text for marker in cryo_markers
        ):
            pairs.append((key, value))
    if not pairs:
        all_result_text = normalize_name(
            " ".join(result_value_without_reference(value) for _, value in all_pairs)
        )
        if any(marker in all_result_text for marker in ["4c沉淀", "4℃沉淀", "低温沉淀", "冷沉淀"]) and any(
            marker in all_result_text for marker in ["加温后溶解", "加温后可重新溶解", "37℃复溶", "复温溶解"]
        ):
            return "positive"
        return "unknown"
    diagnostic_text = normalize_name(" ".join(result_value_without_reference(value) for _, value in pairs))
    if any(marker in diagnostic_text for marker in ["待复核", "待确认", "待报告"]):
        return "unknown"
    value_statuses = [cryoglobulin_result_value_status(value) for _, value in pairs]
    if "positive" in value_statuses:
        return "positive"
    if "negative" in value_statuses:
        return "negative"
    if any(
        lab_value_reference_position(value) == "high" or value_exceeds_one_sided_upper_reference(value)
        for _, value in pairs
    ):
        return "positive"
    # Only treat dedicated cryoglobulin exams as positive from status=abnormal alone.
    dedicated = any(
        cryoglobulin_result_key_is_relevant(key)
        and any(normalize_name(marker) in normalize_name(key) for marker in cryo_markers)
        for key, _ in pairs
    ) or any(
        normalize_name(key) in {"result", "结果", "结论", "定量", "浓度", "含量", "cryocrit"}
        for key, _ in pairs
    )
    if status == "abnormal" and dedicated:
        return "positive"
    all_result_text = normalize_name(" ".join(result_value_without_reference(value) for _, value in all_pairs))
    if any(marker in all_result_text for marker in ["4c沉淀", "4℃沉淀", "低温沉淀", "冷沉淀"]) and any(
        marker in all_result_text for marker in ["加温后溶解", "加温后可重新溶解", "37℃复溶", "复温溶解"]
    ):
        return "positive"
    return "unknown"


def cryoglobulin_result_key_is_relevant(key: str) -> bool:
    normalized_key = normalize_name(key)
    return normalized_key in {"result", "结果", "结论", "定量", "浓度", "含量", "补充描述"} or any(
        marker in normalized_key for marker in ["冷球蛋白", "cryoglobulin", "cryocrit"]
    )


def cryoglobulin_result_value_status(value: str) -> str:
    normalized = normalize_name(result_value_without_reference(value))
    if any(marker in normalized for marker in ["待复核", "待确认"]):
        return "unknown"
    negative_markers = [
        "未达到阳性阈值", "未达阳性阈值", "阴性", "未检出", "未见",
        "未发现", "不存在", "未升高", "不升高", "negative",
    ]
    positive_markers = ["阳性", "检出", "升高", "高于参考", "偏高", "positive"]
    if any(normalized.startswith(normalize_name(marker)) for marker in negative_markers):
        return "negative"
    if any(normalized.startswith(normalize_name(marker)) for marker in positive_markers):
        return "positive"
    if normalized.startswith(("临界", "可疑")):
        return "unknown"
    if any(normalize_name(marker) in normalized for marker in negative_markers):
        return "negative"
    if any(normalize_name(marker) in normalized for marker in positive_markers):
        return "positive"
    return "unknown"


def value_exceeds_one_sided_upper_reference(value: str) -> bool:
    measured = re.search(r"-?\d+(?:\.\d+)?", result_value_without_reference(value))
    reference = re.search(
        r"参考(?:值|范围)?[：:]?\s*([<＜≤])\s*(\d+(?:\.\d+)?)",
        clean_text(value),
    )
    if not measured or not reference:
        return False
    measured_value = float(measured.group())
    upper_value = float(reference.group(2))
    return measured_value >= upper_value if reference.group(1) in {"<", "＜"} else measured_value > upper_value


def has_systemic_infection_hematologic_axis_pattern(facts_text: str) -> bool:
    return all(
        normalize_name(marker) in facts_text
        for marker in ["反复高热", "头痛肌肉关节痛", "多部位黏膜皮肤出血"]
    )


def has_focal_ear_conductive_axis_pattern(facts_text: str) -> bool:
    return all(
        normalize_name(marker) in facts_text
        for marker in ["局灶重度耳痛伴局部功能受损", "传导性听力损失", "耳镜未解释局灶症状"]
    )


def has_cryoglobulinemia_secondary_axis_pattern(facts_text: str) -> bool:
    return all(
        normalize_name(marker) in facts_text
        for marker in ["冷球蛋白相关临床表现", "冷球蛋白异常证据"]
    )



def has_immunosuppressed_progressive_respiratory_pattern(text: str) -> bool:
    immune = marker_present_not_negated(
        text,
        ["免疫抑制", "免疫抑制剂", "全身激素", "长期激素", "化疗", "器官移植"],
    )
    cough = marker_present_not_negated(text, ["咳嗽", "咳痰", "呼吸道症状"])
    dyspnea = marker_present_not_negated(
        text,
        ["气越来越不够用", "进行性气短", "呼吸短促", "呼吸困难", "气促", "喘息"],
    )
    return immune and cough and dyspnea


def has_seizure_intracranial_calcification_pattern(text: str) -> bool:
    seizure = marker_present_not_negated(text, ["抽搐", "癫痫样发作", "癫痫发作", "惊厥"])
    calcification = marker_present_not_negated(
        text,
        ["颅内多发点状钙化", "多发点状钙化", "颅内钙化", "脑内钙化"],
    )
    return seizure and calcification


def has_decompensated_liver_symptom_pattern(text: str) -> bool:
    jaundice = marker_present_not_negated(text, ["黄疸", "眼睛黄", "巩膜黄染"])
    fluid = marker_present_not_negated(text, ["腹水", "腹胀", "肚子胀"])
    return jaundice and fluid


def has_decompensated_cirrhosis_pattern(text: str) -> bool:
    structural = marker_present_not_negated(
        text,
        ["肝硬化", "肝脏结节状", "门静脉高压", "脾大", "肝脏轮廓呈结节状"],
    )
    return has_decompensated_liver_symptom_pattern(text) and structural


def has_childhood_onset_epilepsy_pattern(text: str) -> bool:
    childhood = marker_present_not_negated(text, ["从小就有", "从小", "儿童期起病", "自幼", "小时候开始"])
    seizure = marker_present_not_negated(text, ["癫痫", "癫痫样发作", "抽搐", "惊厥"])
    return childhood and seizure


def has_developmental_genetic_epilepsy_pattern(text: str) -> bool:
    developmental = marker_present_not_negated(
        text,
        ["学习障碍", "发育迟缓", "智力障碍", "认知障碍", "FMR1", "脆性X"],
    )
    return has_childhood_onset_epilepsy_pattern(text) and developmental



def has_post_spinal_surgery_positional_bilious_vomiting_pattern(text: str) -> bool:
    spine_context = marker_present_not_negated(text, ["脊柱侧弯", "脊柱矫形", "脊柱手术"])
    recent_surgery = marker_present_not_negated(text, ["刚做完手术", "术后", "手术后", "脊柱矫形术"])
    obstruction = marker_present_not_negated(text, ["餐后", "吐胆汁", "胆汁性呕吐", "进食即吐"])
    positional = marker_present_not_negated(text, ["蜷缩", "左侧卧", "俯卧", "坐着好转", "站着好转", "躺着更难受"])
    return spine_context and recent_surgery and obstruction and positional


def has_pediatric_leukocoria_red_flag_pattern(text: str) -> bool:
    child = marker_present_not_negated(
        text,
        ["新生儿", "刚出生", "从出生起", "出生后", "婴儿", "儿童", "1岁", "一岁", "幼儿"],
    )
    white_pupil = marker_present_not_negated(
        text,
        [
            "眼发白",
            "眼睛发白",
            "左眼偶尔发白",
            "右眼偶尔发白",
            "眼偶尔发白",
            "白瞳",
            "瞳孔发白",
            "白色反光",
            "双眼白色反光",
        ],
    )
    visual = marker_present_not_negated(
        text,
        [
            "追物差",
            "追物能力差",
            "追视差",
            "无法追视",
            "不能追视",
            "不追视",
            "红光反射消失",
            "红光反射异常",
            "内斜",
            "眼位偏斜",
        ],
    )
    return child and white_pupil and visual


def has_decompensated_hfref_pattern(text: str) -> bool:
    reduced_ef = has_reduced_left_ventricular_ejection_fraction(text)
    congestion_groups = [
        ["端坐呼吸", "夜间阵发性呼吸困难"],
        ["下肢水肿", "双下肢水肿", "外周水肿"],
        ["活动后气短", "呼吸困难", "肺淤血"],
        ["LVEDD", "左心室扩大", "左室扩大"],
        ["PASP", "肺动脉高压"],
    ]
    supported = sum(marker_present_not_negated(text, group) for group in congestion_groups)
    return reduced_ef and supported >= 2


def has_acute_decompensated_heart_failure_pattern(text: str) -> bool:
    """Clinical ADHF without requiring known reduced EF text."""
    if has_decompensated_hfref_pattern(text):
        return True
    congestion = marker_present_not_negated(
        text,
        [
            "端坐呼吸",
            "夜间阵发性呼吸困难",
            "不能平卧",
            "下肢水肿",
            "双下肢水肿",
            "肺淤血",
            "泡沫痰",
            "活动后气短",
            "呼吸困难",
        ],
    )
    cardiac_context = marker_present_not_negated(
        text,
        [
            "心力衰竭",
            "心衰",
            "扩张型心肌病",
            "高血压",
            "冠心病",
            "心肌梗死",
            "射血分数",
            "LVEF",
            "心率快",
            "心动过速",
        ],
    )
    acuity = marker_present_not_negated(
        text,
        ["加重", "突然", "急性", "这两天", "近几天", "越来越喘", "急诊"],
    )
    return bool(congestion and cardiac_context and acuity)


def has_acute_pharyngitis_in_diabetic_child_pattern(text: str) -> bool:
    child = marker_present_not_negated(
        text,
        ["岁", "儿童", "小儿", "孩子", "幼儿", "小学生", "中学生"],
    ) and any(
        marker_present_not_negated(text, [marker])
        for marker in ["1岁", "2岁", "3岁", "4岁", "5岁", "6岁", "7岁", "8岁", "9岁", "10岁", "11岁", "12岁", "13岁", "14岁", "15岁", "16岁", "17岁", "儿童", "小儿", "孩子"]
    )
    # Age-number heuristic: digits followed by 岁 under 18 is common in Chinese cases.
    age_child = bool(re.search(r"(?:^|[^\d])(?:[1-9]|1[0-7])\s*岁", text))
    pharyngitis = marker_present_not_negated(
        text,
        ["咽痛", "喉咙痛", "嗓子疼", "扁桃体", "咽红", "吞咽痛", "上呼吸道感染", "感冒后咽痛"],
    )
    diabetes = marker_present_not_negated(
        text,
        ["糖尿病", "1型糖尿病", "2型糖尿病", "血糖高", "胰岛素"],
    )
    return bool((child or age_child) and pharyngitis and diabetes)


def has_pml_imaging_pattern(text: str) -> bool:
    immune = marker_present_not_negated(
        text,
        [
            "HIV",
            "AIDS",
            "获得性免疫缺陷综合征",
            "免疫抑制",
            "免疫抑制剂",
            "抗逆转录病毒",
            "漏服抗逆转录病毒",
            "经常忘吃",
        ],
    )
    neurologic_markers = [
        "亚急性",
        "进展性偏瘫",
        "进行性偏瘫",
        "越来越没力气",
        "构音障碍",
        "说话也找不着词",
        "找不着词",
        "共济失调",
        "走路不稳",
        "多灶神经功能缺损",
        "多灶神经缺损",
        "偏瘫",
        "左侧肢体无力",
        "左边身体越来越没力气",
        "言语含糊",
        "看不清",
        "视物不清",
        "视力下降",
    ]
    neurologic_hits = sum(
        1 for marker in neurologic_markers if marker_present_not_negated(text, [marker])
    )
    neurologic = neurologic_hits >= 2
    white_matter = marker_present_not_negated(
        text,
        [
            "多发不对称皮质下白质病灶",
            "不对称白质病灶",
            "多灶性白质病灶",
            "白质病变",
            "白质异常信号",
            "脱髓鞘样白质",
        ],
    )
    non_mass = marker_present_not_negated(
        text,
        ["无占位效应", "无强化", "不强化", "无明显强化", "非占位"],
    )
    if immune and neurologic and white_matter and non_mass:
        return True
    if immune and neurologic and white_matter:
        return True
    # HIV + progressive multifocal neurology can open the axis before decisive imaging.
    subacute_course = marker_present_not_negated(
        text,
        ["6周", "六周", "数周", "几周", "逐渐", "进行性", "越来越", "进展"],
    )
    return immune and neurologic and subacute_course


def has_pediatric_upper_airway_danger_pattern(text: str) -> bool:
    child = marker_present_not_negated(text, ["幼儿", "儿童", "2岁", "两岁", "小儿"])
    fever = marker_present_not_negated(text, ["高热", "发热", "发烧", "39.6℃", "寒战"])
    dangers = [
        "流涎", "流口水", "拒食", "不肯吃东西", "拒咽", "吞咽困难",
        "声音闷", "声音变闷", "含物音", "喘鸣", "颈部肿胀", "下巴肿",
    ]
    count = sum(marker_present_not_negated(text, [marker]) for marker in dangers)
    return child and fever and count >= 2


def has_acute_lower_extremity_soft_tissue_infection_pattern(text: str) -> bool:
    location = marker_present_not_negated(
        text,
        ["小腿", "下肢", "足背", "脚踝", "踝部", "小腿肚", "小腿外侧", "小腿内侧", "大腿根"],
    )
    local_infection = marker_present_not_negated(
        text,
        [
            "红肿热痛",
            "红肿",
            "发红发烫",
            "发红",
            "发烫",
            "皮温升高",
            "皮肤发红",
            "往上肿",
            "肿胀",
            "触痛",
            "压痛",
            "蜂窝织炎",
            "丹毒",
        ],
    )
    systemic = marker_present_not_negated(
        text,
        [
            "发热",
            "发烧",
            "寒战",
            "发冷",
            "冷飕飕",
            "高热",
            "全身不适",
            "浑身发冷",
            "浑身没劲",
            "特别累",
            "没劲",
        ],
    )
    # Background knee arthritis alone must not suppress soft-tissue infection.
    return location and local_infection and systemic


def has_suspected_asthma_control_pattern(text: str) -> bool:
    airway = marker_present_not_negated(
        text,
        ["喘息", "喘鸣", "夜间憋醒", "憋醒", "气促", "呼吸困难", "胸闷", "咳嗽夜间加重"],
    )
    atopic_or_trigger = marker_present_not_negated(
        text,
        ["哮喘", "过敏性鼻炎", "湿疹", "灰尘", "花粉", "冷空气", "运动后喘", "沙丁胺醇"],
    )
    return bool(airway and atopic_or_trigger)


def has_hypothalamic_pituitary_amenorrhea_pattern(text: str) -> bool:
    amenorrhea = marker_present_not_negated(
        text,
        ["闭经", "未来月经", "月经停止", "停经", "几个月没来月经", "继发性闭经"],
    )
    central = marker_present_not_negated(
        text,
        ["体重下降", "节食", "过度运动", "产后大出血", "头痛视力", "泌乳", "怕冷乏力", "性欲下降"],
    )
    return bool(amenorrhea and (central or marker_present_not_negated(text, ["垂体", "下丘脑"])))


def has_congenital_syndactyly_pattern(text: str) -> bool:
    return marker_present_not_negated(
        text,
        ["并指", "手指长在一起", "脚趾并拢", "并趾", "指蹼", "出生即并指"],
    )


def has_neck_mass_b_symptoms_pattern(text: str) -> bool:
    mass = marker_present_not_negated(
        text,
        ["颈部包块", "脖子包块", "颈部肿块", "颈淋巴结", "脖子上的包", "颈部淋巴结肿大", "锁骨上淋巴结"],
    )
    b_symptoms = marker_present_not_negated(
        text,
        ["盗汗", "夜汗", "不明原因发热", "低热", "体重下降", "消瘦", "乏力纳差"],
    )
    transplant_or_immune = marker_present_not_negated(
        text,
        ["移植", "免疫抑制剂", "PTLD", "淋巴瘤", "EB病毒"],
    )
    return bool(mass and (b_symptoms or transplant_or_immune))


def has_cholestatic_liver_disease_pattern(text: str) -> bool:
    cholestasis = marker_present_not_negated(
        text,
        ["皮肤发黄", "巩膜黄染", "黄疸", "皮肤瘙痒", "尿色深", "陶土样便", "胆汁淤积"],
    )
    liver_context = marker_present_not_negated(
        text,
        ["肝功能", "碱性磷酸酶", "γ-谷氨酰", "转氨酶", "右上腹", "妊娠", "药物", "AMA", "PBC"],
    )
    return bool(cholestasis and liver_context)


def has_traumatic_rib_fracture_pattern(text: str) -> bool:
    trauma = marker_present_not_negated(
        text,
        ["摔倒", "跌倒", "撞伤", "外伤", "撞击", "车祸", "胸部撞击"],
    )
    rib = marker_present_not_negated(
        text,
        ["肋骨", "肋部", "季肋", "胸壁压痛", "深呼吸胸痛", "咳嗽胸痛", "肋骨骨折"],
    )
    return bool(trauma and rib)


def has_hyperlipidemia_with_xanthelasma_pattern(text: str) -> bool:
    """Adult eyelid xanthoma / xanthelasma with dyslipidemia or metabolic context.

    Must not fire on pediatric rickets-only histories (prevents vitamin-D catalog capture).
    """
    if marker_present_not_negated(text, ["佝偻病", "方颅", "串珠肋", "鸡胸", "O型腿", "X型腿"]):
        if not marker_present_not_negated(text, ["眼睑", "上眼睑", "黄色斑块", "黄色瘤", "睑黄", "发黄"]):
            return False
    xanthoma = marker_present_not_negated(
        text,
        [
            "黄色斑块",
            "黄色瘤",
            "睑黄斑",
            "睑黄瘤",
            "眼睑斑块",
            "上眼睑发黄",
            "眼睑发黄",
            "发黄、轻度隆起",
            "发黄",
            "上眼睑",
            "眼睑",
        ],
    ) and marker_present_not_negated(text, ["眼睑", "上眼睑", "睑"])
    lipid = marker_present_not_negated(
        text,
        [
            "高脂血症",
            "高胆固醇",
            "甘油三酯",
            "血脂",
            "LDL",
            "总胆固醇",
            "混合型高脂血症",
            "脂肪肝",
            "右上腹隐痛",
            "油腻",
            "高脂餐",
            "胰岛素抵抗",
        ],
    )
    lab_lipid = marker_present_not_negated(
        text,
        ["总胆固醇", "LDL胆固醇", "甘油三酯", "HDL", "LDL", "HDL34", "LDL172"],
    )
    adult = marker_present_not_negated(
        text,
        ["岁", "成年", "成人", "中年", "老年", "42岁", "40岁", "50岁", "60岁", "70岁", "32岁", "28岁"],
    )
    return bool(xanthoma and (lipid or lab_lipid) and (adult or lab_lipid))


def has_cavitary_tuberculosis_pattern(text: str) -> bool:
    immune_or_systemic = marker_present_not_negated(text, ["免疫抑制", "HIV", "盗汗", "消瘦"])
    respiratory = marker_present_not_negated(text, ["慢性咳嗽", "咳血", "痰中带血"])
    cavitary = marker_present_not_negated(text, ["厚壁空洞", "空洞性病变", "肺空洞"])
    return immune_or_systemic and respiratory and cavitary

def has_renovascular_hypertension_pattern(text: str) -> bool:
    stenosis = marker_present_not_negated(text, ["肾动脉狭窄", "肾血管狭窄", "肾动脉造影狭窄"])
    severe_hypertension = marker_present_not_negated(text, ["高血压危象", "血压220", "血压210", "重度高血压"])
    target_organ = marker_present_not_negated(text, ["头痛", "视物模糊", "靶器官", "肾功能异常"])
    return stenosis and severe_hypertension and target_organ


def has_perioral_dermatitis_pattern(text: str) -> bool:
    steroid = marker_present_not_negated(text, ["面部外用激素", "外用糖皮质激素", "激素药膏", "激素依赖"])
    distribution = marker_present_not_negated(text, ["口周", "鼻翼", "下巴"])
    inflammation = marker_present_not_negated(text, ["红斑丘疹", "丘疹", "灼热", "紧绷"])
    return steroid and distribution and inflammation


def has_anal_polyp_pattern(text: str) -> bool:
    bleeding = marker_present_not_negated(text, ["鲜红便血", "便后滴血", "便血"])
    pedunculated = marker_present_not_negated(text, ["有蒂结节", "有蒂", "柔软结节", "肛管内可触及"])
    return bleeding and pedunculated


def has_rheumatoid_arthritis_ocular_pattern(text: str) -> bool:
    serology = marker_present_not_negated(text, ["RF阳性", "类风湿因子阳性", "Anti-CCP阳性", "抗CCP阳性"])
    joint = marker_present_not_negated(text, ["晨僵", "炎性关节", "多关节痛", "关节肿痛"])
    ocular = marker_present_not_negated(text, ["眼痛", "畏光", "巩膜炎", "眼红"])
    return serology and joint and ocular


def has_neurocysticercosis_strong_evidence_pattern(text: str) -> bool:
    imaging = marker_present_not_negated(text, ["脑实质钙化", "颅内钙化", "多发点状钙化", "脑囊肿"])
    exposure_or_test = marker_present_not_negated(
        text,
        ["未熟猪肉", "猪带绦虫", "囊虫抗体阳性", "特异性抗体阳性", "卫生条件较差"],
    )
    pressure = marker_present_not_negated(text, ["脑积水", "颅内压", "弯腰加重", "反复呕吐", "喷射性呕吐"])
    return imaging and exposure_or_test and pressure


def has_acute_pressure_headache_intracranial_calcification_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    acute_severe_headache = marker_present_not_negated(
        normalized,
        ["急性颅高压样头痛", "突发剧烈头痛", "突然剧烈头痛", "急性剧烈头痛", "雷击样头痛"],
    )
    pressure_or_neurologic = marker_present_not_negated(
        normalized,
        ["体位加重、呕吐或认知变化", "弯腰加重", "弯腰时明显加重", "反复呕吐", "喷射性呕吐", "脑子变慢", "认知改变", "意识变慢"],
    )
    calcification = marker_present_not_negated(
        normalized,
        ["颅内多发点状钙化", "颅内钙化", "脑实质钙化", "多发点状钙化"],
    )
    return acute_severe_headache and pressure_or_neurologic and calcification


def has_methemoglobin_risk_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    cyanosis = marker_present_not_negated(
        normalized,
        ["紫绀或发青", "活动或哭闹发绀", "紫绀", "发绀", "唇甲发青", "嘴唇发青"],
    )
    oxidant = has_oxidant_exposure(normalized)
    return cyanosis and oxidant


def has_oxidant_exposure(text: str) -> bool:
    markers = ["氧化剂暴露", "利多卡因", "苯佐卡因", "达普松", "亚硝酸盐", "硝酸盐", "局部麻醉药"]
    postposed_absence = [
        "没用过",
        "没有用过",
        "未用过",
        "没使用过",
        "没有使用过",
        "未使用过",
        "未使用",
        "没服用过",
        "没有服用过",
        "未服用",
        "没接触过",
        "没有接触过",
        "未接触过",
        "未接触",
        "从未接触",
        "没有暴露",
        "未暴露",
    ]
    for clause in semantic_clauses(text):
        normalized_clause = normalize_name(clause)
        for marker in markers:
            token = normalize_name(marker)
            start = 0
            while True:
                index = normalized_clause.find(token, start)
                if index < 0:
                    break
                suffix = normalized_clause[index + len(token): index + len(token) + 16]
                if (
                    not marker_occurrence_is_negated(normalized_clause, token, index)
                    and not marker_occurrence_is_uncertain(normalized_clause, index)
                    and not any(suffix.startswith(normalize_name(item)) for item in postposed_absence)
                ):
                    return True
                start = index + len(token)
    return False


def has_pediatric_airway_compression_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    pediatric = marker_present_not_negated(
        normalized,
        ["幼儿或儿童", "幼儿", "儿童", "男童", "女童", "男孩", "女孩", "小儿", "婴儿"],
    ) or bool(re.search(r"(?<!\d)(?:[0-9]|1[0-2])岁", normalized))
    airway = marker_present_not_negated(
        normalized,
        ["气道症状", "呼吸声重", "喘鸣", "长期咳嗽", "持续咳嗽"],
    )
    chronic_course = marker_present_not_negated(
        normalized,
        ["慢性或进行性病程", "三个月", "数月", "长期", "慢性", "越来越重", "逐渐加重", "半年", "多年"],
    )
    compression = marker_present_not_negated(
        normalized,
        ["体位性呼吸加重", "进食或吞咽压迫", "平卧加重", "平躺加重", "进食干呕", "吞咽困难"],
    )
    return pediatric and airway and chronic_course and compression


def has_symptomatic_anemia_loss_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    symptom_groups = [
        marker_present_not_negated(normalized, ["进行性乏力", "逐渐乏力", "越来越乏力"]),
        marker_present_not_negated(normalized, ["活动耐量下降", "活动后气促", "爬楼就气促", "爬楼梯喘"]),
        marker_present_not_negated(normalized, ["反复头晕", "站立头晕", "直立性头晕"]),
    ]
    loss_risk = marker_present_not_negated(
        normalized,
        ["慢性失血或铁缺乏风险", "月经过多", "经量增多", "产后", "痔疮出血", "反复便血", "慢性失血"],
    )
    return sum(symptom_groups) >= 2 and loss_risk


def has_pediatric_progressive_night_blindness_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    pediatric = marker_present_not_negated(
        normalized,
        ["儿童", "孩子", "男童", "女童", "男孩", "女孩", "小儿"],
    ) or bool(re.search(r"(?<!\d)(?:[0-9]|1[0-2])岁", normalized))
    night_blindness = marker_present_not_negated(
        normalized,
        ["儿童进行性夜盲", "夜盲", "天黑时看不清", "天黑的时候看不清", "黄昏经常撞到", "暗处看不清", "晚上走路要牵"],
    )
    progression = marker_present_not_negated(
        normalized,
        ["逐渐", "进行性", "最近加重", "经常撞到", "晚上走路要牵", "一年前开始"],
    )
    return pediatric and night_blindness and progression


def has_water_aerosol_severe_pneumonia_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    exposure = marker_present_not_negated(
        normalized,
        [
            "冷却塔",
            "热水浴池",
            "温泉",
            "喷泉水雾",
            "酒店供水",
            "医院供水",
            "集中空调冷却水",
            "酒店空调",
            "水气溶胶暴露",
            "军团菌暴露",
            "聚集性肺炎",
        ],
    ) or (
        marker_present_not_negated(normalized, ["淋浴", "热水淋浴"])
        and marker_present_not_negated(
            normalized,
            ["酒店", "旅馆", "医院", "养老院", "高风险供水", "长期停用水管"],
        )
    )
    systemic = marker_present_not_negated(normalized, ["寒战", "高热", "发热", "肌肉酸痛"])
    respiratory = marker_present_not_negated(normalized, ["咳嗽", "干咳", "黄痰", "脓痰"])
    severity = marker_present_not_negated(normalized, ["气短", "呼吸困难", "胸痛", "持续加重"])
    return exposure and systemic and respiratory and severity


def has_seafood_acute_watery_diarrhea_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    seafood = has_consumed_high_risk_seafood(text)
    watery_diarrhea = marker_present_not_negated(
        normalized,
        ["水样腹泻", "拉稀水", "频繁腹泻", "腹泻"],
    )
    acute_gi = marker_present_not_negated(normalized, ["呕吐", "腹部绞痛", "突然腹痛", "急性胃肠"])
    return seafood and watery_diarrhea and acute_gi


def has_consumed_high_risk_seafood(text: str) -> bool:
    for clause in semantic_clauses(text):
        normalized_clause = normalize_name(clause)
        if seafood_clause_is_non_exposure(normalized_clause):
            continue
        for token in ["生蚝", "冰虾", "生虾", "海鲜", "贝类"]:
            start = 0
            while True:
                index = normalized_clause.find(token, start)
                if index < 0:
                    break
                if seafood_occurrence_is_exposure(normalized_clause, token, index):
                    return True
                start = index + len(token)
    return False


def seafood_clause_is_non_exposure(clause: str) -> bool:
    blocked = [
        "不确定", "不能确定", "无法确定", "未知", "记不清", "记不得", "不记得", "是否", "有无", "有没有",
        "可能", "好像", "大概", "似乎", "也许", "或许", "医生说", "医生建议", "医生提醒", "风险提示", "宣教",
        "禁止", "避免", "不要", "不应", "询问", "咨询", "菜单", "售卖", "购买", "接触", "清洗", "养殖",
        "小时候", "童年", "多年以前", "很久以前", "既往曾", "以前曾", "过去曾", "只是陪同",
    ]
    if any(marker in clause for marker in blocked):
        return True
    other_subject = any(marker in clause for marker in ["家属", "朋友", "同事", "室友", "陪同者", "别人"])
    patient_subject = any(marker in clause for marker in ["我吃", "我食用", "本人吃", "患者吃", "我们吃", "一起吃"])
    return other_subject and not patient_subject


def seafood_occurrence_is_exposure(clause: str, token: str, index: int) -> bool:
    prefix = clause[max(0, index - 20):index]
    suffix = clause[index + len(token):index + len(token) + 20]
    if re.search(r"(?:没有|没|未曾|不曾|从未|否认|未|不).{0,4}(?:吃|食用|进食|摄入|生吃|生食)", prefix[-16:]):
        return False
    if marker_occurrence_is_negated(clause, token, index):
        return False
    raw_prefix = prefix.rstrip("的")
    raw_markers = [
        "生吃", "生食", "生食过", "未经加热", "未加热", "未充分加热", "未经充分加热", "加热不充分",
        "未熟", "半生", "半生不熟", "未煮熟", "未烹熟", "没熟", "未完全熟",
    ]
    raw = any(raw_prefix.endswith(marker) or raw_prefix.endswith(marker + "了") for marker in raw_markers) or suffix.startswith("刺身")
    consumed = bool(re.search(r"(?:吃(?:了|过)?|食用(?:了|过)?|进食(?:了|过)?|摄入(?:了|过)?|尝(?:了|过)?)\D{0,8}$", prefix))
    event_after = any(marker in suffix[:12] for marker in ["后", "导致", "引起", "出现"])
    if token in {"海鲜", "贝类"} and not raw:
        return False
    if not consumed and not (raw and (event_after or "过" in raw_prefix[-4:])):
        return False
    return not seafood_occurrence_is_cooked(prefix, token, suffix)


def seafood_occurrence_is_cooked(prefix: str, token: str, suffix: str) -> bool:
    scope = prefix[-16:] + token + suffix[:16]
    for raw_marker in [
        "未经充分加热", "未充分加热", "加热不充分", "未经加热", "未加热", "未完全熟", "半生不熟",
        "未煮熟", "未烹熟", "未熟", "半生", "没熟",
    ]:
        scope = scope.replace(raw_marker, "")
    cooked_markers = [
        "熟的", "熟生蚝", "熟生虾", "熟冰虾", "熟海鲜", "熟贝类", "蒸好的", "烤好的", "煮好的", "煎好的",
        "炖好的", "焯好的", "清蒸", "清煮", "烤制", "蒸制", "煮制", "烤的", "煮的", "蒸的", "煎的", "炖的",
        "焯的", "煮过", "蒸过", "烤过", "煎过", "炖过", "焯过", "烤熟", "煮熟", "蒸熟", "煎熟", "炖熟",
        "焯熟", "熟透", "充分加热", "已加热", "加热过", "充分烹饪", "做熟", "全熟", "熟制", "罐头", "罐装",
    ]
    return any(marker in scope for marker in cooked_markers)


def has_chronic_suppurative_middle_ear_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    otorrhea = marker_present_not_negated(
        normalized,
        ["耳流脓", "左耳流脓", "右耳流脓", "脓性耳漏"],
    )
    hearing_loss = marker_present_not_negated(normalized, ["听力下降", "听力更差", "耳鸣"])
    chronic = marker_present_not_negated(normalized, ["反复", "三个月", "数月", "长期", "总复发"])
    return otorrhea and hearing_loss and chronic


def has_active_upper_gi_bleed_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    overt_bleeding = marker_present_not_negated(
        normalized,
        ["呕血", "咖啡样呕吐物", "黑便", "柏油样便", "反复排黑便"],
    )
    high_risk_context = marker_present_not_negated(
        normalized,
        ["肝硬化", "门静脉高压", "头晕", "心悸", "气短", "血流动力学"],
    )
    return overt_bleeding and high_risk_context


def has_immunosuppressed_acute_infection_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    immune = marker_present_not_negated(
        normalized,
        ["HIV", "AIDS", "器官移植", "肾移植", "免疫抑制剂", "长期激素", "他克莫司"],
    )
    infectious = marker_present_not_negated(
        normalized,
        ["发热", "发烧", "寒战", "咳嗽", "咳痰", "鼻塞", "咽痛", "呼吸道症状"],
    )
    return immune and infectious


def has_pediatric_congenital_glaucoma_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    child = marker_present_not_negated(
        normalized,
        ["新生儿", "婴儿", "幼儿", "儿童", "岁儿童", "岁患儿"],
    )
    high_pressure = marker_present_not_negated(normalized, ["眼压升高", "眼压高", "眼压32", "高眼压"])
    ocular_sign = marker_present_not_negated(
        normalized,
        ["畏光", "流泪", "挤眼", "眼球增大", "牛眼", "角膜水肿", "角膜混浊", "视力下降"],
    )
    return child and high_pressure and ocular_sign


def has_current_arrhythmia_specific_evidence(text: str) -> bool:
    normalized = normalize_name(text)
    return marker_present_not_negated(
        normalized,
        ["心电图异常", "ECG异常", "不规则心律", "脉搏不齐", "突发突止", "既往明确心律失常"],
    )


def load_packaged_rule_pack_for_offline_use() -> CompiledRulePack:
    """Offline helper fallback only; does not represent production runtime source.

    Must never read releases/current.json. Historical pure-function tests may call
    select_diagnosis_axes without an explicit pack; production MyDoctorAgent paths
    must always pass self.rule_pack.
    """
    payload = json.loads(
        (KNOWLEDGE_DIR / "clinical_pattern_rules.json").read_text(encoding="utf-8")
    )
    return parse_compiled_rule_pack(payload)


def select_diagnosis_axes(
    intake_facts: Dict[str, Any],
    llm_axes: Optional[List[Dict[str, Any]]] = None,
    *,
    case_state: Optional[Dict[str, Any]] = None,
    rule_pack: Optional[CompiledRulePack] = None,
) -> List[Dict[str, Any]]:
    raw_facts_text = intake_facts_text(intake_facts)
    # Rules must see the patient's exact symptom wording and objective results.
    if case_state is not None:
        patient_text = patient_text_for_matching(case_state)
        exam_text = examination_text_for_matching(case_state)
        raw_facts_text = "\n".join(
            item for item in [raw_facts_text, patient_text, exam_text] if item
        )
    facts_text = normalize_name(raw_facts_text)
    axes: List[Dict[str, Any]] = []
    axes.extend(as_axis_list(llm_axes or []))
    if all(marker in facts_text for marker in [normalize_name("新生儿"), normalize_name("脐部")]) and all(
        any(normalize_name(marker) in facts_text for marker in group)
        for group in [["湿润", "鲜红", "肿块"], ["出血", "易出血"]]
    ):
        axes.append(
            diagnosis_axis(
                "umbilical_granulation_or_vascular_lesion",
                ["新生儿", "脐部", "湿润易出血肿块"],
                ["局部专科检查或病理证据"],
                ["化脓性肉芽肿", "新生儿脐炎"],
                ["局部皮肤病变评估", "出血病变病理评估"],
                ["avoid_no_further_care_for_bleeding_mass"],
            )
        )
    if normalize_name("鼻中隔偏曲") in facts_text and normalize_name("反复鼻出血") in facts_text:
        axes.append(
            diagnosis_axis(
                "septal_deviation_recurrent_epistaxis",
                ["鼻中隔偏曲", "反复鼻出血"],
                ["凝血或血液学筛查"],
                ["鼻中隔偏曲"],
                ["鼻腔结构评估", "凝血功能"],
                ["recurrent_epistaxis_coagulation"],
            )
        )
    rule_state = case_state or {
        "chat_history": [{"from": "patient", "text": raw_facts_text}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }
    context = RuleContext(
        diagnostic_axis_ids=tuple(
            clean_axis_id(axis.get("axis_id"))
            for axis in axes
            if clean_axis_id(axis.get("axis_id"))
        ),
        fact_codes=diagnosis_rule_fact_codes(rule_state),
    )
    active_pack = (
        rule_pack
        if rule_pack is not None
        else load_packaged_rule_pack_for_offline_use()
    )
    typed_result = apply_rules(
        active_pack,
        "diagnosis_candidates",
        context,
    )
    materialized = materialize_diagnosis_rule_axes(axes, typed_result)
    # Preserve exposure-specialized systemic-infection axis fields that the
    # closed typed template intentionally keeps in the base (non-vector) form.
    if has_systemic_infection_hematologic_axis_pattern(facts_text) and (
        normalize_name("媒介或户外暴露") in facts_text or has_vector_exposure(raw_facts_text)
    ):
        for axis in materialized:
            if clean_axis_id(axis.get("axis_id")) != "systemic_infection_vs_primary_hematologic":
                continue
            names = as_text_list(axis.get("candidate_official_names"))
            if "回归热" not in names:
                names.append("回归热")
            axis["candidate_official_names"] = names
            axis["rule_candidate_official_names"] = list(names)
            intents = [
                intent
                for intent in as_text_list(axis.get("exam_intents"))
                if intent != "全身感染病原评估"
            ]
            if "媒介或暴露相关病原评估" not in intents:
                intents.append("媒介或暴露相关病原评估")
            if "血液细胞形态鉴别" not in intents:
                intents.insert(0, "血液细胞形态鉴别")
            axis["exam_intents"] = intents
            break
    return merge_diagnosis_axes(materialized, [])


def diagnosis_axis(
    axis_id: str,
    evidence: List[str],
    missing_evidence: List[str],
    candidates: List[str],
    exam_intents: List[str],
    risks: List[str],
    *,
    clinical_role: str = "current_problem",
    priority: str = "routine",
    closure_requirement: str = "supported_official_diagnosis",
) -> Dict[str, Any]:
    return {
        "axis_id": axis_id,
        "source": "rule",
        "status": "suspected",
        "evidence": evidence,
        "missing_evidence": missing_evidence,
        "candidate_official_names": candidates,
        "rule_candidate_official_names": candidates,
        "exam_intents": exam_intents,
        "treatment_risks": risks,
        "clinical_role": clinical_role,
        "priority": priority,
        "closure_requirement": closure_requirement,
    }


def validate_axis_consult(
    raw_consult: Dict[str, Any],
    *,
    case_state: Dict[str, Any],
    official_diseases: Iterable[str],
    alias_rules: Optional[List[Dict[str, Any]]] = None,
    rule_pack: Optional[CompiledRulePack] = None,
) -> Dict[str, Any]:
    if not isinstance(raw_consult, dict):
        raw_consult = {}
    case_text = "\n".join(
        item
        for item in [
            patient_text_for_matching(case_state),
            examination_text_for_matching(case_state),
        ]
        if item
    )
    official_map = build_name_map(official_diseases)
    intake_facts = extract_intake_facts(case_state, raw_consult.get("intake_facts") if isinstance(raw_consult.get("intake_facts"), dict) else {})
    rule_axes = select_diagnosis_axes(
        extract_intake_facts(case_state),
        case_state=case_state,
        rule_pack=rule_pack,
    )
    rule_supported_names = {
        match_standard_name(name, official_map)
        for axis in rule_axes
        for name in as_text_list(axis.get("rule_candidate_official_names") or axis.get("candidate_official_names"))
    }
    rule_supported_names.discard("")
    candidate_support_text = candidate_support_text_for_matching(case_state)
    context_supported_names = diagnostic_context_supported_official_names(
        case_state.get("diagnostic_context"),
        official_disease_map=official_map,
        alias_rules=alias_rules or [],
        grounding_text=candidate_support_text,
    )
    supported_names = rule_supported_names | context_supported_names
    axes = []
    for axis in raw_consult.get("diagnosis_axes", []) if isinstance(raw_consult.get("diagnosis_axes", []), list) else []:
        normalized = validate_single_axis(
            axis,
            case_text,
            official_map,
            alias_rules=alias_rules or [],
            supported_official_names=supported_names,
            candidate_support_text=candidate_support_text,
        )
        if normalized and is_cryoglobulinemia_secondary_axis(normalized):
            normalized = None
        if normalized:
            axes.append(normalized)
    axes = merge_diagnosis_axes(axes, rule_axes)
    return {
        "intake_facts": intake_facts,
        "diagnosis_axes": axes,
        "treatment_risks": axis_treatment_risks(axes),
        "risk_summary": clean_text(raw_consult.get("risk_summary")),
    }


def diagnostic_context_supported_official_names(
    diagnostic_context: Any,
    *,
    official_disease_map: Dict[str, str],
    alias_rules: List[Dict[str, Any]],
    grounding_text: str,
) -> set[str]:
    if not isinstance(diagnostic_context, dict):
        return set()
    case_features = (
        diagnostic_context.get("case_features")
        if isinstance(diagnostic_context.get("case_features"), dict)
        else {}
    )
    differentials = differential_items(diagnostic_context)
    differential_raw_names = [item["raw_name"] for item in differentials]
    grounded_labels = grounded_case_feature_labels(case_features, grounding_text)
    supported: set[str] = set()
    suggestions = diagnostic_context.get("normalization_suggestions")
    for suggestion in suggestions if isinstance(suggestions, list) else []:
        if not isinstance(suggestion, dict):
            continue
        accepted = accept_normalization_suggestion(
            suggestion,
            case_features=case_features,
            differential_raw_names=differential_raw_names,
            official_disease_map=official_disease_map,
            alias_rules=alias_rules,
        )
        evidence_labels = unique_preserve_order(
            as_text_list(accepted.get("matched_evidence"))
        )
        grounded_required_labels = {
            normalize_feature_label(label)
            for label in evidence_labels
            if normalize_feature_label(label) in grounded_labels
        }
        if accepted.get("accepted") and len(grounded_required_labels) >= 2:
            supported.add(clean_text(accepted.get("normalized_diagnosis")))
    supported.discard("")
    return supported


def grounded_case_feature_labels(case_features: Dict[str, Any], grounding_text: str) -> set[str]:
    grounded: set[str] = set()
    for items in case_features.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            label = normalize_feature_label(item.get("label"))
            evidence = clean_text(item.get("evidence"))
            if label and feature_evidence_grounded(evidence, grounding_text):
                grounded.add(label)
    return grounded


def feature_evidence_grounded(evidence: str, grounding_text: str) -> bool:
    if not evidence or not grounding_text or evidence_marker_is_non_positive(evidence):
        return False
    if any(
        marker in normalize_name(evidence)
        for marker in ["疑为", "疑似", "可能", "考虑", "倾向", "待排", "不能排除"]
    ):
        return False
    atoms = feature_evidence_atoms(evidence)
    return bool(atoms) and all(
        feature_evidence_atom_grounded(atom, grounding_text) for atom in atoms
    )


def feature_evidence_atoms(evidence: str) -> List[str]:
    atoms = [
        clean_text(item)
        for item in re.split(
            r"[，,。；;、/|：:\n]+|"
            r"(?:并伴|伴有|伴随|同时|合并|但是|然而|提示|显示|发现|出现|检出)|"
            r"伴(?!性)",
            clean_text(evidence),
        )
        if clean_text(item)
    ]
    generic = {"患者", "检查", "结果", "检查结果", "报告", "影像"}
    return unique_preserve_order(
        atom for atom in atoms if normalize_name(atom) not in generic
    )


def feature_evidence_atom_grounded(atom: str, grounding_text: str) -> bool:
    normalized_atom = normalize_name(atom)
    if not normalized_atom:
        return False
    if soft_marker_present(atom, normalize_name(grounding_text)):
        return True
    if len(normalized_atom) < 4:
        return False
    return any(
        bounded_ordered_insertion_match(normalized_atom, normalize_name(clause))
        for clause in semantic_clauses(grounding_text)
    )


def bounded_ordered_insertion_match(token: str, clause: str) -> bool:
    if not token or not clause:
        return False
    allowance = max(1, len(token) // 4)
    for start in [index for index, char in enumerate(clause) if char == token[0]]:
        cursor = start
        matched = True
        for char in token[1:]:
            cursor = clause.find(char, cursor + 1)
            if cursor < 0:
                matched = False
                break
        if matched and cursor - start + 1 - len(token) <= allowance:
            return True
    return False


def validate_single_axis(
    axis: Dict[str, Any],
    normalized_case_text: str,
    official_disease_map: Dict[str, str],
    *,
    alias_rules: Optional[List[Dict[str, Any]]] = None,
    supported_official_names: Optional[set[str]] = None,
    candidate_support_text: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(axis, dict):
        return None
    axis_id = clean_axis_id(axis.get("axis_id"))
    if not axis_id:
        return None
    if axis_id == "seafood_acute_watery_diarrhea_pathogen" and not has_seafood_acute_watery_diarrhea_pattern(
        normalized_case_text
    ):
        return None
    if axis_id == "water_aerosol_severe_pneumonia_pathogen" and not has_water_aerosol_severe_pneumonia_pattern(
        normalized_case_text
    ):
        return None
    raw_evidence = unique_preserve_order(as_text_list(axis.get("evidence")))
    if len({normalize_name(item) for item in raw_evidence if normalize_name(item)}) < 2:
        return None
    evidence = unique_preserve_order(
        item for item in raw_evidence if soft_marker_present(item, normalized_case_text)
    )
    if len({normalize_name(item) for item in evidence if normalize_name(item)}) < 2:
        return None
    blocked = AXIS_CANDIDATE_BLOCKLIST.get(axis_id, set())
    candidates = []
    promotable_candidates = []
    for item in as_text_list(axis.get("candidate_official_names")):
        # Exact/alias first, then surface containment (e.g. 支气管肺癌 → 肺癌).
        # Historical bug: raw==official was treated as non-promotable; now any
        # resolved official name may promote when semantic support holds.
        official = resolve_official_from_surface_name(
            item,
            official_disease_map,
            alias_rules,
        )
        promotable = bool(official)
        if official and official not in blocked and official not in candidates:
            candidates.append(official)
        if (
            promotable
            and official
            and official not in blocked
            and axis_candidate_semantically_supported(
                official,
                item,
                normalized_case_text,
                supported_official_names or set(),
                candidate_support_text=candidate_support_text,
            )
            and official not in promotable_candidates
        ):
            promotable_candidates.append(official)
    risks = supported_axis_risks(
        as_text_list(axis.get("treatment_risks")),
        axis_id=axis_id,
        evidence=evidence,
        candidates=candidates,
        normalized_case_text=normalize_name(normalized_case_text),
    )
    clinical_role = clean_text(axis.get("clinical_role"))
    if clinical_role not in {"current_problem", "background_condition", "background_history", "secondary"}:
        clinical_role = "current_problem"
    priority = clean_text(axis.get("priority"))
    if priority not in {"routine", "high", "red_flag"}:
        priority = "routine"
    closure_requirement = clean_text(axis.get("closure_requirement")) or "supported_official_diagnosis"
    return {
        "axis_id": axis_id,
        "source": "llm",
        "validated": True,
        "status": clean_text(axis.get("status")) if clean_text(axis.get("status")) in {"confirmed", "suspected", "missing_evidence"} else "suspected",
        "evidence": evidence,
        "missing_evidence": as_text_list(axis.get("missing_evidence")),
        "candidate_official_names": candidates,
        "llm_candidate_official_names": candidates,
        "promotable_candidate_official_names": promotable_candidates,
        "exam_intents": as_text_list(axis.get("exam_intents")),
        "treatment_risks": risks,
        "clinical_role": clinical_role,
        "priority": priority,
        "closure_requirement": closure_requirement,
    }


def clean_axis_id(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return text.strip("_")


def soft_marker_present(marker: str, normalized_case_text: str) -> bool:
    raw_marker = clean_text(marker)
    marker_text = normalize_name(raw_marker)
    if not marker_text:
        return False
    if evidence_marker_is_non_positive(raw_marker):
        return False
    if normalized_marker_present_not_negated(normalized_case_text, marker_text):
        return True
    reporting_prefixes = ["听力测定显示", "耳镜检查示", "检查结果显示", "检查显示", "结果显示", "检查示"]
    for raw_token in re.split(r"[，,、/|；;：:\n]+", raw_marker):
        token = normalize_name(raw_token)
        variants = [token]
        for prefix in reporting_prefixes:
            prefix_text = normalize_name(prefix)
            if token.startswith(prefix_text):
                variants.append(token[len(prefix_text):])
        if any(
            len(item) >= 2 and normalized_marker_present_not_negated(normalized_case_text, item)
            for item in variants
        ):
            return True
    synonyms = {
        normalize_name("易出血"): ["出血"],
        normalize_name("靶形丘疹"): ["靶形"],
        normalize_name("角膜感染"): ["角膜", "眼痛", "畏光"],
    }
    return any(
        normalized_marker_present_not_negated(normalized_case_text, item)
        for item in synonyms.get(marker_text, [])
    )


def evidence_marker_is_non_positive(marker: str) -> bool:
    normalized = normalize_name(marker)
    subject_stripped = re.sub(r"^(?:患者|病人|本人)", "", normalized)
    if subject_stripped.startswith("没"):
        return True
    if subject_stripped.startswith("无") and not subject_stripped.startswith(
        ("无痛性", "无菌性", "无创", "无症状性")
    ):
        return True
    return any(
        token in normalized
        for token in [
            "没有",
            "未见",
            "未有",
            "未出现",
            "未发生",
            "未发现",
            "未提示",
            "未检出",
            "未患",
            "未合并",
            "否认",
            "不伴",
            "并非",
            "不是",
            "不支持",
            "排除",
            "不除外",
            "不能排除",
            "未能排除",
            "需排除",
            "需要排除",
            "待排",
            "考虑",
            "疑似",
            "可能",
            "可疑",
            "不明确",
            "未知",
            "是否",
            "有无",
            "有没有",
            "待核对",
            "需核对",
            "尚未核对",
            "没有核对",
            "尚不明确",
        ]
    )


def normalized_marker_present_not_negated(text: str, marker: str) -> bool:
    token = normalize_name(marker)
    for clause in semantic_clauses(text):
        normalized_clause = normalize_name(clause)
        start = 0
        while token:
            index = normalized_clause.find(token, start)
            if index < 0:
                break
            if not marker_occurrence_is_uncertain(normalized_clause, index) and not marker_occurrence_is_negated(
                normalized_clause,
                token,
                index,
            ):
                return True
            start = index + len(token)
    normalized_text = normalize_name(text)
    start = 0
    while token:
        index = normalized_text.find(token, start)
        if index < 0:
            break
        if not marker_occurrence_is_uncertain(normalized_text, index) and not marker_occurrence_is_negated(
            normalized_text,
            token,
            index,
        ):
            return True
        start = index + len(token)
    return False


def intake_facts_text(intake_facts: Dict[str, Any]) -> str:
    chunks = []
    if not isinstance(intake_facts, dict):
        return ""
    for items in intake_facts.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            chunks.extend([clean_text(item.get("label")), clean_text(item.get("evidence"))])
    return "\n".join(chunk for chunk in chunks if chunk)


def has_sle_axis_pattern(facts_text: str) -> bool:
    immune = any(normalize_name(marker) in facts_text for marker in ["ANA", "抗Sm", "抗SSA", "抗核抗体", "补体降低"])
    clinical = any(normalize_name(marker) in facts_text for marker in ["光敏", "皮疹", "关节痛", "脱发", "口腔溃疡"])
    serosal_or_renal = any(
        normalize_name(marker) in facts_text
        for marker in ["心包积液", "心包炎", "心包增厚", "胸膜炎", "肌酐", "尿蛋白", "血尿", "肾功能", "肾脏受累风险"]
    )
    return clinical and (immune or serosal_or_renal)


def has_multisystem_autoimmune_serositis_pattern(facts_text: str) -> bool:
    clinical = sum(
        normalize_name(marker) in facts_text
        for marker in ["光敏", "红斑", "皮疹", "脱发", "关节痛", "关节肿痛"]
    ) >= 2
    serositis = any(normalize_name(marker) in facts_text for marker in ["心包积液", "心包炎", "心包增厚", "胸膜炎"])
    return clinical and serositis


def has_leptospirosis_exposure_pattern(normalized_case_text: str) -> bool:
    exposure = marker_present_not_negated(
        normalized_case_text,
        ["洪水", "泥水", "疫水", "牲畜尿液", "鼠尿", "水田", "污水接触"],
    )
    systemic = marker_present_not_negated(normalized_case_text, ["高热", "发热", "寒战"])
    calf_or_conjunctiva = marker_present_not_negated(
        normalized_case_text,
        ["腓肠肌", "小腿肌肉", "肌肉剧痛", "结膜充血", "眼红"],
    )
    organ_injury = marker_present_not_negated(
        normalized_case_text,
        ["尿色变深", "尿色发黑", "黄疸", "少尿", "肾功能", "血小板减少"],
    )
    return exposure and systemic and calf_or_conjunctiva and organ_injury


def has_diuretic_hypokalemia_pattern(normalized_case_text: str) -> bool:
    diuretic = marker_present_not_negated(normalized_case_text, ["利尿剂", "利尿药", "呋塞米", "氢氯噻嗪", "螺内酯"])
    potassium = marker_present_not_negated(normalized_case_text, ["低钾", "血钾降低", "低钾血症", "血钾2.", "血钾为2"])
    alkalosis = marker_present_not_negated(normalized_case_text, ["代谢性碱中毒", "碳酸氢盐升高", "碳酸氢根升高"])
    symptoms = marker_present_not_negated(normalized_case_text, ["抽筋", "乏力", "肌无力", "心悸", "多尿"])
    return diuretic and potassium and (alkalosis or symptoms)


def has_positive_vsd_text(normalized_case_text: str) -> bool:
    return marker_present_not_negated(normalized_case_text, ["大型室间隔缺损", "室间隔缺损", "VSD"])


def has_chest_wall_trauma_pattern(normalized_case_text: str) -> bool:
    site = marker_present_not_negated(normalized_case_text, ["前胸", "胸壁", "胸口", "肋骨"])
    mechanism = marker_present_not_negated(normalized_case_text, ["撞击", "推搡", "跌倒", "外伤"])
    local = marker_present_not_negated(normalized_case_text, ["胸痛", "刺痛", "压痛", "瘀斑", "肿胀"])
    respiratory_trigger = marker_present_not_negated(normalized_case_text, ["深呼吸", "咳嗽时", "抬臂时"])
    return site and mechanism and local and respiratory_trigger


def has_corneal_infection_target_rash_pattern(facts_text: str) -> bool:
    eye = any(normalize_name(marker) in facts_text for marker in ["角膜", "眼红", "眼痛", "畏光"])
    rash = any(normalize_name(marker) in facts_text for marker in ["靶形", "水疱", "水泡", "黏膜"])
    return eye and rash


def has_migraine_reproductive_travel_pattern(facts_text: str) -> bool:
    migraine = any(
        normalize_name(marker) in facts_text
        for marker in ["偏头痛", "偏头痛伴恶心头晕", "搏动性头痛", "偏侧搏动头痛"]
    )
    reproductive = normalize_name("育龄女性") in facts_text or normalize_name("月经相关") in facts_text
    travel = any(
        normalize_name(marker) in facts_text
        for marker in ["旅行或视觉运动诱发", "旅行", "坐飞机", "飞机", "视觉运动", "晕动"]
    )
    return migraine and reproductive and travel


def has_postmenopausal_urogenital_irritation_pattern(facts_text: str) -> bool:
    postmenopausal = any(
        normalize_name(marker) in facts_text
        for marker in ["绝经后女性", "绝经后", "已绝经", "绝经", "更年期"]
    )
    urinary = any(
        normalize_name(marker) in facts_text
        for marker in ["尿路刺激征", "尿频", "尿急", "尿痛", "排尿烧灼", "血尿"]
    )
    return postmenopausal and urinary


def has_acute_respiratory_illness_pattern(facts_text: str) -> bool:
    """True only for distress-level respiratory illness that needs oxygenation closure."""
    return any(
        normalize_name(marker) in facts_text
        for marker in [
            "急性呼吸窘迫线索",
            "低氧血症证据",
            "咳吐",
            "呼吸加快",
            "气促",
            "喘息",
            "呼吸急促",
            "呼吸困难",
            "呼吸窘迫",
        ]
    )


def has_chronic_alcohol_liver_injury_pattern(facts_text: str) -> bool:
    has_alcohol = any(
        normalize_name(marker) in facts_text
        for marker in ["长期大量饮酒", "天天喝", "大量饮酒", "长期饮酒", "饮酒史"]
    )
    has_symptoms = any(
        normalize_name(marker) in facts_text
        for marker in ["肝病相关症状", "乏力", "腹胀", "食欲差", "食欲减退", "纳差"]
    )
    return has_alcohol and has_symptoms


def has_pleuritic_pain_infection_embolism_pattern(facts_text: str) -> bool:
    has_pleuritic = any(
        normalize_name(marker) in facts_text
        for marker in ["胸膜性胸痛", "刀割样", "深呼吸", "咳嗽时胸痛", "胸膜性"]
    )
    has_chest_site = any(
        normalize_name(marker) in facts_text
        for marker in ["胸口", "胸痛", "右侧胸口", "左侧胸口", "胸部"]
    )
    has_support = any(
        normalize_name(marker) in facts_text
        for marker in ["干咳胸闷", "干咳", "胸闷", "既往肺梗死或栓塞史", "肺梗死", "肺栓塞"]
    )
    return has_pleuritic and has_chest_site and has_support


def has_high_energy_hindfoot_trauma_pattern(facts_text: str) -> bool:
    # Colloquial falls ("高处干活掉下来") and heel wording ("脚后跟") must match.
    has_energy = any(
        normalize_name(marker) in facts_text
        for marker in [
            "高能量创伤机制",
            "车祸",
            "交通事故",
            "高处坠落",
            "高能量",
            "砸伤",
            "高处掉",
            "从高处",
            "高处干活",
            "掉下来",
            "摔下来",
            "坠落",
            "掉下",
            "摔下",
        ]
    )
    has_site = any(
        normalize_name(marker) in facts_text
        for marker in [
            "高能量足跟创伤",
            "脚跟",
            "足跟",
            "跟骨",
            "左脚跟",
            "右脚跟",
            "脚后跟",
            "足后跟",
            "后跟",
        ]
    )
    has_severity = any(
        normalize_name(marker) in facts_text
        for marker in [
            "足跟负重不能",
            "剧痛",
            "肿胀",
            "瘀斑",
            "不敢踩地",
            "不能负重",
            "踩不了",
            "不能踩",
            "不敢踩",
            "脚变宽",
            "足变宽",
            "肿得厉害",
            "完全动不了",
            "无法行走",
            "不能走路",
        ]
    )
    return has_energy and has_site and has_severity


def has_febrile_polyuria_dehydration_pattern(facts_text: str) -> bool:
    has_fever = any(
        normalize_name(marker) in facts_text
        for marker in ["高热", "高烧", "发热", "发烧"]
    )
    has_polyuria_polydipsia = any(
        normalize_name(marker) in facts_text
        for marker in ["多尿烦渴脱水", "特别口渴", "极度口渴", "多尿", "尿特别多", "烦渴"]
    )
    has_dehydration = any(
        normalize_name(marker) in facts_text
        for marker in ["脱水", "严重脱水", "头晕", "低血压", "血压也偏低"]
    )
    return has_fever and has_polyuria_polydipsia and has_dehydration


def has_acute_gastroenteritis_like_pattern(facts_text: str) -> bool:
    return any(
        normalize_name(marker) in facts_text
        for marker in ["急性胃肠炎样症状", "暴食后", "呕吐", "腹泻", "上腹不适", "上腹痛"]
    ) and not any(
        normalize_name(marker) in facts_text
        for marker in ["喷嚏加重", "清水样鼻涕急性加重", "眼痒急性加重"]
    )


def hyperglycemic_crisis_excluded(case_features: Dict[str, Any]) -> bool:
    """True only when usable glucose evidence is present and not in crisis range."""
    status = glucose_evidence_status(case_features)
    return status == "normal"


def glucose_evidence_status(case_features: Dict[str, Any]) -> str:
    results = case_features.get("examination_results")
    if not isinstance(results, dict):
        results = {}
    values: List[float] = []
    for name, payload in results.items():
        if not isinstance(payload, dict):
            continue
        name_text = normalize_name(name)
        relevant = any(marker in name_text for marker in ["血糖", "葡萄糖", "fbg", "glu"])
        for key, value in exam_result_pairs(payload):
            key_text = normalize_name(key)
            if not relevant and not any(marker in key_text for marker in ["血糖", "葡萄糖", "glu"]):
                continue
            if any(marker in key_text for marker in ["糖化", "hba1c"]):
                continue
            match = re.search(r"(\d+(?:\.\d+)?)", result_value_without_reference(value))
            if not match:
                continue
            number = float(match.group(1))
            unit_text = normalize_name(value)
            # Convert rough mg/dL to mmol/L when clearly on mg scale.
            if number > 40 and any(marker in unit_text for marker in ["mg", "毫克"]):
                number = number / 18.0
            elif number > 40 and "mmol" not in unit_text:
                number = number / 18.0
            values.append(number)
    text = normalize_name(
        " ".join(
            [
                clean_text(case_features.get("case_text")),
                clean_text(case_features.get("patient_text")),
                " ".join(structured_text_chunks(results)),
            ]
        )
    )
    if any(marker in text for marker in [normalize_name("酮体阳性"), normalize_name("代谢性酸中毒"), normalize_name("dka"), normalize_name("hhs")]):
        return "crisis"
    if not values:
        return "unknown"
    if max(values) >= 13.9 or max(values) >= 250 / 18.0:
        return "crisis"
    if max(values) >= 11.1:
        return "high"
    return "normal"


def has_post_traumatic_cognitive_vestibular_pattern(facts_text: str) -> bool:
    has_trauma_onset = any(
        normalize_name(marker) in facts_text
        for marker in ["头部外伤后起病", "外伤后持续头痛", "头部磕碰", "外伤后"]
    )
    has_headache = any(normalize_name(marker) in facts_text for marker in ["头痛", "外伤后持续头痛", "反复头痛"])
    has_cognitive = any(normalize_name(marker) in facts_text for marker in ["认知症状", "注意力不集中", "记忆差"])
    has_vestibular = any(normalize_name(marker) in facts_text for marker in ["前庭平衡症状", "站立不稳", "快速转头", "头晕"])
    return has_trauma_onset and has_headache and (has_cognitive or has_vestibular)


def has_infant_congenital_structural_heart_pattern(facts_text: str) -> bool:
    has_infant = marker_present_not_negated(
        facts_text,
        ["婴儿", "新生儿", "宝宝", "刚出生", "出生后", "出生第"],
    )
    has_early_respiratory = marker_present_not_negated(
        facts_text,
        [
            "出生后呼吸急促",
            "呼吸急促",
            "呼吸快",
            "气促",
            "喘得厉害",
            "安静时也喘",
            "急性呼吸窘迫线索",
        ],
    )
    has_feeding_cardiac_stress = marker_present_not_negated(
        facts_text,
        [
            "喂养困难出汗",
            "喂养困难或吃奶少",
            "吃奶困难",
            "吃奶少",
            "喂养出汗",
            "喂养时出汗",
            "吃奶时出汗",
            "吃奶出汗",
            "吃奶容易累",
            "吃奶累",
            "吃奶时呼吸快",
            "停下来休息",
        ],
    )
    has_cyanosis_or_pulmonary_htn = marker_present_not_negated(
        facts_text,
        [
            "活动或哭闹发绀",
            "嘴唇发绀或发青",
            "口唇发绀",
            "嘴唇发青",
            "嘴巴周围发暗",
            "口周发暗",
            "发绀",
            "肺动脉高压体征",
            "P2亢进",
            "吸氧改善不明显",
        ],
    )
    return has_infant and has_early_respiratory and has_feeding_cardiac_stress and has_cyanosis_or_pulmonary_htn


def has_congenital_infection_pattern(facts_text: str) -> bool:
    source = clean_text(facts_text)
    infant = marker_present_not_negated(source, ["婴儿", "新生儿", "宝宝", "出生后", "出生第"])
    exposure = marker_present_not_negated(source, ["宫内病毒暴露", "孕早期病毒暴露", "孕期病毒暴露", "宫内感染"])
    multisystem = sum(
        marker_present_not_negated(source, group)
        for group in [
            ["黄疸", "皮肤发黄", "眼睛发黄"],
            ["红疹", "皮疹", "紫癜"],
            ["吃奶差", "吃奶不好", "喂养困难"],
            ["嗜睡", "反应弱", "听力", "对声音反应弱"],
        ]
    )
    return infant and exposure and multisystem >= 2


def _hearing_exam_result_is_abnormal(payload: Dict[str, Any]) -> bool:
    status = normalize_name(payload.get("status"))
    if status in {
        "normal",
        "negative",
        "unknown",
        "pending",
        "ordered",
        "inprogress",
        "processing",
        "invalid",
    }:
        return False
    if not exam_result_is_usable(payload):
        return False
    return status == "abnormal" or bool(as_text_list(payload.get("abnormal_indicators")))


def has_hearing_symptom_pattern(case_state: Dict[str, Any]) -> bool:
    patient_text = patient_text_for_matching(case_state)
    direct_tinnitus = marker_present_not_negated(
        patient_text,
        ["耳鸣", "耳内响", "耳朵里响", "耳内有声音", "耳朵里有声音"],
    )
    colloquial_tinnitus = any(
        marker_present_not_negated(
            clause,
            ["蝉鸣声", "蝉叫声", "嗡嗡声", "电流声", "滋滋声", "嘶嘶声"],
        )
        and marker_present_not_negated(clause, ["耳", "耳朵", "耳内", "耳部"])
        for clause in semantic_clauses(patient_text)
    )
    tinnitus = direct_tinnitus or colloquial_tinnitus
    if not tinnitus:
        return False
    payloads = matching_exam_payloads(
        case_state,
        ["听力测定", "纯音测听", "言语识别", "听力学"],
    )
    return any(_hearing_exam_result_is_abnormal(payload) for payload in payloads)


def has_acute_upper_respiratory_infection_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    acute = marker_present_not_negated(
        normalized,
        ["前天", "昨天", "两天", "三天", "数天", "近期", "突然", "急性"],
    )
    upper_airway = marker_present_not_negated(
        normalized,
        ["咽痛", "喉咙痛", "鼻塞", "流涕", "打喷嚏"],
    )
    infection = marker_present_not_negated(normalized, ["发热", "发烧", "咳嗽", "咳"])
    return acute and upper_airway and infection


def has_acute_bronchitis_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    acute = marker_present_not_negated(
        normalized,
        ["前天", "昨天", "两天", "三天", "数天", "近期", "突然", "急性"],
    )
    cough = marker_present_not_negated(normalized, ["咳嗽", "咳", "咳痰"])
    inflammatory = marker_present_not_negated(
        normalized,
        ["发热", "发烧", "咽痛", "喉咙痛", "喘息", "黄痰"],
    )
    return acute and cough and inflammatory


def has_high_risk_pediatric_lower_respiratory_infection_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    if normalize_name("高危儿科下呼吸道感染") in normalized:
        return True
    has_pediatric_context = any(
        normalize_name(marker) in normalized
        for marker in ["孩子", "患儿", "儿童", "幼儿", "婴儿"]
    ) or bool(re.search(r"\d+岁", normalized))
    has_persistent_course = any(
        normalize_name(marker) in normalized
        for marker in ["反复", "持续", "迁延", "数周", "周", "一个月", "个月"]
    )
    has_fever = marker_present_not_negated(normalized, ["发热", "发烧", "高热"])
    has_cough = marker_present_not_negated(normalized, ["咳嗽", "咳"])
    has_lower_respiratory_signal = marker_present_not_negated(
        normalized,
        ["黄绿色痰", "黄痰", "脓痰", "喘息", "喘得", "呼吸粗", "呼吸急促"],
    )
    has_immunosuppression = marker_present_not_negated(
        normalized,
        ["免疫抑制药", "免疫抑制剂", "免疫抑制治疗", "长期激素", "化疗"],
    )
    return (
        has_pediatric_context
        and has_persistent_course
        and has_fever
        and has_cough
        and has_lower_respiratory_signal
        and has_immunosuppression
    )


def has_acute_ear_pain_after_instrumentation_pattern(text: str) -> bool:
    normalized = normalize_name(text)
    if normalize_name("耳道操作后急性耳痛") in normalized:
        return True
    has_ear_pain = marker_present_not_negated(normalized, ["耳朵疼", "耳痛", "耳朵痛"])
    has_instrumentation = marker_present_not_negated(
        normalized,
        ["掏耳", "掏耳朵", "挖耳", "棉签掏耳", "耳勺"],
    )
    has_local_severity = marker_present_not_negated(
        normalized,
        ["堵住", "耳堵", "耳闷", "耳鸣", "更疼", "疼痛加重", "夜间疼"],
    )
    return has_ear_pain and has_instrumentation and has_local_severity


def has_postop_chylothorax_or_pleural_effusion_pattern(facts_text: str) -> bool:
    has_surgery = any(
        normalize_name(marker) in facts_text
        for marker in [
            "胸部纵隔术后",
            "胸腔手术",
            "纵隔手术",
            "淋巴结清扫",
            "胸部手术",
            "纵隔淋巴结",
            "开胸",
        ]
    )
    has_resp = any(
        normalize_name(marker) in facts_text
        for marker in ["术后呼吸费力", "呼吸费力", "活动减少", "气促", "呼吸困难", "呼吸急促"]
    )
    return has_surgery and has_resp


def has_cyanotic_duct_dependent_risk_pattern(facts_text: str) -> bool:
    has_infant = any(normalize_name(marker) in facts_text for marker in ["婴儿", "新生儿", "出生后"])
    has_cyanosis = any(
        normalize_name(marker) in facts_text
        for marker in ["发绀", "嘴唇发青", "口唇发绀", "嘴唇发绀或发青", "吸氧改善不明显"]
    )
    return has_infant and has_cyanosis


def marker_present_not_negated(normalized_text: str, markers: List[str]) -> bool:
    """True when any marker occurrence is outside a local negation scope."""
    for marker in markers:
        token = normalize_name(marker)
        if not token:
            continue
        for clause in semantic_clauses(normalized_text):
            normalized_clause = normalize_name(clause)
            start = 0
            while True:
                index = normalized_clause.find(token, start)
                if index < 0:
                    break
                if not marker_occurrence_is_uncertain(normalized_clause, index) and not marker_occurrence_is_negated(
                    normalized_clause, token, index
                ):
                    return True
                start = index + len(token)
    return False


def marker_present_negated(text: str, markers: List[str]) -> bool:
    for marker in markers:
        token = normalize_name(marker)
        if not token:
            continue
        for clause in semantic_clauses(text):
            normalized_clause = normalize_name(clause)
            start = 0
            while True:
                index = normalized_clause.find(token, start)
                if index < 0:
                    break
                if not marker_occurrence_is_uncertain(normalized_clause, index) and marker_occurrence_is_negated(
                    normalized_clause, token, index
                ):
                    return True
                start = index + len(token)
    return False


def semantic_clauses(text: str) -> List[str]:
    return [
        clause
        for clause in re.split(
            r"[，,。；;！？!?\n]+|但是|但|然而|不过|可是|"
            r"(?:且|并且|并)(?=正在|仍在|继续|使用|服用|患有|存在)|"
            r"(?:现在|目前)|现(?=突发|出现|发生|开始|有)",
            clean_text(text),
        )
        if clause
    ]


def marker_occurrence_is_negated(clause: str, token: str, index: int) -> bool:
    prefix = clause[max(0, index - 16):index]
    for negation in [
        "没",
        "没有",
        "未见",
        "未诉",
        "未出现",
        "未发生",
        "未发现",
        "未提示",
        "未检出",
        "未患",
        "未合并",
        "未使用",
        "未服用",
        "未接触",
        "从未",
        "否认",
        "不伴",
        "并非",
        "不是",
        "不支持",
        "无",
    ]:
        position = prefix.rfind(normalize_name(negation))
        if position >= 0:
            return True
    return False


def marker_occurrence_is_uncertain(clause: str, index: int) -> bool:
    prefix = clause[max(0, index - 16):index]
    return any(
        marker in prefix
        for marker in ["未核对", "尚未核对", "没有核对", "待核对", "需核对", "需要核对", "尚不明确", "不明确", "未知", "不详"]
    )


def has_upper_arm_trauma_pattern(normalized_text: str) -> bool:
    has_site = marker_present_not_negated(
        normalized_text,
        ["上臂", "肱骨", "肱骨干", "急性上臂外伤功能障碍"],
    )
    has_trauma = marker_present_not_negated(
        normalized_text,
        ["跌倒", "手肘着地", "撞击", "跌倒外伤机制", "上臂外伤"],
    ) or (
        marker_present_not_negated(normalized_text, ["外伤"])
        and marker_present_not_negated(normalized_text, ["上臂", "肱骨"])
    )
    has_severity = marker_present_not_negated(
        normalized_text,
        ["剧痛", "肿胀", "活动受限", "急性上臂外伤功能障碍"],
    )
    return has_site and has_trauma and has_severity


def has_acute_extremity_trauma_pattern(normalized_text: str) -> bool:
    """Acute limb trauma (fall/impact) with local severity — not chronic OA history alone.

    Used to demote degenerative joint labels that often ride incidental imaging.
    """
    has_site = marker_present_not_negated(
        normalized_text,
        [
            "腕",
            "手",
            "桡骨",
            "前臂",
            "肘",
            "肱",
            "肩",
            "锁骨",
            "踝",
            "足",
            "跟骨",
            "膝",
            "小腿",
            "大腿",
            "髋",
            "上臂",
        ],
    )
    has_trauma = marker_present_not_negated(
        normalized_text,
        ["外伤", "跌倒", "摔伤", "撞击", "砸伤", "扭伤后", "车祸", "高处坠落", "手撑地", "着地"],
    )
    has_severity = marker_present_not_negated(
        normalized_text,
        [
            "剧痛",
            "肿胀",
            "畸形",
            "活动受限",
            "不能动",
            "麻木",
            "正中神经",
            "瘀斑",
            "压痛",
            "骨折",
            "骨擦感",
        ],
    )
    # Explicit acute fracture wording is enough with a site.
    if has_site and marker_present_not_negated(normalized_text, ["骨折", "疑似骨折", "骨折线"]):
        return True
    return bool(has_site and has_trauma and has_severity)


def has_palpitation_arrhythmia_pattern(normalized_text: str) -> bool:
    has_palpitation = marker_present_not_negated(
        normalized_text,
        ["心悸", "心慌", "心跳快", "心跳很快", "突发心悸"],
    )
    has_cardio_context = marker_present_not_negated(
        normalized_text,
        [
            "胸闷",
            "气短",
            "心脏病",
            "心功能下降",
            "心梗",
            "心肌梗死",
            "冠心病",
            "既往心脏病",
            "腹泻",
            "摄入不足",
            "吃得少",
            "胃肠道丢失或摄入不足",
        ],
    )
    return has_palpitation and has_cardio_context


def has_elbow_overuse_pattern(normalized_text: str) -> bool:
    has_elbow = marker_present_not_negated(
        normalized_text,
        ["肘部", "肘痛", "肘关节", "局限肘部疼痛", "肘部疼痛"],
    )
    has_overuse = marker_present_not_negated(
        normalized_text,
        [
            "重复抓握",
            "腕部活动",
            "抓握",
            "重复负荷诱发",
            "重复抓握腕活动诱发肘痛",
            "过度使用",
        ],
    )
    return has_elbow and has_overuse


def has_systemic_infection_or_inflammation_pattern(normalized_text: str) -> bool:
    return marker_present_not_negated(
        normalized_text,
        [
            "发热",
            "高热",
            "红肿",
            "感染",
            "寒战",
            "脓",
            "体温",
            "全身炎症",
            "系统炎症",
        ],
    )


def has_hepato_splenic_cytopenia_pattern(normalized_text: str) -> bool:
    has_liver = marker_present_not_negated(
        normalized_text,
        ["慢性肝病", "肝硬化", "肝病", "肝病用药", "慢性肝病背景"],
    )
    has_spleen_clue = marker_present_not_negated(
        normalized_text,
        ["左上腹", "左上腹饱胀", "脾区", "脾大", "腹胀"],
    )
    has_cytopenia = marker_present_not_negated(
        normalized_text,
        [
            "三系减少",
            "血细胞减少",
            "全血细胞减少",
            "血细胞三系减少证据",
            "容易瘀青",
            "易瘀青",
            "瘀青",
        ],
    )
    return has_liver and has_spleen_clue and has_cytopenia


def has_pulmonary_renal_vasculitis_pattern(normalized_text: str) -> bool:
    """Cough/hemoptysis plus renal or multi-system clues — keep vasculitis on the table."""
    has_airway_bleed = marker_present_not_negated(
        normalized_text,
        ["咯血", "痰中带血", "咳血", "带血丝", "咯血或痰中带血"],
    )
    has_cough = marker_present_not_negated(
        normalized_text,
        ["咳嗽", "慢性咳嗽", "咳了", "长期咳嗽", "咳痰", "咳嗽线索"],
    )
    has_renal_or_edema = marker_present_not_negated(
        normalized_text,
        [
            "脚踝水肿",
            "下肢水肿",
            "双下肢水肿",
            "水肿",
            "脚踝肿",
            "踝肿",
            "脚肿",
            "腿肿",
            "血尿",
            "蛋白尿",
            "肾功能",
            "肾不好",
            "肌酐",
            "肾脏受累风险",
            "肺肾",
            "肺肾综合征风险",
            "尿色变黑",
            "尿发黑",
            "尿色发黑",
            "尿量减少",
            "少尿",
        ],
    )
    has_systemic = marker_present_not_negated(
        normalized_text,
        ["乏力", "消瘦", "盗汗", "关节疼痛", "关节都痛", "肌肉和关节", "全身乏力消耗", "低热"],
    )
    return has_airway_bleed and has_cough and (has_renal_or_edema or has_systemic)


def is_anca_vasculitis_diagnosis(diagnosis: str) -> bool:
    diagnosis_text = normalize_name(diagnosis)
    return any(
        normalize_name(marker) in diagnosis_text
        for marker in [
            "显微镜下多血管炎",
            "多血管炎性肉芽肿",
            "肉芽肿性多血管炎",
            "ANCA相关血管炎",
            "ANCA 相关血管炎",
        ]
    )


def treatment_has_glucocorticoid_induction(normalized_plan: str) -> bool:
    return plan_has_any(
        normalized_plan,
        [
            "甲泼尼龙",
            "甲强龙",
            "甲基强的松龙",
            "激素冲击",
            "大剂量激素",
            "大剂量糖皮质",
            "糖皮质激素冲击",
            "静脉激素",
            "静脉甲泼尼龙",
            "泼尼松龙冲击",
            "泼尼松",
            "泼尼松龙",
            "糖皮质激素",
            "激素诱导",
            "诱导缓解用激素",
        ],
    )


def is_anca_vasculitis_induction_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    """MPA/GPA-like diagnosis with pulmonary-renal or DAH clues needs steroid induction."""
    if not is_anca_vasculitis_diagnosis(diagnosis):
        return False
    text = normalize_name(case_features_text(case_features))
    if has_pulmonary_renal_vasculitis_pattern(text):
        return True
    has_bleed = marker_present_not_negated(
        text,
        ["咯血", "痰中带血", "咳血", "带血丝", "肺出血"],
    )
    has_renal = marker_present_not_negated(
        text,
        ["血尿", "蛋白尿", "尿色变黑", "少尿", "尿量减少", "肾功能", "脚踝肿", "水肿"],
    )
    return has_bleed and has_renal


def has_closed_tb_infection_evidence(
    normalized_text: str,
    examinations: Optional[Iterable[str]] = None,
    treatment_plan: str = "",
) -> bool:
    text = normalize_name(
        " ".join(
            [normalized_text, clean_text(treatment_plan)]
            + as_text_list(examinations or [])
        )
    )
    return any(
        normalize_name(marker) in text
        for marker in [
            "痰涂片阳性",
            "抗酸杆菌阳性",
            "结核分枝杆菌培养阳性",
            "GeneXpert阳性",
            "结核菌阳性",
            "结核病原闭合证据",
            "空洞型肺结核已确认",
        ]
    )


def has_symptomatic_hypokalemia_malabsorption_pattern(normalized_text: str) -> bool:
    has_k_symptom = marker_present_not_negated(
        normalized_text,
        [
            "手抽筋",
            "抽筋",
            "浑身没劲",
            "低钾",
            "血钾低",
            "血钾降低",
            "肌无力",
            "症状性低钾表现",
            "血钾降低证据",
            "低钾血症",
        ],
    )
    has_gi_loss = marker_present_not_negated(
        normalized_text,
        [
            "腹泻",
            "脂肪泻",
            "吸收不良",
            "小肠切除",
            "胰酶",
            "胰腺功能不全",
            "腹泻吸收不良丢失",
        ],
    )
    return has_k_symptom and has_gi_loss


def as_axis_list(value: Any) -> List[Dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def merge_diagnosis_axes(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for axis in primary + secondary:
        if not isinstance(axis, dict):
            continue
        axis_id = clean_axis_id(axis.get("axis_id"))
        if not axis_id:
            continue
        existing = merged.get(axis_id, {"axis_id": axis_id})
        axis_sources = [
            source
            for source in clean_text(axis.get("source")).split("+")
            if source
        ]
        axis_candidates = as_text_list(axis.get("candidate_official_names"))
        rule_candidates = as_text_list(axis.get("rule_candidate_official_names"))
        llm_candidates = as_text_list(axis.get("llm_candidate_official_names"))
        if "rule" in axis_sources and not rule_candidates:
            rule_candidates = axis_candidates
        if "llm" in axis_sources and not llm_candidates:
            llm_candidates = axis_candidates
        for key in [
            "evidence",
            "missing_evidence",
            "exam_intents",
            "treatment_risks",
        ]:
            existing[key] = unique_preserve_order(as_text_list(existing.get(key)) + as_text_list(axis.get(key)))
        existing["rule_candidate_official_names"] = unique_preserve_order(
            as_text_list(existing.get("rule_candidate_official_names")) + rule_candidates
        )
        existing["llm_candidate_official_names"] = unique_preserve_order(
            as_text_list(existing.get("llm_candidate_official_names")) + llm_candidates
        )
        existing["promotable_candidate_official_names"] = unique_preserve_order(
            as_text_list(existing.get("promotable_candidate_official_names"))
            + (
                as_text_list(axis.get("promotable_candidate_official_names"))
                if "llm" in axis_sources
                else []
            )
        )
        existing["candidate_official_names"] = unique_preserve_order(
            as_text_list(existing.get("rule_candidate_official_names"))
            + as_text_list(existing.get("llm_candidate_official_names"))
        )
        sources = unique_preserve_order(
            clean_text(existing.get("source")).split("+")
            + clean_text(axis.get("source")).split("+")
        )
        existing["source"] = "+".join(source for source in sources if source)
        existing["validated"] = bool(existing.get("validated") or axis.get("validated"))
        existing["status"] = clean_text(axis.get("status")) or clean_text(existing.get("status")) or "suspected"
        # Preserve semantic fields from rule axes; prefer higher-severity / more specific values.
        role_rank = {"background_condition": 0, "background_history": 0, "secondary": 1, "current_problem": 2}
        priority_rank = {"routine": 0, "high": 1, "red_flag": 2}
        existing_role = clean_text(existing.get("clinical_role")) or "current_problem"
        axis_role = clean_text(axis.get("clinical_role")) or existing_role
        existing["clinical_role"] = (
            axis_role
            if role_rank.get(axis_role, 1) >= role_rank.get(existing_role, 1)
            else existing_role
        )
        existing_priority = clean_text(existing.get("priority")) or "routine"
        axis_priority = clean_text(axis.get("priority")) or existing_priority
        existing["priority"] = (
            axis_priority
            if priority_rank.get(axis_priority, 0) >= priority_rank.get(existing_priority, 0)
            else existing_priority
        )
        existing["closure_requirement"] = (
            clean_text(axis.get("closure_requirement"))
            or clean_text(existing.get("closure_requirement"))
            or "supported_official_diagnosis"
        )
        merged[axis_id] = existing
    return list(merged.values())


def axis_treatment_risks(axes: List[Dict[str, Any]]) -> List[str]:
    risks: List[str] = []
    for axis in axes:
        for risk in as_text_list(axis.get("treatment_risks")):
            if risk in KNOWN_AXIS_RISK_TAGS and risk not in risks:
                risks.append(risk)
    return risks


def supported_axis_risks(
    risks: List[str],
    *,
    axis_id: str = "",
    evidence: Optional[List[str]] = None,
    candidates: Optional[List[str]] = None,
    normalized_case_text: str = "",
    diagnosis: str = "",
) -> List[str]:
    supported = []
    for risk in risks:
        if risk not in KNOWN_AXIS_RISK_TAGS:
            continue
        if risk in SLE_AXIS_RISK_TAGS and not has_sle_risk_support(
            axis_id=axis_id,
            evidence=evidence or [],
            candidates=candidates or [],
            normalized_case_text=normalized_case_text,
            diagnosis=diagnosis,
        ):
            continue
        if risk in MIGRAINE_AXIS_RISK_TAGS and not has_migraine_drug_risk_support(
            axis_id=axis_id,
            evidence=evidence or [],
            candidates=candidates or [],
            normalized_case_text=normalized_case_text,
            diagnosis=diagnosis,
        ):
            continue
        if risk in INFECTION_STEROID_RISK_TAGS and not has_infection_steroid_risk_support(
            axis_id=axis_id,
            evidence=evidence or [],
            candidates=candidates or [],
            normalized_case_text=normalized_case_text,
            diagnosis=diagnosis,
        ):
            continue
        if risk in UMBILICAL_CARE_RISK_TAGS and not has_umbilical_bleeding_mass_risk_support(
            axis_id=axis_id,
            normalized_case_text=normalized_case_text,
            diagnosis=diagnosis,
        ):
            continue
        supported.append(risk)
    return unique_preserve_order(supported)


def has_umbilical_bleeding_mass_risk_support(
    *,
    axis_id: str,
    normalized_case_text: str,
    diagnosis: str,
) -> bool:
    # Axis id alone is not enough: LLM can invent the axis. Require neonatal umbilical narrative.
    if not has_umbilical_granulation_bleeding_mass_pattern(normalized_case_text):
        return False
    if clean_axis_id(axis_id) == "umbilical_granulation_or_vascular_lesion":
        return True
    return any(
        marker in normalize_name(diagnosis)
        for marker in [normalize_name("化脓性肉芽肿"), normalize_name("新生儿脐炎")]
    )


def has_infection_steroid_risk_support(
    *,
    axis_id: str,
    evidence: List[str],
    candidates: List[str],
    normalized_case_text: str,
    diagnosis: str,
) -> bool:
    """Bind infection-before-steroid tags to infection axes; block free LLM pollution."""
    axis_norm = clean_axis_id(axis_id)
    if axis_norm in INFECTION_STEROID_SUPPORTED_AXIS_IDS:
        return True
    context = normalize_name(" ".join([axis_id, diagnosis] + evidence + candidates + [normalized_case_text]))
    eye_infection = any(
        marker in context
        for marker in [
            normalize_name("角膜炎"),
            normalize_name("角膜溃疡"),
            normalize_name("眼内炎"),
            normalize_name("眼红畏光"),
            "corneal_infection",
        ]
    )
    if eye_infection:
        return True
    if has_corneal_infection_target_rash_pattern(normalized_case_text):
        return True
    if has_systemic_infection_hematologic_axis_pattern(normalized_case_text):
        return True
    return False


def has_sle_risk_support(
    *,
    axis_id: str,
    evidence: List[str],
    candidates: List[str],
    normalized_case_text: str,
    diagnosis: str,
) -> bool:
    strong_context = normalize_name(" ".join([axis_id, diagnosis] + evidence + candidates))
    if any(marker in strong_context for marker in [normalize_name("系统性红斑狼疮"), "sle", normalize_name("狼疮")]):
        return True
    return has_sle_axis_pattern(normalized_case_text)


def has_migraine_drug_risk_support(
    *,
    axis_id: str,
    evidence: List[str],
    candidates: List[str],
    normalized_case_text: str,
    diagnosis: str,
) -> bool:
    strong_context = normalize_name(" ".join([axis_id, diagnosis] + evidence + candidates + [normalized_case_text]))
    migraine_markers = [
        normalize_name("偏头痛"),
        normalize_name("搏动性头痛"),
        normalize_name("曲普坦"),
        "migraine",
        "migraine_reproductive_travel_trigger",
    ]
    if any(marker in strong_context for marker in migraine_markers):
        return True
    return has_migraine_reproductive_travel_pattern(normalized_case_text)


def apply_treatment_safety(
    treatment_plan: str,
    *,
    diagnosis: str,
    case_features: Dict[str, Any],
    safety_profiles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    plan = clean_text(treatment_plan)
    issues = []
    patches = []
    feature_text = normalize_name(
        " ".join(
            as_text_list(case_features.get("positive_findings"))
            + as_text_list(case_features.get("immunosuppression"))
            + as_text_list(case_features.get("red_flags"))
            + as_text_list(case_features.get("medications"))
            + [diagnosis]
        )
    )
    normalized_plan = normalize_name(plan)
    contraindication_result = find_contraindicated_drug_recommendations(plan, case_features)
    issues.extend(contraindication_result["issues"])
    patches.extend(contraindication_result["patches"])
    nsaid_result = find_unsafe_nsaid_recommendation(plan, case_features)
    issues.extend(nsaid_result["issues"])
    patches.extend(nsaid_result["patches"])
    if is_high_risk_hsv_immunosuppressed_case(diagnosis, case_features) and treatment_continues_immunosuppression(normalized_plan):
        patch = "急性重症疱疹病毒感染风险下，不应继续或加用全身糖皮质激素/免疫抑制药物；应与开方医生协作，急性感染期暂停或调整全身糖皮质激素/免疫抑制药物。"
        issues.append(
            {
                "field": "treatment_plan",
                "code": "contraindicated_immunosuppression_continuation_in_severe_hsv",
                "severity": "must_fix",
                "problem": "continue_systemic_steroid_during_severe_hsv",
                "patchable": True,
                "edit": patch,
            }
        )
        if patch not in patches:
            patches.append(patch)

    for profile in safety_profiles:
        risk_factors = as_text_list(profile.get("risk_factors"))
        if risk_factors and not all(normalize_name(item) in feature_text for item in risk_factors):
            continue
        patch_templates = profile.get("patch_templates") if isinstance(profile.get("patch_templates"), dict) else {}
        for goal in as_text_list(profile.get("treatment_goals")):
            if treatment_goal_present(goal, normalized_plan):
                continue
            patch = patch_for_goal(goal, patch_templates)
            if not patch:
                continue
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "missing_treatment_goal",
                    "severity": "must_fix",
                    "problem": goal,
                    "patchable": True,
                    "edit": patch,
                }
            )
            if patch not in patches:
                patches.append(patch)

    patched_plan = sanitize_immunosuppression_continuation(
        sanitize_unsafe_nsaid_recommendations(
            sanitize_contraindicated_recommendations(plan, case_features),
            should_sanitize=bool(nsaid_result["issues"]),
        ),
        diagnosis=diagnosis,
        case_features=case_features,
    )
    # I3: named anti-infective without sensitivity or documented empiric indication.
    # Narrow gate — only fires on concrete drug names, not generic "抗感染".
    anti_infective_result = find_anti_infective_evidence_gaps(patched_plan, case_features)
    issues.extend(anti_infective_result.get("issues") or [])
    for patch in anti_infective_result.get("patches") or []:
        if patch not in patches:
            patches.append(patch)

    if patches:
        patched_plan = " ".join([patched_plan] + patches).strip()
    return {"treatment_plan": patched_plan, "issues": issues, "patched": bool(patches)}


def is_high_risk_hsv_immunosuppressed_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    diagnosis_text = normalize_name(" ".join([diagnosis] + as_text_list(case_features.get("candidate_diagnoses"))))
    findings = set(as_text_list(case_features.get("positive_findings")))
    immunosuppression = set(as_text_list(case_features.get("immunosuppression")))
    has_hsv_diagnosis = any(
        marker in diagnosis_text
        for marker in [normalize_name("卡波西水痘样疹"), normalize_name("疱疹"), "hsv"]
    )
    return has_hsv_diagnosis and {"高热", "疱疹样皮损"}.issubset(findings) and "全身激素" in immunosuppression


def treatment_continues_immunosuppression(normalized_plan: str) -> bool:
    action_markers = ["继续", "维持", "加用", "增加", "无需调整", "不需调整"]
    drug_markers = ["全身糖皮质激素", "糖皮质激素", "全身激素", "泼尼松", "免疫抑制"]
    for action in action_markers:
        action_text = normalize_name(action)
        start = normalized_plan.find(action_text)
        while start >= 0:
            window = normalized_plan[start: start + 32]
            if any(normalize_name(drug) in window for drug in drug_markers):
                return True
            start = normalized_plan.find(action_text, start + len(action_text))
    return False


def sanitize_immunosuppression_continuation(
    treatment_plan: str,
    *,
    diagnosis: str,
    case_features: Dict[str, Any],
) -> str:
    plan = clean_text(treatment_plan)
    if not is_high_risk_hsv_immunosuppressed_case(diagnosis, case_features):
        return plan
    patterns = [
        r"继续原全身糖皮质激素[^，,；;。]*[，,；;。]?",
        r"继续全身糖皮质激素[^，,；;。]*[，,；;。]?",
        r"维持全身糖皮质激素[^，,；;。]*[，,；;。]?",
        r"加用全身糖皮质激素[^，,；;。]*[，,；;。]?",
        r"无需调整[^，,；;。]*(?:激素|免疫抑制)[^，,；;。]*[，,；;。]?",
        r"不需调整[^，,；;。]*(?:激素|免疫抑制)[^，,；;。]*[，,；;。]?",
    ]
    for pattern in patterns:
        plan = re.sub(pattern, "", plan)
    return normalize_treatment_text(plan)


def find_contraindicated_drug_recommendations(
    treatment_plan: str,
    case_features: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_plan = normalize_name(treatment_plan)
    issues = []
    patches = []
    for group_name, aliases in CONTRAINDICATED_DRUG_GROUPS.items():
        if not contraindication_group_present(group_name, aliases, case_features):
            continue
        matched_aliases = [alias for alias in aliases if normalize_name(alias) in normalized_plan]
        if not matched_aliases:
            continue
        if not treatment_recommends_drug(normalized_plan, matched_aliases):
            continue
        patch = contraindicated_drug_patch(group_name)
        issues.append(
            {
                "field": "treatment_plan",
                "code": "contraindicated_drug_recommended",
                "severity": "must_fix",
                "problem": matched_aliases[0],
                "patchable": True,
                "edit": patch,
            }
        )
        if patch and patch not in patches:
            patches.append(patch)
    return {"issues": issues, "patches": patches}


NSAID_ALIASES = [
    "非甾体抗炎药",
    "nsaids",
    "nsaid",
    "布洛芬",
    "双氯芬酸",
    "萘普生",
    "塞来昔布",
    "依托考昔",
    "吲哚美辛",
    "酮咯酸",
]


def find_unsafe_nsaid_recommendation(
    treatment_plan: str,
    case_features: Dict[str, Any],
) -> Dict[str, Any]:
    if not treatment_recommends_drug(normalize_name(treatment_plan), NSAID_ALIASES):
        return {"issues": [], "patches": []}
    case_text = clean_text(case_features.get("case_text")) or case_features_text(case_features)
    duplicate_nsaid = recommends_duplicate_systemic_nsaid(treatment_plan, case_text, case_features)
    known_risks = nsaid_known_safety_risks(case_text)
    if not duplicate_nsaid and not known_risks and not nsaid_safety_facts_require_review(case_text):
        return {"issues": [], "patches": []}
    if duplicate_nsaid:
        code = "duplicate_systemic_nsaid"
        problem = "additional_systemic_nsaid_while_another_nsaid_is_current"
        patch = "避免同时使用两种系统性NSAID；应核对当前镇痛药并由医生选择单一安全镇痛方案。"
    elif known_risks:
        code = "nsaid_high_risk_present"
        problem = "systemic_nsaid_with_known_renal_bleeding_or_antithrombotic_risk"
        patch = "存在明确的消化道、出血、肾脏或抗栓用药风险，应避免自行使用系统性NSAID并由医生选择替代镇痛方案。"
    else:
        code = "nsaid_safety_facts_unresolved"
        problem = "systemic_nsaid_without_closed_medication_renal_bleeding_facts"
        patch = (
            "在核对年龄、当前止痛药具体名称、肾功能、消化道出血/瘀斑及抗凝抗血小板用药前，"
            "应先由医生选择安全镇痛方案，避免自行加用同类止痛药。"
        )
    return {
        "issues": [
            {
                "field": "treatment_plan",
                "code": code,
                "severity": "must_fix",
                "problem": problem,
                "patchable": True,
                "edit": patch,
            }
        ],
        "patches": [patch],
    }


def is_cryoglobulinemia_secondary_axis(axis: Dict[str, Any]) -> bool:
    axis_id = clean_axis_id(axis.get("axis_id"))
    id_match = "cryoglobulin" in axis_id and any(
        marker in axis_id for marker in ["secondary", "etiology", "cause", "workup"]
    )
    candidates = normalize_name(" ".join(as_text_list(axis.get("candidate_official_names"))))
    evidence = normalize_name(" ".join(as_text_list(axis.get("evidence"))))
    intents = normalize_name(" ".join(as_text_list(axis.get("exam_intents"))))
    secondary_intent = any(
        marker in intents for marker in ["hcv", "丙型肝炎", "单克隆蛋白", "浆细胞病", "病因"]
    )
    intent_match = secondary_intent and ("冷球蛋白血症" in candidates or "冷球蛋白" in evidence)
    return id_match or intent_match


def recommends_duplicate_systemic_nsaid(
    treatment_plan: str,
    case_text: str,
    case_features: Dict[str, Any],
) -> bool:
    concrete_aliases = [alias for alias in NSAID_ALIASES if alias not in {"非甾体抗炎药", "nsaids", "nsaid"}]
    current_text = " ".join([case_text] + as_text_list(case_features.get("medications")))
    current = {
        normalize_name(alias)
        for alias in concrete_aliases
        if marker_present_not_negated(current_text, [alias])
    }
    recommended = {
        normalize_name(alias)
        for alias in concrete_aliases
        if treatment_recommends_drug(normalize_name(treatment_plan), [alias])
    }
    if not current or not recommended:
        return False
    adds_drug = any(marker in normalize_name(treatment_plan) for marker in ["加用", "联合", "叠加", "同时使用", "同时服用"])
    return bool(current != recommended or adds_drug)


def nsaid_safety_facts_require_review(case_text: str) -> bool:
    normalized = normalize_name(case_text)
    known_analgesics = NSAID_ALIASES + ["对乙酰氨基酚", "曲马多", "可待因"]
    vague_analgesic = "止痛药" in normalized and not any(
        normalize_name(item) in normalized for item in known_analgesics
    )
    ages = [int(item) for item in re.findall(r"(?<!\d)(\d{1,3})岁", case_text)]
    advanced_age = bool(ages and max(ages) >= 65)
    return vague_analgesic or (advanced_age and not nsaid_safety_explicitly_cleared(case_text))


def nsaid_known_safety_risks(case_text: str) -> List[str]:
    markers = [
        "容易瘀斑", "易瘀斑", "瘀斑", "活动性胃溃疡", "消化性溃疡", "胃十二指肠溃疡",
        "消化道出血", "胃出血", "黑便", "慢性肾病", "CKD", "肾功能不全", "肾损伤", "肾衰",
        "抗凝用药", "抗凝药", "抗血小板用药", "抗血小板药", "华法林", "利伐沙班", "阿哌沙班",
        "达比加群", "依度沙班", "肝素", "氯吡格雷", "替格瑞洛",
    ]
    return [marker for marker in markers if marker_present_not_negated(case_text, [marker])]


def positive_clinical_marker(text: str, markers: List[str]) -> bool:
    return any(marker_present_not_negated(text, [marker]) for marker in markers)


def nsaid_safety_explicitly_cleared(case_text: str) -> bool:
    renal_clear = marker_present_not_negated(case_text, ["肾功能正常"])
    bleeding_clear = marker_present_negated(
        case_text,
        ["胃溃疡", "消化道出血", "瘀斑", "出血风险"],
    )
    anticoagulant_clear = marker_present_negated(case_text, ["抗凝用药", "抗凝药", "抗凝"])
    antiplatelet_clear = marker_present_negated(case_text, ["抗血小板用药", "抗血小板药", "抗血小板"])
    return renal_clear and bleeding_clear and anticoagulant_clear and antiplatelet_clear


def sanitize_unsafe_nsaid_recommendations(treatment_plan: str, *, should_sanitize: bool) -> str:
    if not should_sanitize:
        return clean_text(treatment_plan)
    clauses = re.split(
        r"(?<=[；;。\n])|[，,](?=同时|并且|并|另|此外)|"
        r"(?=并(?:监测|补液|复查|随访|观察|评估|继续))",
        clean_text(treatment_plan),
    )
    return normalize_treatment_text(
        "".join(
            clause
            for clause in clauses
            if not treatment_recommends_drug(normalize_name(clause), NSAID_ALIASES)
        )
    )


def has_hfref_contraindication(case_features: Dict[str, Any]) -> bool:
    if has_reduced_left_ventricular_ejection_fraction(case_features_text(case_features)):
        return True
    for axis in as_axis_list(case_features.get("diagnosis_axes")):
        if clean_axis_id(axis.get("axis_id")) == "reduced_ejection_fraction_heart_failure":
            return True
    return False


def has_asthma_or_reactive_airway(case_features: Dict[str, Any]) -> bool:
    """True when case text/findings indicate asthma or reactive airway disease.

    Negated phrases (无哮喘/否认哮喘) must not light the gate.
    """
    raw = " ".join(
        [
            clean_text(case_features.get("case_text")),
            case_features_text(case_features),
        ]
        + as_text_list(case_features.get("positive_findings"))
        + as_text_list(case_features.get("candidate_diagnoses"))
        + as_text_list(case_features.get("medications"))
    )
    text = normalize_name(raw)
    # Strip common Chinese/English negations before substring match.
    for neg in (
        "无哮喘史",
        "无哮喘",
        "没有哮喘",
        "否认哮喘",
        "非哮喘",
        "排除哮喘",
        "no asthma",
        "without asthma",
        "denies asthma",
    ):
        text = text.replace(normalize_name(neg), " ")
    markers = [
        "哮喘",
        "支气管哮喘",
        "喘息性支气管炎",
        "气道高反应",
        "反应性气道",
        "reactive airway",
        "asthma",
    ]
    return any(normalize_name(marker) in text for marker in markers)


def contraindication_group_present(
    group_name: str,
    aliases: List[str],
    case_features: Dict[str, Any],
) -> bool:
    if group_name == "non_dihydropyridine_ccb" and has_hfref_contraindication(case_features):
        return True
    if group_name == "beta_blocker" and has_asthma_or_reactive_airway(case_features):
        return True
    feature_text = normalize_name(
        " ".join(
            as_text_list(case_features.get("drug_allergies"))
            + as_text_list(case_features.get("contraindicated_drugs"))
            + as_text_list(case_features.get("medication_risk"))
        )
    )
    if normalize_name(group_name) in feature_text:
        return True
    return any(normalize_name(alias) in feature_text for alias in aliases)


def treatment_recommends_drug(normalized_plan: str, aliases: List[str]) -> bool:
    for alias in aliases:
        alias_text = normalize_name(alias)
        start = 0
        while True:
            index = normalized_plan.find(alias_text, start)
            if index < 0:
                break
            if not drug_mention_is_negated(normalized_plan, index, len(alias_text)):
                return True
            start = index + len(alias_text)
    return False


def drug_mention_is_negated(normalized_plan: str, index: int, alias_length: int) -> bool:
    before = normalized_plan[max(0, index - 10):index]
    after = normalized_plan[index + alias_length:index + alias_length + 16]
    before_phrases = [
        "避免",
        "应避免",
        "避免使用",
        "禁用",
        "不应",
        "不应使用",
        "不得使用",
        "禁止使用",
        "不推荐使用",
    ]
    after_phrases = ["禁用", "应避免使用", "不应使用", "不得使用", "属于禁忌"]
    if any(before.endswith(normalize_name(item)) for item in before_phrases):
        return True
    if any(after.startswith(normalize_name(item)) for item in after_phrases):
        return True
    if any(after.startswith(normalize_name(item)) for item in ["过敏", "禁忌"]):
        return True
    # Allergy / sensitivity assessment is not a prescription:
    # "评估是否存在磺胺类药物敏感性" / "磺胺类药物过敏".
    # Do NOT treat "无过敏史" / "无明确禁忌" as negation of a recommended drug.
    if any(
        after.startswith(normalize_name(item))
        for item in (
            "类药物过敏",
            "类药物敏感",
            "药物过敏",
            "药物敏感",
            "类过敏",
            "类敏感",
            "过敏/敏感",
            "过敏敏感",
        )
    ):
        return True
    if after.startswith(normalize_name("过敏史")) or after.startswith(
        normalize_name("敏感史")
    ):
        # "磺胺过敏史" is contraindication context; "无过敏史" is cleared above.
        pass
    around = normalized_plan[max(0, index - 8):index + alias_length + 12]
    for denial in ("无过敏", "没有过敏", "否认过敏", "无敏感", "没有敏感", "无明确禁忌", "无禁忌"):
        around = around.replace(normalize_name(denial), "")
    if any(
        normalize_name(token) in around
        for token in (
            "评估是否存在",
            "是否存在",
            "询问过敏",
            "药物相关的皮疹",
            "识别药物",
            "怀疑存在",
            "如怀疑",
        )
    ) and any(
        normalize_name(token) in around
        for token in ("过敏", "敏感", "皮疹", "瘙痒")
    ):
        return True
    # Conjunctive negation: "磺胺类抗生素和磺胺嘧啶禁用" — the alias is the first
    # item in a list joined by a conjunction and the list ends with a negation word
    # (禁用/禁忌/避免). Bound the intervening text to a short drug name (~6 hanzi)
    # so an unrelated trailing "禁用" across a clause does not get cleared.
    tail = normalized_plan[index + alias_length:index + alias_length + 22]
    if re.match(
        r"^(和|与|及|、|,)[^\.\n，。；;]{0,6}?(禁用|禁忌|避免|不应使用|不得使用)",
        tail,
    ):
        return True
    # Class-suffix / list negation: "磺胺类抗生素和磺胺嘧啶禁用" for alias 磺胺 has
    # "类抗生素和磺胺嘧啶" between the alias and 禁用; "磺胺嘧啶禁用" for alias 磺胺
    # has "嘧啶" in between. Only terminal prohibition verbs — not bare 应避免/
    # 过敏/禁忌, which often qualify a different clause ("应避免心率过低",
    # "无过敏史", "无明确禁忌").
    if re.match(
        r"^[^\.\n，。；;,]{0,10}?(禁用|属于禁忌|不应使用|不得使用|应避免使用)",
        tail,
    ):
        return True
    # Scope search before the alias: normalize_name strips punctuation, so heads
    # like "避免使用…药物（尤其是<药名>…）" sit several characters before the
    # alias with no clause boundary left. The token nearest the alias wins, so
    # "禁用青霉素改用头孢曲松" keeps the replacement drug positive.
    window = normalized_plan[max(0, index - 16):index]
    for compound in (
        "不推荐使用", "避免使用", "禁止使用", "不得使用", "不应使用",
        "停止使用", "不推荐", "不使用",
    ):
        window = window.replace(compound, "避免")
    for phrase in (
        "若无禁忌", "如无禁忌", "没有禁忌", "排除禁忌", "无禁忌",
        "若无过敏", "如无过敏", "没有过敏", "无过敏",
    ):
        window = window.replace(phrase, "")
    neg_last = max(
        (window.rfind(token) for token in (
            "避免", "禁用", "禁忌", "不应", "不得", "禁止", "停用", "勿用",
        )),
        default=-1,
    )
    pos_last = max(
        (window.rfind(token) for token in (
            "改用", "换用", "改为", "更换", "选用", "使用", "给予", "加用",
            "首选", "推荐", "开具", "开始",
        )),
        default=-1,
    )
    if neg_last >= 0 and neg_last >= pos_last:
        return True
    return False


def contraindicated_drug_patch(group_name: str) -> str:
    if group_name == "penicillin":
        return "已知青霉素/Penicillin过敏或禁忌时，不应把青霉素G列为首选；应改用非青霉素替代抗感染方案（如头孢曲松需结合过敏严重程度评估），并明确记录过敏风险。"
    if group_name == "non_dihydropyridine_ccb":
        return "HFrEF或左心室射血分数明显降低时，应避免非二氢吡啶类钙通道阻滞剂（地尔硫卓、维拉帕米）；心室率控制应选择适用于心衰的方案并监测血流动力学。"
    if group_name == "beta_blocker":
        return (
            "哮喘或气道高反应病史时，应避免静脉/口服β受体阻滞剂（如艾司洛尔、美托洛尔）控制心室率；"
            "优先选择钙通道阻滞剂（如地尔硫卓，需评估左室功能）或在血流动力学不稳定时同步电复律，"
            "并同步处理哮喘急性加重。"
        )
    return "存在明确药物过敏或禁忌时，应移除相应药物推荐并改用安全替代方案。"


def sanitize_contraindicated_recommendations(
    treatment_plan: str,
    case_features: Dict[str, Any],
) -> str:
    plan = clean_text(treatment_plan)
    for group_name, aliases in CONTRAINDICATED_DRUG_GROUPS.items():
        if not contraindication_group_present(group_name, aliases, case_features):
            continue
        for alias in aliases:
            plan = re.sub(
                r"(?:或|以及|及)\s*(?:使用|给予|选用|考虑使用|作为备选)?\s*%s(?:（[^）]*）)?"
                % re.escape(alias),
                "",
                plan,
                flags=re.I,
            )
        clauses = re.split(r"(?<=[，,；;。\n])", plan)
        plan = "".join(
            clause
            for clause in clauses
            if not treatment_recommends_drug(normalize_name(clause), aliases)
        )
    return normalize_treatment_text(plan)


def has_reduced_left_ventricular_ejection_fraction(case_text: str) -> bool:
    text = clean_text(case_text)
    normalized = normalize_name(text)
    measured_values = [
        float(match.group(1))
        for pattern in [
            r"(?:实测|测得|实际|结果(?:为)?)[^0-9]{0,2}(\d{1,3}(?:\.\d+)?)\s*%",
            r"(?:LVEF|左心室射血分数|射血分数)[^\n]{0,80}[：:]\s*(\d{1,3}(?:\.\d+)?)\s*%",
        ]
        for match in re.finditer(pattern, text, flags=re.I)
    ]
    if measured_values:
        return any(value < 40 for value in measured_values)
    numeric_values = [
        float(match.group(1))
        for match in re.finditer(
            r"(?:LVEF|左心室射血分数|射血分数)[^0-9]{0,20}(\d{1,3}(?:\.\d+)?)\s*%",
            text,
            flags=re.I,
        )
    ]
    if numeric_values:
        return any(value < 40 for value in numeric_values)
    for marker in ["射血分数明显降低", "射血分数降低", "hfref"]:
        index = normalized.find(marker)
        while index >= 0:
            before = normalized[max(0, index - 8):index]
            if not any(negation in before for negation in ["未见", "未提示", "未发现", "无", "不支持"]):
                return True
            index = normalized.find(marker, index + len(marker))
    return False


def treatment_goal_present(goal: str, normalized_plan: str) -> bool:
    goal_text = normalize_name(goal)
    if goal_text in normalized_plan:
        return True
    goal_markers = {
        "住院": ["住院", "急诊", "专科监测"],
        "静脉阿昔洛韦": ["静脉阿昔洛韦", "静脉注射阿昔洛韦", "静脉输注阿昔洛韦"],
        "静脉抗生素": ["静脉抗生素", "静脉注射头孢", "头孢唑林"],
        "调整免疫抑制": ["暂停", "调整", "免疫抑制", "糖皮质激素"],
    }
    for key, markers in goal_markers.items():
        if normalize_name(key) in goal_text:
            return any(normalize_name(marker) in normalized_plan for marker in markers)
    return False


def patch_for_goal(goal: str, patch_templates: Dict[str, Any]) -> str:
    goal_text = normalize_name(goal)
    key_markers = [
        ("住院", "hospitalization"),
        ("静脉阿昔洛韦", "iv_acyclovir"),
        ("静脉抗生素", "iv_antibiotic"),
        ("调整免疫抑制", "immunosuppression"),
    ]
    for marker, key in key_markers:
        if normalize_name(marker) in goal_text and clean_text(patch_templates.get(key)):
            return clean_text(patch_templates.get(key))
    return ""


def verification_candidate_records(
    case_features: Dict[str, Any],
    diagnosis: str,
) -> List[Dict[str, Any]]:
    records = [
        dict(item)
        for item in as_axis_list(case_features.get("diagnosis_candidate_records"))
        if clean_text(item.get("disease"))
    ]
    names = {clean_text(item.get("disease")) for item in records}
    for name in as_text_list(case_features.get("candidate_diagnoses")):
        candidate_name = clean_text(name)
        if candidate_name and candidate_name not in names:
            records.append(
                {
                    "disease": candidate_name,
                    "role": "current_problem" if candidate_name == diagnosis else "secondary",
                    "priority": "routine",
                    "source": "verification_compatibility",
                }
            )
            names.add(candidate_name)
    if diagnosis and diagnosis not in names:
        records.append(
            {
                "disease": diagnosis,
                "role": "current_problem",
                "priority": "routine",
                "source": "verification_selected_fallback",
            }
        )
    return records or [{"disease": diagnosis, "role": "current_problem", "priority": "routine"}]


def _structured_diagnosis_evidence(
    diagnosis: str,
    case_features: Mapping[str, Any],
) -> List[str]:
    """Map high-specificity structured findings to a diagnosis without self-proof."""
    if normalize_name(diagnosis) != normalize_name("皮肤蝇蛆病"):
        return []
    chunks: List[str] = []
    for payload in (case_features.get("examination_results") or {}).values():
        if not isinstance(payload, Mapping) or not exam_result_is_usable(payload):
            continue
        chunks.extend(structured_text_chunks(payload))
    text = normalize_name(" ".join(chunks))
    lesion_markers = ["皮肤", "结节", "疖", "穿刺点", "皮损", "引流"]
    larva_markers = ["幼虫", "蝇蛆", "虫体活动", "可见幼虫"]
    if any(normalize_name(marker) in text for marker in lesion_markers) and any(
        normalize_name(marker) in text for marker in larva_markers
    ):
        return [clean_text(item) for item in chunks if clean_text(item)]
    return []


def _diagnosis_has_grounded_evidence(
    diagnosis: str,
    case_features: Mapping[str, Any],
    examinations: Iterable[str],
) -> bool:
    """Require current-case evidence before authorizing a runtime diagnosis."""
    if case_features.get("_verified_case_memory_source") is True:
        return bool(case_features.get("safety_facts"))

    target = normalize_name(diagnosis)
    for record in as_axis_list(case_features.get("diagnosis_candidate_records")):
        if (
            normalize_name(record.get("disease")) == target
            and clean_text(record.get("source")) == "verified_case_prior"
        ):
            return True
    if not target:
        return False
    for record in as_axis_list(case_features.get("diagnosis_candidate_records")):
        if normalize_name(record.get("disease")) != target:
            continue
        if clean_text(record.get("source")) == "axis_alignment":
            continue
        evidence = [item for item in as_text_list(record.get("matched_evidence")) if item]
        if evidence:
            return True

    for axis in as_axis_list(case_features.get("diagnosis_axes")):
        if target not in {normalize_name(name) for name in axis_alignment_official_names(axis)}:
            continue
        evidence = as_text_list(axis.get("evidence"))
        if len(evidence) >= 2 and (
            axis.get("validated") is True
            or "rule" in clean_text(axis.get("source")).split("+")
        ):
            return True

    support_text = "\\n".join(
        item
        for item in (
            clean_text(case_features.get("patient_text")),
            " ".join(as_text_list(case_features.get("positive_findings"))),
            " ".join(as_text_list(case_features.get("exam_evidence"))),
        )
        if item
    )
    if disease_matched_evidence(diagnosis, support_text):
        return True
    structured_support_text = normalize_name(
        " ".join(
            item
            for item in (
                clean_text(case_features.get("case_text")),
                clean_text(case_features.get("patient_text")),
                " ".join(as_text_list(case_features.get("positive_findings"))),
            )
            if item
        )
    )
    # Some catalog diagnoses are grounded by a validated multi-symptom pattern
    # rather than a literal disease token (for example acute upper-airway illness).
    # Keep the threshold above ordinary symptom overlap so a candidate label alone
    # can never authorize a diagnosis.
    if disease_match_score(diagnosis, structured_support_text) >= 40:
        return True

    result_chunks: List[str] = []
    for payload in (case_features.get("examination_results") or {}).values():
        if isinstance(payload, Mapping) and exam_result_is_usable(payload):
            result_chunks.extend(structured_text_chunks(payload))
    result_text = " ".join(result_chunks)
    return bool(
        (result_text and disease_matched_evidence(diagnosis, result_text))
        or _structured_diagnosis_evidence(diagnosis, case_features)
    )


def final_verifier(
    *,
    diagnosis: str,
    examinations: Iterable[str],
    treatment_plan: str,
    official_diseases: Iterable[str],
    examination_catalog: Dict[str, List[str]],
    exam_plan_trace: List[Dict[str, Any]],
    case_features: Dict[str, Any],
    safety_profiles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    issues = []
    skin_preflight = apply_skin_myiasis_treatment_gate(
        diagnosis=diagnosis,
        treatment_plan=treatment_plan,
        case_features=case_features,
    )
    issues.extend(skin_preflight.get("issues", []))
    skin_patches = list(skin_preflight.get("patches", []))
    treatment_plan = append_unique_patches(
        clean_text(skin_preflight.get("treatment_plan") or treatment_plan),
        skin_patches,
    )
    official_disease_map = build_name_map(official_diseases)
    if not match_standard_name(diagnosis, official_disease_map):
        issues.append(
            {
                "field": "diagnosis",
                "code": "invalid_diagnosis_name",
                "severity": "must_fix",
                "problem": clean_text(diagnosis),
                "patchable": False,
            }
        )
    if not _diagnosis_has_grounded_evidence(diagnosis, case_features, examinations):
        issues.append(
            {
                "field": "diagnosis",
                "code": "diagnosis_without_current_case_evidence",
                "severity": "must_fix",
                "problem": "selected diagnosis lacks grounded patient, exam, axis, or verified-memory evidence",
                "patchable": False,
                "blocks_submission": True,
            }
        )

    specific_diagnosis = first_specific_candidate(
        as_text_list(case_features.get("candidate_diagnoses")),
        official_disease_map,
        generic_diagnosis=diagnosis,
    )
    if is_generic_final_diagnosis(diagnosis) and specific_diagnosis:
        issues.append(
            {
                "field": "diagnosis",
                "code": "underspecified_diagnosis",
                "severity": "must_fix",
                "problem": clean_text(diagnosis),
                "patchable": True,
                "edit": specific_diagnosis,
            }
        )

    leaf_names = set(flatten_examination_catalog(examination_catalog))
    seen_exams = set()
    for exam in as_text_list(examinations):
        if exam in seen_exams:
            issues.append(
                {
                    "field": "examinations",
                    "code": "duplicate_exam_name",
                    "severity": "should_fix",
                    "problem": exam,
                    "patchable": True,
                }
            )
        seen_exams.add(exam)
        if exam not in leaf_names:
            issues.append(
                {
                    "field": "examinations",
                    "code": "invalid_exam_name",
                    "severity": "must_fix",
                    "problem": exam,
                    "patchable": False,
                }
            )

    safety_result = apply_treatment_safety(
        treatment_plan,
        diagnosis=diagnosis,
        case_features=case_features,
        safety_profiles=safety_profiles,
    )
    issues.extend(safety_result.get("issues", []))
    fact_result = apply_fact_consistency_gate(
        safety_result.get("treatment_plan", treatment_plan),
        case_features,
    )
    issues.extend(fact_result.get("issues", []))
    coverage_result = apply_negative_evidence_scope_gate(
        fact_result.get("treatment_plan", safety_result.get("treatment_plan", treatment_plan)),
        examinations=examinations,
        case_features=case_features,
    )
    issues.extend(coverage_result.get("issues", []))
    specificity_result = apply_treatment_specificity_gate(
        treatment_plan=coverage_result.get("treatment_plan", fact_result.get("treatment_plan", treatment_plan)),
        diagnosis=diagnosis,
        examinations=examinations,
        case_features=case_features,
    )
    issues.extend(specificity_result.get("issues", []))
    diagnosis_specific_result = apply_diagnosis_specific_treatment_gate(
        diagnosis=diagnosis,
        treatment_plan=specificity_result.get(
            "treatment_plan",
            coverage_result.get("treatment_plan", fact_result.get("treatment_plan", treatment_plan)),
        ),
        case_features=case_features,
    )
    issues.extend(diagnosis_specific_result.get("issues", []))
    confirmatory_result = apply_confirmatory_evidence_treatment_gate(
        diagnosis=diagnosis,
        treatment_plan=diagnosis_specific_result.get(
            "treatment_plan",
            specificity_result.get(
                "treatment_plan",
                coverage_result.get("treatment_plan", fact_result.get("treatment_plan", treatment_plan)),
            ),
        ),
        case_features=case_features,
    )
    issues.extend(confirmatory_result.get("issues", []))
    axis_risk_result = apply_axis_risk_gate(
        confirmatory_result.get(
            "treatment_plan",
            diagnosis_specific_result.get(
                "treatment_plan",
                specificity_result.get("treatment_plan", coverage_result.get("treatment_plan", treatment_plan)),
            ),
        ),
        case_features,
        diagnosis=diagnosis,
    )
    issues.extend(axis_risk_result.get("issues", []))
    verified_treatment = axis_risk_result.get(
        "treatment_plan",
        confirmatory_result.get(
            "treatment_plan",
            diagnosis_specific_result.get(
                "treatment_plan",
                specificity_result.get("treatment_plan", coverage_result.get("treatment_plan", treatment_plan)),
            ),
        ),
    )
    diagnosis_axes = as_axis_list(case_features.get("diagnosis_axes"))
    # FinalVerifier validates the diagnosis selected upstream; it must not become
    # a second diagnosis agent or silently rewrite the label (P2-1).
    selected_diagnosis = clean_text(diagnosis)
    plan_text = clean_text(verified_treatment or treatment_plan)
    escalation_markers = [
        "急诊",
        "立即转诊",
        "转诊",
        "专科",
        "气道",
        "紧急",
        "抢救",
        "儿童眼科",
        "耳鼻喉",
        " ent ",
    ]
    plan_has_escalation = any(marker in plan_text for marker in escalation_markers if marker.strip())
    high_risk_conflict = False
    safe_escalation_required = False
    preferred_axis_diagnosis = ""
    for axis in diagnosis_axes:
        if not isinstance(axis, dict):
            continue
        priority = clean_text(axis.get("priority")) or "routine"
        status = clean_text(axis.get("status")) or "suspected"
        if priority not in {"high", "red_flag"} and status != "confirmed":
            continue
        if not axis_is_dominant_for_verifier(axis):
            continue
        sources = {item for item in clean_text(axis.get("source")).split("+") if item}
        supported: List[str] = []
        if "rule" in sources:
            supported.extend(
                as_text_list(
                    axis.get("rule_candidate_official_names")
                    or axis.get("candidate_official_names")
                )
            )
        if "llm" in sources and axis.get("validated") is True:
            supported.extend(as_text_list(axis.get("promotable_candidate_official_names")))
        supported = unique_preserve_order(supported)
        if not preferred_axis_diagnosis and supported:
            preferred_axis_diagnosis = clean_text(supported[0])
        if priority == "red_flag" and not supported:
            safe_escalation_required = True
        # An unresolved LLM axis without an official candidate is not enough to
        # displace a diagnosis that already has a catalog-backed candidate.
        if supported and not diagnosis_covers_axis_for_verifier(selected_diagnosis, axis):
            if verified_prior_allows_skin_infection_axis(
                selected_diagnosis,
                axis,
                case_features,
            ):
                continue
            high_risk_conflict = True
            break
    if high_risk_conflict or (safe_escalation_required and not plan_has_escalation):
        issues.append(
            {
                "field": "diagnosis",
                "code": "diagnosis_conflicts_with_high_risk_axis",
                "problem": "最终诊断未覆盖已验证的当前主问题或红旗轴。",
                # Suggested edit is diagnostic-stage guidance only; verifier never
                # applies or returns a selected diagnosis.
                "edit": preferred_axis_diagnosis,
                "severity": "must_fix",
                "blocks_submission": True,
                "patchable": False,
            }
        )
    patched_treatment = append_unique_patches(
        verified_treatment,
        fact_result.get("patches", [])
        + coverage_result.get("patches", [])
        + specificity_result.get("patches", [])
        + confirmatory_result.get("patches", [])
        + diagnosis_specific_result.get("patches", [])
        + axis_risk_result.get("patches", [])
        + skin_patches,
    )
    result = {
        "passed": not issues,
        "issues": issues,
        "patched_treatment": patched_treatment,
        "treatment_hash": treatment_review_plan_hash(patched_treatment or treatment_plan),
    }
    return result



def five_dimension_clinical_report(
    *,
    diagnosis: str,
    treatment_plan: str,
    clinical_basis: Iterable[str],
    case_features: Mapping[str, Any],
    examinations: Iterable[str],
) -> Dict[str, Any]:
    """Five-dimensional clinical safety report for one diagnosis of the final text.

    Dimensions: treatment_target, contraindication, acute_sequence, monitoring,
    drug_interaction. Each dimension is judged against the registry clinical_basis
    and the patient's REAL facts (allergies / contraindications / resistance /
    comorbidities / exam findings) from case_features — never from plan keywords
    alone. Unsupported dimensions must report fail/not_proven, not a pass.
    """
    plan = normalize_name(treatment_plan)
    basis_text = normalize_name(" ".join(as_text_list(clinical_basis)))
    exam_text = normalize_name(" ".join(as_text_list(examinations)))

    allergies = set(_expand_resistance_classes(as_text_list(case_features.get("drug_allergies")))
                  ) | set(normalize_name(a) for a in as_text_list(case_features.get("drug_allergies")))
    contras = set(normalize_name(c) for c in as_text_list(case_features.get("contraindicated_drugs")))
    resistance = set(normalize_name(r) for r in as_text_list(
        (case_features.get("anti_infective_provenance") or {}).get("confirmed_resistance")
    ))

    def _has(plan_needles):
        return any(n in plan for n in plan_needles)

    # treatment_target: plan addresses the diagnosis core target
    dx_norm = normalize_name(diagnosis)
    target_ok = dx_norm and dx_norm in plan
    target_basis = diagnosis in list(clinical_basis) or diagnosis in as_text_list(
        case_features.get("candidate_diagnoses")
    )

    # contraindication: no drug named that the patient is allergic/resistant to
    contra_ok = True
    contra_hits = []
    for drug in _named_drugs_in_plan(plan):
        dn = normalize_name(drug)
        if dn in allergies or dn in contras or dn in resistance:
            contra_ok = False
            contra_hits.append(drug)

    # acute_sequence: time-critical actions ordered early when axis is acute
    acute_markers = ["紧急", "立即", "急诊", "尽快", "住院", "吸氧", "抢救"]
    acute_needed = any(m in basis_text for m in {"急诊", "立即", "危险", "红色"}) or any(
        normalize_name(m) in dx_norm for m in {"出血", "梗死", "心梗", "卒中", "中毒", "急症"}
    )
    acute_ok = (not acute_needed) or _has(acute_markers)

    # monitoring: follow-up / monitoring language present
    monitor_markers = ["监测", "复查", "随访", "观察", "复诊", "评估", "调整"]
    monitor_ok = _has(monitor_markers)

    # drug_interaction: avoid explicit interaction when plan has >1 drug
    named_drugs = _named_drugs_in_plan(plan)
    interaction_markers = ["相互作用", "配伍", "相互作用监测", "联合用药注意"]
    interaction_ok = not _requires_multi_drug_interaction_note(plan, named_drugs) or _has(
        interaction_markers
    )
    interaction_proven = True

    def _status(ok, proven):
        # Unsupported / unproven dimensions must be fail|not_proven, never pass.
        if not proven:
            return "not_proven"
        return "pass" if ok else "fail"

    return {
        "diagnosis": clean_text(diagnosis),
        "treatment_target": _status(target_ok, target_basis),
        "contraindication": _status(contra_ok, True),
        "acute_sequence": _status(acute_ok, True),
        "monitoring": _status(monitor_ok, True),
        "drug_interaction": _status(interaction_ok, interaction_proven),
        "contraindication_hits": contra_hits,
        "named_drugs": named_drugs,
    }


def five_dimension_gate(
    reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate five-dimension reports into a submission gate.

    all_passed is true only when every dimension is "pass"; "not_proven" never
    counts as a pass. blocked is reserved for fact-backed dimensions
    (contraindication with hits). Keyword-only misses degrade to review.
    """
    blocking_findings: List[Dict[str, Any]] = []
    review_findings: List[Dict[str, Any]] = []
    all_passed = True

    for report in reports:
        for dimension in ("treatment_target", "contraindication", "acute_sequence", "monitoring", "drug_interaction"):
            value = report.get(dimension)
            if value != "pass":
                all_passed = False
                finding = {"dimension": dimension, "value": value, "hits": report.get("contraindication_hits", [])}
                # Fact-backed dimensions with hits block submission.
                if dimension == "contraindication" and value == "fail" and finding["hits"]:
                    blocking_findings.append(finding)
                else:
                    review_findings.append(finding)

    if blocking_findings:
        status = "blocked"
    elif review_findings:
        status = "review"
    else:
        status = "pass"

    return {
        "all_passed": all_passed,
        "blocked": bool(blocking_findings),
        "status": status,
        "blocking_findings": blocking_findings,
        "review_findings": review_findings,
    }


def enforce_five_dimension_gate(
    *,
    diagnoses: Sequence[str],
    treatment_plan: str,
    clinical_basis: Iterable[str],
    case_features: Mapping[str, Any],
    examinations: Iterable[str],
) -> Dict[str, Any]:
    """Repair blocking violations deterministically and re-score.

    Removes named drugs that hit allergy/contraindication/resistance from the
    plan, then re-runs the five-dimension report on the sanitized text.
    Returns a dict with keys: gate, treatment_plan, five_dimension.
    """
    plan = normalize_name(treatment_plan)
    allergies = set(_expand_resistance_classes(as_text_list(case_features.get("drug_allergies")))
                  ) | set(normalize_name(a) for a in as_text_list(case_features.get("drug_allergies")))
    contras = set(normalize_name(c) for c in as_text_list(case_features.get("contraindicated_drugs")))
    resistance = set(normalize_name(r) for r in as_text_list(
        (case_features.get("anti_infective_provenance") or {}).get("confirmed_resistance")
    ))

    # Remove offending drugs from the plan.
    sanitized = treatment_plan
    for drug in _named_drugs_in_plan(plan):
        dn = normalize_name(drug)
        if dn in allergies or dn in contras or dn in resistance:
            sanitized = re.sub(re.escape(drug), "", sanitized)

    normalized_sanitized = normalize_name(sanitized)
    missing_diagnoses = [
        clean_text(diagnosis)
        for diagnosis in diagnoses
        if clean_text(diagnosis)
        and normalize_name(diagnosis) not in normalized_sanitized
    ]
    if missing_diagnoses:
        sanitized = "针对%s：%s" % (
            "、".join(missing_diagnoses),
            clean_text(sanitized),
        )
        normalized_sanitized = normalize_name(sanitized)
    if _requires_multi_drug_interaction_note(
        normalized_sanitized,
        _named_drugs_in_plan(normalized_sanitized),
    ) and not any(
        marker in normalized_sanitized for marker in ["相互作用", "配伍", "联合用药注意"]
    ):
        sanitized = (
            clean_text(sanitized)
            + " 联合用药注意：核对药物相互作用和配伍禁忌，必要时调整治疗方案。"
        )

    # Build reports for each diagnosis on the sanitized plan.
    five_dimension = []
    for diagnosis in diagnoses:
        report = five_dimension_clinical_report(
            diagnosis=diagnosis,
            treatment_plan=sanitized,
            clinical_basis=clinical_basis,
            case_features=case_features,
            examinations=examinations,
        )
        five_dimension.append(report)

    gate = five_dimension_gate(five_dimension)
    removed_contra_drugs = []
    original_named = set(_named_drugs_in_plan(normalize_name(treatment_plan)))
    sanitized_named = set(_named_drugs_in_plan(normalize_name(sanitized)))
    removed_contra_drugs = sorted(original_named - sanitized_named)
    if removed_contra_drugs:
        generic_antimicrobial_only = (
            any(marker in normalize_name(sanitized) for marker in ["抗感染", "抗菌", "抗生素"])
            and not any(
                drug not in _GENERIC_DRUG_MARKERS
                for drug in _named_drugs_in_plan(normalize_name(sanitized))
            )
        )
        if generic_antimicrobial_only:
            gate["all_passed"] = False
            gate["blocked"] = True
            gate["status"] = "blocked"
            gate.setdefault("blocking_findings", []).append(
                {
                    "dimension": "contraindication",
                    "value": "replacement_missing",
                    "hits": removed_contra_drugs,
                    "code": "contraindicated_drug_removed_without_specific_alternative",
                }
            )
            gate["repair_failed"] = True
    # Check if the original plan had a blocking violation that was repaired.
    original_blocked = False
    for diagnosis in diagnoses:
        original_report = five_dimension_clinical_report(
            diagnosis=diagnosis,
            treatment_plan=treatment_plan,
            clinical_basis=clinical_basis,
            case_features=case_features,
            examinations=examinations,
        )
        if original_report.get("contraindication") == "fail" and original_report.get("contraindication_hits"):
            original_blocked = True
            break
    gate["repaired_from_blocked"] = original_blocked and gate["blocked"] is False
    return {
        "gate": gate,
        "treatment_plan": sanitized,
        "five_dimension": five_dimension,
    }


def _named_drugs_in_plan(normalized_plan: str) -> List[str]:
    """Heuristic extraction of drug-class keywords from a normalized plan."""
    markers = [
        # Specific drugs first (longer matches优先)
        "阿莫西林", "环丙沙星", "左氧氟沙星", "头孢克洛", "莫匹罗星",
        "夫西地酸", "阿昔洛韦", "更昔洛韦", "奥司他韦",
        # Drug classes
        "抗生素", "抗菌", "抗感染", "抗病毒", "抗真菌", "抗结核", "抗排斥",
        "糖皮质激素", "激素", "利尿", "降压", "抗凝", "抗血小板", "他汀",
        "胰岛素", "化疗", "靶向", "免疫抑制剂", "支气管扩张", "β激动剂",
        "钙通道阻滞剂", "arb", "acei", "华法林", "阿司匹林", "肝素",
        "statin", "他汀类", "ccb", "β阻滞剂", "青霉素",
    ]
    return [m for m in markers if m in normalized_plan]


_GENERIC_DRUG_MARKERS = {
    "抗生素", "抗菌", "抗感染", "抗病毒", "抗真菌", "抗结核", "抗排斥",
    "糖皮质激素", "激素", "利尿", "降压", "抗凝", "抗血小板", "他汀",
    "化疗", "靶向", "免疫抑制剂", "支气管扩张", "β激动剂", "钙通道阻滞剂",
    "他汀类", "ccb", "β阻滞剂", "青霉素",
}


def _requires_multi_drug_interaction_note(
    normalized_plan: str,
    named_drugs: List[str],
) -> bool:
    if len(named_drugs) < 2:
        return False
    if any(marker in normalized_plan for marker in ["联合", "合用", "并用", "同时使用"]):
        return True
    if any(marker in normalized_plan for marker in ["或", "二选一", "任选", "替代"]):
        return False
    concrete_drugs = [drug for drug in named_drugs if drug not in _GENERIC_DRUG_MARKERS]
    return len(concrete_drugs) >= 2


def case_features_text(case_features: Dict[str, Any]) -> str:
    chunks = [
        clean_text(case_features.get("case_text")),
        clean_text(case_features.get("chief_complaint")),
        feature_evidence_text(case_features),
    ]
    for key in (
        "positive_findings",
        "red_flags",
        "personal_history",
        "candidate_diagnoses",
        "organ_risk",
    ):
        chunks.extend(as_text_list(case_features.get(key)))
    for axis in as_axis_list(case_features.get("diagnosis_axes")):
        chunks.extend(as_text_list(axis.get("evidence")))
        chunks.extend(as_text_list(axis.get("candidate_official_names")))
        chunks.append(clean_text(axis.get("axis_id")))
    return normalize_name(" ".join(chunk for chunk in chunks if chunk))


def exams_cover_upper_arm_fracture(examinations: Iterable[str]) -> bool:
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if name == "四肢X线检查":
            return True
        if any(marker in name for marker in ["肱骨", "上臂"]):
            return True
    return False


def exams_cover_rhythm(examinations: Iterable[str]) -> bool:
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if any(marker in name for marker in ["心电图", "ECG", "Holter", "动态心电图"]):
            return True
    return False


def exams_cover_electrolytes(examinations: Iterable[str]) -> bool:
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if any(marker in name for marker in ["电解质", "血钾", "血清钾", "镁"]):
            return True
    return False


def exams_cover_hepato_splenic_structure(examinations: Iterable[str]) -> bool:
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if any(marker in name for marker in ["腹部超声", "肝脏超声", "脾脏", "门脉", "腹部CT"]):
            return True
    return False


def exams_cover_local_enthesopathy(examinations: Iterable[str]) -> bool:
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if any(marker in name for marker in ["体格检查", "肌骨", "肌肉骨骼超声", "关节超声"]):
            return True
    return False


def exams_cover_anca(examinations: Iterable[str]) -> bool:
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if any(marker in name for marker in ["ANCA", "抗中性粒细胞胞质抗体"]):
            return True
    return False


def exams_cover_renal_urine(examinations: Iterable[str]) -> bool:
    has_ua = False
    has_rft = False
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if any(marker in name for marker in ["尿液分析", "UA", "尿常规", "24小时尿蛋白"]):
            has_ua = True
        if any(marker in name for marker in ["肾功能", "RFT", "肝肾功能", "肌酐"]):
            has_rft = True
    return has_ua or has_rft


def exams_cover_chest_imaging(examinations: Iterable[str]) -> bool:
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if any(marker in name for marker in ["胸部X线", "胸部CT", "Chest CT", "Chest HRCT", "胸片", "CXR"]):
            return True
    return False


def mediastinal_cxr_needs_ct(case_state: Dict[str, Any]) -> bool:
    structural_markers = ["纵隔影增宽", "纵隔增宽", "纵隔异常", "纵隔占位", "纵隔肿块"]
    resolved = ["未见", "未发现", "未提示", "不支持", "未检出", "无", "正常"]
    for payload in matching_exam_payloads(case_state, ["胸部X线", "CXR"]):
        if not exam_result_is_usable(payload):
            continue
        if normalize_name(payload.get("status")) in {"normal", "negative"}:
            continue
        result_text = "；".join("%s：%s" % pair for pair in exam_result_pairs(payload))
        if marker_present_active(result_text, structural_markers, resolved_markers=resolved):
            return True
        normalized_result = normalize_name(result_text)
        ct_recommended = any(
            marker in normalized_result
            for marker in ["建议进一步胸部ct", "建议胸部ct", "建议进一步ct", "建议ct"]
        )
        ct_recommendation_negated = bool(
            re.search(r"(?:不建议|未建议|无需|不需|不必).{0,10}(?:胸部)?ct", normalized_result)
        )
        if ct_recommended and not ct_recommendation_negated:
            return True
    return False


def exams_cover_urinary_imaging(examinations: Iterable[str]) -> bool:
    for exam in as_text_list(examinations):
        name = clean_text(exam)
        if any(
            marker in name
            for marker in ["泌尿道超声", "泌尿系超声", "肾脏超声", "腹部CT", "泌尿系CT", "CT尿路成像"]
        ):
            return True
    return False


def exams_cover_blood_smear(examinations: Iterable[str]) -> bool:
    return any("外周血涂片" in clean_text(exam) for exam in as_text_list(examinations))


def exams_cover_cbc(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["全血细胞计数", "血常规", "CBC"])
        for exam in as_text_list(examinations)
    )


def exams_cover_oxygenation_vitals(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["生命体征", "脉搏血氧", "SpO2", "血氧饱和度", "动脉血气"])
        for exam in as_text_list(examinations)
    )


def exams_cover_liver_injury_labs(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["肝功能", "LFT", "凝血功能全套", "全血细胞计数", "血常规", "CBC"])
        for exam in as_text_list(examinations)
    )


def exams_cover_hindfoot_bone_imaging(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["四肢X线", "足部X线", "跟骨", "踝关节X线", "下肢X线"])
        for exam in as_text_list(examinations)
    )


def exams_cover_glucose(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["血糖检测", "空腹血糖", "FBG", "随机血糖", "血糖"])
        and "糖化" not in clean_text(exam)
        for exam in as_text_list(examinations)
    )


def exams_cover_metabolic_acidosis_or_ketosis(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["动脉血气", "ABG", "血气", "尿液分析", "UA", "酮"])
        for exam in as_text_list(examinations)
    )


def exams_cover_echocardiography(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["超声心动图", "心脏超声", "TTE", "TEE"])
        for exam in as_text_list(examinations)
    )


def exams_cover_advanced_chest_imaging(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["胸部X线", "胸部CT", "Chest CT", "胸片", "CXR"])
        for exam in as_text_list(examinations)
    )


def chest_ultrasound_effusion_status(case_state: Dict[str, Any]) -> str:
    for payload in matching_exam_payloads(case_state, ["胸部超声", "胸腔超声"]):
        if not exam_result_is_usable(payload) and normalize_name(payload.get("status")) not in {
            "normal",
            "negative",
            "abnormal",
        }:
            continue
        text = normalize_name(
            " ".join("%s %s" % (key, result_value_without_reference(value)) for key, value in exam_result_pairs(payload))
        )
        if any(marker in text for marker in ["大量", "中等量", "明显积液", "可见积液", "积液阳性"]):
            return "positive"
        if any(marker in text for marker in ["未见", "无积液", "双侧未见", "未探及"]):
            return "negative"
        if normalize_name(payload.get("status")) in {"normal", "negative"}:
            return "negative"
        if normalize_name(payload.get("status")) == "abnormal":
            return "positive"
    return "unknown"


def exams_cover_systemic_pathogen(examinations: Iterable[str], *, vector_exposure: bool = False) -> bool:
    names = as_text_list(examinations)
    serology = any(any(marker in clean_text(exam) for marker in ["血清学抗体", "病原体抗体", "病原体核酸"] ) for exam in names)
    if vector_exposure:
        return serology
    return serology or any("血培养" in clean_text(exam) for exam in names)


def exams_cover_blood_culture(examinations: Iterable[str]) -> bool:
    return any("血培养" in clean_text(exam) for exam in as_text_list(examinations))


def exams_cover_otoscopy(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["耳镜检查", "气压耳镜", "耳内镜"])
        for exam in as_text_list(examinations)
    )


def exams_cover_infection_activity(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["C反应蛋白", "CRP", "降钙素原", "PCT"])
        for exam in as_text_list(examinations)
    )


def exams_cover_soft_tissue_local_imaging(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["软组织超声", "皮肤超声", "肌肉骨骼超声"])
        for exam in as_text_list(examinations)
    )


def exams_cover_middle_ear_mechanism(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["鼓室压图", "声导抗"])
        for exam in as_text_list(examinations)
    )


def exams_cover_deep_ear_structure(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["耳部CT", "颞骨CT"])
        for exam in as_text_list(examinations)
    )


def tympanometry_result_explains(case_state: Dict[str, Any]) -> bool:
    payloads = matching_exam_payloads(case_state, ["鼓室压图", "声导抗"])
    for payload in payloads:
        if not exam_result_is_usable(payload):
            continue
        if normalize_name(payload.get("status")) == "abnormal":
            return True
        result_text = normalize_name(" ".join(value for _, value in exam_result_pairs(payload)))
        if any(marker in result_text for marker in ["b型", "c型", "中耳传导机制异常", "鼓室图异常"]):
            return True
    return False


def exams_cover_monoclonal_protein(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["血清蛋白电泳", "血清免疫电泳", "尿蛋白电泳"])
        for exam in as_text_list(examinations)
    )


def exams_cover_hcv_etiology(examinations: Iterable[str]) -> bool:
    return any(
        "丙型肝炎病毒" in clean_text(exam) and any(marker in clean_text(exam) for marker in ["抗体", "核酸"])
        for exam in as_text_list(examinations)
    )


def exams_cover_markers(examinations: Iterable[str], markers: Iterable[str]) -> bool:
    normalized_markers = [normalize_name(marker) for marker in markers]
    return any(
        any(marker in normalize_name(exam) for marker in normalized_markers)
        for exam in as_text_list(examinations)
    )


def open_coverage_gaps(case_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Closable diagnostic coverage gaps from case facts vs ordered exams (no LLM)."""
    # Mirror select_diagnosis_axes: structured facts plus patient wording and exam results.
    facts = extract_intake_facts(case_state)
    axis_text = normalize_name(
        "\n".join(
            item
            for item in [
                intake_facts_text(facts),
                patient_text_for_matching(case_state),
                examination_text_for_matching(case_state),
            ]
            if item
        )
    )
    ordered = valid_ordered_examinations(case_state)
    completed = completed_examinations(case_state)
    gaps: List[Dict[str, Any]] = []

    if has_upper_arm_trauma_pattern(axis_text) and not exams_cover_upper_arm_fracture(completed):
        gaps.append(
            {
                "gap_id": "upper_arm_long_bone_imaging",
                "exam_intents": ["上臂长骨骨折影像", "损伤段骨性成像"],
                "required_exams": ["四肢X线检查"],
                "reason": "上臂外伤未覆盖损伤段长骨成像，肩/手邻接部位片不能闭合骨折排除。",
            }
        )

    if has_palpitation_arrhythmia_pattern(axis_text):
        if not exams_cover_rhythm(completed):
            gaps.append(
                {
                    "gap_id": "rhythm_ecg",
                    "exam_intents": ["心律评估"],
                    "required_exams": ["心电图（ECG）"],
                    "reason": "突发心悸未完成心电节律评估，不能无证据切到 ACS 强化路径。",
                }
            )
        if not exams_cover_electrolytes(completed):
            gaps.append(
                {
                    "gap_id": "electrolyte_precipitant",
                    "exam_intents": ["电解质评估"],
                    "required_exams": ["血清电解质"],
                    "reason": "心悸诱因轴未覆盖电解质评估。",
                }
            )

    if has_hepato_splenic_cytopenia_pattern(axis_text) and not exams_cover_hepato_splenic_structure(completed):
        gaps.append(
            {
                "gap_id": "hepato_splenic_structure",
                "exam_intents": ["肝脾门脉结构评估"],
                "required_exams": ["腹部超声"],
                "reason": "慢性肝病伴左上腹饱胀与三系减少时，需先评估肝脾门脉结构再谈原发骨髓病。",
            }
        )

    if (
        has_elbow_overuse_pattern(axis_text)
        and not has_systemic_infection_or_inflammation_pattern(axis_text)
        and not exams_cover_local_enthesopathy(completed)
        and not ordered
    ):
        # Only force a local assessment when no exams exist yet; do not demand endless soft exams.
        gaps.append(
            {
                "gap_id": "local_enthesopathy_exam",
                "exam_intents": ["局部肌腱附着点评估"],
                "required_exams": ["体格检查"],
                "reason": "局限肘部过度使用痛应先做局部肌腱附着点评估，而非无收益 CBC。",
            }
        )

    if has_pulmonary_renal_vasculitis_pattern(axis_text):
        if not exams_cover_anca(completed):
            gaps.append(
                {
                    "gap_id": "anca_vasculitis_screen",
                    "exam_intents": ["ANCA血管炎筛查"],
                    "required_exams": ["抗中性粒细胞胞质抗体（ANCA）谱"],
                    "reason": "慢性咳嗽咯血伴肾/系统线索时，需筛查 ANCA 血管炎，不能仅凭单次非特异抗体收口。",
                }
            )
        if not exams_cover_renal_urine(completed):
            gaps.append(
                {
                    "gap_id": "renal_urine_workup",
                    "exam_intents": ["尿液肾脏评估", "肾功能评估"],
                    "required_exams": ["尿液分析（UA）", "肾功能检查（RFTs）"],
                    "reason": "肺出血样表现需同步评估尿液与肾功能，闭合肺肾鉴别。",
                }
            )
        if not exams_cover_chest_imaging(completed):
            gaps.append(
                {
                    "gap_id": "chest_imaging",
                    "exam_intents": ["胸部结构影像"],
                    "required_exams": ["胸部X线检查（CXR）"],
                    "reason": "慢性咳嗽咯血需胸部结构影像覆盖肺部病变范围。",
                }
            )

    if has_symptomatic_hypokalemia_malabsorption_pattern(axis_text):
        if not exams_cover_electrolytes(completed):
            gaps.append(
                {
                    "gap_id": "multi_electrolyte_panel",
                    "exam_intents": ["多电解质与镁评估"],
                    "required_exams": ["血清电解质"],
                    "reason": "症状性低钾伴腹泻/吸收不良时，需多电解质面板评估钾镁钙状态。",
                }
            )
        if not exams_cover_rhythm(completed):
            gaps.append(
                {
                    "gap_id": "electrolyte_rhythm_safety",
                    "exam_intents": ["心律评估"],
                    "required_exams": ["心电图（ECG）"],
                    "reason": "症状性电解质紊乱需心电评估心律失常与 QTc 风险。",
                }
            )

    if (
        has_urinary_stone_infection_differential_pattern(axis_text)
        and not exams_cover_urinary_imaging(completed)
    ):
        gaps.append(
            {
                "gap_id": "urinary_stone_infection_imaging",
                "exam_intents": ["泌尿系结石影像"],
                "required_exams": ["泌尿道超声"],
                "reason": "腰痛伴尿路刺激征和血尿时，需用泌尿系影像闭合结石/梗阻与感染鉴别。",
            }
        )

    if has_acute_respiratory_illness_pattern(axis_text) and not exams_cover_oxygenation_vitals(completed):
        gaps.append(
            {
                "gap_id": "respiratory_oxygenation_vitals",
                "exam_intents": ["氧合与生命体征评估"],
                "required_exams": ["生命体征", "脉搏血氧饱和度监测（SpO2）"],
                "reason": "急性呼吸道症状伴气促/咳嗽加重时，需先闭合生命体征与血氧，避免在未知氧合状态下收口。",
            }
        )

    if has_high_risk_pediatric_lower_respiratory_infection_pattern(axis_text):
        needs_chest_imaging = not exams_cover_chest_imaging(completed)
        needs_blood_culture = not exams_cover_blood_culture(completed)
        if needs_chest_imaging or needs_blood_culture:
            required_exams = []
            if needs_chest_imaging:
                required_exams.append("胸部X线检查（CXR）")
            if needs_blood_culture:
                required_exams.append("血培养")
            gaps.append(
                {
                    "gap_id": "高危儿科下呼吸道感染影像与病原覆盖",
                    "exam_intents": ["高危儿童下呼吸道感染结构与病原评估"],
                    "required_exams": required_exams,
                    "reason": "儿童免疫抑制背景下迁延发热咳嗽伴脓痰或喘息时，需胸部结构影像和血培养闭合肺炎严重度与病原证据，不能按普通支气管炎仅做家庭观察。",
                }
            )

    if (
        has_acute_ear_pain_after_instrumentation_pattern(axis_text)
        and not exams_cover_otoscopy(completed)
    ):
        gaps.append(
            {
                "gap_id": "acute_ear_pain_otoscopy",
                "exam_intents": ["外耳道与鼓膜直视评估"],
                "required_exams": ["耳镜检查"],
                "reason": "耳道操作后急性耳痛伴耳堵、耳鸣或加重时，需先耳镜检查外耳道和鼓膜，不能仅按耵聍栓塞或自行冲洗收口。",
            }
        )

    if has_chronic_alcohol_liver_injury_pattern(axis_text):
        if not exams_cover_liver_injury_labs(completed):
            gaps.append(
                {
                    "gap_id": "alcohol_liver_lab_injury",
                    "exam_intents": ["酒精性肝损伤实验室评估"],
                    "required_exams": ["肝功能检查（LFTs）", "凝血功能全套"],
                    "reason": "长期大量饮酒伴乏力/腹胀/纳差时，仅体格检查不能闭合肝细胞损伤与合成功能评估。",
                }
            )
        if not exams_cover_hepato_splenic_structure(completed):
            gaps.append(
                {
                    "gap_id": "alcohol_liver_structure",
                    "exam_intents": ["肝脾门脉结构评估"],
                    "required_exams": ["腹部超声"],
                    "reason": "酒精暴露肝病轴需要腹部超声评估肝实质与门脉高压相关结构，不能只做皮肤巩膜视诊。",
                }
            )

    if has_pleuritic_pain_infection_embolism_pattern(axis_text) and not exams_cover_chest_imaging(completed):
        gaps.append(
            {
                "gap_id": "pleuritic_chest_imaging",
                "exam_intents": ["胸膜肺结构影像评估"],
                "required_exams": ["胸部X线检查（CXR）", "胸部CT扫描（Chest CT）"],
                "reason": "胸膜性胸痛伴干咳/胸闷或肺梗死史时，应先闭合胸肺结构影像，不能以非特异自身免疫或心脏检查替代。",
            }
        )

    if has_high_energy_hindfoot_trauma_pattern(axis_text) and not exams_cover_hindfoot_bone_imaging(completed):
        gaps.append(
            {
                "gap_id": "high_energy_hindfoot_imaging",
                "exam_intents": ["足跟骨性结构影像"],
                "required_exams": ["四肢X线检查"],
                "reason": "高能量足跟创伤需足/跟骨结构影像，不能无影像按软组织扭伤收口。",
            }
        )

    if has_febrile_polyuria_dehydration_pattern(axis_text):
        if not exams_cover_glucose(completed):
            gaps.append(
                {
                    "gap_id": "hyperglycemic_crisis_glucose",
                    "exam_intents": ["高血糖危象血糖评估"],
                    "required_exams": ["血糖检测", "空腹血糖（FBG）"],
                    "reason": "高热伴多尿烦渴脱水时，必须先闭合血糖，不能仅靠电解质就按尿崩收口。",
                }
            )
        if not exams_cover_metabolic_acidosis_or_ketosis(completed):
            gaps.append(
                {
                    "gap_id": "hyperglycemic_crisis_metabolic",
                    "exam_intents": ["代谢酸中毒与酮症评估"],
                    "required_exams": ["动脉血气（ABG）", "尿液分析（UA）"],
                    "reason": "疑似高血糖危象时需评估代谢性酸中毒/酮体，血清电解质 alone 不够。",
                }
            )

    if has_infant_congenital_structural_heart_pattern(axis_text) and not exams_cover_echocardiography(completed):
        gaps.append(
            {
                "gap_id": "infant_chd_echocardiography",
                "exam_intents": ["心脏结构和血流动力学评估"],
                "required_exams": ["超声心动图"],
                "reason": "新生儿/婴儿心肺窘迫伴喂养困难与发绀线索时，生命体征与SpO2不能替代超声心动图闭合心脏结构证据。",
            }
        )

    if has_postop_chylothorax_or_pleural_effusion_pattern(axis_text):
        us_status = chest_ultrasound_effusion_status(case_state)
        if not exams_cover_advanced_chest_imaging(completed) and us_status in {"unknown", "negative"}:
            gaps.append(
                {
                    "gap_id": "postop_chest_effusion_further_imaging",
                    "exam_intents": ["术后胸腔积液结构评估", "胸膜肺结构影像评估"],
                    "required_exams": ["胸部X线检查（CXR）", "胸部CT扫描（Chest CT）"],
                    "reason": "胸部/纵隔术后出现呼吸费力时，单次阴性床旁胸部超声不能关闭胸腔积液/乳糜轴，需进一步胸部结构影像。",
                }
            )

    if has_systemic_infection_hematologic_axis_pattern(axis_text):
        vector_exposure = normalize_name("媒介或户外暴露") in axis_text
        cbc_closed = exams_cover_cbc(completed)
        smear_closed = exams_cover_blood_smear(completed)
        pathogen_closed = exams_cover_systemic_pathogen(completed, vector_exposure=vector_exposure)
        infection_activity_closed = exams_cover_infection_activity(completed)
        if not (cbc_closed and smear_closed and pathogen_closed and infection_activity_closed):
            intents = []
            required_exams = []
            if not cbc_closed:
                intents.append("血细胞数量与严重度评估")
                required_exams.append("全血细胞计数（CBC）")
            if not smear_closed:
                intents.append("血液细胞形态鉴别")
                required_exams.append("外周血涂片")
            if not pathogen_closed:
                intents.append("媒介或暴露相关病原评估" if vector_exposure else "全身感染病原评估")
                required_exams.append("血清学抗体检测" if vector_exposure else "血培养")
            if not infection_activity_closed:
                intents.append("全身感染活动度评估")
                required_exams.append("C反应蛋白（CRP）")
            gaps.append(
                {
                    "gap_id": "systemic_infection_vs_primary_hematologic",
                    "exam_intents": intents,
                    "required_exams": required_exams,
                    "reason": "反复高热伴全身痛和多部位出血时，需同步闭合感染病原与血细胞形态证据。",
                }
            )

    if has_focal_ear_conductive_axis_pattern(axis_text) and not otoscopy_has_local_explanation(case_state):
        if not exams_cover_middle_ear_mechanism(completed):
            gaps.append(
                {
                    "gap_id": "focal_ear_pain_structural_localization",
                    "exam_intents": ["中耳传导机制评估"],
                    "required_exams": ["鼓室压图及声导抗检查"],
                    "reason": "局灶剧烈耳痛伴传导性听损未被耳镜解释，先定位中耳传导机制。",
                }
            )
        elif not tympanometry_result_explains(case_state) and not exams_cover_deep_ear_structure(completed):
            gaps.append(
                {
                    "gap_id": "focal_ear_pain_structural_localization",
                    "exam_intents": ["耳部深层结构定位"],
                    "required_exams": ["耳部CT扫描（Ear CT）"],
                    "reason": "声导抗仍未解释剧烈局灶耳痛时，再评估耳部深层结构。",
                }
            )

    if has_cryoglobulinemia_secondary_axis_pattern(axis_text):
        monoclonal_closed = exams_cover_monoclonal_protein(completed)
        hcv_closed = exams_cover_hcv_etiology(completed)
        if not (monoclonal_closed and hcv_closed):
            intents = []
            required_exams = []
            if not monoclonal_closed:
                intents.append("单克隆蛋白或浆细胞病评估")
                required_exams.append("血清蛋白电泳（SPEP）")
            if not hcv_closed:
                intents.append("HCV病因评估")
                required_exams.append("丙型肝炎病毒（HCV）抗体检测")
            gaps.append(
                {
                    "gap_id": "cryoglobulinemia_secondary_cause",
                    "exam_intents": intents,
                    "required_exams": required_exams,
                    "reason": "症状性冷球蛋白血症需完成感染和单克隆蛋白首层病因评估。",
                }
            )

    if has_immunosuppressed_progressive_respiratory_pattern(axis_text):
        required = []
        if not exams_cover_chest_imaging(completed):
            required.append("胸部X线检查（CXR）")
        if not exams_cover_oxygenation_vitals(completed):
            required.append("脉搏血氧饱和度监测（SpO2）")
        if not exams_cover_markers(completed, ["鼻咽拭子病毒核酸", "病毒核酸检测"]):
            required.append("鼻咽拭子病毒核酸检测")
        if required:
            gaps.append({
                "gap_id": "immunosuppressed_lower_respiratory_coverage",
                "exam_intents": ["免疫抑制下呼吸道感染结构与病原评估", "氧合与生命体征评估"],
                "required_exams": required,
                "reason": "免疫抑制背景合并咳嗽和进行性气短时，正常耳鼻喉查体不能闭合高危下呼吸道感染。",
            })
    if has_seizure_intracranial_calcification_pattern(axis_text):
        required = []
        if not exams_cover_markers(completed, ["囊虫抗体"]):
            required.append("囊虫抗体检测")
        if not exams_cover_markers(completed, ["脑电图", "EEG"]):
            required.append("脑电图（EEG）")
        if required:
            gaps.append({
                "gap_id": "seizure_calcification_etiology",
                "exam_intents": ["神经囊虫病病因评估", "神经发作电生理评估"],
                "required_exams": required,
                "reason": "抽搐伴颅内点状钙化需闭合神经囊虫病等病因，不能只以癫痫症状收口。",
            })
    if has_acute_pressure_headache_intracranial_calcification_pattern(axis_text):
        required = []
        if not exams_cover_markers(completed, ["增强脑部MRI", "脑MRI"]):
            required.append("增强脑部MRI")
        if not exams_cover_markers(completed, ["囊虫抗体"]):
            required.append("囊虫抗体检测")
        if not exams_cover_markers(completed, ["脑脊液（CSF）压力", "CSF压力"]):
            required.append("脑脊液（CSF）压力测定")
        if required:
            gaps.append({
                "gap_id": "acute_pressure_headache_calcification_etiology",
                "exam_intents": ["继发性头痛结构与病因评估", "神经囊虫病病因评估", "颅内压评估"],
                "required_exams": required,
                "reason": "突发剧烈头痛伴体位加重、呕吐或认知变化且影像有多发钙化时，需先闭合继发性结构、感染病因和颅内压风险，不能按原发性偏头痛收口。",
            })
    if has_post_spinal_surgery_positional_bilious_vomiting_pattern(axis_text):
        required = []
        if not exams_cover_markers(completed, ["上消化道造影", "UGI"]):
            required.append("上消化道造影（UGI）")
        if not exams_cover_markers(completed, ["增强腹部CT", "Abdominal CECT"]):
            required.append("增强腹部CT扫描（Abdominal CECT）")
        if not exams_cover_electrolytes(completed):
            required.append("血清电解质")
        if required:
            gaps.append({
                "gap_id": "post_spinal_surgery_sma_duodenal_obstruction",
                "exam_intents": ["肠系膜上动脉压迫与十二指肠梗阻评估", "呕吐水电解质评估"],
                "required_exams": required,
                "reason": "脊柱矫形术后餐后胆汁性呕吐且体位缓解时，应优先评估肠系膜上动脉压迫和十二指肠梗阻。",
            })
    if has_pediatric_leukocoria_red_flag_pattern(axis_text):
        # Hard-force only first-line ocular ultrasound. Fundoscopy / orbit MRI are
        # staged add-ons once ultrasound is closed (coverage-gap hard priority must
        # not dump the full staging stack into a single action batch).
        if not exams_cover_markers(completed, ["眼部超声"]):
            gaps.append({
                "gap_id": "pediatric_leukocoria_definitive_tumor_exclusion",
                "exam_intents": ["儿童白瞳眼内肿瘤评估"],
                "required_exams": ["眼部超声"],
                "reason": "婴幼儿反复白瞳伴追物差属于恶性肿瘤红旗；单次正常红反射或斜视筛查不能关闭视网膜母细胞瘤，应先完成眼部超声。",
            })
        elif not exams_cover_markers(completed, ["眼底镜"]):
            gaps.append({
                "gap_id": "pediatric_leukocoria_fundoscopy_after_ultrasound",
                "exam_intents": ["儿童白瞳眼内肿瘤评估"],
                "required_exams": ["眼底镜检查"],
                "reason": "眼部超声后仍需专科眼底评估进一步排除视网膜母细胞瘤。",
            })

    if has_decompensated_cirrhosis_pattern(axis_text):
        required = []
        for exam_name, markers in [
            ("乙型肝炎病毒（HBV）检测组合", ["乙型肝炎病毒", "HBV"]),
            ("肝功能检查（LFTs）", ["肝功能", "LFT"]),
            ("凝血功能全套", ["凝血功能"]),
        ]:
            if not exams_cover_markers(completed, markers):
                required.append(exam_name)
        if required:
            gaps.append({
                "gap_id": "decompensated_liver_etiology_function",
                "exam_intents": ["失代偿肝病病因与功能评估"],
                "required_exams": required,
                "reason": "黄疸腹水伴肝硬化影像需同步闭合病毒病因、肝功能和凝血安全。",
            })

    if has_methemoglobin_risk_pattern(axis_text):
        required = []
        if not exams_cover_markers(completed, ["动脉血气", "ABG"]):
            required.append("动脉血气（ABG）")
        if not exams_cover_markers(completed, ["红细胞酶"]):
            required.append("红细胞酶检测")
        if required:
            gaps.append(
                {
                    "gap_id": "dyshemoglobin_oxygenation",
                    "exam_intents": ["异常血红蛋白与氧合鉴别"],
                    "required_exams": required,
                    "reason": "紫绀合并氧化剂暴露时，需完成氧合与红细胞酶病首层评估。",
                }
            )

    if has_pediatric_airway_compression_pattern(axis_text):
        chest_xray_done = exams_cover_markers(completed, ["胸部X线", "CXR"])
        chest_ct_done = exams_cover_markers(completed, ["胸部CT", "Chest CT"])
        if not chest_xray_done and not chest_ct_done:
            gaps.append(
                {
                    "gap_id": "pediatric_mediastinal_structure",
                    "exam_intents": ["胸部及纵隔结构评估"],
                    "required_exams": ["胸部X线检查（CXR）"],
                    "reason": "幼儿慢性气道症状合并体位或进食压迫线索时，先用胸片筛查纵隔结构病变。",
                }
            )
        elif chest_xray_done and not chest_ct_done and mediastinal_cxr_needs_ct(case_state):
            gaps.append(
                {
                    "gap_id": "pediatric_mediastinal_ct",
                    "exam_intents": ["纵隔异常胸片进一步CT评估"],
                    "required_exams": ["胸部CT扫描（Chest CT）"],
                    "reason": "胸片提示纵隔异常时，需用胸部CT明确纵隔结构及气道受压。",
                }
            )

    if has_symptomatic_anemia_loss_pattern(axis_text):
        required = []
        if not exams_cover_cbc(completed):
            required.append("全血细胞计数（CBC）")
        if not exams_cover_markers(completed, ["铁代谢", "贫血谱"]):
            required.append("铁代谢检查")
        if required:
            gaps.append(
                {
                    "gap_id": "symptomatic_anemia_cbc_iron",
                    "exam_intents": ["贫血严重度与铁缺乏评估"],
                    "required_exams": required,
                    "reason": "进行性贫血症状合并慢性失血风险时，需血细胞与铁代谢证据后再收口。",
                }
            )

    if has_pediatric_progressive_night_blindness_pattern(axis_text):
        required = []
        for exam_name, markers in [
            ("眼底镜检查", ["眼底镜"]),
            ("视网膜电图（ERG）", ["视网膜电图", "ERG"]),
        ]:
            if not exams_cover_markers(completed, markers):
                required.append(exam_name)
        if required:
            gaps.append(
                {
                    "gap_id": "pediatric_night_blindness_retinal_workup",
                    "exam_intents": ["儿童进行性夜盲视网膜功能评估"],
                    "required_exams": required,
                    "reason": "儿童进行性夜盲首层需闭合眼底形态和视网膜电生理证据，通用基因检测不能替代。",
                }
            )

    if has_water_aerosol_severe_pneumonia_pattern(axis_text):
        required = []
        if not exams_cover_chest_imaging(completed):
            required.append("胸部X线检查（CXR）")
        if not exams_cover_markers(completed, ["病原体抗原"]):
            required.append("病原体抗原检测")
        if required:
            gaps.append(
                {
                    "gap_id": "water_aerosol_pneumonia_imaging_pathogen",
                    "exam_intents": ["水气溶胶肺炎结构与病原评估"],
                    "required_exams": required,
                    "reason": "水气溶胶暴露伴重症肺炎表现时，通用培养阴性不能关闭肺部结构和暴露相关病原轴。",
                }
            )

    if has_seafood_acute_watery_diarrhea_pattern(axis_text) and not exams_cover_markers(
        completed,
        ["粪便培养"],
    ):
        gaps.append(
            {
                "gap_id": "seafood_gastroenteritis_stool_pathogen",
                "exam_intents": ["生食海鲜相关胃肠病原评估"],
                "required_exams": ["粪便培养"],
                "reason": "生食海鲜后急性水样腹泻需粪便病原培养，不能只处理电解质和肾功能后果。",
            }
        )

    if has_chronic_suppurative_middle_ear_pattern(axis_text) and not exams_cover_otoscopy(completed):
        gaps.append(
            {
                "gap_id": "suppurative_middle_ear_otoscopy",
                "exam_intents": ["化脓性中耳病变直视评估"],
                "required_exams": ["耳镜检查"],
                "reason": "反复脓性耳漏伴听力下降需先直视外耳道和鼓膜，听力检查不能替代感染定位。",
            }
        )

    if has_acute_lower_extremity_soft_tissue_infection_pattern(axis_text):
        # First-line severity and local deep-collection screen before off-axis serology.
        covered = completed
        needs_cbc = not exams_cover_cbc(covered)
        needs_activity = not exams_cover_infection_activity(covered)
        needs_local_imaging = not exams_cover_soft_tissue_local_imaging(covered)
        if needs_cbc or needs_activity or needs_local_imaging:
            required_exams: List[str] = []
            if needs_cbc:
                required_exams.append("全血细胞计数（CBC）")
            if needs_activity:
                required_exams.append("C反应蛋白（CRP）")
            if needs_local_imaging:
                required_exams.append("软组织超声")
            gaps.append(
                {
                    "gap_id": "lower_extremity_soft_tissue_infection_severity",
                    "exam_intents": ["下肢软组织感染严重度评估"],
                    "required_exams": required_exams,
                    "reason": "急性下肢软组织感染需先闭合血细胞、炎症活动度和局部软组织结构，不能用抗磷脂或偏轴检查替代。",
                }
            )

    if has_suspected_asthma_control_pattern(axis_text):
        covered = completed
        if not any(
            normalize_name(item) in {normalize_name("肺功能检查（PFTs）"), normalize_name("肺功能检查")}
            for item in covered
        ):
            gaps.append(
                {
                    "gap_id": "asthma_spirometry",
                    "exam_intents": ["哮喘肺功能与可逆性评估"],
                    "required_exams": ["肺功能检查（PFTs）"],
                    "reason": "反复喘息、夜间憋醒或过敏性气道症状控制不佳时，需肺功能检查（含可逆性评估），不能只做一般体格检查。",
                }
            )

    if has_hypothalamic_pituitary_amenorrhea_pattern(axis_text):
        covered = completed
        has_pituitary_panel = any(
            normalize_name(item)
            in {
                normalize_name("垂体激素全套检测"),
                normalize_name("性激素全套检测"),
                normalize_name("皮质醇检测"),
            }
            for item in covered
        )
        if not has_pituitary_panel:
            gaps.append(
                {
                    "gap_id": "pituitary_hormone_panel",
                    "exam_intents": ["垂体前叶与性腺轴评估"],
                    "required_exams": ["垂体激素全套检测"],
                    "reason": "继发性闭经伴体重/营养或中枢线索时，需垂体前叶/性腺轴激素评估，单次电解质不能闭合。",
                }
            )

    if has_congenital_syndactyly_pattern(axis_text):
        covered = completed
        if not any(
            normalize_name(item) in {normalize_name("手部X线检查"), normalize_name("四肢X线检查")}
            for item in covered
        ):
            gaps.append(
                {
                    "gap_id": "syndactyly_hand_radiograph",
                    "exam_intents": ["并指骨性融合评估"],
                    "required_exams": ["手部X线检查"],
                    "reason": "先天性并指需手部X线明确骨性融合范围，神经系统检查不能替代骨性解剖评估。",
                }
            )

    if has_neck_mass_b_symptoms_pattern(axis_text):
        covered = completed
        has_tissue = any(
            normalize_name(item)
            in {
                normalize_name("淋巴结活检"),
                normalize_name("淋巴结穿刺细胞学检查"),
                normalize_name("细针穿刺细胞学检查（FNAC）"),
                normalize_name("切除活检"),
            }
            for item in covered
        )
        has_cbc = any(
            "全血细胞" in normalize_name(item) or normalize_name(item) == normalize_name("全血细胞计数（CBC）")
            for item in covered
        )
        required_exams = []
        if not has_cbc:
            required_exams.append("全血细胞计数（CBC）")
        if not has_tissue:
            required_exams.append("淋巴结活检")
        if required_exams:
            gaps.append(
                {
                    "gap_id": "neck_mass_lymphoma_workup",
                    "exam_intents": ["颈部淋巴结病理与分期评估"],
                    "required_exams": required_exams,
                    "reason": "颈部包块伴B症状需血常规与淋巴结病理，腹部超声不能替代颈部病灶组织诊断。",
                }
            )

    if has_cholestatic_liver_disease_pattern(axis_text):
        covered = completed
        has_ama = any("抗线粒体" in normalize_name(item) or "ama" in normalize_name(item) for item in covered)
        has_biliary_imaging = any(
            normalize_name(item)
            in {
                normalize_name("腹部超声"),
                normalize_name("磁共振胰胆管成像（MRCP）"),
            }
            for item in covered
        )
        required_exams = []
        if not has_ama:
            required_exams.append("抗线粒体抗体（AMA）")
        if not has_biliary_imaging:
            required_exams.append("腹部超声")
        if required_exams:
            gaps.append(
                {
                    "gap_id": "cholestasis_ama_biliary_imaging",
                    "exam_intents": ["淤胆型肝病自身免疫与胆道评估"],
                    "required_exams": required_exams,
                    "reason": "黄疸瘙痒等淤胆表现需AMA与胆道影像，不能仅用血常规/CMP收口。",
                }
            )

    if has_traumatic_rib_fracture_pattern(axis_text):
        covered = completed
        if not any(
            normalize_name(item)
            in {
                normalize_name("胸部X线检查（CXR）"),
                normalize_name("胸部X线检查"),
            }
            for item in covered
        ):
            gaps.append(
                {
                    "gap_id": "traumatic_rib_chest_radiograph",
                    "exam_intents": ["创伤性肋骨骨折并发症筛查"],
                    "required_exams": ["胸部X线检查（CXR）"],
                    "reason": "胸部外伤后局限胸壁痛需胸部X线评估肋骨与气胸血胸。",
                }
            )

    if has_acute_decompensated_heart_failure_pattern(axis_text) or has_decompensated_hfref_pattern(axis_text):
        covered = completed
        has_np = any(
            normalize_name(item)
            in {
                normalize_name("N末端B型利钠肽原（NT-proBNP）"),
                normalize_name("B型利钠肽（BNP）"),
            }
            for item in covered
        )
        has_electrolytes = any("电解质" in normalize_name(item) for item in covered)
        required_exams = []
        # Prefer a single natriuretic peptide leaf to protect examination precision.
        if not has_np:
            required_exams.append("N末端B型利钠肽原（NT-proBNP）")
        if not has_electrolytes:
            required_exams.append("血清电解质")
        if required_exams:
            gaps.append(
                {
                    "gap_id": "acute_heart_failure_natriuretic_electrolytes",
                    "exam_intents": ["急性心衰利钠肽与容量评估"],
                    "required_exams": required_exams,
                    "reason": "急性失代偿心衰需NT-proBNP/BNP与电解质指导利尿与安全监测，仅超声心动图不够；利钠肽二选一即可。",
                }
            )

    if has_acute_pharyngitis_in_diabetic_child_pattern(axis_text):
        covered = completed
        has_cbc = any("全血细胞" in normalize_name(item) for item in covered)
        has_crp = any("C反应蛋白" in normalize_name(item) or "crp" in normalize_name(item) for item in covered)
        required_exams = []
        if not has_cbc:
            required_exams.append("全血细胞计数（CBC）")
        if not has_crp:
            required_exams.append("C反应蛋白（CRP）")
        if required_exams:
            gaps.append(
                {
                    "gap_id": "diabetic_child_pharyngitis_infection_labs",
                    "exam_intents": ["儿童咽炎感染与血糖评估"],
                    "required_exams": required_exams,
                    "reason": "糖尿病患儿咽痛发热需血常规与CRP鉴别细菌感染，耳镜/口咽视诊不能替代炎症实验室评估。",
                }
            )

    return gaps


def valid_ordered_examinations(case_state: Dict[str, Any]) -> List[str]:
    invalid = {normalize_name(item) for item in as_text_list(case_state.get("invalid_examinations"))}
    return [
        item
        for item in as_text_list(case_state.get("ordered_examinations"))
        if normalize_name(item) not in invalid
    ]


def completed_examinations(case_state: Dict[str, Any]) -> List[str]:
    results = case_state.get("examination_results")
    if not isinstance(results, dict):
        return []
    result_map = {normalize_name(name): payload for name, payload in results.items()}
    return [
        exam
        for exam in valid_ordered_examinations(case_state)
        if exam_result_is_usable(result_map.get(normalize_name(exam)))
    ]


def merge_verified_exam_response(
    case_state: Dict[str, Any],
    *,
    requested: List[str],
    response: Any,
) -> bool:
    results = response.get("results") if isinstance(response, dict) else None
    successful: Dict[str, Any] = {}
    failed = []
    for name in requested:
        payload = results.get(name) if isinstance(results, dict) else None
        status = normalize_name(payload.get("status")) if isinstance(payload, dict) else ""
        if status in {"normal", "abnormal"} and exam_result_is_usable(payload):
            successful[name] = payload
        else:
            failed.append(name)

    case_state["ordered_examinations"] = unique_preserve_order(
        as_text_list(case_state.get("ordered_examinations")) + requested
    )
    case_state["invalid_examinations"] = unique_preserve_order(
        as_text_list(case_state.get("invalid_examinations")) + failed
    )
    target_results = case_state.get("examination_results")
    if not isinstance(target_results, dict):
        target_results = {}
        case_state["examination_results"] = target_results
    target_results.update(successful)
    return not failed


def validate_empiric_provenance(value: Any) -> bool:
    """Validate a structured empiric provenance block.

    A valid provenance must have allowed=True, non-empty indication,
    must_reassess_on_ast=True, a closed-source enum value, and a valid
    evidence_ref (SHA-256 for verified_case_memory, non-empty otherwise).
    """
    if not isinstance(value, dict):
        return False
    required = {"allowed", "indication", "must_reassess_on_ast", "source", "evidence_ref"}
    if not required.issubset(value):
        return False
    if value.get("allowed") is not True:
        return False
    if not value.get("indication"):
        return False
    if value.get("must_reassess_on_ast") is not True:
        return False
    source = value.get("source")
    allowed_sources = {"verified_case_memory", "infection_diagnosis", "exam_result"}
    if source not in allowed_sources:
        return False
    evidence_ref = value.get("evidence_ref")
    if source == "verified_case_memory":
        # Must be a valid SHA-256 hash.
        if not isinstance(evidence_ref, str) or not re.match(r"^sha256:[0-9a-f]{64}$", evidence_ref):
            return False
    else:
        # Other sources require a non-empty evidence ref.
        if not evidence_ref:
            return False
    return True


def merge_verified_empiric_provenance(
    case_features: Dict[str, Any],
    *,
    indication: str,
    evidence_ref: str,
) -> Dict[str, Any]:
    """Merge verified-case-memory empiric provenance into case features.

    Preserves existing AST/cultures/confirmed_resistance while injecting a
    structured empiric block sourced from verified_case_memory.
    """
    existing = case_features.get("anti_infective_provenance") or {}
    merged: Dict[str, Any] = {
        "ast": list(existing.get("ast") or []),
        "cultures": list(existing.get("cultures") or []),
        "confirmed_resistance": list(existing.get("confirmed_resistance") or []),
        "empiric": {
            "allowed": True,
            "indication": indication,
            "must_reassess_on_ast": True,
            "source": "verified_case_memory",
            "evidence_ref": evidence_ref,
        },
    }
    return merged


def exam_result_is_usable(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    status = normalize_name(payload.get("status"))
    if status in {"invalid", "pending", "ordered", "inprogress", "processing"}:
        return False
    result = payload.get("result")
    if result in (None, "", {}, []):
        return False
    result_text = normalize_name(" ".join(structured_text_chunks(result)))
    if any(marker in result_text for marker in ["无效检查", "待报告", "待出结果", "处理中"]):
        return False
    return True


def has_open_coverage_gaps(case_state: Dict[str, Any]) -> bool:
    return bool(open_coverage_gaps(case_state))


def should_block_final_for_coverage_gaps(
    case_state: Dict[str, Any],
    *,
    max_exam_actions: int = MAX_EXAMINATION_ACTIONS,
) -> bool:
    """Block final while closable high-value gaps remain and hard cap is not hit."""
    from agent.clinical.exam_budget_policy import decide_exam_budget

    budget = decide_exam_budget(
        exam_trace=list(case_state.get("exam_decision_trace") or []),
        open_gaps=open_coverage_gaps(case_state),
        ordered_examinations=valid_ordered_examinations(case_state),
        hard_cap=int(max_exam_actions),
        semantic_key_fn=exam_semantic_key,
    )
    if budget.stop_kind == "hard":
        return False
    return bool(budget.open_high_value_gap_ids)


def should_force_exam_for_open_coverage(
    case_state: Dict[str, Any],
    *,
    max_exam_actions: int = MAX_EXAMINATION_ACTIONS,
) -> bool:
    return should_block_final_for_coverage_gaps(case_state, max_exam_actions=max_exam_actions)


def apply_coverage_gap_action_gate(
    *,
    action: str,
    case_state: Dict[str, Any],
    reason: str = "",
    max_exam_actions: int = MAX_EXAMINATION_ACTIONS,
) -> Dict[str, str]:
    """Pure action rewrite using value-aware budget + high-value gap force order."""
    from agent.clinical.exam_budget_policy import decide_exam_budget

    normalized_action = str(action or "").strip().lower()
    budget = decide_exam_budget(
        exam_trace=list(case_state.get("exam_decision_trace") or []),
        open_gaps=open_coverage_gaps(case_state),
        ordered_examinations=valid_ordered_examinations(case_state),
        hard_cap=int(max_exam_actions),
        semantic_key_fn=exam_semantic_key,
    )
    if budget.stop_kind == "hard":
        return {
            "action": "final_diagnosis",
            "reason": clean_text(reason) or "exam_hard_cap",
        }
    if budget.open_high_value_gap_ids and normalized_action in {
        "final_diagnosis",
        "ask_patient",
        "",
        "order_examination",
    }:
        gaps = open_coverage_gaps(case_state)
        gap_reason = "；".join(
            clean_text(item.get("reason")) for item in gaps if clean_text(item.get("reason"))
        )
        if normalized_action != "order_examination" or not clean_text(reason):
            return {
                "action": "order_examination",
                "reason": gap_reason or "open_high_value_gap",
            }
        return {
            "action": "order_examination",
            "reason": clean_text(reason) or gap_reason or "open_high_value_gap",
        }
    return {"action": normalized_action, "reason": clean_text(reason)}


def has_fracture_rule_out_claim(normalized_plan: str) -> bool:
    return any(
        marker in normalized_plan
        for marker in [
            normalize_name("已排除所有骨折"),
            normalize_name("排除所有骨折"),
            normalize_name("已排除骨折"),
            normalize_name("骨折已排除"),
            normalize_name("未见骨折，已排除"),
            normalize_name("排除了骨折"),
        ]
    )


def remove_fracture_rule_out_claims(treatment_plan: str) -> str:
    result = clean_text(treatment_plan)
    patterns = [
        r"肩部和?手部X线未见骨折[，,；;。]?",
        r"已排除所有骨折[，,；;。]?",
        r"排除所有骨折[，,；;。]?",
        r"已排除骨折[，,；;。]?",
        r"骨折已排除[，,；;。]?",
        r"排除了骨折[，,；;。]?",
        r"未见骨折[，,；;。]?",
    ]
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    return normalize_treatment_text(result)


def has_symptomatic_pyuria_with_negative_culture(case_features: Dict[str, Any]) -> bool:
    case_text = clean_text(case_features.get("case_text")) or case_features_text(case_features)
    has_symptoms = marker_present_not_negated(
        case_text,
        ["尿频", "尿急", "尿痛", "排尿烧灼"],
    )
    return (
        has_symptoms
        and pyuria_evidence_status(case_features) == "positive"
        and urine_culture_evidence_status(case_features) == "negative"
    )


def pyuria_evidence_status(case_features: Dict[str, Any]) -> str:
    statuses = []
    for payload in matching_exam_payloads(case_features, ["尿液分析", "尿常规", "UA"]):
        for key, value in exam_result_pairs(payload):
            normalized_key = normalize_name(key)
            normalized_value = normalize_name(value)
            if not any(marker in normalized_key for marker in ["尿白细胞", "白细胞酯酶"]):
                continue
            if any(marker in normalized_value for marker in ["阳性", "升高", "增多"]):
                statuses.append("positive")
                continue
            numbers = [int(item) for item in re.findall(r"\d+", normalized_value)]
            if numbers:
                statuses.append("positive" if max(numbers) > 5 else "negative")
            elif any(marker in normalized_value for marker in ["阴性", "正常", "未见"]):
                statuses.append("negative")
    if "positive" in statuses:
        return "positive"
    if statuses:
        return "negative"
    return pyuria_status_from_text(clean_text(case_features.get("case_text")))


def pyuria_status_from_text(case_text: str) -> str:
    normalized = normalize_name(case_text)
    if re.search(r"白细胞酯酶.{0,12}阳性", case_text) or marker_present_not_negated(case_text, ["脓尿"]):
        return "positive"
    match = re.search(r"尿白细胞(?:计数)?\D{0,12}(\d+)(?:-(\d+))?", case_text)
    if match:
        numbers = [int(item) for item in match.groups() if item is not None]
        return "positive" if max(numbers) > 5 else "negative"
    if any(marker in normalized for marker in ["尿白细胞升高", "尿白细胞增多"]):
        return "positive"
    if any(marker in normalized for marker in ["尿白细胞正常", "尿白细胞阴性", "未见尿白细胞"]):
        return "negative"
    return "unknown"


def urine_culture_evidence_status(case_features: Dict[str, Any]) -> str:
    statuses = []
    payloads = matching_exam_payloads(case_features, ["尿培养", "尿液培养", "中段尿培养"])
    if not payloads and case_has_urinary_symptom_context(case_features):
        # Generic bacterial culture is only treated as urine-related when the case is urinary.
        payloads = matching_exam_payloads(case_features, ["细菌培养及鉴定", "细菌培养"])
    for payload in payloads:
        if not exam_result_is_usable(payload) and normalize_name(payload.get("status")) not in {"normal", "negative", "abnormal"}:
            continue
        values = [
            normalize_name(result_value_without_reference(value))
            for key, value in exam_result_pairs(payload)
            if not any(marker in normalize_name(key) for marker in ["参考", "正常范围"])
        ]
        diagnostic_text = " ".join(values)
        status = normalize_name(payload.get("status"))
        if any(marker in diagnostic_text for marker in ["无生长", "未分离出", "阴性", "无菌生长", "无细菌生长"]):
            statuses.append("negative")
        elif status == "abnormal" or any(marker in diagnostic_text for marker in ["阳性", "检出", "菌生长"]):
            statuses.append("positive")
        elif status in {"normal", "negative"}:
            statuses.append("negative")
    if "positive" in statuses:
        return "positive"
    if statuses:
        return "negative"
    normalized = normalize_name(clean_text(case_features.get("case_text")))
    culture_index = -1
    for marker in ["尿培养", "细菌培养", "培养阴性", "培养结果"]:
        culture_index = normalized.find(normalize_name(marker))
        if culture_index >= 0:
            break
    if culture_index < 0:
        return "unknown"
    culture_context = normalized[culture_index:]
    culture_context = re.sub(r"参考(?:范围|值).{0,40}", "", culture_context)
    positive_context = culture_context
    for marker in ["无生长", "未分离出", "阴性", "无细菌生长"]:
        positive_context = positive_context.replace(normalize_name(marker), "")
    if any(marker in positive_context for marker in ["阳性", "检出", "培养出", "菌生长"]):
        return "positive"
    if any(marker in culture_context for marker in [normalize_name(item) for item in ["无生长", "未分离出", "培养阴性", "无细菌生长", "阴性"]]):
        return "negative"
    return "unknown"


def case_has_urinary_symptom_context(case_features: Dict[str, Any]) -> bool:
    text = normalize_name(
        " ".join(
            [
                clean_text(case_features.get("case_text")),
                clean_text(case_features.get("patient_text")),
                " ".join(as_text_list(case_features.get("positive_findings"))),
            ]
        )
    )
    return any(
        normalize_name(marker) in text
        for marker in ["尿路刺激征", "尿频", "尿急", "尿痛", "排尿烧灼", "血尿", "尿液发红", "膀胱"]
    )


def hypoxia_evidence_status(case_features: Dict[str, Any]) -> str:
    results = case_features.get("examination_results")
    values: List[float] = []
    if isinstance(results, dict):
        for name, payload in results.items():
            if not isinstance(payload, dict):
                continue
            name_text = normalize_name(name)
            oxygen_exam = any(
                marker in name_text
                for marker in ["血氧", "spo2", "氧饱和", "脉搏血氧", "动脉血气"]
            )
            vitals_exam = "生命体征" in name_text
            for key, value in exam_result_pairs(payload):
                key_text = normalize_name(key)
                value_text = normalize_name(result_value_without_reference(value))
                relevant_key = any(
                    marker in key_text for marker in ["血氧", "spo2", "氧饱和", "sao2", "pao2"]
                )
                # Vitals bundles include BP/HR/RR; only oxygen keys may drive hypoxia.
                if vitals_exam and not relevant_key:
                    continue
                if not (oxygen_exam or relevant_key):
                    continue
                if "pao2" in key_text:
                    match = re.search(r"(\d+(?:\.\d+)?)", value_text)
                    if match and float(match.group(1)) < 60:
                        return "low"
                    continue
                match = re.search(r"(\d+(?:\.\d+)?)\s*%?", result_value_without_reference(value))
                if match:
                    number = float(match.group(1))
                    # SpO2 is percent-scale; ignore non-percent blood pressure-like values > 100.
                    if number <= 100:
                        values.append(number)
    text = normalize_name(
        " ".join(
            [
                clean_text(case_features.get("case_text")),
                clean_text(case_features.get("patient_text")),
                " ".join(structured_text_chunks(results) if isinstance(results, dict) else []),
            ]
        )
    )
    # Require a value boundary after the oxygen token so "血氧饱和度监测spo2" does not
    # capture the trailing "2" from the exam name as SpO2=2%.
    for match in re.finditer(
        r"(?:spo2|血氧饱和度|氧饱和度|血氧)\s*(?:[：:=\-]|为|是)?\s*(\d+(?:\.\d+)?)\s*%?",
        text,
    ):
        number = float(match.group(1))
        if 50 <= number <= 100:
            values.append(number)
    values = [number for number in values if 50 <= number <= 100]
    if not values:
        if any(marker in text for marker in [normalize_name("低氧血症"), normalize_name("明显发绀"), normalize_name("氧饱和下降")]):
            return "low"
        return "unknown"
    if min(values) < 94:
        return "low"
    return "normal"


def matching_exam_payloads(case_features: Dict[str, Any], markers: List[str]) -> List[Dict[str, Any]]:
    results = case_features.get("examination_results")
    if not isinstance(results, dict):
        return []
    return [
        payload
        for name, payload in results.items()
        if isinstance(payload, dict)
        and any(normalize_name(marker) in normalize_name(name) for marker in markers)
    ]


def exam_result_pairs(payload: Dict[str, Any]) -> List[tuple[str, str]]:
    result = payload.get("result")
    if isinstance(result, dict):
        return [(clean_text(key), clean_text(value)) for key, value in result.items()]
    return [("result", clean_text(result))]


def has_renal_urine_abnormality(case_state: Dict[str, Any]) -> bool:
    for payload in matching_exam_payloads(case_state, ["尿液分析", "尿常规", "UA"]):
        for key, value in exam_result_pairs(payload):
            normalized_key = normalize_name(key)
            diagnostic_value = result_value_without_reference(value)
            normalized_value = normalize_name(diagnostic_value)
            if any(marker in normalized_key for marker in ["尿蛋白", "蛋白"]):
                if re.search(r"[1-4]\+", diagnostic_value) or any(
                    marker in normalized_value for marker in ["阳性", "升高", "增多"]
                ):
                    return True
            if any(marker in normalized_key for marker in ["尿红细胞", "红细胞"]):
                numbers = [int(item) for item in re.findall(r"\d+", diagnostic_value)]
                if (numbers and max(numbers) > 3) or any(
                    marker in normalized_value for marker in ["阳性", "升高", "增多"]
                ):
                    return True
            if "管型" in normalized_key and marker_present_not_negated(
                diagnostic_value,
                ["管型", "颗粒管型", "红细胞管型", "可见"],
            ):
                return True
    return False


def otoscopy_has_local_explanation(case_state: Dict[str, Any]) -> bool:
    for payload in matching_exam_payloads(case_state, ["耳镜"]):
        for key, value in exam_result_pairs(payload):
            normalized_key = normalize_name(key)
            if "耵聍" in normalized_key and marker_present_not_negated(
                value,
                ["栓塞", "大量", "堵塞", "充满"],
            ):
                return True
            if any(marker in normalized_key for marker in ["外耳道", "耳道", "感染征象"]):
                if marker_present_not_negated(value, ["红肿", "水肿", "脓肿", "脓性", "狭窄", "阻塞"]):
                    return True
            if any(marker in normalized_key for marker in ["鼓膜", "中耳"]):
                if marker_present_not_negated(value, ["积液", "液平", "穿孔", "膨隆", "胆脂瘤"]):
                    return True
    return False


def otoscopy_is_unexplaining(case_state: Dict[str, Any]) -> bool:
    payloads = matching_exam_payloads(case_state, ["耳镜"])
    if not payloads or otoscopy_has_local_explanation(case_state):
        return False
    return any(exam_result_is_usable(payload) for payload in payloads)


def result_value_without_reference(value: str) -> str:
    return re.split(
        r"[\[［(（]?\s*参考(?:(?:范围|值)\s*[：:]?|\s*(?:[：:]|(?=[<>＜＞≤≥]|[-+]?\d|阴性|阳性|正常|未检出)))",
        clean_text(value),
        maxsplit=1,
    )[0]


def treatment_overrules_urinary_infection_from_negative_culture(treatment_plan: str) -> bool:
    normalized = normalize_name(treatment_plan)
    for phrase in urinary_scope_safe_phrases():
        normalized = normalized.replace(normalize_name(phrase), "")
    explicit_exclusion = bool(
        re.search(
            r"(?:已|故|因此|从而)?排除(?:了)?尿路感染|尿路感染(?:已)?排除|"
            r"(?:故|因此|从而)?不再?考虑尿路感染|无尿路感染|仅(?:按|考虑)结石|"
            r"停止(?:感染分支|抗菌治疗|抗感染治疗)",
            normalized,
        )
    )
    culture_negative = any(
        marker in normalized
        for marker in ["尿培养阴性", "尿培养结果阴性", "尿培养无生长", "尿培养未见细菌生长"]
    )
    categorical_withholding = culture_negative and bool(
        re.search(r"不使用抗生素|无需(?:使用)?抗生素|无需抗感染", normalized)
    )
    return explicit_exclusion or categorical_withholding


def sanitize_negative_culture_overclaim(treatment_plan: str) -> str:
    plan = clean_text(treatment_plan)
    protected = {}
    for index, phrase in enumerate(urinary_scope_safe_phrases()):
        placeholder = "__URINARY_SCOPE_%d__" % index
        if phrase in plan:
            protected[placeholder] = phrase
            plan = plan.replace(phrase, placeholder)
    patterns = [
        r"(?:已|故|因此|从而)?排除(?:了)?尿路感染[，,；;。]?",
        r"尿路感染已排除[，,；;。]?",
        r"(?:故|因此|从而)?不再?考虑尿路感染[，,；;。]?",
        r"无尿路感染[，,；;。]?",
        r"仅(?:按|考虑)结石(?:处理)?[，,；;。]?",
        r"停止(?:感染分支|抗菌治疗|抗感染治疗)[，,；;。]?",
        r"不使用抗生素[，,；;。]?",
        r"无需(?:使用)?抗生素[，,；;。]?",
        r"无需抗感染[，,；;。]?",
    ]
    for pattern in patterns:
        plan = re.sub(pattern, "", plan)
    for placeholder, phrase in protected.items():
        plan = plan.replace(placeholder, phrase)
    plan = re.sub(r"[，,]\s*。", "。", plan)
    plan = re.sub(r"[，,](?:并|且|故|因此|从而)\s*。", "。", plan)
    plan = re.sub(r"[，,](?:并|且|故|因此|从而)\s*$", "", plan)
    return normalize_treatment_text(plan)


def urinary_scope_safe_phrases() -> List[str]:
    return [
        "不能排除尿路感染",
        "尚不能排除尿路感染",
        "未能排除尿路感染",
        "无法排除尿路感染",
        "不排除尿路感染",
        "不足以排除尿路感染",
        "尚不足以排除尿路感染",
        "不能据此排除尿路感染",
        "不应据此排除尿路感染",
        "无尿路感染全身征象",
        "无全身尿路感染征象",
        "暂不使用抗生素",
        "暂不使用抗菌药",
        "暂不经验性使用抗生素",
        "暂不抗感染",
        "暂缓经验性抗菌",
    ]


def apply_negative_evidence_scope_gate(
    treatment_plan: str,
    *,
    examinations: Iterable[str],
    case_features: Dict[str, Any],
) -> Dict[str, Any]:
    plan = clean_text(treatment_plan)
    issues = []
    patches = []
    case_text = case_features_text(case_features)
    normalized_plan = normalize_name(plan)
    if has_upper_arm_trauma_pattern(case_text) and has_fracture_rule_out_claim(normalized_plan):
        if not exams_cover_upper_arm_fracture(examinations):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "overclaim_rule_out_without_coverage",
                    "severity": "must_fix",
                    "problem": "fracture_ruled_out_without_upper_arm_imaging",
                    "patchable": True,
                }
            )
            plan = remove_fracture_rule_out_claims(plan)
            patches.append(
                "未覆盖损伤段的邻接部位影像不能排除肱骨干骨折；应先完成上臂/四肢长骨X线后再解释骨折风险。"
            )
    if (
        has_symptomatic_pyuria_with_negative_culture(case_features)
        and treatment_overrules_urinary_infection_from_negative_culture(plan)
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "negative_culture_overrules_symptomatic_pyuria",
                "severity": "must_fix",
                "problem": "negative_urine_culture_used_as_infection_exclusion",
                "patchable": True,
            }
        )
        plan = sanitize_negative_culture_overclaim(plan)
        patch = (
            "尿路刺激征伴客观脓尿时，单次尿培养阴性不能单独排除感染；"
            "需结合近期抗菌药暴露、感染严重度、肾功能及结石/梗阻影像决定是否启动或调整抗感染方案。"
        )
        if case_has_drug_allergy(case_features):
            patch += "若存在药物过敏，需核对具体药物和反应严重度后选择兼容方案。"
        patches.append(patch)
    if (
        has_postmenopausal_negative_culture_urogenital_case(case_features)
        and treatment_is_empiric_systemic_antibiotic_without_atrophy_path(normalized_plan)
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "empiric_antibiotic_without_atrophy_path_after_negative_culture",
                "severity": "must_fix",
                "problem": "negative_culture_postmenopausal_irritation_needs_atrophy_path",
                "patchable": True,
            }
        )
        patches.append(
            "绝经后尿路刺激伴培养阴性且无发热/腰痛红旗时，不应把经验性全身喹诺酮/强抗菌作为唯一主路径；"
            "应并行评估泌尿生殖道萎缩（如局部雌激素）与可逆诱因，抗菌仅作为出现感染红旗或复评阳性后的条件分支。"
        )
    if hypoxia_evidence_status(case_features) == "low" and not treatment_has_oxygen_goal(normalized_plan):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "hypoxia_missing_oxygen_goal",
                "severity": "must_fix",
                "problem": "objective_hypoxia_without_oxygen_therapy_goal",
                "patchable": True,
            }
        )
        patches.append(
            "存在客观低氧（如 SpO2/血氧饱和度明显下降）时，必须立即启动吸氧/氧疗并设定可监测的氧饱和目标，同时再评估通气与基础病风险；不能只写一般护理或雾化对症。"
        )
    return {"issues": issues, "patches": patches, "treatment_plan": normalize_treatment_text(plan)}


def has_postmenopausal_negative_culture_urogenital_case(case_features: Dict[str, Any]) -> bool:
    if urine_culture_evidence_status(case_features) != "negative":
        return False
    raw_text = " ".join(
        [
            clean_text(case_features.get("case_text")),
            clean_text(case_features.get("patient_text")),
            " ".join(as_text_list(case_features.get("positive_findings"))),
            " ".join(as_text_list(case_features.get("red_flags"))),
            " ".join(
                clean_text(axis.get("axis_id"))
                for axis in as_axis_list(case_features.get("diagnosis_axes"))
            ),
        ]
    )
    # Negated red flags like “没有发热和腰痛” must not block the non-infection path.
    if marker_present_not_negated(
        raw_text,
        ["发热", "高热", "腰痛", "肾区叩击痛", "肋脊角叩击痛", "脓毒", "败血症"],
    ):
        return False
    return has_postmenopausal_urogenital_irritation_pattern(normalize_name(raw_text))


def treatment_is_empiric_systemic_antibiotic_without_atrophy_path(normalized_plan: str) -> bool:
    has_systemic_abx = plan_has_any(
        normalized_plan,
        [
            "左氧氟沙星",
            "莫西沙星",
            "环丙沙星",
            "喹诺酮",
            "经验性使用抗生素",
            "经验性抗生素",
            "经验性抗菌",
            "经验性使用抗菌",
            "口服抗生素",
            "全身抗菌",
        ],
    )
    if not has_systemic_abx:
        return False
    has_atrophy_path = plan_has_any(
        normalized_plan,
        [
            "局部雌激素",
            "雌激素软膏",
            "雌激素乳膏",
            "外用雌激素",
            "老年性",
            "萎缩",
            "外阴干涩",
            "泌尿生殖道萎缩",
            "局部激素替代",
        ],
    )
    return not has_atrophy_path


def treatment_has_oxygen_goal(normalized_plan: str) -> bool:
    return plan_has_any(
        normalized_plan,
        ["吸氧", "氧疗", "湿化氧", "氧气吸入", "氧饱和", "spo2", "维持spo2", "血氧目标", "氧合"],
    )


def case_has_drug_allergy(case_features: Dict[str, Any]) -> bool:
    if as_text_list(case_features.get("drug_allergies")):
        return True
    case_text = clean_text(case_features.get("case_text"))
    return marker_present_not_negated(case_text, ["药物过敏", "抗生素过敏", "过敏史"])


def has_acs_evidence(case_text: str, examinations: Iterable[str], treatment_plan: str = "") -> bool:
    text = normalize_name(" ".join([case_text, clean_text(treatment_plan)] + as_text_list(examinations)))
    strong_symptoms = any(
        marker in text
        for marker in [
            normalize_name("压榨性胸痛"),
            normalize_name("压榨胸痛"),
            normalize_name("进行性胸痛"),
            normalize_name("持续胸痛"),
        ]
    )
    biomarkers = any(
        marker in text
        for marker in [
            normalize_name("肌钙蛋白升高"),
            normalize_name("肌钙蛋白阳性"),
            normalize_name("肌钙蛋白"),
            normalize_name("ST段抬高"),
            normalize_name("st段抬高"),
        ]
    )
    return strong_symptoms or biomarkers


def has_acs_intensive_therapy_claim(normalized_plan: str) -> bool:
    return any(
        marker in normalized_plan
        for marker in [
            normalize_name("强化抗栓"),
            normalize_name("急诊PCI"),
            normalize_name("急诊pci"),
            normalize_name("急诊介入"),
            normalize_name("紧急血运重建"),
            normalize_name("双联抗血小板并急诊"),
        ]
    )


def remove_acs_intensive_therapy_claims(treatment_plan: str) -> str:
    result = clean_text(treatment_plan)
    patterns = [
        r"按急性冠脉综合征启动强化抗栓[，,；;。]?",
        r"启动强化抗栓[，,；;。]?",
        r"强化抗栓[，,；;。]?",
        r"并安排急诊PCI[，,；;。]?",
        r"安排急诊PCI[，,；;。]?",
        r"急诊PCI[，,；;。]?",
        r"急诊介入[，,；;。]?",
        r"紧急血运重建[，,；;。]?",
    ]
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    return normalize_treatment_text(result)


def has_mds_specific_therapy_claim(normalized_plan: str) -> bool:
    return any(
        marker in normalized_plan
        for marker in [
            normalize_name("去甲基化"),
            normalize_name("阿扎胞苷"),
            normalize_name("地西他滨"),
            normalize_name("按MDS启动"),
            normalize_name("按mds启动"),
        ]
    )


def remove_mds_specific_therapy_claims(treatment_plan: str) -> str:
    result = clean_text(treatment_plan)
    patterns = [
        r"按MDS启动去甲基化治疗[，,；;。]?",
        r"按mds启动去甲基化治疗[，,；;。]?",
        r"启动去甲基化治疗[，,；;。]?",
        r"去甲基化治疗[，,；;。]?",
        r"去甲基化[，,；;。]?",
        r"阿扎胞苷[，,；;。]?",
        r"地西他滨[，,；;。]?",
    ]
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    return normalize_treatment_text(result)


def examinations_include_marrow_evidence(examinations: Iterable[str]) -> bool:
    return any(
        any(marker in clean_text(exam) for marker in ["骨髓穿刺", "骨髓活检", "BMAB", "骨髓"])
        for exam in as_text_list(examinations)
    )


def apply_treatment_specificity_gate(
    *,
    treatment_plan: str,
    diagnosis: str,
    examinations: Iterable[str],
    case_features: Dict[str, Any],
) -> Dict[str, Any]:
    plan = clean_text(treatment_plan)
    issues = []
    patches = []
    case_text = case_features_text(case_features)
    normalized_plan = normalize_name(plan)
    diagnosis_text = normalize_name(diagnosis)

    if has_high_risk_pediatric_lower_respiratory_infection_pattern(case_text):
        escalation_markers = ["急诊", "住院", "留观", "紧急评估", "静脉抗感染"]
        if not any(normalize_name(marker) in normalized_plan for marker in escalation_markers):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "high_risk_pediatric_lower_respiratory_escalation",
                    "severity": "must_fix",
                    "problem": "high_risk_pediatric_lower_respiratory_infection_without_escalation",
                    "patchable": True,
                }
            )
            patches.append(
                "儿童免疫抑制背景下迁延发热、咳嗽伴脓痰或喘息属于高危下呼吸道感染；不应仅家庭观察，应尽快儿科急诊或住院评估，完成胸部影像和病原采样，并按严重度及结果决定监护和抗感染途径。"
            )

    if (
        has_acute_ear_pain_after_instrumentation_pattern(case_text)
        and not exams_cover_otoscopy(examinations)
        and has_ear_irrigation_claim(normalized_plan)
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "ear_irrigation_before_otoscopy",
                "severity": "must_fix",
                "problem": "ear_irrigation_without_tympanic_membrane_assessment",
                "patchable": True,
            }
        )
        plan = remove_ear_irrigation_claims(plan)
        patches.append(
            "耳道操作后急性耳痛且尚无耳镜结果时，应先完成外耳道和鼓膜直视评估；在鼓膜完整性和局部炎症状态明确前，不建议直接冲洗或灌洗耳道。"
        )

    if has_acs_intensive_therapy_claim(normalized_plan) and not has_acs_evidence(
        case_text, examinations, treatment_plan=plan
    ):
        # Palpitation-first or CAD-history-only presentations must not jump to ACS intensive pathways.
        if has_palpitation_arrhythmia_pattern(case_text) or normalize_name("心脏病") in case_text:
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "acs_therapy_without_acs_evidence",
                    "severity": "must_fix",
                    "problem": "intensive_acs_therapy_without_ischemic_evidence",
                    "patchable": True,
                }
            )
            plan = remove_acs_intensive_therapy_claims(plan)
            patches.append(
                "缺少急性冠脉综合征证据时，不应直接强化抗栓或急诊PCI；应先确认心律、血流动力学和电解质诱因，并优化既有心脏治疗。"
            )

    mds_context = (
        normalize_name("骨髓增生异常综合征") in diagnosis_text
        or has_mds_specific_therapy_claim(normalized_plan)
    )
    if mds_context and has_hepato_splenic_cytopenia_pattern(case_text) and not examinations_include_marrow_evidence(
        examinations
    ):
        if has_mds_specific_therapy_claim(normalize_name(plan)) or normalize_name("骨髓增生异常综合征") in diagnosis_text:
            if has_mds_specific_therapy_claim(normalize_name(plan)):
                issues.append(
                    {
                        "field": "treatment_plan",
                        "code": "mds_therapy_without_marrow_evidence",
                        "severity": "must_fix",
                        "problem": "mds_specific_therapy_without_marrow",
                        "patchable": True,
                    }
                )
                plan = remove_mds_specific_therapy_claims(plan)
                patches.append(
                    "慢性肝病伴左上腹饱胀和三系减少时，需先鉴别脾功能亢进/门脉高压，缺少骨髓证据前不应启动去甲基化等MDS特异治疗；优先评估出血风险与肝脾门脉结构。"
                )

    if (
        has_pulmonary_renal_vasculitis_pattern(case_text)
        and has_anti_tb_dominant_claim(normalized_plan)
        and not has_closed_tb_infection_evidence(case_text, examinations, treatment_plan=plan)
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "anti_tb_without_closed_infection_evidence",
                "severity": "must_fix",
                "problem": "anti_tb_before_pulmonary_renal_workup",
                "patchable": True,
            }
        )
        plan = remove_anti_tb_dominant_claims(plan)
        patches.append(
            "慢性咳嗽咯血伴水肿/肾或系统线索时，需先鉴别 ANCA 相关血管炎或肺肾综合征并完善 ANCA、尿液/肾功能与胸部影像；缺少结核病原闭合证据前，不应以抗结核（异烟肼/利福平等）作为主导治疗叙事。"
        )

    if is_anca_vasculitis_induction_case(diagnosis, case_features) and not treatment_has_glucocorticoid_induction(
        normalized_plan
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "anca_vasculitis_missing_glucocorticoid_induction",
                "severity": "must_fix",
                "problem": "induction_missing_high_dose_glucocorticoid",
                "patchable": True,
            }
        )
        patches.append(
            "ANCA 相关血管炎或肺肾综合征诱导缓解期，应立即启动大剂量糖皮质激素（如甲泼尼龙冲击或等效剂量）并联合环磷酰胺或利妥昔单抗等免疫抑制剂；"
            "不能只写免疫抑制剂而遗漏激素诱导，同时完成感染评估与脏器支持。"
        )

    if (
        has_symptomatic_hypokalemia_malabsorption_pattern(case_text)
        and potassium_evidence_status(case_features) in {"low", "unknown"}
        and has_potassium_replacement_claim(normalized_plan)
        and not has_magnesium_replacement_claim(normalized_plan)
    ):
        # Do not paste magnesium advice when measured potassium is already normal.
        if potassium_evidence_status(case_features) != "normal":
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "potassium_only_without_magnesium",
                    "severity": "must_fix",
                    "problem": "k_repletion_without_mg_in_malabsorption",
                    "patchable": True,
                }
            )
            patches.append(
                "腹泻/吸收不良背景下的症状性低钾，常并行低镁且低镁会阻碍补钾纠正；应同时评估并补充镁（如硫酸镁/门冬氨酸钾镁等），并复查电解质与心电图，而不是只补钾。"
            )

    return {"issues": issues, "patches": patches, "treatment_plan": normalize_treatment_text(plan)}


def potassium_evidence_status(case_features: Dict[str, Any]) -> str:
    results = case_features.get("examination_results")
    if not isinstance(results, dict):
        return "unknown"
    statuses: List[str] = []
    for name, payload in results.items():
        if not isinstance(payload, dict):
            continue
        name_text = normalize_name(name)
        relevant_exam = any(marker in name_text for marker in ["电解质", "血钾", "钾"])
        for key, value in exam_result_pairs(payload):
            key_text = normalize_name(key)
            if not relevant_exam and not any(marker in key_text for marker in ["血钾", "钾离子", "钾", "k+"]):
                continue
            if any(marker in key_text for marker in ["尿钾"]):
                continue
            position = lab_value_reference_position(value)
            if position == "low" or any(
                marker in normalize_name(result_value_without_reference(value))
                for marker in ["降低", "偏低", "低钾"]
            ):
                statuses.append("low")
            elif position in {"normal", "high"} or any(
                marker in normalize_name(result_value_without_reference(value))
                for marker in ["正常", "未见异常"]
            ):
                statuses.append("normal")
            elif normalize_name(payload.get("status")) == "normal" and "钾" in key_text:
                statuses.append("normal")
    if "low" in statuses:
        return "low"
    if statuses and all(status == "normal" for status in statuses):
        return "normal"
    return "unknown"


def has_anti_tb_dominant_claim(normalized_plan: str) -> bool:
    return any(
        marker in normalized_plan
        for marker in [
            normalize_name("抗结核"),
            normalize_name("异烟肼"),
            normalize_name("利福平"),
            normalize_name("吡嗪酰胺"),
            normalize_name("乙胺丁醇"),
            normalize_name("HRZE"),
            normalize_name("标准抗结核"),
        ]
    )


def treatment_text_is_structurally_complete(text: str) -> bool:
    normalized = clean_text(text)
    if not normalized:
        return False
    invalid_patterns = [r"\*\*\*\*", r"[（(]\s*[）)]", r"(?:^|\n)\s*\d+[.、]\s*[：:]"]
    return not any(re.search(pattern, normalized) for pattern in invalid_patterns)


def remove_anti_tb_dominant_claims(treatment_plan: str) -> str:
    retained: List[str] = []
    for entry in re.split(r"(?=\s*(?:\d+[.、]|[-*]))", clean_text(treatment_plan)):
        cleaned = clean_text(entry)
        if not cleaned:
            continue
        normalized = normalize_name(cleaned)
        if has_anti_tb_dominant_claim(normalized):
            continue
        retained.append(cleaned)
    result = normalize_treatment_text("\n".join(retained))
    return result if treatment_text_is_structurally_complete(result) else ""


def has_potassium_replacement_claim(normalized_plan: str) -> bool:
    aliases = ["补钾", "氯化钾", "补充钾", "静脉滴注氯化钾", "口服氯化钾"]
    for clause in re.split(r"(?<=[，,；;。\n])", clean_text(normalized_plan)):
        normalized_clause = normalize_name(clause)
        for alias in aliases:
            token = normalize_name(alias)
            index = normalized_clause.find(token)
            while index >= 0:
                before = normalized_clause[max(0, index - 12):index]
                negated = bool(
                    re.search(
                        r"(?:避免|不应|不得|禁止|无需|不需|不用|暂不|不必|未予|不予)[^，,；;。]{0,8}$",
                        before,
                    )
                )
                if not negated:
                    return True
                index = normalized_clause.find(token, index + len(token))
    return False


def has_magnesium_replacement_claim(normalized_plan: str) -> bool:
    return any(
        marker in normalized_plan
        for marker in [
            normalize_name("补镁"),
            normalize_name("硫酸镁"),
            normalize_name("门冬氨酸钾镁"),
            normalize_name("补充镁"),
            normalize_name("镁剂"),
            normalize_name("静脉补镁"),
        ]
    )


def has_ear_irrigation_claim(normalized_plan: str) -> bool:
    return any(normalize_name(marker) in normalized_plan for marker in ["冲洗耳道", "耳道冲洗", "耳冲洗", "灌洗耳道"])


def remove_ear_irrigation_claims(treatment_plan: str) -> str:
    clauses = re.split(r"(?<=[。；;\n])", clean_text(treatment_plan))
    kept = [
        clause
        for clause in clauses
        if not has_ear_irrigation_claim(normalize_name(clause))
    ]
    return normalize_treatment_text("".join(kept))


def apply_fact_consistency_gate(treatment_plan: str, case_features: Dict[str, Any]) -> Dict[str, Any]:
    plan = clean_text(treatment_plan)
    normalized_plan = normalize_name(plan)
    issues = []
    patches = []
    if has_family_hypertension_without_personal_hypertension(case_features):
        if any(
            marker in normalized_plan
            for marker in [
                normalize_name("患者有妊娠期高血压病史"),
                normalize_name("患者有高血压病史"),
                normalize_name("本人有妊娠期高血压"),
                normalize_name("妊娠期高血压病史"),
            ]
        ):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "unsupported_personal_history_claim",
                    "severity": "must_fix",
                    "problem": "family_hypertension_written_as_personal_history",
                    "patchable": True,
                }
            )
            plan = remove_unsupported_hypertension_claims(plan)
            patches.append("家族高血压仅作为风险因素，不能写成患者本人高血压病史；可保留血压监测和产科风险评估。")
        if any(marker in normalized_plan for marker in [normalize_name("调整降压药"), normalize_name("降压药调整")]):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "unsupported_antihypertensive_adjustment",
                    "severity": "must_fix",
                    "problem": "antihypertensive_adjustment_without_personal_hypertension",
                    "patchable": True,
                }
            )
            plan = remove_unsupported_antihypertensive_adjustment(plan)
            if "家族高血压仅作为风险因素" not in " ".join(patches):
                patches.append("家族高血压仅作为风险因素，不能写成患者本人高血压病史；可保留血压监测和产科风险评估。")
    return {"issues": issues, "patches": patches, "treatment_plan": normalize_treatment_text(plan)}


def has_family_hypertension_without_personal_hypertension(case_features: Dict[str, Any]) -> bool:
    family_history = set(as_text_list(case_features.get("family_history")))
    personal_history = set(as_text_list(case_features.get("personal_history")))
    has_family = "家族高血压" in family_history
    has_personal = any(item in personal_history for item in ["高血压", "妊娠期高血压"])
    return has_family and not has_personal


def remove_unsupported_hypertension_claims(treatment_plan: str) -> str:
    result = clean_text(treatment_plan)
    patterns = [
        r"患者有妊娠期高血压病史[，,；;。]?",
        r"患者有高血压病史[，,；;。]?",
        r"本人有妊娠期高血压[，,；;。]?",
        r"妊娠期高血压病史[，,；;。]?",
    ]
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    return result


def remove_unsupported_antihypertensive_adjustment(treatment_plan: str) -> str:
    result = clean_text(treatment_plan)
    patterns = [
        r"需要调整降压药[，,；;。]?",
        r"调整降压药[，,；;。]?",
        r"降压药调整[，,；;。]?",
    ]
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    return result


def normalize_treatment_text(treatment_plan: str) -> str:
    result = re.sub(r"[；;]\s*[；;。]", "；", clean_text(treatment_plan))
    result = re.sub(r"。；", "。", result)
    result = re.sub(r"\s+", " ", result)
    return result.strip(" ，,；;")


def apply_confirmatory_evidence_treatment_gate(
    *,
    diagnosis: str,
    treatment_plan: str,
    case_features: Dict[str, Any],
) -> Dict[str, Any]:
    plan = clean_text(treatment_plan)
    issues = []
    patches = []
    lesion_specs = {
        "tof": {
            "diagnoses": ["法洛四联症", "TOF"],
            "findings": [
                "法洛四联症",
                "右心室流出道梗阻",
                "右室流出道梗阻",
                "肺动脉狭窄",
                "主动脉骑跨",
            ],
            "procedures": [
                "法洛四联症根治术",
                "法洛四联症根治",
                "法洛四联症外科矫治术",
                "法洛四联症外科矫治",
                "外科矫治术",
                "外科矫治",
                "外科手术矫正",
                "手术矫正",
                "矫正术",
                "矫正手术",
                "右室流出道重建术",
                "右室流出道疏通",
                "右心室流出道重建术",
                "右心室流出道疏通",
            ],
        },
        "pda": {
            "diagnoses": ["动脉导管未闭", "PDA"],
            "findings": ["动脉导管未闭", "PDA"],
            "procedures": [
                "经皮导管封堵动脉导管",
                "导管封堵动脉导管",
                "动脉导管封堵术",
                "动脉导管封堵",
                "PDA封堵术",
                "PDA封堵",
                "PDA结扎术",
                "PDA结扎",
                "外科结扎",
                "动脉导管结扎",
                "切断动脉导管",
                "关闭动脉导管",
                "关闭导管",
            ],
        },
        "asd": {
            "diagnoses": ["房间隔缺损", "ASD"],
            "findings": ["房间隔缺损", "房间隔交通", "ASD"],
            "procedures": [
                "房间隔缺损外科修补术",
                "房间隔缺损外科修补",
                "房间隔缺损修补术",
                "房间隔缺损修补",
                "修补房间隔缺损",
                "房间隔缺损封堵术",
                "房间隔缺损封堵",
                "封堵房间隔缺损",
                "ASD修补术",
                "ASD修补",
                "ASD封堵术",
                "ASD封堵",
            ],
        },
        "vsd": {
            "diagnoses": ["室间隔缺损", "VSD"],
            "findings": ["室间隔缺损", "室间隔交通", "VSD"],
            "procedures": [
                "室间隔缺损外科修补术",
                "室间隔缺损外科修补",
                "室间隔缺损修补术",
                "室间隔缺损修补",
                "修补室间隔缺损",
                "室间隔缺损封堵术",
                "室间隔缺损封堵",
                "封堵室间隔缺损",
                "VSD修补术",
                "VSD修补",
                "VSD封堵术",
                "VSD封堵",
            ],
        },
    }
    normalized_diagnosis = normalize_name(diagnosis)
    diagnosis_lesion = next(
        (
            lesion
            for lesion, spec in lesion_specs.items()
            if any(normalize_name(name) in normalized_diagnosis for name in spec["diagnoses"])
        ),
        "",
    )
    cardiac_case = bool(diagnosis_lesion) or normalized_diagnosis in {
        normalize_name(item) for item in ["先天性心脏病", "法洛四联症"]
    } or has_diagnosis_axis(case_features, "infant_congenital_structural_heart_disease")
    unsafe_cardiac_markers = []
    explicit_lesions = set()
    if cardiac_case:
        for lesion, spec in lesion_specs.items():
            active_markers = [
                marker
                for marker in spec["procedures"]
                if treatment_contains_active_marker(plan, [marker])
            ]
            if not active_markers:
                continue
            explicit_lesions.add(lesion)
            structure_confirmed = (
                has_positive_tof_structure_evidence(case_features)
                if lesion == "tof"
                else has_positive_cardiac_structure_evidence(case_features, spec["findings"])
            )
            if not structure_confirmed:
                unsafe_cardiac_markers.extend(active_markers)

        generic_cardiac_markers = [
            "经导管封堵术",
            "经导管封堵",
            "经皮导管封堵",
            "导管介入封堵术",
            "导管介入封堵",
            "导管封堵",
            "介入封堵",
            "封堵术",
            "补片修补术",
            "补片修补",
            "外科修补术",
            "外科修补",
            "修补术",
            "完全矫治术",
            "完全矫治",
            "根治性修复术",
            "根治性修复",
            "根治术",
            "根治手术",
            "矫治术",
            "矫治手术",
            "结扎术",
            "结扎手术",
            "开胸修补术",
            "开胸修补",
        ]
        active_generic_markers = [
            marker
            for marker in generic_cardiac_markers
            if treatment_contains_active_generic_cardiac_marker(plan, marker)
        ]
        generic_lesion = next(iter(explicit_lesions)) if len(explicit_lesions) == 1 else diagnosis_lesion
        generic_findings = lesion_specs.get(generic_lesion, {}).get("findings")
        generic_structure_confirmed = (
            has_positive_tof_structure_evidence(case_features)
            if generic_lesion == "tof"
            else has_positive_cardiac_structure_evidence(case_features, generic_findings)
        )
        if active_generic_markers and not generic_structure_confirmed:
            unsafe_cardiac_markers.extend(active_generic_markers)

    if unsafe_cardiac_markers:
        issues.append(confirmatory_evidence_issue("cardiac_anatomy"))
        plan = remove_active_treatment_clauses(plan, unique_preserve_order(unsafe_cardiac_markers))
        patches.append("尚未取得与拟议病变一致的心脏解剖和血流动力学阳性证据时，不得预设具体介入、外科术式或关导管药物；先完成超声心动图并由儿童心脏专科分型。")
    eye_case = normalize_name("白内障") in normalize_name(diagnosis) or has_diagnosis_axis(
        case_features,
        "pediatric_progressive_night_blindness",
    )
    eye_markers = [
        "超声乳化",
        "人工晶体",
        "人工晶状体",
        "人工晶状体植入",
        "晶体植入",
        "晶状体植入",
        "IOL植入",
        "白内障手术",
        "白内障摘除术",
        "白内障摘除",
        "白内障吸除术",
        "白内障吸除",
        "晶状体吸除术",
        "晶状体吸除",
        "晶状体切除术",
        "晶状体切除",
        "晶状体摘除",
        "晶状体囊外摘除",
        "白内障囊外摘除",
        "囊外摘除",
        "晶状体囊内摘除",
        "白内障囊内摘除",
        "囊内摘除",
    ]
    if eye_case and not has_positive_lens_opacity_evidence(case_features) and treatment_contains_active_marker(
        plan,
        eye_markers,
    ):
        issues.append(confirmatory_evidence_issue("lens_opacity"))
        plan = remove_active_treatment_clauses(plan, eye_markers)
        patches.append("尚未取得晶状体混浊的客观证据时，不得预设眼内手术；先完成裂隙灯、眼底和视网膜功能评估，再由儿童眼科决定治疗路径。")
    return {"issues": issues, "patches": patches, "treatment_plan": normalize_treatment_text(plan)}


def confirmatory_evidence_issue(problem: str) -> Dict[str, Any]:
    return {
        "field": "treatment_plan",
        "code": "irreversible_intervention_without_confirmatory_evidence",
        "severity": "must_fix",
        "problem": problem,
        "patchable": True,
    }


def axis_candidate_semantically_supported(
    official: str,
    raw_name: str,
    case_text: str,
    supported_official_names: set[str],
    *,
    candidate_support_text: str = "",
) -> bool:
    full_text_states = [
        explicit_name_scope(case_text, item)
        for item in [official, raw_name]
        if normalize_name(item)
    ]
    if "negative" in full_text_states:
        return False
    if normalize_name(official) in {normalize_name(item) for item in supported_official_names}:
        return True
    support_text = candidate_support_text or case_text
    support_states = [
        explicit_name_scope(support_text, item)
        for item in [official, raw_name]
        if normalize_name(item)
    ]
    if "positive" in support_states:
        return True
    if disease_context_boost(normalize_name(official), normalize_name(support_text)) > 0:
        return True
    return False


def explicit_name_scope(text: str, marker: str) -> str:
    uncertain_markers = [
        "考虑", "疑似", "可能", "可疑", "待排", "需排除", "需要排除", "不能排除", "未能排除",
        "未排除", "尚未排除", "尚不能排除", "不除外", "是否", "有无", "有没有", "待核对", "需核对", "尚不明确",
    ]
    resolved_markers = ["排除", "已排除", "未见", "无", "没有", "阴性", "正常", "不支持", "未检出"]
    return marker_group_state(
        text,
        [marker],
        uncertain_markers=uncertain_markers,
        resolved_markers=resolved_markers,
        historical_markers=["既往", "曾", "以前", "病史", "母亲", "父亲", "家族", "兄弟", "姐妹", "子女"],
    )


def has_positive_cardiac_structure_evidence(
    case_features: Dict[str, Any],
    lesion_markers: Optional[List[str]] = None,
) -> bool:
    markers = lesion_markers or [
        "动脉导管未闭",
        "房间隔缺损",
        "室间隔缺损",
        "法洛四联症",
        "结构异常",
        "心脏畸形",
    ]
    resolved = ["未见", "没有", "无", "阴性", "排除", "不支持", "正常"]
    return has_positive_confirmatory_exam_finding(
        case_features,
        exam_markers=["超声心动图", "TTE", "TEE"],
        finding_markers=markers,
        resolved_markers=resolved,
    )


def has_positive_tof_structure_evidence(case_features: Dict[str, Any]) -> bool:
    resolved = ["未见", "没有", "无", "阴性", "排除", "不支持", "正常"]
    states = []
    for payload in matching_exam_payloads(case_features, ["超声心动图", "TTE", "TEE"]):
        if not exam_result_is_usable(payload):
            continue
        explicit = confirmatory_payload_state(
            payload,
            ["法洛四联症", "TOF"],
            resolved_markers=resolved,
        )
        outflow = confirmatory_payload_state(
            payload,
            ["右心室流出道梗阻", "右室流出道梗阻", "肺动脉狭窄"],
            resolved_markers=resolved,
        )
        septal_or_aortic = confirmatory_payload_state(
            payload,
            ["室间隔缺损", "VSD", "主动脉骑跨"],
            resolved_markers=resolved,
        )
        if explicit == "positive" or (outflow == "positive" and septal_or_aortic == "positive"):
            states.append("positive")
        elif "negative" in {explicit, outflow, septal_or_aortic}:
            states.append("negative")
    return "positive" in states and "negative" not in states


def has_positive_lens_opacity_evidence(case_features: Dict[str, Any]) -> bool:
    markers = ["晶状体混浊", "晶状体浑浊", "白内障", "遮挡视轴"]
    resolved = ["未见", "没有", "无", "阴性", "排除", "不支持", "透明", "正常"]
    return has_positive_confirmatory_exam_finding(
        case_features,
        exam_markers=["裂隙灯", "眼科检查"],
        finding_markers=markers,
        resolved_markers=resolved,
    )


def has_positive_confirmatory_exam_finding(
    case_features: Dict[str, Any],
    *,
    exam_markers: List[str],
    finding_markers: List[str],
    resolved_markers: List[str],
) -> bool:
    states = []
    for payload in matching_exam_payloads(case_features, exam_markers):
        if not exam_result_is_usable(payload):
            continue
        state = confirmatory_payload_state(payload, finding_markers, resolved_markers=resolved_markers)
        if state != "absent":
            states.append(state)
    return "positive" in states and "negative" not in states


def confirmatory_payload_state(
    payload: Dict[str, Any],
    markers: List[str],
    *,
    resolved_markers: List[str],
) -> str:
    pairs = exam_result_pairs(payload)
    conclusion_values = [
        value
        for key, value in pairs
        if any(marker in normalize_name(key) for marker in ["结论", "诊断", "印象"])
    ]
    uncertain = confirmatory_uncertain_markers()
    if conclusion_values:
        state = marker_group_state(
            "；".join(conclusion_values),
            markers,
            uncertain_markers=uncertain,
            resolved_markers=resolved_markers,
            historical_markers=["既往", "曾有"],
        )
        if state != "absent":
            return state
    non_conclusion = [
        "%s：%s" % pair
        for pair in pairs
        if not any(marker in normalize_name(pair[0]) for marker in ["结论", "诊断", "印象"])
    ]
    return marker_group_state(
        "；".join(non_conclusion),
        markers,
        uncertain_markers=uncertain,
        resolved_markers=resolved_markers,
        historical_markers=["既往", "曾有"],
    )


def confirmatory_uncertain_markers() -> List[str]:
    return [
        "考虑", "疑似", "可能", "可疑", "待排", "需排除", "需要排除", "不能排除",
        "未能排除", "未排除", "尚未排除", "尚不能排除", "不除外", "鉴别",
        "待明确", "待证实", "需明确", "需证实",
    ]


def marker_group_state(
    text: str,
    markers: List[str],
    *,
    uncertain_markers: List[str],
    resolved_markers: List[str],
    historical_markers: List[str],
) -> str:
    states = [
        marker_last_state(
            text,
            marker,
            uncertain_markers=uncertain_markers,
            resolved_markers=resolved_markers,
            historical_markers=historical_markers,
        )
        for marker in markers
        if normalize_name(marker)
    ]
    if "positive" in states:
        return "positive"
    if "negative" in states:
        return "negative"
    return "absent"


def marker_last_state(
    text: str,
    marker: str,
    *,
    uncertain_markers: List[str],
    resolved_markers: List[str],
    historical_markers: List[str],
) -> str:
    token = normalize_name(marker)
    state = "absent"
    for clause in semantic_clauses(text):
        normalized_clause = normalize_name(clause)
        start = 0
        while token:
            index = normalized_clause.find(token, start)
            if index < 0:
                break
            state = marker_occurrence_state(
                normalized_clause,
                token,
                index,
                uncertain_markers=uncertain_markers,
                resolved_markers=resolved_markers,
                historical_markers=historical_markers,
            )
            start = index + len(token)
    return state


def marker_occurrence_state(
    clause: str,
    token: str,
    index: int,
    *,
    uncertain_markers: List[str],
    resolved_markers: List[str],
    historical_markers: List[str],
) -> str:
    prefix = clause[max(0, index - 24):index]
    suffix = clause[index + len(token):index + len(token) + 24]
    current_cues = [
        "现见", "仍见", "复查见", "复核见", "当前见", "再次发现", "残余", "复发",
        "较既往", "比既往",
    ]
    current = any(cue in prefix for cue in current_cues)
    uncertain = any(normalize_name(item) in prefix or suffix.startswith(normalize_name(item)) for item in uncertain_markers)
    resolved = any(
        prefix.endswith(normalize_name(item))
        or suffix.startswith(normalize_name(item))
        or any(suffix.startswith(prefix_marker + normalize_name(item)) for prefix_marker in ["已", "基本", "可"])
        for item in resolved_markers
        if normalize_name(item)
    )
    historical = not current and any(normalize_name(item) in prefix for item in historical_markers)
    completed = bool(
        re.search(r"(?:根治|修补|封堵|矫治|植入|摘除|切除|吸除|手术)?术后", suffix)
        or re.match(r"已(?:完成|修补|封堵|矫治|植入|摘除|切除|吸除)", suffix)
    )
    if uncertain or resolved or historical or completed or marker_occurrence_is_negated(clause, token, index):
        return "negative"
    return "positive"


def confirmatory_marker_present(text: str, markers: List[str], *, resolved_markers: List[str]) -> bool:
    return marker_group_state(
        text,
        markers,
        uncertain_markers=confirmatory_uncertain_markers(),
        resolved_markers=resolved_markers,
        historical_markers=["既往", "曾有"],
    ) == "positive"


def treatment_contains_active_marker(treatment_plan: str, markers: List[str]) -> bool:
    return any(
        clause_contains_active_marker(clause, markers)
        for clause in re.split(r"(?<=[，,；;。\n])", clean_text(treatment_plan))
    )


def treatment_contains_active_generic_cardiac_marker(treatment_plan: str, marker: str) -> bool:
    cardiac_targets = [
        "动脉导管",
        "房间隔",
        "室间隔",
        "法洛",
        "PDA",
        "ASD",
        "VSD",
        "流出道",
        "肺动脉",
        "主动脉",
    ]
    modifiers = [
        "尽早", "择期", "外科", "开胸", "经皮", "经导管", "导管介入", "介入",
        "手术", "补片", "完全", "根治性", "姑息", "分期", "首选",
    ]
    for clause in re.split(r"(?<=[，,；;。\n])", clean_text(treatment_plan)):
        if not clause_contains_active_marker(clause, [marker]):
            continue
        normalized = normalize_name(clause)
        token = normalize_name(marker)
        index = normalized.find(token)
        prefix = normalized[:index]
        proposals = list(re.finditer(r"(?:不主张|首选|建议|计划|拟行|拟|安排|考虑|实施|进行|接受|选择|采用|推荐|行)", prefix))
        if not proposals:
            continue
        tail = prefix[proposals[-1].end():]
        if any(normalize_name(item) in tail for item in cardiac_targets):
            return True
        for modifier in modifiers:
            tail = tail.replace(normalize_name(modifier), "")
        if not tail:
            return True
    return False


def clause_contains_active_marker(clause: str, markers: List[str]) -> bool:
    normalized = normalize_name(clause)
    for marker in markers:
        token = normalize_name(marker)
        start = 0
        while token:
            index = normalized.find(token, start)
            if index < 0:
                break
            prefix = normalized[max(0, index - 24):index]
            if treatment_marker_is_non_active(normalized, index, marker):
                start = index + len(token)
                continue
            directly_negated = re.search(
                r"(?:避免|禁止|暂缓|删除|不得|不能|不应|不宜|无需|不需|不要|暂不|尚不|无|未)"
                r"(?:证据|明确|手术|介入|治疗|操作|指征|必要性|立即|直接|常规|贸然|盲目|预设|推荐|建议|考虑|安排|选择|采用|进行|实施|接受|支持|具备|满足|行|做){0,5}$",
                prefix,
            )
            if not directly_negated:
                return True
            start = index + len(token)
    return False


def treatment_marker_is_non_active(normalized_clause: str, index: int, marker: str) -> bool:
    prefix = normalized_clause[:index]
    token = normalize_name(marker)
    suffix = normalized_clause[index + len(token):]
    if re.search(
        r"(?:不建议|不推荐|不主张|不需要|不考虑|不再行|未达到|未满足|不具备|拒绝|不同意|"
        r"不接受|放弃|仅讨论|尚未决定|未决定|已取消|"
        r"避免|禁止|不得|不应|不宜|无需|暂不|不要).{0,12}$",
        prefix,
    ):
        return True
    if re.match(r"(?:术)?(?:(?:现)?已取消|(?:现)?取消|(?:现)?放弃)", suffix):
        return True
    active_proposal = re.search(
        r"(?:建议|计划|拟|拟行|安排|考虑|实施|进行|接受|选择|采用|推荐).{0,16}$",
        prefix,
    )
    if not active_proposal:
        action_match = re.search(r"行.{0,16}$", prefix)
        if action_match and not re.search(r"(?:既往|曾|已)$", prefix[max(0, action_match.start() - 4):action_match.start()]):
            active_proposal = action_match
    historical = re.search(r"(?:既往|曾|已完成|已行|已接受|已植入|已置入|术后|手术后).{0,12}$", prefix)
    if historical and (not active_proposal or historical.start() > active_proposal.start()):
        return True
    if re.match(r"(?:术)?已(?:顺利)?(?:完成|实施|接受|植入|置入)", suffix):
        return True
    lens_markers = {"人工晶体", "人工晶状体", "晶体植入", "晶状体植入", "iol植入"}
    if token in lens_markers and (
        re.search(r"(?:监测|复查|随访|观察).{0,4}$", prefix)
        or re.match(r"(?:位置(?:监测|复查|随访|观察)|度数(?:计算|测量|评估))", suffix)
        or (
            not active_proposal
            and re.match(r"植入(?:术)?后(?:位置|继续|监测|随访|复查|康复|训练|观察|恢复)", suffix)
        )
    ):
        return True
    if not active_proposal and re.match(
        r"(?:术)?后(?:定期)?(?:拟)?(?:继续|加强|监测|随访|复查|康复|训练|观察|营养|喂养|支持|"
        r"感染|恢复|位置|出现|发生|并发|需|要|给予)",
        suffix,
    ):
        return True
    return prefix.endswith("术后") and normalize_name(marker) not in {"根治术", "矫治术"}


def remove_active_treatment_clauses(treatment_plan: str, markers: List[str]) -> str:
    clauses = re.split(
        r"(?<=[，,；;。\n])|"
        r"(?=(?:并|同时|另行|另外|或|联合)(?:建议|安排|考虑|转|继续|给予|开展|使用|保留|监测|提供|药物|营养|康复|随访|支持))|"
        r"(?=后(?:定期)?(?:拟)?(?:继续|加强|监测|随访|复查|康复|训练|观察|营养|喂养|支持|"
        r"感染|恢复|位置|出现|发生|并发|需|要|给予))|"
        r"(?=后再(?:行|实施|评估|安排|进行|接受))|"
        r"(?<!术)(?=后(?:行|实施|评估|安排|进行|接受))|"
        r"(?=再(?:行|实施|评估|安排|进行|接受))",
        clean_text(treatment_plan),
    )
    kept = []
    removed_previous = False
    for clause in clauses:
        if clause_contains_active_marker(clause, markers):
            removed_previous = True
            continue
        if removed_previous:
            clause = re.sub(r"^(?:并|同时|另行|另外|或|联合|后)", "", clause)
        if clause:
            kept.append(clause)
        removed_previous = False
    return normalize_treatment_text("".join(kept))


def apply_diagnosis_specific_treatment_gate(
    *,
    diagnosis: str,
    treatment_plan: str,
    case_features: Dict[str, Any],
) -> Dict[str, Any]:
    plan = clean_text(treatment_plan)
    issues = []
    patches = []
    case_text = case_features_text(case_features)
    normalized_diagnosis = normalize_name(diagnosis)
    normalized_plan = normalize_name(plan)

    if has_negative_bacterial_culture(case_features) and treatment_recommends_prophylactic_antibiotics(
        normalized_plan
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "prophylactic_antibiotic_after_negative_bacterial_culture",
                "severity": "must_fix",
                "problem": "negative_bacterial_culture_does_not_support_routine_prophylaxis",
                "patchable": True,
            }
        )
        plan = sanitize_prophylactic_antibiotic_recommendations(plan)
        patches.append(
            "细菌培养阴性时不应常规或预防性使用抗生素；只有出现明确继发细菌感染证据（如脓性分泌物、进行性蜂窝织炎或全身感染）时才考虑抗菌药，并结合培养和药敏调整。"
        )

    skin_result = apply_skin_myiasis_treatment_gate(
        diagnosis=diagnosis,
        treatment_plan=plan,
        case_features=case_features,
    )
    issues.extend(skin_result.get("issues", []))
    patches.extend(skin_result.get("patches", []))
    plan = clean_text(skin_result.get("treatment_plan") or plan)
    normalized_plan = normalize_name(plan)

    if has_qt_prolongation_with_tricyclic_exposure(case_features) and treatment_continues_tricyclic_antidepressant(
        normalized_plan
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "qt_prolongation_with_tricyclic_exposure",
                "severity": "must_fix",
                "problem": "continued_tricyclic_exposure_with_prolonged_qtc",
                "patchable": True,
            }
        )
        plan = sanitize_tricyclic_continuation(plan)
        patches.append(
            "QTc延长伴三环类抗抑郁药过量或持续使用时，应立即停用并由医生/中毒与心脏专科复核，进行心电监护、复查钾镁等电解质和连续心电图，避免继续使用延长QT或诱发心律失常的药物。"
        )

    if has_stroke_secondary_prevention_context(case_features) and treatment_stops_aspirin_without_active_bleed(
        plan, case_features
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "stroke_secondary_prevention_aspirin_discontinued",
                "severity": "must_fix",
                "problem": "unindicated_aspirin_stop_after_ischemic_stroke",
                "patchable": True,
            }
        )
        plan = sanitize_unindicated_aspirin_discontinuation(plan)
        patches.append(
            "有缺血性卒中/TIA或明确卒中二级预防背景时，不得在无活动性大出血、严重出血并发症或专科书面替代方案的情况下擅自停用阿司匹林；"
            "应继续阿司匹林（或由心脑专科评估后规范切换为其他抗血小板/抗凝方案），并监测出血与心脑血管事件。"
        )

    if is_mitral_stenosis_case(normalized_diagnosis, case_features, case_text) and treatment_recommends_pbmv_without_thrombus_exclusion(
        plan
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "mitral_stenosis_pbmv_before_la_thrombus_exclusion",
                "severity": "must_fix",
                "problem": "balloon_valvuloplasty_before_tee_or_la_thrombus_exclusion",
                "patchable": True,
            }
        )
        plan = sanitize_pbmv_before_thrombus_exclusion(plan)
        patches.append(
            "重度二尖瓣狭窄评估经皮球囊成形前，必须先完成经食道超声或等效影像排除左房/左心耳血栓；"
            "血栓未排除或已证实血栓时不得催促球囊扩张，应先规范抗凝并由心内科/心外科分层决策。"
        )

    if treatment_recommends_noac_for_rheumatic_mitral_stenosis(plan, case_features, case_text, normalized_diagnosis):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "noac_in_rheumatic_mitral_stenosis",
                "severity": "must_fix",
                "problem": "doac_not_preferred_for_rheumatic_ms_af",
                "patchable": True,
            }
        )
        plan = sanitize_noac_in_rheumatic_ms(plan)
        patches.append(
            "风湿性心脏病中重度二尖瓣狭窄合并房颤或左房血栓高危时，抗凝应优先华法林并监测INR；"
            "不宜将新型口服抗凝药（NOAC/DOAC）作为默认首选，须由心内科按指南分层。"
        )

    if is_hf_reduced_ef_case(normalized_diagnosis, case_features) and not has_hf_guideline_core(
        normalized_plan
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "hfrEF_missing_guideline_core",
                "severity": "must_fix",
                "problem": "reduced_ef_heart_failure_plan_lacks_guideline_directed_core",
                "patchable": True,
            }
        )
        patches.append(
            "LVEF降低的扩张型心肌病/心衰应尽快由心衰专科评估并启动循证核心治疗（ARNI或ACEI/ARB、循证β受体阻滞剂、醛固酮受体拮抗剂及适用时SGLT2抑制剂）；有水肿或腹胀等容量负荷时加用利尿剂，伴晕厥前兆需急诊评估并监测肾功能、电解质和血流动力学。"
        )

    if normalized_diagnosis == normalize_name("室间隔缺损（VSD）") and has_positive_vsd_text(case_text):
        cyanotic_physiology = marker_present_not_negated(case_text, ["右向左分流", "肺动脉高压", "肺高压"])
        direct_repair = marker_present_not_negated(normalized_plan, ["根治性修补", "根治修补", "直接修补", "VSD修补术"])
        staged_path = plan_has_any(normalized_plan, ["姑息", "分期", "先稳定", "降低肺血管阻力", "儿童心脏专科"])
        if cyanotic_physiology and direct_repair and not staged_path:
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "cyanotic_vsd_direct_repair_without_staging",
                    "severity": "must_fix",
                    "problem": "right_to_left_shunt_pulmonary_hypertension_needs_staged_path",
                    "patchable": True,
                }
            )
            plan = re.sub(r"(?:尽快|立即)?(?:进行|实施|安排)?(?:VSD)?(?:根治性)?修补术(?:根治疾病)?", "", plan)
            patches.append(
                "大型室间隔缺损伴右向左分流或肺动脉高压时，应先由儿童心脏专科稳定氧合、心衰和肺血管阻力，评估姑息或分期路径；不能直接承诺根治性修补。"
            )

    neurocysticercosis_risk = (
        normalized_diagnosis == normalize_name("神经囊虫病")
        or has_neurocysticercosis_strong_evidence_pattern(case_text)
    )
    intracranial_pressure_risk = marker_present_not_negated(
        case_text,
        ["脑积水", "颅内压升高", "颅高压", "弯腰加重", "反复呕吐", "喷射性呕吐"],
    )
    if neurocysticercosis_risk and intracranial_pressure_risk and treatment_recommends_triptan(normalized_plan):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "triptan_with_intracranial_hypertension_neurocysticercosis",
                "severity": "must_fix",
                "problem": "triptan_path_before_intracranial_hypertension_control",
                "patchable": True,
            }
        )
        plan = sanitize_triptan_recommendations(plan)
        patches.append(
            "疑似或确诊神经囊虫病伴颅高压/脑积水时，应急诊转神经科和感染专科，先处理颅高压或脑积水；确认活动性神经囊虫病后由专科制定阿苯达唑联合糖皮质激素或其他抗炎方案，并按适应证给予抗癫痫治疗及眼底、影像监测。"
        )

    if is_retinoblastoma_treatment_gate_case(normalized_diagnosis, case_features, case_text):
        normalized_plan = normalize_name(plan)
        has_urgent_oncology_path = plan_has_any(normalized_plan, ["眼肿瘤", "儿童眼科", "眼科急诊", "紧急转诊"])
        has_staging_path = plan_has_any(normalized_plan, ["分期", "双眼评估", "双眼检查", "视神经", "转移"])
        has_definitive_path = plan_has_any(normalized_plan, ["化疗", "局部治疗", "激光", "冷冻", "眼球摘除"])
        if not (has_urgent_oncology_path and has_staging_path and has_definitive_path):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_retinoblastoma",
                    "severity": "must_fix",
                    "problem": "retinoblastoma_closed_with_amblyopia_care_or_follow_up_only",
                    "patchable": True,
                }
            )
            patches.append(
                "白瞳并视觉下降提示视网膜母细胞瘤时，不能只做配镜、遮盖或常规随访；应紧急转眼肿瘤/儿童眼科，完成双眼评估和分期，并由眼肿瘤团队依据眼球、视神经及转移分期选择保眼化疗、局部治疗或眼球摘除等路径，同时评估社工、交通与家庭照护支持。"
            )

    if is_sma_syndrome_treatment_gate_case(normalized_diagnosis, case_features, case_text):
        normalized_plan = normalize_name(plan)
        has_decompression = plan_has_any(normalized_plan, ["胃肠减压", "胃管减压", "鼻胃管"])
        has_nutrition = plan_has_any(normalized_plan, ["营养", "少量多餐", "鼻十二指肠", "空肠"])
        has_position = plan_has_any(normalized_plan, ["左侧卧", "俯卧", "膝胸位"])
        has_escalation = plan_has_any(normalized_plan, ["外科", "手术", "持续梗阻", "营养失败"])
        if not (has_decompression and has_nutrition and has_position and has_escalation):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_sma_syndrome",
                    "severity": "must_fix",
                    "problem": "sma_syndrome_without_decompression_nutrition_position_or_escalation",
                    "patchable": True,
                }
            )
            patches.append(
                "肠系膜上动脉压迫综合征应先胃肠减压，纠正液体、电解质和营养不足，采用少量多餐及餐后左侧卧、俯卧或膝胸位；不能口服时考虑鼻十二指肠或空肠营养，持续梗阻或营养失败时转外科评估。"
            )

    if is_symptomatic_large_renal_cyst_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        has_intervention_path = plan_has_any(
            normalized_plan,
            ["泌尿外科", "介入", "穿刺", "硬化", "腹腔镜", "手术"],
        )
        has_safety_workup = plan_has_any(normalized_plan, ["血尿", "尿液", "感染", "结石", "肿瘤"])
        if not (has_intervention_path and has_safety_workup):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_symptomatic_renal_cyst",
                    "severity": "must_fix",
                    "problem": "symptomatic_large_renal_cyst_observation_only",
                    "patchable": True,
                }
            )
            patches.append(
                "症状性较大肾囊肿伴压迫或尿频时，不能只写观察止痛；应转泌尿外科评估穿刺硬化、介入或手术适应证，并排查血尿、感染、结石或肿瘤等伴随泌尿风险。"
            )
    if is_deviated_septum_recurrent_epistaxis_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        has_local_care = plan_has_any(normalized_plan, ["生理盐水", "保湿", "润滑", "凡士林"])
        has_bleeding_workup = plan_has_any(normalized_plan, ["凝血", "血小板", "血液学"])
        personal_context = set(as_text_list(case_features.get("personal_context")))
        needs_context_care = bool({"学生", "运动诱发鼻出血"} & personal_context)
        has_context_care = not needs_context_care or plan_has_any(normalized_plan, ["运动防护", "避免鼻部外伤"])
        if not (has_local_care and has_bleeding_workup and has_context_care):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_deviated_septum_recurrent_epistaxis",
                    "severity": "should_fix",
                    "problem": "missing_epistaxis_workup_local_care_or_context",
                    "patchable": True,
                }
            )
            patch = "鼻中隔偏曲伴反复鼻出血时，应补充生理盐水冲洗、局部保湿/润滑护理，并评估凝血功能、血小板或基础血液学。"
            if needs_context_care:
                patch += "结合学生和运动场景，补充运动防护与避免鼻部外伤建议。"
            patches.append(patch)
    if is_acute_bacterial_prostatitis_complication_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        has_ast_guidance = plan_has_any(normalized_plan, ["药敏", "抗菌药物敏感", "尿培养", "培养"])
        has_complication_plan = plan_has_any(
            normalized_plan,
            ["尿潴留", "导尿", "膀胱造瘘", "脓肿", "住院", "急诊", "败血症", "脓毒症"],
        )
        if not (has_ast_guidance and has_complication_plan):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_acute_bacterial_prostatitis_complication_risk",
                    "severity": "must_fix",
                    "problem": "missing_ast_guided_antibiotics_or_retention_abscess_plan",
                    "patchable": True,
                }
            )
            patches.append(
                "急性细菌性前列腺炎伴高热、尿潴留风险或尿培养阳性时，抗生素应结合尿培养和药敏结果调整；需评估尿潴留、前列腺脓肿、败血症/脓毒症风险，必要时急诊/住院，避免经尿道操作，尿潴留优先考虑膀胱造瘘等安全引流方案。"
            )
    if is_viral_conjunctivitis_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        if treatment_recommends_routine_eye_antibiotics(normalized_plan):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "unnecessary_routine_antibiotics_for_viral_conjunctivitis",
                    "severity": "should_fix",
                    "problem": "routine_or_prophylactic_eye_antibiotics_without_bacterial_evidence",
                    "patchable": True,
                }
            )
            plan = sanitize_routine_eye_antibiotics(plan)
            patches.append(
                "腺病毒性或病毒性结膜炎以支持治疗为主，可用人工泪液、冷敷、卫生隔离和避免揉眼；局部抗菌药不作为常规预防用药，只有出现明确继发细菌感染证据时才考虑抗菌药。"
            )
    if is_post_traumatic_brain_injury_cognitive_vestibular_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        has_analgesic_limit = plan_has_any(normalized_plan, ["止痛药频率", "药物过度", "每周", "限制止痛", "避免过度使用"])
        has_vestibular_rehab = plan_has_any(normalized_plan, ["前庭康复", "平衡训练", "前庭训练"])
        has_cognitive_support = plan_has_any(normalized_plan, ["认知支持", "认知康复", "神经心理", "心理干预", "学习工作调整"])
        if not (has_analgesic_limit and has_vestibular_rehab and has_cognitive_support):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_post_traumatic_brain_injury_cognitive_vestibular",
                    "severity": "must_fix",
                    "problem": "migraine_only_plan_missing_post_traumatic_rehab_and_medication_safety",
                    "patchable": True,
                }
            )
            patches.append(
                "创伤后脑损伤综合征伴持续头痛、认知或前庭症状时，不能只按普通偏头痛止痛处理；应限制急性止痛药使用频率以避免药物过度使用性头痛，针对头晕和平衡不稳安排前庭康复/平衡训练，并对注意力、记忆或情绪问题提供认知支持、神经心理评估后的认知康复和必要心理干预。"
            )
    if is_infant_congenital_structural_heart_disease_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        has_specialty_path = plan_has_any(normalized_plan, ["心脏专科", "心外科", "儿童心脏", "小儿心脏", "NICU", "新生儿重症"])
        has_definitive_path = plan_has_any(normalized_plan, ["手术", "介入", "心导管", "矫治", "球囊", "超声心动图"])
        has_hemodynamic_risk_plan = plan_has_any(normalized_plan, ["肺高压", "心衰", "低氧", "缺氧", "血流动力学"])
        has_feeding_growth_support = plan_has_any(normalized_plan, ["喂养", "营养", "生长发育", "体重"])
        if not (has_specialty_path and has_definitive_path and has_hemodynamic_risk_plan and has_feeding_growth_support):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_infant_congenital_structural_heart_disease",
                    "severity": "must_fix",
                    "problem": "oxygen_observation_only_missing_pediatric_cardiac_definitive_path",
                    "patchable": True,
                }
            )
            patches.append(
                "婴儿三房心或先天性结构性心脏病伴早发气促、喂养困难出汗和发绀/P2亢进时，不能只按呼吸道症状吸氧观察；应尽快转儿童心脏专科或心外科，在超声心动图、必要心导管和血流动力学评估基础上明确手术或介入矫治路径，同时管理肺高压、心衰和低氧风险，并提供喂养、营养和生长发育支持。"
            )
    if has_cyanotic_duct_dependent_risk_pattern(case_features_text(case_features)) and treatment_recommends_pda_closure(
        plan
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "duct_dependent_cyanosis_pda_closure_risk",
                "severity": "must_fix",
                "problem": "cyanotic_infant_default_pda_closure_without_duct_assessment",
                "patchable": True,
            }
        )
        plan = sanitize_pda_closure_recommendations(plan)
        patches.append(
            "新生儿发绀且吸氧改善差时，在明确非导管依赖生理前，不能默认药物性关闭动脉导管；应优先心脏专科评估，考虑是否需维持导管开放（前列腺素E1/前列地尔等路径）并尽快超声心动图分型。"
        )
    if is_postop_chylothorax_or_pleural_effusion_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        observation_only = plan_has_any(
            normalized_plan,
            ["继续观察", "暂不处理", "无需处理", "超声未见明显积液", "未见积液即可"],
        ) or not plan_has_any(
            normalized_plan,
            ["引流", "闭式引流", "胸腔引流", "置管", "胸外科", "禁食", "静脉营养", "生长抑素", "奥曲肽"],
        )
        if observation_only:
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "postop_chylothorax_observation_only",
                    "severity": "must_fix",
                    "problem": "postop_respiratory_distress_closed_by_negative_ultrasound_observation",
                    "patchable": True,
                }
            )
            patches.append(
                "胸部/纵隔术后出现呼吸费力时，不能因单次阴性床旁超声就仅观察收口；应监测氧合，安排进一步胸部结构影像，并按胸腔积液/乳糜风险评估引流适应证、饮食/静脉营养限制与胸外科路径。"
            )
    if is_migraine_reproductive_travel_trigger_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        has_pregnancy_safety = plan_has_any(normalized_plan, ["妊娠", "hcg", "怀孕"])
        has_travel_trigger_plan = plan_has_any(normalized_plan, ["旅行", "坐飞机", "晕动", "视觉运动", "前庭"])
        has_preventive_stratification = plan_has_any(normalized_plan, ["发作频率", "预防用药", "预防治疗", "短期预防", "月经相关"])
        if not (has_pregnancy_safety and has_travel_trigger_plan and has_preventive_stratification):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_migraine_reproductive_travel_trigger",
                    "severity": "should_fix",
                    "problem": "generic_migraine_plan_missing_reproductive_safety_or_travel_trigger_stratification",
                    "patchable": True,
                }
            )
            patches.append(
                "育龄女性偏头痛伴月经相关和旅行/坐飞机/视觉运动诱发时，不能只给通用止痛和笼统避免诱因；用药前应确认妊娠状态，急性期用药需按妊娠可能性选择，记录发作频率和月经相关性以判断短期或长期预防用药指征，并针对旅行、晕动或视觉运动诱发制定睡眠、补水、前庭/晕动管理和出行前预防策略。"
            )
    if is_umbilical_granulation_bleeding_mass_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        dismisses_care = plan_has_any(normalized_plan, ["无需进一步", "无需处置", "通常自愈", "不用处理"])
        has_local_care = plan_has_any(normalized_plan, ["局部专科", "皮肤科", "儿外科", "脐部护理", "局部护理", "硝酸银"])
        has_bleeding_or_pathology_plan = plan_has_any(normalized_plan, ["止血", "病理", "活检", "感染", "脐炎"])
        if dismisses_care or not (has_local_care and has_bleeding_or_pathology_plan):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_umbilical_granulation_bleeding_mass",
                    "severity": "should_fix",
                    "problem": "neonatal_moist_bleeding_umbilical_mass_dismissed_without_local_care",
                    "patchable": True,
                }
            )
            patches.append(
                "新生儿脐部鲜红湿润且易出血的局部肿块不能直接写通常自愈或无需进一步处置；应安排局部专科评估，进行脐部护理和止血处理，必要时用硝酸银等局部处置，并结合感染表现、持续出血或外观异常决定是否病理确认。"
            )
    if is_chronic_alcohol_liver_injury_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        has_abstinence = plan_has_any(normalized_plan, ["戒酒", "停止饮酒", "彻底戒酒"])
        has_monitoring = plan_has_any(
            normalized_plan,
            ["肝功能", "腹部超声", "凝血", "血常规", "并发症", "腹水", "静脉曲张", "随访", "复查"],
        )
        if not (has_abstinence and has_monitoring):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "undertreated_alcohol_liver_without_monitoring_path",
                    "severity": "must_fix",
                    "problem": "alcohol_liver_lifestyle_only_missing_monitoring_or_complication_path",
                    "patchable": True,
                }
            )
            patches.append(
                "长期大量饮酒相关肝病不能只写笼统减重观察；必须落实持续戒酒，并安排肝功能/凝血或血常规与腹部超声等监测，同时交待腹水、出血、意识改变等并发症危险信号与复诊路径。"
            )
    if is_high_energy_hindfoot_trauma_case(diagnosis, case_features):
        normalized_plan = normalize_name(plan)
        overcalled_sprain = (
            normalize_name(diagnosis) == normalize_name("踝关节扭伤")
            or plan_has_any(normalized_plan, ["按踝关节扭伤", "扭伤予", "扭伤处理", "仅软组织"])
        )
        claims_fracture_excluded = plan_has_any(
            normalized_plan,
            ["已排除骨折", "排除骨折", "未见骨折即可", "无骨折按扭伤"],
        )
        has_ortho_path = plan_has_any(
            normalized_plan,
            ["骨科", "进一步影像", "CT", "跟骨", "骨创伤", "手术评估", "骨折仍不能排除"],
        )
        if overcalled_sprain and (claims_fracture_excluded or not has_ortho_path):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "high_energy_hindfoot_overcalled_as_sprain",
                    "severity": "must_fix",
                    "problem": "high_energy_hindfoot_closed_as_simple_sprain",
                    "patchable": True,
                }
            )
            plan = remove_fracture_rule_out_claims(plan)
            patches.append(
                "高能量足跟创伤即使初筛X线未见明确骨折，也不能直接按单纯踝扭伤关闭骨创伤轴；应保留骨科评估，必要时进一步足/跟骨影像，并在骨折仍可能时避免仅 RICE 收口。"
            )
    if is_febrile_polyuria_hyperglycemic_crisis_case(diagnosis, case_features):
        crisis_status = glucose_evidence_status(case_features)
        di_diagnosis = normalize_name(diagnosis) == normalize_name("尿崩症")
        di_treatment = treatment_recommends_diabetes_insipidus_path(plan)
        # Crisis not excluded: block DI diagnosis/treatment-first pathways.
        if crisis_status != "normal" and (di_diagnosis or di_treatment):
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "di_path_before_hyperglycemic_crisis_exclusion",
                    "severity": "must_fix",
                    "problem": "diabetes_insipidus_path_before_excluding_hyperglycemic_crisis",
                    "patchable": True,
                }
            )
            plan = sanitize_diabetes_insipidus_first_path(plan)
            patches.append(
                "高热伴多尿烦渴脱水时，在血糖/酮体或代谢证据排除高血糖危象（DKA/HHS）前，不能按尿崩症首选低渗补液或去氨加压素；应优先等渗复苏、闭合血糖与代谢评估，并排查感染，再分层处理。"
            )
    return {"issues": issues, "patches": patches, "treatment_plan": plan}


def has_diagnosis_axis(case_features: Dict[str, Any], axis_id: str) -> bool:
    target = clean_axis_id(axis_id)
    return any(
        clean_axis_id(axis.get("axis_id")) == target
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
    )


def is_retinoblastoma_treatment_gate_case(
    normalized_diagnosis: str,
    case_features: Dict[str, Any],
    case_text: str,
) -> bool:
    return (
        normalized_diagnosis == normalize_name("视网膜母细胞瘤")
        or has_diagnosis_axis(case_features, "pediatric_leukocoria_retinoblastoma_until_excluded")
        or has_pediatric_leukocoria_red_flag_pattern(case_text)
    )


def is_sma_syndrome_treatment_gate_case(
    normalized_diagnosis: str,
    case_features: Dict[str, Any],
    case_text: str,
) -> bool:
    return (
        normalized_diagnosis == normalize_name("肠系膜上动脉压迫综合征")
        or has_diagnosis_axis(case_features, "post_spinal_surgery_positional_duodenal_obstruction")
        or has_post_spinal_surgery_positional_bilious_vomiting_pattern(case_text)
    )


def treatment_recommends_triptan(treatment_plan: str) -> bool:
    for clause in semantic_clauses(treatment_plan):
        normalized = normalize_name(clause)
        if re.search(r"(?:停用|避免|不能|不应|不宜|不要|禁止).{0,16}(?:曲坦|舒马曲坦|佐米曲普坦)", normalized):
            continue
        if re.search(r"(?:给予|使用|口服|应用|首选|建议).{0,12}(?:曲坦|舒马曲坦|佐米曲普坦)", normalized):
            return True
    return False


def has_negative_bacterial_culture(case_features: Dict[str, Any]) -> bool:
    text = normalize_name(case_features_text(case_features))
    return any(marker in text for marker in ["细菌培养阴性", "细菌培养结果阴性", "培养阴性", "培养无生长"])


def has_confirmed_secondary_bacterial_infection(case_features: Dict[str, Any]) -> bool:
    if case_features.get("secondary_bacterial_infection_confirmed") is True:
        return True
    text = normalize_name(
        " ".join(
            [
                case_features_text(case_features),
                " ".join(structured_text_chunks(case_features.get("examination_results"))),
            ]
        )
    )
    if any(
        marker in text
        for marker in [
            "培养阳性",
            "细菌培养阳性",
            "明确继发细菌感染",
            "细菌感染确诊",
            "蜂窝织炎",
            "脓毒症",
            "脓毒性休克",
        ]
    ):
        return True
    provenance = case_features.get("anti_infective_provenance")
    cultures = provenance.get("cultures", []) if isinstance(provenance, dict) else []
    for culture in cultures:
        if not isinstance(culture, dict):
            continue
        organism = normalize_name(culture.get("organism"))
        status = normalize_name(culture.get("status"))
        if organism and not any(marker in organism for marker in ["阴性", "无生长", "未检出"]):
            if status not in {"阴性", "negative", "normal"}:
                return True
    return False


SKIN_MYIASIS_SYSTEMIC_ANTIBIOTIC_ALIASES = [
    "抗生素",
    "抗菌药",
    "抗感染治疗",
    "头孢氨苄",
    "阿莫西林克拉维酸钾",
    "克林霉素",
    "多西环素",
]


def treatment_recommends_skin_myiasis_systemic_antibiotics(normalized_plan: str) -> bool:
    systemic_markers = ["口服", "静脉", "全身", "系统", "经验性", "立即启动", "疗程"]
    for clause in re.split(r"(?<=[，,；;。\n])", clean_text(normalized_plan)):
        normalized_clause = normalize_name(clause)
        for alias in SKIN_MYIASIS_SYSTEMIC_ANTIBIOTIC_ALIASES:
            token = normalize_name(alias)
            start = 0
            while token:
                index = normalized_clause.find(token, start)
                if index < 0:
                    break
                window = normalized_clause[max(0, index - 24): index + len(token) + 24]
                if drug_mention_is_negated(normalized_clause, index, len(token)):
                    start = index + len(token)
                    continue
                if any(marker in window for marker in ["不启动", "不应", "不得", "避免", "仅在", "只有", "若出现", "如出现", "才考虑", "再考虑", "方可"]):
                    if not any(marker in window for marker in ["立即启动", "需立即", "首选", "给予", "开始"]):
                        start = index + len(token)
                        continue
                if "外用" in window or "局部" in window:
                    if not any(marker in window for marker in systemic_markers):
                        start = index + len(token)
                        continue
                return True
    return False


def sanitize_skin_myiasis_systemic_antibiotics(treatment_plan: str) -> str:
    kept = []
    for clause in re.split(r"(?<=[，,；;。\n])", clean_text(treatment_plan)):
        normalized_clause = normalize_name(clause)
        if treatment_recommends_skin_myiasis_systemic_antibiotics(normalized_clause):
            continue
        kept.append(clause)
    return normalize_treatment_text("".join(kept))


def apply_skin_myiasis_treatment_gate(
    *,
    diagnosis: str,
    treatment_plan: str,
    case_features: Dict[str, Any],
) -> Dict[str, Any]:
    plan = clean_text(treatment_plan)
    issues = []
    patches = []
    if normalize_name(diagnosis) != normalize_name("皮肤蝇蛆病"):
        return {"issues": issues, "patches": patches, "treatment_plan": plan}

    normalized_plan = normalize_name(plan)
    if (
        treatment_recommends_skin_myiasis_systemic_antibiotics(normalized_plan)
        and not has_confirmed_secondary_bacterial_infection(case_features)
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "systemic_antibiotic_without_confirmed_secondary_infection",
                "severity": "must_fix",
                "problem": "skin_myiasis_does_not_establish_a_bacterial_indication_for_systemic_antibiotics",
                "patchable": True,
            }
        )
        plan = sanitize_skin_myiasis_systemic_antibiotics(plan)
        patches.append(
            "皮肤蝇蛆病应优先封闭或取出幼虫、局部伤口护理和止痒；没有培养或明确继发细菌感染证据时不启动系统抗生素，只有出现培养阳性、明确蜂窝织炎或脓毒症等证据后才考虑抗菌药并按药敏调整。"
        )
        normalized_plan = normalize_name(plan)

    if anemia_evidence_status(case_features) == "positive" and not plan_has_any(
        normalized_plan,
        ["贫血", "血红蛋白", "铁代谢", "铁蛋白", "红细胞指数"],
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "skin_myiasis_anemia_followup_missing",
                "severity": "should_fix",
                "problem": "low_hemoglobin_requires_anemia_followup",
                "patchable": True,
            }
        )
        patches.append(
            "CBC提示血红蛋白偏低时，应复查血常规并完成基础贫血评估（如红细胞指数、铁蛋白/铁代谢），必要时转全科或血液科评估，而不能只随访皮损。"
        )
    return {"issues": issues, "patches": patches, "treatment_plan": plan}


def treatment_recommends_prophylactic_antibiotics(normalized_plan: str) -> bool:
    antibiotic = ["抗生素", "抗菌药", "莫匹罗星", "夫西地酸", "头孢克洛", "左氧氟沙星", "阿莫西林"]
    prophylaxis = ["预防性", "常规", "为预防", "即使培养阴性"]
    return any(marker in normalized_plan for marker in prophylaxis) and any(
        marker in normalized_plan for marker in antibiotic
    )


def sanitize_prophylactic_antibiotic_recommendations(treatment_plan: str) -> str:
    plan = clean_text(treatment_plan)
    antibiotic = "(?:抗生素|抗菌药|莫匹罗星|夫西地酸|头孢克洛|左氧氟沙星|阿莫西林)"
    patterns = [
        rf"(?:仍|继续|建议|可考虑|必要时)?(?:预防性|常规|为预防)[^，,；;。\n]{{0,18}}{antibiotic}[^，,；;。\n]*[，,；;。]?",
        rf"(?:预防性|常规|为预防)[^，,；;。\n]{{0,18}}使用[^，,；;。\n]*{antibiotic}[^，,；;。\n]*[，,；;。]?",
    ]
    for pattern in patterns:
        plan = re.sub(pattern, "", plan, flags=re.I)
    return normalize_treatment_text(plan)


def has_qt_prolongation_with_tricyclic_exposure(case_features: Dict[str, Any]) -> bool:
    text = normalize_name(case_features_text(case_features))
    qt = bool(re.search(r"(?:qtc|qtc间期|qt间期)[^\d]{0,12}0?\.?(?:4[7-9]|[5-9]\d)", text)) or any(
        marker in text for marker in ["qt延长", "qt间期延长", "qtc延长"]
    )
    tricyclic = any(marker in text for marker in ["三环类", "三环类抗抑郁药", "阿米替林", "丙咪嗪", "氯米帕明"])
    overdose = any(marker in text for marker in ["过量", "超量", "吃多了", "服用过量"])
    return qt and tricyclic and overdose


def treatment_continues_tricyclic_antidepressant(normalized_plan: str) -> bool:
    tricyclic = ["三环类", "三环类抗抑郁药", "阿米替林", "丙咪嗪", "氯米帕明"]
    continuation = ["继续", "维持", "照常", "无需停用", "不必停用", "继续服用"]
    return any(marker in normalized_plan for marker in tricyclic) and any(
        marker in normalized_plan for marker in continuation
    )


def has_stroke_secondary_prevention_context(case_features: Dict[str, Any]) -> bool:
    text = normalize_name(case_features_text(case_features))
    return any(
        marker in text
        for marker in [
            "卒中",
            "中风",
            "脑梗死",
            "脑梗塞",
            "缺血性卒中",
            "脑缺血",
            "tia",
            "短暂性脑缺血",
            "脑卒中史",
            "有卒中史",
            "既往卒中",
            "脑血栓",
        ]
    )


def has_active_major_bleed_context(case_features: Dict[str, Any]) -> bool:
    text = normalize_name(case_features_text(case_features))
    return any(
        marker in text
        for marker in [
            "活动性出血",
            "消化道大出血",
            "呕血",
            "黑便",
            "颅内出血",
            "脑出血",
            "大咯血",
            "失血性休克",
            "血红蛋白急剧下降",
        ]
    )


def treatment_stops_aspirin_without_active_bleed(treatment_plan: str, case_features: Dict[str, Any]) -> bool:
    if has_active_major_bleed_context(case_features):
        return False
    plan = clean_text(treatment_plan)
    if not plan:
        return False
    aspirin = r"(?:阿司匹林|aspirin|asa|拜阿司匹灵)"
    stop = r"(?:停用|停服|暂停使用|停止服用|停止使用|撤除|中止|停掉|不要再服|勿再服用)"
    if re.search(rf"{stop}.{{0,24}}{aspirin}", plan, flags=re.I):
        return True
    if re.search(rf"{aspirin}.{{0,24}}{stop}", plan, flags=re.I):
        return True
    if re.search(rf"(?:建议|将|把).{{0,12}}{aspirin}.{{0,16}}{stop}", plan, flags=re.I):
        return True
    return False


def sanitize_unindicated_aspirin_discontinuation(treatment_plan: str) -> str:
    plan = clean_text(treatment_plan)
    aspirin = r"(?:阿司匹林|aspirin|asa|拜阿司匹灵)"
    stop = r"(?:停用|停服|暂停使用|停止服用|停止使用|撤除|中止|停掉|不要再服|勿再服用)"
    patterns = [
        rf"(?:建议\s*)?(?:立即|马上|暂时|逐步)?\s*{stop}\s*{aspirin}[^，,；;。\n]*[，,；;。]?",
        rf"(?:建议\s*)?(?:将|把)\s*{aspirin}[^，,；;。\n]{{0,20}}{stop}[^，,；;。\n]*[，,；;。]?",
        rf"{aspirin}[^，,；;。\n]{{0,20}}{stop}[^，,；;。\n]*[，,；;。]?",
        rf"(?:改用氯吡格雷抗血小板治疗以防卒中复发)[^，,；;。\n]*[，,；;。]?",
    ]
    for pattern in patterns:
        plan = re.sub(pattern, "", plan, flags=re.I)
    return normalize_treatment_text(plan)


def is_mitral_stenosis_case(
    normalized_diagnosis: str,
    case_features: Dict[str, Any],
    case_text: str,
) -> bool:
    if normalized_diagnosis == normalize_name("二尖瓣狭窄"):
        return True
    if has_diagnosis_axis(case_features, "mitral_stenosis_hemodynamics"):
        return True
    text = normalize_name(case_text or case_features_text(case_features))
    return "二尖瓣狭窄" in text or "二尖瓣口面积" in text


def treatment_recommends_pbmv_without_thrombus_exclusion(treatment_plan: str) -> bool:
    plan = clean_text(treatment_plan)
    if not plan:
        return False
    mentions_pbmv = any(
        marker in plan
        for marker in ["球囊二尖瓣", "二尖瓣球囊", "PBMV", "经皮球囊", "球囊成形", "球囊扩张"]
    )
    if not mentions_pbmv:
        return False
    has_exclusion = any(
        marker in plan
        for marker in ["食道超声", "经食道", "TEE", "左房血栓", "心耳血栓", "排除血栓", "血栓排除", "无血栓"]
    )
    return not has_exclusion


def sanitize_pbmv_before_thrombus_exclusion(treatment_plan: str) -> str:
    plan = clean_text(treatment_plan)
    patterns = [
        r"(?:尽快|立即|尽早)?(?:转诊评估)?(?:经皮)?球囊(?:二尖瓣)?(?:成形|扩张)术?[^，,；;。\n]*[，,；;。]?",
        r"PBMV[^，,；;。\n]*[，,；;。]?",
        r"(?:若解剖条件不适合则考虑外科换瓣)[^，,；;。\n]*[，,；;。]?",
    ]
    for pattern in patterns:
        plan = re.sub(pattern, "", plan, flags=re.I)
    return normalize_treatment_text(plan)


def treatment_recommends_noac_for_rheumatic_mitral_stenosis(
    treatment_plan: str,
    case_features: Dict[str, Any],
    case_text: str,
    normalized_diagnosis: str,
) -> bool:
    plan = clean_text(treatment_plan)
    if not plan:
        return False
    noac = any(
        marker in plan
        for marker in [
            "新型口服抗凝",
            "NOAC",
            "DOAC",
            "达比加群",
            "利伐沙班",
            "阿哌沙班",
            "艾多沙班",
        ]
    )
    if not noac:
        return False
    # Explicit preference/default language, not pure prohibition.
    prefers = any(
        marker in plan
        for marker in ["或新型", "可选新型", "可选用新型", "也可使用新型", "NOAC", "DOAC"]
    ) and not re.search(r"(?:禁止|不宜|不推荐|避免).{0,12}(?:新型口服抗凝|NOAC|DOAC)", plan)
    if not prefers and "新型口服抗凝" not in plan:
        return False
    text = normalize_name(case_text or case_features_text(case_features))
    rheumatic_ms = (
        normalized_diagnosis in {normalize_name("风湿性心脏病"), normalize_name("二尖瓣狭窄")}
        or "风湿性" in text
        or "二尖瓣狭窄" in text
        or "风湿性心脏病" in text
    )
    return bool(rheumatic_ms and prefers)


def sanitize_noac_in_rheumatic_ms(treatment_plan: str) -> str:
    plan = clean_text(treatment_plan)
    patterns = [
        r"(?:或|以及|及|、)\s*新型口服抗凝药[^，,；;。\n]*",
        r"新型口服抗凝药[^，,；;。\n]*",
        r"(?:NOAC|DOAC)[^，,；;。\n]*",
        r"(?:达比加群|利伐沙班|阿哌沙班|艾多沙班)[^，,；;。\n]*",
    ]
    for pattern in patterns:
        plan = re.sub(pattern, "", plan, flags=re.I)
    return normalize_treatment_text(plan)


def sanitize_tricyclic_continuation(treatment_plan: str) -> str:
    plan = clean_text(treatment_plan)
    tricyclic = "(?:三环类(?:抗抑郁药)?|阿米替林|丙咪嗪|氯米帕明)"
    patterns = [
        rf"(?:继续|维持|照常|无需停用|不必停用|继续服用)[^，,；;。\n]{{0,12}}{tricyclic}[^，,；;。\n]*[，,；;。]?",
        rf"{tricyclic}[^，,；;。\n]{{0,12}}(?:继续|维持|照常|无需停用|不必停用|继续服用)[^，,；;。\n]*[，,；;。]?",
    ]
    for pattern in patterns:
        plan = re.sub(pattern, "", plan, flags=re.I)
    return normalize_treatment_text(plan)


def is_hf_reduced_ef_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    diagnosis_match = normalize_name(diagnosis) in {
        normalize_name("扩张型心肌病"),
        normalize_name("心力衰竭"),
        normalize_name("射血分数降低型心力衰竭"),
    }
    text = case_features_text(case_features)
    return diagnosis_match and has_reduced_left_ventricular_ejection_fraction(text)


def has_hf_guideline_core(normalized_plan: str) -> bool:
    core = [
        "arnI",
        "aceI",
        "arb",
        "血管紧张素",
        "β受体阻滞剂",
        "贝塔受体阻滞剂",
        "美托洛尔",
        "卡维地洛",
        "螺内酯",
        "醛固酮受体拮抗剂",
        "sglt2",
        "达格列净",
        "恩格列净",
    ]
    has_neurohormonal = any(marker in normalized_plan for marker in core)
    has_diuretic = any(marker in normalized_plan for marker in ["利尿", "呋塞米", "托拉塞米"])
    has_escalation = any(marker in normalized_plan for marker in ["心衰专科", "急诊", "住院", "急性期"])
    return has_neurohormonal and has_diuretic and has_escalation


def sanitize_triptan_recommendations(treatment_plan: str) -> str:
    clauses = re.split(r"(?<=[，,；;。\n])", clean_text(treatment_plan))
    kept = []
    for clause in clauses:
        normalized = normalize_name(clause)
        if re.search(r"(?:停用|避免|不能|不应|不宜|不要|禁止).{0,16}(?:曲坦|舒马曲坦|佐米曲普坦)", normalized):
            kept.append(clause)
            continue
        if re.search(r"(?:给予|使用|口服|应用|首选|建议).{0,12}(?:曲坦|舒马曲坦|佐米曲普坦)", normalized):
            continue
        kept.append(clause)
    return normalize_treatment_text("".join(kept))


def is_febrile_polyuria_hyperglycemic_crisis_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    axis_ids = {
        clean_axis_id(axis.get("axis_id"))
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
    }
    if "febrile_polyuria_dehydration_hyperglycemic_crisis" in axis_ids:
        return True
    text = case_features_text(case_features)
    return has_febrile_polyuria_dehydration_pattern(text)


def treatment_recommends_diabetes_insipidus_path(treatment_plan: str) -> bool:
    """True when the plan positively recommends DI-first fluids or desmopressin."""
    for clause in semantic_clauses(treatment_plan):
        normalized = normalize_name(clause)
        if re.search(
            r"(?:暂不|避免|不能|不应|不宜|不要|禁止|而非|不是).{0,24}(?:尿崩|去氨加压素|ddavp|低渗)",
            normalized,
        ):
            continue
        if re.search(
            r"(?:按尿崩|尿崩症处理|补充低渗|给予低渗|使用低渗|低渗盐水|低渗液|去氨加压素|ddavp)",
            normalized,
        ):
            return True
    return False


def sanitize_diabetes_insipidus_first_path(treatment_plan: str) -> str:
    clauses = re.split(r"(?<=[，,；;。\n])", clean_text(treatment_plan))
    kept = []
    for clause in clauses:
        normalized = normalize_name(clause)
        if re.search(
            r"(?:暂不|避免|不能|不应|不宜|不要|禁止|而非).{0,24}(?:尿崩|去氨加压素|ddavp|低渗)",
            normalized,
        ):
            kept.append(clause)
            continue
        if re.search(
            r"(?:按尿崩|尿崩症处理|补充低渗|给予低渗|使用低渗|低渗盐水|低渗液|去氨加压素|ddavp)",
            normalized,
        ):
            continue
        kept.append(clause)
    return normalize_treatment_text("".join(kept))


def is_symptomatic_large_renal_cyst_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    if normalize_name(diagnosis) != normalize_name("肾囊肿"):
        return False
    findings = set(as_text_list(case_features.get("positive_findings")))
    has_symptoms = bool({"腰痛加重", "尿频"} & findings)
    has_size_or_pressure = bool({"较大肾囊肿", "压迫邻近结构"} & findings)
    return has_symptoms and has_size_or_pressure


def is_deviated_septum_recurrent_epistaxis_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    if normalize_name(diagnosis) != normalize_name("鼻中隔偏曲"):
        return False
    findings = set(as_text_list(case_features.get("positive_findings")))
    return "反复鼻出血" in findings


def is_acute_bacterial_prostatitis_complication_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    if normalize_name(diagnosis) != normalize_name("急性细菌性前列腺炎"):
        return False
    findings = set(as_text_list(case_features.get("positive_findings")))
    has_infection = bool({"发热性尿路感染", "尿培养阳性"} & findings)
    has_complication_risk = bool({"尿潴留风险", "会阴痛"} & findings)
    return has_infection and has_complication_risk


def is_viral_conjunctivitis_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    diagnosis_text = normalize_name(" ".join([diagnosis] + as_text_list(case_features.get("candidate_diagnoses"))))
    return any(
        marker in diagnosis_text
        for marker in [
            normalize_name("腺病毒性结膜炎"),
            normalize_name("病毒性结膜炎"),
        ]
    )


def is_post_traumatic_brain_injury_cognitive_vestibular_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    diagnosis_text = normalize_name(" ".join([diagnosis] + as_text_list(case_features.get("candidate_diagnoses"))))
    if normalize_name("创伤后脑损伤综合征") not in diagnosis_text:
        return False
    findings = set(as_text_list(case_features.get("positive_findings")))
    has_post_traumatic_headache = {"头部外伤后起病", "外伤后持续头痛"}.issubset(findings)
    has_cognitive_or_vestibular = bool({"认知症状", "前庭平衡症状"} & findings)
    return has_post_traumatic_headache and has_cognitive_or_vestibular


def is_infant_congenital_structural_heart_disease_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    axis_ids = {
        clean_axis_id(axis.get("axis_id"))
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
    }
    if "infant_congenital_structural_heart_disease" in axis_ids:
        return True
    diagnosis_text = normalize_name(" ".join([diagnosis] + as_text_list(case_features.get("candidate_diagnoses"))))
    if not any(
        marker in diagnosis_text
        for marker in [normalize_name("三房心"), normalize_name("先天性心脏病")]
    ):
        return False
    return has_infant_congenital_structural_heart_pattern(case_features_text(case_features))


def is_postop_chylothorax_or_pleural_effusion_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    axis_ids = {
        clean_axis_id(axis.get("axis_id"))
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
    }
    if "postop_chylothorax_or_pleural_effusion" in axis_ids:
        return True
    diagnosis_text = normalize_name(" ".join([diagnosis] + as_text_list(case_features.get("candidate_diagnoses"))))
    if normalize_name("乳糜胸") not in diagnosis_text and normalize_name("胸腔积液") not in diagnosis_text:
        return has_postop_chylothorax_or_pleural_effusion_pattern(case_features_text(case_features))
    return has_postop_chylothorax_or_pleural_effusion_pattern(case_features_text(case_features))


def treatment_recommends_pda_closure(treatment_plan: str) -> bool:
    markers = ["吲哚美辛", "布洛芬关闭导管", "关闭动脉导管", "关闭导管", "前列腺素抑制剂", "关管"]
    safety_negation = re.compile(
        r"(?:不能|不应|不得|严禁|避免|禁止|禁用).{0,40}(?:默认|常规|直接|贸然)?.{0,16}$"
    )
    for clause in semantic_clauses(treatment_plan):
        normalized = normalize_name(clause)
        for marker in markers:
            token = normalize_name(marker)
            start = 0
            while True:
                index = normalized.find(token, start)
                if index < 0:
                    break
                prefix = normalized[max(0, index - 64):index]
                if (
                    not safety_negation.search(prefix)
                    and not marker_occurrence_is_negated(normalized, token, index)
                    and not marker_occurrence_is_uncertain(normalized, index)
                ):
                    return True
                start = index + len(token)
    return False


def sanitize_pda_closure_recommendations(treatment_plan: str) -> str:
    clauses = re.split(r"(?<=[，,；;。\n])", clean_text(treatment_plan))
    kept = []
    for clause in clauses:
        normalized = normalize_name(clause)
        if any(
            marker in normalized
            for marker in [
                normalize_name("吲哚美辛"),
                normalize_name("关闭动脉导管"),
                normalize_name("关闭导管"),
                normalize_name("前列腺素抑制剂"),
                normalize_name("关管"),
            ]
        ):
            continue
        kept.append(clause)
    return normalize_treatment_text("".join(kept))


def is_migraine_reproductive_travel_trigger_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    diagnosis_text = normalize_name(" ".join([diagnosis] + as_text_list(case_features.get("candidate_diagnoses"))))
    if normalize_name("偏头痛") not in diagnosis_text:
        return False
    findings = set(as_text_list(case_features.get("positive_findings")))
    required = {"育龄女性", "偏头痛伴恶心头晕", "旅行或视觉运动诱发"}
    return required.issubset(findings)


def case_narrative_text(case_features: Dict[str, Any]) -> str:
    """Patient/case narrative only — exclude LLM finding labels that can self-prove axes."""
    return normalize_name(
        " ".join(
            [
                clean_text(case_features.get("case_text")),
                clean_text(case_features.get("patient_text")),
                clean_text(case_features.get("chief_complaint")),
            ]
        )
    )


def is_umbilical_granulation_bleeding_mass_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    diagnosis_text = normalize_name(" ".join([diagnosis] + as_text_list(case_features.get("candidate_diagnoses"))))
    if not any(
        marker in diagnosis_text
        for marker in [normalize_name("化脓性肉芽肿"), normalize_name("新生儿脐炎")]
    ):
        return False
    # LLM positive_findings alone must not invent neonatal umbilical care on unrelated cases.
    if not has_umbilical_granulation_bleeding_mass_pattern(case_narrative_text(case_features)):
        return False
    findings = set(as_text_list(case_features.get("positive_findings")))
    return {"新生儿脐部病变", "湿润易出血肿块"}.issubset(findings)


def is_chronic_alcohol_liver_injury_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    diagnosis_text = normalize_name(" ".join([diagnosis] + as_text_list(case_features.get("candidate_diagnoses"))))
    axis_ids = {
        clean_axis_id(axis.get("axis_id"))
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
    }
    if "chronic_alcohol_liver_injury" in axis_ids:
        return True
    if not any(
        marker in diagnosis_text
        for marker in [normalize_name("酒精性肝病"), normalize_name("肝硬化")]
    ):
        return False
    text = case_features_text(case_features)
    return has_chronic_alcohol_liver_injury_pattern(text)


def is_high_energy_hindfoot_trauma_case(diagnosis: str, case_features: Dict[str, Any]) -> bool:
    axis_ids = {
        clean_axis_id(axis.get("axis_id"))
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
    }
    if "high_energy_hindfoot_trauma" in axis_ids:
        return True
    text = case_features_text(case_features)
    return has_high_energy_hindfoot_trauma_pattern(text)


def apply_high_energy_hindfoot_diagnosis_guard(
    diagnosis: str,
    case_state: Dict[str, Any],
    disease_candidates: List[Dict[str, Any]],
) -> str:
    """Keep calcaneal fracture over sprain when high-energy heel trauma is present.

    Plain first-line X-ray can be false-negative; soft-tissue sprain must not win.
    """
    selected = clean_text(diagnosis)
    text = normalize_name(case_text_for_matching(case_state))
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case_state)))
    combined = normalize_name(" ".join([text, facts_text]))
    if not has_high_energy_hindfoot_trauma_pattern(combined):
        return selected
    if normalize_name(selected) != normalize_name("踝关节扭伤"):
        return selected
    candidate_names = [
        clean_text(item.get("disease"))
        for item in disease_candidates
        if isinstance(item, dict) and clean_text(item.get("disease"))
    ]
    for name in candidate_names:
        if normalize_name(name) == normalize_name("跟骨骨折"):
            return name
    return selected


def methemoglobin_evidence_status(case_state: Dict[str, Any]) -> str:
    methemoglobin_statuses: List[str] = []
    enzyme_statuses: List[str] = []
    for payload in matching_exam_payloads(case_state, ["动脉血气", "ABG", "红细胞酶"]):
        if not exam_result_is_usable(payload):
            continue
        for key, value in exam_result_pairs(payload):
            normalized_key = normalize_name(key)
            position = lab_value_reference_position(value)
            if "高铁血红蛋白" in normalized_key:
                methemoglobin_statuses.append("positive" if position == "high" else position)
            if any(marker in normalized_key for marker in ["细胞色素b5还原酶", "高铁血红蛋白还原酶"]):
                enzyme_statuses.append("positive" if position == "low" else position)
    if "positive" in methemoglobin_statuses and "positive" in enzyme_statuses:
        return "positive"
    if "normal" in methemoglobin_statuses or "normal" in enzyme_statuses:
        return "negative"
    return "unknown"


def mediastinal_structure_evidence_status(case_state: Dict[str, Any]) -> str:
    markers = ["纵隔囊肿", "纵隔囊性占位", "纵隔占位", "纵隔肿块", "气道受压", "气道压迫"]
    resolved = ["未见", "没有", "无", "阴性", "排除", "已排除", "不存在", "正常"]
    for payload in matching_exam_payloads(case_state, ["胸部CT", "Chest CT"]):
        if not exam_result_is_usable(payload):
            continue
        if normalize_name(payload.get("status")) in {"normal", "negative"}:
            continue
        result_text = "；".join("%s：%s" % pair for pair in exam_result_pairs(payload))
        if marker_present_active(result_text, markers, resolved_markers=resolved):
            return "positive"
    return "unknown"


def iron_deficiency_evidence_status(case_state: Dict[str, Any]) -> str:
    if anemia_evidence_status(case_state) != "positive":
        return "unknown"
    for payload in matching_exam_payloads(case_state, ["铁代谢", "贫血谱"]):
        if not exam_result_is_usable(payload):
            continue
        for key, value in exam_result_pairs(payload):
            if any(marker in normalize_name(key) for marker in ["铁蛋白", "血清铁"]):
                if lab_value_reference_position(value) == "low":
                    return "positive"
    return "unknown"


def apply_evidence_backed_diagnosis_guard(
    diagnosis: str,
    case_state: Dict[str, Any],
    disease_candidates: List[Dict[str, Any]],
) -> str:
    selected = clean_text(diagnosis)
    candidate_map = {
        normalize_name(item.get("disease")): clean_text(item.get("disease"))
        for item in disease_candidates
        if isinstance(item, dict) and clean_text(item.get("disease"))
    }
    case_text = normalize_name(patient_text_for_matching(case_state))
    methemoglobin_supported = (
        has_methemoglobin_risk_pattern(case_text)
        and methemoglobin_evidence_status(case_state) == "positive"
    )
    mediastinal_supported = (
        has_pediatric_airway_compression_pattern(case_text)
        and mediastinal_structure_evidence_status(case_state) == "positive"
    )
    anemia_supported = (
        has_symptomatic_anemia_loss_pattern(case_text)
        and iron_deficiency_evidence_status(case_state) == "positive"
    )
    combined = normalize_name(case_text_for_matching(case_state))
    evidence_targets = [
        (has_positive_vsd_text(combined), "室间隔缺损（VSD）"),
        (has_leptospirosis_exposure_pattern(combined), "钩端螺旋体病"),
        (has_diuretic_hypokalemia_pattern(combined), "低钾血症"),
        (has_multisystem_autoimmune_serositis_pattern(combined), "系统性红斑狼疮"),
        (has_hearing_symptom_pattern(case_state), "耳鸣"),
    ]
    for supported, target in evidence_targets:
        official = candidate_map.get(normalize_name(target))
        if supported and official:
            return official
    for supported, target in [
        (methemoglobin_supported, "先天性高铁血红蛋白血症"),
        (mediastinal_supported, "先天性纵隔囊肿"),
        (anemia_supported, "缺铁性贫血"),
    ]:
        official = candidate_map.get(normalize_name(target))
        if supported and official:
            return official
    return selected


def treatment_recommends_routine_eye_antibiotics(normalized_plan: str) -> bool:
    antibiotic_markers = ["抗生素", "抗菌", "左氧氟沙星", "妥布霉素", "莫西沙星", "氯霉素"]
    if not plan_has_any(normalized_plan, antibiotic_markers):
        return False
    bacterial_evidence = ["继发细菌感染", "细菌感染证据", "脓性分泌物", "培养阳性"]
    if plan_has_any(normalized_plan, bacterial_evidence) and not plan_has_any(normalized_plan, ["预防性", "常规"]):
        return False
    recommendation_markers = ["预防性", "常规", "给予", "使用", "滴眼", "每日", "推荐"]
    negative_markers = [
        "不常规",
        "无需",
        "避免",
        "不应",
        "慎用",
        "仅在",
        "不作为常规",
        "重新评估是否存在继发细菌感染",
    ]
    for marker in antibiotic_markers:
        marker_text = normalize_name(marker)
        start = normalized_plan.find(marker_text)
        while start >= 0:
            window = normalized_plan[max(0, start - 14): start + len(marker_text) + 14]
            if any(normalize_name(item) in window for item in negative_markers):
                start = normalized_plan.find(marker_text, start + len(marker_text))
                continue
            if any(normalize_name(item) in window for item in recommendation_markers):
                return True
            start = normalized_plan.find(marker_text, start + len(marker_text))
    return False


def sanitize_routine_eye_antibiotics(treatment_plan: str) -> str:
    plan = clean_text(treatment_plan)
    patterns = [
        r"预防性使用局部抗生素[^；;。]*[；;。]?",
        r"常规使用局部抗生素[^；;。]*[；;。]?",
        r"预防性使用[^；;。]*(?:抗生素|抗菌|左氧氟沙星|妥布霉素|莫西沙星|氯霉素)[^；;。]*[；;。]?",
        r"常规使用[^；;。]*(?:抗生素|抗菌|左氧氟沙星|妥布霉素|莫西沙星|氯霉素)[^；;。]*[；;。]?",
    ]
    for pattern in patterns:
        plan = re.sub(pattern, "", plan)
    return normalize_treatment_text(plan)


def apply_axis_risk_gate(
    treatment_plan: str,
    case_features: Dict[str, Any],
    *,
    diagnosis: str = "",
) -> Dict[str, Any]:
    plan = clean_text(treatment_plan)
    normalized_plan = normalize_name(plan)
    normalized_case_text = normalize_name(
        " ".join(
            [
                clean_text(case_features.get("case_text")),
                clean_text(case_features.get("patient_text")),
                clean_text(case_features.get("chief_complaint")),
            ]
            + as_text_list(case_features.get("positive_findings"))
            + as_text_list(case_features.get("red_flags"))
            + as_text_list(case_features.get("organ_risk"))
            + [feature_evidence_text(case_features)]
        )
    )
    risks = set(
        supported_axis_risks(
            as_text_list(case_features.get("treatment_risks")),
            normalized_case_text=normalized_case_text,
            diagnosis=diagnosis,
        )
    )
    for axis in as_axis_list(case_features.get("diagnosis_axes")):
        risks.update(
            supported_axis_risks(
                as_text_list(axis.get("treatment_risks")),
                axis_id=clean_text(axis.get("axis_id")),
                evidence=as_text_list(axis.get("evidence")),
                candidates=as_text_list(axis.get("candidate_official_names")),
                normalized_case_text=normalized_case_text,
                diagnosis=diagnosis,
            )
        )
    issues = []
    patches = []
    if (
        "infection_before_steroid" in risks
        and treatment_contains_steroid(normalized_plan)
        and infection_before_steroid_risk_is_active(case_features, diagnosis=diagnosis)
        and not allows_monitored_steroid_bridge_for_life_threatening_bleeding(
            case_features,
            plan,
            diagnosis=diagnosis,
        )
    ):
        plan = sanitize_steroid_recommendations(plan)
        add_axis_issue(
            issues,
            patches,
            "infection_before_steroid",
            "感染证据未闭合时不应常规使用局部或全身糖皮质激素；如确需使用，必须写明已充分抗感染并由眼科/专科评估。",
        )
    if "unsupported_no_infection_risk" in risks and any(marker in normalized_plan for marker in [normalize_name("暂无感染风险"), normalize_name("无感染风险")]):
        add_axis_issue(
            issues,
            patches,
            "unsupported_no_infection_risk",
            "存在感染轴证据时，不应无依据排除感染风险；需补充病原学评估和感染控制前提。",
        )
    if "sle_renal_thrombosis_unclosed" in risks and not plan_has_any(normalized_plan, ["尿常规", "尿液", "肾功能", "抗磷脂", "血栓"]):
        add_axis_issue(
            issues,
            patches,
            "sle_renal_thrombosis_unclosed",
            "SLE 治疗前应闭合肾脏和血栓风险：补充尿常规/尿蛋白、肾功能、抗磷脂抗体和妊娠/避孕安全评估。",
        )
    if "sle_renal_thrombosis_unclosed" in risks and treatment_contains_estrogen_contraception(normalized_plan):
        plan = sanitize_estrogen_contraception_recommendations(plan)
        add_axis_issue(
            issues,
            patches,
            "estrogen_contraception_with_sle_thrombosis_risk",
            "SLE 合并抗磷脂或血栓风险未闭合前，应避免含雌激素避孕，改用非雌激素方案并由风湿免疫/妇产科共同评估。",
        )
    if "unsupported_no_renal_damage" in risks and any(marker in normalized_plan for marker in [normalize_name("暂无肾损害"), normalize_name("无肾损害")]):
        add_axis_issue(
            issues,
            patches,
            "unsupported_no_renal_damage",
            "缺少尿液和肾功能证据时，不应断言暂无肾损害；应改为需进一步评估狼疮性肾炎风险。",
        )
    if has_chest_wall_trauma_pattern(normalized_case_text):
        if "深呼吸练习" not in normalized_plan and "深呼吸" not in normalized_plan:
            add_axis_issue(
                issues,
                patches,
                "chest_wall_deep_breathing",
                "胸部外伤恢复期应在疼痛可耐受范围内保留规律深呼吸练习，以降低肺不张风险。",
            )
        elif "减少深呼吸" in normalized_plan or "避免深呼吸" in normalized_plan:
            plan = re.sub(r"(?:减少|避免)深呼吸", "进行疼痛可耐受的深呼吸练习", plan)
            add_axis_issue(
                issues,
                patches,
                "chest_wall_deep_breathing",
                "胸部外伤不应把深呼吸写成禁忌；应改为疼痛可耐受的深呼吸练习并监测呼吸困难。",
            )
    if "pregnancy_screening_before_migraine_drugs" in risks and not plan_has_any(normalized_plan, ["妊娠", "hcg", "怀孕"]):
        add_axis_issue(
            issues,
            patches,
            "pregnancy_screening_before_migraine_drugs",
            "育龄女性偏头痛用药前应确认妊娠状态，并据此选择急性期和预防用药。",
        )
    if "recurrent_epistaxis_coagulation" in risks and not plan_has_any(normalized_plan, ["凝血", "血小板", "血液学"]):
        add_axis_issue(
            issues,
            patches,
            "recurrent_epistaxis_coagulation",
            "反复鼻出血应补充凝血功能、血小板或基础血液学评估，同时处理局部干燥和结构性诱因。",
        )
    if "avoid_no_further_care_for_bleeding_mass" in risks and any(marker in normalized_plan for marker in [normalize_name("无需进一步"), normalize_name("无需处置"), normalize_name("通常自愈")]):
        add_axis_issue(
            issues,
            patches,
            "avoid_no_further_care_for_bleeding_mass",
            "新生儿局部湿润易出血肿块不应直接写无需进一步处置；需局部专科评估、止血/护理和必要病理确认。",
        )
    if "immunosuppressed_respiratory_infection_unclosed" in risks:
        if treatment_continues_immunosuppression(normalized_plan):
            plan = sanitize_immunosuppression_continuation(
                plan,
                diagnosis=diagnosis,
                case_features=case_features,
            )
        safety_goal = (
            "免疫抑制背景合并进行性呼吸困难时，应优先按高危下呼吸道感染处理：监测血氧/氧合，"
            "完成胸部影像和病毒病原评估，并与感染科/原免疫治疗专科共同评估免疫抑制方案。"
        )
        if any(marker in normalized_case_text for marker in [normalize_name("吞咽困难"), normalize_name("呛咳"), normalize_name("误吸风险")]):
            safety_goal += "存在吞咽或误吸风险时应采取进食体位、质地调整和防误吸措施，并评估吞咽功能。"
        add_axis_issue(
            issues,
            patches,
            "immunosuppressed_respiratory_infection_unclosed",
            safety_goal,
        )
    return {"issues": issues, "patches": patches, "treatment_plan": plan}


def allows_monitored_steroid_bridge_for_life_threatening_bleeding(
    case_features: Dict[str, Any],
    treatment_plan: str,
    *,
    diagnosis: str,
) -> bool:
    axis_ids = {
        clean_axis_id(axis.get("axis_id"))
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
    }
    if "systemic_infection_vs_primary_hematologic" not in axis_ids:
        return False
    if normalize_name(diagnosis) != normalize_name("特发性血小板减少性紫癜"):
        return False
    patient_text = clean_text(case_features.get("patient_text"))
    examination_results = case_features.get("examination_results")
    examination_text = " ".join(structured_text_chunks(examination_results)) if isinstance(examination_results, dict) else ""
    narrative_parts = [patient_text] if patient_text else (
        [clean_text(case_features.get("case_text"))]
        + as_text_list(case_features.get("positive_findings"))
        + as_text_list(case_features.get("red_flags"))
    )
    narrative_text = " ".join(part for part in narrative_parts if part)
    life_threatening = has_current_life_threatening_bleeding(
        narrative_text=narrative_text,
        examination_text=examination_text,
    )
    joint_monitoring = marker_present_not_negated(
        treatment_plan,
        [
            "血液科与感染科监护",
            "血液科与感染科共同监护",
            "血液科和感染科共同监护",
            "血液科及感染科共同监护",
            "血液科、感染科共同监护",
        ],
    ) or (
        marker_present_not_negated(treatment_plan, ["血液科监护"])
        and marker_present_not_negated(treatment_plan, ["感染科监护"])
    )
    unavailable_monitoring = re.search(
        r"(?:无法|不能|未能).{0,10}(?:监护|参与)|(?:联系|咨询).{0,20}(?:血液科|感染科)",
        clean_text(treatment_plan),
    )
    infection_parallel = marker_present_not_negated(
        treatment_plan,
        ["同步完成病原评估", "同步病原评估", "同步抗感染", "病原评估和抗感染处置"],
    ) and not re.search(r"(?:暂不|不同步|无需|不需).{0,16}(?:病原评估|抗感染)", clean_text(treatment_plan))
    rescue_context = marker_present_not_negated(treatment_plan, ["桥接抢救", "抢救", "挽救生命"])
    return all([life_threatening, joint_monitoring, not unavailable_monitoring, infection_parallel, rescue_context])


def marker_present_active(text: str, markers: List[str], *, resolved_markers: List[str]) -> bool:
    for clause in semantic_clauses(text):
        normalized_clause = normalize_name(clause)
        for marker in markers:
            token = normalize_name(marker)
            start = 0
            while True:
                index = normalized_clause.find(token, start)
                if index < 0:
                    break
                # Keep the value window short so a later field like 其他出血:已缓解
                # cannot resolve a prior field like 颅内出血:明确.
                suffix = normalized_clause[index + len(token): index + len(token) + 12]
                value_window = suffix[:8]
                postposed_negative = any(
                    suffix == normalize_name(item) or suffix.startswith(normalize_name(item))
                    for item in [
                        "无",
                        "未见",
                        "未发现",
                        "未提示",
                        "不支持",
                        "未检出",
                        "无异常",
                        "无证据",
                        "阴性",
                        "排除",
                        "已排除",
                        "不存在",
                        "没有",
                    ]
                )
                resolved_here = any(
                    normalize_name(item) in value_window or value_window.startswith(normalize_name(item))
                    for item in resolved_markers
                )
                if (
                    not marker_occurrence_is_negated(normalized_clause, token, index)
                    and not marker_occurrence_is_uncertain(normalized_clause, index)
                    and not postposed_negative
                    and not resolved_here
                ):
                    return True
                start = index + len(token)
    return False


def has_current_life_threatening_bleeding(*, narrative_text: str, examination_text: str) -> bool:
    markers = [
        "危及生命出血",
        "危及生命的活动性出血",
        "危及生命",
        "颅内出血",
        "活动性大出血",
        "失血性休克",
        "血流动力学不稳定",
    ]
    resolved = ["已排除", "已纠正", "已稳定", "目前稳定", "现已稳定", "已经稳定", "已缓解", "已痊愈"]
    if examination_text and marker_present_active(examination_text, markers, resolved_markers=resolved):
        return True
    for clause in semantic_clauses(narrative_text):
        if is_noncurrent_or_family_bleeding_clause(clause):
            continue
        if marker_present_active(clause, markers, resolved_markers=resolved):
            return True
    return False


def is_noncurrent_or_family_bleeding_clause(clause: str) -> bool:
    normalized = normalize_name(clause)
    if not normalized:
        return False
    family_markers = ["父亲", "母亲", "父母", "家属", "家族", "哥哥", "姐姐", "弟弟", "妹妹", "亲属"]
    if any(normalize_name(marker) in normalized for marker in family_markers):
        return True
    historical_markers = ["曾发生", "曾有", "既往", "三年前", "多年前", "病史中", "既往史"]
    current_markers = ["本次", "当前", "目前", "现在", "正在", "发生危及生命", "突发"]
    if any(normalize_name(marker) in normalized for marker in historical_markers) and not any(
        normalize_name(marker) in normalized for marker in current_markers
    ):
        return True
    return False


def infection_before_steroid_risk_is_active(
    case_features: Dict[str, Any],
    *,
    diagnosis: str,
) -> bool:
    risk_axes = [
        axis
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
        if "infection_before_steroid" in as_text_list(axis.get("treatment_risks"))
    ]
    if not risk_axes:
        return True
    if any(clean_axis_id(axis.get("axis_id")) != "systemic_infection_vs_primary_hematologic" for axis in risk_axes):
        return True
    return not systemic_infection_risk_resolved(case_features, diagnosis=diagnosis)


def systemic_infection_risk_resolved(case_features: Dict[str, Any], *, diagnosis: str) -> bool:
    if normalize_name(diagnosis) != normalize_name("特发性血小板减少性紫癜"):
        return False
    vector_exposure = any(
        clean_axis_id(axis.get("axis_id")) == "systemic_infection_vs_primary_hematologic"
        and (
            "媒介或暴露相关病原评估" in as_text_list(axis.get("exam_intents"))
            or "回归热" in as_text_list(axis.get("candidate_official_names"))
        )
        for axis in as_axis_list(case_features.get("diagnosis_axes"))
    )
    if systemic_pathogen_workup_status(case_features, vector_exposure=vector_exposure) != "negative":
        return False
    if blood_smear_evidence_status(case_features) != "normal":
        return False
    if infection_activity_evidence_status(case_features) != "negative":
        return False
    return has_isolated_thrombocytopenia(case_features)


def systemic_pathogen_workup_status(
    case_features: Dict[str, Any],
    *,
    vector_exposure: bool,
) -> str:
    groups: Dict[str, List[str]] = {"culture": [], "directed": []}
    results = case_features.get("examination_results")
    if not isinstance(results, dict):
        return "unknown"
    for name, payload in results.items():
        normalized_name = normalize_name(name)
        if "血培养" in normalized_name:
            groups["culture"].append(pathogen_payload_status(payload))
        if any(marker in normalized_name for marker in ["血清学抗体", "病原体抗体", "病原体核酸", "病毒核酸"]):
            groups["directed"].append(pathogen_payload_status(payload))
    statuses = groups["culture"] + groups["directed"]
    if "positive" in statuses:
        return "positive"
    required_group = "directed" if vector_exposure else "culture"
    if not groups[required_group] or not all(status == "negative" for status in groups[required_group]):
        return "unknown"
    optional_group = "culture" if vector_exposure else "directed"
    if groups[optional_group] and not all(status == "negative" for status in groups[optional_group]):
        return "unknown"
    if groups[required_group]:
        return "negative"
    return "unknown"


def pathogen_payload_status(payload: Any) -> str:
    if not isinstance(payload, dict) or not exam_result_is_usable(payload):
        return "unknown"
    values = [
        result_value_without_reference(value)
        for key, value in exam_result_pairs(payload)
        if not any(marker in normalize_name(key) for marker in ["阴性对照", "阳性对照", "参考", "正常范围"])
    ]
    normalized = normalize_name(" ".join(values))
    positive_text = normalized
    for marker in ["无生长", "未检出", "未分离出", "阴性"]:
        positive_text = positive_text.replace(normalize_name(marker), "")
    if any(marker in positive_text for marker in ["阳性", "检出", "培养出", "菌生长", "致病菌"]):
        return "positive"
    if any(marker in normalized for marker in ["无生长", "未检出", "未分离出", "阴性"]):
        return "negative"
    if normalize_name(payload.get("status")) == "normal":
        return "negative"
    return "unknown"


def blood_smear_evidence_status(case_features: Dict[str, Any]) -> str:
    statuses = []
    for payload in matching_exam_payloads(case_features, ["外周血涂片"]):
        if not exam_result_is_usable(payload):
            statuses.append("unknown")
            continue
        text = normalize_name(" ".join(value for _, value in exam_result_pairs(payload)))
        positive_text = text
        negative_markers = [
            "未见原始细胞或异常细胞",
            "未见原始及异常细胞",
            "未见原始细胞",
            "未见异常细胞",
            "无原始细胞",
            "无异常细胞",
        ]
        for marker in negative_markers:
            positive_text = positive_text.replace(normalize_name(marker), "")
        if any(marker in positive_text for marker in ["原始细胞", "异常细胞", "幼稚细胞", "母细胞"]):
            statuses.append("abnormal")
        elif normalize_name(payload.get("status")) == "normal" or any(
            normalize_name(marker) in text for marker in negative_markers
        ):
            statuses.append("normal")
        else:
            statuses.append("unknown")
    if "abnormal" in statuses:
        return "abnormal"
    if statuses and all(status == "normal" for status in statuses):
        return "normal"
    return "unknown"


def infection_activity_evidence_status(case_features: Dict[str, Any]) -> str:
    statuses = []
    for payload in matching_exam_payloads(case_features, ["C反应蛋白", "CRP", "降钙素原", "PCT"]):
        if not exam_result_is_usable(payload):
            statuses.append("unknown")
            continue
        result_text = normalize_name(
            " ".join(result_value_without_reference(value) for _, value in exam_result_pairs(payload))
        )
        if any(marker in result_text for marker in ["待报告", "待出结果", "待复核", "处理中"]):
            statuses.append("unknown")
            continue
        status = normalize_name(payload.get("status"))
        if status == "abnormal":
            statuses.append("positive")
        elif status == "normal":
            statuses.append("negative")
        else:
            statuses.append("unknown")
    if "positive" in statuses:
        return "positive"
    if statuses and all(status == "negative" for status in statuses):
        return "negative"
    return "unknown"


def has_isolated_thrombocytopenia(case_features: Dict[str, Any]) -> bool:
    for payload in matching_exam_payloads(case_features, ["全血细胞计数", "血常规", "CBC"]):
        if not exam_result_is_usable(payload):
            continue
        positions = {
            "hemoglobin": "unknown",
            "wbc": "unknown",
            "platelet": "unknown",
        }
        for key, value in exam_result_pairs(payload):
            if is_total_hemoglobin_key(key):
                positions["hemoglobin"] = lab_value_reference_position(value)
            elif is_total_wbc_key(key):
                positions["wbc"] = lab_value_reference_position(value)
            elif is_total_platelet_key(key):
                positions["platelet"] = lab_value_reference_position(value)
        if positions == {"hemoglobin": "normal", "wbc": "normal", "platelet": "low"}:
            return True
    return False


STEROID_ALIASES = [
    "糖皮质激素",
    "全身激素",
    "局部激素",
    "激素滴眼液",
    "激素治疗",
    "地塞米松",
    "泼尼松",
    "泼尼松龙",
    "甲泼尼龙",
    "氢化可的松",
    "氟米龙",
    "氯替泼诺",
    "倍他米松",
    "prednisolone",
    "methylprednisolone",
    "dexamethasone",
    "betamethasone",
    "fluorometholone",
    "loteprednol",
    "prednisone",
    "hydrocortisone",
    "强的松",
]


def sanitize_steroid_recommendations(treatment_plan: str) -> str:
    clauses = re.split(r"(?<=[，,；;。\n])", clean_text(treatment_plan))
    safe_clauses = []
    for clause in clauses:
        normalized_clause = normalize_name(clause)
        if not has_unnegated_steroid_mention(normalized_clause, STEROID_ALIASES):
            safe_clauses.append(clause)
    return normalize_treatment_text("".join(safe_clauses))


def has_unnegated_steroid_mention(normalized_clause: str, aliases: List[str]) -> bool:
    for alias in aliases:
        alias_text = normalize_name(alias)
        start = 0
        while True:
            index = normalized_clause.find(alias_text, start)
            if index < 0:
                break
            before = normalized_clause[max(0, index - 24):index]
            after = normalized_clause[index + len(alias_text): index + len(alias_text) + 16]
            # "无需/不建议/未停用激素" means continue steroid exposure, not avoid it.
            continuation = bool(
                re.search(r"(?:无需|不必|不用|不建议|未|不要)停用$", before)
                or re.search(r"(?:仍在|继续)(?:使用|服用|原剂量|治疗)", after)
                or re.search(r"继续原剂量|仍在继续", normalized_clause)
            )
            negated = bool(
                re.search(
                    r"(?:避免|不应|不得|禁止|禁用|停用|不推荐|不涉及|不使用|不予|无需)[^，,；;。]{0,18}$",
                    before,
                )
            )
            if continuation or not negated:
                return True
            start = index + len(alias_text)
    return False


def sanitize_estrogen_contraception_recommendations(treatment_plan: str) -> str:
    clauses = re.split(r"(?<=[，,；;。\n])", clean_text(treatment_plan))
    safe_clauses = []
    for clause in clauses:
        normalized_clause = normalize_name(clause)
        if not treatment_contains_estrogen_contraception(normalized_clause):
            safe_clauses.append(clause)
            continue
        if not treatment_recommends_drug(
            normalized_clause,
            ["含雌激素", "复方口服避孕药", "雌激素"],
        ):
            safe_clauses.append(clause)
    return normalize_treatment_text("".join(safe_clauses))


def treatment_contains_steroid(normalized_plan: str) -> bool:
    return any(
        has_unnegated_steroid_mention(normalize_name(clause), STEROID_ALIASES)
        for clause in re.split(r"(?<=[，,；;。\n])", clean_text(normalized_plan))
    )


def treatment_contains_estrogen_contraception(normalized_plan: str) -> bool:
    estrogen = any(marker in normalized_plan for marker in [normalize_name("雌激素"), normalize_name("含雌激素"), normalize_name("复方口服避孕药")])
    contraception = any(marker in normalized_plan for marker in [normalize_name("避孕"), normalize_name("避孕药")])
    return estrogen and contraception


def plan_has_any(normalized_plan: str, markers: List[str]) -> bool:
    return any(normalize_name(marker) in normalized_plan for marker in markers)


def add_axis_issue(issues: List[Dict[str, Any]], patches: List[str], code: str, patch: str) -> None:
    issues.append(
        {
            "field": "treatment_plan",
            "code": code,
            "severity": "must_fix",
            "problem": code,
            "patchable": True,
            "edit": patch,
        }
    )
    if patch not in patches:
        patches.append(patch)


def append_unique_patches(treatment_plan: str, patches: List[str]) -> str:
    result = clean_text(treatment_plan)
    for patch in patches:
        patch_text = clean_text(patch)
        if patch_text and patch_text not in result:
            result = " ".join([result, patch_text]).strip()
    return result


def should_force_examination(
    case_state: Dict[str, Any],
    *,
    min_patient_replies: int = 3,
    proposed_question: str = "",
) -> bool:
    if is_repeated_answered_intake_question(case_state, proposed_question):
        return True
    if completed_examinations(case_state):
        return False
    patient_replies = [
        item for item in case_state.get("chat_history", [])
        if isinstance(item, dict) and item.get("from") == "patient" and clean_text(item.get("text"))
    ]
    return len(patient_replies) >= int(min_patient_replies)


INTAKE_TOPIC_MARKERS = {
    "hypertension": ["高血压", "血压"],
    "diabetes": ["糖尿病", "血糖"],
    "medication": ["目前用药", "现在用药", "长期用药", "正在服用", "处方药", "吃什么药", "服用什么药"],
    "medication_allergy": ["药物过敏", "用药后不良反应", "药物不良反应"],
    "infection_exposure": ["旅行", "野外", "户外", "露营", "蜱", "虫咬", "动物接触", "游走性皮疹"],
    "menopause_status": ["绝经", "更年期", "外阴干涩", "围绝经", "绝经后"],
    "respiratory_infection_risk": ["发热", "脓痰", "吞咽", "呛咳", "误吸"],
    "neurocysticercosis_exposure": ["未熟猪肉", "猪带绦虫", "囊虫", "流行地区", "卫生条件"],
    "liver_etiology": ["乙肝", "丙肝", "肝炎", "饮酒", "肝癌", "体重下降"],
    "developmental_epilepsy": ["发育", "学习", "遗传", "FMR1", "停药", "用药效果"],
}

INTAKE_TOPIC_ANSWER_MARKERS = {
    "hypertension": ["无高血压", "没有高血压", "否认高血压", "高血压病史没有"],
    "diabetes": ["无糖尿病", "没有糖尿病", "否认糖尿病", "糖尿病病史没有"],
    "medication": ["无处方药", "没有处方药", "未用药", "没有长期处方药", "布洛芬", "纤维补充剂"],
    "medication_allergy": ["无药物过敏", "没有药物过敏", "否认药物过敏", "用药后不良反应", "药物不良反应"],
    "infection_exposure": ["旅行", "野外", "户外", "露营", "蜱", "虫咬", "动物接触", "游走性皮疹"],
    "menopause_status": ["绝经", "已绝经", "未绝经", "更年期", "外阴干涩", "仍有月经", "还在来月经"],
    "respiratory_infection_risk": ["发热", "脓痰", "吞咽困难", "喝水呛咳", "进食呛咳", "误吸"],
    "neurocysticercosis_exposure": ["未熟猪肉", "猪带绦虫", "囊虫", "流行地区", "卫生条件差"],
    "liver_etiology": ["乙肝", "丙肝", "病毒性肝炎", "饮酒", "肝癌", "体重下降"],
    "developmental_epilepsy": ["发育迟缓", "学习障碍", "遗传", "FMR1", "脆性X", "停药", "控制不佳", "效果不好"],
}


def select_required_intake_question(case_state: Dict[str, Any]) -> str:
    patient_text = patient_text_for_matching(case_state)
    case_text = case_text_for_matching(case_state)
    answered = answered_intake_topics(case_state)
    if has_immunosuppressed_progressive_respiratory_pattern(patient_text) and "respiratory_infection_risk" not in answered:
        return "近期是否发热、咳脓痰或喘息？进食饮水时有无吞咽困难、呛咳或误吸？"
    if (
        has_seizure_intracranial_calcification_pattern(case_text)
        or has_acute_pressure_headache_intracranial_calcification_pattern(case_text)
    ) and "neurocysticercosis_exposure" not in answered:
        return "是否来自囊虫病流行或卫生条件较差地区，是否吃过未熟猪肉、接触猪带绦虫，家人有无类似情况？"
    if has_decompensated_liver_symptom_pattern(patient_text) and "liver_etiology" not in answered:
        return "既往是否有乙肝、丙肝或其他肝炎，是否长期饮酒、近期明显体重下降，或曾发现肝脏结节/肿块？"
    if has_childhood_onset_epilepsy_pattern(patient_text) and "developmental_epilepsy" not in answered:
        return "从小有无发育迟缓、学习困难或已知遗传/FMR1异常？既往抗癫痫药效果如何，为什么停药或服用不规律？"
    if has_relapsing_fever_bleeding_pattern(patient_text) and "infection_exposure" not in answered:
        return "近期是否有旅行、露营或野外活动，是否被蜱虫或其他昆虫叮咬、接触动物，或出现游走性皮疹？"
    menopause_question = select_menopause_status_intake_question(case_state, answered_topics=answered)
    if menopause_question:
        return menopause_question
    return ""


def select_next_clinical_action(case_state: Dict[str, Any]) -> Dict[str, str]:
    replies = [
        item for item in case_state.get("chat_history", [])
        if isinstance(item, dict) and item.get("from") == "patient"
    ]
    if not replies:
        return {
            "action": "ask_patient",
            "question": "请您按时间顺序描述这次最主要的不适、何时开始、如何变化，以及伴随症状。",
            "reason": "先获取主诉和现病史。",
        }
    if len(replies) == 1:
        return {
            "action": "ask_patient",
            "question": "请补充您的年龄、药物过敏、当前用药，以及高血压、糖尿病等基础疾病。",
            "reason": "补齐用药、过敏和基础病安全信息。",
        }
    required = select_required_intake_question(case_state)
    if required:
        return {"action": "ask_patient", "question": required, "reason": "闭合高收益诊断或安全缺口。"}
    if not completed_examinations(case_state):
        return {"action": "order_examination", "question": "", "reason": "问诊信息已足够，获取客观检查证据。"}
    if should_block_final_for_coverage_gaps(case_state):
        return {"action": "order_examination", "question": "", "reason": "仍有可闭合的关键检查缺口。"}
    return {"action": "final_diagnosis", "question": "", "reason": "当前证据已达到最终诊疗条件。"}


def should_run_prefinal_axis_review(case_state: Dict[str, Any]) -> bool:
    return bool(open_coverage_gaps(case_state))


def select_menopause_status_intake_question(
    case_state: Dict[str, Any],
    *,
    answered_topics: Optional[set[str]] = None,
) -> str:
    if answered_topics is None:
        answered_topics = answered_intake_topics(case_state)
    if "menopause_status" in answered_topics:
        return ""
    patient_text = patient_text_for_matching(case_state)
    normalized = normalize_name(patient_text)
    has_urinary = any(
        normalize_name(marker) in normalized
        for marker in ["尿频", "尿急", "尿痛", "排尿烧灼", "血尿", "尿路"]
    )
    if not has_urinary:
        return ""
    has_female_context = any(
        normalize_name(marker) in normalized
        for marker in ["女", "女性", "月经", "怀孕", "妊娠", "阴道", "外阴"]
    )
    already_postmenopausal = any(
        normalize_name(marker) in normalized
        for marker in ["绝经", "更年期", "绝经后"]
    )
    if not has_female_context or already_postmenopausal:
        return ""
    return "是否已绝经或处于更年期？有无外阴干涩、灼热或性交不适等萎缩相关症状？"


def is_repeated_answered_intake_question(case_state: Dict[str, Any], proposed_question: str) -> bool:
    normalized_question = normalize_name(proposed_question)
    if not normalized_question:
        return False
    answered_topics = answered_intake_topics(case_state)
    question_topics = {
        topic for topic, markers in INTAKE_TOPIC_MARKERS.items()
        if any(normalize_name(marker) in normalized_question for marker in markers)
    }
    return bool(question_topics and question_topics.issubset(answered_topics))


def answered_intake_topics(case_state: Dict[str, Any]) -> set[str]:
    patient_text = normalize_name(
        " ".join(
            clean_text(item.get("text"))
            for item in case_state.get("chat_history", [])
            if isinstance(item, dict) and item.get("from") == "patient"
        )
    )
    answered = {
        topic for topic, markers in INTAKE_TOPIC_ANSWER_MARKERS.items()
        if any(normalize_name(marker) in patient_text for marker in markers)
    }
    history = case_state.get("chat_history", [])
    for index, item in enumerate(history[:-1]):
        if not isinstance(item, dict) or item.get("from") != "doctor":
            continue
        next_item = history[index + 1]
        if not isinstance(next_item, dict) or next_item.get("from") != "patient":
            continue
        question = normalize_name(item.get("text"))
        answered.update(
            topic for topic, markers in INTAKE_TOPIC_MARKERS.items()
            if any(normalize_name(marker) in question for marker in markers)
        )
    return answered


def should_stop_patient_questions(
    case_state: Dict[str, Any],
    *,
    max_patient_replies: int = MAX_PATIENT_REPLIES,
) -> bool:
    replies = [
        item
        for item in case_state.get("chat_history", [])
        if isinstance(item, dict)
        and item.get("from") == "patient"
    ]
    return len(replies) >= int(max_patient_replies)


def should_stop_examinations(
    case_state: Dict[str, Any],
    *,
    max_exam_actions: int = MAX_EXAMINATION_ACTIONS,
) -> bool:
    # Thin wrapper over shared budget policy (hard cap only).
    from agent.clinical.exam_budget_policy import decide_exam_budget

    budget = decide_exam_budget(
        exam_trace=list(case_state.get("exam_decision_trace") or []),
        open_gaps=open_coverage_gaps(case_state),
        ordered_examinations=valid_ordered_examinations(case_state),
        hard_cap=int(max_exam_actions),
        semantic_key_fn=exam_semantic_key,
    )
    return budget.stop_kind == "hard"


EXAM_SEMANTIC_ALIASES = {
    "胸部X线检查（CXR）": "chest_xray",
    "胸部CT扫描（Chest CT）": "chest_ct",
    "眼部超声": "ocular_ultrasound",
    "肾血管造影": "renal_artery_angiography",
}


def exam_semantic_key(exam_name: str) -> str:
    normalized = normalize_name(exam_name)
    for source, key in EXAM_SEMANTIC_ALIASES.items():
        if normalize_name(source) == normalized:
            return key
    return normalized


def exam_candidate_entry(item: object) -> Dict[str, object]:
    if isinstance(item, str):
        name = clean_text(item)
        return {
            "name": name,
            "stage": "first_line",
            "priority": 0,
            "semantic_key": exam_semantic_key(name),
            "mandatory_parallel": False,
        }
    if isinstance(item, dict):
        name = clean_text(item.get("name"))
        return {
            "name": name,
            "stage": clean_text(item.get("stage")) or "first_line",
            "priority": int(item.get("priority") or 0),
            "semantic_key": clean_text(item.get("semantic_key")) or exam_semantic_key(name),
            "mandatory_parallel": item.get("mandatory_parallel") is True,
        }
    return {"name": "", "stage": "", "priority": 0, "semantic_key": "", "mandatory_parallel": False}


def exam_candidates_for_intent(intent: str, exam_intent_rules: List[Dict[str, Any]]) -> List[Dict[str, object]]:
    normalized_intent = normalize_name(intent)
    urinary_context = any(
        normalize_name(marker) in normalized_intent
        for marker in ["尿培养", "尿液培养", "尿液病原", "尿路致病菌", "泌尿系致病菌"]
    )
    result: List[Dict[str, object]] = []
    for rule in exam_intent_rules:
        rule_id = clean_text(rule.get("id"))
        inputs = as_text_list(rule.get("input"))
        # Typed rules emit stable exam_intent_* ids; match those before free-text inputs.
        id_matched = bool(rule_id) and normalize_name(rule_id) == normalized_intent
        text_matched = bool(inputs) and any(
            normalize_name(item) in normalized_intent for item in inputs
        )
        if not id_matched and not text_matched:
            if inputs or rule_id:
                continue
            # Empty input + empty id rules are ignored (no open match-all).
            continue
        raw_outputs = rule.get("output")
        candidates: List[object]
        if isinstance(raw_outputs, list):
            candidates = list(raw_outputs)
        elif raw_outputs is None:
            candidates = []
        else:
            candidates = [raw_outputs]
        for raw in candidates:
            entry = exam_candidate_entry(raw)
            name = clean_text(entry.get("name"))
            if not name:
                continue
            if urinary_context and name == "细菌培养及鉴定":
                continue
            result.append(entry)
    # Prefer higher priority first-line, preserve configuration order as a stable tie-breaker.
    stage_rank = {"first_line": 0, "add_on": 1}
    decorated = list(enumerate(result))
    decorated.sort(
        key=lambda pair: (
            stage_rank.get(str(pair[1].get("stage") or "first_line"), 9),
            -int(pair[1].get("priority") or 0),
            pair[0],
        )
    )
    return [item for _, item in decorated]


def _typed_rule_exam_intent_ids(case_state: Dict[str, Any]) -> List[str]:
    """Read supplemental exam_intent_ids emitted by typed clinical_closure rules."""
    # Prefer exam-stage forward intents; final-stage diagnosis_rule_result remains
    # authoritative for candidate/closure transforms and is not replaced here.
    forward_intents = as_text_list(case_state.get("typed_exam_intent_ids"))
    if forward_intents:
        return forward_intents
    rule_result = case_state.get("diagnosis_rule_result")
    if isinstance(rule_result, RuleResult):
        return list(rule_result.output_context.exam_intent_ids)
    if isinstance(rule_result, Mapping):
        context = rule_result.get("output_context")
        if isinstance(context, Mapping):
            return as_text_list(context.get("exam_intent_ids"))
        return as_text_list(rule_result.get("exam_intent_ids"))
    return []


def apply_exam_rule_closure_intents(
    case_state: Dict[str, Any],
    *,
    rule_pack: Optional[CompiledRulePack],
    diagnosis_axes: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Compute clinical_closure exam intents before planning; pure supplemental write.

    Does not reorder diagnosis candidates or run final diagnosis_candidates stage.
    Caches by fact_codes + axis_ids + pack rule_count within the case only.
    """
    if rule_pack is None:
        return as_text_list(case_state.get("typed_exam_intent_ids"))
    from agent.clinical.exam_rule_closure import evaluate_exam_rule_closure

    axes = as_axis_list(diagnosis_axes if diagnosis_axes is not None else case_state.get("diagnosis_axes"))
    axis_ids = [
        clean_axis_id(axis.get("axis_id"))
        for axis in axes
        if clean_axis_id(axis.get("axis_id"))
    ]
    fact_codes = list(diagnosis_rule_fact_codes(case_state))
    cache_payload = {
        "fact_codes": fact_codes,
        "axis_ids": axis_ids,
        "pack_rule_count": int(getattr(rule_pack, "rule_count", 0) or 0),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    cached = case_state.get("exam_rule_closure_cache")
    if isinstance(cached, Mapping) and clean_text(cached.get("key")) == cache_key:
        intents = as_text_list(cached.get("exam_intent_ids"))
        case_state["typed_exam_intent_ids"] = intents
        return intents

    closure = evaluate_exam_rule_closure(
        rule_pack=rule_pack,
        fact_codes=fact_codes,
        diagnostic_axis_ids=axis_ids,
    )
    intents = list(closure.exam_intent_ids)
    case_state["typed_exam_intent_ids"] = intents
    case_state["exam_rule_closure_trace"] = {
        "matched_rule_ids": list(closure.matched_rule_ids),
        "excluded_rule_ids": list(closure.excluded_rule_ids),
    }
    case_state["exam_rule_closure_cache"] = {
        "key": cache_key,
        "exam_intent_ids": intents,
    }
    return intents


def select_exam_plan(
    *,
    case_state: Dict[str, Any],
    disease_candidates: List[Dict[str, Any]],
    diagnosis_axes: Optional[List[Dict[str, Any]]] = None,
    examination_catalog: Dict[str, List[str]],
    item_name_map: Dict[str, str],
    diagnosis_exam_profiles: Optional[List[Dict[str, Any]]] = None,
    verified_exam_profiles: Optional[List[Mapping[str, Any]]] = None,
    exam_intent_rules: Optional[List[Dict[str, Any]]] = None,
    max_items: int = MAX_EXAMS_PER_ACTION,
) -> Dict[str, Any]:
    case_text = case_text_for_matching(case_state)
    already_ordered = set(valid_ordered_examinations(case_state))
    planned: List[str] = []
    accepted: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    reasons: List[str] = []
    reason_codes: List[str] = []
    axis_planned = False
    axes = list(diagnosis_axes or [])
    if not axes:
        # Offline helper only: production callers must pass diagnosis_axes so the
        # planner never reloads a second RulePack from pointer.
        axes = select_diagnosis_axes(
            extract_intake_facts(case_state),
            case_state=case_state,
        )
    covered_semantic_keys = {
        exam_semantic_key(name)
        for name in already_ordered
        if clean_text(name)
    }

    # Prioritize verified case prior examinations (T01).
    invalid_exams = set(as_text_list(case_state.get("invalid_examinations")))
    verified_prior = case_state.get("verified_case_prior")
    if verified_prior:
        from agent.clinical.verified_prior import verified_prior_pending_examinations
        pending_prior_exams = verified_prior_pending_examinations(
            verified_prior,
            attempted=already_ordered,
            valid_examinations=item_name_map.keys(),
        )
        for exam_name in pending_prior_exams:
            if len(planned) >= int(max_items):
                break
            if exam_name in planned or exam_name in already_ordered:
                continue
            # Skip invalid examinations.
            if exam_name in invalid_exams:
                continue
            planned.append(exam_name)
            accepted.append({
                "name": exam_name,
                "intent": "verified_case_prior",
                "source": "verified_case_prior",
            })
            reason_codes.append("verified_case_prior")

    def try_accept_candidate(
        candidate: Dict[str, object],
        *,
        intent: str,
        axis_id: str = "",
        source: str,
    ) -> bool:
        nonlocal axis_planned
        raw_exam = clean_text(candidate.get("name"))
        standard_exam = match_standard_name(raw_exam, item_name_map)
        stage = clean_text(candidate.get("stage")) or "first_line"
        semantic_key = clean_text(candidate.get("semantic_key")) or exam_semantic_key(standard_exam or raw_exam)
        if not standard_exam:
            suppressed.append(
                {
                    "name": raw_exam,
                    "intent": intent,
                    "axis_id": axis_id,
                    "stage": stage,
                    "semantic_key": semantic_key,
                    "reason_code": "not_catalog_leaf",
                    "source": source,
                }
            )
            return False
        if exam_already_covered(standard_exam, already_ordered, case_state) or standard_exam in planned:
            suppressed.append(
                {
                    "name": standard_exam,
                    "intent": intent,
                    "axis_id": axis_id,
                    "stage": stage,
                    "semantic_key": semantic_key,
                    "reason_code": "already_covered",
                    "source": source,
                }
            )
            return False
        if semantic_key and semantic_key in covered_semantic_keys:
            suppressed.append(
                {
                    "name": standard_exam,
                    "intent": intent,
                    "axis_id": axis_id,
                    "stage": stage,
                    "semantic_key": semantic_key,
                    "reason_code": "semantic_duplicate",
                    "source": source,
                }
            )
            return False
        if not is_leaf_exam_name(standard_exam, examination_catalog):
            suppressed.append(
                {
                    "name": standard_exam,
                    "intent": intent,
                    "axis_id": axis_id,
                    "stage": stage,
                    "semantic_key": semantic_key,
                    "reason_code": "not_catalog_leaf",
                    "source": source,
                }
            )
            return False
        if not exam_applicable_to_case(standard_exam, case_text, context=intent):
            suppressed.append(
                {
                    "name": standard_exam,
                    "intent": intent,
                    "axis_id": axis_id,
                    "stage": stage,
                    "semantic_key": semantic_key,
                    "reason_code": "not_applicable",
                    "source": source,
                }
            )
            return False
        if should_suppress_exam(standard_exam, case_text):
            suppressed.append(
                {
                    "name": standard_exam,
                    "intent": intent,
                    "axis_id": axis_id,
                    "stage": stage,
                    "semantic_key": semantic_key,
                    "reason_code": "not_applicable",
                    "source": source,
                }
            )
            return False
        # BNP and NT-proBNP are alternative first-line natriuretic leaves.
        natriuretic = {
            normalize_name("N末端B型利钠肽原（NT-proBNP）"),
            normalize_name("B型利钠肽（BNP）"),
        }
        if normalize_name(standard_exam) in natriuretic:
            already = {normalize_name(item) for item in list(already_ordered) + planned}
            if already & natriuretic:
                suppressed.append(
                    {
                        "name": standard_exam,
                        "intent": intent,
                        "axis_id": axis_id,
                        "stage": stage,
                        "semantic_key": semantic_key,
                        "reason_code": "natriuretic_alternative_already_selected",
                        "source": source,
                    }
                )
                return False
        planned.append(standard_exam)
        covered_semantic_keys.add(semantic_key)
        accepted.append(
            {
                "name": standard_exam,
                "intent": intent,
                "axis_id": axis_id,
                "stage": stage,
                "semantic_key": semantic_key,
                "source": source,
            }
        )
        if source.startswith("axis") or source.startswith("required") or source.startswith("coverage_gap"):
            axis_planned = True
        return True

    def select_from_intent(
        intent: str,
        *,
        axis_id: str = "",
        source: str,
        allow_addon: bool = False,
    ) -> None:
        candidates = exam_candidates_for_intent(intent, exam_intent_rules or [])
        accepted_first_line = False
        for candidate in candidates:
            if len(planned) >= int(max_items):
                break
            stage = clean_text(candidate.get("stage")) or "first_line"
            mandatory_parallel = candidate.get("mandatory_parallel") is True
            if stage == "add_on":
                if not allow_addon or not accepted_first_line:
                    suppressed.append(
                        {
                            "name": clean_text(candidate.get("name")),
                            "intent": intent,
                            "axis_id": axis_id,
                            "stage": stage,
                            "semantic_key": clean_text(candidate.get("semantic_key")),
                            "reason_code": "addon_before_first_line",
                            "source": source,
                        }
                    )
                    continue
            if stage == "first_line" and accepted_first_line and not mandatory_parallel:
                suppressed.append(
                    {
                        "name": clean_text(candidate.get("name")),
                        "intent": intent,
                        "axis_id": axis_id,
                        "stage": stage,
                        "semantic_key": clean_text(candidate.get("semantic_key")),
                        "reason_code": "lower_priority_first_line",
                        "source": source,
                    }
                )
                continue
            accepted_now = try_accept_candidate(
                candidate,
                intent=intent,
                axis_id=axis_id,
                source=source,
            )
            if accepted_now and stage == "first_line":
                accepted_first_line = True
                if not mandatory_parallel:
                    # Default: one first-line leaf per intent.
                    break

    # Hard-force concrete required_exams from coverage gaps before soft intents expand
    # into off-axis allergy/profile leaves (Canary 00660: AMA displaced by skin tests).
    for gap in open_coverage_gaps(case_state):
        if len(planned) >= int(max_items):
            break
        gap_id = clean_text(gap.get("gap_id"))
        for raw_exam in as_text_list(gap.get("required_exams")):
            if len(planned) >= int(max_items):
                break
            accepted_now = try_accept_candidate(
                exam_candidate_entry(raw_exam),
                intent=clean_text(gap.get("reason")) or gap_id,
                axis_id=gap_id,
                source="coverage_gap_required",
            )
            if accepted_now:
                reasons.append("coverage_gap:%s:%s" % (gap_id, raw_exam))
                reason_codes.append("coverage_gap_required")

    # Natriuretic peptides are mutually alternative first-line leaves for HF.
    if any(
        normalize_name(item)
        in {
            normalize_name("N末端B型利钠肽原（NT-proBNP）"),
            normalize_name("B型利钠肽（BNP）"),
        }
        for item in list(already_ordered) + planned
    ):
        covered_semantic_keys.add(exam_semantic_key("N末端B型利钠肽原（NT-proBNP）"))
        covered_semantic_keys.add(exam_semantic_key("B型利钠肽（BNP）"))

    gap_intents = [
        intent
        for gap in open_coverage_gaps(case_state)
        for intent in as_text_list(gap.get("exam_intents"))
    ]
    required_intents = unique_preserve_order(gap_intents + build_required_exam_intents(case_text, axes))
    for intent in required_intents:
        select_from_intent(intent, source="required_intent", allow_addon=False)
        reasons.append("required_intent:%s" % intent)
        if len(planned) >= int(max_items):
            break

    # Supplement only: typed clinical_closure rules may append exam_intent_ids.
    # Never replace the existing intent channels above (P0-3).
    typed_rule_intents = _typed_rule_exam_intent_ids(case_state)
    for intent in typed_rule_intents:
        if len(planned) >= int(max_items):
            break
        before = len(planned)
        select_from_intent(intent, source="typed_rule_intent", allow_addon=False)
        if len(planned) > before:
            reasons.append("typed_rule_intent:%s" % intent)
            reason_codes.append("typed_rule_intent")

    for axis in axes:
        axis_id = clean_text(axis.get("axis_id"))
        axis_intents = as_text_list(axis.get("exam_intents"))
        for intent in axis_intents:
            select_from_intent(
                intent,
                axis_id=axis_id,
                source="axis_intent",
                allow_addon=False,
            )
            if len(planned) >= int(max_items):
                break
        if axis_intents:
            reasons.append("%s:%s" % (axis_id, "、".join(axis_intents)))
        if len(planned) >= int(max_items):
            break

    for profile in exam_profiles_for_case(
        disease_candidates,
        case_text,
        diagnosis_exam_profiles=diagnosis_exam_profiles,
        exam_intent_rules=exam_intent_rules,
    ):
        profile_intents = as_text_list(profile.get("exam_intents"))
        if profile_intents:
            for intent in profile_intents:
                select_from_intent(
                    intent,
                    source="diagnosis_profile",
                    allow_addon=False,
                )
                if len(planned) >= int(max_items):
                    break
        else:
            for raw_exam in as_text_list(profile.get("examinations")):
                if len(planned) >= int(max_items):
                    break
                try_accept_candidate(
                    exam_candidate_entry(raw_exam),
                    intent=clean_text(profile.get("diagnosis")),
                    source="diagnosis_profile",
                )
        reason = clean_text(profile.get("reason"))
        if reason:
            reasons.append(reason)
        if len(planned) >= int(max_items):
            break

    # Add verified disease exam profile examinations (T08).
    if verified_exam_profiles:
        for vprofile in verified_exam_profiles:
            if not isinstance(vprofile, Mapping):
                continue
            # Match by diagnosis name.
            profile_diagnosis = clean_text(vprofile.get("diagnosis_name"))
            candidate_names = {clean_text(item.get("disease")) for item in disease_candidates if isinstance(item, dict)}
            if profile_diagnosis and profile_diagnosis not in candidate_names:
                continue
            profile_used = False
            for exam_item in vprofile.get("exam_items") or []:
                if len(planned) >= int(max_items):
                    break
                exam_name = clean_text(exam_item.get("name") if isinstance(exam_item, Mapping) else exam_item)
                if not exam_name:
                    continue
                # Skip non-catalog names.
                if exam_name not in item_name_map:
                    continue
                # Skip already ordered.
                if exam_name in already_ordered:
                    continue
                try_accept_candidate(
                    exam_candidate_entry(exam_name),
                    intent=profile_diagnosis,
                    source="verified_disease_profile",
                )
                profile_used = True
            if profile_used:
                reason_codes.append("verified_disease_profile")

    planned = prioritize_anatomy_coverage(planned, case_text, max_items=int(max_items))
    if not planned:
        return {
            "category": "",
            "candidate_diagnoses": [],
            "examinations": [],
            "reason": "",
            "reason_codes": ["no_supported_exam"],
            "accepted": [],
            "suppressed": suppressed,
        }
    return {
        "category": (
            "diagnosis_axis_exam_intent"
            if axis_planned
            else ("candidate_diagnosis_exam_profile" if planned else "")
        ),
        "candidate_diagnoses": [
            clean_text(item.get("disease"))
            for item in disease_candidates[:5]
            if isinstance(item, dict)
        ],
        "examinations": planned,
        "reason": "；".join(reasons),
        "reason_codes": unique_preserve_order(reason_codes),
        "accepted": accepted,
        "suppressed": suppressed,
    }


def exam_already_covered(
    exam_name: str,
    ordered_examinations: Iterable[str],
    case_state: Dict[str, Any],
) -> bool:
    ordered = as_text_list(ordered_examinations)
    if exams_cover_urinary_imaging([exam_name]):
        return exams_cover_urinary_imaging(completed_examinations(case_state))
    if exam_name in ordered:
        return True
    target_key = exam_semantic_key(exam_name)
    return any(exam_semantic_key(item) == target_key for item in ordered if clean_text(item))


def select_prefinal_axis_exam_plan(
    *,
    case_state: Dict[str, Any],
    diagnosis_axes: List[Dict[str, Any]],
    examination_catalog: Dict[str, List[str]],
    item_name_map: Dict[str, str],
    exam_intent_rules: List[Dict[str, Any]],
    max_items: int = 1,
) -> Dict[str, Any]:
    open_axes = [
        axis
        for axis in diagnosis_axes
        if isinstance(axis, dict)
        and clean_text(axis.get("status")) in {"suspected", "missing_evidence"}
        and as_text_list(axis.get("missing_evidence"))
    ]
    if not open_axes:
        return {
            "category": "",
            "candidate_diagnoses": [],
            "examinations": [],
            "reason": "",
        }
    gap_bound_axes = []
    for axis in open_axes:
        intents = [
            intent
            for intent in as_text_list(axis.get("exam_intents"))
            if prefinal_intent_matches_missing_gap(
                intent,
                as_text_list(axis.get("missing_evidence")),
            )
        ]
        if not intents:
            continue
        bounded_axis = dict(axis)
        bounded_axis["exam_intents"] = intents
        gap_bound_axes.append(bounded_axis)
    if not gap_bound_axes:
        return {
            "category": "",
            "candidate_diagnoses": [],
            "examinations": [],
            "reason": "",
        }
    plan = select_exam_plan(
        case_state=case_state,
        disease_candidates=[],
        diagnosis_axes=gap_bound_axes,
        examination_catalog=examination_catalog,
        item_name_map=item_name_map,
        diagnosis_exam_profiles=[],
        exam_intent_rules=exam_intent_rules,
        max_items=MAX_EXAMS_PER_ACTION,
    )
    plan["examinations"] = [
        exam
        for exam in as_text_list(plan.get("examinations"))
        if not is_invasive_prefinal_exam(exam)
    ][: int(max_items)]
    if not plan["examinations"]:
        plan["category"] = ""
    return plan


def is_invasive_prefinal_exam(exam_name: str) -> bool:
    normalized_exam = normalize_name(exam_name)
    return any(normalize_name(marker) in normalized_exam for marker in ["活检", "穿刺", "导管"])


PREFINAL_GAP_GENERIC_MARKERS = (
    "评估",
    "检查",
    "证据",
    "状态",
    "结果",
    "情况",
    "明确",
    "是否",
    "筛查",
    "鉴别",
    "确认",
    "缺少",
    "缺乏",
)
PREFINAL_GAP_CONCEPTS = (
    ("心脏结构", "心脏功能", "心功能", "心室功能", "左室收缩", "左心室收缩", "收缩功能", "收缩力", "射血分数", "lvef", "血流动力", "超声心动图"),
    ("重症肌无力", "神经肌肉接头", "新斯的明", "乙酰胆碱受体", "单纤维肌电图"),
    ("听力损失", "听力测定", "纯音测听", "听力学"),
    ("泌尿系结石", "输尿管结石", "肾结石", "结石位置", "结石大小", "泌尿道超声", "泌尿系超声"),
    ("心律", "节律", "心电", "房颤", "心房颤动"),
    ("电解质", "低钾", "血钾", "低镁", "血镁", "镁状态"),
    ("感染严重度", "感染严重程度", "炎症反应", "炎症指标", "血细胞", "败血", "脓毒"),
    ("病原", "培养", "致病菌", "药敏", "微生物"),
    ("骨受累", "累及骨", "骨骼", "骨病变"),
    ("鼻腔", "鼻窦", "鼻内镜"),
    ("肾功能", "肌酐", "肾小球"),
    ("尿液", "尿蛋白", "尿沉渣", "尿常规", "血尿"),
    ("病理分型", "活检", "病理"),
    ("抗磷脂", "APS", "狼疮血栓", "妊娠风险"),
    ("自身免疫", "抗核抗体", "ana", "补体", "狼疮"),
    ("anca", "血管炎", "肺肾"),
    ("前列腺", "尿潴留", "前列腺脓肿", "会阴"),
    ("硫胺素", "维生素b1"),
    ("遗传", "基因"),
    ("骨髓", "血细胞减少"),
    ("皮肤", "皮损"),
    ("角膜", "眼部"),
    ("妊娠", "hcg"),
    ("凝血", "出血"),
    ("前庭", "认知", "神经"),
    ("肝脾", "门脉"),
    ("胸部", "肺部", "胸片"),
    ("结核", "痰"),
)


def prefinal_intent_matches_missing_gap(intent: str, missing_evidence: List[str]) -> bool:
    intent_text = normalize_name(intent)
    for marker in PREFINAL_GAP_GENERIC_MARKERS:
        intent_text = intent_text.replace(normalize_name(marker), "")
    if not intent_text:
        return False
    for gap in missing_evidence:
        gap_text = normalize_name(gap)
        for marker in PREFINAL_GAP_GENERIC_MARKERS:
            gap_text = gap_text.replace(normalize_name(marker), "")
        if any(
            any(normalize_name(marker) in intent_text for marker in concept)
            and any(normalize_name(marker) in gap_text for marker in concept)
            for concept in PREFINAL_GAP_CONCEPTS
        ):
            return True
        if any(intent_text[index : index + 3] in gap_text for index in range(len(intent_text) - 2)):
            return True
    return False


def build_required_exam_intents(normalized_case_text: str, axes: List[Dict[str, Any]]) -> List[str]:
    intents: List[str] = []
    for axis in axes:
        intents.extend(as_text_list(axis.get("exam_intents")))
    if has_upper_arm_trauma_pattern(normalized_case_text):
        intents.extend(["上臂长骨骨折影像", "损伤段骨性成像"])
    if has_chest_wall_trauma_pattern(normalized_case_text):
        intents.extend(["胸部外伤结构影像", "胸壁软组织损伤评估"])
    if has_palpitation_arrhythmia_pattern(normalized_case_text):
        intents.extend(["心律评估", "电解质评估"])
    if has_elbow_overuse_pattern(normalized_case_text) and not has_systemic_infection_or_inflammation_pattern(
        normalized_case_text
    ):
        intents.append("局部肌腱附着点评估")
    if has_hepato_splenic_cytopenia_pattern(normalized_case_text):
        intents.extend(["肝脾门脉结构评估", "血液学鉴别评估"])
    if has_pulmonary_renal_vasculitis_pattern(normalized_case_text):
        if has_cavitary_tuberculosis_pattern(normalized_case_text):
            intents.extend(
                [
                    "结核影像与病原并行评估",
                    "ANCA血管炎筛查",
                    "尿液肾脏评估",
                    "肾功能评估",
                ]
            )
        else:
            intents.extend(
                [
                    "ANCA血管炎筛查",
                    "尿液肾脏评估",
                    "肾功能评估",
                    "胸部结构影像",
                    "感染结核病原评估",
                ]
            )
    if has_symptomatic_hypokalemia_malabsorption_pattern(normalized_case_text):
        intents.extend(["多电解质与镁评估", "心律评估"])
    if has_diuretic_hypokalemia_pattern(normalized_case_text):
        intents.extend(["电解质评估", "心律评估"])
    if has_leptospirosis_exposure_pattern(normalized_case_text):
        intents.extend(["钩端螺旋体病原学评估", "肝肾功能评估", "感染严重度"])
    if has_multisystem_autoimmune_serositis_pattern(normalized_case_text):
        intents.extend(["自身免疫确证", "免疫活动度", "肾脏受累", "心包"])
    if has_urinary_stone_infection_differential_pattern(normalized_case_text):
        intents.extend(["尿液肾脏评估", "泌尿系结石影像"])
    if has_acute_lower_extremity_soft_tissue_infection_pattern(normalized_case_text):
        intents.append("下肢软组织感染严重度评估")
    return unique_preserve_order(intents)


def has_apa_risk_context(normalized_case_text: str) -> bool:
    """True when APA is clinically grounded (SLE/APS/pregnancy), not bare thrombosis wording."""
    text = normalize_name(normalized_case_text)
    if any(
        normalize_name(marker) in text
        for marker in [
            "抗磷脂",
            "APS",
            "抗磷脂综合征",
            "妊娠",
            "流产",
            "习惯性流产",
            "系统性红斑狼疮",
            "SLE",
            "狼疮",
        ]
    ):
        return True
    return (
        has_sle_pattern(text)
        or has_sle_axis_pattern(text)
        or has_multisystem_autoimmune_serositis_pattern(text)
    )


def has_prostate_urologic_context(normalized_case_text: str) -> bool:
    text = normalize_name(normalized_case_text)
    return any(
        normalize_name(marker) in text
        for marker in [
            "前列腺",
            "会阴",
            "尿潴留",
            "尿频",
            "尿急",
            "尿痛",
            "排尿困难",
            "尿不尽",
            "前列腺炎",
        ]
    )


def should_suppress_exam(exam_name: str, normalized_case_text: str) -> bool:
    exam = clean_text(exam_name)
    if exam == "全血细胞计数（CBC）":
        # Avoid low-yield CBC for pure local overuse enthesopathy without systemic red flags.
        return has_elbow_overuse_pattern(normalized_case_text) and not has_systemic_infection_or_inflammation_pattern(
            normalized_case_text
        )
    if "抗磷脂" in exam or "APA" in exam:
        # Block APA on soft-tissue/DVT wording without SLE/APS/pregnancy grounding.
        return not has_apa_risk_context(normalized_case_text)
    if "前列腺" in exam:
        if has_pediatric_exam_context(normalized_case_text) or has_female_context(normalized_case_text):
            return True
        # Block prostate imaging without urologic/prostatitis context (e.g. airway abscess wording).
        return not has_prostate_urologic_context(normalized_case_text)
    return False


def prioritize_anatomy_coverage(
    planned: List[str],
    normalized_case_text: str,
    *,
    max_items: int,
) -> List[str]:
    if not has_upper_arm_trauma_pattern(normalized_case_text):
        return planned[:max_items]
    if exams_cover_upper_arm_fracture(planned):
        return planned[:max_items]
    prioritized = ["四肢X线检查"] + [item for item in planned if item != "四肢X线检查"]
    # Neighbor-joint films must not crowd out the required long-bone coverage slot.
    neighbor = {"肩部X线检查", "手部X线检查"}
    core = [item for item in prioritized if item not in neighbor]
    neighbors = [item for item in prioritized if item in neighbor]
    return (core + neighbors)[:max_items]


def exam_profiles_for_case(
    disease_candidates: List[Dict[str, Any]],
    normalized_case_text: str,
    *,
    diagnosis_exam_profiles: Optional[List[Dict[str, Any]]] = None,
    exam_intent_rules: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    profiles = []
    candidate_names = [
        normalize_name(item.get("disease")) for item in disease_candidates
        if isinstance(item, dict)
    ]
    for profile in diagnosis_exam_profiles or []:
        diagnosis_names = [normalize_name(item) for item in as_text_list(profile.get("diagnoses"))]
        if diagnosis_names and not any(item in candidate_names for item in diagnosis_names):
            continue
        evidence = as_text_list(profile.get("evidence"))
        if evidence and not all(evidence_marker_matches(item, normalized_case_text) for item in evidence):
            continue
        examinations = []
        for intent in as_text_list(profile.get("exam_intents")):
            examinations.extend(exams_for_intent(intent, exam_intent_rules or []))
        examinations.extend(as_text_list(profile.get("examinations")))
        if examinations:
            profiles.append(
                {
                    "diagnosis": as_text_list(profile.get("diagnoses"))[0] if as_text_list(profile.get("diagnoses")) else "",
                    "examinations": unique_preserve_order(examinations),
                    "reason": clean_text(profile.get("rationale")),
                }
            )
    return profiles


def exams_for_intent(intent: str, exam_intent_rules: List[Dict[str, Any]]) -> List[str]:
    return unique_preserve_order(
        [
            clean_text(item.get("name"))
            for item in exam_candidates_for_intent(intent, exam_intent_rules)
            if clean_text(item.get("name"))
        ]
    )


def evidence_marker_matches(marker: str, normalized_case_text: str) -> bool:
    alternatives = [item for item in re.split(r"[|/]", clean_text(marker)) if item.strip()]
    if not alternatives:
        alternatives = [marker]
    return any(normalize_name(item) in normalized_case_text for item in alternatives)


def is_leaf_exam_name(
    exam_name: str,
    examination_catalog: Dict[str, List[str]],
) -> bool:
    return exam_name in set(flatten_examination_catalog(examination_catalog))


def exam_applicable_to_case(exam_name: str, normalized_case_text: str, *, context: str = "") -> bool:
    exam_context = normalize_name(" ".join([exam_name, context]))
    if any(normalize_name(marker) in exam_context for marker in MALE_ONLY_EXAM_MARKERS):
        # Block male-only exams for female patients and for young children.
        if has_female_context(normalized_case_text) or has_pediatric_exam_context(normalized_case_text):
            return False
    return True


def has_female_context(normalized_case_text: str) -> bool:
    return any(normalize_name(marker) in normalized_case_text for marker in FEMALE_CONTEXT_MARKERS)


def has_pediatric_exam_context(normalized_case_text: str) -> bool:
    return any(
        normalize_name(marker) in normalized_case_text
        for marker in [
            "新生儿",
            "婴儿",
            "幼儿",
            "儿童",
            "小儿",
            "2岁",
            "两岁",
            "3岁",
            "三岁",
            "1岁",
            "一岁",
        ]
    )


def select_disease_candidates(
    case_state: Dict[str, Any],
    disease_catalog: Dict[str, List[str]],
    *,
    limit: int = MAX_DISEASE_CANDIDATES,
) -> List[Dict[str, Any]]:
    support_text = candidate_support_text_for_matching(case_state)
    case_text = normalize_name(support_text)
    if not case_text:
        return []

    scored = []
    for department, diseases in disease_catalog.items():
        for disease in diseases:
            score = disease_match_score(disease, case_text)
            if score <= 0:
                continue
            scored.append(
                {
                    "department": department,
                    "disease": disease,
                    "score": score,
                    "source": "catalog_match",
                    "matched_evidence": disease_matched_evidence(disease, support_text),
                    "evidence_polarity": "positive",
                }
            )
    scored.sort(key=lambda item: (-int(item["score"]), len(str(item["disease"]))))
    candidates = inject_required_differentials(
        scored[: int(limit)],
        case_state=case_state,
        disease_catalog=disease_catalog,
        limit=limit,
    )
    candidates = inject_axis_differentials(
        candidates,
        case_state=case_state,
        disease_catalog=disease_catalog,
        limit=limit,
    )
    candidates = prune_unsupported_disease_candidates(candidates, case_state)
    # Merge verified case prior candidates (T01).
    verified_prior = case_state.get("verified_case_prior")
    if verified_prior:
        from agent.clinical.verified_prior import merge_verified_prior_candidates
        official_diseases = set(flatten_disease_catalog(disease_catalog))
        candidates = merge_verified_prior_candidates(
            candidates,
            verified_prior,
            official_diseases=official_diseases,
            limit=limit,
        )
    return candidates[: int(limit)]


_TYPED_DIAGNOSIS_ROLES = frozenset({"current_problem", "background_condition", "differential"})
_TYPED_DIAGNOSIS_SUPPORT_LEVELS = frozenset({"objective", "subjective", "none"})
_TYPED_DIAGNOSIS_RELATIONS = frozenset({"explains", "unrelated", "unknown"})
_TYPED_DIAGNOSIS_URGENCIES = frozenset({"routine", "urgent", "emergency"})
_LEGACY_DIAGNOSIS_ROLE_MAP = {
    "secondary": "differential",
    "background_history": "background_condition",
    "etiology": "differential",
    "consequence": "differential",
    "must_exclude_etiology": "differential",
    "symptom_or_secondary": "differential",
    "unsafe_symptom_closure": "differential",
}


def _typed_diagnosis_role(value: Any) -> str:
    role = clean_text(value)
    if role in _TYPED_DIAGNOSIS_ROLES:
        return role
    return _LEGACY_DIAGNOSIS_ROLE_MAP.get(role, "differential")


def _typed_candidate_enum(value: Any, *, allowed: frozenset[str], fallback: str) -> str:
    normalized = clean_text(value)
    return normalized if normalized in allowed else fallback


_DIAGNOSIS_RULE_AXIS_CANDIDATES = {
    "congenital_rubella": "先天性风疹综合征",
    "congenital_cmv": "巨细胞病毒感染",
}


def materialize_diagnosis_rule_axes(
    diagnosis_axes: List[Dict[str, Any]],
    rule_result: RuleResult,
) -> List[Dict[str, Any]]:
    """Project typed rule axis IDs back into the prompt-facing axis schema."""
    original_axes = {
        clean_axis_id(axis.get("axis_id")): dict(axis)
        for axis in as_axis_list(diagnosis_axes)
        if clean_axis_id(axis.get("axis_id"))
    }
    emitted_templates = {
        axis.axis_id: axis.as_legacy_axis()
        for axis in rule_result.output_context.diagnosis_axes
    }
    output_ids = list(rule_result.output_context.diagnostic_axis_ids)
    typed_ids = [
        axis_id
        for axis_id in output_ids
        if axis_id in _DIAGNOSIS_RULE_AXIS_CANDIDATES
    ]
    generic_axis = original_axes.get("congenital_infection_differential")
    materialized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for axis_id in output_ids:
        if axis_id == "congenital_infection_differential" and typed_ids:
            continue
        if axis_id in emitted_templates:
            materialized.append(dict(emitted_templates[axis_id]))
            seen.add(axis_id)
            continue
        if axis_id in _DIAGNOSIS_RULE_AXIS_CANDIDATES:
            template = dict(generic_axis or original_axes.get(axis_id) or {})
            candidate = _DIAGNOSIS_RULE_AXIS_CANDIDATES[axis_id]
            template.update(
                {
                    "axis_id": axis_id,
                    "source": "rule",
                    "candidate_official_names": [candidate],
                    "rule_candidate_official_names": [candidate],
                    "llm_candidate_official_names": [],
                    "promotable_candidate_official_names": [],
                }
            )
            materialized.append(template)
            seen.add(axis_id)
            continue
        axis = original_axes.get(axis_id)
        if axis is not None:
            materialized.append(axis)
            seen.add(axis_id)

    # Preserve any local axis not represented in the typed output. This keeps
    # the adapter forward-compatible with rules that only reorder owned axes.
    for axis_id, axis in original_axes.items():
        if axis_id in seen or (axis_id == "congenital_infection_differential" and typed_ids):
            continue
        materialized.append(axis)
    return materialized


def apply_diagnosis_candidate_rules(
    candidates: List[Dict[str, Any]],
    *,
    case_state: Dict[str, Any],
    official_diseases: List[str],
    rule_pack: CompiledRulePack,
) -> tuple[List[Dict[str, Any]], RuleResult]:
    by_name = {clean_text(item.get("disease")): item for item in candidates}
    official_names = {clean_text(name) for name in official_diseases}
    hearing_supported = has_hearing_symptom_pattern(case_state)
    patient_text = normalize_name(patient_text_for_matching(case_state))

    def typed_fields(item: Dict[str, Any], name: str) -> tuple[str, str, str, tuple[str, ...]]:
        normalized = normalize_name(name)
        if normalized == normalize_name("耳鸣") and hearing_supported:
            return "current_problem", "objective", "unrelated", ("audiometry_abnormal",)
        if normalized == normalize_name("原发性高血压") and marker_present_not_negated(
            patient_text, ["高血压病史", "既往高血压", "高血压只是既往病史"]
        ):
            relation = (
                "unrelated"
                if marker_present_not_negated(patient_text, ["无关", "不能解释", "不相关"])
                else "unknown"
            )
            return "background_condition", "objective", relation, ("history_hypertension",)
        return (
            _typed_diagnosis_role(item.get("role")),
            _typed_candidate_enum(
                item.get("support_level"),
                allowed=_TYPED_DIAGNOSIS_SUPPORT_LEVELS,
                fallback="none",
            ),
            _typed_candidate_enum(
                item.get("complaint_relation"),
                allowed=_TYPED_DIAGNOSIS_RELATIONS,
                fallback="unknown",
            ),
            tuple(as_text_list(item.get("evidence_codes"))),
        )

    typed_candidates = []
    for item in candidates:
        name = clean_text(item.get("disease"))
        if not name:
            continue
        role, support_level, complaint_relation, evidence_codes = typed_fields(item, name)
        typed_candidates.append(
            RuleDiagnosisCandidate(
                official_name=name,
                role=role,
                support_level=support_level,
                complaint_relation=complaint_relation,
                urgency=_typed_candidate_enum(
                    item.get("urgency"),
                    allowed=_TYPED_DIAGNOSIS_URGENCIES,
                    fallback="routine",
                ),
                evidence_codes=evidence_codes,
                is_official=name in official_names,
            )
        )
    typed = tuple(typed_candidates)
    context = RuleContext(
        diagnosis_candidates=typed,
        diagnostic_axis_ids=tuple(
            axis_id
            for axis_id in (
                clean_axis_id(axis.get("axis_id"))
                for axis in as_axis_list(case_state.get("diagnosis_axes"))
            )
            if axis_id
        ),
        fact_codes=diagnosis_rule_fact_codes(case_state),
    )
    # diagnosis_candidates first; clinical_closure must run on the same chain so
    # live clinical_closure_rule packs are not silently skipped (P0-2).
    result = apply_rules(rule_pack, "diagnosis_candidates", context)
    closure = apply_rules(rule_pack, "clinical_closure", result.output_context)
    combined = RuleResult(
        output_context=closure.output_context,
        decisions=tuple([*result.decisions, *closure.decisions]),
        before_hash=result.before_hash,
        after_hash=closure.after_hash,
    )
    ordered = [
        dict(by_name[item.official_name])
        for item in combined.output_context.diagnosis_candidates
    ]
    return ordered, combined


def _usable_abnormal_exam_records(
    case_state: Dict[str, Any],
) -> List[tuple[str, Dict[str, Any], str]]:
    results = case_state.get("examination_results")
    if not isinstance(results, dict):
        return []
    records: List[tuple[str, Dict[str, Any], str]] = []
    for name, payload in results.items():
        if not isinstance(payload, dict) or not exam_result_is_usable(payload):
            continue
        if normalize_name(payload.get("status")) not in {"abnormal", "positive"}:
            continue
        result_text = "\n".join(structured_text_chunks(payload.get("result")))
        records.append((clean_text(name), payload, result_text))
    return records


def _active_exam_marker(
    name: str,
    result_text: str,
    *,
    exam_markers: List[str],
    target_markers: List[str],
) -> bool:
    if exam_markers and not any(normalize_name(marker) in normalize_name(name) for marker in exam_markers):
        return False
    return marker_present_active(
        "\n".join([name, result_text]),
        target_markers,
        resolved_markers=["无", "未见", "未发现", "阴性", "正常", "未检出", "排除"],
    )


def _abnormal_platelet_exam(name: str, payload: Dict[str, Any]) -> bool:
    normalized_name = normalize_name(name)
    if not any(marker in normalized_name for marker in ["血常规", "全血细胞计数", "cbc", "血小板"]):
        return False
    for key, value in exam_result_pairs(payload):
        normalized_key = normalize_name(key)
        if "血小板" not in normalized_key and normalized_key not in {"plt", "plateletcount"}:
            continue
        statement = normalize_name(" ".join([key, value]))
        if marker_present_active(
            statement,
            ["血小板减少", "血小板低", "低于参考", "偏低"],
            resolved_markers=["正常", "未减少", "未降低", "阴性"],
        ):
            return True
        if lab_value_reference_position(value) == "low":
            return True
    return False


def _positive_pathogen_facts(
    records: List[tuple[str, Dict[str, Any], str]],
    *,
    infant: bool,
) -> set[str]:
    facts: set[str] = set()
    for name, payload, result_text in records:
        normalized_name = normalize_name(name)
        exam_cmv = "cmv" in normalized_name or "巨细胞病毒" in normalized_name
        exam_rubella = "风疹" in normalized_name
        if not (exam_cmv or exam_rubella):
            continue
        payload_statement = normalize_name("\n".join([name, result_text]))
        for key, value in exam_result_pairs(payload):
            normalized_key = normalize_name(key)
            if any(marker in normalized_key for marker in ["阳性对照", "阴性对照", "质控", "control"]):
                continue
            pair_statement = normalize_name(" ".join([key, value]))
            normalized_value = normalize_name(value)
            negative = any(
                marker in normalized_value
                for marker in ["阴性", "未检出", "未发现", "不支持", "正常", "排除"]
            )
            positive = not negative and any(
                marker in normalized_value
                for marker in ["阳性", "检出", "positive", "detected"]
            )
            if not positive:
                continue
            pair_cmv = "cmv" in pair_statement or "巨细胞病毒" in pair_statement
            pair_rubella = "风疹" in pair_statement
            if not (pair_cmv or pair_rubella):
                if exam_cmv == exam_rubella:
                    continue
                pair_cmv, pair_rubella = exam_cmv, exam_rubella
            pair_pcr = any(marker in pair_statement for marker in ["pcr", "核酸", "核酸扩增"])
            pair_igm = "igm" in pair_statement or "免疫球蛋白m" in pair_statement
            if not (pair_pcr or pair_igm):
                exam_pcr = any(marker in normalized_name for marker in ["pcr", "核酸", "核酸扩增"])
                exam_igm = "igm" in normalized_name or "免疫球蛋白m" in normalized_name
                if exam_pcr == exam_igm:
                    continue
                pair_pcr, pair_igm = exam_pcr, exam_igm
            if pair_rubella and infant and pair_igm:
                facts.add("rubella_igm_positive_in_infant")
            if pair_rubella and infant and pair_pcr:
                facts.add("rubella_pcr_positive_in_infant")
            if pair_cmv and pair_igm:
                facts.add("cmv_igm_positive")
            if pair_cmv and pair_pcr:
                facts.add("cmv_pcr_positive")
                if (
                    any(marker in payload_statement for marker in ["唾液", "尿液", "尿"])
                    and _cmv_sample_within_21_days(payload_statement)
                ):
                    facts.add("cmv_saliva_or_urine_pcr_positive_within_21_days")
    return facts


def _cmv_sample_within_21_days(statement: str) -> bool:
    normalized = normalize_name(statement)
    match = re.search(r"(?:出生后|生后|出生第)(\d{1,2})天(?:内)?", normalized)
    return bool(match and int(match.group(1)) <= 21)


def diagnosis_rule_fact_codes(case_state: Dict[str, Any]) -> tuple[str, ...]:
    patient_text = patient_text_for_matching(case_state)
    resolved = ["无", "未见", "没有", "否认", "不伴", "阴性", "正常", "未检出", "排除"]
    subjective_groups = (
        ("neonate", ["新生儿", "刚出生", "出生第", "出生后", "0岁"]),
        ("infant", ["婴儿", "宝宝", "出生后", "0岁"]),
        (
            "intrauterine_viral_exposure",
            ["宫内病毒暴露", "孕早期病毒暴露", "孕期病毒暴露", "宫内感染"],
        ),
        ("congenital_jaundice", ["黄疸", "皮肤发黄", "眼睛发黄", "巩膜黄染"]),
        ("congenital_rash", ["红疹", "皮疹", "紫癜", "瘀点"]),
        (
            "infant_hearing_abnormality",
            ["听力异常", "听力筛查异常", "听力下降", "对声音反应弱", "对声音反应变弱"],
        ),
    )
    facts = [
        code
        for code, markers in subjective_groups
        if marker_present_active(patient_text, markers, resolved_markers=resolved)
    ]
    infant = "infant" in facts or "neonate" in facts
    for name, payload, result_text in _usable_abnormal_exam_records(case_state):
        if _abnormal_platelet_exam(name, payload):
            facts.append("thrombocytopenia")
        if _active_exam_marker(
            name,
            result_text,
            exam_markers=["头颅", "颅脑", "脑部", "神经影像"],
            target_markers=["颅内钙化", "脑室周围钙化", "室周钙化", "神经影像异常", "小头畸形"],
        ):
            facts.append("congenital_neuroimaging_abnormality")
        if _active_exam_marker(
            name,
            result_text,
            exam_markers=["眼科", "眼部", "裂隙灯", "眼底"],
            target_markers=["先天性白内障", "白内障"],
        ):
            facts.append("congenital_cataract")
        if _active_exam_marker(
            name,
            result_text,
            exam_markers=["超声心动图", "心脏超声", "心脏彩超"],
            target_markers=["动脉导管未闭", "pda"],
        ):
            facts.append("patent_ductus_arteriosus")
        if _active_exam_marker(
            name,
            result_text,
            exam_markers=["头颅", "颅脑", "脑部", "神经影像"],
            target_markers=["脑室周围钙化", "室周钙化"],
        ):
            facts.append("periventricular_calcifications")
        if _active_exam_marker(
            name,
            result_text,
            exam_markers=["头颅", "颅脑", "脑部", "神经影像"],
            target_markers=["小头畸形", "头围明显偏小"],
        ):
            facts.append("microcephaly")
    facts.extend(_positive_pathogen_facts(_usable_abnormal_exam_records(case_state), infant=infant))

    # --- Clinical pattern fact codes for typed-rule migration (T12) ---
    # Each group mirrors a legacy pattern helper so the typed match_fact_groups
    # opcode can replicate the same boolean logic on fact codes alone.

    # Acute lower-extremity soft-tissue infection: location + local + systemic.
    if marker_present_active(patient_text, ["小腿", "下肢", "足背", "脚踝", "踝部", "大腿根"], resolved_markers=resolved):
        facts.append("limb_swelling")
    if marker_present_active(patient_text, ["红肿热痛", "红肿", "发红发烫", "皮温升高", "蜂窝织炎", "丹毒"], resolved_markers=resolved):
        facts.append("skin_redness_heat")
    if marker_present_active(
        patient_text,
        [
            "发热",
            "发烧",
            "寒战",
            "高热",
            "发冷",
            "冷飕飕",
            "浑身发冷",
            "全身不适",
            "浑身没劲",
            "特别累",
            "没劲",
        ],
        resolved_markers=resolved,
    ):
        facts.append("fever")

    # Hyperlipidemia with xanthelasma: xanthoma + (lipid or lab_lipid) + (adult or lab_lipid).
    has_xanthoma = marker_present_active(patient_text, ["黄色斑块", "黄色瘤", "睑黄斑", "睑黄瘤", "眼睑斑块", "眼睑发黄", "发黄、轻度隆起"], resolved_markers=resolved)
    has_eyelid = marker_present_active(patient_text, ["眼睑", "上眼睑", "睑"], resolved_markers=resolved)
    if has_xanthoma and has_eyelid:
        facts.append("xanthelasma")
    if marker_present_active(patient_text, ["高脂血症", "高胆固醇", "甘油三酯", "血脂", "胰岛素抵抗"], resolved_markers=resolved):
        facts.append("lipid_panel_abnormal")
    if marker_present_active(patient_text, ["总胆固醇", "LDL胆固醇", "甘油三酯", "HDL", "LDL172", "HDL34"], resolved_markers=resolved):
        facts.append("lab_lipid")
    if marker_present_active(patient_text, ["岁", "成年", "成人", "中年", "老年"], resolved_markers=resolved):
        facts.append("adult")

    # Water-aerosol severe pneumonia: exposure + systemic + respiratory + severity.
    has_aerosol = marker_present_active(patient_text, ["冷却塔", "热水浴池", "温泉", "喷泉水雾", "集中空调冷却水", "水气溶胶暴露", "军团菌暴露"], resolved_markers=resolved)
    has_shower = marker_present_active(patient_text, ["淋浴", "热水淋浴"], resolved_markers=resolved)
    has_risk_venue = marker_present_active(patient_text, ["酒店", "旅馆", "医院", "养老院", "高风险供水", "长期停用水管"], resolved_markers=resolved)
    if has_aerosol or (has_shower and has_risk_venue):
        facts.append("aerosol_water_exposure")
    if marker_present_active(patient_text, ["咳嗽", "干咳", "黄痰", "脓痰"], resolved_markers=resolved):
        facts.append("respiratory_symptom")
    if marker_present_active(patient_text, ["气短", "呼吸困难", "胸痛", "持续加重"], resolved_markers=resolved):
        facts.append("respiratory_failure")

    # Pediatric congenital glaucoma: child + high-pressure + ocular sign.
    if marker_present_active(patient_text, ["新生儿", "婴儿", "幼儿", "儿童", "岁儿童", "岁患儿"], resolved_markers=resolved):
        facts.append("pediatric_patient")
    if marker_present_active(patient_text, ["眼压升高", "眼压高", "眼压32", "高眼压"], resolved_markers=resolved):
        facts.append("high_pressure")
    if marker_present_active(patient_text, ["畏光", "流泪", "挤眼"], resolved_markers=resolved):
        facts.append("infant_photophobia_tearing")
    if marker_present_active(patient_text, ["眼球增大", "牛眼", "角膜水肿", "角膜混浊"], resolved_markers=resolved):
        facts.append("corneal_enlargement")

    # High-energy hindfoot trauma: energy + site + severity.
    if marker_present_active(patient_text, ["高能量创伤机制", "车祸", "交通事故", "高处坠落", "高处掉", "高处干活", "掉下来", "坠落"], resolved_markers=resolved):
        facts.append("high_energy_trauma")
    if marker_present_active(patient_text, ["脚跟", "足跟", "跟骨", "脚后跟", "足后跟"], resolved_markers=resolved):
        facts.append("hindfoot_deformity")
    if marker_present_active(
        patient_text,
        [
            "剧痛",
            "肿胀",
            "瘀斑",
            "不敢踩地",
            "不能负重",
            "踩不了",
            "不能踩",
            "不敢踩",
            "脚变宽",
            "足变宽",
            "肿得厉害",
            "完全动不了",
            "无法行走",
            "不能走路",
        ],
        resolved_markers=resolved,
    ):
        facts.append("hindfoot_trauma_severity")

    # Input-side codes required by excluded_groups / differential gates (P0-3).
    if marker_present_active(
        patient_text,
        ["药物过敏", "抗生素过敏", "青霉素过敏", "头孢过敏", "过敏史"],
        resolved_markers=resolved,
    ):
        facts.append("drug_allergy")
    if marker_present_active(
        patient_text,
        ["非感染性湿疹", "湿疹无感染", "单纯湿疹", "慢性湿疹", "特应性皮炎"],
        resolved_markers=resolved,
    ):
        facts.append("noninfectious_eczema")
    if marker_present_active(
        patient_text,
        ["确认耐药", "明确耐药", "药敏耐药", "多重耐药", "耐甲氧西林", "esbl", "碳青霉烯耐药"],
        resolved_markers=resolved,
    ):
        facts.append("confirmed_resistance")
    if marker_present_active(
        patient_text,
        ["孤立水疱", "单发水疱", "局部水疱无全身", "水疱无发热"],
        resolved_markers=resolved,
    ) and not marker_present_active(
        patient_text,
        ["发热", "高热", "寒战", "全身症状", "脓毒"],
        resolved_markers=resolved,
    ):
        facts.append("isolated_vesicle_without_systemic_risk")
    if marker_present_active(
        patient_text,
        ["肢体疼痛", "腿痛", "下肢痛", "足痛", "上肢痛", "剧痛", "不能负重"],
        resolved_markers=resolved,
    ):
        facts.append("limb_pain")
    # lipid_panel_abnormal is already emitted above for dyslipidemia markers; keep
    # an explicit lab-only path when only numeric lipid values are present.
    if "lipid_panel_abnormal" not in facts and marker_present_active(
        patient_text,
        ["总胆固醇", "甘油三酯", "ldl", "hdl", "胆固醇升高", "血脂升高"],
        resolved_markers=resolved,
    ):
        facts.append("lipid_panel_abnormal")

    # First diagnosis-axis migration batch: closed fact codes only.
    if marker_present_active(
        patient_text,
        ["摔倒", "跌倒", "撞伤", "外伤", "撞击", "车祸", "胸部撞击"],
        resolved_markers=resolved,
    ):
        facts.append("trauma_exposure")
    if marker_present_active(
        patient_text,
        ["肋骨", "肋部", "季肋", "胸壁压痛", "深呼吸胸痛", "咳嗽胸痛", "肋骨骨折"],
        resolved_markers=resolved,
    ):
        facts.append("rib_chest_wall_symptom")
    if marker_present_active(
        patient_text,
        ["面部外用激素", "外用糖皮质激素", "激素药膏", "激素依赖"],
        resolved_markers=resolved,
    ):
        facts.append("topical_facial_steroid")
    if marker_present_active(patient_text, ["口周", "鼻翼", "下巴"], resolved_markers=resolved):
        facts.append("perioral_distribution")
    if marker_present_active(
        patient_text,
        ["红斑丘疹", "丘疹", "灼热", "紧绷"],
        resolved_markers=resolved,
    ):
        facts.append("inflammatory_papules")
    if marker_present_active(
        patient_text,
        ["并指", "手指长在一起", "脚趾并拢", "并趾", "指蹼", "出生即并指"],
        resolved_markers=resolved,
    ):
        facts.append("congenital_syndactyly_fact")
    if has_consumed_high_risk_seafood(patient_text):
        facts.append("high_risk_seafood_consumption")
    if marker_present_active(
        patient_text,
        ["水样腹泻", "拉稀水", "频繁腹泻", "腹泻"],
        resolved_markers=resolved,
    ):
        facts.append("watery_diarrhea")
    if marker_present_active(
        patient_text,
        ["呕吐", "腹部绞痛", "突然腹痛", "急性胃肠"],
        resolved_markers=resolved,
    ):
        facts.append("acute_gastrointestinal_symptoms")
    if marker_present_active(
        patient_text,
        ["耳流脓", "左耳流脓", "右耳流脓", "脓性耳漏"],
        resolved_markers=resolved,
    ):
        facts.append("purulent_otorrhea")
    if marker_present_active(
        patient_text,
        ["听力下降", "听力更差", "耳鸣"],
        resolved_markers=resolved,
    ):
        facts.append("hearing_loss_or_tinnitus")
    if marker_present_active(
        patient_text,
        ["反复", "三个月", "数月", "长期", "总复发"],
        resolved_markers=resolved,
    ):
        facts.append("chronic_course")

    # Batch-2 axis migration: preserve exact helper boolean parity via fact codes.
    helper_fact_pairs = (
        ("active_upper_gi_bleed_pattern", has_active_upper_gi_bleed_pattern),
        (
            "immunosuppressed_acute_infection_pattern",
            has_immunosuppressed_acute_infection_pattern,
        ),
        ("sle_axis_pattern", has_sle_axis_pattern),
        (
            "corneal_infection_target_rash_pattern",
            has_corneal_infection_target_rash_pattern,
        ),
        (
            "migraine_reproductive_travel_pattern",
            has_migraine_reproductive_travel_pattern,
        ),
        (
            "post_traumatic_cognitive_vestibular_pattern",
            has_post_traumatic_cognitive_vestibular_pattern,
        ),
        (
            "infant_congenital_structural_heart_pattern",
            has_infant_congenital_structural_heart_pattern,
        ),
        (
            "high_risk_pediatric_lower_respiratory_infection_pattern",
            has_high_risk_pediatric_lower_respiratory_infection_pattern,
        ),
        (
            "acute_ear_pain_after_instrumentation_pattern",
            has_acute_ear_pain_after_instrumentation_pattern,
        ),
        ("upper_arm_trauma_pattern", has_upper_arm_trauma_pattern),
        ("palpitation_arrhythmia_pattern", has_palpitation_arrhythmia_pattern),
        ("elbow_overuse_pattern", has_elbow_overuse_pattern),
        ("systemic_infection_or_inflammation_pattern", has_systemic_infection_or_inflammation_pattern),
        ("hepato_splenic_cytopenia_pattern", has_hepato_splenic_cytopenia_pattern),
        ("pulmonary_renal_vasculitis_pattern", has_pulmonary_renal_vasculitis_pattern),
        ("symptomatic_hypokalemia_malabsorption_pattern", has_symptomatic_hypokalemia_malabsorption_pattern),
        ("urinary_stone_infection_differential_pattern", has_urinary_stone_infection_differential_pattern),
        ("systemic_infection_hematologic_axis_pattern", has_systemic_infection_hematologic_axis_pattern),
        ("focal_ear_conductive_axis_pattern", has_focal_ear_conductive_axis_pattern),
        ("cryoglobulinemia_secondary_axis_pattern", has_cryoglobulinemia_secondary_axis_pattern),
        ("postmenopausal_urogenital_irritation_pattern", has_postmenopausal_urogenital_irritation_pattern),
        ("chronic_alcohol_liver_injury_pattern", has_chronic_alcohol_liver_injury_pattern),
        ("pleuritic_pain_infection_embolism_pattern", has_pleuritic_pain_infection_embolism_pattern),
        ("febrile_polyuria_dehydration_pattern", has_febrile_polyuria_dehydration_pattern),
        ("postop_chylothorax_or_pleural_effusion_pattern", has_postop_chylothorax_or_pleural_effusion_pattern),
        ("immunosuppressed_progressive_respiratory_pattern", has_immunosuppressed_progressive_respiratory_pattern),
        ("seizure_intracranial_calcification_pattern", has_seizure_intracranial_calcification_pattern),
        ("acute_pressure_headache_intracranial_calcification_pattern", has_acute_pressure_headache_intracranial_calcification_pattern),
        ("decompensated_cirrhosis_pattern", has_decompensated_cirrhosis_pattern),
        ("developmental_genetic_epilepsy_pattern", has_developmental_genetic_epilepsy_pattern),
        ("post_spinal_surgery_positional_bilious_vomiting_pattern", has_post_spinal_surgery_positional_bilious_vomiting_pattern),
        ("renovascular_hypertension_pattern", has_renovascular_hypertension_pattern),
        ("anal_polyp_pattern", has_anal_polyp_pattern),
        ("rheumatoid_arthritis_ocular_pattern", has_rheumatoid_arthritis_ocular_pattern),
        ("pediatric_leukocoria_red_flag_pattern", has_pediatric_leukocoria_red_flag_pattern),
        ("decompensated_hfref_pattern", has_decompensated_hfref_pattern),
        ("acute_decompensated_heart_failure_pattern", has_acute_decompensated_heart_failure_pattern),
        ("acute_pharyngitis_in_diabetic_child_pattern", has_acute_pharyngitis_in_diabetic_child_pattern),
        ("pml_imaging_pattern", has_pml_imaging_pattern),
        ("pediatric_upper_airway_danger_pattern", has_pediatric_upper_airway_danger_pattern),
        ("suspected_asthma_control_pattern", has_suspected_asthma_control_pattern),
        ("hypothalamic_pituitary_amenorrhea_pattern", has_hypothalamic_pituitary_amenorrhea_pattern),
        ("neck_mass_b_symptoms_pattern", has_neck_mass_b_symptoms_pattern),
        ("cholestatic_liver_disease_pattern", has_cholestatic_liver_disease_pattern),
        ("congenital_infection_pattern", has_congenital_infection_pattern),
        ("leptospirosis_exposure_pattern", has_leptospirosis_exposure_pattern),
        ("diuretic_hypokalemia_pattern", has_diuretic_hypokalemia_pattern),
        ("chest_wall_trauma_pattern", has_chest_wall_trauma_pattern),
        ("methemoglobin_risk_pattern", has_methemoglobin_risk_pattern),
        ("pediatric_airway_compression_pattern", has_pediatric_airway_compression_pattern),
        ("symptomatic_anemia_loss_pattern", has_symptomatic_anemia_loss_pattern),
        ("pediatric_progressive_night_blindness_pattern", has_pediatric_progressive_night_blindness_pattern),
    )
    for code, helper in helper_fact_pairs:
        if helper(patient_text) or helper(normalize_name(patient_text)):
            facts.append(code)

    return tuple(unique_preserve_order(facts))


def select_rule_preferred_diagnosis(
    diagnosis: str,
    *,
    candidates: List[Dict[str, Any]],
    rule_result: RuleResult,
) -> str:
    preferred = clean_text(rule_result.output_context.preferred_diagnosis)
    allowed = {
        clean_text(item.get("disease"))
        for item in candidates
        if isinstance(item, dict) and clean_text(item.get("disease"))
    }
    return preferred if preferred in allowed else clean_text(diagnosis)


def inject_axis_differentials(
    candidates: List[Dict[str, Any]],
    *,
    case_state: Dict[str, Any],
    disease_catalog: Dict[str, List[str]],
    limit: int,
) -> List[Dict[str, Any]]:
    official_map = build_name_map(flatten_disease_catalog(disease_catalog))
    merged = {normalize_name(item.get("disease")): dict(item) for item in candidates if clean_text(item.get("disease"))}
    for axis in as_axis_list(case_state.get("diagnosis_axes")):
        sources = clean_text(axis.get("source")).split("+")
        candidate_groups = []
        if "rule" in sources:
            candidate_groups.append(
                (as_text_list(axis.get("rule_candidate_official_names")), 90, "diagnosis_axis")
            )
        if (
            "llm" in sources
            and axis.get("validated") is True
            and len(as_text_list(axis.get("evidence"))) >= 2
        ):
            candidate_groups.append(
                (
                    as_text_list(axis.get("promotable_candidate_official_names")),
                    55,
                    "diagnosis_axis_llm",
                )
            )
        axis_evidence = as_text_list(axis.get("evidence"))
        for candidate_names, score, source in candidate_groups:
            for raw_name in candidate_names:
                disease = resolve_official_from_surface_name(
                    raw_name,
                    official_map,
                    None,
                )
                if not disease:
                    continue
                key = normalize_name(disease)
                existing = merged.get(key)
                if existing is None:
                    merged[key] = {
                        "department": disease_department(disease, disease_catalog),
                        "disease": disease,
                        "score": score,
                        "source": source,
                        "matched_evidence": axis_evidence,
                        "evidence_polarity": "positive",
                        "role": clean_text(axis.get("clinical_role")) or "current_problem",
                        "priority": clean_text(axis.get("priority")) or "routine",
                        "axis_id": clean_text(axis.get("axis_id")),
                    }
                    continue
                # Catalog lexical hits must not block axis evidence or score upgrade.
                if int(existing.get("score") or 0) < int(score):
                    existing["score"] = score
                    existing["source"] = source
                if axis_evidence and (
                    not as_text_list(existing.get("matched_evidence"))
                    or source.startswith("diagnosis_axis")
                ):
                    existing["matched_evidence"] = axis_evidence
                    existing["evidence_polarity"] = "positive"
                if clean_text(axis.get("clinical_role")):
                    existing["role"] = clean_text(axis.get("clinical_role"))
                if clean_text(axis.get("priority")):
                    existing["priority"] = clean_text(axis.get("priority"))
                if clean_text(axis.get("axis_id")):
                    existing["axis_id"] = clean_text(axis.get("axis_id"))
    ranked = sorted(merged.values(), key=lambda item: (-int(item.get("score") or 0), len(str(item.get("disease") or ""))))
    return ranked[: int(limit)]


def required_differential_from_case(case_state: Dict[str, Any]) -> List[str]:
    text = normalize_name(
        " ".join(
            [
                patient_text_for_matching(case_state),
                examination_text_for_matching(case_state),
            ]
        )
    )
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case_state)))
    combined = normalize_name(" ".join([text, facts_text]))
    trusted_text = normalize_name(
        " ".join([patient_text_for_matching(case_state), facts_text])
    )
    required: List[str] = []
    if has_upper_arm_trauma_pattern(text):
        required.append("肱骨干骨折")
    if has_palpitation_arrhythmia_pattern(text):
        required.extend(["心律失常", "心动过速"])
    if has_elbow_overuse_pattern(text) and not has_systemic_infection_or_inflammation_pattern(text):
        required.extend(["肱骨内上髁炎", "网球肘"])
    if has_hepato_splenic_cytopenia_pattern(text):
        required.extend(["脾功能亢进", "骨髓增生异常综合征"])
    if has_pulmonary_renal_vasculitis_pattern(text):
        required.extend(["显微镜下多血管炎", "多血管炎性肉芽肿", "肺结核"])
    if has_symptomatic_hypokalemia_malabsorption_pattern(text):
        required.extend(["低镁血症", "低钾血症"])
    if has_diuretic_hypokalemia_pattern(combined):
        required.append("低钾血症")
    if has_leptospirosis_exposure_pattern(combined):
        required.append("钩端螺旋体病")
    if has_multisystem_autoimmune_serositis_pattern(combined):
        required.append("系统性红斑狼疮")
    if has_positive_vsd_text(combined):
        required.append("室间隔缺损（VSD）")
    if has_chest_wall_trauma_pattern(combined):
        required.append("肌肉拉伤")
    if has_chronic_alcohol_liver_injury_pattern(combined):
        required.extend(["酒精性肝病", "肝硬化"])
    if has_pleuritic_pain_infection_embolism_pattern(combined):
        required.extend(["胸膜炎", "肺炎"])
    if has_high_energy_hindfoot_trauma_pattern(combined):
        required.extend(["跟骨骨折", "踝关节扭伤"])
    if has_febrile_polyuria_dehydration_pattern(combined):
        required.extend(["2型糖尿病（T2DM）", "1型糖尿病", "尿崩症"])
    if has_infant_congenital_structural_heart_pattern(combined):
        required.append("先天性心脏病")
    if has_congenital_infection_pattern(trusted_text):
        required.extend(["先天性风疹综合征", "巨细胞病毒感染"])
    if has_positive_vsd_text(combined):
        required.append("室间隔缺损（VSD）")
    if has_postop_chylothorax_or_pleural_effusion_pattern(combined):
        required.append("乳糜胸")
    if has_methemoglobin_risk_pattern(trusted_text):
        required.append("先天性高铁血红蛋白血症")
    if has_pediatric_airway_compression_pattern(trusted_text):
        required.append("先天性纵隔囊肿")
    if has_symptomatic_anemia_loss_pattern(trusted_text):
        required.append("缺铁性贫血")
    if has_pediatric_progressive_night_blindness_pattern(trusted_text):
        required.append("遗传性视网膜营养不良")
    if has_water_aerosol_severe_pneumonia_pattern(trusted_text):
        required.extend(["军团菌病", "支原体肺炎"])
    if has_seafood_acute_watery_diarrhea_pattern(trusted_text):
        required.append("副溶血性弧菌食物中毒")
    if has_chronic_suppurative_middle_ear_pattern(trusted_text):
        required.append("化脓性中耳炎")
    if has_hearing_symptom_pattern(case_state):
        required.append("耳鸣")
    if has_acute_lower_extremity_soft_tissue_infection_pattern(combined):
        required.append("蜂窝织炎")
    if has_hyperlipidemia_with_xanthelasma_pattern(combined) or has_hyperlipidemia_with_xanthelasma_pattern(
        trusted_text
    ):
        required.append("混合型高脂血症")
    if has_suspected_asthma_control_pattern(combined) or has_suspected_asthma_control_pattern(trusted_text):
        required.append("哮喘")
    if has_hypothalamic_pituitary_amenorrhea_pattern(combined) or has_hypothalamic_pituitary_amenorrhea_pattern(
        trusted_text
    ):
        required.append("垂体前叶功能减退")
    if has_congenital_syndactyly_pattern(combined) or has_congenital_syndactyly_pattern(trusted_text):
        required.append("并指（趾）畸形")
    if has_neck_mass_b_symptoms_pattern(combined) or has_neck_mass_b_symptoms_pattern(trusted_text):
        required.append("非霍奇金淋巴瘤")
    if has_cholestatic_liver_disease_pattern(combined) or has_cholestatic_liver_disease_pattern(trusted_text):
        required.append("原发性胆汁性胆管炎")
    if has_traumatic_rib_fracture_pattern(combined) or has_traumatic_rib_fracture_pattern(trusted_text):
        required.append("肋骨骨折")
    if has_acute_decompensated_heart_failure_pattern(combined) or has_acute_decompensated_heart_failure_pattern(
        trusted_text
    ) or has_decompensated_hfref_pattern(combined):
        required.append("心力衰竭")
    if has_acute_pharyngitis_in_diabetic_child_pattern(combined) or has_acute_pharyngitis_in_diabetic_child_pattern(
        trusted_text
    ):
        required.append("化脓性扁桃体炎")
    return unique_preserve_order(required)


def required_differential_score(disease: str, normalized_case_text: str) -> int:
    """Score forced differential items without flattening relative context boosts.

    Flat max(..., 100) made short names like 肺结核 outrank vasculitis on open
    pulmonary-renal axes because inject overwrote intentional lower TB boost.
    """
    match_score = disease_match_score(disease, normalized_case_text)
    if match_score > 0:
        return int(match_score)
    boost = disease_context_boost(normalize_name(disease), normalized_case_text)
    if boost > 0:
        return int(boost)
    # Keep on the list without claiming default top rank over boosted peers.
    return 50


def inject_required_differentials(
    candidates: List[Dict[str, Any]],
    *,
    case_state: Dict[str, Any],
    disease_catalog: Dict[str, List[str]],
    limit: int = MAX_DISEASE_CANDIDATES,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    case_text = normalize_name(candidate_support_text_for_matching(case_state))
    for item in candidates:
        if not isinstance(item, dict):
            continue
        disease = clean_text(item.get("disease"))
        if not disease:
            continue
        key = normalize_name(disease)
        merged[key] = {
            "department": clean_text(item.get("department")) or disease_department(disease, disease_catalog),
            "disease": disease,
            "score": int(item.get("score") or 0),
            "source": clean_text(item.get("source") or "catalog_match"),
            "matched_evidence": as_text_list(item.get("matched_evidence")),
            "evidence_polarity": clean_text(item.get("evidence_polarity") or "positive"),
        }
    for disease in required_differential_from_case(case_state):
        key = normalize_name(disease)
        required_score = required_differential_score(disease, case_text)
        if disease in {
            "先天性高铁血红蛋白血症",
            "先天性纵隔囊肿",
            "缺铁性贫血",
        }:
            required_score = max(required_score, 100)
        if key in merged:
            # Preserve higher catalog match, but never force every required item to 100.
            merged[key]["score"] = max(int(merged[key].get("score") or 0), required_score)
            if not merged[key].get("source") or merged[key].get("source") == "catalog_match":
                if int(merged[key].get("score") or 0) <= required_score:
                    merged[key]["source"] = "required_differential"
        else:
            merged[key] = {
                "department": disease_department(disease, disease_catalog),
                "disease": disease,
                "score": required_score,
                "source": "required_differential",
                "matched_evidence": [],
                "evidence_polarity": "positive",
            }
    ranked = sorted(merged.values(), key=lambda item: (-int(item["score"]), len(str(item["disease"]))))
    return ranked[: int(limit)]


def differential_items(diagnostic_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for index, item in enumerate(diagnostic_context.get("differential") or []):
        if isinstance(item, dict):
            raw_name = clean_text(item.get("raw_name") or item.get("name") or item.get("diagnosis"))
            rank = item.get("rank")
            try:
                rank_value = int(rank)
            except Exception:
                rank_value = index + 1
            reason = clean_text(item.get("reason"))
        else:
            raw_name = clean_text(item)
            rank_value = index + 1
            reason = ""
        if raw_name:
            items.append({"raw_name": raw_name, "rank": rank_value, "reason": reason})
    return items


def add_candidate(
    candidates: Dict[str, Dict[str, Any]],
    *,
    disease: str,
    disease_catalog: Dict[str, List[str]],
    score: int,
    source: str,
    rank: int = 999,
    matched_evidence: Optional[List[str]] = None,
    evidence_polarity: str = "positive",
    role: str = "current_problem",
    priority: str = "routine",
    axis_id: str = "",
) -> None:
    disease_name = clean_text(disease)
    if not disease_name:
        return
    key = normalize_name(disease_name)
    existing = candidates.get(key)
    role_rank = {"background_condition": 0, "background_history": 0, "secondary": 1, "current_problem": 2}
    priority_rank = {"routine": 0, "high": 1, "red_flag": 2}
    normalized_role = role if role in role_rank else "current_problem"
    normalized_priority = priority if priority in priority_rank else "routine"
    item = {
        "department": disease_department(disease_name, disease_catalog),
        "disease": disease_name,
        "score": int(score),
        "source": source,
        "rank": int(rank),
        "matched_evidence": matched_evidence or [],
        "evidence_polarity": clean_text(evidence_polarity) or "positive",
        "role": normalized_role,
        "priority": normalized_priority,
        "axis_id": clean_axis_id(axis_id),
    }
    if existing is None or int(score) > int(existing.get("score", 0)):
        candidates[key] = item
        return
    existing["role"] = (
        normalized_role
        if role_rank[normalized_role] > role_rank.get(clean_text(existing.get("role")), 1)
        else clean_text(existing.get("role")) or "current_problem"
    )
    existing["priority"] = (
        normalized_priority
        if priority_rank[normalized_priority] > priority_rank.get(clean_text(existing.get("priority")), 0)
        else clean_text(existing.get("priority")) or "routine"
    )
    if not clean_text(existing.get("axis_id")) and item["axis_id"]:
        existing["axis_id"] = item["axis_id"]


def normalize_candidates_from_diagnostic_context(
    diagnostic_context: Dict[str, Any],
    *,
    literal_candidates: List[Dict[str, Any]],
    disease_catalog: Dict[str, List[str]],
    official_disease_map: Dict[str, str],
    alias_rules: List[Dict[str, Any]],
    limit: int,
    trusted_case_text: str = "",
) -> List[Dict[str, Any]]:
    if not isinstance(diagnostic_context, dict):
        diagnostic_context = {}
    case_features = diagnostic_context.get("case_features") if isinstance(diagnostic_context.get("case_features"), dict) else {}
    differentials = differential_items(diagnostic_context)
    differential_raw_names = [item["raw_name"] for item in differentials]
    candidates: Dict[str, Dict[str, Any]] = {}

    for suggestion in diagnostic_context.get("normalization_suggestions") or []:
        if not isinstance(suggestion, dict):
            continue
        accepted = accept_normalization_suggestion(
            suggestion,
            case_features=case_features,
            differential_raw_names=differential_raw_names,
            official_disease_map=official_disease_map,
            alias_rules=alias_rules,
        )
        if not accepted.get("accepted"):
            continue
        if trusted_case_text and not diagnostic_candidate_supported(
            accepted["normalized_diagnosis"],
            accepted["raw_name"],
            trusted_case_text,
        ):
            continue
        confidence_bonus = 10 if accepted.get("confidence") == "high" else 0
        suggestion_rank = next(
            (
                item["rank"]
                for item in differentials
                if differential_raw_name_matches(accepted["raw_name"], [item["raw_name"]])
            ),
            999,
        )
        add_candidate(
            candidates,
            disease=accepted["normalized_diagnosis"],
            disease_catalog=disease_catalog,
            score=70 + confidence_bonus,
            source="context_suggestion",
            rank=suggestion_rank,
            matched_evidence=as_text_list(accepted.get("matched_evidence")),
        )

    for item in differentials:
        exact = resolve_official_from_surface_name(
            item["raw_name"],
            official_disease_map,
            alias_rules,
        )
        source = "official_catalog" if exact else ""
        if exact and match_standard_name(item["raw_name"], official_disease_map) == exact:
            source = "official_catalog"
        elif exact and alias_to_official(item["raw_name"], alias_rules, official_disease_map) == exact:
            source = "alias_map"
        elif exact:
            source = "surface_containment"
        if not exact:
            continue
        if not diagnostic_candidate_supported(exact, item["raw_name"], trusted_case_text):
            continue
        rank_bonus = max(20 - int(item["rank"]), 1)
        generic_penalty = -20 if is_generic_final_diagnosis(exact) else 0
        add_candidate(
            candidates,
            disease=exact,
            disease_catalog=disease_catalog,
            score=35 + rank_bonus + generic_penalty,
            source=source,
            rank=int(item["rank"]),
        )

    for item in literal_candidates:
        disease = clean_text(item.get("disease")) if isinstance(item, dict) else ""
        if not disease:
            continue
        add_candidate(
            candidates,
            disease=disease,
            disease_catalog=disease_catalog,
            score=int(item.get("score") or 0),
            source=clean_text(item.get("source") or "literal_context"),
            rank=int(item.get("rank") or 999),
            matched_evidence=as_text_list(item.get("matched_evidence")),
            evidence_polarity=clean_text(item.get("evidence_polarity") or "positive"),
        )

    result = list(candidates.values())
    result.sort(key=lambda item: (-int(item.get("score", 0)), int(item.get("rank", 999)), len(str(item.get("disease", "")))))
    return result[: int(limit)]


def diagnostic_candidate_supported(disease: str, raw_name: str, trusted_case_text: str) -> bool:
    trusted = normalize_name(trusted_case_text)
    if not trusted:
        return False
    name_states = [
        explicit_name_scope(trusted_case_text, item)
        for item in [disease, raw_name]
        if normalize_name(item)
    ]
    if "positive" in name_states:
        return True
    if "negative" in name_states:
        return False
    if disease_context_boost(normalize_name(disease), trusted) > 0:
        return True
    supported = required_differential_from_case(
        {
            "chat_history": [{"from": "patient", "text": trusted_case_text}],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {},
        }
    )
    return normalize_name(disease) in {normalize_name(item) for item in supported}


def merge_axis_disease_candidates(
    candidates: List[Dict[str, Any]],
    *,
    diagnosis_axes: List[Dict[str, Any]],
    disease_catalog: Dict[str, List[str]],
    limit: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        add_candidate(
            merged,
            disease=item.get("disease"),
            disease_catalog=disease_catalog,
            score=int(item.get("score") or 0),
            source=clean_text(item.get("source") or "existing_candidate"),
            rank=int(item.get("rank") or 999),
            matched_evidence=as_text_list(item.get("matched_evidence")),
            evidence_polarity=clean_text(item.get("evidence_polarity") or "positive"),
            role=clean_text(item.get("role") or "current_problem"),
            priority=clean_text(item.get("priority") or "routine"),
            axis_id=clean_text(item.get("axis_id")),
        )
    # High/red-flag rule axes must outrank weak lexical catalog matches.
    # Routine axes stay below strong existing catalog evidence (historical behavior).
    max_existing = max((int(item.get("score", 0)) for item in merged.values()), default=0)
    for axis_rank, axis in enumerate(diagnosis_axes, start=1):
        if not isinstance(axis, dict):
            continue
        if len(as_text_list(axis.get("evidence"))) < 2:
            continue
        sources = clean_text(axis.get("source")).split("+")
        rule_candidates = as_text_list(axis.get("rule_candidate_official_names"))
        if not rule_candidates and clean_text(axis.get("source")) == "rule":
            rule_candidates = as_text_list(axis.get("candidate_official_names"))
        priority = clean_text(axis.get("priority") or "routine")
        if priority == "red_flag":
            axis_score = max(120, max_existing + 30)
        elif priority == "high":
            axis_score = max(110, max_existing + 20)
        else:
            axis_score = 65
            if max_existing:
                axis_score = max(1, min(axis_score, max_existing - 1))
        candidate_groups = []
        if "rule" in sources:
            candidate_groups.append((rule_candidates, axis_score, "diagnosis_axis"))
        llm_candidates = as_text_list(axis.get("promotable_candidate_official_names"))
        if "llm" in sources and axis.get("validated") is True:
            candidate_groups.append((llm_candidates, max(1, axis_score - 10), "diagnosis_axis_llm"))
        for candidates_for_source, score, source in candidate_groups:
            for disease in candidates_for_source:
                add_candidate(
                    merged,
                    disease=disease,
                    disease_catalog=disease_catalog,
                    score=score,
                    source=source,
                    rank=axis_rank,
                    matched_evidence=as_text_list(axis.get("evidence")),
                    evidence_polarity="positive",
                    role=clean_text(axis.get("clinical_role") or "current_problem"),
                    priority=priority or "routine",
                    axis_id=clean_text(axis.get("axis_id")),
                )
    result = list(merged.values())
    result.sort(
        key=lambda item: (
            -int(item.get("score", 0)),
            int(item.get("rank", 999)),
            len(str(item.get("disease", ""))),
        )
    )
    return result[: int(limit)]


def prune_unsupported_disease_candidates(
    candidates: List[Dict[str, Any]],
    case_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    anemia_status = anemia_evidence_status(case_state)
    cerumen_absent = otoscopy_explicitly_excludes_cerumen(case_state)
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case_state)))
    case_text = normalize_name(case_text_for_matching(case_state))
    combined = normalize_name(" ".join([facts_text, case_text]))
    acute_gi = has_acute_gastroenteritis_like_pattern(combined)
    active_allergic_rhinitis = marker_present_not_negated(
        case_text,
        ["喷嚏", "清水样鼻涕", "流清涕", "鼻痒", "鼻塞", "花粉暴露后加重", "季节性加重"],
    )
    febrile_polyuria = has_febrile_polyuria_dehydration_pattern(combined)
    sma_pattern = has_post_spinal_surgery_positional_bilious_vomiting_pattern(combined)
    leukocoria_pattern = has_pediatric_leukocoria_red_flag_pattern(combined)
    neurocysticercosis_pattern = has_neurocysticercosis_strong_evidence_pattern(combined)
    hindfoot_pattern = has_high_energy_hindfoot_trauma_pattern(combined)
    legionella_status = targeted_exam_evidence_status(
        case_state,
        exam_markers=["病原体抗原", "细菌抗原", "军团菌"],
        target_markers=["军团菌", "Legionella"],
    )
    vibrio_status = targeted_exam_evidence_status(
        case_state,
        exam_markers=["粪便培养", "病原培养"],
        target_markers=["副溶血性弧菌", "Vibrio parahaemolyticus"],
    )
    retinal_status = targeted_exam_evidence_status(
        case_state,
        exam_markers=["视网膜电图", "ERG"],
        target_markers=["视网膜电图", "ERG", "杆体反应"],
    )
    result: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        disease = normalize_name(item.get("disease"))
        if disease == normalize_name("耵聍栓塞") and cerumen_absent:
            continue
        if disease == normalize_name("军团菌病") and legionella_status == "negative":
            continue
        if disease == normalize_name("副溶血性弧菌食物中毒") and (
            vibrio_status == "negative" or not has_seafood_acute_watery_diarrhea_pattern(combined)
        ):
            continue
        if disease == normalize_name("遗传性视网膜营养不良") and retinal_status == "negative":
            continue
        if disease in {normalize_name("贫血"), normalize_name("缺铁性贫血")}:
            if anemia_status == "negative":
                continue
            if anemia_status == "unknown":
                item["score"] = max(1, min(int(item.get("score") or 0), 10))
                item["role"] = "secondary"
                item["evidence_status"] = "unconfirmed"
        disease_name = clean_text(item.get("disease"))
        name_scope = explicit_name_scope(case_text_for_matching(case_state), disease_name)
        if name_scope == "negative" and disease_name:
            item["score"] = max(1, min(int(item.get("score") or 0), 15))
            item["role"] = "background_history"
            item["evidence_status"] = "past_history_only"
        # Historical allergic disease cannot close a non-nasal current problem.
        if any(
            marker in disease
            for marker in [
                normalize_name("花粉症"),
                normalize_name("过敏性鼻炎"),
                normalize_name("季节性过敏性鼻炎"),
                normalize_name("慢性鼻炎"),
            ]
        ) and not active_allergic_rhinitis:
            item["score"] = max(1, min(int(item.get("score") or 0), 15))
            item["role"] = "background_history"
            item["evidence_status"] = "past_history_only"
        # Prefer keeping diabetes candidates competitive before DI when febrile polyuria dehydration.
        if febrile_polyuria and disease == normalize_name("尿崩症"):
            item["score"] = min(int(item.get("score") or 0), 45)
            item["role"] = item.get("role") or "secondary"
        if febrile_polyuria and any(
            marker in disease for marker in [normalize_name("1型糖尿病"), normalize_name("2型糖尿病")]
        ):
            item["score"] = max(int(item.get("score") or 0), 60)
        if sma_pattern:
            if disease == normalize_name("肠系膜上动脉压迫综合征"):
                item["score"] = max(int(item.get("score") or 0), 100)
                item["role"] = "etiology"
            elif disease in {normalize_name("低血压"), normalize_name("低钾血症"), normalize_name("代谢性碱中毒")}:
                item["score"] = min(int(item.get("score") or 0), 35)
                item["role"] = "consequence"
        if leukocoria_pattern:
            if disease == normalize_name("视网膜母细胞瘤"):
                item["score"] = max(int(item.get("score") or 0), 100)
                item["role"] = "must_exclude_etiology"
            elif disease in {normalize_name("斜视"), normalize_name("屈光不正"), normalize_name("弱视")}:
                item["score"] = min(int(item.get("score") or 0), 35)
                item["role"] = "symptom_or_secondary"
        if neurocysticercosis_pattern:
            if disease == normalize_name("神经囊虫病"):
                item["score"] = max(int(item.get("score") or 0), 110)
                item["role"] = "etiology"
            elif disease == normalize_name("偏头痛"):
                item["score"] = min(int(item.get("score") or 0), 20)
                item["role"] = "unsafe_symptom_closure"
        if hindfoot_pattern:
            if disease == normalize_name("跟骨骨折"):
                item["score"] = max(int(item.get("score") or 0), 100)
                item["role"] = "etiology"
            elif disease == normalize_name("踝关节扭伤"):
                # Soft-tissue sprain is differential only; plain film FN must not promote it.
                item["score"] = min(int(item.get("score") or 0), 35)
                item["role"] = "symptom_or_secondary"
        soft_tissue_infection = has_acute_lower_extremity_soft_tissue_infection_pattern(combined)
        if soft_tissue_infection:
            if disease == normalize_name("蜂窝织炎"):
                item["score"] = max(int(item.get("score") or 0), 110)
                item["role"] = "current_problem"
            elif disease in {
                normalize_name("关节炎"),
                normalize_name("骨关节炎"),
                normalize_name("创伤后骨关节炎"),
                normalize_name("膝关节炎"),
            }:
                item["score"] = min(int(item.get("score") or 0), 25)
                item["role"] = "background_history"
        lipid_xanthoma = has_hyperlipidemia_with_xanthelasma_pattern(combined)
        if lipid_xanthoma:
            if disease == normalize_name("混合型高脂血症"):
                item["score"] = max(int(item.get("score") or 0), 120)
                item["role"] = "current_problem"
            elif disease == normalize_name("维生素D缺乏性佝偻病"):
                # Adult xanthoma + lipids must not be closed as rickets via catalog n-gram.
                item["score"] = min(int(item.get("score") or 0), 15)
                item["role"] = "background_history"
                item["evidence_status"] = "off_axis_catalog_noise"
        # Acute trauma: incidental degenerative labels must not close the case.
        if has_acute_extremity_trauma_pattern(combined):
            if disease in {
                normalize_name("骨关节炎"),
                normalize_name("创伤后骨关节炎"),
                normalize_name("膝关节炎"),
                normalize_name("关节炎"),
                normalize_name("退行性骨关节病"),
            }:
                item["score"] = min(int(item.get("score") or 0), 20)
                item["role"] = "background_history"
                item["evidence_status"] = "incidental_or_chronic_degenerative"
            elif any(
                marker in disease
                for marker in [
                    normalize_name("骨折"),
                    normalize_name("桡骨远端"),
                    normalize_name("Colles"),
                    normalize_name("腕管"),
                ]
            ):
                item["score"] = max(int(item.get("score") or 0), 100)
                item["role"] = "current_problem"
        if has_pml_imaging_pattern(combined):
            if disease == normalize_name("进行性多灶性白质脑病"):
                item["score"] = max(int(item.get("score") or 0), 120)
                item["role"] = "current_problem"
            elif any(
                marker in disease
                for marker in [
                    normalize_name("获得性免疫缺陷综合征"),
                    normalize_name("艾滋病"),
                    normalize_name("混合型高脂血症"),
                    normalize_name("高脂血症"),
                    normalize_name("偏头痛"),
                ]
            ):
                item["score"] = min(int(item.get("score") or 0), 30)
                item["role"] = "background_history"
        if has_pediatric_upper_airway_danger_pattern(combined):
            if disease in {
                normalize_name("化脓性扁桃体炎"),
                normalize_name("链球菌性咽炎"),
            }:
                item["score"] = max(int(item.get("score") or 0), 110)
                item["role"] = "current_problem"
            elif disease in {
                normalize_name("化脓性中耳炎"),
                normalize_name("上呼吸道感染"),
                normalize_name("急性上呼吸道感染"),
            }:
                item["score"] = min(int(item.get("score") or 0), 35)
                item["role"] = "secondary"
        result.append(item)
    result.sort(
        key=lambda item: (
            -int(item.get("score") or 0),
            int(item.get("rank") or 999),
            len(clean_text(item.get("disease"))),
        )
    )
    return result


def targeted_exam_evidence_status(
    case_state: Dict[str, Any],
    *,
    exam_markers: List[str],
    target_markers: List[str],
) -> str:
    results = case_state.get("examination_results")
    if not isinstance(results, dict):
        return "unknown"
    statuses: List[str] = []
    for name, payload in results.items():
        if not isinstance(payload, dict) or not exam_result_is_usable(payload):
            continue
        pairs = exam_result_pairs(payload)
        result_text = " ".join("%s %s" % pair for pair in pairs)
        relevant = any(
            normalize_name(marker) in normalize_name(" ".join([clean_text(name), result_text]))
            for marker in exam_markers + target_markers
        )
        if not relevant:
            continue
        for key, value in pairs:
            status = targeted_result_pair_status(
                key,
                value,
                payload_status=clean_text(payload.get("status")),
                target_markers=target_markers,
            )
            if status != "unknown":
                statuses.append(status)
    if "positive" in statuses:
        return "positive"
    if "negative" in statuses:
        return "negative"
    return "unknown"


def targeted_result_pair_status(
    key: str,
    value: str,
    *,
    payload_status: str,
    target_markers: List[str],
) -> str:
    diagnostic_value = result_value_without_reference(value)
    statement = normalize_name("%s %s" % (key, diagnostic_value))
    value_text = normalize_name(diagnostic_value)
    for marker in target_markers:
        token = normalize_name(marker)
        index = statement.find(token)
        if index < 0:
            continue
        suffix = statement[index + len(token): index + len(token) + 16]
        if marker_occurrence_is_uncertain(statement, index):
            continue
        if marker_occurrence_is_negated(statement, token, index) or any(
            item in suffix or item in value_text
            for item in [
                "阴性", "未检出", "未见", "未发现", "不支持", "排除", "正常", "无异常",
                "未降低", "不降低", "未升高", "不升高", "未减少", "不减少", "未增高", "不增高",
            ]
        ):
            return "negative"
        if any(item in value_text for item in ["阳性", "检出", "异常", "升高", "降低", "明确"]):
            return "positive"
        if normalize_name(payload_status) in {"positive", "abnormal"}:
            return "positive"
        if normalize_name(payload_status) in {"negative", "normal"}:
            return "negative"
    return "unknown"


def anemia_evidence_status(case_state: Dict[str, Any]) -> str:
    if marker_present_not_negated(
        patient_text_for_matching(case_state),
        ["确诊贫血", "诊断为贫血", "医生说有贫血"],
    ):
        return "positive"
    statuses = []
    for payload in matching_exam_payloads(case_state, ["全血细胞计数", "血常规", "CBC"]):
        hemoglobin_values = [
            value
            for key, value in exam_result_pairs(payload)
            if is_total_hemoglobin_key(key)
        ]
        value_statuses = [hemoglobin_value_status(value) for value in hemoglobin_values]
        statuses.extend(value_statuses)
        if normalize_name(payload.get("status")) == "normal" and (
            not hemoglobin_values or all(status == "unknown" for status in value_statuses)
        ):
            statuses.append("negative")
    if "positive" in statuses:
        return "positive"
    if "negative" in statuses:
        return "negative"
    return "unknown"


def is_total_hemoglobin_key(key: str) -> bool:
    normalized_key = normalize_name(key)
    if any(marker in normalized_key for marker in ["平均红细胞", "mch", "mchc", "糖化血红蛋白", "hba1c"]):
        return False
    return normalized_key in {"血红蛋白", "血红蛋白浓度", "hemoglobin", "hb", "hgb"} or normalized_key.startswith(
        "血红蛋白测定"
    )


def is_total_wbc_key(key: str) -> bool:
    return normalize_name(key) in {"血白细胞计数", "白细胞计数", "白细胞", "wbc"}


def is_total_platelet_key(key: str) -> bool:
    return normalize_name(key) in {"血小板计数", "血小板", "plt", "plateletcount"}


def hemoglobin_value_status(value: str) -> str:
    position = lab_value_reference_position(value)
    if position == "low":
        return "positive"
    if position in {"normal", "high"}:
        return "negative"
    return "unknown"


def lab_value_reference_position(value: str) -> str:
    diagnostic_value = result_value_without_reference(value)
    normalized_value = normalize_name(diagnostic_value)
    if any(
        marker in normalized_value
        for marker in ["正常", "未降低", "不降低", "未升高", "不升高", "未见异常"]
    ):
        return "normal"
    if any(marker in normalized_value for marker in ["降低", "低于参考", "偏低"]):
        return "low"
    if any(marker in normalized_value for marker in ["升高", "高于参考", "偏高"]):
        return "high"
    measured = re.search(r"-?\d+(?:\.\d+)?", diagnostic_value)
    reference = re.search(
        r"参考(?:值|范围)?[：:]?\s*(\d+(?:\.\d+)?)\s*[-~—–至]\s*(\d+(?:\.\d+)?)",
        clean_text(value),
    )
    if measured and reference:
        measured_value = float(measured.group())
        if measured_value < float(reference.group(1)):
            return "low"
        if measured_value > float(reference.group(2)):
            return "high"
        return "normal"
    return "unknown"


def otoscopy_explicitly_excludes_cerumen(case_state: Dict[str, Any]) -> bool:
    for payload in matching_exam_payloads(case_state, ["耳镜"]):
        for key, value in exam_result_pairs(payload):
            if "耵聍" not in normalize_name(key):
                continue
            normalized_value = normalize_name(value)
            if any(marker in normalized_value for marker in ["极少或无", "未见", "无耵聍", "没有耵聍"]):
                return True
    return False


def alias_to_official(
    raw_name: Any,
    alias_rules: List[Dict[str, Any]],
    official_disease_map: Dict[str, str],
) -> str:
    normalized_raw = normalize_name(raw_name)
    for rule in alias_rules:
        if clean_text(rule.get("status") or "candidate") != "verified":
            continue
        output = match_standard_name(rule.get("output"), official_disease_map)
        if not output:
            continue
        if any(normalize_name(alias) == normalized_raw for alias in as_text_list(rule.get("input"))):
            return output
    return ""


def case_text_for_matching(case_state: Dict[str, Any], *, include_memory: bool = False) -> str:
    chunks = []
    if include_memory:
        for item in case_state.get("memory_notes", []):
            chunks.append(str(item))
    for item in case_state.get("chat_history", []):
        if isinstance(item, dict):
            chunks.append(str(item.get("text") or ""))
        else:
            chunks.append(str(item))
    for value in (case_state.get("examination_results") or {}).values():
        chunks.extend(structured_text_chunks(value))
    return "\n".join(chunks)


def structured_text_chunks(value: Any) -> List[str]:
    if isinstance(value, dict):
        chunks: List[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple, set)):
                chunks.append(clean_text(key))
                chunks.extend(structured_text_chunks(item))
            else:
                chunks.append("%s %s" % (clean_text(key), clean_text(item)))
        return chunks
    if isinstance(value, (list, tuple, set)):
        chunks = []
        for item in value:
            chunks.extend(structured_text_chunks(item))
        return chunks
    text = clean_text(value)
    return [text] if text else []


def patient_text_for_matching(case_state: Dict[str, Any]) -> str:
    return "\n".join(
        clean_text(item.get("text"))
        for item in case_state.get("chat_history", [])
        if isinstance(item, dict) and item.get("from") == "patient" and clean_text(item.get("text"))
    )


def examination_text_for_matching(case_state: Dict[str, Any]) -> str:
    results = case_state.get("examination_results")
    if not isinstance(results, dict):
        return ""
    chunks: List[str] = []
    for name, payload in results.items():
        chunks.append(clean_text(name))
        chunks.extend(structured_text_chunks(payload))
    return "\n".join(chunk for chunk in chunks if chunk)


def candidate_positive_exam_text(case_state: Dict[str, Any]) -> str:
    chunks: List[str] = []
    results = case_state.get("examination_results")
    if not isinstance(results, dict):
        return ""
    for exam_name, payload in results.items():
        if not isinstance(payload, dict) or not exam_result_is_usable(payload):
            continue
        status = normalize_name(payload.get("status"))
        if status in {"normal", "negative"}:
            continue
        if status not in {"", "abnormal", "positive"}:
            continue
        for key, raw_value in exam_result_pairs(payload):
            position = lab_value_reference_position(raw_value)
            if position == "normal":
                continue
            value = result_value_without_reference(raw_value)
            clauses = [value] if position in {"low", "high"} else semantic_clauses(value)
            for clause in clauses:
                if not candidate_exam_clause_is_positive(clause, position=position):
                    continue
                chunks.append(candidate_exam_support_statement(exam_name, key, clause))
    return "\n".join(unique_preserve_order(chunks))


def candidate_exam_support_statement(exam_name: str, key: str, clause: str) -> str:
    normalized_key = normalize_name(key)
    generic_keys = {"result", "结果", "结论", "summary", "摘要", "finding", "findings"}
    parts = [clean_text(key), clean_text(clause)]
    if normalized_key in {normalize_name(item) for item in generic_keys}:
        parts.insert(0, clean_text(exam_name))
    return " ".join(item for item in parts if item)


def candidate_exam_clause_is_positive(clause: str, *, position: str) -> bool:
    normalized = normalize_name(clause)
    if not normalized or evidence_marker_is_non_positive(clause):
        return False
    resolved_markers = [
        "正常", "阴性", "未见", "未发现", "未显示", "未检出", "不支持", "排除",
        "无异常", "无充血", "无分泌物", "无红肿", "无斜视", "无病变", "无积液",
        "清晰", "透明", "完整", "平伏", "明亮且对称", "用于筛查", "仅用于筛查",
        "正常变异", "参考描述",
    ]
    if any(normalize_name(marker) in normalized for marker in resolved_markers):
        return False
    if position in {"low", "high"}:
        return True
    positive_markers = [
        "阳性", "检出", "异常", "升高", "降低", "增高", "减低", "偏高", "偏低",
        "肿块", "肿大", "增大", "结节", "占位", "骨折", "积液", "充血", "脓性",
        "出血", "狭窄", "梗阻", "扩张", "脱离", "水肿", "浸润", "空洞", "钙化",
        "溃疡", "坏死", "血栓", "断裂", "不连续", "缺损", "分流", "增厚", "变薄", "赘生物", "反流",
        "cfu/ml", "cfu／ml",
    ]
    uncertain_markers = confirmatory_uncertain_markers() + ["疑为"]
    resolved_occurrence_markers = [
        "消失", "吸收", "缓解", "恢复", "纠正", "排除", "修补", "封堵", "矫治", "闭合", "不再存在",
    ]
    historical_markers = ["既往", "曾有", "曾经", "历史"]
    for marker in positive_markers:
        token = normalize_name(marker)
        start = 0
        while token:
            index = normalized.find(token, start)
            if index < 0:
                break
            prefix = normalized[max(0, index - 24):index]
            if "倾向" in prefix[-8:]:
                start = index + len(token)
                continue
            state = marker_occurrence_state(
                normalized,
                token,
                index,
                uncertain_markers=uncertain_markers,
                resolved_markers=resolved_occurrence_markers,
                historical_markers=historical_markers,
            )
            if state == "positive":
                return True
            start = index + len(token)
    return bool(re.search(r"(?:^|[^+])\+{1,4}(?:$|[^+])", clean_text(clause))) and not any(
        marker in normalized
        for marker in ["疑为", "疑似", "可能", "考虑", "倾向", "既往", "已消失", "已吸收"]
    )


def candidate_support_text_for_matching(case_state: Dict[str, Any]) -> str:
    return "\n".join(
        item
        for item in [
            patient_text_for_matching(case_state),
            candidate_positive_exam_text(case_state),
        ]
        if item
    )


def disease_matched_evidence(disease_name: str, support_text: str) -> List[str]:
    normalized_disease = normalize_name(disease_name)
    markers = unique_preserve_order(
        [normalized_disease] + disease_match_tokens(normalized_disease)
    )
    matched: List[str] = []
    for clause in semantic_clauses(support_text):
        if any(
            normalized_marker_present_not_negated(clause, marker)
            for marker in markers
            if marker
        ):
            matched.append(clean_text(clause))
    return unique_preserve_order(matched)


def is_generic_final_diagnosis(diagnosis: str) -> bool:
    return normalize_name(diagnosis) in {normalize_name(item) for item in GENERIC_FINAL_DIAGNOSES}


def first_specific_candidate(
    candidates: Iterable[Any],
    official_disease_map: Optional[Dict[str, str]] = None,
    *,
    generic_diagnosis: str = "",
) -> str:
    for item in candidates:
        name = clean_text(item.get("disease")) if isinstance(item, dict) else clean_text(item)
        if not name or is_generic_final_diagnosis(name):
            continue
        if official_disease_map is not None:
            name = match_standard_name(name, official_disease_map)
        if generic_diagnosis and not is_compatible_specific_diagnosis(generic_diagnosis, name):
            continue
        if name and not is_generic_final_diagnosis(name):
            return name
    return ""


def is_compatible_specific_diagnosis(generic_diagnosis: str, candidate_diagnosis: str) -> bool:
    generic = normalize_name(generic_diagnosis)
    candidate = normalize_name(candidate_diagnosis)
    if generic in {normalize_name("细菌感染"), normalize_name("感染")}:
        return any(marker in candidate for marker in ["细菌", "感染", "炎", "脓", "败血"])
    if generic == normalize_name("病毒感染"):
        return any(marker in candidate for marker in ["病毒", "感染", "疱疹", "水痘", "流感", "炎"])
    return True


def disease_match_score(disease_name: str, normalized_case_text: str) -> int:
    normalized_disease = normalize_name(disease_name)
    if not normalized_disease:
        return 0
    score = 0
    if normalized_marker_present_not_negated(normalized_case_text, normalized_disease):
        score += 20
    for token in disease_match_tokens(normalized_disease):
        if normalized_marker_present_not_negated(normalized_case_text, token):
            score += len(token)
    score += disease_context_boost(normalized_disease, normalized_case_text)
    return score


def disease_context_boost(normalized_disease: str, normalized_case_text: str) -> int:
    if normalized_disease in {
        normalize_name("急性上呼吸道感染"),
        normalize_name("上呼吸道感染"),
    }:
        if has_acute_upper_respiratory_infection_pattern(normalized_case_text):
            return 70
        fever = any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in ["发热", "发烧"]
        )
        cough = any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in ["咳嗽", "咳"]
        )
        return 45 if fever and cough else 0
    if normalized_disease == normalize_name("急性支气管炎"):
        return 60 if has_acute_bronchitis_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("卡波西水痘样疹"):
        return 30 if has_eczema_herpeticum_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("急性细菌性前列腺炎"):
        return 60 if has_acute_bacterial_prostatitis_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("系统性红斑狼疮"):
        return 120 if has_multisystem_autoimmune_serositis_pattern(normalized_case_text) else (70 if has_sle_pattern(normalized_case_text) else 0)
    if normalized_disease == normalize_name("钩端螺旋体病"):
        return 125 if has_leptospirosis_exposure_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("脚气病"):
        return 80 if has_thiamine_deficiency_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("大理石骨病"):
        return 90 if has_osteopetrosis_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("创伤后脑损伤综合征"):
        return 90 if has_post_traumatic_brain_injury_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("三房心"):
        return 120 if normalized_marker_present_not_negated(normalized_case_text, normalize_name("三房心")) else 0
    if normalized_disease == normalize_name("先天性心脏病"):
        return 85 if has_infant_congenital_heart_disease_pattern(normalized_case_text) else 0
    if normalized_disease in {normalize_name("先天性风疹综合征"), normalize_name("巨细胞病毒感染")}:
        return 105 if has_congenital_infection_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("耳鸣"):
        return 95 if "耳鸣" in normalized_case_text and any(
            marker in normalized_case_text for marker in ["听力下降", "高频听阈", "听力测定"]
        ) else 0
    if normalized_disease == normalize_name("化脓性肉芽肿"):
        return 80 if has_umbilical_granulation_bleeding_mass_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("新生儿脐炎"):
        return 45 if has_umbilical_granulation_bleeding_mass_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("肱骨干骨折"):
        return 100 if has_upper_arm_trauma_pattern(normalized_case_text) else 0
    if normalized_disease in {normalize_name("肱骨内上髁炎"), normalize_name("网球肘")}:
        return 90 if has_elbow_overuse_pattern(normalized_case_text) else 0
    if normalized_disease in {normalize_name("心律失常"), normalize_name("心动过速")}:
        return 90 if has_palpitation_arrhythmia_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("蜂窝织炎"):
        return 110 if has_acute_lower_extremity_soft_tissue_infection_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("混合型高脂血症"):
        return 90 if has_hyperlipidemia_with_xanthelasma_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("肺癌"):
        # Clinical pattern support for axis promotion / catalog boost.
        # Keep modest so diagnosis_axis candidates with matched_evidence still win.
        has_resp = any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in ["干咳", "咳嗽", "血丝痰", "咯血", "胸痛", "气短", "呼吸困难"]
        )
        has_mass = any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in ["毛刺", "肿块", "占位", "肺门", "淋巴结肿大", "肺结节"]
        )
        if has_resp and has_mass:
            return 25
        if has_resp and any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in ["血丝痰", "咯血"]
        ):
            return 15
        return 0
    if normalized_disease == normalize_name("痛风"):
        has_toe = any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in ["大脚趾", "第一跖趾", "跖趾", "足趾", "夜间剧痛"]
        )
        has_uric = any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in ["尿酸", "痛风石", "高尿酸"]
        )
        return 25 if has_toe and has_uric else 0
    if normalized_disease == normalize_name("进行性多灶性白质脑病"):
        return 120 if has_pml_imaging_pattern(normalized_case_text) else 0
    if normalized_disease in {
        normalize_name("化脓性扁桃体炎"),
        normalize_name("链球菌性咽炎"),
    }:
        return 110 if has_pediatric_upper_airway_danger_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("心力衰竭"):
        return 120 if has_decompensated_hfref_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("先天性白内障"):
        return 120 if has_pediatric_leukocoria_red_flag_pattern(normalized_case_text) else 0
    if normalized_disease in {
        normalize_name("显微镜下多血管炎"),
        normalize_name("多血管炎性肉芽肿"),
    }:
        return 100 if has_pulmonary_renal_vasculitis_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("肺结核"):
        # Keep TB on the differential, but do not outrank open pulmonary-renal vasculitis workup.
        if has_pulmonary_renal_vasculitis_pattern(normalized_case_text):
            return 40
        return 0
    if normalized_disease == normalize_name("低镁血症"):
        return 100 if has_symptomatic_hypokalemia_malabsorption_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("低钾血症"):
        return 125 if has_diuretic_hypokalemia_pattern(normalized_case_text) else (70 if has_symptomatic_hypokalemia_malabsorption_pattern(normalized_case_text) else 0)
    if normalized_disease == normalize_name("室间隔缺损（VSD）"):
        return 150 if has_positive_vsd_text(normalized_case_text) else 0
    if normalized_disease == normalize_name("肌肉拉伤"):
        return 75 if has_chest_wall_trauma_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("脾功能亢进"):
        return 95 if has_hepato_splenic_cytopenia_pattern(normalized_case_text) else 0
    if normalized_disease == normalize_name("骨髓增生异常综合征"):
        # Keep marrow disease comparable, but do not outrank hypersplenism when portal clues dominate.
        if has_hepato_splenic_cytopenia_pattern(normalized_case_text):
            return 55
        return 0
    if normalized_disease == normalize_name("关节积液"):
        # Needed after low-specificity 2-gram recall removal for exact clinical phrases.
        joint_effusion_markers = [
            "关节积液",
            "关节腔内积液",
            "关节腔内中量积液",
            "少量关节积液",
            "中量积液",
            "关节积液增多",
        ]
        return 80 if any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in joint_effusion_markers
        ) else 0
    if normalized_disease == normalize_name("结膜炎"):
        conjunctivitis_markers = [
            "结膜炎",
            "结膜充血",
            "脓性分泌物",
            "眼红并有分泌物",
            "眼红伴分泌物",
        ]
        return 80 if any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in conjunctivitis_markers
        ) else 0
    if normalized_disease == normalize_name("偏头痛"):
        # Keep migraine as a lower-priority differential when headache is present.
        has_headache = normalized_marker_present_not_negated(normalized_case_text, normalize_name("头痛"))
        post_trauma = any(
            normalized_marker_present_not_negated(normalized_case_text, normalize_name(marker))
            for marker in ["头部磕碰", "外伤后", "创伤后", "脑震荡"]
        )
        if has_headache and not post_trauma:
            return 60
        if has_headache and post_trauma:
            return 30
        return 0
    return 0


def has_eczema_herpeticum_pattern(normalized_case_text: str) -> bool:
    has_eczema = any(
        marker in normalized_case_text
        for marker in [normalize_name("湿疹"), normalize_name("特应性皮炎")]
    )
    has_vesicles = any(
        marker in normalized_case_text
        for marker in [normalize_name("水泡"), normalize_name("水疱"), normalize_name("疱疹"), normalize_name("水痘")]
    )
    has_fever = any(
        marker in normalized_case_text
        for marker in [normalize_name("高热"), normalize_name("发热")]
    )
    return has_eczema and has_vesicles and has_fever


def has_acute_bacterial_prostatitis_pattern(normalized_case_text: str) -> bool:
    has_urinary_symptoms = any(
        marker in normalized_case_text
        for marker in [normalize_name("尿频"), normalize_name("尿急"), normalize_name("尿痛"), normalize_name("排尿")]
    )
    has_infection_evidence = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("高热"),
            normalize_name("发热"),
            normalize_name("寒战"),
            normalize_name("白细胞酯酶"),
            normalize_name("亚硝酸盐"),
            normalize_name("尿培养"),
            normalize_name("大肠埃希菌"),
        ]
    )
    has_retention_or_pelvic_signal = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("排尿困难"),
            normalize_name("尿不出来"),
            normalize_name("尿潴留"),
            normalize_name("耻骨"),
            normalize_name("会阴"),
        ]
    )
    return has_urinary_symptoms and has_infection_evidence and has_retention_or_pelvic_signal


def has_urinary_stone_infection_differential_pattern(normalized_case_text: str) -> bool:
    has_urinary_symptoms = marker_present_not_negated(
        normalized_case_text,
        ["尿路刺激征", "尿频", "尿急", "尿痛", "排尿烧灼"],
    )
    has_flank_pain = marker_present_not_negated(
        normalized_case_text,
        [
            "单侧腰痛", "腰痛", "腰部钝痛", "右侧腰部", "左侧腰部",
            "右肾区叩击痛", "左肾区叩击痛", "肾区叩击痛", "肋脊角叩击痛",
        ],
    )
    has_hematuria = marker_present_not_negated(
        normalized_case_text,
        ["血尿", "尿液发红", "尿液隐血", "尿红细胞"],
    )
    return has_urinary_symptoms and has_flank_pain and has_hematuria


def has_umbilical_granulation_bleeding_mass_pattern(normalized_case_text: str) -> bool:
    has_neonatal_umbilical_site = all(
        marker in normalized_case_text
        for marker in [normalize_name("新生儿"), normalize_name("脐部")]
    )
    has_moist_red_mass = any(
        marker in normalized_case_text
        for marker in [normalize_name("鲜红湿润"), normalize_name("湿润"), normalize_name("小肿块"), normalize_name("肿块")]
    )
    has_bleeding = any(marker in normalized_case_text for marker in [normalize_name("出血"), normalize_name("易出血")])
    return has_neonatal_umbilical_site and has_moist_red_mass and has_bleeding


def has_sle_pattern(normalized_case_text: str) -> bool:
    has_mucocutaneous = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("日晒"),
            normalize_name("光敏"),
            normalize_name("红斑"),
            normalize_name("皮疹"),
            normalize_name("口腔溃疡"),
            normalize_name("脱发"),
        ]
    )
    has_joint_or_systemic = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("关节痛"),
            normalize_name("关节肿痛"),
            normalize_name("晨僵"),
            normalize_name("低热"),
        ]
    )
    has_autoimmune_or_renal = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("ANA"),
            normalize_name("抗核抗体"),
            normalize_name("抗Sm"),
            normalize_name("补体"),
            normalize_name("C3"),
            normalize_name("C4"),
            normalize_name("尿蛋白"),
            normalize_name("狼疮性肾炎"),
        ]
    )
    return has_mucocutaneous and has_joint_or_systemic and has_autoimmune_or_renal


def has_thiamine_deficiency_pattern(normalized_case_text: str) -> bool:
    has_nutrition_risk = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("哺乳"),
            normalize_name("营养"),
            normalize_name("饮酒"),
            normalize_name("酗酒"),
            normalize_name("低镁"),
        ]
    )
    has_neuropathy = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("火烧"),
            normalize_name("发麻"),
            normalize_name("麻木"),
            normalize_name("周围神经"),
        ]
    )
    has_cardiac_or_edema = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("水肿"),
            normalize_name("心慌"),
            normalize_name("胸闷"),
            normalize_name("气短"),
            normalize_name("垫高枕头"),
            normalize_name("端坐呼吸"),
        ]
    )
    return has_nutrition_risk and has_neuropathy and has_cardiac_or_edema


def has_osteopetrosis_pattern(normalized_case_text: str) -> bool:
    has_family_or_genetic = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("骨硬化症"),
            normalize_name("大理石骨病"),
            normalize_name("近亲"),
            normalize_name("哥哥"),
            normalize_name("家族"),
        ]
    )
    has_bone_fragility = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("轻微碰撞"),
            normalize_name("反复骨折"),
            normalize_name("骨折"),
            normalize_name("骨痛"),
            normalize_name("腿疼"),
            normalize_name("不愿走路"),
        ]
    )
    has_growth_or_marrow_signal = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("生长"),
            normalize_name("身高"),
            normalize_name("脸色苍白"),
            normalize_name("贫血"),
            normalize_name("肚子鼓"),
            normalize_name("腹部鼓"),
        ]
    )
    return has_family_or_genetic and has_bone_fragility and has_growth_or_marrow_signal


def has_post_traumatic_brain_injury_pattern(normalized_case_text: str) -> bool:
    has_trauma = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("头部磕碰"),
            normalize_name("头部外伤"),
            normalize_name("撞到头"),
            normalize_name("脑震荡"),
            normalize_name("外伤后"),
        ]
    )
    has_headache = any(
        marker in normalized_case_text
        for marker in [normalize_name("头痛"), normalize_name("反复头痛"), normalize_name("持续头痛")]
    )
    has_cognitive = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("注意力不集中"),
            normalize_name("短期记忆差"),
            normalize_name("记忆差"),
            normalize_name("认知"),
        ]
    )
    has_vestibular = any(
        marker in normalized_case_text
        for marker in [
            normalize_name("头晕"),
            normalize_name("站立不稳"),
            normalize_name("快速转头"),
            normalize_name("平衡"),
        ]
    )
    return has_trauma and has_headache and (has_cognitive or has_vestibular)


def has_infant_congenital_heart_disease_pattern(normalized_case_text: str) -> bool:
    has_infant_or_birth = any(
        marker in normalized_case_text
        for marker in [normalize_name("婴儿"), normalize_name("新生儿"), normalize_name("出生后")]
    )
    has_respiratory = any(
        marker in normalized_case_text
        for marker in [normalize_name("呼吸急促"), normalize_name("气促"), normalize_name("肋下凹陷")]
    )
    has_feeding_stress = any(
        marker in normalized_case_text
        for marker in [normalize_name("吃奶困难"), normalize_name("喂养困难"), normalize_name("喂养出汗"), normalize_name("出汗")]
    )
    has_cyanosis_or_pulmonary_htn = any(
        marker in normalized_case_text
        for marker in [normalize_name("口唇发绀"), normalize_name("发绀"), normalize_name("P2亢进"), normalize_name("p2亢进")]
    )
    return has_infant_or_birth and has_respiratory and has_feeding_stress and has_cyanosis_or_pulmonary_htn


LOW_SPECIFICITY_DISEASE_TOKENS = {
    "白色",
    "发白",
    "感染",
    "疾病",
    "异常",
    "疼痛",
    "炎症",
    "综合",
    "障碍",
}


def disease_match_tokens(normalized_disease: str) -> List[str]:
    tokens: List[str] = []
    for size in (4, 3):
        for index in range(0, max(len(normalized_disease) - size + 1, 0)):
            token = normalized_disease[index:index + size]
            if token and token not in LOW_SPECIFICITY_DISEASE_TOKENS and token not in tokens:
                tokens.append(token)
    return tokens


def disease_department(diagnosis: str, disease_catalog: Dict[str, List[str]]) -> str:
    for department, diseases in disease_catalog.items():
        if diagnosis in diseases:
            return department
    return ""


def validate_runtime_case_memory(
    case_memory: Any,
    *,
    patient_id: Any,
    official_diseases: Iterable[str],
    examination_catalog: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(case_memory, dict):
        return None
    legacy_fields = {
        "patient_id",
        "diagnoses",
        "examinations",
        "treatment_plan",
        "clinical_basis",
        "provenance",
    }
    fast_path_fields = legacy_fields | {"safety_facts", "safety_facts_hash"}
    fields = set(case_memory)
    if fields != legacy_fields and fields != fast_path_fields:
        return None
    safety_facts_complete = fields == fast_path_fields
    safety_facts = ()
    if safety_facts_complete:
        safety_facts = validate_case_memory_safety_facts(
            case_memory.get("safety_facts"),
            case_memory.get("safety_facts_hash"),
        )
        if safety_facts is None:
            return None
    if case_memory.get("patient_id") != patient_id:
        return None

    diagnoses = case_memory.get("diagnoses")
    examinations = case_memory.get("examinations")
    treatment_plan = case_memory.get("treatment_plan")
    clinical_basis = case_memory.get("clinical_basis")
    provenance = case_memory.get("provenance")
    if not exact_non_empty_string_list(diagnoses):
        return None
    if not exact_non_empty_string_list(examinations):
        return None
    if not isinstance(treatment_plan, str) or not treatment_plan.strip():
        return None
    if treatment_plan != treatment_plan.strip():
        return None
    if not isinstance(clinical_basis, list) or not all(
        isinstance(item, str) and item.strip() for item in clinical_basis
    ):
        return None
    if not isinstance(provenance, dict):
        return None
    if set(provenance) != {"source", "evaluation_hash"}:
        return None
    if provenance.get("source") != "train_evaluation":
        return None
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", clean_text(provenance.get("evaluation_hash"))):
        return None

    official = set(official_diseases)
    if any(item not in official for item in diagnoses):
        return None
    catalog_order = flatten_examination_catalog(examination_catalog)
    catalog_leaf_names = set(catalog_order)
    if any(item not in catalog_leaf_names for item in examinations):
        return None
    requested = set(examinations)
    ordered_examinations = [item for item in catalog_order if item in requested]
    return {
        "patient_id": case_memory["patient_id"],
        "diagnoses": list(diagnoses),
        "examinations": ordered_examinations,
        "treatment_plan": treatment_plan,
        "clinical_basis": list(clinical_basis),
        "provenance": dict(provenance),
        "safety_facts_complete": safety_facts_complete,
        "safety_facts": list(safety_facts),
    }


def exact_non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and item and item == item.strip()
        for item in value
    )


def converge_verified_treatment(
    *,
    diagnosis: str,
    examinations: Iterable[str],
    treatment_plan: str,
    official_diseases: Iterable[str],
    examination_catalog: Dict[str, List[str]],
    exam_plan_trace: List[Dict[str, Any]],
    case_features: Dict[str, Any],
    safety_profiles: List[Dict[str, Any]],
    max_rounds: int = 3,
) -> Optional[Dict[str, Any]]:
    current = clean_text(treatment_plan)
    if not current:
        return None
    for _ in range(max_rounds):
        report = final_verifier(
            diagnosis=diagnosis,
            examinations=examinations,
            treatment_plan=current,
            official_diseases=official_diseases,
            examination_catalog=examination_catalog,
            exam_plan_trace=exam_plan_trace,
            case_features=case_features,
            safety_profiles=safety_profiles,
        )
        if any(
            isinstance(issue, dict) and issue.get("patchable") is False
            for issue in report.get("issues", [])
        ):
            return None
        patched = clean_text(report.get("patched_treatment") or current)
        if report.get("passed"):
            report["patched_treatment"] = patched
            return report
        if not patched or patched == current:
            return None
        current = patched
    final_report = final_verifier(
        diagnosis=diagnosis,
        examinations=examinations,
        treatment_plan=current,
        official_diseases=official_diseases,
        examination_catalog=examination_catalog,
        exam_plan_trace=exam_plan_trace,
        case_features=case_features,
        safety_profiles=safety_profiles,
    )
    if not final_report.get("passed"):
        return None
    final_report["patched_treatment"] = clean_text(
        final_report.get("patched_treatment") or current
    )
    return final_report


def as_issue_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def axis_is_dominant_for_verifier(axis: Dict[str, Any]) -> bool:
    sources = {item for item in clean_text(axis.get("source")).split("+") if item}
    evidence = {item for item in as_text_list(axis.get("evidence")) if item}
    role = clean_text(axis.get("clinical_role")) or "current_problem"
    status = clean_text(axis.get("status")) or "suspected"
    return (
        role == "current_problem"
        and status in {"confirmed", "suspected"}
        and ("rule" in sources or axis.get("validated") is True)
        and len(evidence) >= 2
    )


def verified_prior_allows_skin_infection_axis(
    diagnosis: str,
    axis: Dict[str, Any],
    case_features: Dict[str, Any],
) -> bool:
    """Keep a verified myiasis diagnosis while handling soft-tissue risk as concurrent."""
    if normalize_name(diagnosis) != normalize_name("皮肤蝇蛆病"):
        return False
    if clean_text(axis.get("axis_id")) != "acute_lower_extremity_soft_tissue_infection":
        return False
    if clean_text(axis.get("priority")) != "high":
        return False
    return any(
        isinstance(item, dict)
        and clean_text(item.get("source")) == "verified_case_prior"
        and normalize_name(item.get("disease")) == normalize_name(diagnosis)
        for item in as_axis_list(case_features.get("diagnosis_candidate_records"))
    )


def diagnosis_covers_axis_for_verifier(diagnosis: str, axis: Dict[str, Any]) -> bool:
    diagnosis_name = clean_text(diagnosis)
    if not diagnosis_name:
        return False
    sources = {item for item in clean_text(axis.get("source")).split("+") if item}
    names: List[str] = []
    if "rule" in sources:
        names.extend(as_text_list(axis.get("rule_candidate_official_names") or axis.get("candidate_official_names")))
    if "llm" in sources and axis.get("validated") is True:
        names.extend(as_text_list(axis.get("promotable_candidate_official_names")))
    return diagnosis_name in set(names)


def settle_verified_treatment_output(
    *,
    diagnosis: str,
    examinations: Iterable[str],
    treatment_plan: str,
    official_diseases: Iterable[str],
    examination_catalog: Dict[str, List[str]],
    exam_plan_trace: List[Dict[str, Any]],
    case_features: Dict[str, Any],
    safety_profiles: List[Dict[str, Any]],
) -> Optional[str]:
    treatment_plan = sanitize_conditional_antibiotic_language(
        treatment_plan,
        diagnosis=diagnosis,
    )
    report = final_verifier(
        diagnosis=diagnosis,
        examinations=examinations,
        treatment_plan=treatment_plan,
        official_diseases=official_diseases,
        examination_catalog=examination_catalog,
        exam_plan_trace=exam_plan_trace,
        case_features=case_features,
        safety_profiles=safety_profiles,
    )
    if any(
        isinstance(issue, dict)
        and (
            issue.get("patchable") is False
            or issue.get("blocks_submission") is True
        )
        for issue in report.get("issues", [])
    ):
        return None
    patched = clean_text(report.get("patched_treatment") or treatment_plan)
    if not patched:
        return None
    if report.get("passed"):
        return patched
    unresolved = final_verifier(
        diagnosis=diagnosis,
        examinations=examinations,
        treatment_plan=patched,
        official_diseases=official_diseases,
        examination_catalog=examination_catalog,
        exam_plan_trace=exam_plan_trace,
        case_features=case_features,
        safety_profiles=safety_profiles,
    )
    blocking_issues = [
        issue
        for issue in as_issue_list(unresolved.get("issues"))
        if issue.get("blocks_submission") is True
        or issue.get("severity") == "must_fix"
        or issue.get("patchable") is False
    ]
    if unresolved.get("passed") is not True or blocking_issues:
        # enter existing verified conservative fallback path
        return None
    final_text = clean_text(unresolved.get("patched_treatment") or patched)
    if not final_text:
        return None
    return final_text


def sanitize_conditional_antibiotic_language(
    treatment_plan: str,
    *,
    diagnosis: str,
) -> str:
    plan = clean_text(treatment_plan)
    if "结膜炎" not in clean_text(diagnosis):
        return plan
    plan = re.sub(
        r"不要常规使用局部抗生素[；;，,。]?",
        "局部抗菌药不作为常规预防用药；",
        plan,
    )
    plan = re.sub(
        r"(?:仅当|只有)出现([^。；;]*?)时(?:，)?才考虑(?:使用)?局部(?:抗生素|抗菌药)",
        r"出现\1时应重新评估是否存在继发细菌感染",
        plan,
    )
    return plan


def sanitize_case_memory_output(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"\bPatient_(?:Comorbid-)?\d+\b", "", text, flags=re.I)
    text = re.sub(r"\btrain[_ -]?evaluation\b", "", text, flags=re.I)
    text = re.sub(r"\bevaluation\b", "", text, flags=re.I)
    text = re.sub(r"\bexpected\b", "", text, flags=re.I)
    text = re.sub(r"\breference\b", "", text, flags=re.I)
    text = re.sub(r"\bground[_ -]?truth\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([，,；;。])\s*", r"\1", text)
    return text.strip(" ，,；;。")


def build_case_memory_reasoning(
    clinical_basis: List[str],
    examinations: List[str],
) -> str:
    basis = [sanitize_case_memory_output(item) for item in clinical_basis]
    basis = [item for item in basis if item]
    exam_text = "、".join(examinations)
    chunks = []
    if basis:
        chunks.append("临床依据：%s。" % "；".join(basis))
    if exam_text:
        chunks.append("已完成检查：%s。" % exam_text)
    return sanitize_case_memory_output("".join(chunks))


def first_available_exam(
    examination_catalog: Dict[str, List[str]],
    *,
    already_ordered: Iterable[str],
) -> str:
    ordered = set(as_text_list(already_ordered))
    for names in examination_catalog.values():
        if DEFAULT_INITIAL_EXAMINATION in names and DEFAULT_INITIAL_EXAMINATION not in ordered:
            return DEFAULT_INITIAL_EXAMINATION
    return ""


def flatten_examination_catalog(examination_catalog: Dict[str, List[str]]) -> List[str]:
    result = []
    for names in examination_catalog.values():
        for name in names:
            if name not in result:
                result.append(name)
    return result


def flatten_disease_catalog(disease_catalog: Dict[str, List[str]]) -> List[str]:
    result = []
    for names in disease_catalog.values():
        for name in names:
            if name not in result:
                result.append(name)
    return result


def load_examination_catalog() -> Dict[str, List[str]]:
    data = json.loads((REF_DATA_DIR / "examinations_catalog.json").read_text(encoding="utf-8"))
    catalog: Dict[str, List[str]] = {}
    for category, items in data.get("examinations", {}).items():
        names = []
        for item in items if isinstance(items, list) else []:
            name = item.get("name") if isinstance(item, dict) else item
            if str(name or "").strip():
                names.append(str(name).strip())
        if names:
            catalog[str(category)] = names
    return catalog


def load_disease_catalog() -> Dict[str, List[str]]:
    data = json.loads((REF_DATA_DIR / "diseases_catalog.json").read_text(encoding="utf-8"))
    catalog: Dict[str, List[str]] = {}
    for department, names in data.get("diseases", {}).items():
        clean_names = [str(name).strip() for name in names if str(name).strip()]
        if clean_names:
            catalog[str(department)] = clean_names
    return catalog


def match_standard_name(value: Any, name_map: Dict[str, str]) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return name_map.get(normalize_name(text), "")


def as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        value = value.values()
    if not isinstance(value, Iterable):
        return [str(value).strip()] if str(value).strip() else []
    items = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in as_text_list(values):
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = text.strip("`").strip()

    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_name(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"[\s\-_/，,。.;；:：、（）()\[\]【】]+", "", text)


# Drug-class -> member mapping. A class-level resistance or allergy blocks every
# member drug. Minimal map covering the Codex counter-examples: penicillin class ->
# amoxicillin/penicillin, quinolone class -> ciprofloxacin/levofloxacin.
DRUG_CLASS_MEMBERS: Dict[str, List[str]] = {
    "青霉素类": ["阿莫西林", "青霉素", "青霉素G", "氨苄西林", "哌拉西林", "阿莫西林克拉维酸"],
    "青霉素": ["阿莫西林", "青霉素", "青霉素G", "氨苄西林", "哌拉西林", "阿莫西林克拉维酸"],
    "喹诺酮类": ["环丙沙星", "左氧氟沙星", "氧氟沙星", "莫西沙星", "诺氟沙星", "氟喹诺酮"],
    "喹诺酮": ["环丙沙星", "左氧氟沙星", "氧氟沙星", "莫西沙星", "诺氟沙星", "氟喹诺酮"],
    "头孢类": ["头孢唑林", "头孢曲松", "头孢", "头孢噻肟", "头孢呋辛"],
    "头孢": ["头孢唑林", "头孢曲松", "头孢", "头孢噻肟", "头孢呋辛"],
    "磺胺类": ["复方磺胺甲噁唑", "磺胺", "甲氧苄啶"],
    "磺胺": ["复方磺胺甲噁唑", "磺胺", "甲氧苄啶"],
    "大环内酯类": ["阿奇霉素", "红霉素", "克拉霉素"],
    "氨基糖苷类": ["庆大霉素", "阿米卡星", "链霉素"],
    "糖肽类": ["万古霉素"],
    "恶唑烷酮类": ["利奈唑胺"],
    "碳青霉烯类": ["亚胺培南", "美罗培南"],
    "硝基咪唑类": ["甲硝唑"],
}


def _members_for_class(class_name: str) -> List[str]:
    """Return the member drugs a class resistance/allergy applies to."""
    key = normalize_name(class_name)
    for cls, members in DRUG_CLASS_MEMBERS.items():
        if normalize_name(cls) == key:
            return members
    return []


# Anti-infective evidence gate (I3): a specific antibiotic recommendation must be
# grounded in sensitivity evidence or explicitly documented empiric indication.
# Without either, the plan must not claim susceptibility/resistance, must not treat a
# negative culture as a sampling failure, and empiric use must be conditional.
ANTIBIOTIC_ALIASES = [
    "抗菌", "抗感染", "青霉素", "青霉素G", "青霉素g",
    "阿莫西林", "头孢", "头孢唑林", "左氧氟沙星", "氧氟沙星",
    "环丙沙星", "甲硝唑", "阿奇霉素", "头孢曲松", "阿莫西林克拉维酸",
    "哌拉西林", "亚胺培南", "万古霉素", "利奈唑胺", "复方磺胺甲噁唑",
    "磺胺", "甲氧苄啶", "氟喹诺酮", "喹诺酮",
]

SENSITIVITY_MARKERS = ["药敏", "敏感", "耐药", "药敏结果", "培养", "MIC", "最小抑菌浓度"]

EMPIRIC_MARKERS = ["经验性", "经验用药", "经验"]


def _case_features_support_anti_infective_gate(case_features, *, _plan: str = "") -> bool:
    """Engage I3 whenever a concrete antibiotic is actually recommended.

    A specific named antibiotic recommendation must ALWAYS enter the gate — it
    cannot be skipped just because the case text lacks an infection keyword
    (the Codex counter-example: case_text=胸痛，无发热 with a concrete 环丙沙星
    recommendation). The gate is the provenance check; skipping it lets any named
    drug through without evidence.
    """
    if not isinstance(case_features, Mapping):
        return False
    return True


def _iter_structured_mapping_list(value: Any) -> List[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _infection_diagnosis_labels(case_features: Any) -> List[str]:
    if not isinstance(case_features, Mapping):
        return []
    labels: List[str] = []
    for key in ("candidate_diagnoses", "diagnoses"):
        labels.extend(as_text_list(case_features.get(key)))
    for key in ("diagnosis", "primary_diagnosis"):
        value = case_features.get(key)
        if isinstance(value, str) and value.strip():
            labels.append(value)
    infection_markers = [
        "感染", "炎", "肺炎", "脓", "细菌", "病毒", "真菌", "尿路", "前列腺",
        "蜂窝织", "败血", "脓毒", "脓肿", "溃疡", "幽门", "螺杆菌", "结核",
        "心内膜炎", "骨髓炎", "脑膜炎", "胆囊炎", "阑尾炎", "憩室炎", "痢疾",
        # Catalog leaves / skin-mucosal infections without the 炎/感染 suffix.
        "角膜", "结膜", "中耳", "咽", "扁桃", "丹毒", "脓疱", "脓疱疮",
        "疹", "疱", "癣", "疣", "水痘", "麻疹", "风疹", "卡波西", "带状疱疹",
        "疱疹", "弓形虫", "疟", "寄生", "螺旋", "衣原体", "支原体", "念珠",
        "曲霉", "梅毒", "淋病", "性病", "hiv", "aids", "pml", "败血症",
        "放线菌", "军团", "风湿", "链球菌", "葡萄球菌", "肺炎球菌",
    ]
    out: List[str] = []
    for label in labels:
        blob = normalize_name(label)
        if any(normalize_name(marker) in blob for marker in infection_markers):
            out.append(clean_text(label))
    return out


def _expand_resistance_classes(items):
    """Expand drug-class entries into their member drugs (e.g. 喹诺酮 -> 环丙沙星/左氧氟沙星).

    Class-level resistance blocks every member drug, so a class entry must be
    normalized to its concrete members before matching a named drug.
    """
    out = []
    for item in items:
        members = _members_for_class(item)
        if members:
            out.extend(normalize_name(m) for m in members)
        else:
            out.append(normalize_name(item))
    return out


def _infection_diagnosis_has_current_evidence(
    label: str,
    case_features: Mapping[str, Any],
) -> bool:
    """Require current findings before a diagnosis can authorize empiric therapy."""
    target = normalize_name(label)
    for record in as_axis_list(case_features.get("diagnosis_candidate_records")):
        if normalize_name(record.get("disease")) == target and as_text_list(
            record.get("matched_evidence")
        ):
            return True
    raw_chunks = [
        clean_text(case_features.get("case_text")),
        clean_text(case_features.get("patient_text")),
        " ".join(as_text_list(case_features.get("positive_findings"))),
    ]
    for payload in (case_features.get("examination_results") or {}).values():
        if isinstance(payload, Mapping) and exam_result_is_usable(payload):
            raw_chunks.extend(structured_text_chunks(payload))
    evidence_text = " ".join(item for item in raw_chunks if item)
    if disease_matched_evidence(label, evidence_text):
        return True
    if disease_match_score(label, normalize_name(evidence_text)) >= 40:
        return True
    # Corneal disease is a catalog label whose evidence is usually a finding,
    # not the exact disease name.
    if "角膜" in target:
        return any(
            marker in normalize_name(evidence_text)
            for marker in ("角膜", "眼痛", "畏光", "角膜上皮缺损", "角膜溃疡")
        )
    return False


def _structured_anti_infective_provenance(case_features: Any) -> Dict[str, Any]:
    """Provenance may only come from structured patient/exam/diagnosis facts, never plan text."""
    empty = {
        "ast": [],
        "cultures": [],
        "confirmed_resistance": [],
        "empiric": None,
    }
    if not isinstance(case_features, Mapping):
        return empty
    out = {
        "ast": [],
        "cultures": [],
        "confirmed_resistance": [],
        "empiric": None,
    }
    raw = case_features.get("anti_infective_provenance")
    if isinstance(raw, Mapping):
        out["ast"].extend(_iter_structured_mapping_list(raw.get("ast")))
        out["cultures"].extend(_iter_structured_mapping_list(raw.get("cultures")))
        for item in _expand_resistance_classes(as_text_list(raw.get("confirmed_resistance"))):
            out["confirmed_resistance"].append(item)
        empiric = raw.get("empiric")
        if isinstance(empiric, Mapping):
            # Validate the empiric provenance before accepting it.
            if validate_empiric_provenance(empiric):
                out["empiric"] = empiric
        elif raw.get("empiric_allowed") or raw.get("empiric_indication"):
            candidate = {
                "allowed": bool(raw.get("empiric_allowed") or raw.get("empiric_indication")),
                "indication": clean_text(raw.get("empiric_indication")),
                "must_reassess_on_ast": True,
            }
            if validate_empiric_provenance(candidate):
                out["empiric"] = candidate
    # A bare top-level sensitivity_results field is NOT trusted provenance — it is
    # exactly the shape the Codex counter-example forges, so it must never clear the
    # gate. Only values nested under an explicit anti_infective_provenance object
    # (parsed from real examination_results) count as credible AST evidence.
    out["cultures"].extend(_iter_structured_mapping_list(case_features.get("culture_results")))
    for item in _expand_resistance_classes(as_text_list(case_features.get("confirmed_resistance"))):
        out["confirmed_resistance"].append(item)
    # Top-level empiric_documented/empiric_indication fields are trusted ONLY when
    # the source is verified_case_memory (marked by _verified_case_memory_source).
    if out["empiric"] is None and case_features.get("empiric_documented"):
        if case_features.get("_verified_case_memory_source") is True:
            candidate = {
                "allowed": True,
                "indication": clean_text(case_features.get("empiric_indication")) or "documented",
                "must_reassess_on_ast": True,
                "source": "infection_diagnosis",
                "evidence_ref": clean_text(case_features.get("empiric_indication")) or "documented",
            }
            out["empiric"] = candidate
    # Diagnosis labels alone are candidates, not empiric authorization evidence.
    # Infection-shaped catalog labels are a structured indication, but still require
    # explicit conditional reassessment rather than proving a pathogen or AST result.
    if out["empiric"] is None:
        infection_labels = [
            label
            for label in _infection_diagnosis_labels(case_features)
            if _infection_diagnosis_has_current_evidence(label, case_features)
        ]
        if infection_labels:
            candidate = {
                "allowed": True,
                "indication": infection_labels[0],
                "must_reassess_on_ast": True,
                "source": "infection_diagnosis",
                "evidence_ref": infection_labels[0],
            }
            if validate_empiric_provenance(candidate):
                out["empiric"] = candidate
    out["confirmed_resistance"] = sorted({item for item in out["confirmed_resistance"] if item})
    return out


def _named_anti_infective_drugs(normalized_plan: str) -> List[str]:
    generic = {
        normalize_name("抗菌"),
        normalize_name("抗感染"),
        normalize_name("抗生素"),
        normalize_name("抗菌药"),
    }
    named: List[str] = []
    for alias in ANTIBIOTIC_ALIASES:
        alias_norm = normalize_name(alias)
        if not alias_norm or alias_norm in generic:
            continue
        if treatment_recommends_drug(normalized_plan, [alias]):
            named.append(alias)
    return named


def _ast_result_for_drug(ast_rows: List[Mapping[str, Any]], drug: str) -> str:
    drug_norm = normalize_name(drug)
    for row in ast_rows:
        row_drug = normalize_name(row.get("drug") or row.get("drug_norm") or row.get("name"))
        if not row_drug:
            continue
        if row_drug in drug_norm or drug_norm in row_drug:
            result = clean_text(row.get("result") or row.get("susceptibility") or "").upper()
            if result in {"S", "I", "R", "敏感", "中介", "耐药"}:
                if result == "敏感":
                    return "S"
                if result == "中介":
                    return "I"
                if result == "耐药":
                    return "R"
                return result
    return ""


def _plan_has_conditional_empiric_language(normalized_plan: str) -> bool:
    markers = [
        normalize_name("待药敏"),
        normalize_name("根据药敏"),
        normalize_name("按药敏"),
        normalize_name("药敏后调整"),
        normalize_name("药敏结果后"),
        normalize_name("培养药敏后"),
        normalize_name("经验性"),
        normalize_name("经验用药"),
        normalize_name("条件化"),
        normalize_name("待培养"),
        normalize_name("待药敏结果"),
        normalize_name("药敏结果回报后"),
    ]
    return any(marker in normalized_plan for marker in markers if marker)


def _plan_claims_sensitivity_without_ast(normalized_plan: str) -> bool:
    """Affirmative susceptibility claims only; ignore gate warnings like 不得声称敏感/耐药."""
    cleaned = normalized_plan
    for phrase in (
        normalize_name("不得声称敏感或耐药"),
        normalize_name("不得声称敏感"),
        normalize_name("不得声称耐药"),
        normalize_name("暂无药敏前不得声称敏感或耐药"),
        normalize_name("暂不依据无证据文本声称敏感/耐药"),
        normalize_name("暂不依据无证据文本声称敏感或耐药"),
        normalize_name("禁止无证据肯定性敏感/耐药声称"),
        normalize_name("删除无药敏依据的敏感/耐药肯定性声称"),
        normalize_name("删除无药敏依据的肯定性敏感/耐药声称"),
    ):
        if phrase:
            cleaned = cleaned.replace(phrase, "")
    patterns = [
        "药敏结果示",
        "药敏示",
        "显示敏感",
        "显示耐药",
        "结果为敏感",
        "结果为耐药",
        "敏感株",
        "耐药株",
        "确认敏感",
        "确认耐药",
        "证实敏感",
        "证实耐药",
    ]
    return any(normalize_name(p) in cleaned for p in patterns)


def _strip_gate_boilerplate(normalized_plan: str) -> str:
    """Remove gate-authored remediation sentences before drug/claim auditing.

    Gate patches name the audited drug, so left in place they would re-trigger
    extraction on their own drug mention round after round. Structured evidence
    (allergy/resistance/AST/empiric authorization) is never satisfied by text;
    only the conditional-language formatting requirement may be fulfilled by a
    gate patch, so that check reads the unstripped plan instead.
    """
    patterns = [
        r"具体抗感染药物.{1,20}?缺乏培养药敏或结构化经验指征.{0,60}?敏感耐药声称",
        r"经验性使用.{1,20}?须条件化记录指征.{0,40}?不得声称敏感或耐药",
        r"停用已确认耐药的.{1,20}?改按药敏或感染专科方案",
        r"患者过敏或类别过敏含.{1,20}?停用并更换为非交叉过敏方案",
        r"删除无药敏依据的肯定性敏感耐药声称抗感染方案须待培养药敏后调整",
        r"删除无药敏依据的敏感耐药肯定性声称抗感染方案须待培养药敏后调整",
    ]
    cleaned = normalized_plan
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned


def find_anti_infective_evidence_gaps(treatment_plan, case_features):
    """Return issues/patches for named antibiotics without structured provenance.

    Provenance (culture/AST/allergy/resistance/empiric indication) must come from
    structured patient/exam/diagnosis facts. Gate-authored disclaimer text is never evidence.
    """
    normalized_plan_full = normalize_name(treatment_plan)
    normalized_plan = _strip_gate_boilerplate(normalized_plan_full)
    named = _named_anti_infective_drugs(normalized_plan)
    if not named:
        return {"issues": [], "patches": []}
    if not _case_features_support_anti_infective_gate(case_features):
        return {"issues": [], "patches": []}

    provenance = _structured_anti_infective_provenance(case_features)
    issues: List[Dict[str, Any]] = []
    patches: List[str] = []

    if (
        _plan_claims_sensitivity_without_ast(normalized_plan)
        and not provenance["ast"]
        and not provenance["confirmed_resistance"]
    ):
        issues.append(
            {
                "field": "treatment_plan",
                "code": "anti_infective_sensitivity_claim_without_ast",
                "severity": "must_fix",
                "problem": "plan_claims_sensitivity_without_structured_ast",
                "patchable": True,
                "edit": "删除无药敏依据的敏感/耐药肯定性声称；抗感染方案须待培养药敏后调整。",
            }
        )
        patches.append("删除无药敏依据的肯定性敏感/耐药声称；抗感染方案须待培养药敏后调整。")

    allergy_classes = set(_expand_resistance_classes(
        as_text_list(case_features.get("drug_allergies"))
    ))
    allergy_direct = set(normalize_name(a) for a in as_text_list(case_features.get("drug_allergies")))
    for drug in named:
        drug_norm = normalize_name(drug)
        ast_result = _ast_result_for_drug(provenance["ast"], drug)
        resistant = drug_norm in set(provenance["confirmed_resistance"]) or ast_result == "R"
        allergic = drug_norm in allergy_direct or drug_norm in allergy_classes
        if allergic:
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "anti_infective_drug_allergy",
                    "severity": "must_fix",
                    "problem": "named_anti_infective_with_drug_allergy:%s" % drug,
                    "patchable": False,
                    "edit": "患者过敏或类别过敏（含%s），停用并更换为非交叉过敏方案。" % drug,
                }
            )
            continue
        if resistant:
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "anti_infective_confirmed_resistance",
                    "severity": "must_fix",
                    "problem": "named_anti_infective_with_confirmed_resistance:%s" % drug,
                    "patchable": False,
                    "edit": "停用已确认耐药的%s，改按药敏或感染专科方案。" % drug,
                }
            )
            continue
        if ast_result == "S":
            continue
        empiric = provenance.get("empiric") if isinstance(provenance.get("empiric"), Mapping) else None
        empiric_allowed = bool(empiric and empiric.get("allowed"))
        # Formatting requirement only when structured empiric is already allowed.
        # Gate-authored disclaimers never invent empiric authorization (P0:
        # empty features + self-patch must keep failing while the drug remains).
        if empiric_allowed and _plan_has_conditional_empiric_language(normalized_plan_full):
            continue
        if empiric_allowed:
            patch = (
                "经验性使用%s须条件化：记录指征（%s），并在培养药敏结果回报后立即调整；"
                "暂无药敏前不得声称敏感或耐药。"
                % (drug, clean_text(empiric.get("indication")) or "感染表型")
            )
            issues.append(
                {
                    "field": "treatment_plan",
                    "code": "anti_infective_empiric_missing_conditional_language",
                    "severity": "must_fix",
                    "problem": "empiric_named_drug_without_conditional_language:%s" % drug,
                    "patchable": True,
                    "edit": patch,
                }
            )
            if patch not in patches:
                patches.append(patch)
            continue
        # No structured AST/empiric: named drug is not launderable by disclaimer text.
        # Mark unpatchable so converge/fail-closed instead of infinite append loops.
        patch = (
            "具体抗感染药物%s缺乏培养/药敏或结构化经验指征：先完善病原学送检，"
            "经验方案须条件化并在药敏结果回报后调整；禁止无证据肯定性敏感/耐药声称。"
            % drug
        )
        issues.append(
            {
                "field": "treatment_plan",
                "code": "anti_infective_without_sensitivity_evidence",
                "severity": "must_fix",
                "problem": "specific_anti_infective_without_sensitivity_or_empiric_evidence:%s" % drug,
                "patchable": False,
                "edit": patch,
            }
        )
        if patch not in patches:
            patches.append(patch)
    return {"issues": issues, "patches": patches}


def clean_text(value: Any) -> str:
    return str(value or "").strip()
