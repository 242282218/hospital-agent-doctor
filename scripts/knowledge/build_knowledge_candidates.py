from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from offline.artifacts import content_hash, file_hash, read_json, write_immutable_json
from offline.candidates import load_candidate, write_candidate
from offline.knowledge_rules import knowledge_rule_candidate


BASE_DIR = Path(__file__).resolve().parents[2]
EXPECTED_RULES = {
    "congenital_infection_differential": "diagnosis_differential_rule",
    "congenital_infection_organ_closure": "clinical_closure_rule",
    "symptom_over_background_condition": "diagnosis_priority_rule",
    "negative_culture_antibiotic_stewardship": "treatment_gate_rule",
    "qt_tricyclic_structural_heart_risk": "treatment_gate_rule",
    "hfref_phase_ordered_treatment": "treatment_sequence_rule",
}
DIFFERENTIAL_PARAMETERS = {
    "age_fact_codes": ["neonate", "infant"],
    "exposure_fact_codes": ["intrauterine_viral_exposure"],
    "manifestation_fact_codes": [
        "congenital_jaundice",
        "congenital_rash",
        "infant_hearing_abnormality",
        "thrombocytopenia",
        "congenital_neuroimaging_abnormality",
        "congenital_cataract",
        "patent_ductus_arteriosus",
        "periventricular_calcifications",
        "microcephaly",
    ],
    "rubella_rank_fact_codes": [
        "congenital_cataract",
        "patent_ductus_arteriosus",
        "rubella_igm_positive_in_infant",
        "rubella_pcr_positive_in_infant",
    ],
    "cmv_rank_fact_codes": [
        "periventricular_calcifications",
        "microcephaly",
        "cmv_igm_positive",
        "cmv_pcr_positive",
        "cmv_saliva_or_urine_pcr_positive_within_21_days",
    ],
    "rubella_confirmed_fact_codes": [
        "rubella_igm_positive_in_infant",
        "rubella_pcr_positive_in_infant",
    ],
    "cmv_confirmed_fact_codes": [
        "cmv_saliva_or_urine_pcr_positive_within_21_days"
    ],
    "rubella_axis_id": "congenital_rubella",
    "cmv_axis_id": "congenital_cmv",
}


def _control(
    control_id: str,
    kind: str,
    facts: Sequence[str],
    assertions: Sequence[str],
) -> Dict[str, Any]:
    return {
        "control_id": control_id,
        "kind": kind,
        "facts": list(facts),
        "assertions": list(assertions),
    }


def _audit_runtime(stage: str) -> Dict[str, Any]:
    return {"status": "audit_only", "stage": stage}


def _active_priority_runtime() -> Dict[str, Any]:
    return {
        "status": "active",
        "stage": "diagnosis_candidates",
        "opcode": "promote_supported_current_over_background",
        "parameters": {
            "target_roles": ["current_problem"],
            "target_support_levels": ["objective"],
            "background_roles": ["background_condition"],
            "background_relations": ["unrelated"],
            "excluded_relations": ["explains"],
            "preserve_urgencies": ["emergency"],
            "fallback_policy": "official_catalog_only",
        },
    }


def _active_differential_runtime() -> Dict[str, Any]:
    return {
        "status": "active",
        "stage": "diagnosis_candidates",
        "opcode": "expand_congenital_infection_axes",
        "parameters": deepcopy(DIFFERENTIAL_PARAMETERS),
    }


def _effect(
    *,
    rule_id: str,
    triggers: Sequence[str],
    required_evidence: Sequence[str],
    exclusions: Sequence[str],
    actions: Mapping[str, Any],
    positive_controls: Sequence[Mapping[str, Any]],
    negative_controls: Sequence[Mapping[str, Any]],
    source_refs: Sequence[Mapping[str, str]],
    test_refs: Sequence[Mapping[str, str]],
    priority: int,
    phase: str,
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "clinical-knowledge-candidate/v2",
        "rule_id": rule_id,
        "triggers": list(triggers),
        "required_evidence": list(required_evidence),
        "exclusions": list(exclusions),
        "effect": dict(actions),
        "positive_controls": [dict(item) for item in positive_controls],
        "negative_controls": [dict(item) for item in negative_controls],
        "source_refs": [dict(item) for item in source_refs],
        "test_refs": [dict(item) for item in test_refs],
        "priority": priority,
        "scope": {"phase": phase, "application": "trigger_bound"},
        "review_requirements": [
            "人工核对触发与排除边界",
            "逐项核对正例、近邻负例和合理例外",
            "确认规则不得替代个体化临床判断",
        ],
        "runtime": deepcopy(dict(runtime)),
    }


def _rule_effects(
    source_refs: Sequence[Mapping[str, str]],
    test_refs: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[str, Any]]:
    return {
        "congenital_infection_differential": _effect(
            rule_id="congenital_infection_differential",
            triggers=["新生儿或婴儿存在宫内感染暴露与先天感染表现组合"],
            required_evidence=["黄疸、皮疹、听力异常、血小板异常或神经影像线索之一"],
            exclusions=["仅有非特异单一症状且无宫内感染线索", "已有充分病原证据支持单一方向"],
            actions={
                "add_diagnostic_axes": ["先天性风疹方向", "巨细胞病毒方向"],
                "ranking_policy": "evidence_ordered",
            },
            positive_controls=[
                _control("congenital_multi_system", "positive", ["宫内病毒暴露", "黄疸与皮疹并存"], ["保留两个先天感染方向"]),
                _control("congenital_hearing_signal", "positive", ["婴儿期听力异常", "伴先天感染线索"], ["使用区分性证据排序"]),
                _control("congenital_rubella_rank_signal", "positive", ["宫内病毒暴露", "先天性白内障或动脉导管未闭线索"], ["风疹方向排在巨细胞病毒方向之前"]),
                _control("congenital_cmv_rank_signal", "positive", ["宫内病毒暴露", "脑室周围钙化或小头畸形线索"], ["巨细胞病毒方向排在风疹方向之前"]),
            ],
            negative_controls=[
                _control("isolated_neonatal_jaundice", "near_neighbor", ["孤立轻度黄疸", "无暴露及多系统线索"], ["不触发双轴扩展"]),
                _control("postnatal_infection_pattern", "near_neighbor", ["出生后出现感染表现", "无宫内多系统受累模式"], ["不触发先天感染双轴扩展"]),
                _control("documented_single_pathogen", "reasonable_exception", ["已有可靠单一病原证据"], ["不强制保留证据不支持的另一方向"]),
            ],
            source_refs=source_refs,
            test_refs=test_refs,
            priority=120,
            phase="diagnosis",
            runtime=_active_differential_runtime(),
        ),
        "congenital_infection_organ_closure": _effect(
            rule_id="congenital_infection_organ_closure",
            triggers=["先天感染方向已建立且出现器官受累线索"],
            required_evidence=["心脏、血小板、听力、肝脏或神经线索至少一项"],
            exclusions=["无对应器官线索", "相关器官评估已充分完成且无新变化"],
            actions={
                "add_exam_intent_ids": [
                    "exam_intent_congenital_infection_organ_involvement"
                ],
                "deduplicate": "intent_id",
            },
            positive_controls=[
                _control("hearing_closure", "positive", ["听力异常线索"], ["补听力相关闭环"]),
                _control("hematologic_closure", "positive", ["紫癜或血小板异常线索"], ["补血液受累闭环"]),
            ],
            negative_controls=[
                _control("no_organ_signal", "near_neighbor", ["仅有暴露史", "无器官受累线索"], ["不固定广筛"]),
                _control("unrelated_single_system_abnormality", "near_neighbor", ["无先天感染器官线索", "存在无关单系统异常"], ["不扩展先天感染器官广筛"]),
                _control("closure_already_complete", "reasonable_exception", ["对应器官评估已完成且稳定"], ["不重复添加"]),
            ],
            source_refs=source_refs,
            test_refs=test_refs,
            priority=130,
            phase="closure",
            runtime=_audit_runtime("clinical_closure"),
        ),
        "symptom_over_background_condition": _effect(
            rule_id="symptom_over_background_condition",
            triggers=["当前问题候选有可追溯的同系统客观证据，且基础病与当前问题无解释关系"],
            required_evidence=["与当前问题同系统且可追溯的异常检查或客观体征"],
            exclusions=[
                "基础病能够解释当前问题或相关同系统客观证据",
                "基础病已确诊急症（关键是新发或进展性急性靶器官损害，不是单次血压数值本身）",
                "基础病疑似急症且急性靶器官损害尚待排除",
                "仅有主观症状且同系统检查正常",
                "仅有主观症状且尚未检查同系统客观证据",
                "仅有主观症状且同系统检查结果不确定",
            ],
            actions={
                "priority_policy": "objective_evidence_first",
                "fallback_policy": "official_catalog_only",
            },
            positive_controls=[
                _control("hearing_over_hypertension", "positive", ["进行性耳鸣与听力下降", "存在可追溯的耳部异常检查或客观体征", "高血压病史不能解释耳部客观证据"], ["提升客观证据支持的当前问题候选"]),
                _control("focal_symptom_over_history", "positive", ["局灶症状有同系统可追溯客观异常", "既往病史与当前问题无解释关系"], ["提升客观证据支持的当前问题候选"]),
                _control("severe_elevation_without_acute_target_organ_damage", "positive", ["严重血压升高", "无急性靶器官损害", "耳部异常检查或客观体征充分且可追溯"], ["提升当前问题候选"]),
                _control("transient_elevation_without_acute_target_organ_damage", "positive", ["疼痛或焦虑相关一过性血压升高", "无急性靶器官损害", "耳部异常检查或客观体征充分且可追溯"], ["提升当前问题候选"]),
            ],
            negative_controls=[
                _control("background_explains_current_problem", "near_neighbor", ["基础病能够解释当前问题及其同系统客观证据"], ["不提升当前问题候选"]),
                _control("subjective_symptom_normal_exam", "near_neighbor", ["仅有主观症状", "同系统检查正常"], ["不提升当前问题候选"]),
                _control("subjective_symptom_not_examined", "near_neighbor", ["仅有主观症状", "同系统客观检查尚未完成"], ["不提升当前问题候选"]),
                _control("subjective_symptom_uncertain_result", "near_neighbor", ["仅有主观症状", "同系统检查结果不确定"], ["不提升当前问题候选"]),
                _control("confirmed_hypertensive_emergency", "reasonable_exception", ["严重血压升高", "存在新发或进展性急性靶器官损害", "另有独立耳部客观证据"], ["基础病急症优先", "独立耳部轴不得删除"]),
                _control("suspected_hypertensive_emergency", "reasonable_exception", ["严重血压升高", "急性靶器官损害尚待排除", "另有独立耳部客观证据"], ["先保留基础病急症优先", "独立耳部轴不得删除"]),
            ],
            source_refs=source_refs,
            test_refs=test_refs,
            priority=110,
            phase="diagnosis",
            runtime=_active_priority_runtime(),
        ),
        "negative_culture_antibiotic_stewardship": _effect(
            rule_id="negative_culture_antibiotic_stewardship",
            triggers=["细菌培养阴性且方案包含常规或预防性抗菌"],
            required_evidence=["当前缺少继发细菌感染或全身感染证据"],
            exclusions=["脓毒症", "脓肿", "进行性蜂窝织炎", "脓性分泌物", "取材前已使用抗菌药"],
            actions={
                "remove_treatment_codes": [
                    "routine_antibiotic",
                    "prophylactic_antibiotic",
                ],
                "preserve_treatment_codes": [
                    "etiology_specific_care",
                    "wound_care",
                    "reassessment",
                ],
                "gate_policy": "require_infection_evidence",
            },
            positive_controls=[
                _control("negative_culture_prophylaxis", "positive", ["培养阴性", "无感染征象", "仍计划预防性抗菌"], ["移除预防性抗菌", "保留其他处置"]),
                _control("negative_culture_routine", "positive", ["培养阴性", "无全身感染", "计划常规抗菌"], ["改为观察和复评路径"]),
            ],
            negative_controls=[
                _control("culture_negative_no_antibiotic", "near_neighbor", ["培养阴性", "方案未包含抗菌"], ["不改动方案"]),
                _control("documented_infection_antibiotic_indication", "near_neighbor", ["培养阳性或存在明确局灶感染", "抗菌治疗有明确指征"], ["不移除有指征的抗菌治疗"]),
                _control("sepsis_exception", "reasonable_exception", ["脓毒症表现或血流动力学不稳定"], ["不得仅凭阴性培养停止必要经验治疗"]),
                _control("pretreatment_exception", "reasonable_exception", ["取材前已使用抗菌药", "仍有感染证据"], ["保留临床评估支持的抗菌路径"]),
            ],
            source_refs=source_refs,
            test_refs=test_refs,
            priority=210,
            phase="treatment",
            runtime=_audit_runtime("treatment"),
        ),
        "qt_tricyclic_structural_heart_risk": _effect(
            rule_id="qt_tricyclic_structural_heart_risk",
            triggers=["QT间期延长并存在三环类药物过量或持续暴露"],
            required_evidence=["心电图异常或毒性暴露信息", "结构性心脏病作为风险放大信息时需有病史或检查支持"],
            exclusions=["普通室性早搏但QT正常且无三环类暴露", "药物暴露未发生且无毒性证据"],
            actions={
                "remove_treatment_codes": ["tricyclic_arrhythmogenic_exposure"],
                "preserve_treatment_codes": [
                    "continuous_ecg_monitoring",
                    "electrolyte_recheck",
                    "severity_escalation",
                ],
                "gate_policy": "require_toxicity_evidence",
                "risk_modifier_codes": ["structural_heart_disease"],
            },
            positive_controls=[
                _control("qt_tricyclic_overdose", "positive", ["QT延长", "三环类过量"], ["停止暴露并监护", "按毒性严重度分层"]),
                _control("qt_structural_modifier", "positive", ["QT延长", "持续三环类暴露", "结构性心脏病"], ["提高监护与复核优先级"]),
            ],
            negative_controls=[
                _control("ordinary_pvc", "near_neighbor", ["普通室性早搏", "QT正常", "无三环类暴露"], ["不触发药物毒性门"]),
                _control("no_exposure", "near_neighbor", ["结构性心脏病", "无QT延长及药物暴露"], ["不单独触发"]),
                _control("unstable_arrhythmia", "reasonable_exception", ["低血压或恶性室性心律失常"], ["保留紧急复苏和专科升级"]),
            ],
            source_refs=source_refs,
            test_refs=test_refs,
            priority=230,
            phase="treatment",
            runtime=_audit_runtime("treatment"),
        ),
        "hfref_phase_ordered_treatment": _effect(
            rule_id="hfref_phase_ordered_treatment",
            triggers=["射血分数降低型心衰或明确左室收缩功能降低"],
            required_evidence=["射血分数、容量负荷、灌注、血压、肾功能和血钾中的阶段判定信息"],
            exclusions=["无心衰证据的孤立气短", "射血分数保留且无其他收缩功能降低证据"],
            actions={
                "ordered_patch_codes": [
                    "stabilize_hemodynamics",
                    "decongest_if_overloaded",
                    "titrate_core_therapy_if_stable",
                ],
                "sequence_policy": "acute_before_stable",
                "skip_patch_codes": ["force_diuresis_when_euvolemic"],
            },
            positive_controls=[
                _control("acute_congested_hfref", "positive", ["射血分数降低", "水肿或肺淤血"], ["先稳定并去充血", "稳定后再推进核心药物"]),
                _control("stable_hfref", "positive", ["射血分数降低", "血流动力学稳定"], ["评估并有序启动或滴定核心药物"]),
            ],
            negative_controls=[
                _control("stable_euvolemic", "near_neighbor", ["状态稳定", "无容量负荷"], ["不强制利尿"]),
                _control("no_hf_evidence", "near_neighbor", ["孤立气短", "无结构或功能证据"], ["不触发心衰序列"]),
                _control("low_pressure_exception", "reasonable_exception", ["低血压或低灌注"], ["暂缓不耐受药物并先稳定"]),
                _control("renal_potassium_exception", "reasonable_exception", ["高钾或进行性肾损害"], ["保留暂缓和复核空间"]),
            ],
            source_refs=source_refs,
            test_refs=test_refs,
            priority=220,
            phase="treatment",
            runtime=_audit_runtime("treatment"),
        ),
    }


def _file_refs(project_root: Path, paths: Iterable[Path]) -> list[Dict[str, str]]:
    project_root = Path(project_root).resolve()
    refs: Dict[str, str] = {}
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError("explicit input file missing: %s" % path)
        try:
            relative = path.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("input file must be inside project root: %s" % path) from exc
        relative_text = relative.as_posix()
        refs[relative_text] = file_hash(path)
    if not refs:
        raise ValueError("at least one explicit input file is required")
    return [{"path": path, "sha256": refs[path]} for path in sorted(refs)]


def _verify_existing(
    batch_dir: Path,
    *,
    project_root: Path,
    receipt: Mapping[str, Any],
    checklist: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_files = {
        "source_receipt.json",
        "review_checklist.json",
        *("candidates/%s.json" % rule_id for rule_id in candidates),
    }
    actual_files = {
        path.relative_to(batch_dir).as_posix()
        for path in batch_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("existing knowledge candidate batch has unexpected files")
    if read_json(batch_dir / "source_receipt.json") != dict(receipt):
        raise FileExistsError("existing source receipt differs")
    if read_json(batch_dir / "review_checklist.json") != dict(checklist):
        raise FileExistsError("existing review checklist differs")
    if (batch_dir / "decisions").exists() or list(batch_dir.rglob("*decision*")):
        raise ValueError("knowledge candidate batch must not contain decisions")
    for rule_id, expected in candidates.items():
        loaded = load_candidate(batch_dir / "candidates" / (rule_id + ".json"), project_root=project_root)
        if loaded != dict(expected):
            raise FileExistsError("existing candidate differs: %s" % rule_id)


def build_knowledge_candidate_batch(
    *,
    project_root: Path,
    source_files: Sequence[Path],
    test_files: Sequence[Path],
    artifact_root: Path,
    supersedes_batch_id: str | None = None,
) -> Dict[str, Any]:
    project_root = Path(project_root).resolve()
    source_refs = _file_refs(project_root, source_files)
    test_refs = _file_refs(project_root, test_files)
    effects = _rule_effects(source_refs, test_refs)
    if set(effects) != set(EXPECTED_RULES):
        raise ValueError("knowledge rule inventory mismatch")
    candidates = {
        rule_id: knowledge_rule_candidate(
            candidate_type=EXPECTED_RULES[rule_id],
            effect=effects[rule_id],
            project_root=project_root,
        )
        for rule_id in sorted(EXPECTED_RULES)
    }
    batch_core = {
        "schema_version": "knowledge-candidate-batch/v2",
        "effect_contract_version": "typed-effect/v1",
        "source_files": source_refs,
        "test_files": test_refs,
        "candidate_hashes": {
            rule_id: candidates[rule_id]["candidate_hash"] for rule_id in sorted(candidates)
        },
        "supersedes_batch_id": supersedes_batch_id,
    }
    batch_id = content_hash(batch_core)
    receipt = {
        "schema_version": "knowledge-source-receipt/v1",
        "batch_id": batch_id,
        "source_files": source_refs,
        "test_files": test_refs,
        "candidate_hashes": batch_core["candidate_hashes"],
    }
    checklist = {
        "schema_version": "knowledge-review-checklist/v2",
        "batch_id": batch_id,
        "approval_status": "pending_user_review",
        "review_package_role": "final_review_package_candidate",
        "supersedes_batch_id": supersedes_batch_id,
        "rule_ids": sorted(EXPECTED_RULES),
        "checks": [
            "触发与证据边界已人工复核",
            "排除、近邻负例与合理例外已人工复核",
            "来源与测试文件哈希已复算",
            "type-specific effect 白名单与短结构化原语已复核",
            "无病例标识或答案式内容泄漏",
        ],
    }
    batch_dir = Path(artifact_root) / batch_id
    if batch_dir.exists():
        _verify_existing(
            batch_dir,
            project_root=project_root,
            receipt=receipt,
            checklist=checklist,
            candidates=candidates,
        )
        return {"batch_id": batch_id, "batch_dir": str(batch_dir), "reused": True}

    for rule_id, candidate in candidates.items():
        write_candidate(batch_dir / "candidates" / (rule_id + ".json"), candidate)
    write_immutable_json(batch_dir / "source_receipt.json", receipt)
    write_immutable_json(batch_dir / "review_checklist.json", checklist)
    return {"batch_id": batch_id, "batch_dir": str(batch_dir), "reused": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pending typed knowledge candidates offline.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, action="append", required=True)
    parser.add_argument("--test-file", type=Path, action="append", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--supersedes-batch-id")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    return build_knowledge_candidate_batch(
        project_root=args.project_root,
        source_files=args.source_file,
        test_files=args.test_file,
        artifact_root=args.artifact_root,
        supersedes_batch_id=args.supersedes_batch_id,
    )


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
