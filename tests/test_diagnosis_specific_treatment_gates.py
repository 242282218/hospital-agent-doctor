"""Diagnosis-specific treatment safety gate regressions."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_diagnosis_specific_treatment_gate,
    converge_verified_treatment,
    final_verifier,
)


def features(text: str, *, axis_id: str | None = None) -> dict:
    result = {
        "case_text": text,
        "positive_findings": [text],
        "candidate_diagnoses": [],
    }
    if axis_id:
        result["diagnosis_axes"] = [{"axis_id": axis_id, "evidence": [text]}]
    return result


def issue_codes(result: dict) -> set[str]:
    return {item["code"] for item in result["issues"]}


def test_neurocysticercosis_with_intracranial_hypertension_removes_triptan_path() -> None:
    case = features(
        "剧烈头痛、反复呕吐且弯腰加重；头颅CT示多发脑实质钙化和梗阻性脑积水，囊虫抗体阳性。"
    )

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="神经囊虫病",
        treatment_plan="考虑偏头痛，给予舒马曲坦止痛并门诊随访。",
        case_features=case,
    )

    assert "triptan_with_intracranial_hypertension_neurocysticercosis" in issue_codes(result)
    assert "舒马曲坦" not in result["treatment_plan"]
    patch = "".join(result["patches"])
    assert "颅高压" in patch or "脑积水" in patch
    assert "阿苯达唑" in patch
    assert "糖皮质激素" in patch or "抗炎" in patch
    assert "抗癫痫" in patch


def test_calcification_without_strong_etiologic_evidence_does_not_force_antiparasitic_treatment() -> None:
    case = features("既往头颅CT偶然发现单发钙化灶，目前无头痛、呕吐或脑积水，也无囊虫暴露证据。")

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="癫痫",
        treatment_plan="按癫痫专科方案随访。",
        case_features=case,
    )

    assert "triptan_with_intracranial_hypertension_neurocysticercosis" not in issue_codes(result)
    assert not any("阿苯达唑" in patch for patch in result["patches"])


def test_retinoblastoma_cannot_close_with_amblyopia_care_only() -> None:
    case = features(
        "1岁儿童反复左眼白瞳、内斜并追物能力下降。",
        axis_id="pediatric_leukocoria_retinoblastoma_until_excluded",
    )

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="视网膜母细胞瘤",
        treatment_plan="先配镜并进行遮盖训练，三个月后复查。",
        case_features=case,
    )

    assert "undertreated_retinoblastoma" in issue_codes(result)
    patch = "".join(result["patches"])
    assert "眼肿瘤" in patch or "儿童眼科" in patch
    assert "分期" in patch and "双眼" in patch
    assert "化疗" in patch and "局部治疗" in patch and "眼球摘除" in patch
    assert "交通" in patch or "照护" in patch or "社工" in patch


def test_sma_syndrome_requires_decompression_nutrition_position_and_surgical_escalation() -> None:
    case = features(
        "脊柱侧弯手术后餐后腹胀并胆汁性呕吐，躺着更差，左侧卧或蜷缩缓解。",
        axis_id="post_spinal_surgery_positional_duodenal_obstruction",
    )

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="肠系膜上动脉压迫综合征",
        treatment_plan="补液、止吐，采取半卧位并观察。",
        case_features=case,
    )

    assert "undertreated_sma_syndrome" in issue_codes(result)
    patch = "".join(result["patches"])
    assert "胃肠减压" in patch
    assert "电解质" in patch and "营养" in patch
    assert "左侧卧" in patch or "膝胸位" in patch
    assert "少量多餐" in patch
    assert "鼻十二指肠" in patch or "空肠营养" in patch
    assert "外科" in patch


def test_final_verifier_merges_retinoblastoma_treatment_patch() -> None:
    case = features(
        "1岁儿童反复左眼白瞳、内斜并追物能力下降。",
        axis_id="pediatric_leukocoria_retinoblastoma_until_excluded",
    )

    result = final_verifier(
        diagnosis="视网膜母细胞瘤",
        examinations=["眼部超声"],
        treatment_plan="配镜、遮盖训练并定期随访。",
        official_diseases=["视网膜母细胞瘤"],
        examination_catalog={"眼科检查": ["眼部超声"]},
        exam_plan_trace=[],
        case_features=case,
        safety_profiles=[],
    )

    assert "undertreated_retinoblastoma" in issue_codes(result)
    assert "眼肿瘤" in result["patched_treatment"] or "儿童眼科" in result["patched_treatment"]
    assert "分期" in result["patched_treatment"]


def test_negative_bacterial_culture_blocks_prophylactic_antibiotics_for_skin_maggot_case() -> None:
    case = features(
        "10岁儿童赤脚接触沙土后足趾出现黑点小疙瘩，抓破后红肿疼痛；细菌培养阴性。"
    )

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="皮肤蝇蛆病",
        treatment_plan="取出幼虫后，因培养阴性仍预防性使用莫匹罗星，必要时口服抗生素。",
        case_features=case,
    )

    assert "prophylactic_antibiotic_after_negative_bacterial_culture" in issue_codes(result)
    assert "预防性使用莫匹罗星" not in result["treatment_plan"]
    assert "取出幼虫" in result["treatment_plan"]
    assert "继发细菌感染证据" in "".join(result["patches"])


def test_skin_myiasis_requires_confirmed_secondary_bacterial_infection_for_systemic_antibiotics() -> None:
    case = features(
        "小腿和前臂反复长痛痒结节，有浆液性至脓性引流，近期低热、乏力和食欲差。"
    )
    case["examination_results"] = {
        "体格检查": {
            "status": "abnormal",
            "result": {"皮肤状态": "多个疖样结节，中央有穿刺点，并有浆液性至脓性引流；封闭后可见幼虫活动"},
        },
        "全血细胞计数（CBC）": {
            "status": "abnormal",
            "result": {
                "血红蛋白": "108 g/L［参考范围：120-160］",
                "血白细胞计数": "12.8 x 10^9/L［参考范围：4.5-11.0］",
            },
        },
    }

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="皮肤蝇蛆病",
        treatment_plan=(
            "取出幼虫后需立即启动经验性抗生素治疗，首选头孢氨苄或阿莫西林克拉维酸钾，"
            "疗程7-10天；同时监测伤口。"
        ),
        case_features=case,
    )

    assert "systemic_antibiotic_without_confirmed_secondary_infection" in issue_codes(result)
    assert "头孢氨苄" not in result["treatment_plan"]
    assert "阿莫西林克拉维酸钾" not in result["treatment_plan"]
    assert "培养或明确继发细菌感染" in "".join(result["patches"])


def test_skin_myiasis_keeps_antibiotic_when_secondary_infection_is_confirmed() -> None:
    case = features(
        "皮肤蝇蛆病病灶周围红肿热痛范围扩大，细菌培养阳性，明确继发蜂窝织炎。"
    )

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="皮肤蝇蛆病",
        treatment_plan="完整取出幼虫后，因培养阳性和蜂窝织炎给予口服头孢氨苄，并按药敏调整。",
        case_features=case,
    )

    assert "systemic_antibiotic_without_confirmed_secondary_infection" not in issue_codes(result)
    assert "头孢氨苄" in result["treatment_plan"]


def test_skin_myiasis_low_hemoglobin_requires_anemia_followup() -> None:
    case = features("反复皮肤结节伴乏力和食欲差。")
    case["examination_results"] = {
        "全血细胞计数（CBC）": {
            "status": "abnormal",
            "result": {"血红蛋白": "108 g/L［参考范围：120-160］"},
        }
    }

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="皮肤蝇蛆病",
        treatment_plan="完成幼虫移除，监测血常规并安排一周后复诊。",
        case_features=case,
    )

    assert "skin_myiasis_anemia_followup_missing" in issue_codes(result)
    assert "贫血" in "".join(result["patches"])
    assert "铁代谢" in "".join(result["patches"])


def test_skin_myiasis_plan_converges_before_antibiotic_evidence_gate() -> None:
    case = features("小腿和前臂反复长痛痒结节，有渗液，伴乏力和食欲差。")
    case["examination_results"] = {
        "体格检查": {
            "status": "abnormal",
            "result": {"皮肤状态": "多个疖样结节，中央有穿刺点，并有浆液性至脓性引流；封闭后可见幼虫活动"},
        },
        "全血细胞计数（CBC）": {
            "status": "abnormal",
            "result": {"血红蛋白": "108 g/L［参考范围：120-160］"},
        },
    }

    result = converge_verified_treatment(
        diagnosis="皮肤蝇蛆病",
        examinations=["体格检查", "全血细胞计数（CBC）"],
        treatment_plan="取出幼虫后需立即启动经验性抗生素治疗，首选头孢氨苄，疗程7-10天。",
        official_diseases=["皮肤蝇蛆病"],
        examination_catalog={"皮肤科": ["体格检查", "全血细胞计数（CBC）"]},
        exam_plan_trace=[],
        case_features=case,
        safety_profiles=[],
    )

    assert result is not None
    assert result["passed"] is True
    assert "头孢氨苄" not in result["patched_treatment"]
    assert "铁代谢" in result["patched_treatment"]


def test_qt_prolongation_and_tricyclic_overdose_blocks_arrhythmogenic_plan() -> None:
    case = features(
        "心电图示频发室性早搏和QTc 0.49秒；近期三环类抗抑郁药服用过量，既往有冠心病和心肌病。"
    )

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="室性早搏（PVCs）",
        treatment_plan="给予美托洛尔控制早搏，继续三环类抗抑郁药并观察。",
        case_features=case,
    )

    assert "qt_prolongation_with_tricyclic_exposure" in issue_codes(result)
    patch = "".join(result["patches"])
    assert "立即停用" in patch
    assert "心电监护" in patch or "电解质" in patch
    assert "美托洛尔" in result["treatment_plan"]


def test_dilated_cardiomyopathy_plan_requires_hf_core_and_acute_escalation() -> None:
    case = features(
        "活动后气短、下肢水肿和腹胀；超声心动图示左心室扩张，LVEF 28%，伴头晕接近晕厥。"
    )

    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="扩张型心肌病",
        treatment_plan="建议低盐饮食，定期复查超声心动图，适度运动。",
        case_features=case,
    )

    assert "hfrEF_missing_guideline_core" in issue_codes(result)
    patch = "".join(result["patches"])
    assert "ARNI" in patch or "ACEI" in patch or "血管紧张素" in patch
    assert "利尿" in patch and "心衰专科" in patch
