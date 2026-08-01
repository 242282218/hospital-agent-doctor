"""Prompt templates for the baseline doctor agent."""

from __future__ import annotations

import json
from string import Template
from typing import Any, Dict


DOCTOR_SYSTEM_PROMPT = """你是一名医生，正在根据患者信息完成一次诊疗。

基本要求：
- 根据已有对话、已开检查和检查结果做判断。
- 不要编造病史或检查结果；检查结果以系统返回为准。
- 面向患者的问题使用中文。
- 检查名称、检查类别、科室名称和诊断名称要优先使用给定的标准名称。
- 按用户消息要求输出 JSON，不要输出 Markdown 或额外说明。
"""


JSON_REPAIR_SYSTEM_PROMPT = """请把输入改写为合法 JSON 对象。
只输出 JSON，不要输出解释、Markdown、代码围栏或额外文本。"""


NEXT_ACTION_PROMPT = """根据当前病例状态，选择下一步操作。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

可选动作：
- ask_patient：问诊。选择该动作时，同时输出这一步要问患者的问题。
- order_examination：开检查。后续会先选检查类别，再从该类别中选择具体检查。
- final_diagnosis：进入诊断和治疗。后续会先选科室，再从该科室中选择疾病并给出治疗方案。

判断要点：
- 在诊断开始阶段，或者缺少诊断疾病所必需的关键信息时，优先 ask_patient。
- 在进入 final_diagnosis 前，必须确认药物过敏、禁忌药、当前用药、妊娠/哺乳或儿童老人等关键安全信息；缺失时优先 ask_patient。
- 同时确认会改变治疗的基础病或合并症、职业或日常活动需求、依从性障碍和治疗偏好；缺失时用一个简洁问题合并询问。
- 不要因为已经问过固定轮数而跳过仍能改变诊断或治疗的非重复问题，也不要重复询问患者已经回答的信息。
- 如果某项检查会影响诊断或治疗，选择 order_examination。
- 如果信息已经足够，选择 final_diagnosis。

请只返回 JSON：
{
  "action": "ask_patient | order_examination | final_diagnosis",
  "question": "action=ask_patient 时填写一个中文问题，否则为空字符串",
  "reason": "简要说明选择这个动作的原因"
}
"""


EXAM_CATEGORY_PROMPT = """请选择下一步最合适的检查类别。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

可选检查类别：
$exam_categories

要求：
- 只能从“可选检查类别”中选择一个标准类别名称。
- 选择最能帮助确认诊断或指导治疗的检查类别。
- 避免选择已经无法提供增量信息的类别。

请只返回 JSON：
{
  "category": "一个标准检查类别名称",
  "reason": "选择该类别的原因"
}
"""


EXAM_ITEM_PROMPT = """请从指定检查类别中选择这次要开的具体检查。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

检查类别：
$category

该类别下的标准检查名称：
$exam_items

要求：
- 只能从“该类别下的标准检查名称”中选择。
- 选择当前确实需要的检查，不要为了凑数量而选择无关检查。
- 不要选择已经做过的检查。

请只返回 JSON：
{
  "examinations": ["标准检查名称1", "标准检查名称2"],
  "reason": "选择这些检查的原因"
}
"""


DEPARTMENT_PROMPT = """请选择最可能负责当前诊断的科室。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

可选科室：
$departments

要求：
- 只能从“可选科室”中选择一个标准科室名称。
- 根据患者表现、检查结果和主要鉴别诊断选择。

请只返回 JSON：
{
  "department": "一个标准科室名称",
  "reason": "选择该科室的原因"
}
"""


DISEASE_CANDIDATE_PROMPT = """请从候选标准疾病名称中选择最可能的诊断，并给出治疗方案。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

候选标准疾病名称：
$disease_candidates

要求：
- diagnosis 只能填写一个“候选标准疾病名称”中的 disease。
- role=secondary 或 evidence_status=unconfirmed 的候选只能作为并行鉴别；缺少对应客观证据时，不得覆盖能解释主诉和已确认异常的主轴候选。
- clinical_role=background_condition 的候选不能覆盖有病例证据支持的 current_problem 候选。
- priority=red_flag 的主轴必须闭合；候选不足时在 reasoning 中明确标记 candidate_pool_inconsistent，不要把弱候选解释成已确诊。
- diagnosis 与 reasoning 必须一致；不得先说明候选严重不符，再因“只能从候选选择”提交该候选。
- 如果候选中存在比当前年龄或科室更贴近症状模式的疾病，优先选择该疾病，不要被年龄或单一科室锁死。
- treatment_plan 用中文给出治疗方案，重点说明有效性、个性化和安全性：
  1. 有效性：针对诊断给出主要治疗、必要药物或操作。
  2. 个性化：结合患者病史、症状、检查结果、年龄/妊娠/基础病/过敏史等因素调整。
  3. 安全性：说明禁忌、监测、复诊和需要立即就医的危险信号。
- reasoning 简要说明诊断依据和治疗考虑。

请只返回 JSON：
{
  "diagnosis": "一个候选标准疾病名称",
  "treatment_plan": "具体治疗方案",
  "reasoning": "诊断和治疗推理"
}
"""


DISEASE_AND_TREATMENT_PROMPT = """请在指定科室中选择最可能的疾病，并给出治疗方案。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

科室：
$department

该科室下的标准疾病名称：
$diseases

要求：
- diagnosis 只能填写一个“该科室下的标准疾病名称”。
- treatment_plan 用中文给出治疗方案，重点说明有效性、个性化和安全性：
  1. 有效性：针对诊断给出主要治疗、必要药物或操作。
  2. 个性化：结合患者病史、症状、检查结果、年龄/妊娠/基础病/过敏史等因素调整。
  3. 安全性：说明禁忌、监测、复诊和需要立即就医的危险信号。
- reasoning 简要说明诊断依据和治疗考虑。

请只返回 JSON：
{
  "diagnosis": "一个标准疾病名称",
  "treatment_plan": "具体治疗方案",
  "reasoning": "诊断和治疗推理"
}
"""


TREATMENT_REVIEW_PROMPT = """请作为 final 前治疗复核助手，只修订治疗方案，不改变最终诊断。

最终诊断：
$diagnosis

初步治疗方案：
$treatment_plan

可引用证据目录：
$review_evidence_catalog

要求：
- 只有证据目录条目可以建立患者事实；不得编造，也不得从初步方案或目录外内容推断病史、检查结果、禁忌或个人背景。
- 每个 edit 只能引用上述证据目录中的 id；不得自行改写或拼接 evidence id。
- 修复所有 must_fix 问题；发现禁忌药或不安全操作时必须明确删除相应推荐，不能只写“谨慎使用”。
- polarity=negative 的证据可用于删除不受支持的治疗，但不能用于新增正向病史、合并症、当前用药或禁忌事实。
- source=diagnosis 且 id=diagnosis:final 只提供诊断上下文，不能单独授权 delete、replace、新药、患者敏感事实或延迟治疗。
- consistency_status=conflicted 时，禁止减少或替换已有器官急症治疗；只允许由其他正向证据支持的急诊、转诊、监测、禁忌或相互作用安全追加。
- polarity=missing 只能用于 append 检查、监测、复评或会诊闭环，不得删除或延迟已有治疗。
- 按最终诊断复核标准治疗目标、必要操作、监测、复诊和危险信号，避免只有笼统对症处理。
- 结合真实存在的年龄、妊娠状态、基础病或合并症、当前用药、过敏/禁忌、器官功能调整方案。
- 结合真实存在的职业或日常活动需求、依从性障碍、费用限制和治疗偏好给出可执行建议；缺少证据时不要补写。
- 使用 delete、replace 或 append 原子 edit。delete/replace 的 target 必须逐字来自初步治疗方案且各 edit 不得重叠；append 的 target 必须为空字符串。
- 没有可靠修订时返回空 edits，不要复述原方案。
- 不引用 evaluation、expected 或 reference，不输出推测性的标准答案。

请只返回 JSON：
{
  "edits": [
    {
      "edit_id": "本次响应内唯一的简短 id",
      "operation": "delete | replace | append",
      "target": "delete/replace 时为原方案精确子串；append 时为空字符串",
      "replacement": "delete 时为空字符串；replace/append 时为新文本",
      "evidence_refs": ["证据目录中的稳定 id"]
    }
  ],
  "revision_summary": ["本次修订解决的问题"]
}
"""


DIAGNOSTIC_CONTEXT_PROMPT = """请根据患者对话、已开检查和检查结果，输出结构化诊断上下文。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

要求：
- 不要编造未提供的信息；所有 evidence 必须来自历史对话或检查结果。
- case_features 使用固定槽位，但槽位内 label 可以开放增长。
- differential 列出 5-10 个临床鉴别诊断，raw_name 可以是临床常用表达。
- normalization_suggestions 只建议把 raw_name 映射到更合适的官方疾病名；没有足够证据时留空。
- suggested_official_name 必须尽量使用标准中文疾病名；不要输出症状或检查名。
- supporting_feature_labels 必须引用 case_features 中已经出现的 label。

请只返回 JSON：
{
  "case_features": {
    "demographics": [{"label": "人群特征A", "evidence": "来自病例的人群信息", "confidence": "high"}],
    "symptom_clusters": [{"label": "症状组合A", "evidence": "来自问诊的症状组合", "confidence": "high"}],
    "exam_evidence": [{"label": "检查证据A", "evidence": "来自检查结果的阳性或阴性证据", "confidence": "high"}],
    "microbiology": [{"label": "病原学证据A", "evidence": "来自培养、核酸或抗体等结果", "confidence": "high"}],
    "organ_risk": [{"label": "器官受累风险A", "evidence": "来自症状、体征或检查的器官风险", "confidence": "medium"}],
    "medication_risk": [],
    "red_flags": [{"label": "危险信号A", "evidence": "来自病例的危险信号", "confidence": "high"}]
  },
  "differential": [
    {"raw_name": "疾病名称或临床表达", "rank": 1, "reason": "简要依据"}
  ],
  "normalization_suggestions": [
    {
      "raw_name": "临床表达A",
      "suggested_official_name": "官方疾病名A",
      "confidence": "medium",
      "supporting_feature_labels": ["证据标签A", "证据标签B"],
      "rationale": "简要说明为什么可以映射"
    }
  ],
  "reasoning": "简要说明主要鉴别思路"
}
"""


DIAGNOSTIC_AXIS_CONSULT_PROMPT = """请作为诊断轴会诊助手，基于当前病例输出结构化临床轴和风险闭环。

记忆摘要：
$memory_notes

历史对话：
$chat_history

已开检查及结果：
$examinations

当前候选疾病：
$disease_candidates

要求：
- 不要编造未提供的信息；所有 evidence 必须能在历史对话或检查结果中找到。
- diagnosis_axes 描述综合征、解剖部位、器官/人群风险或治疗安全轴，不要写病例 ID。
- 每个轴必须明确 clinical_role、priority 和 closure_requirement；既往已治疗、缓解且无当前活动证据的疾病必须标为 background_condition。
- 活动性出血、气道威胁或不可逆器官损害风险标为 red_flag；免疫抑制宿主的新发感染综合征至少标为 high。
- closure_requirement 必须描述 final 前可执行的诊断、检查、转诊或安全处置闭环，不能只写“进一步检查”。
- candidate_official_names 只能填可能的标准疾病名；不确定时少填或留空。
- exam_intents 只写最多 1-3 个能改变主诊断或治疗安全的检查目的，不写 expected 答案列表；不要加入常规监测、无差别广筛或已经完成的检查。
- treatment_risks 只写与本病例证据和当前轴直接相关的通用风险标签；不得复制与器官、人群或风险主题无关的示例标签。

请只返回 JSON：
{
  "intake_facts": {
    "demographics": [{"label": "人群特征", "evidence": "病例原文证据", "confidence": "high"}],
    "anatomic_sites": [{"label": "部位", "evidence": "病例原文证据", "confidence": "high"}],
    "symptom_clusters": [{"label": "症状组合", "evidence": "病例原文证据", "confidence": "high"}],
    "exam_evidence": [{"label": "检查证据", "evidence": "检查结果证据", "confidence": "high"}],
    "organ_risk": [{"label": "器官风险", "evidence": "病例原文或检查证据", "confidence": "medium"}],
    "medication_risk": [],
    "infection_risk": [],
    "bleeding_risk": [],
    "trigger_factors": []
  },
  "diagnosis_axes": [
    {
      "axis_id": "short_snake_case_axis_id",
      "status": "confirmed | suspected | missing_evidence",
      "clinical_role": "current_problem | background_condition | secondary",
      "priority": "routine | high | red_flag",
      "closure_requirement": "final 前必须完成的诊断、检查、转诊或安全处置闭环",
      "evidence": ["病例中真实存在的证据短语"],
      "missing_evidence": ["仍需补齐的关键证据"],
      "candidate_official_names": ["标准疾病名"],
      "exam_intents": ["检查目的"],
      "treatment_risks": ["通用风险标签"]
    }
  ],
  "risk_summary": "一句话说明 final 前最需要闭合的风险"
}
"""


EVALUATION_REFLECTION_PROMPT = """训练病例已经完成，并收到了评估结果。请根据患者对话记录和评估明细写一段可复用的简短反思。

患者对话记录：
$chat_history

评估明细：
$evaluation_details

要求：
- reflection 是唯一会写入 memory 的字段。
- reflection.profile 用 1-2 句话概括患者简介，重点保留从对话记录中可复用的症状线索、关键背景或特殊风险。
- 评估明细只包含 diagnosisDetail、examinationDetail、treatmentDetail；反思时分别对照其中的 submitted/ordered、expected/reference、matched、reasoning 和分数信息。
- 诊断、检查、治疗反思要写清楚本次遗漏或做对的要点。
- 内容要短，适合后续病例作为参考摘要，不要复制长篇参考治疗原文。
- 不得复制完整 expected 检查列表；只能把它压缩为“症状/体征 -> 检查目的 -> 标准检查类别或关键叶子名”的短策略。
- 不得复制 reference 治疗原文；只能保留可解释的治疗目标、禁忌和安全约束。
- 不要写入病例 ID、完整患者对话、完整标准答案或可被当作单病例答案索引的表述。

请只返回 JSON：
{
  "reflection": {
    "profile": "患者简介",
    "diagnosis_reflection": "诊断方面的经验",
    "examination_reflection": "检查选择方面的经验",
    "treatment_reflection": "治疗方案方面的经验",
    "future_strategy": "以后遇到类似病例的简短策略"
  }
}
"""


def format_prompt(template: str, variables: Dict[str, Any]) -> str:
    """Format a prompt with JSON-safe values."""
    prepared = {}
    for key, value in variables.items():
        if isinstance(value, str):
            prepared[key] = value
        else:
            prepared[key] = json.dumps(value, ensure_ascii=False, indent=2)
    return Template(template).safe_substitute(prepared)

# De-anchored second opinion: the reviewer never sees the first diagnosis, the
# draft treatment or the first reasoning, so it cannot simply ratify them.
REQUIRED_RUNTIME_PROMPT_KEYS = frozenset(
    {
        "DOCTOR_SYSTEM_PROMPT",
        "JSON_REPAIR_SYSTEM_PROMPT",
        "DEPARTMENT_PROMPT",
        "DIAGNOSTIC_AXIS_CONSULT_PROMPT",
        "DIAGNOSTIC_CONTEXT_PROMPT",
        "DIAGNOSIS_INDEPENDENT_REVIEW_PROMPT",
        "DISEASE_CANDIDATE_PROMPT",
        "DISEASE_AND_TREATMENT_PROMPT",
        "EVALUATION_REFLECTION_PROMPT",
        "TREATMENT_REVIEW_PROMPT",
    }
)


DIAGNOSIS_INDEPENDENT_REVIEW_PROMPT = """请作为独立诊断复核医生，在不知道任何既有诊断结论的前提下重新判断最可能的诊断。

只允许从给定的官方候选疾病名称中选择，不得自造名称。

结构化病例摘要：
$case_summary

已完成检查与结果：
$examination_results

官方候选疾病名称：
$official_candidates

可引用证据目录（ID 对应的临床事实）：
$evidence_catalog

候选疾病与可支持其的证据 ID：
$candidate_evidence

请只输出 JSON：
{
  "recommended_diagnosis": "候选列表中的一个标准疾病名称",
  "supporting_evidence_ids": ["允许引用的证据 ID"],
  "contradicting_evidence_ids": ["允许引用的证据 ID"],
  "confidence": "high | medium | low"
}
"""
