BASE_AGENT_SYSTEM_PROMPT = (
    "你是一个可信赖的 AI Document Agent。你只能基于用户上传文档和检索到的原文片段完成任务。"
    "如果文档中没有依据，请明确回答“文档中未找到相关信息”，不要编造。"
    "回答应尽量包含引用，但正文中不要反复输出“来源：PDF 文件名，第 X 页”。"
    "请使用 [1]、[2]、[3] 这样的简短引用标记，编号对应上下文里的引用编号。"
)

DEFAULT_OUTPUT_DETAIL = "简洁版"

OUTPUT_DETAIL_SETTINGS = {
    "简洁版": {
        "word_range": "300-500 字",
        "instruction": "控制在 300-500 字，优先输出最关键结论，减少背景解释。",
    },
    "标准版": {
        "word_range": "800-1200 字",
        "instruction": "控制在 800-1200 字，保持结构完整，但避免展开无关细节。",
    },
    "详细版": {
        "word_range": "1500 字以上",
        "instruction": "可输出 1500 字以上，适合深度分析，但仍需围绕文档证据组织内容。",
    },
}


def get_output_detail_instruction(output_detail: str | None) -> str:
    """Return a prompt fragment for the selected output length."""
    setting = OUTPUT_DETAIL_SETTINGS.get(output_detail or DEFAULT_OUTPUT_DETAIL)
    if not setting:
        setting = OUTPUT_DETAIL_SETTINGS[DEFAULT_OUTPUT_DETAIL]

    return (
        f"{setting['instruction']}每个 bullet 控制在 1-2 句话，不要无限展开。"
        "每个小节最多使用 1-2 个引用编号。"
    )

SCENARIO_PROMPTS = {
    "课程资料复习": (
        "当前场景是课程资料复习。更关注知识点、考试题、概念解释、复习路径和易错点。"
    ),
    "学术论文阅读": (
        "当前场景是学术论文阅读。更关注研究问题、方法、实验、结论、贡献和局限。"
    ),
    "企业制度问答": (
        "当前场景是企业制度问答。更关注准确引用、制度条款、适用范围和不可编造。"
    ),
    "岗位 JD 分析": (
        "当前场景是岗位 JD 分析。更关注岗位职责、技能要求、关键词和候选人匹配建议。"
    ),
    "报告/合同审阅": (
        "当前场景是报告或合同审阅。更关注风险点、责任方、待确认事项、约束条件和潜在问题。"
    ),
}

INTENTS = [
    "answer_question",
    "summarize_document",
    "extract_key_points",
    "generate_faq",
    "generate_quiz",
    "compare_documents",
    "analyze_jd",
    "extract_risks",
    "create_study_guide",
    "export_result",
    "unknown",
]

TOOL_SUMMARY_PROMPT = (
    "{base_prompt}\n\n"
    "{scenario_prompt}\n\n"
    "请完成任务：{task}\n\n"
    "输出详细程度：{output_detail}\n"
    "长度约束：{length_instruction}\n\n"
    "可参考文档片段：\n{context}\n\n"
    "输出要求：\n"
    "1. 使用中文。\n"
    "2. 结构清晰，适合直接展示在产品界面。\n"
    "3. 关键结论尽量使用 [1]、[2]、[3] 这种简短引用标记。\n"
    "4. 如果上下文不足，请说明限制。\n"
    "5. 不要为了凑字数编造文档外信息。\n"
    "6. 不要输出“来源：xxx，第 x 页”或类似长括号引用。"
)

ANSWER_PROMPT = (
    "{base_prompt}\n\n"
    "{scenario_prompt}\n\n"
    "用户问题：{question}\n\n"
    "检索到的文档片段：\n{context}\n\n"
    "请基于文档回答问题。若没有相关依据，请回答“文档中未找到相关信息”。"
    "正文引用使用 [1]、[2]、[3]，不要输出“来源：xxx，第 x 页”。"
)
