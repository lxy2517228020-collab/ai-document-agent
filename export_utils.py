from datetime import datetime


def export_to_markdown(content: str) -> str:
    """Wrap one result as Markdown text."""
    return f"# Agent 执行结果\n\n导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content}"


def normalize_source(source: dict) -> dict:
    """Normalize old and new citation payloads before export."""
    if not isinstance(source, dict):
        return {}

    return {
        "file_name": source.get("file_name") or source.get("file") or "未知文件",
        "page": source.get("page", "未知页码"),
        "snippet": source.get("snippet", ""),
    }


def history_to_markdown(history: list[dict]) -> str:
    """Export task history as Markdown."""
    lines = ["# AI Document Agent 历史记录", ""]
    lines.append(f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for index, item in enumerate(history, start=1):
        lines.extend(
            [
                f"## 任务 {index}",
                "",
                f"**时间**：{item['created_at']}",
                "",
                f"**Intent**：`{item['intent']}`",
                "",
                f"**用户任务**：{item['user_task']}",
                "",
                "**执行计划**：",
            ]
        )

        for step in item.get("plan", []):
            lines.append(f"- {step['step']}. {step['description']}（工具：`{step['tool']}`）")

        lines.extend(["", "**最终结果**：", "", item["answer"], "", "**引用来源**："])

        for raw_source in item.get("sources", []):
            source = normalize_source(raw_source)
            if source:
                lines.append(f"- {source['file_name']}，第 {source['page']} 页：{source['snippet']}")

        feedback = item.get("feedback") or "未反馈"
        lines.extend(["", f"**反馈**：{feedback}", ""])

    return "\n".join(lines)


def history_to_txt(history: list[dict]) -> str:
    """Export task history as plain text."""
    lines = ["AI Document Agent 历史记录", f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]

    for index, item in enumerate(history, start=1):
        lines.extend(
            [
                f"任务 {index}",
                f"时间：{item['created_at']}",
                f"Intent：{item['intent']}",
                f"用户任务：{item['user_task']}",
                "执行计划：",
            ]
        )

        for step in item.get("plan", []):
            lines.append(f"- {step['step']}. {step['description']}（工具：{step['tool']}）")

        lines.extend(["最终结果：", item["answer"], "引用来源："])

        for raw_source in item.get("sources", []):
            source = normalize_source(raw_source)
            if source:
                lines.append(f"- {source['file_name']}，第 {source['page']} 页：{source['snippet']}")

        lines.extend([f"反馈：{item.get('feedback') or '未反馈'}", "-" * 60, ""])

    return "\n".join(lines)
