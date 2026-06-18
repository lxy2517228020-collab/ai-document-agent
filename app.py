import copy
import hashlib
import html
import json
import re

import streamlit as st

from agent import DocumentAgent
from export_utils import history_to_markdown, history_to_txt
from memory import (
    FEEDBACK_LABELS,
    add_task_history,
    average_response_time,
    clear_history,
    init_session_state,
    set_feedback,
)
from prompts import OUTPUT_DETAIL_SETTINGS, SCENARIO_PROMPTS
from rag import (
    DEFAULT_MODEL_MODE,
    MODEL_MODES,
    build_knowledge_base,
    delete_document,
    empty_knowledge_base,
    fallback_model_message,
    get_api_key,
    get_embedding_model_name,
    get_model_config,
    get_model_name,
    list_available_generation_models,
    rebuild_knowledge_base,
    uploaded_files_signature,
)


def apply_styles() -> None:
    """Small product-style polish while keeping Streamlit simple."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 4rem;
            max-width: 1320px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        .doc-card {
            border: 1px solid #e6e8eb;
            border-radius: 8px;
            padding: 0.8rem;
            background: #ffffff;
            margin-bottom: 0.7rem;
        }
        .muted {
            color: #667085;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .source-card {
            border-left: 3px solid #4f46e5;
            padding: 0.55rem 0.75rem;
            margin: 0.55rem 0;
            background: #f8fafc;
        }
        .source-snippet {
            color: #475467;
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .citation {
            color: #2563eb;
            font-size: 0.75em;
            font-weight: 600;
            cursor: pointer;
            display: inline-block;
            margin-left: 2px;
            position: relative;
            vertical-align: super;
        }
        .citation:hover {
            color: #1d4ed8;
            text-decoration: underline;
        }
        .citation:hover::after {
            background: #111827;
            border-radius: 6px;
            bottom: 1.45em;
            color: #ffffff;
            content: attr(data-tooltip);
            font-size: 0.78rem;
            font-weight: 400;
            left: 0;
            line-height: 1.45;
            max-width: 360px;
            min-width: 240px;
            padding: 0.45rem 0.55rem;
            position: absolute;
            text-align: left;
            text-decoration: none;
            white-space: normal;
            z-index: 9999;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render API status, history, and metrics in the sidebar."""
    with st.sidebar:
        st.subheader("运行状态")
        st.selectbox(
            "模型模式",
            options=list(MODEL_MODES.keys()),
            key="model_mode",
            help=(
                "快速模式更适合演示；稳定模式默认推荐；高质量模式结果更完整，"
                "但可能更慢、更容易触发限流。"
            ),
        )
        st.selectbox(
            "输出详细程度",
            options=list(OUTPUT_DETAIL_SETTINGS.keys()),
            key="detail_level",
            help="用于平衡回答完整度和响应速度。简洁版更快，详细版更完整但耗时更长。",
        )
        st.caption("简洁版：适合快速演示，响应更快")
        st.caption("标准版：内容更完整")
        st.caption("详细版：适合深度分析，但响应更慢")
        st.session_state.output_detail = st.session_state.detail_level
        st.checkbox("启用快速路由", key="fast_routing")
        st.checkbox("启用深度评估", key="deep_evaluation")

        if get_api_key():
            st.success("已读取 GOOGLE_API_KEY")
        else:
            st.error("未找到 GOOGLE_API_KEY")
            st.info("本地使用 .env；Streamlit Cloud 使用 Secrets。")

        st.caption(f"当前模型：{get_model_name()}")
        st.caption(f"当前模式：{st.session_state.model_mode}")
        detail = OUTPUT_DETAIL_SETTINGS.get(st.session_state.detail_level, {})
        st.caption(f"输出详细程度：{st.session_state.detail_level}（{detail.get('word_range', '-')}）")
        st.caption(f"Embedding：{get_embedding_model_name()}")
        st.caption(f"快速路由：{'开启' if st.session_state.fast_routing else '关闭'}")
        st.caption(f"深度评估：{'开启' if st.session_state.deep_evaluation else '关闭'}")

        with st.expander("模型调试", expanded=False):
            model_config = get_model_config()
            st.json(model_config)
            if st.button("检测可用模型", use_container_width=True):
                with st.spinner("正在调用 Gemini list_models..."):
                    try:
                        st.session_state.available_generation_models = list_available_generation_models()
                    except Exception as exc:
                        st.session_state.available_generation_models = []
                        st.error(f"检测失败：{exc}")

            if st.session_state.get("available_generation_models"):
                st.write(st.session_state.available_generation_models)

            st.checkbox("显示 citation debug", key="show_citation_debug")

        st.divider()
        st.subheader("数据看板")
        stats = st.session_state.stats
        kb = st.session_state.knowledge_base
        col1, col2 = st.columns(2)
        col1.metric("总提问", stats["total_tasks"])
        col2.metric("工具调用", stats["tool_calls"])
        col1.metric("文档数", len(kb.get("file_records", [])))
        col2.metric("文本块", len(kb.get("chunks", [])))
        col1.metric("Gemini 调用", stats["gemini_calls"])
        col2.metric("缓存命中", stats["cache_hits"])
        col1.metric("Retry 次数", stats["retry_count"])
        col2.metric("没帮助", stats["unhelpful"])
        st.metric("平均响应时间", f"{average_response_time(stats):.2f}s")

        st.divider()
        st.subheader("历史任务")
        if not st.session_state.task_history:
            st.caption("暂无历史任务。")
        else:
            for item in reversed(st.session_state.task_history[-8:]):
                st.caption(f"{item['created_at']} · {item['intent']}")
                st.write(item["user_task"][:60])

        if st.button("清空历史", use_container_width=True, disabled=not st.session_state.task_history):
            clear_history(st)
            st.rerun()


def render_dashboard() -> None:
    """Render top-level product metrics."""
    stats = st.session_state.stats
    kb = st.session_state.knowledge_base
    cols = st.columns(6)
    cols[0].metric("文档数量", len(kb.get("file_records", [])))
    cols[1].metric("文本块数量", len(kb.get("chunks", [])))
    cols[2].metric("任务次数", stats["total_tasks"])
    cols[3].metric("工具调用", stats["tool_calls"])
    cols[4].metric("Gemini 调用", stats["gemini_calls"])
    cols[5].metric("缓存命中", stats["cache_hits"])


def cache_key_for_task(user_task: str, scenario: str) -> str:
    """Build a stable cache key for identical task + document + mode."""
    payload = {
        "task": user_task.strip(),
        "scenario": scenario,
        "file_signature": st.session_state.knowledge_base.get("file_signature"),
        "model_mode": st.session_state.get("model_mode"),
        "detail_level": st.session_state.get("detail_level"),
        "fast_routing": st.session_state.get("fast_routing"),
        "deep_evaluation": st.session_state.get("deep_evaluation"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def handle_upload(uploaded_files) -> None:
    """Build knowledge base automatically after upload."""
    if not uploaded_files:
        return

    current_signature = uploaded_files_signature(uploaded_files)
    kb = st.session_state.knowledge_base
    if current_signature == kb.get("file_signature"):
        return

    if not get_api_key():
        st.warning("请先配置 GOOGLE_API_KEY，再上传 PDF 建库。")
        return

    cache_key = repr(current_signature)
    if cache_key in st.session_state.knowledge_base_cache:
        st.session_state.knowledge_base = st.session_state.knowledge_base_cache[cache_key]
        st.success("已复用缓存中的知识库和向量库，跳过重复 Embedding。")
        return

    st.info("首次上传文档会较慢，因为需要解析 PDF、生成 Embedding 并建立向量库；后续提问会复用向量库。")

    with st.spinner("正在解析 PDF、切分文本、生成 Gemini Embedding 并建立 FAISS 知识库..."):
        try:
            st.session_state.knowledge_base = build_knowledge_base(uploaded_files)
        except Exception as exc:
            st.error(
                "知识库建立失败。若看到 503/429/timeout，可能是 Gemini 模型高峰期或免费额度繁忙；"
                "请稍后重试，或切换到快速模式。"
            )
            st.caption(str(exc))
        else:
            st.session_state.knowledge_base_cache[cache_key] = st.session_state.knowledge_base
            st.success("知识库已建立。现在可以让 Agent 执行文档任务。")


def render_knowledge_base_tab(uploaded_files) -> None:
    """Render knowledge base management controls."""
    st.subheader("知识库管理")
    st.caption("单个 PDF 建议控制在 Streamlit 上传限制内。扫描版 PDF 需要先做 OCR，否则可能无法提取文本。")

    handle_upload(uploaded_files)
    kb = st.session_state.knowledge_base
    records = kb.get("file_records", [])

    if not records:
        st.info("请先上传 PDF，系统会自动建立 FAISS 知识库。")
        return

    confirm_changes = st.checkbox("我确认要执行删除或清空知识库操作")

    for record in records:
        col_info, col_action = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"""
                <div class="doc-card">
                  <strong>{record["name"]}</strong>
                  <div class="muted">
                    页数：{record["pages"]} · 文本块：{record["chunks"]} · 大小：{record["size"]} bytes<br>
                    上传时间：{record["uploaded_at"]} · 建库状态：{record["status"]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_action:
            if st.button(
                "删除",
                key=f"delete_{record['name']}",
                disabled=not confirm_changes,
                use_container_width=True,
            ):
                st.session_state.knowledge_base = delete_document(kb, record["name"])
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("重新生成向量库", use_container_width=True):
            with st.spinner("正在重新生成 FAISS 向量库..."):
                st.session_state.knowledge_base = rebuild_knowledge_base(kb)
            st.success("向量库已重新生成。")
            st.rerun()
    with col2:
        if st.button("清空知识库", disabled=not confirm_changes, use_container_width=True):
            st.session_state.knowledge_base = empty_knowledge_base()
            st.success("知识库已清空。")
            st.rerun()


def compact_debug_value(value, max_chars: int = 500):
    """Return a compact debug-safe representation."""
    text = repr(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def read_source_value(source, keys: list[str], metadata_keys: list[str] | None = None):
    """Read one source value from dicts, objects, or LangChain Document metadata."""
    metadata_keys = metadata_keys or keys

    if isinstance(source, dict):
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value

        metadata = source.get("metadata")
        if isinstance(metadata, dict):
            for key in metadata_keys:
                value = metadata.get(key)
                if value not in (None, ""):
                    return value

    for key in keys:
        value = getattr(source, key, None)
        if value not in (None, ""):
            return value

    metadata = getattr(source, "metadata", None)
    if isinstance(metadata, dict):
        for key in metadata_keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return value

    return None


def normalize_source(source) -> dict:
    """Normalize dicts and LangChain Document objects into the citation schema."""
    if source is None:
        return {}

    file_name = read_source_value(
        source,
        ["file_name", "filename", "source", "doc_name", "file"],
        metadata_keys=["file_name", "filename", "source", "doc_name", "file"],
    )
    page = read_source_value(
        source,
        ["page", "page_number", "page_num"],
        metadata_keys=["page", "page_number", "page_num"],
    )
    snippet = read_source_value(
        source,
        ["snippet", "content", "text", "page_content"],
        metadata_keys=["snippet", "content", "text", "page_content"],
    )

    return {
        "file_name": str(file_name or "未知文件"),
        "page": str(page if page not in (None, "") else "未知页码"),
        "snippet": str(snippet or "暂无片段"),
    }


def normalize_sources(sources, dedupe: bool = False) -> list[dict]:
    """Normalize sources while optionally deduping by file + page."""
    normalized_sources = []
    seen = set()
    if not isinstance(sources, (list, tuple)):
        sources = []

    for source in sources:
        normalized = normalize_source(source)
        if not normalized:
            continue

        key = (normalized["file_name"], normalized["page"])
        if dedupe and key in seen:
            continue

        normalized_sources.append(normalized)
        seen.add(key)

    return normalized_sources


def truncate_snippet(text: str, max_chars: int = 140) -> str:
    """Keep citation snippets readable in the UI."""
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def citation_index_for_legacy(file_name: str, page: str, sources: list[dict]) -> int:
    """Map old-style source text to the current source number."""
    file_name = file_name.strip()
    page = page.strip()
    for index, source in enumerate(sources, start=1):
        same_file = file_name in source["file_name"] or source["file_name"] in file_name
        same_page = str(source["page"]) == page
        if same_file and same_page:
            return index
    return 1 if sources else 0


def replace_legacy_citations(answer: str, sources: list[dict]) -> str:
    """Convert ugly inline source parentheses into compact numeric markers."""
    pattern = re.compile(
        r"[（(]\s*来源[:：]\s*([^，,）)]+)\s*[，,]\s*第?\s*([0-9]+)\s*页\s*[）)]"
    )

    def replace(match: re.Match) -> str:
        index = citation_index_for_legacy(match.group(1), match.group(2), sources)
        return f"[{index}]" if index else ""

    return pattern.sub(replace, answer)


def ensure_answer_has_citation(answer: str, sources: list[dict]) -> str:
    """Add one compact citation when the model forgot all markers."""
    if not sources or re.search(r"\[\d+(?:\s*,\s*\d+)*\]", answer) or "文档中未找到相关信息" in answer:
        return answer

    lines = answer.splitlines()
    for index, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            lines[index] = f"{line.rstrip()} [1]"
            return "\n".join(lines)

    return f"{answer.rstrip()} [1]"


def citation_sup_html(index: int, sources: list[dict]) -> str:
    """Build one cited superscript with a source tooltip."""
    if index < 1 or index > len(sources):
        tooltip = html.escape("未找到对应来源", quote=True)
        return f'<sup class="citation" title="{tooltip}" data-tooltip="{tooltip}">{index}</sup>'

    source = sources[index - 1]
    tooltip = (
        f"文件：{source['file_name']}｜"
        f"页码：第 {source['page']} 页｜"
        f"片段：{truncate_snippet(source['snippet'], max_chars=120)}"
    )
    tooltip = html.escape(tooltip, quote=True)
    return f'<sup class="citation" title="{tooltip}" data-tooltip="{tooltip}">{index}</sup>'


def build_citation_debug(raw_sources, normalized_sources: list[dict], used_indexes: list[int]) -> dict:
    """Build citation debug info for diagnosing source mapping."""
    first_source = None
    if isinstance(raw_sources, (list, tuple)) and raw_sources:
        first_source = raw_sources[0]

    mappings = []
    for index in used_indexes:
        source = normalized_sources[index - 1] if 1 <= index <= len(normalized_sources) else None
        mappings.append(
            {
                "citation_number": index,
                "source_index": index - 1,
                "resolved": source or "未找到对应来源",
            }
        )

    return {
        "sources_type": type(raw_sources).__name__,
        "sources_length": len(raw_sources) if isinstance(raw_sources, (list, tuple)) else 0,
        "sources_0_structure": compact_debug_value(first_source) if first_source is not None else None,
        "normalized_sources_length": len(normalized_sources),
        "citation_mappings": mappings,
    }


def render_answer_with_citations(answer: str, sources: list[dict]) -> None:
    """Render answer markdown and turn [1] / [1, 3] markers into tooltip superscripts."""
    normalized_sources = normalize_sources(sources)
    answer = replace_legacy_citations(answer, normalized_sources)
    answer = ensure_answer_has_citation(answer, normalized_sources)
    used_indexes = []

    def replace_marker(match: re.Match) -> str:
        raw_numbers = match.group(1)
        superscripts = []

        for raw_number in re.split(r"\s*,\s*", raw_numbers):
            try:
                index = int(raw_number)
            except ValueError:
                superscripts.append(f"[{raw_number}]")
                continue

            used_indexes.append(index)
            superscripts.append(citation_sup_html(index, normalized_sources))

        return "".join(superscripts)

    html_answer = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", replace_marker, answer)
    st.markdown(html_answer, unsafe_allow_html=True)

    if st.session_state.get("show_citation_debug"):
        with st.expander("Citation Debug", expanded=False):
            st.json(build_citation_debug(sources, normalized_sources, used_indexes))


def render_source_card(index: int, source: dict) -> None:
    """Render one citation source card."""
    st.markdown(
        f"""
        <div class="source-card">
          <strong>来源 {index}</strong>
          <div class="source-snippet">
            <div>文件：{html.escape(str(source["file_name"]))}</div>
            <div>页码：第 {html.escape(str(source["page"]))} 页</div>
            <div>原文片段：{html.escape(truncate_snippet(source["snippet"]))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], expanded: bool = True, answer_text: str = "") -> None:
    """Render citation sources."""
    normalized_sources = normalize_sources(sources, dedupe=True)

    if not normalized_sources:
        if "来源" in answer_text:
            st.info("本次未返回结构化引用来源。")
        return

    visible_sources = normalized_sources[:5]
    hidden_sources = normalized_sources[5:]

    with st.expander("引用来源", expanded=expanded):
        for index, source in enumerate(visible_sources, start=1):
            render_source_card(index, source)

    if hidden_sources:
        with st.expander("查看更多引用来源", expanded=False):
            for offset, source in enumerate(hidden_sources, start=6):
                render_source_card(offset, source)


def build_plain_text_result(record: dict) -> str:
    """Build a copy-friendly plain text answer with citation sources."""
    answer = str(record.get("answer", "")).strip()
    sources = normalize_sources(record.get("sources", []), dedupe=True)
    lines = ["Agent 回答正文", "", answer, "", "引用来源"]

    if not sources:
        lines.append("本次未返回结构化引用来源。")
    else:
        for index, source in enumerate(sources, start=1):
            lines.extend(
                [
                    f"{index}. 文件：{source['file_name']}",
                    f"   页码：第 {source['page']} 页",
                    f"   原文片段：{truncate_snippet(source['snippet'], max_chars=180)}",
                ]
            )

    return "\n".join(lines).strip()


def build_markdown_result(record: dict) -> str:
    """Build a Markdown export for one Agent result."""
    answer = str(record.get("answer", "")).strip()
    sources = normalize_sources(record.get("sources", []), dedupe=True)
    lines = ["# Agent 执行结果", "", "## Agent 回答正文", "", answer, "", "## 引用来源"]

    if not sources:
        lines.append("本次未返回结构化引用来源。")
    else:
        for index, source in enumerate(sources, start=1):
            lines.extend(
                [
                    f"### 来源 {index}",
                    f"- 文件：{source['file_name']}",
                    f"- 页码：第 {source['page']} 页",
                    f"- 原文片段：{truncate_snippet(source['snippet'], max_chars=220)}",
                    "",
                ]
            )

    return "\n".join(lines).strip()


def render_copy_export_area(record: dict, key_prefix: str) -> None:
    """Render a copy-friendly plain text result and single-result downloads."""
    plain_text_result = build_plain_text_result(record)
    markdown_result = build_markdown_result(record)

    with st.expander("复制纯文本结果", expanded=False):
        st.text_area(
            "纯文本结果",
            value=plain_text_result,
            height=240,
            key=f"{key_prefix}_plain_text",
            help="点击文本框后可以使用 Command + A / Command + C 复制，不需要选中页面正文。",
        )
        col_txt, col_md = st.columns(2)
        with col_txt:
            st.download_button(
                label="下载 TXT",
                data=plain_text_result,
                file_name="agent_result.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"{key_prefix}_download_txt",
            )
        with col_md:
            st.download_button(
                label="下载 Markdown",
                data=markdown_result,
                file_name="agent_result.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"{key_prefix}_download_md",
            )


def render_agent_process(result: dict) -> None:
    """Visualize router, plan, tool calls, and evaluation."""
    with st.expander("Agent 执行过程", expanded=True):
        st.markdown(f"**Intent**：`{result['router_result']['intent']}`")
        st.json(result["router_result"])

        st.markdown("**执行计划**")
        for step in result["plan"]:
            st.write(f"{step['step']}. {step['description']} · 工具：`{step['tool']}`")

        st.markdown("**工具调用记录**")
        if result["tool_calls"]:
            for call in result["tool_calls"]:
                st.write(f"{call['step']}. `{call['tool']}`")
                st.caption(call.get("summary", ""))
        else:
            st.caption("本次没有调用工具。")

        st.markdown("**可信度评估**")
        st.json(result["evaluation"])
        st.markdown("**模型使用情况**")
        st.json(
            {
                "原始选择模型": result.get("selected_model"),
                "实际使用模型": result.get("actual_model"),
                "是否 fallback": result.get("model_fallback", False),
                "fallback 模型": result.get("fallback_model"),
                "fallback 原因": result.get("fallback_reason"),
            }
        )
        if result.get("retry_events"):
            st.markdown("**Retry 记录**")
            st.json(result["retry_events"])
        if result.get("errors"):
            st.markdown("**失败原因**")
            st.json(result["errors"])
        st.caption(
            f"响应时间：{result['response_time']} 秒 · "
            f"Gemini 调用：{result.get('gemini_calls', 0)} · "
            f"Retry：{result.get('retry_count', 0)} · "
            f"缓存命中：{'是' if result.get('cache_hit') else '否'} · "
            f"输出详细程度：{result.get('detail_level') or result.get('output_detail') or st.session_state.get('detail_level', '-')}"
        )


def render_feedback(record: dict) -> None:
    """Render feedback buttons for one answer."""
    current = record.get("feedback")
    cols = st.columns(6)

    for index, (feedback_key, label) in enumerate(FEEDBACK_LABELS.items()):
        with cols[index]:
            if st.button(
                label,
                key=f"feedback_{record['id']}_{feedback_key}",
                type="primary" if current == feedback_key else "secondary",
                use_container_width=True,
            ):
                set_feedback(st, record["id"], feedback_key)
                st.rerun()


def render_result(record: dict, expanded_process: bool = False) -> None:
    """Render one history item."""
    with st.chat_message("user"):
        st.write(record["user_task"])

    with st.chat_message("assistant"):
        render_answer_with_citations(record["answer"], record.get("sources", []))
        metric_cols = st.columns(5)
        metric_cols[0].metric("响应时间", f"{record.get('response_time', 0):.2f}s")
        metric_cols[1].metric("Gemini 调用", record.get("gemini_calls", 0))
        metric_cols[2].metric("Retry", record.get("retry_count", 0))
        metric_cols[3].metric("缓存命中", "是" if record.get("cache_hit") else "否")
        metric_cols[4].metric(
            "输出详细程度",
            record.get("detail_level") or record.get("output_detail") or st.session_state.detail_level,
        )
        model_cols = st.columns(3)
        model_cols[0].metric("原始模型", record.get("selected_model") or "-")
        model_cols[1].metric("实际模型", record.get("actual_model") or "-")
        model_cols[2].metric("模型 Fallback", "是" if record.get("model_fallback") else "否")
        if record.get("model_fallback"):
            st.info(fallback_model_message())
        if record.get("errors") or "当前 Gemini 模型繁忙" in record.get("answer", ""):
            st.warning(
                "Gemini 当前可能处于高峰期、免费额度繁忙或临时限流。"
                "建议稍后重试、切换快速模式，或换用轻量 Flash 模型。"
            )
        render_sources(record.get("sources", []), expanded=False, answer_text=record.get("answer", ""))
        render_copy_export_area(record, key_prefix=f"latest_{record['id']}")
        if expanded_process:
            render_agent_process(record)
        else:
            with st.expander("Agent 执行过程", expanded=False):
                st.markdown(f"**Intent**：`{record['intent']}`")
                for call in record.get("tool_calls", []):
                    st.write(f"{call['step']}. `{call['tool']}`：{call.get('summary', '')}")
        render_feedback(record)


def render_agent_tab(scenario: str) -> None:
    """Render the Agent task entry and latest result."""
    st.subheader("让 Agent 帮你完成任务")
    st.caption("可以输入普通问题，也可以输入任务型指令，例如：总结 PDF、生成 FAQ、对比两份文档、提取风险点、分析岗位 JD。")

    user_task = st.text_area(
        "自然语言任务",
        placeholder="例如：帮我把这份论文整理成面试讲解稿，并生成 10 个复习问题",
        height=110,
    )
    run_clicked = st.button("运行 Agent", type="primary", use_container_width=True)

    if run_clicked:
        if not user_task.strip():
            st.warning("请输入一个任务。")
        else:
            with st.spinner("Agent 正在识别意图、规划步骤并调用工具..."):
                try:
                    task_key = cache_key_for_task(user_task, scenario)
                    if task_key in st.session_state.agent_result_cache:
                        result = copy.deepcopy(st.session_state.agent_result_cache[task_key])
                        result["cache_hit"] = True
                        result["response_time"] = 0.0
                        result["gemini_calls"] = 0
                        result["retry_count"] = 0
                        result["retry_events"] = []
                        result["errors"] = []
                        result["model_fallback"] = False
                    else:
                        agent = DocumentAgent(
                            st.session_state.knowledge_base,
                            scenario,
                            fast_routing=st.session_state.fast_routing,
                            deep_evaluation=st.session_state.deep_evaluation,
                            detail_level=st.session_state.detail_level,
                        )
                        result = agent.execute(user_task.strip(), st.session_state.last_agent_result)
                        st.session_state.agent_result_cache[task_key] = copy.deepcopy(result)
                except Exception as exc:
                    st.error(
                        "Agent 执行失败。若遇到 503 UNAVAILABLE、429 rate limit 或 timeout，"
                        "可能是 Gemini 高峰期繁忙。请稍后重试，或切换到快速模式。"
                    )
                    st.caption(str(exc))
                else:
                    add_task_history(st, result)
                    st.rerun()

    if st.session_state.last_agent_result:
        st.divider()
        st.subheader("最近一次执行结果")
        render_result(st.session_state.last_agent_result, expanded_process=True)
    else:
        st.info("上传 PDF 后，在这里输入任务，Agent 会自动识别 intent、生成计划并调用工具。")


def render_history_tab() -> None:
    """Render task history and export actions."""
    st.subheader("历史记录与导出")

    if not st.session_state.task_history:
        st.info("还没有历史任务。")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "导出 Markdown",
            data=history_to_markdown(st.session_state.task_history),
            file_name="document_agent_history.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "导出 TXT",
            data=history_to_txt(st.session_state.task_history),
            file_name="document_agent_history.txt",
            mime="text/plain",
            use_container_width=True,
        )

    for record in reversed(st.session_state.task_history):
        with st.expander(f"{record['created_at']} · {record['intent']} · {record['user_task'][:50]}", expanded=False):
            render_answer_with_citations(record["answer"], record.get("sources", []))
            render_sources(record.get("sources", []), expanded=False, answer_text=record.get("answer", ""))
            render_copy_export_area(record, key_prefix=f"history_{record['id']}")
            render_agent_process(record)


def main() -> None:
    st.set_page_config(page_title="AI Document Agent", page_icon="📚", layout="wide")
    init_session_state(st)
    apply_styles()

    st.title("AI Document Agent")
    st.caption("Gemini API + LangChain + FAISS + Streamlit · 支持 RAG 检索、任务规划、工具调用和引用溯源")

    render_sidebar()

    top_left, top_right = st.columns([2, 1])
    with top_left:
        uploaded_files = st.file_uploader(
            "上传 PDF 文档",
            type=["pdf"],
            accept_multiple_files=True,
            help="支持一个或多个 PDF。上传完成后会自动建立 FAISS 知识库。",
        )
    with top_right:
        scenario = st.selectbox("场景化模式", options=list(SCENARIO_PROMPTS.keys()))

    render_dashboard()
    st.divider()

    agent_tab, kb_tab, history_tab = st.tabs(["Agent 工作台", "知识库管理", "历史与导出"])
    with agent_tab:
        render_agent_tab(scenario)
    with kb_tab:
        render_knowledge_base_tab(uploaded_files)
    with history_tab:
        render_history_tab()


if __name__ == "__main__":
    main()
