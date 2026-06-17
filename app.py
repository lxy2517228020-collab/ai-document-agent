import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

DEFAULT_CHAT_MODEL = "gemini-3.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"

SCENARIO_PROMPTS = {
    "课程资料复习": (
        "你是一个课程复习助教。回答时要帮助学生抓住考点、概念关系和易错点。"
        "如果适合，请用条目化方式说明，并给出可复习的重点。"
    ),
    "学术论文阅读": (
        "你是一个学术论文阅读助手。回答时关注研究问题、方法、实验、结论、局限和贡献。"
        "保持严谨，不要夸大论文没有明确支持的结论。"
    ),
    "企业制度问答": (
        "你是一个企业制度问答助手。回答时要准确、稳健，优先引用制度原文。"
        "如果上下文没有明确依据，请提醒用户需要查看原制度或咨询负责人。"
    ),
    "岗位 JD 分析": (
        "你是一个岗位 JD 分析助手。回答时关注职责、能力要求、关键词、候选人匹配点和准备建议。"
        "不要凭空补充 JD 中没有出现的硬性要求。"
    ),
}

QUICK_ACTIONS = {
    "总结全文": "请对知识库中的文档做一份结构化全文总结，包含核心主题、关键结论和适用场景。",
    "提取关键词": "请提取 15 到 25 个关键词，并按主题分组。每个关键词后给一句简短解释。",
    "生成知识点大纲": "请生成一份层级清晰的知识点大纲，适合快速复习或汇报。",
    "生成 10 个复习问题": "请生成 10 个复习问题，并附上简短参考答案。",
    "生成 FAQ": "请生成一份 FAQ，包含常见问题和清晰回答。",
}


def get_streamlit_secret(name: str) -> str | None:
    """Read one value from Streamlit Secrets when running on Streamlit Cloud."""
    try:
        value = st.secrets.get(name)
    except Exception:
        return None

    if value is None:
        return None

    return str(value)


def get_config_value(name: str, default: str | None = None) -> str | None:
    """Read configuration from Streamlit Secrets first, then environment variables."""
    return get_streamlit_secret(name) or os.getenv(name) or default


def get_api_key() -> str | None:
    """Read the Gemini API key from Streamlit Secrets, environment, or local .env."""
    api_key = get_config_value("GOOGLE_API_KEY")

    if api_key:
        # LangChain's Google integration also reads GOOGLE_API_KEY from os.environ.
        os.environ["GOOGLE_API_KEY"] = api_key

    return api_key


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model used to answer questions."""
    get_api_key()
    model_name = get_config_value("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    default_temperature = "1.0" if model_name.startswith("gemini-3") else "0.2"
    temperature = float(get_config_value("GEMINI_TEMPERATURE", default_temperature))
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embedding model used by FAISS."""
    get_api_key()
    model_name = get_config_value("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(model=model_name)


def init_session_state() -> None:
    """Initialize all Streamlit state used by the product prototype."""
    defaults = {
        "vector_store": None,
        "chunks": [],
        "documents": [],
        "file_signature": None,
        "file_records": [],
        "stats": {
            "files": 0,
            "pages": 0,
            "chunks": 0,
            "questions": 0,
            "helpful": 0,
            "unhelpful": 0,
        },
        "chat_history": [],
        "quick_outputs": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def uploaded_files_signature(uploaded_files) -> tuple[tuple[str, int], ...]:
    """Build a signature so Streamlit does not rebuild the index unnecessarily."""
    return tuple((uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files)


def apply_product_styles() -> None:
    """Add a restrained product-like layer over Streamlit defaults."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
        .kb-card {
            border: 1px solid #e6e8eb;
            border-radius: 8px;
            padding: 0.8rem;
            margin-bottom: 0.65rem;
            background: #ffffff;
        }
        .kb-meta {
            color: #667085;
            font-size: 0.82rem;
            line-height: 1.45;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_pdf_documents(uploaded_files) -> tuple[list[Document], list[dict]]:
    """Save uploaded PDFs temporarily, then load their text page by page."""
    documents: list[Document] = []
    file_records: list[dict] = []
    uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix or ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            for page in pages:
                # Keep source metadata on every page so answers can cite it later.
                page.metadata["source"] = uploaded_file.name
                page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
                documents.append(page)

            file_records.append(
                {
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "pages": len(pages),
                    "uploaded_at": uploaded_at,
                    "status": "已建库",
                }
            )
        except Exception:
            file_records.append(
                {
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "pages": 0,
                    "uploaded_at": uploaded_at,
                    "status": "建库失败",
                }
            )
            raise
        finally:
            Path(temp_path).unlink(missing_ok=True)

    return documents, file_records


def split_documents(documents: list[Document]) -> list[Document]:
    """Split PDF pages into chunks that are suitable for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(uploaded_files) -> tuple[FAISS, list[Document], list[Document], list[dict]]:
    """Load PDFs, split text, embed chunks, and create a FAISS vector store."""
    documents, file_records = load_pdf_documents(uploaded_files)
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("没有从 PDF 中读取到可用文本。请确认 PDF 不是纯扫描图片。")

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, documents, chunks, file_records


def format_context(docs: list[Document]) -> str:
    """Format retrieved documents as source-aware context for the LLM."""
    context_parts = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知文件")
        page = doc.metadata.get("page", "未知页码")
        context_parts.append(
            f"[片段 {index}] 来源：{source}，页码：{page}\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def extract_message_text(message) -> str:
    """Return text from LangChain AIMessage across Gemini model response shapes."""
    text_attr = getattr(message, "text", None)
    if text_attr:
        text = text_attr() if callable(text_attr) else text_attr
        if text:
            return text

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(part for part in text_parts if part)

    return str(content)


def source_payload(docs: list[Document]) -> list[dict]:
    """Convert retrieved documents to simple dictionaries for display and export."""
    sources = []
    seen = set()

    for doc in docs:
        snippet = " ".join(doc.page_content.split())[:500]
        source = {
            "file": doc.metadata.get("source", "未知文件"),
            "page": doc.metadata.get("page", "未知页码"),
            "snippet": snippet,
        }
        key = (source["file"], source["page"], source["snippet"][:120])

        if key not in seen:
            sources.append(source)
            seen.add(key)

    return sources


def get_scenario_prompt(scenario: str) -> str:
    """Return the system prompt for the selected use case."""
    return SCENARIO_PROMPTS.get(scenario, SCENARIO_PROMPTS["课程资料复习"])


def answer_question(
    question: str,
    vector_store: FAISS,
    scenario: str,
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and ask Gemini to answer with citations."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{scenario_prompt}\n\n"
                "你也是一个严谨的 PDF 知识库问答助手。"
                "请只根据提供的上下文回答问题；如果上下文没有答案，请明确说不知道。"
                "回答必须展示引用来源，格式为：（来源：PDF 文件名，第 X 页）。"
                "不要编造页码、文件名或上下文中没有的事实。",
            ),
            (
                "human",
                "问题：{question}\n\n"
                "可参考的 PDF 原文片段：\n{context}\n\n"
                "请用中文回答，并在关键结论后标注来源。",
            ),
        ]
    )

    llm = get_chat_model()
    messages = prompt.format_messages(
        scenario_prompt=get_scenario_prompt(scenario),
        question=question,
        context=format_context(docs),
    )
    response = llm.invoke(messages)

    return extract_message_text(response), docs


def corpus_preview(chunks: list[Document], limit: int = 14000) -> str:
    """Create a compact corpus preview for document-level quick actions."""
    parts = []
    total_chars = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "未知文件")
        page = chunk.metadata.get("page", "未知页码")
        text = " ".join(chunk.page_content.split())
        block = f"来源：{source}，页码：{page}\n{text}"

        if total_chars + len(block) > limit:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)


def run_quick_action(action_name: str, scenario: str) -> str:
    """Run a document-level analysis action against the current knowledge base."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{scenario_prompt}\n\n"
                "你正在为一个 AI 文档知识库生成可直接使用的分析结果。"
                "请严格基于给定 PDF 片段输出；如果材料不足，请说明限制。"
                "必要时引用来源，格式为：（来源：PDF 文件名，第 X 页）。",
            ),
            (
                "human",
                "任务：{task}\n\n"
                "PDF 片段：\n{context}\n\n"
                "请用中文输出，结构清晰，适合直接放入知识库产品界面。",
            ),
        ]
    )
    llm = get_chat_model()
    messages = prompt.format_messages(
        scenario_prompt=get_scenario_prompt(scenario),
        task=QUICK_ACTIONS[action_name],
        context=corpus_preview(st.session_state.chunks),
    )
    return extract_message_text(llm.invoke(messages))


def set_feedback(message_index: int, value: str) -> None:
    """Record one feedback vote per assistant answer and keep counters consistent."""
    history = st.session_state.chat_history
    previous = history[message_index].get("feedback")

    if previous == value:
        return

    if previous == "helpful":
        st.session_state.stats["helpful"] -= 1
    elif previous == "unhelpful":
        st.session_state.stats["unhelpful"] -= 1

    history[message_index]["feedback"] = value

    if value == "helpful":
        st.session_state.stats["helpful"] += 1
    else:
        st.session_state.stats["unhelpful"] += 1


def render_sources(sources: list[dict], expanded: bool = True) -> None:
    """Display cited PDF snippets."""
    if not sources:
        return

    with st.expander("引用来源", expanded=expanded):
        for source in sources:
            st.markdown(
                f"""
                <div class="source-card">
                  <strong>{source["file"]}，第 {source["page"]} 页</strong>
                  <div class="source-snippet">{source["snippet"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_feedback(message_index: int) -> None:
    """Render helpful / unhelpful feedback buttons for one answer."""
    current = st.session_state.chat_history[message_index].get("feedback")
    col1, col2, col3 = st.columns([1, 1, 5])

    with col1:
        if st.button(
            "有帮助",
            key=f"helpful_{message_index}",
            type="primary" if current == "helpful" else "secondary",
        ):
            set_feedback(message_index, "helpful")
            st.rerun()

    with col2:
        if st.button(
            "没帮助",
            key=f"unhelpful_{message_index}",
            type="primary" if current == "unhelpful" else "secondary",
        ):
            set_feedback(message_index, "unhelpful")
            st.rerun()

    with col3:
        if current:
            st.caption("已记录反馈")


def markdown_export() -> str:
    """Export chat history as Markdown."""
    lines = ["# PDF 知识库问答记录", ""]
    lines.append(f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for index, item in enumerate(st.session_state.chat_history, start=1):
        lines.extend(
            [
                f"## 问答 {index}",
                "",
                f"**时间**：{item['created_at']}",
                "",
                f"**使用场景**：{item['scenario']}",
                "",
                f"**问题**：{item['question']}",
                "",
                "**回答**：",
                "",
                item["answer"],
                "",
                "**引用来源**：",
            ]
        )

        for source in item["sources"]:
            lines.append(f"- {source['file']}，第 {source['page']} 页：{source['snippet']}")

        lines.extend(["", f"**反馈**：{item.get('feedback') or '未反馈'}", ""])

    return "\n".join(lines)


def txt_export() -> str:
    """Export chat history as plain text."""
    lines = ["PDF 知识库问答记录", f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]

    for index, item in enumerate(st.session_state.chat_history, start=1):
        lines.extend(
            [
                f"问答 {index}",
                f"时间：{item['created_at']}",
                f"使用场景：{item['scenario']}",
                f"问题：{item['question']}",
                "回答：",
                item["answer"],
                "引用来源：",
            ]
        )

        for source in item["sources"]:
            lines.append(f"- {source['file']}，第 {source['page']} 页：{source['snippet']}")

        lines.extend([f"反馈：{item.get('feedback') or '未反馈'}", "-" * 60, ""])

    return "\n".join(lines)


def render_sidebar(uploaded_files) -> None:
    """Render API status, upload control, and knowledge base management."""
    with st.sidebar:
        st.subheader("知识库管理")

        if get_api_key():
            st.success("已读取 GOOGLE_API_KEY")
        else:
            st.error("未找到 GOOGLE_API_KEY")
            st.info("请在环境变量或 .env 文件中配置 GOOGLE_API_KEY。")

        st.caption(f"聊天模型：{get_config_value('GEMINI_CHAT_MODEL', DEFAULT_CHAT_MODEL)}")
        st.caption(f"Embedding：{get_config_value('GEMINI_EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL)}")

        st.divider()

        if uploaded_files:
            for record in st.session_state.file_records:
                st.markdown(
                    f"""
                    <div class="kb-card">
                      <strong>{record["name"]}</strong>
                      <div class="kb-meta">
                        页数：{record["pages"]}<br>
                        上传：{record["uploaded_at"]}<br>
                        状态：{record["status"]}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("上传 PDF 后，这里会显示文件、页数、上传时间和建库状态。")


def handle_upload(uploaded_files) -> None:
    """Build or refresh the knowledge base when uploaded files change."""
    if not uploaded_files:
        return

    current_signature = uploaded_files_signature(uploaded_files)
    if current_signature == st.session_state.file_signature:
        return

    if not get_api_key():
        st.warning("请先配置 GOOGLE_API_KEY，然后重新上传 PDF 或刷新页面。")
        return

    pending_records = [
        {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "pages": "-",
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "建库中",
        }
        for uploaded_file in uploaded_files
    ]
    st.session_state.file_records = pending_records

    with st.spinner("正在读取 PDF、切分文本、生成向量并建立 FAISS 知识库..."):
        try:
            vector_store, documents, chunks, file_records = build_vector_store(uploaded_files)
        except Exception as exc:
            for record in st.session_state.file_records:
                record["status"] = "建库失败"
            st.session_state.vector_store = None
            st.session_state.file_signature = None
            st.error(f"知识库建立失败：{exc}")
        else:
            st.session_state.vector_store = vector_store
            st.session_state.documents = documents
            st.session_state.chunks = chunks
            st.session_state.file_signature = current_signature
            st.session_state.file_records = file_records
            st.session_state.quick_outputs = {}
            st.session_state.stats.update(
                {
                    "files": len(file_records),
                    "pages": len(documents),
                    "chunks": len(chunks),
                }
            )
            st.success("知识库已建立，可以开始提问或使用快捷分析。")


def render_dashboard() -> None:
    """Render product metrics."""
    stats = st.session_state.stats
    cols = st.columns(6)
    cols[0].metric("文件数", stats["files"])
    cols[1].metric("文档页数", stats["pages"])
    cols[2].metric("文本块", stats["chunks"])
    cols[3].metric("提问次数", stats["questions"])
    cols[4].metric("有帮助", stats["helpful"])
    cols[5].metric("没帮助", stats["unhelpful"])


def render_chat_tab(scenario: str) -> None:
    """Render chat history and handle new questions."""
    st.subheader("知识库问答")

    if not st.session_state.chat_history:
        st.info("上传并建库后，在下方输入问题。回答会保留在当前会话历史中。")

    for index, item in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            render_sources(item["sources"], expanded=False)
            render_feedback(index)

    question = st.chat_input("请输入你想问 PDF 的问题")

    if not question:
        return

    if st.session_state.vector_store is None:
        st.warning("请先上传 PDF 并等待知识库建立完成。")
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索相关内容并生成回答..."):
            try:
                answer, source_docs = answer_question(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    scenario=scenario,
                )
            except Exception as exc:
                st.error(f"回答生成失败：{exc}")
                return

            sources = source_payload(source_docs)
            st.markdown(answer)
            render_sources(sources)

    st.session_state.chat_history.append(
        {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario,
            "question": question,
            "answer": answer,
            "sources": sources,
            "feedback": None,
        }
    )
    st.session_state.stats["questions"] += 1
    st.rerun()


def render_quick_actions_tab(scenario: str) -> None:
    """Render document-level quick action buttons and results."""
    st.subheader("文档智能处理")

    if st.session_state.vector_store is None:
        st.info("上传 PDF 并完成建库后，可在这里生成总结、关键词、大纲、复习题和 FAQ。")
        return

    cols = st.columns([1, 1, 1, 1, 1])
    action_names = list(QUICK_ACTIONS.keys())

    for index, action_name in enumerate(action_names):
        with cols[index]:
            if st.button(action_name, use_container_width=True):
                with st.spinner(f"正在生成：{action_name}..."):
                    try:
                        st.session_state.quick_outputs[action_name] = run_quick_action(
                            action_name,
                            scenario,
                        )
                    except Exception as exc:
                        st.error(f"{action_name} 生成失败：{exc}")

    for action_name, output in st.session_state.quick_outputs.items():
        with st.expander(action_name, expanded=True):
            st.markdown(output)


def render_export_tab() -> None:
    """Render conversation history and export controls."""
    st.subheader("对话历史与导出")

    if not st.session_state.chat_history:
        st.info("还没有问答记录。开始提问后，可以在这里导出 Markdown 或 TXT。")
        return

    for index, item in enumerate(st.session_state.chat_history, start=1):
        with st.expander(f"{index}. {item['question']}", expanded=False):
            st.caption(f"{item['created_at']} · {item['scenario']}")
            st.markdown(item["answer"])
            render_sources(item["sources"], expanded=False)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "导出 Markdown",
            data=markdown_export(),
            file_name="pdf_qa_history.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "导出 TXT",
            data=txt_export(),
            file_name="pdf_qa_history.txt",
            mime="text/plain",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="AI 文档知识库", page_icon="📚", layout="wide")
    init_session_state()
    apply_product_styles()

    st.title("AI 文档知识库工作台")
    st.caption("Gemini API + LangChain + FAISS + Streamlit")

    top_left, top_right = st.columns([2, 1])

    with top_left:
        uploaded_files = st.file_uploader(
            "上传 PDF 文档",
            type=["pdf"],
            accept_multiple_files=True,
            help="支持一个或多个 PDF。上传后会自动读取文本、切分并建立 FAISS 向量知识库。",
        )

    with top_right:
        scenario = st.selectbox(
            "使用场景",
            options=list(SCENARIO_PROMPTS.keys()),
            help="不同场景会切换不同的 system prompt。",
        )

    handle_upload(uploaded_files)
    render_sidebar(uploaded_files)
    render_dashboard()

    st.divider()

    qa_tab, quick_tab, export_tab = st.tabs(["问答工作台", "快捷分析", "历史导出"])

    with qa_tab:
        render_chat_tab(scenario)

    with quick_tab:
        render_quick_actions_tab(scenario)

    with export_tab:
        render_export_tab()


if __name__ == "__main__":
    main()
