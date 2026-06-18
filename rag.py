import os
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

DEFAULT_MODEL_MODE = "稳定模式"
MODEL_MODES = {
    "快速模式": {
        "model": "gemini-2.5-flash-lite",
        "fallback_model": "gemini-2.5-flash",
        "description": "轻量 Flash-Lite 模型，若不可用会自动切换到 gemini-2.5-flash。",
    },
    "稳定模式": {
        "model": "gemini-2.5-flash",
        "fallback_model": "gemini-2.5-flash",
        "description": "默认推荐，兼顾速度、成本和稳定性。",
    },
    "高质量模式": {
        "model": "gemini-2.5-pro",
        "fallback_model": "gemini-2.5-flash",
        "description": "更强模型，若不可用会自动切换到 gemini-2.5-flash。",
    },
}
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_FALLBACK_MODEL = "gemini-2.5-flash"
RETRY_DELAYS = [2, 4, 8]


class GeminiServiceError(RuntimeError):
    """Friendly wrapper for Gemini transient failures after retry."""

    def __init__(self, operation: str, attempts: int, last_error: Exception):
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            "当前 Gemini 模型繁忙，请稍后重试，或切换到快速模式。"
        )


def get_streamlit_secret(name: str) -> str | None:
    """Read a value from Streamlit Secrets when available."""
    try:
        value = st.secrets.get(name)
    except Exception:
        return None

    if value is None:
        return None

    return str(value)


def get_config_value(name: str, default: str | None = None) -> str | None:
    """Read config from Streamlit Secrets first, then environment variables."""
    return get_streamlit_secret(name) or os.getenv(name) or default


def get_api_key() -> str | None:
    """Read GOOGLE_API_KEY from Streamlit Secrets, env vars, or local .env."""
    api_key = get_config_value("GOOGLE_API_KEY")

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    return api_key


def get_model_mode() -> str:
    """Read the selected model mode from Streamlit state."""
    try:
        mode = st.session_state.get("model_mode", DEFAULT_MODEL_MODE)
    except Exception:
        mode = DEFAULT_MODEL_MODE

    return mode if mode in MODEL_MODES else DEFAULT_MODEL_MODE


def get_model_name() -> str:
    """Resolve the current chat model name from mode, secrets, or env vars."""
    return get_model_config()["selected_model"]


def get_model_config() -> dict:
    """Return current model mode, selected model, and fallback model."""
    mode = get_model_mode()
    mode_config = MODEL_MODES[mode]
    mode_model = mode_config["model"]
    fallback_model = mode_config.get("fallback_model", DEFAULT_FALLBACK_MODEL)

    if mode == "快速模式":
        selected_model = get_config_value("GEMINI_FAST_MODEL", mode_model)
    elif mode == "高质量模式":
        selected_model = get_config_value("GEMINI_QUALITY_MODEL", mode_model)
    else:
        selected_model = get_config_value("GEMINI_STABLE_MODEL", mode_model)

    fallback_model = get_config_value("GEMINI_FALLBACK_MODEL", fallback_model)

    return {
        "mode": mode,
        "selected_model": selected_model,
        "fallback_model": fallback_model,
        "description": mode_config["description"],
    }


def get_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model from the current centralized config."""
    get_api_key()
    model_name = model_name or get_model_config()["selected_model"]
    default_temperature = "1.0" if model_name.startswith("gemini-3") else "0.2"
    temperature = float(get_config_value("GEMINI_TEMPERATURE", default_temperature))
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def is_not_found_error(error: Exception) -> bool:
    """Return True for non-retryable Gemini model name/API version errors."""
    message = str(error).lower()
    return (
        "404" in message
        or "not_found" in message
        or "not found" in message
        or "is not supported for generatecontent" in message
    )


def fallback_model_message() -> str:
    """Message shown when the selected generation model is unavailable."""
    return "当前选择的模型不可用，已自动切换到 gemini-2.5-flash。"


def list_available_generation_models() -> list[str]:
    """List Gemini models that support generateContent.

    This helper is intentionally not called on app startup to avoid extra latency.
    """
    get_api_key()

    try:
        from google import genai
    except Exception:
        return []

    client = genai.Client(api_key=get_api_key())
    models = client.models.list()
    available_models = []

    for model in models:
        methods = getattr(model, "supported_generation_methods", None)
        if methods is None:
            methods = getattr(model, "supported_actions", [])
        if methods and "generateContent" not in methods:
            continue

        name = getattr(model, "name", "")
        if name.startswith("models/"):
            name = name.removeprefix("models/")
        if name:
            available_models.append(name)

    return sorted(set(available_models))


def get_embedding_model_name() -> str:
    """Resolve the embedding model name from one central place."""
    return get_config_value("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Backward-compatible alias for get_llm()."""
    return get_llm()


@st.cache_resource(show_spinner=False)
def _cached_embeddings(model_name: str) -> GoogleGenerativeAIEmbeddings:
    """Cache the embedding client so repeated calls do not reinitialize it."""
    return GoogleGenerativeAIEmbeddings(model=model_name)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create or reuse the Gemini embedding model."""
    get_api_key()
    return _cached_embeddings(get_embedding_model_name())


def is_retryable_error(error: Exception) -> bool:
    """Return True when the error looks like a transient Gemini/API issue."""
    if is_not_found_error(error):
        return False

    message = str(error).lower()
    retryable_terms = [
        "503",
        "unavailable",
        "429",
        "rate limit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "resource exhausted",
        "service unavailable",
    ]
    return any(term in message for term in retryable_terms)


def call_with_retry(operation_name: str, fn):
    """Run Gemini work with exponential backoff and friendly final error."""
    last_error = None

    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            result = fn()
            return result, {"operation": operation_name, "retries": attempt, "error": None}
        except Exception as exc:
            last_error = exc
            if attempt >= len(RETRY_DELAYS) or not is_retryable_error(exc):
                if is_retryable_error(exc):
                    raise GeminiServiceError(operation_name, attempt + 1, exc) from exc
                raise

            time.sleep(RETRY_DELAYS[attempt])

    raise GeminiServiceError(operation_name, len(RETRY_DELAYS) + 1, last_error)


def gemini_busy_message() -> str:
    """User-facing message for transient Gemini failures."""
    return (
        "当前 Gemini 模型繁忙，请稍后重试，或切换到快速模式。\n\n"
        "可能原因：模型高峰期、免费额度繁忙、临时限流或网络超时。\n\n"
        "建议操作：稍后重试、切换到快速模式，或换用更轻量的 Flash 模型。"
    )


def uploaded_files_signature(uploaded_files) -> tuple[tuple[str, int], ...]:
    """Build a signature so unchanged uploads do not rebuild the index."""
    return tuple((uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files)


def load_uploaded_pdfs(uploaded_files) -> tuple[list[Document], list[dict]]:
    """Load uploaded PDF files and keep friendly metadata for citations."""
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
                page.metadata["source"] = uploaded_file.name
                page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
                documents.append(page)

            file_records.append(
                {
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "pages": len(pages),
                    "chunks": 0,
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
                    "chunks": 0,
                    "uploaded_at": uploaded_at,
                    "status": "建库失败",
                }
            )
            raise
        finally:
            Path(temp_path).unlink(missing_ok=True)

    return documents, file_records


def split_documents(documents: list[Document]) -> list[Document]:
    """Split PDF pages into retrieval-friendly chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list[Document]) -> FAISS | None:
    """Create a FAISS vector store from chunks."""
    if not chunks:
        return None

    vector_store, _meta = call_with_retry(
        "Gemini Embedding 建库",
        lambda: FAISS.from_documents(chunks, get_embeddings()),
    )
    return vector_store


def build_knowledge_base(uploaded_files) -> dict:
    """Load PDFs, split text, generate embeddings, and build FAISS."""
    documents, file_records = load_uploaded_pdfs(uploaded_files)
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("没有从 PDF 中读取到可用文本。请确认 PDF 不是纯扫描图片。")

    chunk_counts = Counter(chunk.metadata.get("source", "未知文件") for chunk in chunks)
    for record in file_records:
        record["chunks"] = chunk_counts.get(record["name"], 0)

    vector_store = build_vector_store(chunks)

    return {
        "vector_store": vector_store,
        "documents": documents,
        "chunks": chunks,
        "file_records": file_records,
        "file_signature": uploaded_files_signature(uploaded_files),
    }


def empty_knowledge_base() -> dict:
    """Return an empty knowledge base object."""
    return {
        "vector_store": None,
        "documents": [],
        "chunks": [],
        "file_records": [],
        "file_signature": None,
    }


def rebuild_knowledge_base(knowledge_base: dict) -> dict:
    """Regenerate FAISS from existing chunks."""
    chunks = knowledge_base.get("chunks", [])
    knowledge_base["vector_store"] = build_vector_store(chunks)

    for record in knowledge_base.get("file_records", []):
        record["status"] = "已重新建库"

    return knowledge_base


def delete_document(knowledge_base: dict, doc_name: str) -> dict:
    """Remove one document and rebuild FAISS from remaining chunks."""
    documents = [
        doc
        for doc in knowledge_base.get("documents", [])
        if doc.metadata.get("source") != doc_name
    ]
    chunks = [
        chunk
        for chunk in knowledge_base.get("chunks", [])
        if chunk.metadata.get("source") != doc_name
    ]
    file_records = [
        record
        for record in knowledge_base.get("file_records", [])
        if record.get("name") != doc_name
    ]

    knowledge_base["documents"] = documents
    knowledge_base["chunks"] = chunks
    knowledge_base["file_records"] = file_records
    knowledge_base["vector_store"] = build_vector_store(chunks)
    knowledge_base["file_signature"] = None

    return knowledge_base


def get_document_names(knowledge_base: dict) -> list[str]:
    """Return document names currently stored in the knowledge base."""
    return [record["name"] for record in knowledge_base.get("file_records", [])]


def source_payload(docs: list[Document]) -> list[dict]:
    """Convert retrieved docs into display/export-friendly source dicts."""
    sources = []
    seen = set()

    for doc in docs:
        snippet = " ".join(doc.page_content.split())[:500]
        source = {
            "file_name": doc.metadata.get("source", "未知文件"),
            "page": doc.metadata.get("page", "未知页码"),
            "snippet": snippet,
        }
        key = (source["file_name"], source["page"])

        if key not in seen:
            sources.append(source)
            seen.add(key)

    return sources


def format_context(docs: list[Document]) -> str:
    """Format documents as LLM context with citation metadata."""
    parts = []
    source_index = {}

    for doc in docs:
        source = doc.metadata.get("source", "未知文件")
        page = doc.metadata.get("page", "未知页码")
        key = (source, page)
        if key not in source_index:
            source_index[key] = len(source_index) + 1
        index = source_index[key]
        parts.append(f"[引用 {index}] 文件：{source}，页码：第 {page} 页\n{doc.page_content}")

    return "\n\n".join(parts)


def filter_chunks_by_document(chunks: list[Document], doc_name: str | None) -> list[Document]:
    """Return chunks from one document, or all chunks when doc_name is empty."""
    if not doc_name:
        return chunks

    return [chunk for chunk in chunks if chunk.metadata.get("source") == doc_name]


def corpus_preview(
    chunks: list[Document],
    doc_name: str | None = None,
    limit: int = 14000,
) -> tuple[str, list[dict]]:
    """Create compact context for document-level tools."""
    selected_chunks = filter_chunks_by_document(chunks, doc_name)
    parts = []
    total_chars = 0
    used_docs = []
    source_index = {}

    for chunk in selected_chunks:
        source = chunk.metadata.get("source", "未知文件")
        page = chunk.metadata.get("page", "未知页码")
        key = (source, page)
        index = source_index.get(key, len(source_index) + 1)
        text = " ".join(chunk.page_content.split())
        block = f"[引用 {index}] 文件：{source}，页码：第 {page} 页\n{text}"

        if total_chars + len(block) > limit:
            break

        if key not in source_index:
            source_index[key] = index
            used_docs.append(chunk)

        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts), source_payload(used_docs)


def retrieve_passages(
    knowledge_base: dict,
    query: str,
    k: int = 5,
    target_documents: list[str] | None = None,
) -> list[Document]:
    """Retrieve relevant passages from FAISS and optionally filter by document name."""
    vector_store = knowledge_base.get("vector_store")
    if vector_store is None:
        return []

    search_k = max(k, 8) if target_documents else k
    docs = vector_store.similarity_search(query, k=search_k)

    if target_documents:
        target_set = set(target_documents)
        docs = [doc for doc in docs if doc.metadata.get("source") in target_set]

    return docs[:k]


def extract_message_text(message) -> str:
    """Return text from LangChain AIMessage across Gemini response shapes."""
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
