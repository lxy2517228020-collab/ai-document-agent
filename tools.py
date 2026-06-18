from langchain_core.prompts import ChatPromptTemplate

from export_utils import export_to_markdown
from prompts import (
    ANSWER_PROMPT,
    BASE_AGENT_SYSTEM_PROMPT,
    DEFAULT_OUTPUT_DETAIL,
    SCENARIO_PROMPTS,
    TOOL_SUMMARY_PROMPT,
    get_output_detail_instruction,
)
from rag import (
    GeminiServiceError,
    call_with_retry,
    corpus_preview,
    extract_message_text,
    fallback_model_message,
    format_context,
    get_llm,
    get_model_config,
    gemini_busy_message,
    is_not_found_error,
    retrieve_passages as rag_retrieve_passages,
    source_payload,
)


class AgentTools:
    """Tool layer used by the Document Agent executor."""

    def __init__(
        self,
        knowledge_base: dict,
        scenario: str,
        deep_evaluation: bool = False,
        detail_level: str = DEFAULT_OUTPUT_DETAIL,
        output_detail: str | None = None,
    ):
        self.knowledge_base = knowledge_base
        self.scenario = scenario
        self.deep_evaluation = deep_evaluation
        self.detail_level = output_detail or detail_level or DEFAULT_OUTPUT_DETAIL
        self.output_detail = self.detail_level
        self.length_instruction = get_output_detail_instruction(self.detail_level)
        self.model_config = get_model_config()
        self.llm = get_llm(self.model_config["selected_model"])
        self.metrics = {
            "gemini_calls": 0,
            "retry_count": 0,
            "retry_events": [],
            "errors": [],
            "selected_model": self.model_config["selected_model"],
            "actual_model": self.model_config["selected_model"],
            "fallback_model": self.model_config["fallback_model"],
            "model_fallback": False,
            "fallback_reason": None,
            "detail_level": self.detail_level,
            "output_detail": self.detail_level,
        }

    @property
    def chunks(self):
        return self.knowledge_base.get("chunks", [])

    @property
    def scenario_prompt(self) -> str:
        return SCENARIO_PROMPTS.get(self.scenario, SCENARIO_PROMPTS["课程资料复习"])

    def _invoke(self, template: str, **kwargs) -> str:
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(
            base_prompt=BASE_AGENT_SYSTEM_PROMPT,
            scenario_prompt=self.scenario_prompt,
            **kwargs,
        )
        self.metrics["gemini_calls"] += 1

        try:
            response, retry_meta = call_with_retry(
                "Gemini LLM 调用",
                lambda: self.llm.invoke(messages),
            )
        except GeminiServiceError as exc:
            self.metrics["retry_count"] += max(0, exc.attempts - 1)
            self.metrics["errors"].append(
                {
                    "operation": exc.operation,
                    "attempts": exc.attempts,
                    "reason": str(exc.last_error),
                }
            )
            raise
        except Exception as exc:
            if not is_not_found_error(exc):
                raise

            fallback_model = self.model_config["fallback_model"]
            self.metrics["model_fallback"] = True
            self.metrics["fallback_reason"] = str(exc)
            self.metrics["actual_model"] = fallback_model
            self.metrics["errors"].append(
                {
                    "operation": "Gemini 模型 fallback",
                    "attempts": 1,
                    "reason": str(exc),
                    "message": fallback_model_message(),
                }
            )
            self.llm = get_llm(fallback_model)
            self.metrics["gemini_calls"] += 1

            try:
                response, retry_meta = call_with_retry(
                    "Gemini LLM fallback 调用",
                    lambda: self.llm.invoke(messages),
                )
            except GeminiServiceError as fallback_exc:
                self.metrics["retry_count"] += max(0, fallback_exc.attempts - 1)
                self.metrics["errors"].append(
                    {
                        "operation": fallback_exc.operation,
                        "attempts": fallback_exc.attempts,
                        "reason": str(fallback_exc.last_error),
                    }
                )
                raise

        if retry_meta["retries"]:
            self.metrics["retry_count"] += retry_meta["retries"]
            self.metrics["retry_events"].append(retry_meta)

        return extract_message_text(response)

    def _resolve_detail(self, detail_level: str | None = None) -> tuple[str, str]:
        """Resolve detail level and prompt instruction for one tool call."""
        level = detail_level or self.detail_level or DEFAULT_OUTPUT_DETAIL
        return level, get_output_detail_instruction(level)

    def _document_task(
        self,
        task: str,
        doc_name: str | None = None,
        limit: int = 14000,
        detail_level: str | None = None,
    ) -> dict:
        context, sources = corpus_preview(self.chunks, doc_name=doc_name, limit=limit)

        if not context:
            return {
                "content": "文档中未找到相关信息。",
                "sources": [],
                "summary": "没有可用文档片段。",
            }

        resolved_detail, length_instruction = self._resolve_detail(detail_level)
        content = self._invoke(
            TOOL_SUMMARY_PROMPT,
            task=task,
            context=context,
            output_detail=resolved_detail,
            length_instruction=length_instruction,
        )
        return {
            "content": content,
            "sources": sources,
            "summary": content[:240],
        }

    def retrieve_passages(
        self,
        query: str,
        k: int = 5,
        target_documents: list[str] | None = None,
    ) -> dict:
        """Retrieve relevant text chunks from FAISS."""
        docs = rag_retrieve_passages(
            self.knowledge_base,
            query=query,
            k=k,
            target_documents=target_documents,
        )
        return {
            "content": format_context(docs) if docs else "文档中未找到相关信息。",
            "sources": source_payload(docs),
            "documents": docs,
            "summary": f"检索到 {len(docs)} 个相关片段。",
        }

    def answer_question(self, question: str, retrieved: dict) -> dict:
        """Generate a cited answer from retrieved passages."""
        docs = retrieved.get("documents", [])

        if not docs:
            return {
                "content": "文档中未找到相关信息。",
                "sources": [],
                "summary": "没有检索到可回答的片段。",
            }

        content = self._invoke(
            ANSWER_PROMPT,
            question=question,
            context=format_context(docs),
        )
        return {
            "content": content,
            "sources": source_payload(docs),
            "summary": content[:240],
        }

    def summarize_document(self, doc_name: str | None = None, detail_level: str | None = None) -> dict:
        """Summarize one document or the whole knowledge base."""
        return self._document_task(
            "总结文档主题、结构、核心观点和关键结论。每个小节最多 1-2 个引用编号。",
            doc_name,
            detail_level=detail_level,
        )

    def extract_key_points(self, doc_name: str | None = None, detail_level: str | None = None) -> dict:
        """Extract key points from one document or the whole knowledge base."""
        return self._document_task(
            "提取核心知识点、关键概念、重要结论，并按主题分组。每个小节最多 1-2 个引用编号。",
            doc_name,
            detail_level=detail_level,
        )

    def generate_faq(self, doc_name: str | None = None, detail_level: str | None = None) -> dict:
        """Generate FAQ from one document or the whole knowledge base."""
        return self._document_task(
            "生成 FAQ，包含常见问题和基于文档的答案。每个答案最多 1 个引用编号。",
            doc_name,
            detail_level=detail_level,
        )

    def generate_quiz(
        self,
        doc_name: str | None = None,
        num_questions: int = 10,
        detail_level: str | None = None,
    ) -> dict:
        """Generate review questions."""
        return self._document_task(
            f"生成 {num_questions} 个复习问题，并附简短参考答案。",
            doc_name,
            detail_level=detail_level,
        )

    def compare_documents(self, doc_a: str | None = None, doc_b: str | None = None) -> dict:
        """Compare two documents."""
        if not doc_a or not doc_b:
            return {
                "content": "请至少上传或指定两份文档，才能进行对比。",
                "sources": [],
                "summary": "缺少对比文档。",
            }

        context_a, sources_a = corpus_preview(self.chunks, doc_name=doc_a, limit=7000)
        context_b, sources_b = corpus_preview(self.chunks, doc_name=doc_b, limit=7000)

        if not context_a or not context_b:
            return {
                "content": "文档中未找到足够信息进行对比。",
                "sources": sources_a + sources_b,
                "summary": "对比上下文不足。",
            }

        content = self._invoke(
            TOOL_SUMMARY_PROMPT,
            task=(
                f"对比两份文档的差异。文档 A：{doc_a}；文档 B：{doc_b}。"
                "请从目标、结构、重点、结论、风险或适用场景等维度对比。"
            ),
            context=f"文档 A：\n{context_a}\n\n文档 B：\n{context_b}",
            output_detail=self.output_detail,
            length_instruction=self.length_instruction,
        )
        return {
            "content": content,
            "sources": sources_a + sources_b,
            "summary": content[:240],
        }

    def analyze_jd(self, doc_name: str | None = None, detail_level: str | None = None) -> dict:
        """Analyze job description requirements."""
        resolved_detail, _instruction = self._resolve_detail(detail_level)
        task = (
            "分析岗位 JD，包括岗位职责、能力要求、核心关键词和候选人匹配建议。"
            "所有结论必须来自文档，不要泛泛展开，不要输出长括号来源。"
        )
        if resolved_detail == "简洁版":
            task += (
                "\n简洁版必须使用以下固定结构："
                "\n- 岗位职责：3 条"
                "\n- 能力要求：3 条"
                "\n- 核心关键词：6 个"
                "\n- 候选人匹配建议：3 条"
                "\n不要输出“面试准备方向”。"
                "\n每个 bullet 控制在 1-2 句话，每个小节最多 1 个引用编号。"
            )
        elif resolved_detail == "标准版":
            task += (
                "\n标准版必须使用以下固定结构："
                "\n- 岗位职责：4 条"
                "\n- 能力要求：4 条"
                "\n- 核心关键词：8 个"
                "\n- 候选人匹配建议：4 条"
                "\n不要默认输出“面试准备方向”。"
                "\n每个 bullet 控制在 1-2 句话，每个小节最多 1-2 个引用编号。"
            )
        else:
            task += (
                "\n详细版可以加入“面试准备方向”，但仍需围绕文档证据组织内容。"
                "\n每个 bullet 控制在 1-2 句话。"
            )
        return self._document_task(
            task,
            doc_name,
            detail_level=resolved_detail,
        )

    def extract_risks(self, doc_name: str | None = None) -> dict:
        """Extract risks, issues, and open questions."""
        return self._document_task(
            "提取风险点、问题点、责任方、潜在影响和待确认事项。",
            doc_name,
        )

    def create_study_guide(self, doc_name: str | None = None) -> dict:
        """Create a study guide."""
        return self._document_task(
            "生成考试或复习资料，包括知识点大纲、核心概念、复习路线和练习问题。",
            doc_name,
        )

    def export_to_markdown(self, content: str) -> dict:
        """Export current result to Markdown text."""
        markdown = export_to_markdown(content)
        return {
            "content": markdown,
            "sources": [],
            "summary": "已生成 Markdown 文本。",
        }

    def evaluate_answer(self, answer: str, sources: list[dict]) -> dict:
        """Evaluate citation and grounding quality with rules by default."""
        structured_sources = [self._normalize_source(source) for source in sources]
        structured_sources = [source for source in structured_sources if source]
        has_sources = bool(structured_sources)
        says_not_found = "文档中未找到相关信息" in answer
        has_file_name = any(source.get("file_name") for source in structured_sources)
        has_page = any(source.get("page") not in (None, "", "未知页码") for source in structured_sources)
        has_snippet = any(source.get("snippet") for source in structured_sources)
        has_citation_text = has_sources and has_file_name and has_page

        if says_not_found:
            confidence = "low"
        elif has_sources and has_file_name and has_page and has_snippet:
            confidence = "high"
        elif has_sources:
            confidence = "medium"
        else:
            confidence = "low"

        result = {
            "has_sources": has_sources,
            "has_file_name": has_file_name,
            "has_page": has_page,
            "has_snippet": has_snippet,
            "has_citation_text": has_citation_text,
            "grounded": has_sources or says_not_found,
            "confidence": confidence,
            "mode": "rule",
            "summary": f"引用来源：{'有' if has_sources else '无'}；置信度：{confidence}",
        }

        if not self.deep_evaluation:
            return result

        try:
            deep_result = self._invoke(
                (
                    "{base_prompt}\n\n{scenario_prompt}\n\n"
                    "请评估下面回答是否基于文档、引用是否充分、是否存在明显幻觉。"
                    "请输出简短中文结论和 high/medium/low 置信度。\n\n"
                    "回答：{answer}\n\n引用来源：{sources}"
                ),
                answer=answer,
                sources=structured_sources,
            )
        except Exception as exc:
            result["deep_evaluation_error"] = str(exc)
            return result

        result["mode"] = "rule+llm"
        result["deep_evaluation"] = deep_result
        return result

    @staticmethod
    def _normalize_source(source: dict) -> dict:
        """Normalize legacy source payloads into the current citation schema."""
        if not isinstance(source, dict):
            return {}

        file_name = source.get("file_name") or source.get("file")
        return {
            "file_name": file_name or "未知文件",
            "page": source.get("page", "未知页码"),
            "snippet": source.get("snippet", ""),
        }


def friendly_gemini_failure() -> dict:
    """Return a standard failed tool payload for transient Gemini failures."""
    return {
        "content": gemini_busy_message(),
        "sources": [],
        "summary": "Gemini 模型繁忙或限流，已触发友好失败提示。",
    }
