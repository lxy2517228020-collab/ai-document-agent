import json
import re
import time
from dataclasses import dataclass

from rag import GeminiServiceError, get_document_names
from tools import AgentTools, friendly_gemini_failure


@dataclass
class IntentResult:
    intent: str
    task_goal: str
    target_documents: list[str]
    need_retrieval: bool
    need_citation: bool
    steps: list[str]

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "task_goal": self.task_goal,
            "target_documents": self.target_documents,
            "need_retrieval": self.need_retrieval,
            "need_citation": self.need_citation,
            "steps": self.steps,
        }


class IntentRouter:
    """Classify natural language tasks into structured intents."""

    def route(self, user_task: str, document_names: list[str]) -> dict:
        text = user_task.lower()
        target_documents = self._detect_target_documents(user_task, document_names)

        if self._has_any(text, ["导出", "markdown", "txt", "下载"]):
            intent = "export_result"
        elif self._has_any(text, ["对比", "比较", "差异", "不同"]):
            intent = "compare_documents"
        elif self._has_any(text, ["岗位", "jd", "职位", "招聘", "候选人"]):
            intent = "analyze_jd"
        elif self._has_any(text, ["风险", "问题点", "待确认", "合同", "责任方"]):
            intent = "extract_risks"
        elif self._has_any(text, ["faq", "常见问题"]):
            intent = "generate_faq"
        elif self._has_any(text, ["复习题", "练习题", "测试题", "quiz", "题目"]):
            intent = "generate_quiz"
        elif self._has_any(text, ["复习大纲", "复习资料", "学习指南", "考试", "讲解稿"]):
            intent = "create_study_guide"
        elif self._has_any(text, ["知识点", "核心要点", "关键点", "关键词", "要点"]):
            intent = "extract_key_points"
        elif self._has_any(text, ["总结", "摘要", "概括", "整理"]):
            intent = "summarize_document"
        elif self._looks_like_question(user_task):
            intent = "answer_question"
        else:
            intent = "unknown"

        need_retrieval = intent not in {"export_result", "unknown"}
        need_citation = intent != "export_result"
        steps = self._default_steps(intent)

        return IntentResult(
            intent=intent,
            task_goal=user_task,
            target_documents=target_documents,
            need_retrieval=need_retrieval,
            need_citation=need_citation,
            steps=steps,
        ).to_dict()

    @staticmethod
    def _has_any(text: str, keywords: list[str]) -> bool:
        return any(keyword.lower() in text for keyword in keywords)

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        question_words = ["什么", "为什么", "如何", "怎么", "是否", "哪些", "多少", "吗", "?"]
        return any(word in text for word in question_words)

    @staticmethod
    def _detect_target_documents(user_task: str, document_names: list[str]) -> list[str]:
        return [name for name in document_names if name in user_task]

    @staticmethod
    def _default_steps(intent: str) -> list[str]:
        mapping = {
            "answer_question": ["理解问题", "检索相关片段", "基于文档生成回答", "评估引用质量"],
            "summarize_document": ["提取文档核心内容", "总结主题和结构", "输出带引用总结"],
            "extract_key_points": ["提取核心片段", "识别关键概念", "按主题输出知识点"],
            "generate_faq": ["识别高频问题", "生成 FAQ", "补充引用来源"],
            "generate_quiz": ["提取可考知识点", "生成复习题", "生成参考答案"],
            "compare_documents": ["读取两份文档", "提取对比维度", "输出差异分析"],
            "analyze_jd": ["提取 JD 内容", "分析职责和要求", "生成匹配建议"],
            "extract_risks": ["检索风险相关内容", "识别风险点", "输出待确认事项"],
            "create_study_guide": ["检索核心内容", "总结主题", "生成大纲", "生成复习问题"],
            "export_result": ["整理最近结果", "生成 Markdown 文本"],
            "unknown": ["识别失败", "提示用户补充任务"],
        }
        return mapping.get(intent, mapping["unknown"])


class AgentPlanner:
    """Create executable tool plans from intents."""

    def plan(self, router_result: dict, document_names: list[str]) -> list[dict]:
        intent = router_result["intent"]
        targets = router_result.get("target_documents") or []
        doc_name = targets[0] if len(targets) == 1 else None

        if intent == "answer_question":
            return [
                self._step(
                    1,
                    "retrieve_passages",
                    {
                        "query": router_result["task_goal"],
                        "k": 5,
                        "target_documents": targets or document_names,
                    },
                    "检索相关文档片段",
                ),
                self._step(2, "answer_question", {}, "基于检索片段生成回答"),
                self._step(3, "evaluate_answer", {}, "评估引用和可信度"),
            ]

        if intent == "compare_documents":
            doc_a, doc_b = self._pick_two_documents(targets, document_names)
            return [
                self._step(1, "compare_documents", {"doc_a": doc_a, "doc_b": doc_b}, "对比两份文档"),
                self._step(2, "evaluate_answer", {}, "评估引用和可信度"),
            ]

        single_tool_map = {
            "summarize_document": ("summarize_document", "生成文档总结"),
            "extract_key_points": ("extract_key_points", "提取核心知识点"),
            "generate_faq": ("generate_faq", "生成 FAQ"),
            "generate_quiz": ("generate_quiz", "生成复习问题"),
            "analyze_jd": ("analyze_jd", "分析岗位 JD"),
            "extract_risks": ("extract_risks", "提取风险点"),
            "create_study_guide": ("create_study_guide", "生成复习大纲"),
        }

        if intent in single_tool_map:
            tool, description = single_tool_map[intent]
            args = {"doc_name": doc_name}
            if intent == "generate_quiz":
                args["num_questions"] = self._extract_number(router_result["task_goal"], default=10)
            return [
                self._step(1, tool, args, description),
                self._step(2, "evaluate_answer", {}, "评估引用和可信度"),
            ]

        if intent == "export_result":
            return [self._step(1, "export_to_markdown", {}, "导出最近一次结果为 Markdown")]

        return [self._step(1, "unknown", {}, "无法识别任务，提示用户补充信息")]

    @staticmethod
    def _step(step: int, tool: str, args: dict, description: str) -> dict:
        return {
            "step": step,
            "tool": tool,
            "args": args,
            "description": description,
        }

    @staticmethod
    def _pick_two_documents(targets: list[str], document_names: list[str]) -> tuple[str | None, str | None]:
        candidates = targets or document_names
        if len(candidates) >= 2:
            return candidates[0], candidates[1]
        return None, None

    @staticmethod
    def _extract_number(text: str, default: int = 10) -> int:
        match = re.search(r"\d+", text)
        if not match:
            return default
        return max(1, min(30, int(match.group())))


class DocumentAgent:
    """Intent router + planner + multi-step executor."""

    def __init__(
        self,
        knowledge_base: dict,
        scenario: str,
        fast_routing: bool = True,
        deep_evaluation: bool = False,
        detail_level: str = "简洁版",
        output_detail: str | None = None,
    ):
        self.knowledge_base = knowledge_base
        self.scenario = scenario
        self.fast_routing = fast_routing
        self.deep_evaluation = deep_evaluation
        self.detail_level = output_detail or detail_level
        self.output_detail = self.detail_level
        self.router = IntentRouter()
        self.planner = AgentPlanner()
        self.tools = AgentTools(
            knowledge_base,
            scenario,
            deep_evaluation=deep_evaluation,
            detail_level=self.detail_level,
        )

    def execute(self, user_task: str, last_result: dict | None = None) -> dict:
        start_time = time.perf_counter()
        document_names = get_document_names(self.knowledge_base)
        router_result = self.router.route(user_task, document_names)
        router_result = self._fill_target_documents(router_result, document_names)
        router_result["routing_mode"] = "rule"
        router_result["planner_mode"] = "fixed"
        router_result["fast_routing"] = self.fast_routing
        if self.fast_routing:
            router_result["performance_note"] = "使用规则路由，跳过 LLM Planner 以提升速度。"

        if router_result["intent"] == "unknown" and not self.fast_routing:
            router_result = self._llm_router_fallback(user_task, document_names, router_result)
            router_result = self._fill_target_documents(router_result, document_names)

        plan = self.planner.plan(router_result, document_names)
        tool_calls = []
        current_result = None
        last_evaluation = None
        sources = []

        if not document_names and router_result["intent"] != "export_result":
            answer = "请先上传 PDF 并完成知识库构建，然后再让 Agent 执行文档任务。"
            evaluation = self.tools.evaluate_answer(answer, [])
            return self._result(user_task, router_result, plan, tool_calls, answer, [], evaluation, start_time, self.tools.metrics)

        for step in plan:
            tool_name = step["tool"]
            args = dict(step.get("args", {}))
            try:
                output = self._run_step(tool_name, args, user_task, current_result, last_result)
            except GeminiServiceError as exc:
                output = friendly_gemini_failure()
                output["summary"] = (
                    f"{output['summary']} 重试 {max(0, exc.attempts - 1)} 次后失败：{exc.last_error}"
                )
            except Exception as exc:
                output = {
                    "content": f"Agent 执行该步骤时遇到错误：{exc}",
                    "sources": [],
                    "summary": f"工具执行失败：{exc}",
                }

            if output is None:
                output = {
                    "content": "文档中未找到相关信息。",
                    "sources": [],
                    "summary": "工具没有返回结果。",
                }

            current_result = output
            if tool_name == "evaluate_answer" and output.get("evaluation"):
                last_evaluation = output["evaluation"]
            if output.get("sources"):
                sources.extend(output["sources"])

            tool_calls.append(
                {
                    "step": step["step"],
                    "tool": tool_name,
                    "args": args,
                    "summary": output.get("summary", "")[:300],
                }
            )

        answer = current_result.get("content", "文档中未找到相关信息。") if current_result else "文档中未找到相关信息。"
        sources = self._dedupe_sources(sources)
        evaluation = last_evaluation or self.tools.evaluate_answer(answer, sources)

        return self._result(
            user_task,
            router_result,
            plan,
            tool_calls,
            answer,
            sources,
            evaluation,
            start_time,
            self.tools.metrics,
        )

    def _llm_router_fallback(self, user_task: str, document_names: list[str], original: dict) -> dict:
        """Use LLM routing only for unknown tasks when fast routing is disabled."""
        prompt = (
            "{base_prompt}\n\n{scenario_prompt}\n\n"
            "请把用户文档任务分类为以下 intent 之一："
            "answer_question, summarize_document, extract_key_points, generate_faq, generate_quiz, "
            "compare_documents, analyze_jd, extract_risks, create_study_guide, export_result, unknown。\n"
            "只输出 JSON，不要输出解释。字段：intent, task_goal, target_documents, need_retrieval, need_citation, steps。\n\n"
            "可用文档名：{document_names}\n用户任务：{user_task}"
        )
        try:
            raw = self.tools._invoke(
                prompt,
                document_names=document_names,
                user_task=user_task,
            )
            parsed = json.loads(raw.strip().strip("`").replace("json\n", "", 1))
            parsed["routing_mode"] = "llm_fallback"
            parsed["planner_mode"] = "fixed"
            parsed["fast_routing"] = self.fast_routing
            parsed["performance_note"] = "规则路由未识别，已启用 LLM Router fallback。"
            return parsed
        except Exception as exc:
            original["routing_mode"] = "rule_failed"
            original["llm_router_error"] = str(exc)
            return original

    @staticmethod
    def _fill_target_documents(router_result: dict, document_names: list[str]) -> dict:
        """Default target documents to the current knowledge base when omitted."""
        if document_names and not router_result.get("target_documents"):
            router_result["target_documents"] = list(document_names)
        return router_result

    def _run_step(
        self,
        tool_name: str,
        args: dict,
        user_task: str,
        current_result: dict | None,
        last_result: dict | None,
    ) -> dict | None:
        if tool_name == "retrieve_passages":
            return self.tools.retrieve_passages(
                query=args.get("query", user_task),
                k=args.get("k", 5),
                target_documents=args.get("target_documents"),
            )
        if tool_name == "answer_question":
            return self.tools.answer_question(user_task, current_result or {})
        if tool_name == "summarize_document":
            return self.tools.summarize_document(args.get("doc_name"), detail_level=self.detail_level)
        if tool_name == "extract_key_points":
            return self.tools.extract_key_points(args.get("doc_name"), detail_level=self.detail_level)
        if tool_name == "generate_faq":
            return self.tools.generate_faq(args.get("doc_name"), detail_level=self.detail_level)
        if tool_name == "generate_quiz":
            return self.tools.generate_quiz(
                args.get("doc_name"),
                args.get("num_questions", 10),
                detail_level=self.detail_level,
            )
        if tool_name == "compare_documents":
            return self.tools.compare_documents(args.get("doc_a"), args.get("doc_b"))
        if tool_name == "analyze_jd":
            return self.tools.analyze_jd(args.get("doc_name"), detail_level=self.detail_level)
        if tool_name == "extract_risks":
            return self.tools.extract_risks(args.get("doc_name"))
        if tool_name == "create_study_guide":
            return self.tools.create_study_guide(args.get("doc_name"))
        if tool_name == "export_to_markdown":
            content = last_result["answer"] if last_result else "暂无可导出的 Agent 结果。"
            return self.tools.export_to_markdown(content)
        if tool_name == "evaluate_answer":
            answer = current_result.get("content", "") if current_result else ""
            sources = current_result.get("sources", []) if current_result else []
            evaluation = self.tools.evaluate_answer(answer, sources)
            return {
                "content": answer or "文档中未找到相关信息。",
                "sources": sources,
                "evaluation": evaluation,
                "summary": json.dumps(evaluation, ensure_ascii=False),
            }
        return {
            "content": "我暂时无法识别这个任务。你可以尝试输入：总结文档、生成 FAQ、生成复习题、分析 JD、提取风险点或直接提问。",
            "sources": [],
            "summary": "未知任务。",
        }

    @staticmethod
    def _dedupe_sources(sources: list[dict]) -> list[dict]:
        deduped = []
        seen = set()
        for source in sources:
            normalized = DocumentAgent._normalize_source(source)
            if not normalized:
                continue
            key = (
                normalized.get("file_name"),
                normalized.get("page"),
            )
            if key not in seen:
                deduped.append(normalized)
                seen.add(key)
        return deduped

    @staticmethod
    def _normalize_source(source: dict) -> dict:
        if not isinstance(source, dict):
            return {}

        return {
            "file_name": source.get("file_name") or source.get("file") or "未知文件",
            "page": source.get("page", "未知页码"),
            "snippet": source.get("snippet", ""),
        }

    @staticmethod
    def _result(
        user_task: str,
        router_result: dict,
        plan: list[dict],
        tool_calls: list[dict],
        answer: str,
        sources: list[dict],
        evaluation: dict,
        start_time: float,
        metrics: dict,
    ) -> dict:
        return {
            "user_task": user_task,
            "router_result": router_result,
            "plan": plan,
            "tool_calls": tool_calls,
            "answer": answer,
            "sources": sources,
            "evaluation": evaluation,
            "response_time": round(time.perf_counter() - start_time, 2),
            "gemini_calls": metrics.get("gemini_calls", 0),
            "retry_count": metrics.get("retry_count", 0),
            "retry_events": metrics.get("retry_events", []),
            "errors": metrics.get("errors", []),
            "selected_model": metrics.get("selected_model"),
            "actual_model": metrics.get("actual_model"),
            "model_fallback": metrics.get("model_fallback", False),
            "fallback_reason": metrics.get("fallback_reason"),
            "fallback_model": metrics.get("fallback_model"),
            "detail_level": metrics.get("detail_level") or metrics.get("output_detail"),
            "output_detail": metrics.get("output_detail"),
            "cache_hit": False,
        }
