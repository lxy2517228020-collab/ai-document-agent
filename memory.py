from datetime import datetime

from rag import empty_knowledge_base


FEEDBACK_LABELS = {
    "helpful": "有帮助",
    "unhelpful": "没帮助",
    "inaccurate": "回答不准确",
    "weak_citation": "引用不足",
    "too_long": "太长",
    "too_short": "太短",
}


def default_stats() -> dict:
    """Return dashboard counters."""
    return {
        "total_tasks": 0,
        "helpful": 0,
        "unhelpful": 0,
        "inaccurate": 0,
        "weak_citation": 0,
        "too_long": 0,
        "too_short": 0,
        "tool_calls": 0,
        "gemini_calls": 0,
        "retry_count": 0,
        "cache_hits": 0,
        "response_times": [],
    }


def init_session_state(st) -> None:
    """Initialize Streamlit session_state."""
    defaults = {
        "knowledge_base": empty_knowledge_base(),
        "task_history": [],
        "stats": default_stats(),
        "last_agent_result": None,
        "pending_clear_kb": False,
        "knowledge_base_cache": {},
        "agent_result_cache": {},
        "model_mode": "稳定模式",
        "detail_level": "简洁版",
        "output_detail": "简洁版",
        "fast_routing": True,
        "deep_evaluation": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.session_state.output_detail = st.session_state.detail_level


def add_task_history(st, result: dict) -> None:
    """Save one Agent result to history and update metrics."""
    record = {
        "id": len(st.session_state.task_history),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_task": result["user_task"],
        "intent": result["router_result"]["intent"],
        "router_result": result["router_result"],
        "plan": result["plan"],
        "tool_calls": result["tool_calls"],
        "answer": result["answer"],
        "sources": result["sources"],
        "evaluation": result["evaluation"],
        "response_time": result["response_time"],
        "gemini_calls": result.get("gemini_calls", 0),
        "retry_count": result.get("retry_count", 0),
        "cache_hit": result.get("cache_hit", False),
        "retry_events": result.get("retry_events", []),
        "errors": result.get("errors", []),
        "selected_model": result.get("selected_model"),
        "actual_model": result.get("actual_model"),
        "model_fallback": result.get("model_fallback", False),
        "fallback_reason": result.get("fallback_reason"),
        "fallback_model": result.get("fallback_model"),
        "detail_level": result.get("detail_level") or result.get("output_detail"),
        "output_detail": result.get("output_detail") or result.get("detail_level"),
        "feedback": None,
    }
    st.session_state.task_history.append(record)
    st.session_state.last_agent_result = record

    st.session_state.stats["total_tasks"] += 1
    st.session_state.stats["tool_calls"] += len(result.get("tool_calls", []))
    st.session_state.stats["gemini_calls"] += result.get("gemini_calls", 0)
    st.session_state.stats["retry_count"] += result.get("retry_count", 0)
    if result.get("cache_hit"):
        st.session_state.stats["cache_hits"] += 1
    st.session_state.stats["response_times"].append(result["response_time"])


def clear_history(st) -> None:
    """Clear history and feedback metrics but keep the knowledge base."""
    st.session_state.task_history = []
    st.session_state.last_agent_result = None
    st.session_state.stats = default_stats()


def set_feedback(st, record_id: int, feedback_key: str) -> None:
    """Record one feedback category per answer and keep counters consistent."""
    history = st.session_state.task_history
    if record_id < 0 or record_id >= len(history):
        return

    previous = history[record_id].get("feedback")
    if previous == feedback_key:
        return

    if previous in FEEDBACK_LABELS:
        st.session_state.stats[previous] = max(0, st.session_state.stats[previous] - 1)

    history[record_id]["feedback"] = feedback_key
    st.session_state.stats[feedback_key] += 1


def average_response_time(stats: dict) -> float:
    """Calculate average Agent response time."""
    response_times = stats.get("response_times", [])
    if not response_times:
        return 0.0

    return sum(response_times) / len(response_times)
