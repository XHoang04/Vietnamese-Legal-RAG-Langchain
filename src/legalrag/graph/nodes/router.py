"""Router node classifying intent, scope, and complexity."""

import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState
from legalrag.llm.structured import RouterOutput, invoke_structured


def router_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    question = state.get("question", "")
    has_upload = state.get("has_upload", False)
    clarifications_used = state.get("clarifications_used", 0)
    max_clarifications = deps.config.app.graph.max_clarifications

    llm = deps.llm_factory.get_llm("router", contains_upload=has_upload)

    history_msgs = state.get("messages", [])
    history_str = "\n".join(
        f"{m.type}: {m.content}" for m in history_msgs if hasattr(m, "content")
    )

    prompt = (
        f"Lĩnh vực: {deps.config.domain.domain}\n"
        f"Lịch sử hội thoại:\n{history_str}\n"
        f"Người dùng có tài liệu upload: {has_upload}\n"
        f"Câu hỏi: {question}\n"
        "Phân loại câu hỏi thành RouterOutput."
    )

    router_out: RouterOutput = invoke_structured(
        llm, RouterOutput, [{"role": "user", "content": prompt}]
    )

    intent = router_out.intent
    scope = router_out.scope
    complexity = router_out.complexity
    standalone = router_out.standalone_question or question
    if not standalone or standalone.strip() in ["câu hỏi", ""]:
        standalone = question
    clarifying_q = router_out.clarifying_question

    # Enforce rules by code
    if not has_upload:
        scope = "corpus"

    if clarifications_used >= max_clarifications and intent == "needs_clarification":
        intent = "legal_query"
        clarifying_q = None

    duration_ms = (time.perf_counter() - start) * 1000
    return {
        "intent": intent,
        "scope": scope,
        "complexity": complexity,
        "standalone_question": standalone,
        "query": standalone,
        "clarifying_question": clarifying_q,
        "llm_calls": 1,
        "trace": [TraceEvent(node="router", ms=duration_ms, note=f"intent={intent}, scope={scope}, comp={complexity}")],
    }
