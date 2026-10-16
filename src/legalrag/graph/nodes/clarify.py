"""Clarification node using LangGraph interrupt."""

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState


def clarify_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    clarifying_q = state.get("clarifying_question", "Bạn có thể nói rõ hơn yêu cầu tra cứu không?")
    clarifications_used = state.get("clarifications_used", 0)

    # Interrupt and wait for user clarification response
    user_response = interrupt({
        "thread_id": state.get("thread_id", ""),
        "question": clarifying_q,
    })

    # When resumed, user_response is provided
    answer_text = str(user_response) if user_response else ""

    ms = (time.perf_counter() - start) * 1000
    return {
        "clarifications_used": clarifications_used + 1,
        "question": answer_text,
        "messages": [HumanMessage(content=answer_text)],
        "trace": [TraceEvent(node="clarify", ms=ms, note=f"clarified: {answer_text[:50]}")],
    }
