"""Guardrail node for checking input length and prompt injection."""

import re
import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState


def guardrail_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    question = state.get("question", "")
    cfg = deps.config.app.graph.guardrail

    is_blocked = False
    note = "ok"

    if len(question) > cfg.max_chars:
        is_blocked = True
        note = f"blocked: length {len(question)} > {cfg.max_chars}"
    else:
        for pat in cfg.patterns:
            if re.search(pat, question, re.IGNORECASE):
                is_blocked = True
                note = f"blocked: pattern '{pat}'"
                break

    duration_ms = (time.perf_counter() - start) * 1000
    updates: dict[str, Any] = {
        "trace": [TraceEvent(node="guardrail", ms=duration_ms, note=note)]
    }

    if is_blocked:
        updates["intent"] = "blocked"

    return updates
