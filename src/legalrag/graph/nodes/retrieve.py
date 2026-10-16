"""Retrieve node calling Retriever."""

import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState


def retrieve_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    query = state.get("query") or state.get("standalone_question") or state.get("question", "")
    scope = state.get("scope", "corpus")
    session_id = state.get("session_id")

    contexts = deps.retriever.retrieve(
        query=query,
        scope=scope,
        session_id=session_id,
    )

    top_score = contexts[0].best_score if contexts else -999.0

    ms = (time.perf_counter() - start) * 1000
    return {
        "contexts": contexts,
        "top_score": top_score,
        "trace": [TraceEvent(node="retrieve", ms=ms, note=f"retrieved {len(contexts)} contexts, top_score={top_score:.2f}")],
    }
