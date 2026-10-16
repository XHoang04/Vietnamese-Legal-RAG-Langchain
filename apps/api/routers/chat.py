"""Chat SSE streaming endpoints."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from apps.api.auth import verify_api_key
from apps.api.dependencies import get_graph, get_session_manager, rate_limit_dependency
from apps.api.schemas import ChatRequest, ResumeRequest
from legalrag.core.errors import LegalRAGError
from legalrag.sessions.manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit_dependency)],
)


async def _run_and_stream(
    graph: CompiledStateGraph,
    inputs_or_command: Any,
    graph_config: dict[str, Any],
    thread_id: str,
    session_id: str,
) -> AsyncGenerator[dict[str, str], None]:
    """Execute LangGraph execution and yield Server-Sent Events (SSE)."""
    start_time = time.perf_counter()
    nodes_executed: list[str] = []
    final_state_values: dict[str, Any] = {}

    try:
        # Run graph in a separate thread to prevent blocking asyncio event loop
        loop = asyncio.get_running_loop()

        def _execute_step_by_step():
            step_results = []
            for chunk in graph.stream(inputs_or_command, config=graph_config, stream_mode="updates"):
                step_results.append(chunk)
            return step_results

        # Execute stream
        chunks = await loop.run_in_executor(None, _execute_step_by_step)

        for chunk in chunks:
            for node_name, updates in chunk.items():
                nodes_executed.append(node_name)
                # Emit step end event
                yield {
                    "event": "step",
                    "data": json.dumps({"node": node_name, "status": "end"}),
                }
                if isinstance(updates, dict):
                    final_state_values.update(updates)

        # Check if the execution interrupted (e.g. clarifying question)
        state_snapshot = graph.get_state(graph_config)
        if state_snapshot.tasks and len(state_snapshot.tasks) > 0:
            task = state_snapshot.tasks[0]
            if task.interrupts:
                interrupt_data = task.interrupts[0].value
                q = interrupt_data.get("question", "Vui lòng làm rõ yêu cầu của bạn.") if isinstance(interrupt_data, dict) else str(interrupt_data)
                yield {
                    "event": "interrupt",
                    "data": json.dumps({"thread_id": thread_id, "question": q}),
                }
                return

        # Read state values from snapshot
        state_values = state_snapshot.values if state_snapshot else final_state_values
        citations = state_values.get("citations", [])
        warnings = state_values.get("warnings", [])
        answer = state_values.get("answer", "")
        llm_calls = state_values.get("llm_calls", 0)

        # 1. Emit citations event
        if citations:
            citations_data = [c if isinstance(c, dict) else c.model_dump() for c in citations]
            yield {
                "event": "citations",
                "data": json.dumps(citations_data, ensure_ascii=False),
            }

        # 2. Emit warnings event
        if warnings:
            yield {
                "event": "warnings",
                "data": json.dumps(warnings, ensure_ascii=False),
            }

        # 3. Emit tokens in simulated chunks (ADR 004: tokens emitted after verification)
        if answer:
            words = answer.split(" ")
            for i, w in enumerate(words):
                prefix = "" if i == 0 else " "
                yield {
                    "event": "token",
                    "data": json.dumps({"text": prefix + w}, ensure_ascii=False),
                }

        # 4. Emit done event
        total_latency = (time.perf_counter() - start_time) * 1000
        yield {
            "event": "done",
            "data": json.dumps({
                "thread_id": thread_id,
                "session_id": session_id,
                "metrics": {
                    "latency_ms": round(total_latency, 2),
                    "llm_calls": llm_calls,
                    "nodes": nodes_executed,
                },
            }, ensure_ascii=False),
        }

    except LegalRAGError as lre:
        logger.error(f"Domain error during chat stream: {lre.code} - {lre.message}")
        yield {
            "event": "error",
            "data": json.dumps({"code": lre.code, "message": lre.message}, ensure_ascii=False),
        }
    except Exception as e:
        logger.exception("Unexpected error during chat stream")
        yield {
            "event": "error",
            "data": json.dumps({"code": "INTERNAL_ERROR", "message": str(e)}, ensure_ascii=False),
        }


@router.post(
    "",
    summary="Chat with Legal RAG assistant via SSE",
)
async def chat_stream(
    request: ChatRequest,
    graph: CompiledStateGraph = Depends(get_graph),
    session_manager: SessionManager = Depends(get_session_manager),
) -> EventSourceResponse:
    """
    Initiate conversation with Legal RAG assistant.
    Returns a stream of Server-Sent Events (SSE).
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"

    # Determine if upload exists in session
    session_docs = session_manager.list_session_documents(session_id)
    has_upload = len(session_docs) > 0 or bool(request.document_ids)

    graph_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    inputs = {
        "question": request.message,
        "thread_id": thread_id,
        "session_id": session_id,
        "has_upload": has_upload,
        "messages": [],
    }

    return EventSourceResponse(
        _run_and_stream(graph, inputs, graph_config, thread_id, session_id),
        media_type="text/event-stream",
    )


@router.post(
    "/{thread_id}/resume",
    summary="Resume interrupted conversation thread",
)
async def resume_chat_stream(
    thread_id: str,
    request: ResumeRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> EventSourceResponse:
    """
    Resume conversation after a clarifying question (LangGraph interrupt).
    Returns SSE events.
    """
    graph_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    state_snapshot = graph.get_state(graph_config)
    if not state_snapshot or not state_snapshot.tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "THREAD_NOT_FOUND", "message": f"Không tìm thấy luồng hội thoại {thread_id} đang chờ làm rõ."},
        )

    session_id = state_snapshot.values.get("session_id", "default")
    command = Command(resume=request.answer)

    return EventSourceResponse(
        _run_and_stream(graph, command, graph_config, thread_id, session_id),
        media_type="text/event-stream",
    )
