"""Rewrite node generating improved query when retrieved context was insufficient."""

import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState
from legalrag.llm.structured import RewriteOutput, invoke_structured


def rewrite_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    retries = state.get("retries", 0)
    current_query = state.get("query") or state.get("standalone_question") or state.get("question", "")

    llm = deps.llm_factory.get_llm("rewriter", contains_upload=state.get("has_upload", False))

    prompt = (
        f"Truy vấn hiện tại: {current_query}\n"
        "Truy vấn trên chưa tìm được đủ điều khoản pháp luật phù hợp.\n"
        "Hãy viết lại truy vấn, sử dụng thêm thuật ngữ pháp lý chuẩn xác hoặc từ đồng nghĩa để tra cứu tốt hơn.\n"
        "Trả về RewriteOutput."
    )

    out: RewriteOutput = invoke_structured(llm, RewriteOutput, [{"role": "user", "content": prompt}])
    new_query = out.query or current_query

    ms = (time.perf_counter() - start) * 1000
    return {
        "query": new_query,
        "retries": retries + 1,
        "llm_calls": 1,
        "trace": [TraceEvent(node="rewrite", ms=ms, note=f"retry={retries + 1}, new_query={new_query[:50]}")],
    }
