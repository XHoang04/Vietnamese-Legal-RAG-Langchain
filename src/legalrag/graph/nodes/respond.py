"""Static / fast response nodes for refusal, chitchat, and out of scope."""

import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState


def respond_refusal_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    answer = "Yêu cầu của bạn đã bị từ chối do không phù hợp với tiêu chuẩn an toàn hoặc có dấu hiệu can thiệp hệ thống."
    ms = (time.perf_counter() - start) * 1000
    return {
        "answer": answer,
        "citations": [],
        "warnings": [],
        "trace": [TraceEvent(node="respond_refusal", ms=ms, note="blocked")],
    }


def respond_chitchat_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    answer = "Xin chào! Tôi là trợ lý tra cứu văn bản pháp luật Việt Nam. Tôi có thể giúp gì cho bạn về các quy định pháp luật?"
    ms = (time.perf_counter() - start) * 1000
    return {
        "answer": answer,
        "citations": [],
        "warnings": [],
        "trace": [TraceEvent(node="respond_chitchat", ms=ms, note="chitchat")],
    }


def respond_out_of_scope_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    domain = deps.config.domain.domain
    answer = f"Câu hỏi của bạn nằm ngoài phạm vi dữ liệu pháp luật hiện tại của hệ thống (lĩnh vực: {domain}). Vui lòng đặt câu hỏi liên quan đến lĩnh vực này."
    ms = (time.perf_counter() - start) * 1000
    return {
        "answer": answer,
        "citations": [],
        "warnings": [],
        "trace": [TraceEvent(node="respond_out_of_scope", ms=ms, note="out_of_scope")],
    }
