"""Give up node returning graceful failure message with closest documents found."""

import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState


def give_up_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    contexts = state.get("contexts", [])

    lines = [
        "Hiện tại hệ thống không tìm thấy đủ căn cứ pháp lý trong kho dữ liệu để trả lời chính xác câu hỏi của bạn.",
        "Gợi ý: bạn có thể thử đặt lại câu hỏi với số hiệu văn bản cụ thể hoặc sử dụng các từ khóa chuyên ngành.",
    ]

    if contexts:
        lines.append("\nMột số văn bản gần nhất được tìm thấy:")
        seen = set()
        for c in contexts[:3]:
            if c.doc_id not in seen:
                seen.add(c.doc_id)
                lines.append(f"- {c.title} ({c.breadcrumb})")

    answer = "\n".join(lines)
    ms = (time.perf_counter() - start) * 1000
    return {
        "answer": answer,
        "citations": [],
        "warnings": [],
        "trace": [TraceEvent(node="give_up", ms=ms, note="gave up")],
    }
