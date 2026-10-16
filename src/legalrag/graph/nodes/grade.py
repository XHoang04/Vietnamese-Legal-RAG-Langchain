"""Grade node assessing relevance/sufficiency of retrieved contexts."""

import time
from typing import Any

from legalrag.core.errors import ThresholdNotCalibratedError
from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState
from legalrag.llm.structured import GradeOutput, invoke_structured


def grade_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    contexts = state.get("contexts", [])
    top_score = state.get("top_score", -999.0)
    cfg = deps.config.app.graph.grade
    mode = cfg.mode

    if not contexts:
        ms = (time.perf_counter() - start) * 1000
        return {
            "grade": "insufficient",
            "trace": [TraceEvent(node="grade", ms=ms, note="insufficient: 0 contexts")],
        }

    grade_result = "insufficient"
    llm_calls = 0

    if mode == "threshold":
        if cfg.tau_high is None:
            raise ThresholdNotCalibratedError(
                "tau_high threshold is null. Please calibrate thresholds using script 07 or set tau_high in config."
            )
        grade_result = "ok" if top_score >= cfg.tau_high else "insufficient"

    elif mode == "llm":
        llm = deps.llm_factory.get_llm("grader", contains_upload=state.get("has_upload", False))
        query = state.get("query") or state.get("question", "")
        ctx_text = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(contexts))
        prompt = (
            f"Câu hỏi: {query}\n\n"
            f"Các ngữ cảnh:\n{ctx_text}\n\n"
            "Đánh giá xem ngữ cảnh trên có đủ căn cứ trả lời câu hỏi không. Trả về GradeOutput."
        )
        out: GradeOutput = invoke_structured(llm, GradeOutput, [{"role": "user", "content": prompt}])
        llm_calls = 1
        grade_result = "ok" if out.sufficient else "insufficient"

    elif mode == "hybrid":
        if cfg.tau_high is None or cfg.tau_low is None:
            raise ThresholdNotCalibratedError(
                "tau_high and tau_low thresholds must be configured for hybrid grade mode."
            )
        if top_score >= cfg.tau_high:
            grade_result = "ok"
        elif top_score < cfg.tau_low:
            grade_result = "insufficient"
        else:
            # Borderline score: ask LLM
            llm = deps.llm_factory.get_llm("grader", contains_upload=state.get("has_upload", False))
            query = state.get("query") or state.get("question", "")
            ctx_text = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(contexts))
            prompt = (
                f"Câu hỏi: {query}\n\n"
                f"Các ngữ cảnh:\n{ctx_text}\n\n"
                "Đánh giá xem ngữ cảnh trên có đủ căn cứ trả lời câu hỏi không. Trả về GradeOutput."
            )
            out: GradeOutput = invoke_structured(llm, GradeOutput, [{"role": "user", "content": prompt}])
            llm_calls = 1
            grade_result = "ok" if out.sufficient else "insufficient"

    ms = (time.perf_counter() - start) * 1000
    updates: dict[str, Any] = {
        "grade": grade_result,
        "trace": [TraceEvent(node="grade", ms=ms, note=f"mode={mode}, result={grade_result}")],
    }
    if llm_calls > 0:
        updates["llm_calls"] = llm_calls

    return updates
