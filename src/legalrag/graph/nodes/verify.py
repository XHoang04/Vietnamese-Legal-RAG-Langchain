"""Verify node ensuring citations and claims are grounded in context."""

import re
import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState
from legalrag.llm.structured import VerificationOutput, invoke_structured


def verify_code_rules(answer: str, contexts: list[Any], insufficient_context: bool) -> list[str]:
    """Verify groundness using deterministic code checks."""
    unsupported = []
    num_contexts = len(contexts)

    # Find citations [n]
    citations = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)]

    # Rule (a): all citations within 1..num_contexts
    for c in citations:
        if c < 1 or c > num_contexts:
            unsupported.append(f"Chỉ số trích dẫn [{c}] không tồn tại trong danh sách ngữ cảnh.")

    # Rule (b): if not insufficient_context, must have at least one citation
    if not insufficient_context and not citations:
        unsupported.append("Câu trả lời không có trích dẫn nguồn [n].")

    # Collect texts of cited contexts
    cited_texts = []
    for c in citations:
        if 1 <= c <= num_contexts:
            cited_texts.append(contexts[c - 1].text)
    combined_cited = " ".join(cited_texts)

    # Rule (c): article references like 'Điều 5' must appear in cited context
    articles = re.findall(r"Điều\s+\d+[a-zA-Z]?", answer, re.IGNORECASE)
    for art in articles:
        if not re.search(re.escape(art), combined_cited, re.IGNORECASE):
            unsupported.append(f"Viện dẫn {art} không xuất hiện trong ngữ cảnh được trích dẫn.")

    return unsupported


def verify_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    draft = state.get("draft")
    contexts = state.get("contexts", [])
    regen_count = state.get("regen_count", 0)
    cfg = deps.config.app.graph.verify

    if not draft:
        ms = (time.perf_counter() - start) * 1000
        return {
            "verification": VerificationOutput(unsupported_claims=["Không có bản thảo câu trả lời."]),
            "regen_count": regen_count + 1,
            "trace": [TraceEvent(node="verify", ms=ms, note="failed: no draft")],
        }

    unsupported = verify_code_rules(draft.answer, contexts, draft.insufficient_context)
    llm_calls = 0

    if not unsupported and cfg.mode == "llm":
        # Additional NLI check via LLM
        llm = deps.llm_factory.get_llm("verifier", contains_upload=state.get("has_upload", False))
        ctx_text = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(contexts))
        prompt = (
            f"Ngữ cảnh:\n{ctx_text}\n\n"
            f"Câu trả lời:\n{draft.answer}\n\n"
            "Kiểm tra xem từng khẳng định trong câu trả lời có được chứng minh bởi ngữ cảnh không. "
            "Trả về VerificationOutput liệt kê unsupported_claims (để trống nếu tất cả đều có căn cứ)."
        )
        ver_out: VerificationOutput = invoke_structured(llm, VerificationOutput, [{"role": "user", "content": prompt}])
        llm_calls = 1
        unsupported.extend(ver_out.unsupported_claims)

    passed = len(unsupported) == 0
    new_regen_count = regen_count if passed else (regen_count + 1)

    ms = (time.perf_counter() - start) * 1000
    updates: dict[str, Any] = {
        "verification": VerificationOutput(unsupported_claims=unsupported),
        "regen_count": new_regen_count,
        "trace": [TraceEvent(node="verify", ms=ms, note=f"passed={passed}, claims_len={len(unsupported)}")],
    }
    if llm_calls > 0:
        updates["llm_calls"] = llm_calls

    return updates
