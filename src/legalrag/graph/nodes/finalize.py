"""Finalize node assembling final response, citations, and warnings strictly in code."""

import re
import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import Citation, RAGState
from legalrag.validity import templates


def finalize_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    draft = state.get("draft")
    contexts = state.get("contexts", [])
    validity_map = state.get("validity", {})
    snapshot_date = deps.config.app.snapshot_as_of

    body = draft.answer if draft else ""

    # Identify citations mentioned in body [n]
    cited_indices = sorted(list({int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", body)}))

    citations: list[Citation] = []
    warnings_list: list[str] = []

    for n in cited_indices:
        if 1 <= n <= len(contexts):
            ctx = contexts[n - 1]
            citation = Citation(
                n=n,
                doc_id=ctx.doc_id,
                title=ctx.title,
                doc_number=ctx.doc_number,
                article_no=ctx.article_no,
                breadcrumb=ctx.breadcrumb,
                url=ctx.url,
                status=ctx.status,
                source=ctx.source,
            )
            citations.append(citation)

            # Check validity warnings for this doc
            if ctx.doc_id in validity_map:
                for w in validity_map[ctx.doc_id].warnings:
                    if w not in warnings_list:
                        warnings_list.append(w)

    # Format assembled response
    response_sections = [body]

    if citations:
        response_sections.append("\n\n### Nguồn trích dẫn:")
        for cit in citations:
            url_part = f" - [Liên kết]({cit.url})" if cit.url else ""
            status_part = f" [{cit.status.value}]" if cit.status else ""
            response_sections.append(f"[{cit.n}] {cit.breadcrumb}{status_part}{url_part}")

    if warnings_list:
        response_sections.append("\n\n### Cảnh báo hiệu lực:")
        for w in warnings_list:
            response_sections.append(f"- ⚠️ {w}")

    # Snapshot disclaimer
    response_sections.append(f"\n\n_{templates.snapshot_notice(snapshot_date)}_")

    final_answer = "\n".join(response_sections)
    ms = (time.perf_counter() - start) * 1000
    return {
        "answer": final_answer,
        "citations": citations,
        "warnings": warnings_list,
        "trace": [TraceEvent(node="finalize", ms=ms, note=f"assembled {len(citations)} citations, {len(warnings_list)} warnings")],
    }
