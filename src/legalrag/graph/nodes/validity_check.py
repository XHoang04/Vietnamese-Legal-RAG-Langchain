"""Validity check node checking document status and relation warnings."""

import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState
from legalrag.validity import templates
from legalrag.validity.checker import ValidityInfo, expand_replacements


def validity_check_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    start = time.perf_counter()
    contexts = state.get("contexts", [])
    validity_cfg = deps.config.app.validity

    validity_map: dict[str, ValidityInfo] = {}

    if validity_cfg.enabled:
        corpus_doc_ids = list({c.doc_id for c in contexts if c.source == "corpus"})
        if corpus_doc_ids:
            validity_map = deps.validity_checker.check(corpus_doc_ids)

        # Upload docs are not verified in corpus SQL
        for c in contexts:
            if c.source == "upload" and c.doc_id not in validity_map:
                validity_map[c.doc_id] = ValidityInfo(
                    doc_id=c.doc_id,
                    status=None,  # type: ignore[arg-type]
                    warnings=[templates.warn_upload()],
                )

        # Section 11.3: Expand replacements if flag is enabled
        if getattr(validity_cfg, "expand_replacements", False):
            query = state.get("query") or state.get("standalone_question") or state.get("question", "")
            contexts = expand_replacements(
                retriever=deps.retriever,
                query=query,
                contexts=contexts,
                validity_map=validity_map,
                max_parents=2,
            )

    ms = (time.perf_counter() - start) * 1000
    return {
        "contexts": contexts,
        "validity": validity_map,
        "trace": [TraceEvent(node="validity_check", ms=ms, note=f"checked {len(validity_map)} docs")],
    }

