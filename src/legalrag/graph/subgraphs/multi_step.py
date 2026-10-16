"""Multi-step query decomposition and parallel retrieval subgraph."""

import time
from typing import Any

from legalrag.core.telemetry import TraceEvent
from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState
from legalrag.llm.structured import PlannerOutput, invoke_structured
from legalrag.retrieval.retriever import RetrievedContext


def multi_step_subgraph_node(state: RAGState, deps: GraphDeps) -> dict[str, Any]:
    """Decompose complex query into sub-questions and merge retrieved contexts."""
    start = time.perf_counter()
    question = state.get("standalone_question") or state.get("question", "")
    scope = state.get("scope", "corpus")
    session_id = state.get("session_id")
    max_subs = deps.config.app.graph.multi_step.max_sub_questions

    llm = deps.llm_factory.get_llm("planner", contains_upload=state.get("has_upload", False))

    prompt = (
        f"Câu hỏi pháp lý phức tạp: {question}\n"
        f"Hãy chia nhỏ câu hỏi này thành tối đa {max_subs} câu hỏi con độc lập để tra cứu song song.\n"
        "Trả về PlannerOutput."
    )

    planner_out: PlannerOutput = invoke_structured(llm, PlannerOutput, [{"role": "user", "content": prompt}])
    sub_questions = planner_out.sub_questions or [question]
    sub_questions = sub_questions[:max_subs]

    # Retrieve for each sub-question
    all_contexts: dict[str, RetrievedContext] = {}
    for sq in sub_questions:
        res = deps.retriever.retrieve(sq, scope=scope, session_id=session_id)
        for c in res:
            c.from_subquestion = sq
            if c.parent_id not in all_contexts or c.best_score > all_contexts[c.parent_id].best_score:
                all_contexts[c.parent_id] = c

    # Sort merged contexts by score descending and prune by token budget
    sorted_contexts = sorted(all_contexts.values(), key=lambda x: x.best_score, reverse=True)
    budget = getattr(deps.config.retrieval, "context_token_budget", 4000)
    pruned_contexts: list[RetrievedContext] = []
    total_tokens = 0
    for c in sorted_contexts:
        approx_tokens = max(1, len(c.text.split()))
        if total_tokens + approx_tokens > budget and pruned_contexts:
            break
        pruned_contexts.append(c)
        total_tokens += approx_tokens

    top_score = pruned_contexts[0].best_score if pruned_contexts else -999.0

    ms = (time.perf_counter() - start) * 1000
    return {
        "contexts": pruned_contexts,
        "top_score": top_score,
        "llm_calls": 1,
        "trace": [TraceEvent(node="multi_step_subgraph", ms=ms, note=f"sub_questions={len(sub_questions)}, merged={len(pruned_contexts)}")],
    }

