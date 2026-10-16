"""Conditional edge routing logic for the LangGraph pipeline."""

from typing import Literal

from legalrag.graph.deps import GraphDeps
from legalrag.graph.state import RAGState


def route_after_guardrail(state: RAGState) -> Literal["respond_refusal", "router"]:
    if state.get("intent") == "blocked":
        return "respond_refusal"
    return "router"


def route_after_router(state: RAGState, deps: GraphDeps) -> Literal[
    "respond_chitchat", "respond_out_of_scope", "clarify", "retrieve", "multi_step_subgraph"
]:
    intent = state.get("intent", "legal_query")
    complexity = state.get("complexity", "simple")

    if intent == "chitchat":
        return "respond_chitchat"
    if intent == "out_of_scope":
        return "respond_out_of_scope"
    if intent == "needs_clarification":
        return "clarify"

    # legal_query
    if complexity == "multi_step" and deps.config.app.graph.multi_step.enabled:
        return "multi_step_subgraph"

    return "retrieve"


def route_after_grade(state: RAGState, deps: GraphDeps) -> Literal["validity_check", "rewrite", "give_up"]:
    grade = state.get("grade")
    retries = state.get("retries", 0)
    max_retries = deps.config.app.graph.max_retries

    if grade == "ok":
        return "validity_check"

    if retries < max_retries:
        return "rewrite"

    return "give_up"


def route_after_verify(state: RAGState) -> Literal["finalize", "generate", "give_up"]:
    ver = state.get("verification")
    regen_count = state.get("regen_count", 0)

    # Passed verification if no unsupported claims
    if ver and len(ver.unsupported_claims) == 0:
        return "finalize"

    # Allow 1 regeneration attempt
    if regen_count <= 1:
        return "generate"

    return "give_up"
