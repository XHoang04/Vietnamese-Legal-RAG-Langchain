"""LangGraph builder connecting all nodes, conditional edges, and dependencies."""

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from legalrag.graph.checkpointer import build_checkpointer
from legalrag.graph.deps import GraphDeps
from legalrag.graph.edges import (
    route_after_grade,
    route_after_guardrail,
    route_after_router,
    route_after_verify,
)
from legalrag.graph.nodes.clarify import clarify_node
from legalrag.graph.nodes.finalize import finalize_node
from legalrag.graph.nodes.generate import generate_node
from legalrag.graph.nodes.give_up import give_up_node
from legalrag.graph.nodes.grade import grade_node
from legalrag.graph.nodes.guardrail import guardrail_node
from legalrag.graph.nodes.respond import (
    respond_chitchat_node,
    respond_out_of_scope_node,
    respond_refusal_node,
)
from legalrag.graph.nodes.retrieve import retrieve_node
from legalrag.graph.nodes.rewrite import rewrite_node
from legalrag.graph.nodes.router import router_node
from legalrag.graph.nodes.validity_check import validity_check_node
from legalrag.graph.nodes.verify import verify_node
from legalrag.graph.state import RAGState
from legalrag.graph.subgraphs.multi_step import multi_step_subgraph_node


def build_graph(deps: GraphDeps, checkpointer: Any = None) -> CompiledStateGraph:
    """Build and compile the Legal RAG StateGraph with injected dependencies."""
    workflow = StateGraph(RAGState)

    # Wrap nodes with injected dependencies using partial
    workflow.add_node("guardrail", partial(guardrail_node, deps=deps))
    workflow.add_node("router", partial(router_node, deps=deps))
    workflow.add_node("respond_refusal", partial(respond_refusal_node, deps=deps))
    workflow.add_node("respond_chitchat", partial(respond_chitchat_node, deps=deps))
    workflow.add_node("respond_out_of_scope", partial(respond_out_of_scope_node, deps=deps))
    workflow.add_node("clarify", partial(clarify_node, deps=deps))
    workflow.add_node("retrieve", partial(retrieve_node, deps=deps))
    workflow.add_node("multi_step_subgraph", partial(multi_step_subgraph_node, deps=deps))
    workflow.add_node("grade", partial(grade_node, deps=deps))
    workflow.add_node("rewrite", partial(rewrite_node, deps=deps))
    workflow.add_node("validity_check", partial(validity_check_node, deps=deps))
    workflow.add_node("generate", partial(generate_node, deps=deps))
    workflow.add_node("verify", partial(verify_node, deps=deps))
    workflow.add_node("give_up", partial(give_up_node, deps=deps))
    workflow.add_node("finalize", partial(finalize_node, deps=deps))

    # Edges
    workflow.add_edge(START, "guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "respond_refusal": "respond_refusal",
            "router": "router",
        },
    )

    workflow.add_conditional_edges(
        "router",
        partial(route_after_router, deps=deps),
        {
            "respond_chitchat": "respond_chitchat",
            "respond_out_of_scope": "respond_out_of_scope",
            "clarify": "clarify",
            "retrieve": "retrieve",
            "multi_step_subgraph": "multi_step_subgraph",
        },
    )

    # After clarify, loop back to router
    workflow.add_edge("clarify", "router")

    # After response nodes, end
    workflow.add_edge("respond_refusal", END)
    workflow.add_edge("respond_chitchat", END)
    workflow.add_edge("respond_out_of_scope", END)

    # Retrieval flows to grade
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("multi_step_subgraph", "grade")

    # Grade branching
    workflow.add_conditional_edges(
        "grade",
        partial(route_after_grade, deps=deps),
        {
            "validity_check": "validity_check",
            "rewrite": "rewrite",
            "give_up": "give_up",
        },
    )

    # Rewrite loops back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Generation pipeline
    workflow.add_edge("validity_check", "generate")
    workflow.add_edge("generate", "verify")

    # Verification branching
    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "finalize": "finalize",
            "generate": "generate",
            "give_up": "give_up",
        },
    )

    workflow.add_edge("finalize", END)
    workflow.add_edge("give_up", END)

    if checkpointer is None:
        checkpointer = build_checkpointer(deps.config.app)

    return workflow.compile(checkpointer=checkpointer)
