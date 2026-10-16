"""Graph state definitions for Legal RAG."""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from legalrag.core.telemetry import TraceEvent
from legalrag.data.models import Citation
from legalrag.llm.structured import GenerationOutput, VerificationOutput
from legalrag.retrieval.retriever import RetrievedContext
from legalrag.validity.checker import ValidityInfo


class RAGState(TypedDict, total=False):
    question: str
    thread_id: str
    session_id: str | None
    has_upload: bool
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Literal["legal_query", "chitchat", "out_of_scope", "needs_clarification", "blocked"]
    scope: Literal["corpus", "upload", "both"]
    complexity: Literal["simple", "multi_step"]
    standalone_question: str
    clarifying_question: str | None
    clarifications_used: int
    query: str
    retries: int
    contexts: list[RetrievedContext]
    grade: Literal["ok", "insufficient"] | None
    top_score: float | None
    validity: dict[str, ValidityInfo]
    draft: GenerationOutput | None
    verification: VerificationOutput | None
    regen_count: int
    answer: str
    citations: list[Citation]
    warnings: list[str]
    trace: Annotated[list[TraceEvent], operator.add]
    llm_calls: Annotated[int, operator.add]
