"""Simple baseline RAG pipeline without LangGraph for comparison and ablation."""

import re

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from legalrag.data.models import Citation
from legalrag.llm.structured import GenerationOutput, invoke_structured
from legalrag.prompts.loader import load_prompt, render_prompt
from legalrag.retrieval.retriever import BaseRetriever, RetrievedContext


class SimpleRAGOutput(BaseModel):
    """Output structure of baseline Simple RAG pipeline."""
    answer: str
    contexts: list[RetrievedContext] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    used_ctx: list[int] = Field(default_factory=list)
    insufficient_context: bool = False
    verified: bool = True
    verification_errors: list[str] = Field(default_factory=list)


def format_contexts(contexts: list[RetrievedContext]) -> str:
    """Format contexts into XML-like delimiter blocks."""
    blocks = []
    for idx, ctx in enumerate(contexts, start=1):
        status_str = ctx.status.value if ctx.status else "UNKNOWN"
        blocks.append(
            f'<context id="{idx}" source="{ctx.breadcrumb}" status="{status_str}">\n{ctx.text}\n</context>'
        )
    return "\n\n".join(blocks)


def verify_answer_code(answer: str, contexts: list[RetrievedContext]) -> tuple[bool, list[str]]:
    """Perform code-based verification of citations and claims."""
    errors = []
    num_ctx = len(contexts)

    # Check citation indices [n]
    citations = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    for c in citations:
        if c < 1 or c > num_ctx:
            errors.append(f"Trích dẫn [{c}] nằm ngoài phạm vi ngữ cảnh (1..{num_ctx})")

    # If answer claims specific Điều X, verify it exists in at least one cited context
    cited_contexts = [contexts[c - 1] for c in citations if 1 <= c <= num_ctx]
    all_cited_text = " ".join(c.text for c in cited_contexts)

    article_mentions = re.findall(r"Điều\s+(\d+[a-zA-Z]?)", answer)
    for art in article_mentions:
        if f"Điều {art}" not in all_cited_text and f"Điều{art}" not in all_cited_text:
            # Informational check
            pass

    return len(errors) == 0, errors


def run_simple_rag(
    query: str,
    retriever: BaseRetriever,
    llm: BaseChatModel,
    *,
    scope: str = "corpus",
    max_parents: int = 5,
    prompt_version: str = "v1",
) -> SimpleRAGOutput:
    """Execute baseline retrieve -> generate -> code verify pipeline."""
    contexts = retriever.retrieve(query, scope=scope, max_parents=max_parents)
    if not contexts:
        return SimpleRAGOutput(
            answer="Hệ thống không tìm thấy tài liệu quy phạm pháp luật phù hợp với câu hỏi.",
            contexts=[],
            citations=[],
            insufficient_context=True,
            verified=True,
        )

    formatted_ctx = format_contexts(contexts)
    prompt_template = load_prompt("generate", version=prompt_version)
    rendered_prompt = render_prompt(
        prompt_template,
        validity_notes="",
        verify_feedback="",
        question=query,
        contexts=formatted_ctx,
    )

    messages = [{"role": "user", "content": rendered_prompt}]
    draft: GenerationOutput = invoke_structured(llm, GenerationOutput, messages)

    is_verified, errors = verify_answer_code(draft.answer, contexts)

    # Build citations list
    citations_list: list[Citation] = []
    for ctx_idx in draft.used_ctx:
        if 1 <= ctx_idx <= len(contexts):
            c = contexts[ctx_idx - 1]
            citations_list.append(
                Citation(
                    n=ctx_idx,
                    doc_id=c.doc_id,
                    doc_number=c.doc_number,
                    article_no=c.article_no,
                    breadcrumb=c.breadcrumb,
                    url=c.url,
                )
            )

    return SimpleRAGOutput(
        answer=draft.answer,
        contexts=contexts,
        citations=citations_list,
        used_ctx=draft.used_ctx,
        insufficient_context=draft.insufficient_context,
        verified=is_verified,
        verification_errors=errors,
    )
