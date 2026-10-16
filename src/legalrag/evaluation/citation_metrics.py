"""Citation accuracy and citation support metrics."""

import re

from legalrag.data.models import Citation
from legalrag.evaluation.datasets import GoldenArticle
from legalrag.retrieval.retriever import RetrievedContext


def compute_citation_accuracy(
    citations: list[Citation],
    gold_articles: list[GoldenArticle],
) -> float:
    """Compute proportion of citations [n] that point to a ground-truth gold article."""
    if not citations:
        return 1.0 if not gold_articles else 0.0

    gold_set = {(g.doc_id, str(g.article_no)) for g in gold_articles if g.article_no is not None}
    if not gold_set:
        return 1.0

    correct = 0
    for cit in citations:
        pair = (cit.doc_id, str(cit.article_no)) if cit.article_no is not None else None
        if pair and pair in gold_set:
            correct += 1

    return float(correct / len(citations))


def compute_citation_support(
    answer: str,
    citations: list[Citation],
    contexts: list[RetrievedContext],
) -> float:
    """Check whether specific Article numbers mentioned in answer are supported by cited contexts."""
    mentioned_articles = set(re.findall(r"Điều\s+(\d+[a-zA-Z]?)", answer))
    if not mentioned_articles:
        return 1.0

    cited_indices = [getattr(c, "n", getattr(c, "ref_index", 1)) for c in citations]
    cited_contexts = [contexts[idx - 1] for idx in cited_indices if 1 <= idx <= len(contexts)]
    cited_full_text = " ".join(c.text for c in cited_contexts)

    supported = 0
    for art in mentioned_articles:
        if f"Điều {art}" in cited_full_text or f"Điều{art}" in cited_full_text:
            supported += 1

    return float(supported / len(mentioned_articles))
