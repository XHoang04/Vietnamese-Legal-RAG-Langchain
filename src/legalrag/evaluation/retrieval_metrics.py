"""Retrieval metrics evaluation: Recall@k, MRR@k, nDCG@k at Article level."""

import math
from typing import Sequence

from legalrag.evaluation.datasets import GoldenArticle
from legalrag.retrieval.retriever import RetrievedContext


def compute_retrieval_metrics_for_query(
    retrieved: Sequence[RetrievedContext],
    gold_articles: Sequence[GoldenArticle],
    k_list: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Compute Article-level Recall@k, MRR@10, and nDCG@10 for a single query.

    A retrieved parent context is considered a hit if its (doc_id, article_no) matches
    any gold article pair.
    """
    gold_set = {(g.doc_id, str(g.article_no)) for g in gold_articles if g.article_no is not None}
    if not gold_set:
        return {}

    metrics: dict[str, float] = {}

    # Extract ranked unique (doc_id, article_no) from retrieved parents
    ranked_hits: list[int] = []
    seen_articles = set()

    for ctx in retrieved:
        art_key = (ctx.doc_id, str(ctx.article_no)) if ctx.article_no is not None else None
        if art_key and art_key not in seen_articles:
            seen_articles.add(art_key)
            ranked_hits.append(1 if art_key in gold_set else 0)
        elif not art_key:
            ranked_hits.append(0)

    # Compute Recall@k
    for k in k_list:
        hits_in_k = set()
        for ctx in retrieved[:k]:
            if ctx.article_no is not None:
                pair = (ctx.doc_id, str(ctx.article_no))
                if pair in gold_set:
                    hits_in_k.add(pair)
        metrics[f"recall@{k}"] = len(hits_in_k) / len(gold_set)

    # Compute MRR@10
    mrr_10 = 0.0
    for rank, hit in enumerate(ranked_hits[:10], start=1):
        if hit == 1:
            mrr_10 = 1.0 / rank
            break
    metrics["mrr@10"] = mrr_10

    # Compute nDCG@10
    dcg_10 = 0.0
    for rank, hit in enumerate(ranked_hits[:10], start=1):
        if hit == 1:
            dcg_10 += 1.0 / math.log2(rank + 1)

    idcg_10 = 0.0
    for rank in range(1, min(len(gold_set), 10) + 1):
        idcg_10 += 1.0 / math.log2(rank + 1)

    metrics["ndcg@10"] = (dcg_10 / idcg_10) if idcg_10 > 0 else 0.0

    return metrics


def aggregate_retrieval_metrics(query_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-query metric dictionaries into mean scores."""
    if not query_metrics:
        return {}

    all_keys = set().union(*query_metrics)
    summary: dict[str, float] = {}

    for key in sorted(all_keys):
        vals = [qm[key] for qm in query_metrics if key in qm]
        summary[key] = float(sum(vals) / len(vals)) if vals else 0.0

    return summary
