"""Evaluation runner for retrieval metrics on Golden Set splits."""

import time
from typing import Any

import numpy as np

from legalrag.evaluation.datasets import GoldenQuestion
from legalrag.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics_for_query,
)
from legalrag.retrieval.retriever import BaseRetriever


def evaluate_retrieval(
    retriever: BaseRetriever,
    questions: list[GoldenQuestion],
    max_parents: int = 10,
) -> dict[str, Any]:
    """Run retrieval evaluation across a list of GoldenQuestion items and return summary statistics."""
    per_query_results = []
    category_metrics: dict[str, list[dict[str, float]]] = {}
    all_metrics: list[dict[str, float]] = []
    latencies: list[float] = []

    for item in questions:
        if not item.gold_articles:
            # Skip out_of_scope or non-retrieval questions in retrieval metric calculation
            continue

        t0 = time.time()
        retrieved = retriever.retrieve(
            query=item.question,
            scope="corpus",
            max_parents=max_parents,
        )
        elapsed_ms = (time.time() - t0) * 1000.0
        latencies.append(elapsed_ms)

        q_metrics = compute_retrieval_metrics_for_query(
            retrieved=retrieved,
            gold_articles=item.gold_articles,
            k_list=(5, 10),
        )

        all_metrics.append(q_metrics)
        category_metrics.setdefault(item.category, []).append(q_metrics)

        per_query_results.append({
            "id": item.id,
            "category": item.category,
            "question": item.question,
            "metrics": q_metrics,
            "latency_ms": elapsed_ms,
            "num_retrieved": len(retrieved),
        })

    summary = aggregate_retrieval_metrics(all_metrics)

    by_category = {
        cat: aggregate_retrieval_metrics(mets)
        for cat, mets in category_metrics.items()
    }

    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    return {
        "num_evaluated": len(all_metrics),
        "overall": summary,
        "by_category": by_category,
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
        },
        "per_query": per_query_results,
    }
