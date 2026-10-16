"""Unit tests for evaluation metrics, dataset loading, and runner."""

from pathlib import Path

from legalrag.evaluation.datasets import (
    GoldenArticle,
    GoldenQuestion,
    load_golden_set,
    save_golden_set,
)
from legalrag.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics_for_query,
)
from legalrag.retrieval.retriever import RetrievedContext


def test_retrieval_metrics_exact_hit():
    gold = [GoldenArticle(doc_id="doc1", article_no="5")]
    retrieved = [
        RetrievedContext(
            parent_id="p1",
            doc_id="doc1",
            title="Doc 1",
            breadcrumb="Doc 1 > Điều 5",
            article_no="5",
            text="sample text",
        ),
        RetrievedContext(
            parent_id="p2",
            doc_id="doc2",
            title="Doc 2",
            breadcrumb="Doc 2 > Điều 10",
            article_no="10",
            text="sample text 2",
        ),
    ]

    metrics = compute_retrieval_metrics_for_query(retrieved, gold)
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] == 1.0


def test_retrieval_metrics_rank_2_hit():
    gold = [GoldenArticle(doc_id="doc1", article_no="5")]
    retrieved = [
        RetrievedContext(
            parent_id="p2",
            doc_id="doc2",
            title="Doc 2",
            breadcrumb="Doc 2 > Điều 1",
            article_no="1",
            text="sample text 2",
        ),
        RetrievedContext(
            parent_id="p1",
            doc_id="doc1",
            title="Doc 1",
            breadcrumb="Doc 1 > Điều 5",
            article_no="5",
            text="sample text",
        ),
    ]

    metrics = compute_retrieval_metrics_for_query(retrieved, gold)
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr@10"] == 0.5  # Rank 2 -> 1/2
    assert metrics["ndcg@10"] < 1.0


def test_retrieval_metrics_no_hit():
    gold = [GoldenArticle(doc_id="doc1", article_no="5")]
    retrieved = [
        RetrievedContext(
            parent_id="p3",
            doc_id="doc3",
            title="Doc 3",
            breadcrumb="Doc 3 > Điều 1",
            article_no="1",
            text="sample text 3",
        )
    ]

    metrics = compute_retrieval_metrics_for_query(retrieved, gold)
    assert metrics["recall@5"] == 0.0
    assert metrics["mrr@10"] == 0.0
    assert metrics["ndcg@10"] == 0.0


def test_aggregate_metrics():
    m1 = {"recall@5": 1.0, "mrr@10": 1.0}
    m2 = {"recall@5": 0.0, "mrr@10": 0.0}
    agg = aggregate_retrieval_metrics([m1, m2])
    assert agg["recall@5"] == 0.5
    assert agg["mrr@10"] == 0.5


def test_golden_set_io(tmp_path: Path):
    file_path = tmp_path / "test_golden.jsonl"
    sample = [
        GoldenQuestion(
            id="q1",
            category="single",
            question="Thời giờ làm việc?",
            gold_articles=[GoldenArticle(doc_id="doc1", article_no="5")],
            split="dev",
            review_status="accepted",
        )
    ]
    save_golden_set(sample, file_path)
    loaded = load_golden_set(file_path)
    assert len(loaded) == 1
    assert loaded[0].id == "q1"
    assert loaded[0].category == "single"
    assert loaded[0].gold_articles[0].doc_id == "doc1"
    assert loaded[0].gold_articles[0].article_no == "5"
