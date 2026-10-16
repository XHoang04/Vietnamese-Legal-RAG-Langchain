"""Unit tests for hybrid retriever, RRF fusion, parent expansion, and session isolation."""

import json
from pathlib import Path

import pandas as pd
import pytest
from qdrant_client import QdrantClient

from legalrag.core.errors import IndexModelMismatchError
from legalrag.data.models import Status
from legalrag.ingestion.indexer import index_chunks
from legalrag.retrieval.embedder import FakeEmbedder
from legalrag.retrieval.hybrid import reciprocal_rank_fusion
from legalrag.retrieval.reranker import FakeReranker
from legalrag.retrieval.retriever import Retriever
from legalrag.retrieval.sparse import SyllableBM25


def test_reciprocal_rank_fusion_hand_calculated():
    # List 1: [A, B, C]
    # List 2: [B, A, D]
    # rrf_k = 60
    # score(A) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # score(B) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 (tie with A)
    # score(C) = 1/(60+3) = 1/63
    # score(D) = 1/(60+3) = 1/63
    res = reciprocal_rank_fusion([["A", "B", "C"], ["B", "A", "D"]], rrf_k=60)
    scores = dict(res)
    assert pytest.approx(scores["A"], rel=1e-5) == (1 / 61 + 1 / 62)
    assert pytest.approx(scores["B"], rel=1e-5) == (1 / 62 + 1 / 61)
    assert pytest.approx(scores["C"], rel=1e-5) == (1 / 63)
    assert pytest.approx(scores["D"], rel=1e-5) == (1 / 63)
    assert scores["A"] > scores["C"]


@pytest.fixture
def mock_retrieval_setup(tmp_path: Path):
    client = QdrantClient(location=":memory:")
    embedder = FakeEmbedder(dim=32, name="test-embedder")
    bm25 = SyllableBM25(mode="syllable_bigram", avgdl=50.0)

    # 6 sample chunks with 3 parents
    chunks_data = [
        {
            "chunk_id": "doc1:A1:all",
            "parent_id": "doc1:A1",
            "doc_id": "doc1",
            "level": "article",
            "article_no": "1",
            "clause_no": None,
            "breadcrumb": "Luật Lao động > Điều 1",
            "text": "Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày.",
            "embed_text": "Luật Lao động > Điều 1\nThời giờ làm việc bình thường không quá 08 giờ trong 01 ngày.",
            "doc_type": "Luật",
            "doc_number": "45/2019/QH14",
            "title": "Bộ luật Lao động 2019",
            "status": "IN_FORCE",
            "effective_date": "2021-01-01",
            "url": "http://example.com/doc1",
            "source": "corpus",
            "session_id": None,
            "content_hash": "hash1",
        },
        {
            "chunk_id": "doc2:A5:all",
            "parent_id": "doc2:A5",
            "doc_id": "doc2",
            "level": "article",
            "article_no": "5",
            "clause_no": None,
            "breadcrumb": "Nghị định 145 > Điều 5",
            "text": "Tiền lương làm thêm giờ vào ban đêm được tính theo công thức quy định.",
            "embed_text": "Nghị định 145 > Điều 5\nTiền lương làm thêm giờ vào ban đêm được tính theo công thức quy định.",
            "doc_type": "Nghị định",
            "doc_number": "145/2020/NĐ-CP",
            "title": "Nghị định 145/2020/NĐ-CP",
            "status": "EXPIRED",
            "effective_date": "2021-02-01",
            "url": "http://example.com/doc2",
            "source": "corpus",
            "session_id": None,
            "content_hash": "hash2",
        },
        {
            "chunk_id": "upload:up1:W0",
            "parent_id": "upload:up1:P0",
            "doc_id": "upload:up1",
            "level": "flat",
            "article_no": None,
            "clause_no": None,
            "breadcrumb": "Hợp đồng lao động mẫu",
            "text": "Thời giờ làm việc theo thỏa thuận trong hợp đồng của nhân viên A.",
            "embed_text": "Hợp đồng lao động mẫu\nThời giờ làm việc theo thỏa thuận trong hợp đồng của nhân viên A.",
            "doc_type": None,
            "doc_number": None,
            "title": "Hợp đồng cá nhân",
            "status": "UNKNOWN",
            "effective_date": None,
            "url": None,
            "source": "upload",
            "session_id": "session_user_1",
            "content_hash": "hash3",
        },
    ]

    df = pd.DataFrame(chunks_data)
    embeddings_map = {
        row["chunk_id"]: embedder.encode_query(row["embed_text"]).tolist()
        for row in chunks_data
    }

    index_chunks(
        client=client,
        chunks_df=df,
        embeddings_map=embeddings_map,
        embed_model_name="test-embedder",
        dense_dim=32,
        collection_name="test_collection",
        bm25_k1=1.2,
        bm25_b=0.75,
    )

    return client, embedder, bm25, tmp_path


def test_retriever_corpus_and_rerank(mock_retrieval_setup):
    client, embedder, bm25, _ = mock_retrieval_setup
    reranker = FakeReranker()

    retriever = Retriever(
        client=client,
        embedder=embedder,
        bm25=bm25,
        reranker=reranker,
        collection_name="test_collection",
        dense_top_k=5,
        sparse_top_k=5,
        fused_top_k=5,
        rerank_top_k=5,
        max_parents=5,
        hybrid_enabled=True,
    )

    results = retriever.retrieve("thời giờ làm việc", scope="corpus")
    assert len(results) > 0
    # Must retrieve corpus docs, not uploads
    assert all(r.source == "corpus" for r in results)
    # doc1 has "thời giờ làm việc"
    doc_ids = [r.doc_id for r in results]
    assert "doc1" in doc_ids


def test_retriever_session_isolation(mock_retrieval_setup):
    client, embedder, bm25, _ = mock_retrieval_setup

    retriever = Retriever(
        client=client,
        embedder=embedder,
        bm25=bm25,
        reranker=FakeReranker(),
        collection_name="test_collection",
    )

    # Search with session_user_1 should find the upload
    results_user1 = retriever.retrieve("hợp đồng lao động", scope="upload", session_id="session_user_1")
    assert len(results_user1) == 1
    assert results_user1[0].doc_id == "upload:up1"

    # Search with another session should find NOTHING (strict isolation)
    results_user2 = retriever.retrieve("hợp đồng lao động", scope="upload", session_id="session_user_2")
    assert len(results_user2) == 0


def test_retriever_expired_policy(mock_retrieval_setup):
    client, embedder, bm25, _ = mock_retrieval_setup

    # 1. Policy exclude: doc2 is EXPIRED, so it should not appear
    retriever_exclude = Retriever(
        client=client,
        embedder=embedder,
        bm25=bm25,
        collection_name="test_collection",
        expired_policy="exclude",
    )
    res_exclude = retriever_exclude.retrieve("tiền lương làm thêm giờ", scope="corpus")
    assert not any(r.doc_id == "doc2" for r in res_exclude)

    # 2. Policy penalize: doc2 should appear but with lower score
    retriever_penalize = Retriever(
        client=client,
        embedder=embedder,
        bm25=bm25,
        reranker=FakeReranker(),
        collection_name="test_collection",
        expired_policy="penalize",
        status_penalty=10.0,
    )
    res_penalize = retriever_penalize.retrieve("tiền lương làm thêm giờ", scope="corpus")
    doc2_res = [r for r in res_penalize if r.doc_id == "doc2"]
    if doc2_res:
        assert doc2_res[0].status == Status.EXPIRED


def test_retriever_model_mismatch(mock_retrieval_setup):
    client, embedder, bm25, tmp_path = mock_retrieval_setup
    meta_path = tmp_path / "index_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"embed_model": "old-model-xyz"}, f)

    with pytest.raises(IndexModelMismatchError):
        Retriever(
            client=client,
            embedder=embedder,
            bm25=bm25,
            collection_name="test_collection",
            expected_embed_model="new-model-abc",
            index_meta_path=meta_path,
        )
