"""Retriever interface, context models, and Qdrant-backed hybrid implementation."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sqlalchemy import Engine, select

from legalrag.core.errors import IndexModelMismatchError
from legalrag.core.text import count_tokens
from legalrag.data.db import parents_table
from legalrag.data.models import Status
from legalrag.retrieval.embedder import Embedder
from legalrag.retrieval.hybrid import reciprocal_rank_fusion
from legalrag.retrieval.reranker import BaseReranker
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import (
    DEFAULT_COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    build_qdrant_filter,
)

logger = logging.getLogger(__name__)


class RetrievedContext(BaseModel):
    """Context unit retrieved and expanded to parent level for generation."""
    parent_id: str
    doc_id: str
    title: str
    doc_number: str | None = None
    breadcrumb: str
    article_no: str | None = None
    text: str
    matched_chunk_ids: list[str] = Field(default_factory=list)
    best_score: float = 0.0
    source: Literal["corpus", "upload"] = "corpus"
    status: Status | None = None
    url: str | None = None
    from_subquestion: str | None = None
    is_replacement: bool = False
    session_id: str | None = None


@runtime_checkable
class BaseRetriever(Protocol):
    """Protocol for retriever components."""

    def retrieve(
        self,
        query: str,
        *,
        scope: Literal["corpus", "upload", "both"] = "corpus",
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        max_parents: int | None = None,
    ) -> list[RetrievedContext]:
        ...


class FakeRetriever:
    """Mock retriever returning deterministic contexts for testing."""

    def __init__(self, contexts: list[RetrievedContext] | None = None) -> None:
        self.contexts = contexts or []

    def retrieve(
        self,
        query: str,
        *,
        scope: Literal["corpus", "upload", "both"] = "corpus",
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        max_parents: int | None = None,
    ) -> list[RetrievedContext]:
        filtered = []
        for c in self.contexts:
            if scope == "corpus" and c.source != "corpus":
                continue
            if scope == "upload" and (
                c.source != "upload"
                or (c.doc_id.startswith("upload:") and session_id and getattr(c, "session_id", None) != session_id)
            ):
                continue
            if doc_ids and c.doc_id not in doc_ids:
                continue
            filtered.append(c)

        if max_parents is not None:
            filtered = filtered[:max_parents]
        return filtered


class Retriever:
    """Production Qdrant-backed hybrid retriever supporting dense+sparse RRF, reranking, and parent expansion."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        bm25: SyllableBM25 | None = None,
        reranker: BaseReranker | None = None,
        db_engine: Engine | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        dense_top_k: int = 40,
        sparse_top_k: int = 40,
        rrf_k: int = 60,
        fused_top_k: int = 30,
        hybrid_enabled: bool = True,
        rerank_enabled: bool = True,
        rerank_top_k: int = 8,
        parent_child_enabled: bool = True,
        max_parents: int = 5,
        expired_policy: Literal["annotate", "penalize", "exclude"] = "annotate",
        status_penalty: float = 2.0,
        context_token_budget: int = 4000,
        expected_embed_model: str | None = None,
        index_meta_path: Path | str | None = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.db_engine = db_engine
        self.collection_name = collection_name
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.fused_top_k = fused_top_k
        self.hybrid_enabled = hybrid_enabled
        self.rerank_enabled = rerank_enabled
        self.rerank_top_k = rerank_top_k
        self.parent_child_enabled = parent_child_enabled
        self.max_parents = max_parents
        self.expired_policy = expired_policy
        self.status_penalty = status_penalty
        self.context_token_budget = context_token_budget

        # Verify index model match if metadata exists
        if expected_embed_model and index_meta_path:
            meta_file = Path(index_meta_path)
            if meta_file.exists():
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        meta = json.load(f)
                    indexed_model = meta.get("embed_model")
                    if indexed_model and indexed_model != expected_embed_model:
                        raise IndexModelMismatchError(
                            f"Configured embedding model '{expected_embed_model}' does not match indexed model '{indexed_model}'"
                        )
                except IndexModelMismatchError:
                    raise
                except Exception as e:
                    logger.warning("Could not verify index metadata: %s", e)

    def retrieve(
        self,
        query: str,
        *,
        scope: Literal["corpus", "upload", "both"] = "corpus",
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        max_parents: int | None = None,
    ) -> list[RetrievedContext]:
        if not query.strip():
            return []

        q_filter = build_qdrant_filter(
            scope=scope,
            session_id=session_id,
            doc_ids=doc_ids,
            exclude_expired=(self.expired_policy == "exclude"),
        )

        payload_map: dict[str, dict[str, Any]] = {}
        dense_chunk_ids: list[str] = []

        # 1. Dense retrieval
        try:
            dense_query_vec = self.embedder.encode_query(query).tolist()
            dense_points = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_query_vec,
                using=DENSE_VECTOR_NAME,
                query_filter=q_filter,
                limit=self.dense_top_k,
                with_payload=True,
            ).points

            for p in dense_points:
                if p.payload and "chunk_id" in p.payload:
                    cid = str(p.payload["chunk_id"])
                    dense_chunk_ids.append(cid)
                    payload_map[cid] = p.payload
        except Exception as e:
            logger.error("Dense retrieval failed: %s", e)

        # 2. Sparse retrieval
        sparse_chunk_ids: list[str] = []
        if self.hybrid_enabled and self.bm25:
            try:
                sparse_vec = self.bm25.encode_query(query)
                if sparse_vec.indices:
                    sparse_points = self.client.query_points(
                        collection_name=self.collection_name,
                        query=models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                        using=SPARSE_VECTOR_NAME,
                        query_filter=q_filter,
                        limit=self.sparse_top_k,
                        with_payload=True,
                    ).points

                    for p in sparse_points:
                        if p.payload and "chunk_id" in p.payload:
                            cid = str(p.payload["chunk_id"])
                            sparse_chunk_ids.append(cid)
                            if cid not in payload_map:
                                payload_map[cid] = p.payload
            except Exception as e:
                logger.error("Sparse retrieval failed: %s", e)

        # 3. Fusion (RRF)
        if self.hybrid_enabled and sparse_chunk_ids:
            fused = reciprocal_rank_fusion([dense_chunk_ids, sparse_chunk_ids], rrf_k=self.rrf_k)
            candidate_chunk_ids = [cid for cid, _ in fused[:self.fused_top_k]]
            candidate_scores = {cid: score for cid, score in fused[:self.fused_top_k]}
        else:
            candidate_chunk_ids = dense_chunk_ids[:self.fused_top_k]
            candidate_scores = {cid: float(len(candidate_chunk_ids) - i) for i, cid in enumerate(candidate_chunk_ids)}

        if not candidate_chunk_ids:
            return []

        candidates = [payload_map[cid] for cid in candidate_chunk_ids if cid in payload_map]

        # 4. Rerank
        scored_candidates: list[tuple[dict[str, Any], float]] = []
        if self.rerank_enabled and self.reranker and candidates:
            try:
                passages = [f"{c.get('breadcrumb', '')}\n{c.get('text', '')}" for c in candidates]
                logits = self.reranker.rank(query, passages)
                for c, s in zip(candidates, logits, strict=False):
                    score = float(s)
                    if self.expired_policy == "penalize" and c.get("status") == "EXPIRED":
                        score -= self.status_penalty
                    scored_candidates.append((c, score))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                scored_candidates = scored_candidates[:self.rerank_top_k]
            except Exception as e:
                logger.warning("Reranking failed, using fusion scores: %s", e)
                scored_candidates = [(c, candidate_scores.get(c["chunk_id"], 0.0)) for c in candidates[:self.rerank_top_k]]
        else:
            scored_candidates = [(c, candidate_scores.get(c["chunk_id"], 0.0)) for c in candidates[:self.rerank_top_k]]

        # 5. Parent Expansion
        parent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        parent_best_score: dict[str, float] = {}

        for c, score in scored_candidates:
            pid = str(c.get("parent_id") or c["chunk_id"])
            parent_groups[pid].append(c)
            if pid not in parent_best_score or score > parent_best_score[pid]:
                parent_best_score[pid] = score

        # Query parents table in DB for complete parent texts
        db_parents: dict[str, dict[str, Any]] = {}
        if self.db_engine and parent_groups:
            try:
                with self.db_engine.connect() as conn:
                    query_stmt = select(parents_table).where(parents_table.c.parent_id.in_(list(parent_groups.keys())))
                    for r in conn.execute(query_stmt):
                        db_parents[r.parent_id] = {
                            "parent_id": r.parent_id,
                            "doc_id": r.doc_id,
                            "article_no": r.article_no,
                            "article_title": r.article_title,
                            "breadcrumb": r.breadcrumb,
                            "text": r.text,
                            "token_count": r.token_count,
                            "source": r.source,
                        }
            except Exception as e:
                logger.warning("Could not query parents table: %s", e)

        # Assemble RetrievedContext objects
        contexts: list[RetrievedContext] = []
        for pid, score in sorted(parent_best_score.items(), key=lambda x: x[1], reverse=True):
            chunks = parent_groups[pid]
            sample = chunks[0]
            matched_cids = [str(chk["chunk_id"]) for chk in chunks]

            raw_status = sample.get("status")
            status_enum = None
            if raw_status:
                try:
                    status_enum = Status(raw_status)
                except ValueError:
                    status_enum = Status.UNKNOWN

            if pid in db_parents:
                parent_row = db_parents[pid]
                parent_text = parent_row["text"]
                breadcrumb = parent_row["breadcrumb"]
                article_no = parent_row["article_no"]
            else:
                parent_text = sample.get("text", "")
                breadcrumb = sample.get("breadcrumb", "")
                article_no = sample.get("article_no")

            doc_title = str(sample.get("title", ""))
            if not doc_title and breadcrumb:
                doc_title = breadcrumb.split(" > ")[0]

            ctx = RetrievedContext(
                parent_id=pid,
                doc_id=str(sample.get("doc_id", "")),
                title=doc_title,
                doc_number=sample.get("doc_number"),
                breadcrumb=breadcrumb,
                article_no=article_no,
                text=parent_text,
                matched_chunk_ids=matched_cids,
                best_score=float(score),
                source=sample.get("source", "corpus"),
                status=status_enum,
                url=sample.get("url"),
                session_id=sample.get("session_id"),
            )
            contexts.append(ctx)

        effective_max = max_parents if max_parents is not None else self.max_parents
        contexts = contexts[:effective_max]

        # 6. Context token budget pruning
        total_tokens = sum(count_tokens(c.text) for c in contexts)
        while total_tokens > self.context_token_budget and len(contexts) > 1:
            popped = contexts.pop()  # Drop lowest-scoring parent
            total_tokens -= count_tokens(popped.text)
            logger.info("Context token budget exceeded; pruned parent %s", popped.parent_id)

        return contexts
