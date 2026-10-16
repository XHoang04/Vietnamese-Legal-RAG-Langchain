"""Indexer for batch upserting dense and sparse chunks into Qdrant idempotently."""

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models

from legalrag.core.ids import make_uuid5
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import (
    DEFAULT_COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    init_qdrant_collection,
)

logger = logging.getLogger(__name__)


def build_qdrant_points(
    chunks_df: pd.DataFrame,
    embeddings_map: dict[str, list[float]],
    bm25: SyllableBM25,
    embed_model_name: str,
    snapshot_as_of: str = "2024-09-10",
) -> list[models.PointStruct]:
    """Convert dataframe rows into Qdrant PointStruct objects with dense + sparse vectors and full payload."""
    points: list[models.PointStruct] = []

    for _, row in chunks_df.iterrows():
        chunk_id = str(row["chunk_id"])
        if chunk_id not in embeddings_map:
            continue

        dense_vec = embeddings_map[chunk_id]
        embed_text = str(row.get("embed_text", ""))
        sparse_vec = bm25.encode_document(embed_text)

        point_id = make_uuid5(chunk_id)

        # Build payload matching Section 9.3 specification
        payload = {
            "chunk_id": chunk_id,
            "parent_id": str(row.get("parent_id", "")),
            "doc_id": str(row.get("doc_id", "")),
            "level": str(row.get("level", "")),
            "article_no": str(row["article_no"]) if pd.notna(row.get("article_no")) else None,
            "clause_no": str(row["clause_no"]) if pd.notna(row.get("clause_no")) else None,
            "breadcrumb": str(row.get("breadcrumb", "")),
            "text": str(row.get("text", "")),
            "doc_type": str(row["doc_type"]) if pd.notna(row.get("doc_type")) else None,
            "doc_number": str(row["doc_number"]) if pd.notna(row.get("doc_number")) else None,
            "title": str(row.get("title", "")),
            "status": str(row.get("status", "UNKNOWN")),
            "effective_date": str(row["effective_date"]) if pd.notna(row.get("effective_date")) else None,
            "url": str(row["url"]) if pd.notna(row.get("url")) else None,
            "source": str(row.get("source", "corpus")),
            "session_id": str(row["session_id"]) if pd.notna(row.get("session_id")) else None,
            "content_hash": str(row.get("content_hash", "")),
            "embed_model": embed_model_name,
            "snapshot_as_of": snapshot_as_of,
        }

        point = models.PointStruct(
            id=point_id,
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                ),
            },
            payload=payload,
        )
        points.append(point)

    return points


def index_chunks(
    client: QdrantClient,
    chunks_df: pd.DataFrame,
    embeddings_map: dict[str, list[float]],
    embed_model_name: str,
    dense_dim: int,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    state_file: Path | str | None = None,
    meta_file: Path | str | None = None,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
    snapshot_as_of: str = "2024-09-10",
    batch_size: int = 256,
    recreate: bool = False,
) -> dict[str, Any]:
    """Idempotently index chunks into Qdrant collection with state tracking."""
    state_path = Path(state_file) if state_file else None
    meta_path = Path(meta_file) if meta_file else None

    # Load existing state if present
    previous_state: dict[str, str] = {}
    if state_path and state_path.exists() and not recreate:
        try:
            state_df = pd.read_parquet(state_path)
            for _, r in state_df.iterrows():
                previous_state[str(r["chunk_id"])] = f"{r['content_hash']}_{r['embed_model']}"
        except Exception as e:
            logger.warning("Could not read previous index state: %s", e)

    init_qdrant_collection(
        client=client,
        collection_name=collection_name,
        dense_dim=dense_dim,
        distance=models.Distance.COSINE,
        recreate=recreate,
    )

    # Calculate average doc length for BM25
    corpus_embed_texts = [str(t) for t in chunks_df["embed_text"].tolist()]
    avgdl = float(np.mean([len(t.split()) for t in corpus_embed_texts])) if corpus_embed_texts else 100.0
    bm25 = SyllableBM25(mode="syllable_bigram", k1=bm25_k1, b=bm25_b, avgdl=avgdl)

    # Determine chunks that need upserting
    current_chunk_keys: dict[str, str] = {}
    chunks_to_upsert: list[str] = []

    for _, row in chunks_df.iterrows():
        cid = str(row["chunk_id"])
        chash = str(row["content_hash"])
        state_key = f"{chash}_{embed_model_name}"
        current_chunk_keys[cid] = state_key

        if cid not in previous_state or previous_state[cid] != state_key:
            chunks_to_upsert.append(cid)

    # Determine obsolete points to delete
    chunks_to_delete = [
        cid for cid in previous_state
        if cid not in current_chunk_keys
    ]

    # Delete obsolete chunks
    if chunks_to_delete:
        logger.info("Deleting %d obsolete corpus chunks from Qdrant", len(chunks_to_delete))
        point_ids_to_del = [make_uuid5(cid) for cid in chunks_to_delete]
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids_to_del),
        )

    # Upsert new / modified chunks
    upserted_count = 0
    if chunks_to_upsert:
        logger.info("Preparing to upsert %d chunks (out of %d total)...", len(chunks_to_upsert), len(chunks_df))
        subset_df = chunks_df[chunks_df["chunk_id"].isin(set(chunks_to_upsert))]
        all_points = build_qdrant_points(
            chunks_df=subset_df,
            embeddings_map=embeddings_map,
            bm25=bm25,
            embed_model_name=embed_model_name,
            snapshot_as_of=snapshot_as_of,
        )

        for i in range(0, len(all_points), batch_size):
            batch = all_points[i : i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )
            upserted_count += len(batch)
            logger.info("Upserted batch %d/%d points", upserted_count, len(all_points))
    else:
        logger.info("Index is fully up-to-date. 0 chunks needed upserting.")

    # Save state
    new_state_rows = [
        {"chunk_id": cid, "content_hash": state_key.split("_")[0], "embed_model": embed_model_name}
        for cid, state_key in current_chunk_keys.items()
    ]
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(new_state_rows).to_parquet(state_path, index=False)

    # Save metadata
    meta = {
        "embed_model": embed_model_name,
        "dim": dense_dim,
        "distance": "Cosine",
        "normalized": True,
        "num_chunks": len(chunks_df),
        "upserted_count": upserted_count,
        "deleted_count": len(chunks_to_delete),
        "avgdl": round(avgdl, 2),
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "snapshot_as_of": snapshot_as_of,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if meta_path:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta
