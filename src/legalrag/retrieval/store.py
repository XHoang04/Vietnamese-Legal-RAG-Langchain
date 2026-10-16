"""Qdrant vector store connection, collection management, and filtering."""

import logging
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "legal_chunks"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

PAYLOAD_INDEX_FIELDS = [
    "doc_id",
    "status",
    "source",
    "session_id",
    "doc_type",
    "level",
]


def get_qdrant_client(
    url: str | None = None,
    path: str | Path | None = None,
    location: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """Create a QdrantClient instance based on configuration."""
    if location == ":memory:":
        logger.info("Initializing in-memory QdrantClient")
        return QdrantClient(location=":memory:")

    if url:
        logger.info("Connecting to remote Qdrant at %s", url)
        return QdrantClient(url=url, api_key=api_key, timeout=60)

    if path:
        local_path = Path(path)
        local_path.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing local persistent Qdrant at %s", local_path)
        return QdrantClient(path=str(local_path))

    # Fallback to in-memory
    return QdrantClient(location=":memory:")


def init_qdrant_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    dense_dim: int = 1024,
    distance: models.Distance = models.Distance.COSINE,
    recreate: bool = False,
) -> None:
    """Initialize Qdrant collection with dense and sparse vector configurations and payload indices."""
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        logger.warning("Deleting existing Qdrant collection: %s", collection_name)
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        logger.info(
            "Creating Qdrant collection '%s' (dense_dim=%d, distance=%s)",
            collection_name, dense_dim, distance,
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_dim,
                    distance=distance,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                )
            },
        )

        # Create payload keyword indices
        for field_name in PAYLOAD_INDEX_FIELDS:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug("Payload index creation on '%s' notice: %s", field_name, e)
    else:
        logger.info("Qdrant collection '%s' already exists", collection_name)


def build_qdrant_filter(
    scope: str = "corpus",
    session_id: str | None = None,
    doc_ids: list[str] | None = None,
    exclude_expired: bool = False,
) -> models.Filter:
    """Build Qdrant Filter according to scope, session isolation, and doc_ids.

    - scope='corpus' -> source == 'corpus'
    - scope='upload' -> source == 'upload' AND session_id == <id>
    - scope='both'   -> (source == 'corpus') OR (source == 'upload' AND session_id == <id>)
    - doc_ids filter -> doc_id in doc_ids
    - exclude_expired -> status != 'EXPIRED'
    """
    conditions: list[Any] = []

    # Scope condition
    if scope == "corpus":
        conditions.append(
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value="corpus"),
            )
        )
    elif scope == "upload":
        # Strictly isolate by session_id
        effective_session = session_id or "__NO_SESSION__"
        conditions.append(
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value="upload"),
            )
        )
        conditions.append(
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=effective_session),
            )
        )
    elif scope == "both":
        effective_session = session_id or "__NO_SESSION__"
        conditions.append(
            models.Filter(
                should=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="corpus"),
                    ),
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value="upload"),
                            ),
                            models.FieldCondition(
                                key="session_id",
                                match=models.MatchValue(value=effective_session),
                            ),
                        ]
                    ),
                ]
            )
        )

    # doc_ids filter
    if doc_ids:
        conditions.append(
            models.FieldCondition(
                key="doc_id",
                match=models.MatchAny(any=doc_ids),
            )
        )

    # exclude_expired
    if exclude_expired:
        conditions.append(
            models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="status",
                        match=models.MatchValue(value="EXPIRED"),
                    )
                ]
            )
        )

    return models.Filter(must=conditions)
