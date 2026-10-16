"""Scheduled cleanup of expired uploaded documents and parents."""

import logging
from typing import Any

from qdrant_client import QdrantClient
from sqlalchemy import Engine

from legalrag.core.config import ConfigBundle, load_config
from legalrag.data.db import get_engine
from legalrag.retrieval.embedder import FakeEmbedder
from legalrag.retrieval.store import get_qdrant_client
from legalrag.sessions.manager import SessionManager

logger = logging.getLogger(__name__)


def cleanup_expired_uploads(
    db_engine: Engine | None = None,
    qdrant_client: QdrantClient | None = None,
    config: ConfigBundle | None = None,
) -> dict[str, Any]:
    """Purge all expired upload records, parents, and Qdrant points."""
    cfg = config or load_config()
    engine = db_engine or get_engine(cfg.app.db.url)
    client = qdrant_client or get_qdrant_client(url=cfg.app.qdrant.url, path=cfg.app.qdrant.path)

    # Fake embedder suffices for deletion
    dummy_embedder = FakeEmbedder(dim=64)
    manager = SessionManager(
        db_engine=engine,
        qdrant_client=client,
        embedder=dummy_embedder,
        config=cfg,
    )

    stats = manager.cleanup_expired()
    logger.info("Cleanup completed: deleted %d uploads, %d parents", stats["deleted_uploads"], stats["deleted_parents"])
    return stats
