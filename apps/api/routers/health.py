"""Health and readiness check endpoints."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text

from apps.api.dependencies import get_config, get_db_engine, get_qdrant_client
from legalrag.core.config import ConfigBundle

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness Probe")
async def health_check() -> dict[str, str]:
    """Always returns 200 OK to indicate service is alive."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness Probe")
async def readiness_check(
    engine=Depends(get_db_engine),
    qdrant=Depends(get_qdrant_client),
    config: ConfigBundle = Depends(get_config),
) -> dict[str, str]:
    """
    Checks database connection, Qdrant cluster status, and index metadata consistency.
    Returns 200 if ready, 503 if any subsystem fails.
    """
    # 1. Database check
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Readiness check failed on Database: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "DATABASE_UNAVAILABLE", "message": f"Database check failed: {e}"},
        )

    # 2. Qdrant check
    try:
        qdrant.get_collections()
    except Exception as e:
        logger.error(f"Readiness check failed on Qdrant: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "QDRANT_UNAVAILABLE", "message": f"Qdrant connection failed: {e}"},
        )

    # 3. Model index consistency check
    imeta_file = Path("data/processed/index_meta.json")
    if imeta_file.exists():
        try:
            with open(imeta_file, encoding="utf-8") as f:
                imeta = json.load(f)
            indexed_col = imeta.get("collection")
            if indexed_col and indexed_col != config.app.qdrant.collection:
                logger.warning(
                    f"Index meta collection {indexed_col} differs from app.yaml {config.app.qdrant.collection}"
                )
        except Exception as e:
            logger.warning(f"Could not verify index_meta.json: {e}")

    return {
        "status": "ready",
        "database": "ok",
        "qdrant": "ok",
    }
