"""FastAPI Application Entry Point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.dependencies import get_container
from apps.api.routers import chat, documents, health, sessions
from legalrag.core.errors import LegalRAGError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: initialize container and teardown on shutdown."""
    logger.info("Initializing Legal RAG API Server...")
    container = get_container()
    logger.info(f"API loaded successfully with DB={container.config.app.db.url}")
    yield
    logger.info("Shutting down Legal RAG API Server...")
    if hasattr(container, "qdrant") and hasattr(container.qdrant, "close"):
        try:
            container.qdrant.close()
        except Exception:
            pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from legalrag.core.config import load_config
    app_cfg = load_config().app

    app = FastAPI(
        title="Vietnamese Labor Law Legal RAG API",
        description="REST and SSE API for Vietnamese Labor Law QA with LangGraph, hybrid retrieval, and validity management.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    cors_origins = getattr(app_cfg.api, "cors_origins", ["*"]) or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            code = detail.get("code", "HTTP_ERROR")
            message = detail.get("message", str(detail))
        else:
            code = "HTTP_ERROR"
            message = str(detail)

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": message}},
        )

    @app.exception_handler(LegalRAGError)
    async def legalrag_exception_handler(request: Request, exc: LegalRAGError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if exc.code == "NOT_A_PDF":
            status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        elif exc.code == "FILE_TOO_LARGE":
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        elif exc.code in ["TOO_MANY_PAGES", "TOO_MANY_CHUNKS", "UNSUPPORTED_SCAN_PDF"]:
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif exc.code in ["UPLOAD_REQUIRES_LOCAL_LLM", "LLM_UNAVAILABLE"]:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif exc.code == "UNAUTHORIZED":
            status_code = status.HTTP_401_UNAUTHORIZED
        elif exc.code == "RATE_LIMITED":
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif exc.code == "THREAD_NOT_FOUND":
            status_code = status.HTTP_404_NOT_FOUND

        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    # Root redirect to API documentation
    @app.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    # Register Routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(sessions.router)

    return app


app = create_app()
