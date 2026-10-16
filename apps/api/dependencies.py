"""FastAPI Dependency Injection providers."""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient
from sqlalchemy import Engine, create_engine

from apps.api.auth import verify_api_key
from apps.api.rate_limit import get_rate_limiter
from legalrag.core.config import ConfigBundle, load_config
from legalrag.graph.build import build_graph
from legalrag.graph.checkpointer import build_checkpointer
from legalrag.graph.deps import GraphDeps
from legalrag.llm.factory import LLMFactory
from legalrag.retrieval.embedder import Embedder, FakeEmbedder
from legalrag.retrieval.reranker import VietnameseCrossEncoderReranker
from legalrag.retrieval.retriever import Retriever
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import get_qdrant_client as create_qdrant_client
from legalrag.sessions.manager import SessionManager
from legalrag.validity.checker import ValidityChecker
from legalrag.validity.repository import ValidityRepository

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Singleton container managing all system components and dependencies."""

    def __init__(
        self,
        config_bundle: ConfigBundle | None = None,
        qdrant_client: QdrantClient | None = None,
        db_engine: Engine | None = None,
    ) -> None:
        self.config: ConfigBundle = config_bundle or load_config()
        self.engine: Engine = db_engine or create_engine(self.config.app.db.url, future=True)

        # Initialize Qdrant Client
        app_cfg = self.config.app
        if qdrant_client is not None:
            self.qdrant = qdrant_client
        else:
            try:
                self.qdrant = create_qdrant_client(url=app_cfg.qdrant.url, path=app_cfg.qdrant.path)
            except Exception as e:
                logger.warning(f"Could not open local persistent Qdrant ({e}). Falling back to in-memory client.")
                self.qdrant = QdrantClient(location=":memory:")

        # Check index_meta for embedder
        imeta_file = Path("data/processed/index_meta.json")
        embed_name = "fake-embedder"
        dense_dim = 64
        if imeta_file.exists():
            try:
                with open(imeta_file, encoding="utf-8") as f:
                    imeta = json.load(f)
                    embed_name = imeta.get("embed_model", "fake-embedder")
                    dense_dim = imeta.get("dim", 64)
            except Exception as e:
                logger.warning(f"Failed to read index_meta.json: {e}")

        if embed_name == "fake-embedder":
            self.embedder: Any = FakeEmbedder(dim=dense_dim, name="fake-embedder")
            self.reranker: Any = None
        else:
            try:
                self.embedder = Embedder(model_name=embed_name)
            except Exception:
                self.embedder = FakeEmbedder(dim=dense_dim, name=embed_name)

            retrieval_cfg = self.config.retrieval
            if retrieval_cfg.rerank.enabled:
                rerank_cfg = retrieval_cfg.rerank
                try:
                    self.reranker = VietnameseCrossEncoderReranker(
                        model_name=getattr(rerank_cfg, "model", "AITeamVN/Vietnamese_Reranker"),
                        max_length=getattr(rerank_cfg, "max_length", 1024),
                        device="cpu",
                    )
                except Exception:
                    self.reranker = None
            else:
                self.reranker = None

        retrieval_cfg = self.config.retrieval
        self.bm25 = SyllableBM25(mode="syllable_bigram") if retrieval_cfg.hybrid.enabled else None

        self.retriever = Retriever(
            client=self.qdrant,
            embedder=self.embedder,
            bm25=self.bm25,
            reranker=self.reranker,
            db_engine=self.engine,
            collection_name=app_cfg.qdrant.collection,
            dense_top_k=retrieval_cfg.dense_top_k,
            sparse_top_k=retrieval_cfg.sparse_top_k,
            rrf_k=retrieval_cfg.rrf_k,
            fused_top_k=retrieval_cfg.fused_top_k,
            hybrid_enabled=retrieval_cfg.hybrid.enabled,
            rerank_enabled=retrieval_cfg.rerank.enabled,
            rerank_top_k=retrieval_cfg.rerank.top_k,
            parent_child_enabled=retrieval_cfg.parent_child.enabled,
            max_parents=5,
            expired_policy=retrieval_cfg.expired_policy,
            status_penalty=retrieval_cfg.status_penalty,
            context_token_budget=retrieval_cfg.context_token_budget,
        )

        self.validity_repo = ValidityRepository(self.engine)
        self.validity_checker = ValidityChecker(self.validity_repo, snapshot_as_of=app_cfg.snapshot_as_of)
        self.session_manager = SessionManager(
            db_engine=self.engine,
            qdrant_client=self.qdrant,
            embedder=self.embedder,
            config=self.config,
            bm25=self.bm25,
        )
        self.llm_factory = LLMFactory(self.config)

        self.deps = GraphDeps(
            config=self.config,
            retriever=self.retriever,
            validity_checker=self.validity_checker,
            llm_factory=self.llm_factory,
        )

        checkpointer = build_checkpointer(app_cfg)
        self.graph = build_graph(self.deps, checkpointer=checkpointer)
        self.rate_limiter = get_rate_limiter(rpm=app_cfg.api.rate_limit_per_minute)


_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def set_container(container: ServiceContainer | None) -> None:
    global _container
    _container = container


def get_config(container: ServiceContainer = Depends(get_container)) -> ConfigBundle:
    return container.config


def get_db_engine(container: ServiceContainer = Depends(get_container)) -> Engine:
    return container.engine


def get_qdrant_client(container: ServiceContainer = Depends(get_container)) -> QdrantClient:
    return container.qdrant


def get_retriever(container: ServiceContainer = Depends(get_container)) -> Retriever:
    return container.retriever


def get_session_manager(container: ServiceContainer = Depends(get_container)) -> SessionManager:
    return container.session_manager


def get_validity_checker(container: ServiceContainer = Depends(get_container)) -> ValidityChecker:
    return container.validity_checker


def get_graph(container: ServiceContainer = Depends(get_container)) -> CompiledStateGraph:
    return container.graph


def rate_limit_dependency(
    api_key: str = Depends(verify_api_key),
    container: ServiceContainer = Depends(get_container),
) -> None:
    """Enforces rate limit per API key."""
    container.rate_limiter.check_rate_limit(api_key)
