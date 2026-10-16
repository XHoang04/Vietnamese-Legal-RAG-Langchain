"""Session and uploaded document manager."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sqlalchemy import Engine, delete, select, update
from sqlalchemy.orm import Session

from legalrag.core.config import ConfigBundle
from legalrag.core.errors import TooManyChunksError
from legalrag.data.db import parents_table, uploads_table
from legalrag.ingestion.chunk_models import UploadDocument
from legalrag.ingestion.legal_chunker import chunk_document
from legalrag.ingestion.pdf_loader import load_pdf
from legalrag.retrieval.embedder import Embedder
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages document uploads, session isolation, and lifecycle."""

    def __init__(
        self,
        db_engine: Engine,
        qdrant_client: QdrantClient,
        embedder: Embedder,
        config: ConfigBundle,
        bm25: SyllableBM25 | None = None,
    ) -> None:
        self.engine = db_engine
        self.client = qdrant_client
        self.embedder = embedder
        self.config = config
        self.bm25 = bm25 or SyllableBM25(mode="syllable_bigram")
        self.collection_name = config.app.qdrant.collection

    def create_upload(self, session_id: str, filename: str) -> str:
        """Create a new pending upload record and return upload_id."""
        upload_id = str(uuid.uuid4())
        ttl_hours = getattr(self.config.app.upload, "ttl_hours", 24)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours)

        with Session(self.engine) as session:
            stmt = uploads_table.insert().values(
                upload_id=upload_id,
                session_id=session_id,
                filename=filename,
                pages=None,
                chunks=None,
                status="processing",
                created_at=now,
                expires_at=expires_at,
            )
            session.execute(stmt)
            session.commit()

        return upload_id

    def process_upload(
        self,
        upload_id: str,
        file_input: bytes | str | Path,
        filename: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Validate, parse, chunk, embed, and index an uploaded document.
        Updates upload status to 'ready' or 'failed'.
        """
        ttl_hours = getattr(self.config.app.upload, "ttl_hours", 24)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours)
        doc_id = f"upload:{upload_id}"

        try:
            # 1. Load and validate PDF
            pdf_doc = load_pdf(file_input, filename=filename, upload_config=self.config.app.upload)

            # 2. Chunk document
            source = UploadDocument(
                doc_id=doc_id,
                title=pdf_doc.title,
                full_text=pdf_doc.full_text,
                source="upload",
                session_id=session_id,
            )
            parents, chunks = chunk_document(source, self.config.app.chunking)

            # Check chunk limit
            max_chunks = getattr(self.config.app.upload, "max_chunks", 800)
            if len(chunks) > max_chunks:
                raise TooManyChunksError(
                    f"Tài liệu tạo ra {len(chunks)} đoạn, vượt quá giới hạn cho phép ({max_chunks} đoạn)."
                )

            # 3. Save parents to SQLite
            with Session(self.engine) as session:
                for p in parents:
                    stmt = parents_table.insert().values(
                        parent_id=p.parent_id,
                        doc_id=doc_id,
                        article_no=p.article_no,
                        article_title=p.article_title,
                        breadcrumb=p.breadcrumb,
                        text=p.text,
                        token_count=p.token_count,
                        source="upload",
                        session_id=session_id,
                        expires_at=expires_at,
                    )
                    session.execute(stmt)
                session.commit()

            # 4. Embed and upsert to Qdrant
            points: list[models.PointStruct] = []
            if chunks:
                dense_vectors = self.embedder.encode_documents([c.embed_text for c in chunks])
                for idx, c in enumerate(chunks):
                    vec_dense = dense_vectors[idx].tolist()
                    vectors_dict: dict[str, Any] = {DENSE_VECTOR_NAME: vec_dense}

                    if self.bm25:
                        s_vec = self.bm25.encode_document(c.embed_text)
                        if s_vec.indices:
                            vectors_dict[SPARSE_VECTOR_NAME] = models.SparseVector(
                                indices=s_vec.indices, values=s_vec.values
                            )

                    point_id = str(uuid.uuid4())
                    payload = {
                        "chunk_id": c.chunk_id,
                        "parent_id": c.parent_id,
                        "doc_id": doc_id,
                        "upload_id": upload_id,
                        "session_id": session_id,
                        "title": pdf_doc.title,
                        "breadcrumb": c.breadcrumb,
                        "text": c.text,
                        "source": "upload",
                        "status": None,
                        "token_count": c.token_count,
                    }

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vectors_dict,
                            payload=payload,
                        )
                    )

                self.client.upsert(collection_name=self.collection_name, points=points)

            # 5. Mark as ready
            with Session(self.engine) as session:
                stmt = (
                    update(uploads_table)
                    .where(uploads_table.c.upload_id == upload_id)
                    .values(
                        pages=pdf_doc.total_pages,
                        chunks=len(chunks),
                        status="ready",
                    )
                )
                session.execute(stmt)
                session.commit()

            return {
                "upload_id": upload_id,
                "session_id": session_id,
                "title": pdf_doc.title,
                "pages": pdf_doc.total_pages,
                "chunks": len(chunks),
                "status": "ready",
            }

        except Exception as e:
            logger.error("Failed to process upload %s: %s", upload_id, e)
            with Session(self.engine) as session:
                stmt = (
                    update(uploads_table)
                    .where(uploads_table.c.upload_id == upload_id)
                    .values(status="failed")
                )
                session.execute(stmt)
                session.commit()
            raise

    def get_upload_status(self, upload_id: str) -> dict[str, Any] | None:
        """Get status and metadata for an upload."""
        with Session(self.engine) as session:
            stmt = select(uploads_table).where(uploads_table.c.upload_id == upload_id)
            row = session.execute(stmt).first()
            if not row:
                return None
            return {
                "upload_id": row.upload_id,
                "session_id": row.session_id,
                "filename": row.filename,
                "pages": row.pages,
                "chunks": row.chunks,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            }

    def list_session_documents(self, session_id: str) -> list[dict[str, Any]]:
        """List all uploaded documents associated with a session."""
        with Session(self.engine) as session:
            stmt = (
                select(uploads_table)
                .where(uploads_table.c.session_id == session_id)
                .order_by(uploads_table.c.created_at.desc())
            )
            rows = session.execute(stmt).all()
            return [
                {
                    "upload_id": r.upload_id,
                    "session_id": r.session_id,
                    "filename": r.filename,
                    "pages": r.pages,
                    "chunks": r.chunks,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                }
                for r in rows
            ]

    def delete_upload(self, upload_id: str) -> bool:
        """
        Delete an uploaded document completely:
        1. Points from Qdrant collection
        2. Parents from SQLite parents table
        3. Record from SQLite uploads table
        """
        doc_id = f"upload:{upload_id}"

        # 1. Delete points in Qdrant
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="upload_id",
                            match=models.MatchValue(value=upload_id),
                        )
                    ]
                ),
            )
        except Exception as e:
            logger.warning("Could not delete Qdrant points for upload %s: %s", upload_id, e)

        # 2. Delete parents and upload record in SQL
        with Session(self.engine) as session:
            session.execute(delete(parents_table).where(parents_table.c.doc_id == doc_id))
            stmt = delete(uploads_table).where(uploads_table.c.upload_id == upload_id)
            res = session.execute(stmt)
            session.commit()
            return res.rowcount > 0

    def cleanup_expired(self) -> dict[str, int]:
        """Delete all expired uploads and parents across sessions."""
        now = datetime.now(timezone.utc)
        expired_upload_ids: list[str] = []

        with Session(self.engine) as session:
            stmt = select(uploads_table.c.upload_id).where(uploads_table.c.expires_at <= now)
            expired_upload_ids = [r[0] for r in session.execute(stmt)]

        deleted_uploads = 0
        for uid in expired_upload_ids:
            if self.delete_upload(uid):
                deleted_uploads += 1

        # Also purge any dangling parents past expiration
        with Session(self.engine) as session:
            p_stmt = delete(parents_table).where(
                parents_table.c.expires_at.is_not(None),
                parents_table.c.expires_at <= now,
            )
            p_res = session.execute(p_stmt)
            session.commit()
            deleted_parents = p_res.rowcount

        return {
            "deleted_uploads": deleted_uploads,
            "deleted_parents": deleted_parents,
        }
