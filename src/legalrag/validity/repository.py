"""Repository for querying document metadata and legal validity relations from SQL."""

import logging
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from legalrag.core.text import normalize_doc_number
from legalrag.data.db import documents_table, relations_table

logger = logging.getLogger(__name__)


class ValidityRepository:
    """SQL repository for document validity metadata and relations."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_document_metadata(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Query document metadata for a list of doc_ids."""
        if not doc_ids:
            return {}

        results: dict[str, dict[str, Any]] = {}
        with Session(self.engine) as session:
            stmt = select(
                documents_table.c.doc_id,
                documents_table.c.doc_number,
                documents_table.c.title,
                documents_table.c.url,
                documents_table.c.status,
                documents_table.c.effective_date,
                documents_table.c.expiry_date,
            ).where(documents_table.c.doc_id.in_(doc_ids))

            for row in session.execute(stmt):
                results[row.doc_id] = {
                    "doc_id": row.doc_id,
                    "doc_number": row.doc_number,
                    "title": row.title,
                    "url": row.url,
                    "status": row.status,
                    "effective_date": row.effective_date,
                    "expiry_date": row.expiry_date,
                }

        return results

    def get_relations(self, doc_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Query incoming relations (dst_doc_id in doc_ids) with rel_type in
        ['REPEALS', 'AMENDS', 'REPEALS_PART'].

        Joins with documents to resolve replacing/amending document title and url.
        """
        if not doc_ids:
            return {}

        results: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in doc_ids}
        with Session(self.engine) as session:
            # Query relations where dst_doc_id is in doc_ids
            stmt = (
                select(
                    relations_table.c.src_doc_id,
                    relations_table.c.rel_type,
                    relations_table.c.dst_doc_id,
                    relations_table.c.dst_title,
                    relations_table.c.dst_url,
                    relations_table.c.resolved,
                    documents_table.c.title.label("src_doc_title"),
                    documents_table.c.url.label("src_doc_url"),
                )
                .select_from(
                    relations_table.outerjoin(
                        documents_table,
                        relations_table.c.src_doc_id == documents_table.c.doc_id,
                    )
                )
                .where(
                    relations_table.c.dst_doc_id.in_(doc_ids),
                    relations_table.c.rel_type.in_(["REPEALS", "AMENDS", "REPEALS_PART"]),
                )
            )

            for row in session.execute(stmt):
                dst = row.dst_doc_id
                if dst not in results:
                    results[dst] = []

                # Title resolution: if source doc is in documents table, use its title;
                # otherwise fallback to dst_title recorded on relation edge, or src_doc_id
                title = row.src_doc_title or row.dst_title or row.src_doc_id
                url = row.src_doc_url or row.dst_url

                results[dst].append({
                    "src_doc_id": row.src_doc_id,
                    "rel_type": row.rel_type,
                    "resolved": bool(row.resolved),
                    "title": title,
                    "url": url,
                })

        return results

    def find_doc_id_by_number(self, identifier: str) -> str | None:
        """
        Find doc_id by exact doc_id, doc_number, or normalized doc_number.
        """
        clean_id = identifier.strip()
        if not clean_id:
            return None

        with Session(self.engine) as session:
            # 1. Exact doc_id
            stmt = select(documents_table.c.doc_id).where(documents_table.c.doc_id == clean_id)
            row = session.execute(stmt).first()
            if row:
                return str(row[0])

            # 2. Exact doc_number
            stmt = select(documents_table.c.doc_id).where(documents_table.c.doc_number == clean_id)
            row = session.execute(stmt).first()
            if row:
                return str(row[0])

            # 3. Normalized doc_number
            norm = normalize_doc_number(clean_id)
            if norm:
                stmt = select(documents_table.c.doc_id).where(documents_table.c.doc_number_norm == norm)
                row = session.execute(stmt).first()
                if row:
                    return str(row[0])

        return None

    def get_corpus_validity_stats(self) -> dict[str, Any]:
        """Compute validity statistics across the entire database."""
        with Session(self.engine) as session:
            # Document status counts
            status_stmt = select(documents_table.c.status)
            statuses = [r[0] for r in session.execute(status_stmt)]
            from collections import Counter
            status_counts = dict(Counter(statuses))

            # Relations stats
            rel_stmt = select(relations_table.c.resolved)
            rel_resolved = [bool(r[0]) for r in session.execute(rel_stmt)]
            resolved_count = sum(rel_resolved)
            unresolved_count = len(rel_resolved) - resolved_count

            # Amended count
            amended_stmt = (
                select(relations_table.c.dst_doc_id)
                .where(relations_table.c.rel_type == "AMENDS")
                .distinct()
            )
            amended_doc_ids = {r[0] for r in session.execute(amended_stmt)}

            # Replaced count
            replaced_stmt = (
                select(relations_table.c.dst_doc_id)
                .where(relations_table.c.rel_type == "REPEALS")
                .distinct()
            )
            replaced_doc_ids = {r[0] for r in session.execute(replaced_stmt)}

        return {
            "total_documents": len(statuses),
            "status_counts": status_counts,
            "total_relations": len(rel_resolved),
            "resolved_relations": resolved_count,
            "unresolved_relations": unresolved_count,
            "documents_with_amended_by": len(amended_doc_ids),
            "documents_with_replaced_by": len(replaced_doc_ids),
        }
