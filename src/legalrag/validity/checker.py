"""Validity checker service and models."""

import logging
from datetime import date
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import Engine

from legalrag.data.models import Status
from legalrag.retrieval.retriever import BaseRetriever, RetrievedContext
from legalrag.validity.repository import ValidityRepository
from legalrag.validity.templates import (
    warn_amended,
    warn_expired,
    warn_partially_expired,
    warn_suspended,
    warn_unknown,
)

logger = logging.getLogger(__name__)


class DocRef(BaseModel):
    """Reference to a related legal document."""
    doc_id: str | None = None
    title: str
    url: str | None = None


class ValidityInfo(BaseModel):
    """Validity metadata and warnings for a legal document."""
    doc_id: str
    title: str | None = None
    status: Status | None = Status.UNKNOWN
    effective_date: date | None = None
    expiry_date: date | None = None
    replaced_by: list[DocRef] = Field(default_factory=list)
    amended_by: list[DocRef] = Field(default_factory=list)
    partially_repealed_by: list[DocRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@runtime_checkable
class BaseValidityChecker(Protocol):
    """Protocol for checking legal document validity."""

    def check(self, doc_ids: list[str]) -> dict[str, ValidityInfo]:
        ...


class ValidityChecker(BaseValidityChecker):
    """Concrete validity checker querying SQL database via ValidityRepository."""

    def __init__(
        self,
        repo: ValidityRepository | Engine,
        snapshot_as_of: str | date | None = "2024-09-10",
    ) -> None:
        if isinstance(repo, Engine):
            self.repo = ValidityRepository(repo)
        else:
            self.repo = repo

        if isinstance(snapshot_as_of, str):
            try:
                self.snapshot_date: date | None = date.fromisoformat(snapshot_as_of)
            except ValueError:
                self.snapshot_date = None
        else:
            self.snapshot_date = snapshot_as_of

    def check(self, doc_ids: list[str]) -> dict[str, ValidityInfo]:
        """Check validity status and generate warnings for given document IDs."""
        if not doc_ids:
            return {}

        # De-duplicate while preserving order
        unique_doc_ids = list(dict.fromkeys(doc_ids))

        docs_meta = self.repo.get_document_metadata(unique_doc_ids)
        relations_map = self.repo.get_relations(unique_doc_ids)

        results: dict[str, ValidityInfo] = {}

        for did in unique_doc_ids:
            if did not in docs_meta:
                # Document not found in corpus
                results[did] = ValidityInfo(
                    doc_id=did,
                    status=Status.UNKNOWN,
                    warnings=[warn_unknown(did)],
                )
                continue

            meta = docs_meta[did]
            title = meta.get("title") or did

            # Parse status
            raw_status = meta.get("status")
            status_enum = Status.UNKNOWN
            if raw_status:
                try:
                    status_enum = Status(raw_status)
                except ValueError:
                    status_enum = Status.UNKNOWN

            # Parse dates
            effective_date = meta.get("effective_date")
            if isinstance(effective_date, str):
                try:
                    effective_date = date.fromisoformat(effective_date)
                except ValueError:
                    effective_date = None

            expiry_date = meta.get("expiry_date")
            if isinstance(expiry_date, str):
                try:
                    expiry_date = date.fromisoformat(expiry_date)
                except ValueError:
                    expiry_date = None

            # Process relations
            rels = relations_map.get(did, [])
            replaced_by: list[DocRef] = []
            amended_by: list[DocRef] = []
            partially_repealed_by: list[DocRef] = []

            for r in rels:
                ref = DocRef(
                    doc_id=r["src_doc_id"] if r.get("resolved") else None,
                    title=r["title"],
                    url=r.get("url"),
                )
                rel_type = r.get("rel_type")
                if rel_type == "REPEALS":
                    replaced_by.append(ref)
                elif rel_type == "AMENDS":
                    amended_by.append(ref)
                elif rel_type == "REPEALS_PART":
                    partially_repealed_by.append(ref)

            # Build warnings
            warnings: list[str] = []

            if status_enum == Status.EXPIRED:
                # Expired warning: include expiry_date if present and <= snapshot_as_of (or snapshot_date is None)
                active_expiry = None
                if expiry_date:
                    if self.snapshot_date is None or expiry_date <= self.snapshot_date:
                        active_expiry = expiry_date
                rep_titles = [ref.title for ref in replaced_by]
                warnings.append(warn_expired(title, rep_titles, active_expiry))
            elif status_enum == Status.PARTIALLY_EXPIRED:
                part_titles = [ref.title for ref in partially_repealed_by]
                warnings.append(warn_partially_expired(title, part_titles))
            elif status_enum == Status.SUSPENDED:
                warnings.append(warn_suspended(title))
            elif status_enum == Status.UNKNOWN:
                warnings.append(warn_unknown(title))

            # Amended rule applies regardless of status
            if amended_by:
                amended_titles = [ref.title for ref in amended_by]
                warnings.append(warn_amended(title, amended_titles))

            results[did] = ValidityInfo(
                doc_id=did,
                title=title,
                status=status_enum,
                effective_date=effective_date,
                expiry_date=expiry_date,
                replaced_by=replaced_by,
                amended_by=amended_by,
                partially_repealed_by=partially_repealed_by,
                warnings=warnings,
            )

        return results


def expand_replacements(
    retriever: BaseRetriever,
    query: str,
    contexts: list[RetrievedContext],
    validity_map: dict[str, ValidityInfo],
    max_parents: int = 2,
) -> list[RetrievedContext]:
    """
    Expand contexts with replacing documents (flag validity.expand_replacements).
    If replaced_by doc_id exists and is not yet in contexts, retrieves up to
    max_parents marked with is_replacement=True.
    """
    existing_doc_ids = {c.doc_id for c in contexts}
    replacement_doc_ids: list[str] = []

    for c in contexts:
        info = validity_map.get(c.doc_id)
        if not info:
            continue
        for ref in info.replaced_by:
            if ref.doc_id and ref.doc_id not in existing_doc_ids and ref.doc_id not in replacement_doc_ids:
                replacement_doc_ids.append(ref.doc_id)

    if not replacement_doc_ids:
        return contexts

    logger.info("Expanding replacements for candidate doc_ids: %s", replacement_doc_ids)
    try:
        replacement_contexts = retriever.retrieve(
            query=query,
            doc_ids=replacement_doc_ids,
            max_parents=max_parents,
        )
        for r_ctx in replacement_contexts:
            r_ctx.is_replacement = True
            contexts.append(r_ctx)
    except Exception as e:
        logger.warning("Failed to expand replacements: %s", e)

    return contexts


class FakeValidityChecker:
    """Fake validity checker with preconfigured mock data for tests."""

    def __init__(self, mock_data: dict[str, ValidityInfo] | None = None) -> None:
        self.mock_data = mock_data or {}

    def check(self, doc_ids: list[str]) -> dict[str, ValidityInfo]:
        results: dict[str, ValidityInfo] = {}
        for d in doc_ids:
            if d in self.mock_data:
                results[d] = self.mock_data[d]
            else:
                results[d] = ValidityInfo(
                    doc_id=d,
                    status=Status.IN_FORCE,
                    warnings=[],
                )
        return results
