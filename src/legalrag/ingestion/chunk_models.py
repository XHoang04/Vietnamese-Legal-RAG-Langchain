"""Chunking models and protocol."""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class ChunkSource(Protocol):
    """Protocol for any document that can be chunked."""
    doc_id: str
    title: str
    full_text: str
    source: Literal["corpus", "upload"]
    session_id: str | None


class UploadDocument(BaseModel):
    """Concrete model for uploaded documents implementing ChunkSource."""
    doc_id: str
    title: str
    full_text: str
    source: Literal["corpus", "upload"] = "upload"
    session_id: str | None = None


class Parent(BaseModel):
    """Parent document unit (e.g. an Article, Preamble, or Flat Window)."""
    parent_id: str
    doc_id: str
    article_no: str | None = None
    article_title: str | None = None
    breadcrumb: str
    text: str
    token_count: int
    source: Literal["corpus", "upload"] = "corpus"
    session_id: str | None = None


class Chunk(BaseModel):
    """Child chunk indexed into dense/sparse vector storage."""
    chunk_id: str
    parent_id: str
    doc_id: str
    level: Literal["article", "clause", "flat", "preamble", "appendix"]
    article_no: str | None = None
    clause_no: str | None = None
    breadcrumb: str
    text: str
    embed_text: str
    token_count: int
    content_hash: str
    source: Literal["corpus", "upload"] = "corpus"
    session_id: str | None = None
