"""Deterministic identifier generation for Legal RAG."""

import hashlib
import uuid

NAMESPACE_LEGALRAG = uuid.NAMESPACE_URL


def make_uuid5(name: str) -> str:
    """Generate deterministic UUID5 string based on input name."""
    return str(uuid.uuid5(NAMESPACE_LEGALRAG, name))


def hash_text(text: str) -> str:
    """Generate SHA256 hex digest for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_parent_id(doc_id: str, kind: str, identifier: str | int) -> str:
    """
    Generate deterministic parent_id.
    kind: 'A' (Article), 'P' (Preamble), 'F' (Flat window)
    """
    return f"{doc_id}:{kind}{identifier}"


def make_chunk_id(parent_id: str, suffix: str) -> str:
    """
    Generate deterministic chunk_id.
    suffix: 'all', 'K1', 'W0', etc.
    """
    return f"{parent_id}:{suffix}"
