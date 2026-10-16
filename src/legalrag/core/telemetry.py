"""Telemetry and execution trace models."""

from pydantic import BaseModel


class TraceEvent(BaseModel):
    """Event recording execution inside the RAG graph."""
    node: str
    ms: float
    note: str | None = None
