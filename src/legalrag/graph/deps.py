"""Dependencies passed to the graph and its nodes."""

from dataclasses import dataclass
from typing import Any, Protocol

from legalrag.core.config import ConfigBundle
from legalrag.retrieval.retriever import BaseRetriever
from legalrag.validity.checker import BaseValidityChecker


class LLMFactory(Protocol):
    def get_llm(self, role: str, *, contains_upload: bool = False) -> Any:
        ...


@dataclass
class GraphDeps:
    """Runtime dependencies container injected into all nodes."""
    config: ConfigBundle
    retriever: BaseRetriever
    validity_checker: BaseValidityChecker
    llm_factory: LLMFactory
