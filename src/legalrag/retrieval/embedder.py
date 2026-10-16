"""Embedder protocol and implementations (Fake and SentenceTransformer)."""

import hashlib
import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for dense embedding models."""
    name: str
    dim: int

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        ...

    def encode_query(self, text: str) -> np.ndarray:
        ...


class FakeEmbedder:
    """Deterministic in-memory embedder for tests without model downloads."""

    def __init__(self, dim: int = 64, name: str = "fake-embedder") -> None:
        self.dim = dim
        self.name = name

    def _hash_to_vec(self, text: str) -> np.ndarray:
        # Generate pseudo-random vector deterministically from text sha256
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.array([self._hash_to_vec(t) for t in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._hash_to_vec(text)


class SentenceTransformerEmbedder:
    """SentenceTransformer wrapper supporting CPU/GPU inference and L2 normalization verification."""

    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        max_seq_length: int = 2048,
        device: str = "cpu",
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.device = device
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

        logger.info("Loading SentenceTransformer model %s on %s...", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.is_normalized = self._check_normalization()

    def _check_normalization(self) -> bool:
        """Encode 8 probe texts to test whether embeddings are L2 normalized (~1.0)."""
        probes = [
            "quy định thời giờ làm việc",
            "hợp đồng lao động xác định thời hạn",
            "nghỉ phép hàng năm của người lao động",
            "kỷ luật sa thải và bồi thường thiệt hại",
            "tiền lương làm thêm giờ ban đêm",
            "bảo hiểm xã hội bắt buộc cho người lao động",
            "thỏa ước lao động tập thể doanh nghiệp",
            "tranh chấp lao động cá nhân và tập thể",
        ]
        vecs = self.model.encode(probes, convert_to_numpy=True, normalize_embeddings=False)
        norms = np.linalg.norm(vecs, axis=1)
        normalized = bool(np.allclose(norms, 1.0, atol=1e-2))
        logger.info(
            "Embedding normalization check for %s: norms min=%.4f, max=%.4f, is_normalized=%s",
            self.name, float(norms.min()), float(norms.max()), normalized,
        )
        return normalized

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        prefixed = [f"{self.doc_prefix}{t}" for t in texts] if self.doc_prefix else texts
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        prefixed = f"{self.query_prefix}{text}" if self.query_prefix else text
        vec = self.model.encode(
            prefixed,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)
