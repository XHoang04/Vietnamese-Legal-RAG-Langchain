"""Syllable-based BM25 sparse tokenizer and vector generator."""

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal


def tokenize_syllables(
    text: str,
    mode: Literal["syllable_bigram", "syllable"] = "syllable_bigram",
) -> list[str]:
    """Tokenize Vietnamese text into syllables and optional adjacent bigrams.

    1. Unicode NFC normalization.
    2. Lowercase.
    3. Split by non-alphanumeric characters (preserving Vietnamese diacritics).
    4. Remove empty tokens.
    5. If syllable_bigram mode, append adjacent bigrams joined by '_'.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFC", text).lower()
    # Split by non-alphanumeric characters
    # In python re with Unicode, \w matches unicode letters and digits
    tokens = [t for t in re.split(r"[^\w]+", normalized, flags=re.UNICODE) if t]

    if mode == "syllable":
        return tokens

    # Syllable bigrams
    bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def hash_token(token: str) -> int:
    """Hash token into a 32-bit integer index using blake2b(digest_size=4)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


@dataclass
class SparseVector:
    """Sparse vector representation with indices and float values."""
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=False))


class SyllableBM25:
    """BM25 sparse encoder generating indices and weights for Qdrant sparse vectors."""

    def __init__(
        self,
        mode: Literal["syllable_bigram", "syllable"] = "syllable_bigram",
        k1: float = 1.2,
        b: float = 0.75,
        avgdl: float = 100.0,
        idf_table: dict[str, float] | None = None,
    ) -> None:
        self.mode = mode
        self.k1 = k1
        self.b = b
        self.avgdl = max(avgdl, 1.0)
        self.idf_table = idf_table or {}

    def encode_document(self, text: str) -> SparseVector:
        """Encode document text into sparse BM25 vector."""
        tokens = tokenize_syllables(text, mode=self.mode)
        if not tokens:
            return SparseVector()

        doc_len = len(tokens)
        counts = Counter(tokens)

        # Aggregate weights by hash index in case of rare hash collisions
        index_weights: dict[int, float] = {}
        for token, tf in counts.items():
            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
            weight = (tf * (self.k1 + 1.0)) / denom
            if self.idf_table and token in self.idf_table:
                weight *= self.idf_table[token]
            idx = hash_token(token)
            index_weights[idx] = index_weights.get(idx, 0.0) + weight

        # Sort by indices for deterministic vector representation
        sorted_items = sorted(index_weights.items(), key=lambda x: x[0])
        return SparseVector(
            indices=[idx for idx, _ in sorted_items],
            values=[float(round(w, 5)) for _, w in sorted_items],
        )

    def encode_query(self, text: str) -> SparseVector:
        """Encode query text into sparse vector (unit weights or idf)."""
        tokens = tokenize_syllables(text, mode=self.mode)
        if not tokens:
            return SparseVector()

        index_weights: dict[int, float] = {}
        counts = Counter(tokens)
        for token in counts:
            weight = self.idf_table.get(token, 1.0)
            idx = hash_token(token)
            index_weights[idx] = max(index_weights.get(idx, 0.0), weight)

        sorted_items = sorted(index_weights.items(), key=lambda x: x[0])
        return SparseVector(
            indices=[idx for idx, _ in sorted_items],
            values=[float(round(w, 5)) for _, w in sorted_items],
        )


def compute_idf(corpus_texts: list[str], mode: Literal["syllable_bigram", "syllable"] = "syllable_bigram") -> tuple[dict[str, float], float]:
    """Compute corpus-wide token IDF and average document length (in tokens)."""
    n_docs = len(corpus_texts)
    if n_docs == 0:
        return {}, 1.0

    df: Counter[str] = Counter()
    total_tokens = 0

    for text in corpus_texts:
        tokens = tokenize_syllables(text, mode=mode)
        total_tokens += len(tokens)
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            df[tok] += 1

    avgdl = total_tokens / max(n_docs, 1)

    idf: dict[str, float] = {}
    for tok, doc_freq in df.items():
        # Standard Lucene/BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        idf[tok] = float(math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)))

    return idf, avgdl
