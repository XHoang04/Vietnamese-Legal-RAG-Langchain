"""Unit tests for core utilities."""

from pathlib import Path

from legalrag.core.config import load_config
from legalrag.core.ids import hash_text, make_chunk_id, make_parent_id, make_uuid5
from legalrag.core.text import count_tokens, normalize_doc_number, normalize_whitespace


def test_ids_deterministic() -> None:
    assert make_uuid5("chunk:1") == make_uuid5("chunk:1")
    assert hash_text("hello world") == hash_text("hello world")
    assert make_parent_id("doc1", "A", "5") == "doc1:A5"
    assert make_chunk_id("doc1:A5", "all") == "doc1:A5:all"


def test_text_normalization() -> None:
    raw = "  Điều   5.  Quyền   và   nghĩa vụ\r\n\r\nCủa người   lao động.  "
    norm = normalize_whitespace(raw)
    assert "\r" not in norm
    assert "Điều 5. Quyền và nghĩa vụ" in norm
    assert "Của người lao động." in norm

    assert normalize_doc_number(" 45/2019/QH14 - ") == "452019QH14"
    assert normalize_doc_number("37.2021.QD-UBND") == "372021QDUBND"
    assert count_tokens("Xin chào Việt Nam!") > 0


def test_config_loading() -> None:
    bundle = load_config(Path("configs"))
    assert bundle.app.snapshot_as_of == "2024-09-10"
    assert bundle.domain.domain == "lao_dong"
    assert bundle.retrieval.dense_top_k == 40
    assert bundle.app.graph.mode == "lean"
    assert bundle.app.graph.router.merge_rewrite is True
