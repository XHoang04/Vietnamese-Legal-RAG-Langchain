"""Unit tests for hierarchical legal chunker (Section 8.3 of PLAN.md)."""

from legalrag.core.config import ChunkingConfig
from legalrag.data.models import LegalDocument
from legalrag.ingestion.legal_chunker import chunk_document


def test_chunker_structured_hierarchy() -> None:
    """Test full hierarchy: Chương, Mục, Điều, Khoản, Điểm."""
    text = (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n\n"
        "CHƯƠNG I\nQUY ĐỊNH CHUNG\n\n"
        "Mục 1\nPHẠM VI ĐIỀU CHỈNH\n\n"
        "Điều 1. Phạm vi điều chỉnh\n"
        "1. Bộ luật này quy định tiêu chuẩn lao động; quyền, nghĩa vụ của người lao động.\n"
        "2. Áp dụng đối với các đối tượng sau đây:\n"
        "a) Người lao động làm việc theo hợp đồng lao động;\n"
        "b) Người sử dụng lao động.\n\n"
        "Điều 2. Đối tượng áp dụng\n"
        "Quy định chung áp dụng trên toàn lãnh thổ."
    )

    doc = LegalDocument(
        doc_id="doc_struct",
        title="Bộ luật Lao động",
        full_text=text,
    )

    cfg = ChunkingConfig(
        min_article_tokens_for_split=10,  # low threshold to trigger clause split
        min_chunk_tokens=5,
        max_chunk_tokens=500,
        index_preamble=True,
    )

    parents, chunks = chunk_document(doc, cfg)

    # 1 preamble + 2 articles = 3 parents
    assert len(parents) == 3
    assert parents[0].article_title == "Phần mở đầu"
    assert parents[1].article_no == "1"
    assert parents[2].article_no == "2"

    # Breadcrumb checks
    assert "CHƯƠNG I" in parents[1].breadcrumb
    assert "Mục 1" in parents[1].breadcrumb
    assert "Điều 1" in parents[1].breadcrumb

    # Check clause chunks for Article 1
    art1_chunks = [c for c in chunks if c.parent_id == "doc_struct:A1"]
    assert len(art1_chunks) == 2
    assert art1_chunks[0].clause_no == "1"
    assert art1_chunks[1].clause_no == "2"
    # Point a) and b) stay within Clause 2
    assert "a) Người lao động" in art1_chunks[1].text
    assert "b) Người sử dụng lao động" in art1_chunks[1].text


def test_chunker_mid_sentence_citation_not_split() -> None:
    """Test that 'theo Điều 5' in the middle of a sentence does NOT create a new Article."""
    text = (
        "Điều 10. Chế độ nâng lương\n"
        "Người sử dụng lao động xem xét nâng lương theo Điều 5 của quy chế nội bộ và báo cáo công đoàn.\n"
        "Điều 11. Khen thưởng\n"
        "Quy định về khen thưởng hàng năm."
    )

    doc = LegalDocument(
        doc_id="doc_cite",
        title="Nội quy lao động",
        full_text=text,
    )

    cfg = ChunkingConfig(min_article_tokens_for_split=400)
    parents, chunks = chunk_document(doc, cfg)

    # Must have only Điều 10 and Điều 11, NOT Điều 5
    art_nos = [p.article_no for p in parents if p.article_no]
    assert art_nos == ["10", "11"]


def test_chunker_flat_window_mode() -> None:
    """Test documents without articles (Directive, short Decision) use flat windowing."""
    text = (
        "QUYẾT ĐỊNH\nVề việc chỉ đạo sản xuất vụ đông xuân.\n\n"
        "Ủy ban nhân dân tỉnh yêu cầu các sở ngành tập trung chuẩn bị giống vốn, "
        "thực hiện nghiêm túc lịch thời vụ nông nghiệp trên toàn địa bàn tỉnh.\n\n"
        "Các cơ quan truyền thông tăng cường thông tin tuyên truyền rộng rãi tới bà con nhân dân."
    )

    doc = LegalDocument(
        doc_id="doc_flat",
        title="Chỉ thị về nông nghiệp",
        full_text=text,
    )

    cfg = ChunkingConfig(flat_window_tokens=100, flat_overlap_tokens=20)
    parents, chunks = chunk_document(doc, cfg)

    assert len(parents) >= 1
    assert all(p.article_no is None for p in parents)
    assert all(c.level == "flat" for c in chunks)
    assert chunks[0].breadcrumb == "Chỉ thị về nông nghiệp"


def test_chunker_idempotency() -> None:
    """Test identical input produces identical chunk IDs and content hashes."""
    text = (
        "Điều 1. Tiêu chuẩn\n1. Tiêu chuẩn chung.\n2. Tiêu chuẩn riêng.\n"
        "Điều 2. Hồ sơ\nThủ tục hồ sơ theo mẫu."
    )
    doc = LegalDocument(doc_id="doc_idem", title="Quy chế", full_text=text)
    cfg = ChunkingConfig(min_article_tokens_for_split=5)

    p1, c1 = chunk_document(doc, cfg)
    p2, c2 = chunk_document(doc, cfg)

    assert [p.parent_id for p in p1] == [p.parent_id for p in p2]
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
    assert [c.content_hash for c in c1] == [c.content_hash for c in c2]
