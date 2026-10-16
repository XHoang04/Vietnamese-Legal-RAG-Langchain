"""Unit tests for PDF upload, validation, session isolation, TTL cleanup, and privacy."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymupdf as fitz
import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine

from legalrag.core.config import UploadConfig, load_config
from legalrag.core.errors import (
    FileTooLargeError,
    NotAPDFError,
    TooManyPagesError,
    UnsupportedScanPDFError,
    UploadRequiresLocalLLMError,
)
from legalrag.data.db import init_db
from legalrag.ingestion.pdf_loader import load_pdf
from legalrag.llm.factory import get_llm
from legalrag.retrieval.embedder import FakeEmbedder
from legalrag.retrieval.retriever import Retriever
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import init_qdrant_collection
from legalrag.sessions.manager import SessionManager


def create_in_memory_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate a valid PDF in memory using PyMuPDF."""
    doc = fitz.open()
    fontfile = None
    for candidate in [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            fontfile = candidate
            break

    for text in pages_text:
        page = doc.new_page()
        if fontfile:
            page.insert_font(fontname="f0", fontfile=fontfile)
            page.insert_text((50, 72), text, fontname="f0")
        else:
            page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_validation_not_a_pdf():
    invalid_bytes = b"This is a plain text file, not a pdf"
    with pytest.raises(NotAPDFError) as exc_info:
        load_pdf(invalid_bytes, filename="test.pdf")
    assert "NOT_A_PDF" in exc_info.value.code


def test_pdf_validation_file_too_large():
    valid_pdf = create_in_memory_pdf(["Nội dung tài liệu kiểm tra." * 20])
    cfg = UploadConfig(max_size_mb=0)  # 0 MB threshold
    with pytest.raises(FileTooLargeError) as exc_info:
        load_pdf(valid_pdf, filename="test.pdf", upload_config=cfg)
    assert "FILE_TOO_LARGE" in exc_info.value.code


def test_pdf_validation_too_many_pages():
    two_page_pdf = create_in_memory_pdf([
        "Nội dung trang một với đủ số lượng từ ngữ để không bị xem là scan." * 5,
        "Nội dung trang hai với đủ số lượng từ ngữ để không bị xem là scan." * 5,
    ])
    cfg = UploadConfig(max_pages=1)
    with pytest.raises(TooManyPagesError) as exc_info:
        load_pdf(two_page_pdf, filename="test.pdf", upload_config=cfg)
    assert "TOO_MANY_PAGES" in exc_info.value.code


def test_pdf_validation_scanned_pdf_rejected():
    # Only 5 characters on 2 pages -> avg < 100 chars/page
    scanned_pdf = create_in_memory_pdf(["A", "B"])
    cfg = UploadConfig(min_chars_per_page=100)
    with pytest.raises(UnsupportedScanPDFError) as exc_info:
        load_pdf(scanned_pdf, filename="scanned.pdf", upload_config=cfg)
    assert "PDF dạng ảnh chưa được hỗ trợ" in str(exc_info.value)
    assert "UNSUPPORTED_SCAN_PDF" in exc_info.value.code


def test_pdf_loading_valid():
    content = "Hợp đồng lao động mẫu số 01. Nội dung thỏa thuận về tiền lương và bảo hiểm xã hội." * 5
    pdf_bytes = create_in_memory_pdf([content])
    doc = load_pdf(pdf_bytes, filename="hop_dong_lao_dong.pdf")

    assert doc.title == "hop_dong_lao_dong"
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    assert "Hợp đồng lao động" in doc.full_text


@pytest.fixture
def session_env():
    # SQLite in-memory DB
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    # Qdrant in-memory client
    client = QdrantClient(location=":memory:")
    collection_name = "test_collection"
    init_qdrant_collection(client, collection_name=collection_name, dense_dim=64)

    embedder = FakeEmbedder(dim=64, name="fake-embedder")
    bm25 = SyllableBM25(mode="syllable_bigram")

    config = load_config()
    config.app.qdrant.collection = collection_name

    manager = SessionManager(
        db_engine=engine,
        qdrant_client=client,
        embedder=embedder,
        config=config,
        bm25=bm25,
    )

    return {
        "engine": engine,
        "client": client,
        "embedder": embedder,
        "bm25": bm25,
        "config": config,
        "manager": manager,
        "collection_name": collection_name,
    }


def test_upload_processing_and_session_listing(session_env):
    mgr: SessionManager = session_env["manager"]

    text = (
        "Quy chế làm việc nội bộ công ty.\n"
        "Điều 1. Thời giờ làm việc\n"
        "Thời giờ làm việc bình thường là 8 giờ một ngày và 40 giờ một tuần.\n"
        "Điều 2. Nghỉ phép năm\n"
        "Người lao động làm việc đủ 12 tháng được nghỉ 12 ngày phép năm."
    )
    pdf_bytes = create_in_memory_pdf([text])

    # Session A upload
    upload_id_a = mgr.create_upload(session_id="session_A", filename="quyet_dinh.pdf")
    res = mgr.process_upload(upload_id_a, pdf_bytes, filename="quyet_dinh.pdf", session_id="session_A")

    assert res["status"] == "ready"
    assert res["chunks"] >= 2

    # Verify session isolation in listing
    docs_a = mgr.list_session_documents("session_A")
    assert len(docs_a) == 1
    assert docs_a[0]["upload_id"] == upload_id_a
    assert docs_a[0]["status"] == "ready"

    docs_b = mgr.list_session_documents("session_B")
    assert len(docs_b) == 0


def test_session_isolation_in_retriever(session_env):
    mgr: SessionManager = session_env["manager"]
    client = session_env["client"]
    engine = session_env["engine"]
    embedder = session_env["embedder"]
    bm25 = session_env["bm25"]
    col_name = session_env["collection_name"]

    text_a = (
        "Tài liệu mật của phiên A: Mức lương giám đốc phiên A là 100 triệu đồng một tháng."
    )
    pdf_a = create_in_memory_pdf([text_a * 5])
    uid_a = mgr.create_upload(session_id="session_A", filename="tai_lieu_A.pdf")
    mgr.process_upload(uid_a, pdf_a, filename="tai_lieu_A.pdf", session_id="session_A")

    retriever = Retriever(
        client=client,
        embedder=embedder,
        bm25=bm25,
        reranker=None,
        db_engine=engine,
        collection_name=col_name,
        dense_top_k=5,
        sparse_top_k=5,
        hybrid_enabled=True,
        rerank_enabled=False,
    )

    # Query from session A -> retrieves session A upload
    res_a = retriever.retrieve(
        "Mức lương giám đốc",
        scope="upload",
        session_id="session_A",
    )
    assert len(res_a) > 0
    assert res_a[0].session_id == "session_A"
    assert "Mức lương" in res_a[0].text

    # Query from session B -> returns EMPTY (Zero cross-session data leakage!)
    res_b = retriever.retrieve(
        "Mức lương giám đốc",
        scope="upload",
        session_id="session_B",
    )
    assert len(res_b) == 0


def test_delete_upload(session_env):
    mgr: SessionManager = session_env["manager"]

    text = "Nội dung tạm thời cần xóa." * 10
    pdf_bytes = create_in_memory_pdf([text])
    uid = mgr.create_upload(session_id="session_del", filename="delete_me.pdf")
    mgr.process_upload(uid, pdf_bytes, filename="delete_me.pdf", session_id="session_del")

    # Document exists
    assert mgr.get_upload_status(uid) is not None

    # Delete
    deleted = mgr.delete_upload(uid)
    assert deleted is True

    # Record deleted
    assert mgr.get_upload_status(uid) is None


def test_ttl_cleanup_expired(session_env):
    mgr: SessionManager = session_env["manager"]

    text = "Nội dung sẽ hết hạn." * 10
    pdf_bytes = create_in_memory_pdf([text])

    # Upload 1: active (expires in 24 hours)
    uid_active = mgr.create_upload(session_id="sess_active", filename="active.pdf")
    mgr.process_upload(uid_active, pdf_bytes, filename="active.pdf", session_id="sess_active")

    # Upload 2: expired (manually set expired timestamp in past)
    uid_expired = mgr.create_upload(session_id="sess_expired", filename="expired.pdf")
    mgr.process_upload(uid_expired, pdf_bytes, filename="expired.pdf", session_id="sess_expired")

    from sqlalchemy import update

    from legalrag.data.db import parents_table, uploads_table
    past_time = datetime.now(timezone.utc) - timedelta(hours=2)
    with mgr.engine.begin() as conn:
        conn.execute(
            update(uploads_table)
            .where(uploads_table.c.upload_id == uid_expired)
            .values(expires_at=past_time)
        )
        conn.execute(
            update(parents_table)
            .where(parents_table.c.doc_id == f"upload:{uid_expired}")
            .values(expires_at=past_time)
        )

    # Run cleanup
    cleanup_stats = mgr.cleanup_expired()
    assert cleanup_stats["deleted_uploads"] == 1

    # Active remains, expired purged
    assert mgr.get_upload_status(uid_active) is not None
    assert mgr.get_upload_status(uid_expired) is None


def test_privacy_rule_enforcement():
    """Verify Section 13.5: remote LLM rejected when contains_upload=True unless explicitly allowed."""
    cfg = load_config()
    cfg.app.privacy.allow_remote_for_uploads = False
    cfg.models.profiles["private"]["default"]["provider"] = "google_genai"

    # With contains_upload=True and remote provider, it must raise UploadRequiresLocalLLMError
    with pytest.raises(UploadRequiresLocalLLMError):
        get_llm("generator", contains_upload=True, config_bundle=cfg)
