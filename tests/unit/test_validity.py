"""Unit tests for validity checking, warning templates, and replacement expansion."""

from datetime import date

import pytest
from sqlalchemy import create_engine

from legalrag.data.db import init_db, save_documents_and_relations
from legalrag.data.models import Status
from legalrag.retrieval.retriever import FakeRetriever, RetrievedContext
from legalrag.validity.checker import (
    DocRef,
    ValidityChecker,
    ValidityInfo,
    expand_replacements,
)
from legalrag.validity.repository import ValidityRepository
from legalrag.validity.templates import (
    format_doc_refs,
    snapshot_notice,
    warn_amended,
    warn_expired,
    warn_partially_expired,
    warn_suspended,
    warn_unknown,
    warn_upload,
)


def test_format_doc_refs():
    assert format_doc_refs(["Luật A", "Luật B"]) == "Luật A, Luật B"
    assert format_doc_refs([]) == ""


def test_template_warn_expired():
    d = date(2020, 1, 1)
    msg1 = warn_expired("Luật A", ["Luật B"], d)
    assert msg1 == "Văn bản Luật A đã hết hiệu lực toàn bộ (hết hiệu lực từ 01/01/2020); bị thay thế/bãi bỏ bởi: Luật B."

    msg2 = warn_expired("Luật A", [], d)
    assert msg2 == "Văn bản Luật A đã hết hiệu lực toàn bộ (hết hiệu lực từ 01/01/2020)."

    msg3 = warn_expired("Luật A", ["Luật B"], None)
    assert msg3 == "Văn bản Luật A đã hết hiệu lực toàn bộ; bị thay thế/bãi bỏ bởi: Luật B."

    msg4 = warn_expired("Luật A", [], None)
    assert msg4 == "Văn bản Luật A đã hết hiệu lực toàn bộ."


def test_template_warn_partially_expired():
    msg = warn_partially_expired("Luật C")
    assert "đã hết hiệu lực một phần" in msg
    assert "cần đối chiếu văn bản gốc" in msg

    msg_with_repeal = warn_partially_expired("Luật C", ["Luật D"])
    assert "Đã có văn bản bãi bỏ một phần: Luật D." in msg_with_repeal


def test_template_warn_suspended():
    msg = warn_suspended("Luật E")
    assert msg == "Văn bản Luật E đang bị ngưng hiệu lực."


def test_template_warn_unknown():
    msg = warn_unknown("Luật F")
    assert msg == "Không xác định được trạng thái hiệu lực của văn bản Luật F."


def test_template_warn_amended():
    msg = warn_amended("Luật G", ["Luật H", "Luật K"])
    assert "đã được sửa đổi, bổ sung bởi: Luật H, Luật K." in msg
    assert "Nội dung trích dẫn có thể chưa phản ánh các sửa đổi này." in msg


def test_template_warn_upload_and_snapshot():
    assert warn_upload() == "Tài liệu do bạn tải lên, hệ thống không xác minh hiệu lực."
    assert snapshot_notice("2024-09-10") == "Dữ liệu kho luật cập nhật đến 2024-09-10."


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    docs = [
        {
            "doc_id": "doc_in_force",
            "doc_number": "45/2019/QH14",
            "doc_number_norm": "452019QH14",
            "title": "Bộ luật Lao động 2019",
            "status": "IN_FORCE",
            "effective_date": date(2021, 1, 1),
            "expiry_date": None,
            "url": "http://example.com/45",
        },
        {
            "doc_id": "doc_expired",
            "doc_number": "10/2012/QH13",
            "doc_number_norm": "102012QH13",
            "title": "Bộ luật Lao động 2012",
            "status": "EXPIRED",
            "effective_date": date(2013, 5, 1),
            "expiry_date": date(2021, 1, 1),
            "url": "http://example.com/10",
        },
        {
            "doc_id": "doc_partially_expired",
            "doc_number": "25/2008/QH12",
            "doc_number_norm": "252008QH12",
            "title": "Luật 25/2008/QH12",
            "status": "PARTIALLY_EXPIRED",
            "effective_date": date(2009, 7, 1),
            "expiry_date": None,
            "url": "http://example.com/25",
        },
        {
            "doc_id": "doc_suspended",
            "doc_number": "01/2020/TT-BLDTBXH",
            "doc_number_norm": "012020TTBLDTBXH",
            "title": "Thông tư 01/2020/TT-BLĐTBXH",
            "status": "SUSPENDED",
            "effective_date": date(2020, 3, 1),
            "expiry_date": None,
            "url": "http://example.com/01",
        },
    ]

    relations = [
        # doc_in_force REPEALS doc_expired
        {
            "src_doc_id": "doc_in_force",
            "rel_type": "REPEALS",
            "dst_doc_id": "doc_expired",
            "dst_title": "Bộ luật Lao động 2012",
            "dst_url": "http://example.com/10",
            "raw_key": "canceled_documents",
            "resolved": True,
        },
        # external doc AMENDS doc_in_force
        {
            "src_doc_id": "doc_amender",
            "rel_type": "AMENDS",
            "dst_doc_id": "doc_in_force",
            "dst_title": "Luật sửa đổi bổ sung năm 2023",
            "dst_url": "http://example.com/amender",
            "raw_key": "amended_documents",
            "resolved": False,
        },
    ]

    save_documents_and_relations(engine, docs, relations)
    return engine


def test_validity_checker_all_statuses(sqlite_engine):
    repo = ValidityRepository(sqlite_engine)
    checker = ValidityChecker(repo, snapshot_as_of="2024-09-10")

    doc_ids = [
        "doc_in_force",
        "doc_expired",
        "doc_partially_expired",
        "doc_suspended",
        "doc_non_existent",
    ]

    results = checker.check(doc_ids)

    # 1. doc_in_force: IN_FORCE, has AMENDS warning
    info_in_force = results["doc_in_force"]
    assert info_in_force.status == Status.IN_FORCE
    assert len(info_in_force.amended_by) == 1
    assert "Luật sửa đổi bổ sung năm 2023" in info_in_force.amended_by[0].title
    assert any("đã được sửa đổi, bổ sung bởi" in w for w in info_in_force.warnings)

    # 2. doc_expired: EXPIRED, replaced_by doc_in_force
    info_expired = results["doc_expired"]
    assert info_expired.status == Status.EXPIRED
    assert len(info_expired.replaced_by) == 1
    assert info_expired.replaced_by[0].doc_id == "doc_in_force"
    assert "Bộ luật Lao động 2019" in info_expired.replaced_by[0].title
    assert any("đã hết hiệu lực toàn bộ" in w for w in info_expired.warnings)
    assert any("bị thay thế/bãi bỏ bởi: Bộ luật Lao động 2019" in w for w in info_expired.warnings)

    # 3. doc_partially_expired: PARTIALLY_EXPIRED
    info_part = results["doc_partially_expired"]
    assert info_part.status == Status.PARTIALLY_EXPIRED
    assert any("đã hết hiệu lực một phần" in w for w in info_part.warnings)

    # 4. doc_suspended: SUSPENDED
    info_susp = results["doc_suspended"]
    assert info_susp.status == Status.SUSPENDED
    assert any("đang bị ngưng hiệu lực" in w for w in info_susp.warnings)

    # 5. doc_non_existent: UNKNOWN
    info_unknown = results["doc_non_existent"]
    assert info_unknown.status == Status.UNKNOWN
    assert any("Không xác định được trạng thái hiệu lực" in w for w in info_unknown.warnings)


def test_find_doc_id_by_number(sqlite_engine):
    repo = ValidityRepository(sqlite_engine)

    # Exact doc_id
    assert repo.find_doc_id_by_number("doc_in_force") == "doc_in_force"
    # Exact doc_number
    assert repo.find_doc_id_by_number("45/2019/QH14") == "doc_in_force"
    # Normalized doc_number
    assert repo.find_doc_id_by_number("452019qh14") == "doc_in_force"
    assert repo.find_doc_id_by_number("45/2019/qh14") == "doc_in_force"
    # Non-existent
    assert repo.find_doc_id_by_number("999/9999/QH99") is None


def test_expand_replacements():
    # Setup mock retriever
    replacement_context = RetrievedContext(
        parent_id="doc_in_force:A1",
        doc_id="doc_in_force",
        title="Bộ luật Lao động 2019",
        breadcrumb="Điều 1",
        text="Quy định tiêu chuẩn lao động...",
        best_score=8.5,
    )
    retriever = FakeRetriever([replacement_context])

    initial_context = RetrievedContext(
        parent_id="doc_expired:A1",
        doc_id="doc_expired",
        title="Bộ luật Lao động 2012",
        breadcrumb="Điều 1",
        text="Quy định cũ...",
        best_score=5.0,
    )

    validity_map = {
        "doc_expired": ValidityInfo(
            doc_id="doc_expired",
            status=Status.EXPIRED,
            replaced_by=[DocRef(doc_id="doc_in_force", title="Bộ luật Lao động 2019")],
            warnings=["Hết hiệu lực"],
        )
    }

    contexts = [initial_context]
    expanded = expand_replacements(
        retriever=retriever,
        query="tiêu chuẩn lao động",
        contexts=contexts,
        validity_map=validity_map,
        max_parents=2,
    )

    assert len(expanded) == 2
    assert expanded[0].doc_id == "doc_expired"
    assert expanded[0].is_replacement is False
    assert expanded[1].doc_id == "doc_in_force"
    assert expanded[1].is_replacement is True


def test_corpus_validity_stats(sqlite_engine):
    repo = ValidityRepository(sqlite_engine)
    stats = repo.get_corpus_validity_stats()

    assert stats["total_documents"] == 4
    assert stats["total_relations"] == 2
    assert stats["resolved_relations"] == 1
    assert stats["unresolved_relations"] == 1
    assert stats["documents_with_replaced_by"] == 1
    assert stats["documents_with_amended_by"] == 1
