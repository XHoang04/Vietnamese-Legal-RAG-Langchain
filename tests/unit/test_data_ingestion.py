"""Unit tests for Phase P1 data ingestion components."""

from datetime import date

from legalrag.core.config import FiltersConfig, load_config
from legalrag.data.db import get_engine, init_db, save_documents_and_relations
from legalrag.data.filters import evaluate_document_filter
from legalrag.data.metadata import (
    extract_subject,
    parse_date,
    parse_doc_number,
    parse_status,
)
from legalrag.data.models import LegalDocument, RelType, Status
from legalrag.data.relations import extract_relations_from_schema
from legalrag.data.subset import select_domain_subset


def test_metadata_parsing() -> None:
    # 1. Status parsing
    attr_in_force = {"document_info": ["Trích yếu văn bản", "Tình trạng: Còn hiệu lực"]}
    st, raw = parse_status(attr_in_force)
    assert st == Status.IN_FORCE
    assert raw == "Còn hiệu lực"

    attr_expired = {"document_info": ["Tình trạng: Hết hiệu lực toàn bộ"]}
    st2, _ = parse_status(attr_expired)
    assert st2 == Status.EXPIRED

    attr_part = {"document_info": ["Tình trạng: Hết hiệu lực một phần"]}
    st3, _ = parse_status(attr_part)
    assert st3 == Status.PARTIALLY_EXPIRED

    # 2. Date parsing
    assert parse_date("01/01/2021") == date(2021, 1, 1)
    assert parse_date("31/12/2020") == date(2020, 12, 31)
    assert parse_date("invalid") is None
    assert parse_date("") is None

    # 3. Subject extraction
    subj = extract_subject(attr_in_force)
    assert subj == "Trích yếu văn bản"

    # 4. Doc number parsing
    test_titles = [
        ("Nghị quyết 19/2020/NQ-HĐND", "Nghị quyết", "19/2020/NQ-HĐND", "192020NQHĐND"),
        ("Quyết định 37.2021.QD-UBND", "Quyết định", "37.2021.QD-UBND", "372021QDUBND"),
        ("Thông tư 04/2009/TT-BNN", "Thông tư", "04/2009/TT-BNN", "042009TTBNN"),
    ]
    for raw_title, exp_type, exp_num, exp_norm in test_titles:
        dt, num, norm = parse_doc_number(raw_title)
        assert dt == exp_type
        assert num == exp_num
        assert norm == exp_norm


def test_filter_evaluation() -> None:
    cfg = FiltersConfig(min_text_chars=200, mismatch_overlap_min=0.3)

    # English URL should be discarded
    reason, _, _ = evaluate_document_filter(
        url="http://vbpl.vn/vbpqen-toanvan.aspx?ItemID=1",
        full_text="Long text " * 50,
        subject="Tiêu đề",
        status=Status.IN_FORCE,
        status_raw="Còn hiệu lực",
        cfg=cfg,
    )
    assert reason == "ENGLISH_URL"

    # Short text should be discarded
    reason, _, _ = evaluate_document_filter(
        url="http://vbpl.vn/vbpq-toanvan.aspx?ItemID=1",
        full_text="Quá ngắn",
        subject="Tiêu đề",
        status=Status.IN_FORCE,
        status_raw="Còn hiệu lực",
        cfg=cfg,
    )
    assert reason == "EMPTY_TEXT"

    # Mismatch text should be kept but flagged
    reason, flag, st = evaluate_document_filter(
        url="http://vbpl.vn/vbpq-toanvan.aspx?ItemID=1",
        full_text="Nội dung hoàn toàn về giao thông đường sắt và cầu cống thủy lợi " * 20,
        subject="Quy định tiền lương và bảo hiểm y tế cho người lao động",
        status=Status.IN_FORCE,
        status_raw="Còn hiệu lực",
        cfg=cfg,
    )
    assert reason is None
    assert flag == "METADATA_MISMATCH"
    assert st == Status.UNKNOWN


def test_relation_extraction() -> None:
    schema = {
        "cancel_documents": [
            ["Bộ luật Lao động 2019", "http://vbpl.vn/laodong/Pages/vbpq-toanvan.aspx?ItemID=456"]
        ],
        "amended_documents": [
            ["Nghị định cũ", "http://vbpl.vn/laodong/Pages/vbpq-toanvan.aspx?ItemID=789"]
        ],
    }

    relations = extract_relations_from_schema("123", schema)
    assert len(relations) == 2

    # cancel_documents on R (123) means 456 repeals 123
    repeal_rel = next(r for r in relations if r.rel_type == RelType.REPEALS)
    assert repeal_rel.src_doc_id == "456"
    assert repeal_rel.dst_doc_id == "123"
    assert repeal_rel.resolved is True

    # amended_documents on R (123) means 123 amends 789
    amend_rel = next(r for r in relations if r.rel_type == RelType.AMENDS)
    assert amend_rel.src_doc_id == "123"
    assert amend_rel.dst_doc_id == "789"


def test_subset_selection() -> None:
    bundle = load_config()
    cfg = bundle.domain

    docs = [
        LegalDocument(
            doc_id="1",
            title="Bộ luật Lao động 2019",
            doc_number="45/2019/QH14",
            doc_number_norm="452019QH14",
            doc_type="Bộ luật",
            effective_area="Toàn quốc",
            subject="Quy định tiêu chuẩn lao động",
            status=Status.IN_FORCE,
        ),
        LegalDocument(
            doc_id="2",
            title="Nghị định quy định chi tiết Bộ luật Lao động",
            doc_number="145/2020/NĐ-CP",
            doc_number_norm="1452020NĐCP",
            doc_type="Nghị định",
            effective_area="Toàn quốc",
            subject="Hợp đồng lao động, tiền lương",
            status=Status.IN_FORCE,
        ),
        LegalDocument(
            doc_id="3",
            title="Quyết định về đê điều thủy lợi",
            doc_number="01/2020/QĐ-BNN",
            doc_number_norm="012020QĐBNN",
            doc_type="Quyết định",
            effective_area="Toàn quốc",
            subject="Thủy lợi đê điều",
            status=Status.IN_FORCE,
        ),
    ]

    selected_docs, _, meta = select_domain_subset(docs, [], cfg)
    selected_ids = [d.doc_id for d in selected_docs]
    assert "1" in selected_ids
    assert "2" in selected_ids
    assert "3" not in selected_ids


def test_sql_database_ops() -> None:
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)

    docs = [{
        "doc_id": "doc1",
        "doc_number": "45/2019/QH14",
        "doc_number_norm": "452019QH14",
        "title": "Bộ luật Lao động",
        "subject": "Quy định lao động",
        "doc_type": "Bộ luật",
        "effective_area": "Toàn quốc",
        "status": "IN_FORCE",
        "status_raw": "Còn hiệu lực",
        "effective_date": date(2021, 1, 1),
        "enforced_date": None,
        "expiry_date": None,
        "gazette_date": None,
        "url": "http://vbpl.vn/doc1",
        "last_updated_time": None,
        "metadata_flag": None,
    }]

    rels = [{
        "src_doc_id": "doc1",
        "rel_type": "AMENDS",
        "dst_doc_id": "doc2",
        "dst_title": "Bộ luật cũ",
        "dst_url": "http://vbpl.vn/doc2",
        "raw_key": "amended_documents",
        "resolved": True,
    }]

    save_documents_and_relations(engine, docs, rels)
