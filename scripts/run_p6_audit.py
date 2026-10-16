"""Audit script for Phase P6: Validity Checker verification across domain corpus."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.core.config import load_config
from legalrag.data.db import get_engine
from legalrag.validity.checker import ValidityChecker
from legalrag.validity.repository import ValidityRepository


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    config_bundle = load_config()
    db_engine = get_engine(config_bundle.app.db.url)
    repo = ValidityRepository(db_engine)
    checker = ValidityChecker(repo, snapshot_as_of=config_bundle.app.snapshot_as_of)

    # 1. Corpus Statistics
    stats = repo.get_corpus_validity_stats()
    print("=================================================================")
    print("BÁO CÁO TOÀN DIỆN HIỆU LỰC KHO VĂN BẢN (PHASE P6 AUDIT)")
    print("=================================================================")
    print(f"Tổng số văn bản trong kho      : {stats['total_documents']}")
    print("Phân bố trạng thái hiệu lực:")
    for status, count in sorted(stats["status_counts"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {status:20s}: {count:3d} văn bản ({count / stats['total_documents'] * 100:.1f}%)")

    print(f"\nTổng số quan hệ pháp lý        : {stats['total_relations']}")
    print(f"  - Đã phân giải (resolved)    : {stats['resolved_relations']} ({stats['resolved_relations'] / stats['total_relations'] * 100:.1f}%)")
    print(f"  - Chưa phân giải             : {stats['unresolved_relations']} ({stats['unresolved_relations'] / stats['total_relations'] * 100:.1f}%)")
    print(f"Số văn bản có amended_by       : {stats['documents_with_amended_by']}")
    print(f"Số văn bản có replaced_by      : {stats['documents_with_replaced_by']}")

    # 2. Check 10 hand-picked sample documents across statuses
    print("\n=================================================================")
    print("KIỂM TRA THỰC TẾ 10 VĂN BẢN TIÊU BIỂU TRONG KHO DỮ LIỆU")
    print("=================================================================")

    # Select 10 documents representing different statuses
    sample_doc_numbers = [
        "58/2014/QH13",      # Luật BHXH 2014 (IN_FORCE)
        "32525",             # Nghị định 43/2013/NĐ-CP (IN_FORCE)
        "105812",            # Nghị định 55/2016/NĐ-CP (IN_FORCE)
        "17/2008/QH12",      # Luật 17/2008/QH12 (EXPIRED)
        "13686",             # Nghị định 133/2007/NĐ-CP (EXPIRED)
        "16829",             # Nghị định 03/2006/NĐ-CP (EXPIRED)
        "13422",             # Nghị định 141/2007/NĐ-CP (EXPIRED)
        "25/2008/QH12",      # Luật 25/2008/QH12 (PARTIALLY_EXPIRED + amended)
        "70/2006/QH11",      # Luật 70/2006/QH11 (PARTIALLY_EXPIRED)
        "95691",             # Nghị định 01/2016/NĐ-CP (PARTIALLY_EXPIRED)
    ]

    for idx, num in enumerate(sample_doc_numbers, 1):
        doc_id = repo.find_doc_id_by_number(num)
        if not doc_id:
            print(f"[{idx:2d}] Số hiệu {num}: Không tìm thấy trong kho.")
            continue

        info = checker.check([doc_id])[doc_id]
        print(f"\n[{idx:2d}] {info.title} (Số hiệu: {num}, Doc ID: {doc_id})")
        print(f"     Trạng thái    : {info.status.value if info.status else 'UNKNOWN'}")
        print(f"     Hiệu lực      : {info.effective_date} -> {info.expiry_date or 'hiện tại'}")

        if info.replaced_by:
            rep_str = "; ".join(r.title for r in info.replaced_by)
            print(f"     Thay thế bởi  : {rep_str}")
        if info.amended_by:
            amd_str = "; ".join(r.title for r in info.amended_by)
            print(f"     Sửa đổi bởi   : {amd_str}")
        if info.partially_repealed_by:
            prp_str = "; ".join(r.title for r in info.partially_repealed_by)
            print(f"     Bãi bỏ 1 phần : {prp_str}")

        if info.warnings:
            for w in info.warnings:
                print(f"     [CẢNH BÁO] {w}")
        else:
            print("     [CẢNH BÁO] Không có cảnh báo (Văn bản còn đầy đủ hiệu lực).")

    print("\n=================================================================")
    print("HOÀN THÀNH AUDIT PHASE P6")
    print("=================================================================")


if __name__ == "__main__":
    main()
