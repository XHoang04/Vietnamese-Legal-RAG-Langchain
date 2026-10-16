"""Templates for validity warnings and notices in Vietnamese."""

from datetime import date
from typing import Sequence


def format_doc_refs(refs: Sequence[str]) -> str:
    """Format list of document titles."""
    return ", ".join(refs)


def warn_expired(title: str, replaced_by: Sequence[str] = (), expiry_date: date | None = None) -> str:
    msg = f"Văn bản {title} đã hết hiệu lực toàn bộ"
    if expiry_date:
        msg += f" (hết hiệu lực từ {expiry_date.strftime('%d/%m/%Y')})"
    if replaced_by:
        msg += f"; bị thay thế/bãi bỏ bởi: {format_doc_refs(replaced_by)}"
    return msg + "."


def warn_partially_expired(title: str, partially_repealed_by: Sequence[str] = ()) -> str:
    msg = f"Văn bản {title} đã hết hiệu lực một phần. Hệ thống chưa xác định được điều khoản nào còn hiệu lực, cần đối chiếu văn bản gốc."
    if partially_repealed_by:
        msg += f" Đã có văn bản bãi bỏ một phần: {format_doc_refs(partially_repealed_by)}."
    return msg


def warn_suspended(title: str) -> str:
    return f"Văn bản {title} đang bị ngưng hiệu lực."


def warn_unknown(title: str) -> str:
    return f"Không xác định được trạng thái hiệu lực của văn bản {title}."


def warn_amended(title: str, amended_by: Sequence[str]) -> str:
    return (
        f"Văn bản {title} đã được sửa đổi, bổ sung bởi: {format_doc_refs(amended_by)}. "
        f"Nội dung trích dẫn có thể chưa phản ánh các sửa đổi này."
    )


def warn_upload() -> str:
    return "Tài liệu do bạn tải lên, hệ thống không xác minh hiệu lực."


def snapshot_notice(snapshot_as_of: str = "2024-09-10") -> str:
    return f"Dữ liệu kho luật cập nhật đến {snapshot_as_of}."
