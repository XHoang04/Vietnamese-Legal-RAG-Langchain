"""PDF document loader and validation using PyMuPDF (fitz)."""

import re
from pathlib import Path

import pymupdf as fitz
from pydantic import BaseModel, Field

from legalrag.core.config import UploadConfig
from legalrag.core.errors import (
    FileTooLargeError,
    NotAPDFError,
    TooManyPagesError,
    UnsupportedScanPDFError,
)
from legalrag.core.text import normalize_nfc


class PDFPage(BaseModel):
    """Extracted text of a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int = 0


class PDFDocument(BaseModel):
    """Parsed and validated PDF document structure."""
    title: str
    filename: str
    total_pages: int
    pages: list[PDFPage] = Field(default_factory=list)
    full_text: str


def clean_pdf_text(text: str) -> str:
    """Strip control characters, normalize whitespace, and normalize Unicode NFC."""
    # Replace non-breaking spaces with standard space
    cleaned = text.replace("\xa0", " ")
    # Remove control characters except newline and tab
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return normalize_nfc(cleaned).strip()


def load_pdf(
    file_input: bytes | str | Path,
    filename: str,
    upload_config: UploadConfig | None = None,
) -> PDFDocument:
    """
    Validate, load, and extract text from a PDF file.

    Validations:
    1. First 5 bytes must be b'%PDF-' (NOT_A_PDF).
    2. File size must not exceed upload.max_size_mb (FILE_TOO_LARGE).
    3. Page count must not exceed upload.max_pages (TOO_MANY_PAGES).
    4. Average characters per page must be >= upload.min_chars_per_page (UNSUPPORTED_SCAN_PDF).
    """
    max_size_mb = upload_config.max_size_mb if upload_config else 20
    max_pages = upload_config.max_pages if upload_config else 150
    min_chars_per_page = upload_config.min_chars_per_page if upload_config else 100

    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        with open(p, "rb") as f:
            data = f.read()
    else:
        data = file_input

    # 1. Check size
    max_bytes = max_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"Kích thước tệp ({len(data) / (1024*1024):.1f} MB) vượt quá giới hạn cho phép ({max_size_mb} MB)."
        )

    # 2. Check magic header (%PDF-)
    if len(data) < 5 or data[:5] != b"%PDF-":
        raise NotAPDFError("Tệp tải lên không phải là định dạng PDF hợp lệ (thiếu phần đầu %PDF-).")

    # 3. Open with PyMuPDF
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise NotAPDFError(f"Không thể mở tệp PDF: {e}")

    try:
        total_pages = len(doc)
        if total_pages > max_pages:
            raise TooManyPagesError(
                f"Số trang của tệp ({total_pages} trang) vượt quá giới hạn tối đa ({max_pages} trang)."
            )

        pages: list[PDFPage] = []
        total_chars = 0

        for i in range(total_pages):
            page = doc.load_page(i)
            raw_text = page.get_text()
            cleaned = clean_pdf_text(raw_text)
            c_count = len(cleaned)
            total_chars += c_count
            pages.append(PDFPage(page_number=i + 1, text=cleaned, char_count=c_count))

        # 4. Check for scanned/image-only PDF
        avg_chars = total_chars / max(1, total_pages)
        if avg_chars < min_chars_per_page:
            raise UnsupportedScanPDFError(
                "PDF dạng ảnh chưa được hỗ trợ, hãy dùng PDF có thể chọn văn bản."
            )

        title = Path(filename).stem
        # Assemble full text with page markers
        full_text_blocks = []
        for p_obj in pages:
            if p_obj.text:
                full_text_blocks.append(f"[Trang {p_obj.page_number}]\n{p_obj.text}")
        full_text = "\n\n".join(full_text_blocks)

        return PDFDocument(
            title=title,
            filename=filename,
            total_pages=total_pages,
            pages=pages,
            full_text=full_text,
        )
    finally:
        doc.close()
