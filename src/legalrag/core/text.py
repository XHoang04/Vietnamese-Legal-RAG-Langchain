"""Text normalization and token counting utilities."""

import re
import unicodedata


def normalize_nfc(text: str) -> str:
    """Normalize text to Unicode NFC form."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize text:
    - Unicode NFC
    - Convert CRLF to LF
    - Collapse redundant inline spaces while preserving newlines
    """
    if not text:
        return ""
    text = normalize_nfc(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace multiple spaces/tabs within lines
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def normalize_doc_number(doc_number: str | None) -> str | None:
    """
    Normalize legal document number:
    Uppercase, remove whitespace, '-', '.', and '/'
    """
    if not doc_number:
        return None
    s = normalize_nfc(doc_number).upper()
    return re.sub(r"[\s\-\.\/]", "", s)


def count_tokens(text: str) -> int:
    """
    Count tokens for text.
    Fallback / fast method: splits by whitespace and punctuation.
    """
    if not text:
        return 0
    # Approximate tokens using word count + punctuation tokens for Vietnamese
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(tokens)
