"""Custom error types for Legal RAG."""

from typing import Any


class LegalRAGError(Exception):
    """Base exception for all Legal RAG domain errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "message": self.message}
        if self.details is not None:
            result["details"] = self.details
        return result


class IndexModelMismatchError(LegalRAGError):
    def __init__(self, message: str = "Index embedding model does not match configured model") -> None:
        super().__init__(message, code="INDEX_MODEL_MISMATCH")


class UploadRequiresLocalLLMError(LegalRAGError):
    def __init__(self, message: str = "Processing uploaded documents requires a local LLM under current privacy settings") -> None:
        super().__init__(message, code="UPLOAD_REQUIRES_LOCAL_LLM")


class UnsupportedScanPDFError(LegalRAGError):
    def __init__(self, message: str = "PDF dạng ảnh chưa được hỗ trợ, hãy dùng PDF có thể chọn văn bản") -> None:
        super().__init__(message, code="UNSUPPORTED_SCAN_PDF")


class NotAPDFError(LegalRAGError):
    def __init__(self, message: str = "File is not a valid PDF") -> None:
        super().__init__(message, code="NOT_A_PDF")


class FileTooLargeError(LegalRAGError):
    def __init__(self, message: str = "File size exceeds allowed limit") -> None:
        super().__init__(message, code="FILE_TOO_LARGE")


class TooManyPagesError(LegalRAGError):
    def __init__(self, message: str = "PDF page count exceeds allowed limit") -> None:
        super().__init__(message, code="TOO_MANY_PAGES")


class TooManyChunksError(LegalRAGError):
    def __init__(self, message: str = "Document generated too many chunks") -> None:
        super().__init__(message, code="TOO_MANY_CHUNKS")


class RateLimitedError(LegalRAGError):
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, code="RATE_LIMITED")


class LLMUnavailableError(LegalRAGError):
    def __init__(self, message: str = "LLM service is currently unavailable") -> None:
        super().__init__(message, code="LLM_UNAVAILABLE")


class UnauthorizedError(LegalRAGError):
    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, code="UNAUTHORIZED")


class ThreadNotFoundError(LegalRAGError):
    def __init__(self, message: str = "Chat thread not found") -> None:
        super().__init__(message, code="THREAD_NOT_FOUND")


class ThresholdNotCalibratedError(LegalRAGError):
    def __init__(self, message: str = "tau_high and tau_low thresholds must be calibrated before running in threshold mode") -> None:
        super().__init__(message, code="THRESHOLD_NOT_CALIBRATED")


class StructuredOutputError(LegalRAGError):
    def __init__(self, message: str = "Failed to parse structured output from LLM") -> None:
        super().__init__(message, code="STRUCTURED_OUTPUT_ERROR")
