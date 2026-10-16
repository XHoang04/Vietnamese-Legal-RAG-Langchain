"""API Request and Response Pydantic Schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload for POST /v1/chat."""
    message: str = Field(..., min_length=1, max_length=2000, description="Nội dung câu hỏi của người dùng")
    thread_id: str | None = Field(default=None, description="ID luồng hội thoại để duy trì ngữ cảnh")
    session_id: str | None = Field(default=None, description="ID phiên làm việc để truy cập tài liệu tải lên")
    document_ids: list[str] | None = Field(default=None, description="Danh sách upload_id cụ thể cần truy vấn")


class ResumeRequest(BaseModel):
    """Payload for POST /v1/chat/{thread_id}/resume."""
    answer: str = Field(..., min_length=1, description="Câu trả lời làm rõ của người dùng")


class DocumentUploadResponse(BaseModel):
    """Response for POST /v1/documents."""
    upload_id: str
    filename: str
    status: Literal["processing", "ready", "failed"]
    message: str | None = None


class DocumentStatusResponse(BaseModel):
    """Response for GET /v1/documents/{upload_id}."""
    upload_id: str
    filename: str
    pages: int | None = None
    chunks: int | None = None
    status: Literal["processing", "ready", "failed"]
    created_at: str | None = None
    expires_at: str | None = None
    error: str | None = None


class SessionDocumentItem(BaseModel):
    """Item for GET /v1/sessions/{session_id}/documents."""
    upload_id: str
    filename: str
    pages: int | None = None
    chunks: int | None = None
    status: Literal["processing", "ready", "failed"]
    created_at: str | None = None
    expires_at: str | None = None


class SessionDocumentsResponse(BaseModel):
    """Response for GET /v1/sessions/{session_id}/documents."""
    session_id: str
    documents: list[SessionDocumentItem]


class ErrorDetail(BaseModel):
    """Structured error object."""
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    """Error envelope matching Section 15 specification."""
    error: ErrorDetail
