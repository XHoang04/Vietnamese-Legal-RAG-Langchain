"""Documents upload and management endpoints."""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, UploadFile, status

from apps.api.auth import verify_api_key
from apps.api.dependencies import get_session_manager, rate_limit_dependency
from apps.api.schemas import DocumentStatusResponse, DocumentUploadResponse
from legalrag.sessions.manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit_dependency)],
)


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and process PDF document",
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
    manager: SessionManager = Depends(get_session_manager),
) -> DocumentUploadResponse:
    """
    Upload a PDF document.
    Validates magic bytes (%PDF-), initiates background parsing, chunking, and indexing.
    """
    session_id = x_session_id or f"sess_{uuid.uuid4().hex[:12]}"
    filename = file.filename or "uploaded.pdf"

    # Read file contents
    content = await file.read()

    # Pre-validation of magic bytes (fail fast)
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "NOT_A_PDF", "message": "File tải lên không phải là định dạng PDF hợp lệ (%PDF-)."},
        )

    # Check size immediately
    max_mb = getattr(manager.config.app.upload, "max_size_mb", 20)
    if len(content) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "FILE_TOO_LARGE", "message": f"Dung lượng file vượt quá giới hạn ({max_mb} MB)."},
        )

    # Create upload record with status 'processing'
    upload_id = manager.create_upload(session_id=session_id, filename=filename)

    def _process_in_bg():
        try:
            manager.process_upload(
                upload_id=upload_id,
                file_input=content,
                filename=filename,
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Background upload processing failed for {upload_id}: {e}")

    background_tasks.add_task(_process_in_bg)

    return DocumentUploadResponse(
        upload_id=upload_id,
        filename=filename,
        status="processing",
        message="Tài liệu đã được tiếp nhận và đang được xử lý trong nền.",
    )


@router.get(
    "/{upload_id}",
    response_model=DocumentStatusResponse,
    summary="Get upload status",
)
async def get_document_status(
    upload_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> DocumentStatusResponse:
    """Retrieve indexing status, page count, and chunk count of an uploaded document."""
    info = manager.get_upload_status(upload_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DOCUMENT_NOT_FOUND", "message": f"Không tìm thấy tài liệu upload_id: {upload_id}"},
        )

    return DocumentStatusResponse(
        upload_id=info["upload_id"],
        filename=info["filename"],
        pages=info.get("pages"),
        chunks=info.get("chunks"),
        status=info["status"],
        created_at=info.get("created_at"),
        expires_at=info.get("expires_at"),
        error=info.get("error"),
    )


@router.delete(
    "/{upload_id}",
    summary="Delete uploaded document",
)
async def delete_document(
    upload_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Immediately delete an uploaded document, its vectors, and parent chunks."""
    deleted = manager.delete_upload(upload_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DOCUMENT_NOT_FOUND", "message": f"Không tìm thấy tài liệu upload_id: {upload_id}"},
        )

    return {
        "status": "deleted",
        "upload_id": upload_id,
    }
