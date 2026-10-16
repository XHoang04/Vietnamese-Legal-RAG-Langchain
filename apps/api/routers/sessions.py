"""Sessions document listing endpoints."""

from fastapi import APIRouter, Depends

from apps.api.auth import verify_api_key
from apps.api.dependencies import get_session_manager, rate_limit_dependency
from apps.api.schemas import SessionDocumentItem, SessionDocumentsResponse
from legalrag.sessions.manager import SessionManager

router = APIRouter(
    prefix="/v1/sessions",
    tags=["Sessions"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit_dependency)],
)


@router.get(
    "/{session_id}/documents",
    response_model=SessionDocumentsResponse,
    summary="List documents for a session",
)
async def list_session_documents(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> SessionDocumentsResponse:
    """Retrieve all uploaded documents associated with the given session ID."""
    docs = manager.list_session_documents(session_id)
    items = [
        SessionDocumentItem(
            upload_id=d["upload_id"],
            filename=d["filename"],
            pages=d.get("pages"),
            chunks=d.get("chunks"),
            status=d["status"],
            created_at=d.get("created_at"),
            expires_at=d.get("expires_at"),
        )
        for d in docs
    ]
    return SessionDocumentsResponse(session_id=session_id, documents=items)
