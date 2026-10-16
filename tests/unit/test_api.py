"""Unit tests for FastAPI REST and SSE endpoints."""

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from apps.api.dependencies import ServiceContainer, set_container
from apps.api.main import create_app
from legalrag.core.config import load_config
from legalrag.data.db import init_db
from legalrag.data.models import Citation
from legalrag.retrieval.store import init_qdrant_collection


@pytest.fixture
def api_client(monkeypatch):
    """Fixture providing a test client configured with isolated in-memory mocks."""
    monkeypatch.setenv("API_KEYS", "test-api-key,second-key")

    cfg = load_config()
    cfg.app.api.rate_limit_per_minute = 100
    collection_name = "test_api_collection"
    cfg.app.qdrant.collection = collection_name

    # In-memory DB with StaticPool so all connections share the same memory tables
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)

    # In-memory Qdrant
    qdrant = QdrantClient(location=":memory:")
    init_qdrant_collection(qdrant, collection_name=collection_name, dense_dim=64)

    # Build container with isolated in-memory dependencies
    container = ServiceContainer(config_bundle=cfg, qdrant_client=qdrant, db_engine=engine)
    container.rate_limiter.requests_per_minute = 100
    container.rate_limiter.history.clear()
    set_container(container)

    app = create_app()
    client = TestClient(app)

    yield client, container

    # Teardown
    set_container(None)


def test_health_check(api_client):
    client, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_check(api_client):
    client, _ = api_client
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["database"] == "ok"
    assert data["qdrant"] == "ok"


def test_auth_rejection(api_client):
    client, _ = api_client
    # Missing API Key
    resp = client.get("/v1/documents/fake-id")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    # Wrong API Key
    resp = client.get("/v1/documents/fake-id", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"


def test_rate_limiting(api_client):
    client, container = api_client
    container.config.app.api.rate_limit_per_minute = 3
    container.rate_limiter.requests_per_minute = 3
    container.rate_limiter.history.clear()

    headers = {"X-API-Key": "test-api-key"}

    # 3 allowed requests
    for _ in range(3):
        r = client.get("/v1/sessions/test_sess/documents", headers=headers)
        assert r.status_code == 200

    # 4th request must be rate limited (429)
    r_limited = client.get("/v1/sessions/test_sess/documents", headers=headers)
    assert r_limited.status_code == 429
    assert r_limited.json()["error"]["code"] == "RATE_LIMITED"


def test_upload_not_a_pdf(api_client):
    client, _ = api_client
    headers = {"X-API-Key": "test-api-key"}
    fake_file = io.BytesIO(b"This is plain text, definitely not PDF")

    resp = client.post(
        "/v1/documents",
        files={"file": ("test.txt", fake_file, "text/plain")},
        headers=headers,
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "NOT_A_PDF"


def test_upload_valid_pdf_and_query_status(api_client):
    client, container = api_client
    headers = {"X-API-Key": "test-api-key", "X-Session-Id": "sess_123"}

    # Valid minimal PDF header
    valid_pdf = io.BytesIO(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    resp = client.post(
        "/v1/documents",
        files={"file": ("hop_dong.pdf", valid_pdf, "application/pdf")},
        headers=headers,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "upload_id" in data
    assert data["status"] == "processing"

    upload_id = data["upload_id"]

    # Check status
    st_resp = client.get(f"/v1/documents/{upload_id}", headers=headers)
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["upload_id"] == upload_id
    assert st_data["filename"] == "hop_dong.pdf"

    # List session documents
    list_resp = client.get("/v1/sessions/sess_123/documents", headers=headers)
    assert list_resp.status_code == 200
    docs = list_resp.json()["documents"]
    assert len(docs) >= 1
    assert any(d["upload_id"] == upload_id for d in docs)

    # Delete upload
    del_resp = client.delete(f"/v1/documents/{upload_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"


def test_chat_sse_stream(api_client):
    client, container = api_client
    headers = {"X-API-Key": "test-api-key"}

    # Mock graph stream to yield controlled chunks
    mock_chunks = [
        {"router": {"intent": "legal_query", "scope": "corpus", "complexity": "simple"}},
        {"retrieve": {"contexts": []}},
        {"finalize": {
            "answer": "Quy định về thời giờ làm việc là 8 giờ một ngày theo Điều 105 [1].",
            "citations": [
                Citation(
                    n=1,
                    doc_id="45/2019/QH14",
                    title="Bộ luật Lao động 2019",
                    doc_number="45/2019/QH14",
                    article_no="105",
                    breadcrumb="Điều 105. Thời giờ làm việc bình thường",
                    url="https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-luat-Lao-dong-2019-333670.aspx",
                    status="IN_FORCE",
                    source="corpus",
                )
            ],
            "warnings": ["Dữ liệu cập nhật đến 2024-09-10"],
            "llm_calls": 2,
        }},
    ]

    container.graph.stream = MagicMock(return_value=iter(mock_chunks))
    mock_state = MagicMock()
    mock_state.tasks = []
    mock_state.values = mock_chunks[2]["finalize"]
    container.graph.get_state = MagicMock(return_value=mock_state)

    payload = {
        "message": "Thời giờ làm việc bình thường là mấy tiếng?",
        "thread_id": "thread_test_01",
    }

    resp = client.post("/v1/chat", json=payload, headers=headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    content = resp.text
    # Verify expected SSE events
    assert "event: step" in content
    assert "event: citations" in content
    assert "event: warnings" in content
    assert "event: token" in content
    assert "event: done" in content
    assert "Bộ luật Lao động 2019" in content
