# Multi-stage Dockerfile for Vietnamese Labor Law Legal RAG

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY pyproject.toml .

# Install dependencies including api and ui extras
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.2.0" \
    "pyyaml>=6.0.1" \
    "orjson>=3.10.0" \
    "tenacity>=8.3.0" \
    "numpy>=1.26.0" \
    "langchain-core>=0.2.0" \
    "langgraph>=0.2.0" \
    "langgraph-checkpoint>=2.0.0" \
    "langgraph-checkpoint-sqlite>=2.0.0" \
    "fastapi>=0.111.0" \
    "uvicorn>=0.30.0" \
    "sse-starlette>=2.1.0" \
    "python-multipart>=0.0.9" \
    "httpx>=0.27.0" \
    "streamlit>=1.35.0" \
    "pymupdf>=1.24.0" \
    "sqlalchemy>=2.0.0" \
    "qdrant-client>=1.9.0"

# Copy project source code and configs
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY apps/ /app/apps/
COPY data/ /app/data/

# Install legalrag package in editable mode without reinstalling deps
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
