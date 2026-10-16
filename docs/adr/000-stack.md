# ADR 000: Technology Stack and Version Pinning

- **Ngày quyết định**: 2026-10-16
- **Trạng thái**: Accepted

## Bối cảnh
Dự án Legal RAG đòi hỏi pipeline agentic retrieval và generation cho văn bản pháp luật tiếng Việt, kết hợp kiểm tra hiệu lực văn bản, xử lý câu hỏi nhiều bước và hỏi đáp trên tài liệu người dùng tải lên.

## Quyết định
- **Ngôn ngữ**: Python >= 3.11
- **Điều phối Agentic**: LangGraph (`StateGraph`), checkpoint qua Memory/SQLite/Postgres
- **LLM Abstraction**: LangChain (`langchain-core`, model providers: Google GenAI, Groq, OpenAI-compatible, Ollama)
- **Data Validation & Config**: Pydantic v2, `pydantic-settings`, PyYAML
- **Vector Storage**: Qdrant (hỗ trợ Dense vector + BM25 Sparse vector)
- **Cơ sở dữ liệu quan hệ**: SQLAlchemy Core (SQLite cho dev/test, PostgreSQL cho production)
- **Testing**: `pytest`, `pytest-asyncio` với unit test 100% không gọi mạng / GPU (dùng fake embedder, fake LLM)
- **Linter & Code Quality**: `ruff`
