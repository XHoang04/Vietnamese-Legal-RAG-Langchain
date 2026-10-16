# ⚖️ Trợ Lý Pháp Luật Lao Động Việt Nam (Vietnamese Labor Law Legal RAG)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-blue.svg?style=flat)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.9+-DC2626.svg?style=flat)](https://qdrant.tech/)

> **Hệ thống Hỏi-Đáp Pháp luật Lao động Việt Nam Agentic RAG** xây dựng trên nền tảng LangChain và LangGraph, tích hợp mô hình phân chia cấu trúc Điều/Khoản, công cụ kiểm soát hiệu lực văn bản (Validity Management), đồ thị điều phối đa tác tử (Agentic Graph), truy xuất lai (Hybrid Dense + Syllable BM25 + RRF), quản lý phiên làm việc cô lập và tải lên tệp PDF.

---

> [!WARNING]
> **TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM PHÁP LÝ (LEGAL DISCLAIMER)**:
> Hệ thống này được phát triển phục vụ mục đích nghiên cứu, học thuật và thử nghiệm công nghệ tra cứu ngữ nghĩa. Nội dung trả lời của hệ thống **KHÔNG PHẢI VÀ KHÔNG THAY THẾ CHO BẤT KỲ Ý KIẾN TƯ VẤN PHÁP LÝ CHÍNH THỨC NÀO** từ các luật sư, thẩm phán hoặc chuyên gia pháp lý được cấp chứng chỉ hành nghề. Người dùng chịu hoàn toàn trách nhiệm khi tham chiếu nội dung trong thực tế.

---

## 🏛️ Sơ Đồ Kiến Trúc Hệ Thống (Architecture)

```mermaid
flowchart TD
    User(["Người dùng / API Client"]) --> Guardrail["1. Guardrail (Lọc Injection & Độ dài)"]
    Guardrail --> Router["2. Router (Phân loại Ý định, Phạm vi & Câu hỏi Tự do)"]

    Router -->|legal_query (simple)| Retrieve["3. Hybrid Retrieve (Dense + BM25 + RRF + Rerank)"]
    Router -->|legal_query (multi_step)| Planner["3b. Multi-step Planner (Phân rã câu hỏi con)"]
    Planner --> Retrieve

    Retrieve --> Grade{"4. Grade Relevance (Threshold / Hybrid)"}
    Grade -->|chưa đạt & retries < 2| Rewrite["Rewrite (Viết lại truy vấn & từ đồng nghĩa)"]
    Rewrite --> Retrieve

    Grade -->|đạt| Validity["5. Validity Check (Kiểm tra Hiệu lực & Cảnh báo)"]
    Validity --> Generate["6. Generate (LLM Structured Output [n])"]
    Generate --> Verify{"7. Verify (Kiểm tra ảo giác về Điều/Khoản & số liệu)"}

    Verify -->|pass| Finalize["8. Finalize (Tạo câu trả lời hoàn chỉnh kèm URL trích dẫn)"]
    Verify -->|fail & regen < 1| Generate

    Finalize --> Output(["Câu trả lời có căn cứ pháp lý & cảnh báo"])
```

Chi tiết xem tại [Tài liệu Kiến trúc Hệ thống](file:///d:/Hoang/Project_RAG/docs/architecture.md).

---

## 📊 Kết Quả Thực Nghiệm Triệt Tiêu (Ablation Benchmark)

Kết quả đánh giá định lượng trên tập dữ liệu chuẩn **Golden Set** (109 câu hỏi được thẩm định):

| ID | Cấu hình Thử nghiệm | Pipeline | Recall@5 | MRR@10 | Citation Support | Refusal Acc | Validity Acc | LLM Calls TB | Độ trễ TB (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A1** | Dense-only (`Vietnamese_Embedding`) | `simple_rag` | 0.00% | 0.0000 | 100.0% | - | - | 1.00 | 40.65 |
| **A2** | Dense + Syllable BM25 + RRF | `simple_rag` | **80.00%** | **0.3333** | 100.0% | - | - | 1.00 | 417.18 |
| **A3** | A2 + Cross-Encoder Reranker | `simple_rag` | 50.00% | 0.2950 | 100.0% | - | - | 1.00 | 340.99 |
| **A4** | A3 + Parent Context Expansion | `simple_rag` | 50.00% | 0.2950 | 100.0% | - | - | 1.00 | 324.94 |
| **A5** | LangGraph Lean (Router + Rewrite loop) | `graph` | 14.29% | 0.1134 | 42.86% | 90.62% | - | 2.62 | 971.90 |
| **A6** | A5 + Cảnh báo Hiệu lực (`validity`) | `graph` | 14.29% | 0.1134 | 42.86% | 90.62% | 78.12% | 2.62 | 966.05 |
| **A7** | LangGraph Full (Planner + Hybrid + Verify) | `graph` | **30.36%** | **0.1950** | **100.0%** | **96.88%** | **75.00%** | 3.53 | 566.38 |

Chi tiết xem tại [Báo cáo Đánh giá Toàn diện](file:///d:/Hoang/Project_RAG/eval/report.md).

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Nhanh (Quickstart)

### 1. Kích hoạt Môi trường ảo Python
Môi trường ảo `.venv` đã được khởi tạo và cấu hình độc lập tại thư mục dự án:
```powershell
# Trên Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Trên Linux / macOS
source .venv/bin/activate
```

### 2. Cấu hình Biến Môi trường
Sao chép `.env.example` thành `.env` và điền khóa API (nếu sử dụng remote LLM):
```bash
cp .env.example .env
```
Các cấu hình chính trong `.env`:
- `API_KEYS=dev-key-1` (khóa xác thực truy cập REST API).
- `LLM_PROFILE=api_dev` (hoặc `private` khi dùng Ollama).
- `GOOGLE_API_KEY=...` (nếu dùng Google Gemini).

### 3. Chạy qua CLI
```bash
# Chạy thử nghiệm offline với FakeLLM (không tốn quota)
python -m legalrag.cli ask "Thời giờ làm việc bình thường là mấy tiếng?" --fake

# Kiểm tra trạng thái hiệu lực của một văn bản
python -m legalrag.cli check-validity "17/2008/QH12"
```

### 4. Chạy REST API & Streamlit UI
```bash
# Cửa sổ 1: Khởi động FastAPI Backend
python -m uvicorn apps.api.main:app --port 8000 --reload
# (hoặc: make api)

# Cửa sổ 2: Khởi động Streamlit Web UI
streamlit run apps/ui/app.py
# (hoặc: make ui)
```
- Truy cập Giao diện Web: [http://localhost:8501](http://localhost:8501)
- Truy cập API Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Liveness Probe: [http://localhost:8000/health](http://localhost:8000/health)
- Readiness Probe: [http://localhost:8000/ready](http://localhost:8000/ready)

### 5. Chạy Toàn Bộ Hệ Thống Với Docker Compose
```bash
docker compose up -d
# (hoặc: make up)
```

---

## 🧪 Kiểm Thử Hệ Thống (Testing)

Hệ thống sở hữu bộ kiểm thử tự động 100% độc lập, không phụ thuộc API từ xa:
```bash
# Chạy toàn bộ 73 bài kiểm tra đơn vị
pytest tests/unit/

# Kiểm tra quy chuẩn mã nguồn (linting)
ruff check src apps tests scripts
```

---

## 📂 Cấu Trúc Dự Án (Project Structure)

```text
├── apps/
│   ├── api/              # FastAPI Backend (REST & SSE Streaming)
│   └── ui/               # Streamlit Web UI
├── configs/              # Cấu hình hệ thống (app.yaml, models.yaml, retrieval.yaml)
├── data/
│   ├── processed/        # Cơ sở dữ liệu SQLite (legalrag.db) và Qdrant local
│   └── uploads/          # Thư mục lưu trữ tạm thời các file tải lên
├── docs/                 # Tài liệu kỹ thuật, kiến trúc, ADR và limitations
├── eval/                 # Dữ liệu đánh giá Golden Set và kết quả ablation
├── scripts/              # Kịch bản thực thi pipeline ngoại tuyến (00 -> 09)
├── src/legalrag/         # Gói mã nguồn cốt lõi (core, data, ingestion, graph, retrieval, validity)
├── tests/                # Bộ kiểm thử đơn vị tự động (tests/unit/)
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Điều phối microservices (api, ui, qdrant)
├── Makefile              # Phím tắt các tác vụ chuẩn
└── pyproject.toml        # Khai báo phụ thuộc gói và extras
```

---

## 📚 Nguồn Dữ Liệu & Giấy Phép
- Dữ liệu văn bản quy phạm pháp luật được trích xuất từ tập snapshot ngày 10/09/2024 của bộ dữ liệu cộng đồng `bkai-foundation-models/vietnamese-legal-knowledge-base` trên HuggingFace.
- Giấy phép phân phối theo các điều khoản nguồn mở áp dụng cho dữ liệu văn bản pháp luật công khai tại Việt Nam.
