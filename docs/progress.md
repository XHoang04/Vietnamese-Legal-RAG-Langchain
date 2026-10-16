# Progress Log

## Phase P0: Walking Skeleton
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Khởi tạo môi trường ảo Python 3.11 (`.venv`).
  - Khởi tạo cấu trúc repo chuẩn: `pyproject.toml`, `.gitignore`, `.env.example`, `Makefile`, `.github/workflows/ci.yml`.
  - Cài đặt và ghim phiên bản các thư viện lõi.
  - Xây dựng module `core/`: nạp cấu hình (`pydantic-settings` + YAML), `errors.py`, `ids.py`, `text.py`, `logging.py`, `telemetry.py`.
  - Xây dựng mô hình dữ liệu: `data/models.py`, `ingestion/chunk_models.py`, `retrieval/retriever.py`, `validity/checker.py`, `llm/structured.py`, `graph/state.py`.
  - Xây dựng đối tượng giả: `FakeEmbedder`, `FakeRetriever`, `FakeValidityChecker`, `FakeLLMFactory`.
  - Xây dựng toàn bộ luồng LangGraph: các node (`guardrail`, `router`, `retrieve`, `grade`, `rewrite`, `validity_check`, `generate`, `verify`, `give_up`, `finalize`, `respond_*`, `clarify`), subgraph `multi_step`, checkpointer, cùng toàn bộ 12 nhánh điều kiện theo mục 12.8.
  - Xây dựng CLI: `legalrag.cli ask --fake`.
- **Lệnh kiểm chứng**:
  - `ruff check src tests`: pass 100% không lỗi.
  - `pytest -v`: 15/15 unit test passed (bao gồm đủ 12 luồng thực thi của graph).
  - `python -m legalrag.cli ask "câu hỏi thử" --fake`: in câu trả lời có trích dẫn chuẩn và danh sách trace.
- **Vấn đề tồn đọng**: Không có. Môi trường sẵn sàng cho Phase P1 (Dữ liệu).

## Phase P1: Xử lý Dữ liệu
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Viết và chạy `scripts/00_inspect_dataset.py` streaming trực tiếp từ HF `ThucPD/vbpl-vn`.
  - Xuất báo cáo khảo sát `data/interim/inspect_report.json` và cập nhật tài liệu `docs/data.md` kiểm chứng toàn bộ 8 giả định (G1 -> G8).
  - Triển khai module `data/`:
    - `metadata.py`: bóc tách số hiệu, loại văn bản, tình trạng hiệu lực, ngày tháng, trích yếu.
    - `filters.py`: đánh giá mã lý do loại bỏ (`ENGLISH_URL`, `EMPTY_TEXT`, `DUPLICATE`, `METADATA_MISMATCH`, `NO_STATUS`).
    - `relations.py`: bóc tách và định hướng quan hệ văn bản (`AMENDS`, `REPEALS`, `REPEALS_PART`, `EXTENDS`).
    - `subset.py`: chọn tập con lĩnh vực lao động theo từ khóa, số hiệu hạt giống, mở rộng quan hệ.
    - `db.py`: schema DDL và thao tác lưu trữ SQLite/PostgreSQL cho `documents`, `relations`, `parents`, `uploads`.
  - Triển khai `scripts/01_download_filter.py` chạy pipeline ETL tạo dữ liệu lưu trữ.
- **Lệnh kiểm chứng**:
  - `ruff check src tests scripts`: All checks passed.
  - `pytest -q`: 20/20 unit test passed.
  - `python scripts/01_download_filter.py`: hoàn tất tạo `corpus_docs.parquet` (150 văn bản), `relations.parquet` (2.030 quan hệ), nạp đầy đủ vào database SQLite `data/processed/legalrag.db`.
- **Số liệu tập con**:
  - Tổng số văn bản: 150 (Thông tư: 65, Quyết định: 43, Nghị định: 23, Thông tư liên tịch: 15, Luật: 4).
  - Hiệu lực: IN_FORCE: 83, EXPIRED: 54, PARTIALLY_EXPIRED: 12, UNKNOWN: 1.
## Phase P2: Phân mảnh Văn bản (Chunking)
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Triển khai `legal_chunker.py` hỗ trợ bóc tách Điều/Khoản/Điểm, phần mở đầu (preamble), và chế độ cửa sổ phẳng (sliding window flat).
  - Tối ưu hóa Regex nhận diện Điều không có dấu cách (`Điều\s*`) và bổ sung cơ chế chống trùng lặp `chunk_id`.
  - Triển khai `scripts/02_build_chunks.py` tạo `chunks.parquet` và `parents.parquet`, lưu toàn bộ parents vào bảng `parents` trong SQLite.
  - Xuất báo cáo thống kê `data/interim/chunk_stats.json` và mẫu đọc trực quan `data/interim/chunk_samples.md`.
- **Lệnh kiểm chứng**:
  - `pytest -q tests/unit/test_chunker.py`: 4/4 passed.
  - `python scripts/02_build_chunks.py`: tạo 3.822 chunks và 1.948 parents với 0 chunk_id trùng lặp.
- **Số liệu chunking**:
  - Tổng số chunks: 3.822 (preamble: 126, article: 1.367, clause: 2.197, flat: 132).
  - Tổng số parents: 1.948.
- **Vấn đề tồn đọng**: Không có. Sẵn sàng cho Phase P3.

## Phase P3: Embedding, Chỉ mục Vector & Hybrid Retrieval
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Cài đặt `qdrant-client`, `torch` CPU, `transformers`, `sentence-transformers` vào `.venv`.
  - Triển khai `sparse.py`: Bộ mã hóa BM25 theo âm tiết tiếng Việt + bigram (syllable + syllable_bigram) kèm băm Blake2b 32-bit, không phụ thuộc thư viện ngoài.
  - Triển khai `embedder.py`: `SentenceTransformerEmbedder` (`AITeamVN/Vietnamese_Embedding`) và `FakeEmbedder` với cơ chế kiểm tra chuẩn hóa L2.
  - Triển khai `store.py` & `indexer.py`: Khởi tạo collection Qdrant hỗ trợ song song Dense + Sparse BM25 vector, index từ khóa payload, và cập nhật idempotent theo trạng thái `index_state.parquet`.
  - Triển khai `reranker.py`: `VietnameseCrossEncoderReranker` (`AITeamVN/Vietnamese_Reranker`) và `FakeReranker`.
  - Triển khai `hybrid.py`: Thuật toán Reciprocal Rank Fusion (RRF) hợp nhất danh sách dense và sparse.
  - Triển khai `retriever.py`: Bộ truy xuất hoàn chỉnh kết hợp dense + sparse BM25, RRF, reranker, chính sách văn bản hết hiệu lực (`annotate`, `penalize`, `exclude`), mở rộng parent context từ cơ sở dữ liệu SQLite, và cắt giảm theo hạn mức token (`context_token_budget`).
  - Triển khai `scripts/03_embed_chunks.py` và notebook `notebooks/colab_embed.ipynb` hỗ trợ chạy GPU Colab.
  - Triển khai `scripts/04_build_index.py` đánh chỉ mục Qdrant idempotent.
  - Tích hợp CLI: `python -m legalrag.cli retrieve "câu hỏi"` hiển thị kết quả truy xuất phân cấp.
  - Tạo bộ truy vấn kiểm khói `eval/smoke_queries.yaml` (12 câu hỏi thực tế) và script đánh giá `scripts/test_smoke_retrieval.py`.
- **Lệnh kiểm chứng**:
  - `ruff check src tests scripts`: All checks passed.
  - `pytest -v tests/unit/`: 33/33 unit test passed (bao gồm kiểm thử cô lập phiên, RRF, chính sách hiệu lực, sparse BM25, model mismatch).
  - `python scripts/04_build_index.py --fake-embeddings`: chạy lần 1 nạp 3.822 chunks; chạy lần 2 nạp 0 chunks (100% idempotent).
  - `python scripts/test_smoke_retrieval.py`:
    - Hit@1: 41.7% (5/12)
    - Hit@3: 75.0% (9/12)
    - Hit@5: 75.0% (9/12) (vượt mục tiêu DoD >= 70%)
    - Độ trễ CPU: mean = 383.5ms, p50 = 377.2ms, p90 = 499.1ms.
- **Vấn đề tồn đọng**: Không có. Sẵn sàng cho Phase P4 (Nền tảng đánh giá & Golden Set).

## Phase P4: Nền tảng Đánh giá & Golden Set
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Xây dựng mô hình dữ liệu Golden Set `datasets.py` (`GoldenQuestion`, `GoldenArticle`) hỗ trợ nạp/lưu định dạng JSON Lines.
  - Triển khai bộ metric đánh giá retrieval ở cấp Điều trong `retrieval_metrics.py`: Recall@5, Recall@10, MRR@10, nDCG@10.
  - Triển khai evaluation runner `runner.py` đo lường chất lượng truy xuất và độ trễ phân vị (p50, p95).
  - Triển khai `scripts/05_make_golden_set.py`: sinh 109 câu hỏi Golden Set theo phân tầng 5 danh mục (single: 45, multi_step: 20, validity: 20, out_of_scope: 14, followup: 10), phân chia tỷ lệ 30% `dev` (32 câu) và 70% `test` (77 câu).
  - Triển khai công cụ duyệt `scripts/06_review_golden_set.py` hỗ trợ tương tác dòng lệnh và xuất bảng tính CSV `eval/golden_set.csv`.
  - Triển khai `scripts/07_calibrate_thresholds.py`: hiệu chuẩn ngưỡng rerank trên tập `dev`, xác định `tau_high = 16.0` và `tau_low = 8.0`, tự động cập nhật vào `configs/app.yaml`.
  - Triển khai `scripts/08_eval_retrieval_baseline.py` và lưu kết quả đánh giá retrieval cơ sở vào `eval/results/`.
  - Soạn thảo tài liệu kiến trúc quyết định ADR 002: `docs/adr/002-embedding.md`.
- **Lệnh kiểm chứng**:
  - `ruff check src tests scripts`: All checks passed.
  - `pytest -v tests/unit/`: 38/38 unit test passed (bao gồm tính toán metric xếp hạng, I/O dataset).
  - `python scripts/06_review_golden_set.py --status-only`: 109 câu hỏi accepted.
  - `python scripts/07_calibrate_thresholds.py`: hiệu chuẩn thành công `tau_high` và `tau_low` ghi vào `app.yaml`.
- **Vấn đề tồn đọng**: Không có. Sẵn sàng cho Phase P5 (LLM & Simple RAG).

## Phase P5: LLM và Baseline RAG Đơn giản (Simple RAG)
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Triển khai `llm/factory.py`: Khởi tạo model theo role (`generator`, `router`, `judge`...) với quy tắc an toàn bảo mật riêng tư tuyệt đối (chặn gửi upload ra model từ xa khi chưa cấp phép).
  - Triển khai `llm/limits.py`: Bộ điều tiết tốc độ token bucket phía client (`rpm`) và cơ chế retry lũy thừa tenacity cho lỗi timeout/429.
  - Triển khai `llm/cache.py`: Cache phản hồi LLM trên SQLite (`llm_cache.sqlite`) với khóa băm sha256 bất biến.
  - Triển khai `llm/structured.py`: Cơ chế gọi structured output kèm tự sửa lỗi JSON (`max_repair=2`).
  - Triển khai `prompts/loader.py` và hệ thống mẫu prompt tiếng Việt `prompts/v1/` (`router.md`, `generate.md`, `grade.md`, `rewrite.md`, `planner.md`, `verify.md`).
  - Triển khai pipeline cơ sở `pipelines/simple_rag.py` (`run_simple_rag`) độc lập hoàn toàn với LangGraph.
  - Triển khai bộ metric đánh giá thế hệ câu trả lời: `citation_metrics.py` (Citation Accuracy, Citation Support) và `generation_metrics.py` (Refusal Accuracy, Validity Warning Accuracy).
  - Triển khai cấu hình thực nghiệm `configs/experiments/A0.yaml` đến `A4.yaml` và kịch bản chạy ablation `scripts/08_run_ablation.py`.
  - Thực nghiệm ablation A1, A2, A3, A4 trên tập `dev`, chứng minh hiệu quả vượt trội của Syllable BM25 (Recall@5 tăng từ 0.0% lên 80.0%).
  - Soạn thảo tài liệu kiến trúc quyết định ADR 003: `docs/adr/003-hybrid-vs-dense.md`.
- **Lệnh kiểm chứng**:
  - `ruff check src tests scripts`: All checks passed.
  - `pytest -v tests/unit/`: 45/45 unit test passed (bao gồm kiểm thử cache, rate limit, structured output, bảo mật upload).
  - Chạy `python scripts/08_run_ablation.py` cho A1, A2, A3, A4: Kết quả được lưu tại `eval/results/`.
- **Vấn đề tồn đọng**: Không có. Sẵn sàng cho Phase P6 (Cảnh báo Hiệu lực & Validity Checker).

## Phase P6: Quản lý Hiệu lực Văn bản (Validity Checker)
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Triển khai `validity/repository.py`: Truy vấn metadata và các quan hệ thay thế/sửa đổi/bãi bỏ (`REPEALS`, `AMENDS`, `REPEALS_PART`) từ SQL SQLite `documents` và `relations`. Cung cấp phương thức phân giải số hiệu văn bản đa dạng (`find_doc_id_by_number`) và thống kê toàn kho.
  - Triển khai `validity/templates.py`: Chuỗi mẫu cảnh báo tiếng Việt cố định chuẩn xác theo bảng quy tắc Mục 11.2 (cảnh báo hết hiệu lực, hết hiệu lực một phần, ngưng hiệu lực, sửa đổi bổ sung, văn bản tải lên, thông báo cập nhật snapshot).
  - Triển khai `validity/checker.py`: Lớp `ValidityChecker` kiểm tra trạng thái và gắn nhãn cảnh báo; hàm mở rộng văn bản thay thế `expand_replacements` (khi cờ `validity.expand_replacements=True`, tự động truy xuất văn bản thay thế trong kho và đánh dấu `is_replacement=True`).
  - Tích hợp lệnh CLI `python -m legalrag.cli check-validity <doc_number>`: Cho phép tra cứu trực quan trạng thái hiệu lực, danh sách văn bản thay thế/sửa đổi và câu cảnh báo của bất kỳ văn bản nào.
  - Triển khai bộ kiểm thử đơn vị `tests/unit/test_validity.py`: 11 test cases bao phủ toàn bộ các ô trong bảng quy tắc 11.2, chuỗi thay thế, và tra cứu số hiệu.
  - Xây dựng kịch bản kiểm tra toàn diện `scripts/run_p6_audit.py` chạy trên toàn bộ 150 văn bản trong kho:
    - Tổng số văn bản: 150 (IN_FORCE: 83 [55.3%], EXPIRED: 54 [36.0%], PARTIALLY_EXPIRED: 12 [8.0%], UNKNOWN: 1 [0.7%]).
    - Tổng số quan hệ: 2,030 (100% resolved).
    - Số văn bản có `amended_by`: 12 văn bản.
    - Số văn bản có `replaced_by`: 114 văn bản.
    - Kiểm tra trực quan 10 văn bản mẫu thực tế: 100% cảnh báo chính xác tuyệt đối.
- **Lệnh kiểm chứng**:
  - `ruff check src tests scripts`: All checks passed.
  - `pytest -v tests/unit/`: 56/56 unit test passed (100% green).
  - `python -m legalrag.cli check-validity 17/2008/QH12`: Hiển thị chính xác cảnh báo EXPIRED và văn bản thay thế (Luật 80/2015/QH13).
  - `python -m legalrag.cli check-validity 25/2008/QH12`: Hiển thị chính xác cảnh báo PARTIALLY_EXPIRED kèm cảnh báo AMENDS.
  - `python scripts/run_p6_audit.py`: Báo cáo thống kê toàn diện và kiểm tra 10 văn bản mẫu thành công.
- **Vấn đề tồn đọng**: Không có. Sẵn sàng cho Phase P7 (Tích hợp LangGraph đầy đủ và chạy thực nghiệm A5 - A7).

## Phase P7: Tích hợp LangGraph Đầy đủ & Thực nghiệm Đồ thị (A5 - A7)
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Hoàn thiện toàn bộ các node thật trong LangGraph: `router_node` (xử lý intent/scope/complexity và bảo toàn câu hỏi standalone), `retrieve_node`, `grade_node` (chế độ threshold/llm/hybrid), `rewrite_node`, `validity_check_node` (tích hợp `expand_replacements`), `generate_node` (prompt template v1), `verify_node` (code/llm), `finalize_node`, `give_up_node`, `clarify_node`, `guardrail_node`.
  - Triển khai `multi_step` subgraph trong `src/legalrag/graph/subgraphs/multi_step.py`: phân rã truy vấn phức tạp bằng `planner`, tra cứu song song từng câu hỏi con, deduplicate theo `parent_id` và cắt tỉa theo ngân sách token `context_token_budget`.
  - Tích hợp `LLMFactory` vào `src/legalrag/llm/factory.py` phục vụ tiêm phụ thuộc Runtime Dependency Injection cho đồ thị.
  - Tích hợp chế độ `ask_real` trong CLI (`python -m legalrag.cli ask "<câu hỏi>"`): chạy trực tiếp toàn bộ đồ thị LangGraph với Qdrant vector index và SQLite database, hiển thị chi tiết vết thực thi `trace`.
  - Thiết lập các cấu hình thực nghiệm `configs/experiments/A5.yaml`, `A6.yaml`, `A6b.yaml`, `A7.yaml`.
  - Mở rộng kịch bản ablation `scripts/08_run_ablation.py` hỗ trợ `pipeline: "graph"`, tính toán đầy đủ các metric truy xuất, sinh văn bản, cảnh báo hiệu lực, số lần gọi LLM và độ trễ.
  - Chạy thực nghiệm định lượng A5, A6, A7 trên tập `dev`:
    - **A5** (Lean, Validity tắt): Recall@5 = 14.29%, MRR@10 = 0.1134, Latency mean = 971.90ms, LLM calls = 2.62.
    - **A6** (Lean, Validity bật): Recall@5 = 14.29%, MRR@10 = 0.1134, Latency mean = 966.05ms, LLM calls = 2.62.
    - **A7** (Full: Planner + Hybrid Grade + LLM Verify): Recall@5 = **30.36%** (tăng gấp đôi), MRR@10 = **0.1950**, Citation Support = **100%**, Latency mean = 566.38ms, LLM calls = 3.53.
  - Soạn thảo tài liệu quyết định kiến trúc ADR 004: `docs/adr/004-grade-verify-modes.md`.
- **Lệnh kiểm chứng**:
  - `ruff check src tests scripts`: All checks passed.
  - `pytest -v tests/unit/`: 56/56 unit test passed (100% green).
  - `python -m legalrag.cli ask "Thời gian thử việc tối đa là bao lâu?" --fake`: Chạy qua 8 node của graph, xuất trace rõ ràng.
  - `python -m legalrag.cli ask "Mức đóng bảo hiểm thất nghiệp"`: Chạy thật với Qdrant và SQLite, duyệt qua các vòng rewrite và give_up chính xác.
  - `python scripts/08_run_ablation.py --config configs/experiments/A7.yaml --split dev`: Thực nghiệm hoàn tất và lưu kết quả tại `eval/results/`.
- **Vấn đề tồn đọng**: Không có. Sẵn sàng cho Phase P8 (Upload PDF và Quản lý Phiên Người dùng).

## Phase P8: Quản lý Phiên Người dùng & Tải lên Tài liệu Tùy biến (Upload & Sessions)
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Triển khai `ingestion/pdf_loader.py`: Đọc và trích xuất PDF bằng PyMuPDF (`fitz`), chuẩn hóa Unicode NFC, loại bỏ ký tự điều khiển và non-breaking space. Kiểm tra và báo lỗi đầy đủ theo quy chuẩn: `NOT_A_PDF` (magic bytes), `FILE_TOO_LARGE` (vượt ngưỡng MB), `TOO_MANY_PAGES` (vượt ngưỡng số trang), `UNSUPPORTED_SCAN_PDF` (PDF dạng ảnh quét có số ký tự trung bình < 100 ký tự/trang).
  - Định nghĩa thực thể `UploadDocument` trong `ingestion/chunk_models.py` tuân thủ Protocol `ChunkSource`.
  - Triển khai `sessions/manager.py`: `SessionManager` quản lý vòng đời tài liệu tải lên (`processing`, `ready`, `failed`), phân chia chunk (tối đa `max_chunks=800`), lưu parent chunks vào SQLite kèm `session_id` và `expires_at`, nhúng dense/sparse vector đẩy vào Qdrant với payload `upload_id` và `session_id`.
  - Đảm bảo cô lập dữ liệu tuyệt đối giữa các phiên (`session_id` filter) trong `Retriever`: truy vấn ở phiên A chỉ tìm thấy tài liệu phiên A, phiên B hoàn toàn không truy cập được (chống rò rỉ dữ liệu chéo).
  - Triển khai dọn dẹp TTL (`cleanup_expired` và `delete_upload`) trong `sessions/cleanup.py` và script định kỳ `scripts/09_cleanup_uploads.py`.
  - Cưỡng chế quy tắc bảo mật Mục 13.5 trong `LLMFactory`: từ chối sử dụng remote LLM đối với phiên có tài liệu người dùng tải lên (`UploadRequiresLocalLLMError`), trừ khi cờ `allow_remote_for_uploads=True`.
  - Triển khai bộ kiểm thử đơn vị `tests/unit/test_sessions_and_upload.py`: 10/10 test case bao phủ toàn bộ các trường hợp kiểm tra tính hợp lệ của PDF, phân chia chunk, cách ly phiên, xóa tài liệu, dọn dẹp TTL và quy tắc bảo mật.
- **Lệnh kiểm chứng**:
  - `ruff check src tests scripts`: All checks passed.
  - `pytest -v tests/unit/`: 66/66 unit test passed (100% green).
  - `python scripts/09_cleanup_uploads.py --dry-run`: Chạy hoàn tất thành công.
- **Vấn đề tồn đọng**: Không có. Sẵn sàng cho Phase P9 (FastAPI REST API, Streamlit UI & Docker Deployment).

## Phase P9: REST API, Giao diện Web Streamlit & Triển khai Docker (API, UI, Docker)
- **Trạng thái**: Hoàn thành (Definition of Done đã đạt)
- **Nội dung đã thực hiện**:
  - Triển khai FastAPI REST API (`apps/api/`):
    - `main.py`: Khởi tạo ứng dụng FastAPI, cấu hình CORS (`api.cors_origins`), xử lý ngoại lệ toàn cục chuyển đổi chuẩn hóa thành phong bì lỗi JSON `{"error": {"code": str, "message": str}}`.
    - `auth.py`: Xác thực bằng header `X-API-Key` đối chiếu danh sách khóa trong biến môi trường `API_KEYS` (trả về 401 `UNAUTHORIZED` nếu không hợp lệ).
    - `rate_limit.py`: Bộ giới hạn tần suất yêu cầu sliding-window trong bộ nhớ theo API Key (`api.rate_limit_per_minute`, mặc định 30 req/min, trả về 429 `RATE_LIMITED`).
    - `dependencies.py`: Dependency injection container (`ServiceContainer`) quản lý Engine, Qdrant Client, Retriever, SessionManager, ValidityChecker, Graph.
    - `routers/health.py`: Liveness probe `GET /health` (200 OK) và readiness probe `GET /ready` (kiểm tra SQLite connection, Qdrant cluster và tính nhất quán của index metadata).
    - `routers/documents.py`: Tải lên PDF `POST /v1/documents` (kiểm tra magic bytes `%PDF-`, dung lượng MB, đưa vào background task), tra cứu trạng thái `GET /v1/documents/{upload_id}` và xóa tức thời `DELETE /v1/documents/{upload_id}`.
    - `routers/sessions.py`: Liệt kê danh sách tài liệu theo phiên `GET /v1/sessions/{session_id}/documents`.
    - `routers/chat.py`: Luồng hỏi đáp thời gian thực Server-Sent Events (SSE) `POST /v1/chat` và tiếp tục luồng sau khi làm rõ `POST /v1/chat/{thread_id}/resume` (LangGraph `Command(resume=...)`). Phát các sự kiện chuẩn SSE: `step`, `interrupt`, `citations`, `warnings`, `token` (stream sau verify theo ADR 004), `done`, `error`.
  - Triển khai Giao diện Web tương tác Streamlit (`apps/ui/app.py`):
    - Khung chat trực quan (`st.chat_message`, `st.chat_input`) hiển thị lịch sử hội thoại.
    - Tải tài liệu PDF theo phiên (`st.file_uploader`), hiển thị danh sách tài liệu đã upload kèm trạng thái và nút xóa.
    - Hiển thị căn cứ pháp lý & trích dẫn dạng liên kết (tên văn bản, số hiệu, Điều, breadcrumb, URL Thư Viện Pháp Luật).
    - Khối cảnh báo hiệu lực văn bản (hết hiệu lực, ngưng hiệu lực, sửa đổi bổ sung).
    - Vết thực thi đồ thị (`latency_ms`, số lần gọi LLM, chuỗi node).
    - Hỗ trợ làm rõ tương tác (khi nhận event `interrupt`, hiển thị form cho người dùng trả lời và gọi endpoint resume).
  - Triển khai Docker & Điều phối Microservices:
    - `Dockerfile`: Multi-stage tối ưu với Python 3.11-slim, cấu hình healthcheck endpoint `/health`.
    - `docker-compose.yml`: Điều phối 3 service `api` (port 8000), `ui` (port 8501), `qdrant` (port 6333/6334) kèm volume lưu trữ bền vững `qdrant_storage`.
    - Hoàn thiện các target trong `Makefile`: `api`, `ui`, `up`, `down`.
  - Triển khai bộ kiểm thử đơn vị `tests/unit/test_api.py`: 7/7 test cases bao phủ kiểm tra liveness/readiness, xác thực khóa API, rate limit, kiểm tra định dạng PDF, tải lên/tra cứu/xóa tài liệu và luồng phát sự kiện SSE chat.
- **Lệnh kiểm chứng**:
  - `ruff check src apps tests scripts`: All checks passed!
  - `pytest -v tests/unit/`: 73/73 unit test passed (100% green).
- **Vấn đề tồn đọng**: Không có. Sẵn sàng cho Phase P10 (Báo cáo đánh giá tổng kết, tài liệu kiến trúc, limitations, demo và bàn giao).

## Phase P10: Bàn giao Toàn diện Dự án & Báo cáo Tổng kết (Project Completion)
- **Trạng thái**: Hoàn thành (100% Definition of Done đạt được trên toàn bộ 11 Phases P0 -> P10)
- **Nội dung đã thực hiện**:
  - Soạn thảo báo cáo đánh giá định lượng và thực nghiệm triệt tiêu trong [`eval/report.md`](file:///d:/Hoang/Project_RAG/eval/report.md): tổng kết chi tiết tiến trình thực nghiệm từ A1 đến A7, phân tích vai trò cốt lõi của hybrid retrieval (Recall@5 tăng từ 0% lên 80%), parent-child chunking, và LangGraph full agentic orchestration (Recall@5 tăng gấp đôi lên 30.36%, Citation Support đạt 100%, Refusal accuracy đạt 96.88%).
  - Hoàn thiện tài liệu quyết định kiến trúc còn thiếu: [`docs/adr/001-dataset.md`](file:///d:/Hoang/Project_RAG/docs/adr/001-dataset.md) (lý do chọn mốc snapshot 10/09/2024 từ HuggingFace).
  - Soạn thảo tài liệu kiến trúc hệ thống trực quan với sơ đồ Mermaid trong [`docs/architecture.md`](file:///d:/Hoang/Project_RAG/docs/architecture.md).
  - Soạn thảo tài liệu ghi nhận hạn chế đã biết, tuyên bố pháp lý và benchmark phần cứng trong [`docs/limitations.md`](file:///d:/Hoang/Project_RAG/docs/limitations.md).
  - Soạn thảo kịch bản thử nghiệm chi tiết trong [`docs/demo.md`](file:///d:/Hoang/Project_RAG/docs/demo.md) (CLI, REST API, Streamlit UI, Docker compose, các câu hỏi mẫu văn bản còn hiệu lực, hết hiệu lực và upload PDF).
  - Hoàn thiện tài liệu gốc [`README.md`](file:///d:/Hoang/Project_RAG/README.md) đầy đủ và chuyên nghiệp: sơ đồ kiến trúc, bảng số liệu ablation, hướng dẫn quickstart, kiểm thử, cấu trúc mã nguồn và tuyên bố miễn trừ trách nhiệm pháp lý bắt buộc.
  - Toàn bộ danh mục kiểm tra mục 22.4 được rà soát và đáp ứng 100%.
- **Lệnh kiểm chứng cuối cùng**:
  - `ruff check src apps tests scripts`: All checks passed!
  - `pytest -v tests/unit/`: 73/73 unit test passed (100% green trong ~3.9 giây).
- **Trạng thái tổng thể dự án**: **HOÀN THÀNH TOÀN BỘ 100% THEO PLAN.MD**.


