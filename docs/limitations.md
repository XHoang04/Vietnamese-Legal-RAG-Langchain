# Các Hạn Chế Đã Biết & Đo Lường Hiệu Năng (Limitations & Benchmarks)

Tài liệu này ghi nhận trung thực các giới hạn kỹ thuật, giả định phạm vi, và kết quả đo lường hiệu năng thực tế của hệ thống **Vietnamese Labor Law Legal RAG**.

---

## 1. Các Hạn Chế Đã Biết (Known Limitations)

### 1.1. Mốc Thời gian Dữ liệu Cố định (Data Snapshot)
- Dữ liệu kho luật được cố định tại mốc snapshot **10/09/2024** từ `bkai-foundation-models/vietnamese-legal-knowledge-base`.
- Các văn bản quy phạm pháp luật, nghị định, thông tư hoặc sửa đổi bổ sung ban hành sau ngày 10/09/2024 **hoàn toàn chưa có trong kho dữ liệu**. Người dùng cần kiểm tra thêm các văn bản mới nhất từ Cổng Thông tin điện tử Chính phủ hoặc Thư Viện Pháp Luật.

### 1.2. Trạng thái "Hết hiệu lực một phần" (Partially Expired)
- Dữ liệu nguồn chỉ cung cấp nhãn mức độ văn bản (`status = PARTIALLY_EXPIRED`) và quan hệ bãi bỏ một phần (`REPEALS_PART`). Hệ thống **chưa phân giải tự động được chính xác cụ thể Điều, Khoản, Điểm nào đã bị bãi bỏ**.
- Do đó, khi gặp văn bản này, hệ thống sẽ đưa ra cảnh báo bắt buộc: *"Văn bản {title} đã hết hiệu lực một phần. Hệ thống chưa xác định được điều khoản nào còn hiệu lực, cần đối chiếu văn bản gốc."*

### 1.3. Văn bản Gốc Chưa Hợp nhất (Unconsolidated Source Text)
- Nội dung lưu trữ trong kho luật là toàn văn văn bản gốc ban hành lần đầu, **chưa được hợp nhất văn bản (không phải Văn bản Hợp nhất)**.
- Khi một văn bản đã có văn bản sửa đổi, bổ sung (`AMENDS`), nội dung trích dẫn Điều/Khoản trong văn bản gốc có thể chưa phản ánh các quy định đã được điều chỉnh bởi văn bản mới. Hệ thống giải quyết bằng cảnh báo kèm theo danh sách văn bản sửa đổi.

### 1.4. Tệp PDF Dạng Ảnh Quét (Scanned PDF)
- Hệ thống xử lý PDF dựa trên trích xuất lớp văn bản điện tử trực tiếp (`PyMuPDF`).
- Các tệp PDF dạng ảnh quét hoàn toàn (scanned/raster images) có mật độ ký tự trung bình < 100 ký tự/trang sẽ bị từ chối với mã lỗi `UNSUPPORTED_SCAN_PDF` do hệ thống không tích hợp OCR nặng trên CPU cục bộ.

### 1.5. Giới hạn Tốc độ API Ngoài (Remote LLM Rate Limits)
- Khi sử dụng các dịch vụ LLM miễn phí (như Google AI Studio / Gemini API), tài khoản thường bị giới hạn tốc độ nghiêm ngặt (ví dụ: 15 RPM).
- Hệ thống đã tích hợp client-side rate limiting (token bucket) và retry với exponential backoff (`tenacity`), nhưng nếu vượt quá quota của nhà cung cấp, hệ thống sẽ phản hồi `LLM_UNAVAILABLE`.

### 1.6. Tuyên bố Miễn trừ Trách nhiệm Pháp lý (Legal Disclaimer)
- **Hệ thống này chỉ là một công cụ hỗ trợ tra cứu thông tin học thuật và kỹ thuật, KHÔNG PHẢI VÀ KHÔNG THAY THẾ CHO Ý KIẾN TƯ VẤN PHÁP LÝ CHÍNH THỨC TỪ CÁC LUẬT SƯ HOẶC CHUYÊN GIA PHÁP LÝ ĐƯỢC CẤP PHÉP.**

---

## 2. Đo Lường Hiệu Năng Thực Tế (Performance Benchmarks)

### 2.1. Cấu hình Phần cứng Thử nghiệm
- **Hệ điều hành**: Windows 11 / x86_64
- **CPU**: Intel Core i5 / AMD Ryzen (kiểm thử hoàn toàn trên CPU, không dùng GPU)
- **RAM**: 16 GB
- **Lưu trữ**: NVMe SSD
- **Môi trường**: Python 3.11 trong virtualenv (`.venv`)

### 2.2. Thời gian Thực thi Truy xuất (Retrieval Latency)
- **Dense Vector Search (Qdrant Local)**: ~15ms - 25ms.
- **Sparse BM25 Search (`SyllableBM25`)**: ~5ms - 10ms.
- **Reciprocal Rank Fusion (RRF)**: ~2ms.
- **Vietnamese Cross-Encoder Reranker**:
  - Với `batch_size=8`, `max_length=1024` trên CPU: ~250ms - 450ms.
  - Tổng độ trễ toàn bộ quy trình Hybrid Retrieval (Top 30 -> Rerank Top 8 -> Group Top 5 Parents): **~300ms - 500ms** trên CPU.

### 2.3. Thời gian Thực thi Toàn chuỗi LangGraph (End-to-End Flow)
Đo lường định lượng trên tập `dev` golden set (109 câu hỏi) theo cấu hình thực nghiệm:

| Cấu hình | Recall@5 | MRR@10 | Số lần gọi LLM TB | Độ trễ Trung bình (ms) |
|---|---|---|---|---|
| **A5 (Lean)** | 14.29% | 0.1134 | 2.62 | 971.90 ms |
| **A6 (Lean + Validity)** | 14.29% | 0.1134 | 2.62 | 966.05 ms |
| **A7 (Full: Planner + Verify)** | **30.36%** | **0.1950** | **3.53** | **566.38 ms** |

*Ghi chú*: Cấu hình A7 cho độ trễ trung bình thấp hơn do cơ chế cache phản hồi LLM cục bộ (SQLite cache `data/cache/llm_cache.sqlite`), giảm đáng kể thời gian chờ mạng khi chạy các tác vụ phân tách planner song song.
