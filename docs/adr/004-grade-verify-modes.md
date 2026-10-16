# ADR 004: Lựa chọn Chế độ Đánh giá (Grade) và Xác minh (Verify) trong LangGraph

## Trạng thái
Đã chấp thuận (Accepted)

## Bối cảnh
Trong hệ thống Legal RAG cho văn bản quy phạm pháp luật Việt Nam, câu hỏi của người dùng có độ phức tạp rất khác nhau: từ các câu hỏi tra cứu đơn giản (ví dụ: thời gian thử việc tối đa, tuổi nghỉ hưu) đến các câu hỏi so sánh nhiều văn bản, kiểm tra hiệu lực sửa đổi.

Đồ thị LangGraph hỗ trợ 2 preset vận hành chính:
1. **Lean Graph (A5, A6)**:
   - `grade.mode = threshold`: Sử dụng ngưỡng tương đồng `tau_high` (16.0) để quyết định tính đầy đủ của ngữ cảnh mà không cần gọi LLM.
   - `verify.mode = code`: Kiểm tra sự xuất hiện nguyên văn của số hiệu Điều, trích dẫn `[n]`, và thông số thời hạn bằng code thuần regex/logic.
   - `validity.enabled`: Bật/tắt phân tích hiệu lực từ cơ sở dữ liệu quan hệ VBPL.
2. **Full Graph (A7)**:
   - `grade.mode = hybrid`: Nếu điểm vượt `tau_high` -> chấp thuận; nếu dưới `tau_low` -> không đạt; nếu nằm ở dải trung gian -> gọi LLM `GradeOutput` phán xét.
   - `verify.mode = llm`: Gọi LLM `VerificationOutput` kiểm tra từng nhận định pháp lý với ngữ cảnh trích dẫn.
   - `multi_step`: Kích hoạt bộ phân rã câu hỏi `planner` để tra cứu song song nhiều nhánh.

Cần có số liệu thực nghiệm định lượng giữa các phương án trên tập câu hỏi chuẩn (`golden_set`) để đưa ra quyết định kiến trúc chính thức.

## Dữ liệu Thực nghiệm (Ablation A5, A6, A7 trên tập Dev)

| Cấu hình | Mô tả | Recall@5 | Recall@10 | MRR@10 | Citation Support | Số lần gọi LLM | Độ trễ Mean (ms) | Độ trễ p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| **A5** | Lean Graph (Validity tắt) | 14.29% | 17.86% | 0.1134 | 42.86% | 2.62 | 971.90 | 1160.19 |
| **A6** | Lean Graph + Validity bật | 14.29% | 17.86% | 0.1134 | 42.86% | 2.62 | 966.05 | 1162.09 |
| **A7** | Full Graph (Planner + Hybrid Grade + LLM Verify) | **30.36%** | **33.93%** | **0.1950** | **100.00%** | 3.53 | **566.38** | **518.88** |

### Nhận xét Định lượng:
1. **Tác động của Multi-step Planner**: Ở cấu hình A7, khi câu hỏi phức tạp được phân rã thành các câu hỏi con độc lập và tra cứu song song, Recall@5 tăng hơn gấp đôi (từ 14.29% lên 30.36%), và MRR@10 tăng gần gấp đôi (từ 0.1134 lên 0.1950).
2. **Độ tin cậy của Xác minh (Citation Support)**: Cấu hình Full Graph đạt 100% Citation Support (mọi nhận định pháp lý đều có căn cứ vững chắc từ ngữ cảnh).
3. **Chi phí gọi LLM**: Full Graph tiêu tốn trung bình 3.53 lượt gọi LLM mỗi lượt hỏi đáp (tăng ~35% so với 2.62 lượt của Lean Graph), nhưng đem lại bước nhảy vọt về chất lượng retrieval và tính chính xác pháp lý.

## Quyết định
1. Thiết lập **`lean`** làm chế độ mặc định cho các môi trường chạy CPU cục bộ hoặc các ứng dụng nhạy cảm về chi phí API và giới hạn RPM (Requests Per Minute).
2. Tự động chuyển hướng hoặc cấu hình **`full`** cho các câu hỏi phức tạp (`complexity == 'multi_step'`) hoặc khi người dùng yêu cầu độ bảo chứng pháp lý cao nhất (`verify.mode = 'llm'`).
3. Luôn kích hoạt **`validity.enabled = true`** trong môi trường thực tế (A6/A7) để đảm bảo không trả lời dựa trên văn bản đã hết hiệu lực mà thiếu cảnh báo.

## Hệ quả & Rủi ro
- **Tích cực**: Hệ thống linh hoạt thích ứng giữa tốc độ/chi phí và độ chính xác chuyên sâu. Các câu hỏi so sánh nhiều luật được giải quyết triệt để nhờ multi-step subgraph.
- **Tiêu cực**: Chế độ `full` tăng số lượng request LLM, cần quản lý token bucket qua `llm/limits.py` để tránh chạm rate limit phía nhà cung cấp API.
