# ADR 001: Lựa chọn Tập Dữ liệu Snapshot Pháp luật Lao động (2024-09-10)

## Bối cảnh (Context)
Xây dựng hệ thống Hỏi đáp Pháp luật (Legal RAG) tại Việt Nam đòi hỏi kho dữ liệu văn bản quy phạm pháp luật (VBQPPL) đầy đủ, có cấu trúc Điều/Khoản, và bảo toàn được mạng lưới quan hệ hiệu lực (thay thế, sửa đổi, bổ sung, bãi bỏ). Việc tự thu thập (crawl) trực tiếp từ các cổng thông tin pháp luật thời gian thực đối mặt với rủi ro rate limit, captcha, thiếu metadata quan hệ, và không đảm bảo tính tái lập (reproducibility) cho quá trình đánh giá định lượng và ablation studies.

## Quyết định (Decision)
Hệ thống sử dụng bản snapshot cố định tại mốc ngày **10/09/2024** từ tập dữ liệu `bkai-foundation-models/vietnamese-legal-knowledge-base` trên HuggingFace:
1. **Lĩnh vực trọng tâm**: Pháp luật Lao động và Tiền lương tại Việt Nam.
2. **Kho văn bản gốc**: 150 văn bản pháp quy cốt lõi (Bộ luật Lao động 2019, 2012, các Nghị định và Thông tư hướng dẫn thi hành).
3. **Mạng lưới quan hệ hiệu lực**: 2,030 cạnh quan hệ (`REPEALS`, `AMENDS`, `REPEALS_PART`) được phân giải triệt để (100% resolved ID) vào SQLite database `legalrag.db`.
4. **Mốc đóng băng (Snapshot as-of)**: Cố định tại `2024-09-10`. Mọi suy luận hiệu lực trong hệ thống được đối chiếu chuẩn xác với mốc thời gian này.

## Các lựa chọn đã cân nhắc (Alternatives Considered)
- **Crawl trực tiếp theo thời gian thực (Live Crawling)**:
  - *Ưu điểm*: Cập nhật văn bản mới nhất.
  - *Nhược điểm*: Mất tính ổn định khoa học; kết quả benchmark đánh giá (golden set) sẽ bị trôi dạt (drift) theo thời gian; nguy cơ lỗi mạng và blocking.
- **Dùng toàn bộ kho luật đa lĩnh vực (toàn bộ 16,000+ văn bản)**:
  - *Ưu điểm*: Phạm vi tra cứu rộng.
  - *Nhược điểm*: Vượt quá giới hạn tài nguyên tính toán cục bộ (CPU/RAM); nhiễu khi truy xuất ngữ nghĩa giữa các lĩnh vực pháp luật khác nhau; kéo dài thời gian indexing.

## Hệ quả (Consequences)
- **Tích cực**:
  - Đảm bảo 100% tính tái lập cho mọi thực nghiệm Ablation (A0 - A7).
  - Khởi tạo nhanh chỉ mục vector và đồ thị quan hệ trong vài giây trên máy CPU thông thường.
  - Phù hợp tuyệt đối với môi trường kiểm thử CI/CD độc lập.
- **Hạn chế cần nêu rõ cho người dùng**:
  - Dữ liệu dừng ở ngày 10/09/2024; các văn bản ban hành sau thời điểm này chưa có trong kho.
  - Luôn kèm thông báo cố định trong câu trả lời: *"Dữ liệu kho luật cập nhật đến 2024-09-10"*.
