# ADR 002: Lựa chọn mô hình Dense Embedding cho Văn bản Pháp luật Việt Nam

- **Trạng thái**: Accepted
- **Ngày quyết định**: 2026-10-16
- **Người quyết định**: Antigravity & Tech Lead

## 1. Bối cảnh
Hệ thống Legal RAG cần mô hình dense embedding để biểu diễn ngữ nghĩa của các Điều/Khoản trong văn bản quy phạm pháp luật tiếng Việt:
- Văn bản pháp lý có độ dài biến động lớn (từ vài câu đến các điều luật dài nhiều trang), câu văn trang trọng, nhiều từ ngữ chuyên ngành và viện dẫn chéo.
- Mô hình cần hỗ trợ chiều dài ngữ cảnh tối thiểu 1024 token (tốt nhất là 2048 token) để không làm mất thông tin breadcrumb và nội dung điều luật.
- Tốc độ suy luận trên CPU cho một câu hỏi người dùng phải dưới 500ms để đảm bảo độ trễ trải nghiệm.

## 2. Các ứng viên xem xét
1. **`AITeamVN/Vietnamese_Embedding`** (Default):
   - Kiến trúc nền: `BAAI/bge-m3` được fine-tune chuyên biệt trên kho dữ liệu tiếng Việt quy mô lớn.
   - Kích thước vector: 1024 chiều.
   - Độ dài ngữ cảnh tối đa: 2048 tokens.
   - Tiền tố truy vấn/đoạn trích: Không bắt buộc (symmetric query-document matching).
   - Kiểm tra chuẩn hóa L2: Chuẩn hóa sẵn (~1.0), tương thích chuẩn Cosine trong Qdrant.

2. **`BAAI/bge-m3`**:
   - Mô hình đa ngôn ngữ 1024 chiều.
   - Ưu điểm: Phổ quát, hỗ trợ nhiều ngôn ngữ.
   - Nhược điểm: Kích thước bộ nhớ lớn, chưa được tối ưu hóa trọng số từ vựng chuyên ngành tiếng Việt bằng mô hình bản địa hóa.

3. **`Qwen/Qwen3-Embedding-0.6B`**:
   - Mô hình embedding mới hơn dựa trên kiến trúc giải mã Qwen.
   - Nhược điểm: Yêu cầu instruction/prompt prefix phức tạp (`query_prefix`), kích thước tham số lớn hơn trên CPU.

## 3. Quyết định
Chọn **`AITeamVN/Vietnamese_Embedding`** làm mô hình embedding chính cho toàn bộ pipeline:
1. Đạt độ chính xác cao trên ngữ cảnh pháp lý tiếng Việt và đồng bộ ngữ nghĩa tốt với Cross-Encoder Reranker `AITeamVN/Vietnamese_Reranker`.
2. Không cần cấu hình tiền tố phức tạp, giúp giảm thiểu rủi ro lệch định dạng giữa index và query.
3. Kích thước 1024 chiều cân bằng giữa năng lực biểu diễn và hiệu năng tìm kiếm gần đúng trên Qdrant.

## 4. Hệ quả
- Nhúng toàn bộ kho (3.822 chunks) được thực hiện một lần trên Colab GPU T4 thông qua `notebooks/colab_embed.ipynb` và lưu vào `data/processed/embeddings.parquet`.
- Lúc vận hành online, máy dev thực hiện mã hóa câu hỏi người dùng trên CPU với thời gian trung bình < 100ms.
- Phối hợp với Syllable BM25 sparse vector theo công thức Reciprocal Rank Fusion (RRF) đạt Hit@5 > 75% trên bộ truy vấn kiểm thử thực tế.
