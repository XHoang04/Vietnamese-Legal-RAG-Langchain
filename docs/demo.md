# Hướng Dẫn Thử Nghiệm & Kịch Bản Demo (Demo Guide)

Tài liệu này cung cấp các kịch bản chạy thử nghiệm thực tế hệ thống **Vietnamese Labor Law Legal RAG** qua cả 3 phương thức: **CLI**, **REST API (cURL / Swagger)**, và **Giao diện Web Streamlit**.

---

## 1. Chuẩn Bị Môi Trường

Kích hoạt môi trường ảo Python đã được cài đặt:
```bash
# Trên Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Trên Linux / macOS
source .venv/bin/activate
```

---

## 2. Thử Nghiệm Qua Giao Diện Dòng Lệnh (CLI)

### Kịch bản 1: Hỏi đáp trực tiếp toàn diện qua LangGraph
```bash
# Chạy với mô hình giả lập (không cần mạng, không tốn quota API)
python -m legalrag.cli ask "Thời giờ làm việc bình thường là mấy tiếng?" --fake

# Chạy thực với cơ sở dữ liệu thật và vector index
python -m legalrag.cli ask "Thời gian thử việc tối đa là bao lâu?"
```
*Kết quả quan sát được*:
- Hiển thị nội dung câu trả lời chuẩn xác trích dẫn `[1]`, Điều/Khoản.
- Danh sách trace thực thi chi tiết qua các node: `router`, `retrieve`, `grade`, `validity_check`, `generate`, `verify`, `finalize`.

### Kịch bản 2: Tra cứu trạng thái hiệu lực văn bản
```bash
# 1. Văn bản ĐÃ HẾT HIỆU LỰC (Bị thay thế bởi Luật mới)
python -m legalrag.cli check-validity "17/2008/QH12"

# 2. Văn bản HẾT HIỆU LỰC MỘT PHẦN
python -m legalrag.cli check-validity "25/2008/QH12"

# 3. Văn bản ĐANG CÒN HIỆU LỰC (Bộ luật Lao động 2019)
python -m legalrag.cli check-validity "45/2019/QH14"
```

---

## 3. Khởi Chạy REST API & Streamlit Web UI

### Khởi chạy Backend API
```bash
python -m uvicorn apps.api.main:app --port 8000 --reload
# Hoặc dùng Makefile
make api
```
- **Liveness probe**: `http://localhost:8000/health`
- **Readiness probe**: `http://localhost:8000/ready`
- **Swagger Documentation**: `http://localhost:8000/docs`

### Khởi chạy Giao diện Web Streamlit
Mở một cửa sổ terminal mới (đã kích hoạt `.venv`):
```bash
streamlit run apps/ui/app.py
# Hoặc dùng Makefile
make ui
```
Trình duyệt sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`.

---

## 4. Kịch Bản Thử Nghiệm Trên Giao Diện Web Streamlit

### Kịch bản 1: Tra cứu quy định đang còn hiệu lực
1. Nhập câu hỏi vào ô chat:
   > *"Thời gian thử việc tối đa đối với công việc có trình độ cao đẳng trở lên là bao lâu?"*
2. **Quan sát**:
   - Vết thực thi `step` hiển thị các bước đang xử lý.
   - Trả lời: Tối đa không quá 60 ngày theo Điều 25 Bộ luật Lao động 2019.
   - Accordion **📚 Căn cứ pháp lý & Trích dẫn**: Hiển thị liên kết trực tiếp tới văn bản trên Thư Viện Pháp Luật.
   - Accordion **⏱️ Vết thực thi**: Hiển thị chi tiết số ms và số lần gọi LLM.

### Kịch bản 2: Cảnh báo văn bản đã hết hiệu lực
1. Nhập câu hỏi:
   > *"Theo Bộ luật Lao động 2012, mức lương tối thiểu được quy định như thế nào?"*
2. **Quan sát**:
   - Khung cảnh báo màu vàng hiển thị:
     `⚠️ Văn bản Bộ luật Lao động 2012 đã hết hiệu lực toàn bộ; bị thay thế/bãi bỏ bởi: Bộ luật Lao động 2019.`

### Kịch bản 3: Tải lên tệp PDF quy chế nội bộ & Hỏi đáp cô lập phiên
1. Ở thanh bên trái (Sidebar), tại mục **Tải lên tài liệu PDF**, bấm chọn tệp PDF mẫu (hoặc Hợp đồng lao động).
2. Bấm **Tải lên & Xử lý 📤**.
3. Danh sách **Tài liệu trong phiên** sẽ xuất hiện tài liệu với trạng thái `ready` kèm số trang và số đoạn.
4. Nhập câu hỏi liên quan đến tài liệu vừa tải:
   > *"Quy định về thời giờ làm việc trong quy chế này là gì?"*
5. **Quan sát**:
   - Hệ thống tự động chuyển `scope="both"` hoặc `scope="upload"`.
   - Hiển thị trích dẫn từ tài liệu người dùng tải lên với cảnh báo: *"Tài liệu do bạn tải lên, hệ thống không xác minh hiệu lực."*
6. Bấm nút **Tạo phiên mới 🔄** để đổi sang phiên khác: tài liệu vừa tải ở phiên trước hoàn toàn không thể truy cập được từ phiên mới (đảm bảo cô lập tuyệt đối, chống rò rỉ dữ liệu).

---

## 5. Triển Khai Bằng Docker Compose
Chỉ với một câu lệnh duy nhất để khởi động toàn bộ hạ tầng (Qdrant, API Backend, Streamlit Web):
```bash
docker compose up -d
# Hoặc dùng Makefile
make up
```
- Kiểm tra trạng thái: `docker compose ps`
- Tắt hệ thống: `docker compose down` (hoặc `make down`)
