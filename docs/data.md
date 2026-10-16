# Báo cáo Khảo sát Dữ liệu Pháp luật (ThucPD/vbpl-vn)

- **Thời điểm khảo sát**: 2026-10-16
- **Tập dữ liệu**: Hugging Face `ThucPD/vbpl-vn` (split `train`, snapshot `2024-09-10`)
- **Số mẫu khảo sát**: 5.000 dòng liên tục (đọc streaming, bỏ qua `html_text`)
- **File báo cáo chi tiết**: [`data/interim/inspect_report.json`](file:///d:/Hoang/Project_RAG/data/interim/inspect_report.json)

---

## 1. Kết quả kiểm chứng các GIẢ ĐỊNH (Mục 7.2 PLAN.md)

| Giả định mục 7.2 | Kết quả kiểm chứng | Thực tế quan sát | Kết luận |
|---|---|---|---|
| **G1: Định dạng `title`** | ĐÚNG | Tiêu đề chứa `<Loại văn bản> <số hiệu>`, ví dụ: `Nghị quyết 19/2020/NQ-HĐND`, `Thông tư 04/2009/TT-BNN`, `Quyết định 57/2000/QĐ-UBND`. | Giữ nguyên regex trích xuất số hiệu |
| **G2: Phân bố URL** | ĐÚNG | `vbpq-toanvan`: 81,3% (4.063 dòng), `vbpqen-toanvan` (tiếng Anh): 7,3% (363 dòng), `vbpq-van-ban-goc`: 11,5% (574 dòng). | Lọc bỏ URL chứa `vbpqen` theo `ENGLISH_URL` |
| **G3: Thuộc tính `attribute`** | ĐÚNG | 16 khóa luôn có mặt trong dict: `document_info` chứa trích yếu và dòng `Tình trạng: ...`. `official_number`, `document_type`, `effective_area`, các trường ngày. | Sử dụng đầy đủ các khóa đã phân tích |
| **G4: Giá trị `Tình trạng`** | ĐÚNG (mở rộng nhẹ) | `Còn hiệu lực` (53,7%), `Hết hiệu lực toàn bộ` (39,6%), `Hết hiệu lực một phần` (4,6%), `Ngưng hiệu lực` (0,4%), `Chưa có hiệu lực` (5 dòng), `Chưa xác định` (2 dòng), rỗng (83 dòng). | Ánh xạ vào `Status` Enum (IN_FORCE, EXPIRED, PARTIALLY_EXPIRED, SUSPENDED, UNKNOWN) |
| **G5: Định dạng ngày tháng** | ĐÚNG | 66,8% các trường ngày có định dạng chuẩn `dd/mm/yyyy`. Các trường không khớp chủ yếu do chuỗi rỗng `""` hoặc placeholder `"..."`. | `parse_date` trả `None` khi không đúng regex, không raise |
| **G6: Cấu trúc quan hệ `schema`** | ĐÚNG | Toàn bộ 18 khóa quan hệ xuất hiện đầy đủ trong 100% bản ghi: `amend_documents`, `amended_documents`, `cancel_documents`, `canceled_documents`,... | Giữ nguyên bảng ánh xạ `RelType` |
| **G7: Văn bản hết hiệu lực có văn bản thay thế** | ĐÚNG (66,3%) | Trong số các văn bản `Hết hiệu lực toàn bộ`, 66,3% (1.311/1.978) có danh sách `cancel_documents` rõ ràng. | Dùng `cancel_documents` làm `replaced_by` |
| **G8: Metadata lệch** | ĐÚNG (7,2%) | 7,16% văn bản có độ trùng token giữa trích yếu và 600 ký tự đầu < 0,3. | Gắn nhãn `METADATA_MISMATCH` và gán `status=UNKNOWN` |

---

## 2. Thống kê độ dài và cấu trúc văn bản

- **Phân vị độ dài văn bản (`full_text`)**:
  - `p50` (trung vị): 4.094 ký tự.
  - `p90`: 23.669 ký tự.
  - `p99`: 77.863 ký tự.
  - Tỉ lệ rỗng hoặc ngắn hơn 200 ký tự: **11,74%** (loại bằng bộ lọc `EMPTY_TEXT`).
- **Cấu trúc Điều/Khoản**:
  - Tỉ lệ có dòng bắt đầu bằng `Điều ...`: 21,2% (tính trên toàn bộ loại hình; với Luật/Nghị định/Thông tư tỉ lệ này cao hơn rất nhiều). Các văn bản không có Điều (Quyết định cá biệt, chỉ thị ngắn) được chuyển sang cơ chế flat window chunking.
  - Mật độ xuống dòng trung vị: 25,3 dòng / 1.000 ký tự (đủ mật độ để không bị rơi vào chế độ inline_mode bắt buộc).

---

## 3. Các loại văn bản và phạm vi hiệu lực phổ biến

- **Top loại văn bản**:
  - Quyết định: 2.895 (57,9%)
  - Nghị quyết: 843 (16,9%)
  - Thông tư / Thông tư liên tịch: 633 (12,7%)
  - Chỉ thị: 312 (6,2%)
  - Nghị định: 168 (3,4%)
  - Luật / Bộ luật: 20 (0,4%)
- **Phạm vi hiệu lực**:
  - `Toàn quốc`: 1.291 văn bản (25,8%). Khi cờ `national_only: true` được bật, hệ thống ưu tiên lọc các văn bản áp dụng toàn quốc.
