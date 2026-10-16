# Open Questions

1. Lĩnh vực: Mặc định là lĩnh vực Lao động (`domain: lao_dong`).
2. Nhà cung cấp LLM dùng thực tế: Mặc định cấu hình hỗ trợ Google Gemini qua `langchain-google-genai` hoặc OpenAI/OpenRouter qua `langchain-openai`.
3. Hệ điều hành máy dev: Windows x64.
4. Ollama và local model: Tùy chọn cục bộ cho upload. Nếu không có, cờ `privacy.allow_remote_for_uploads: false` sẽ trả mã lỗi `UPLOAD_REQUIRES_LOCAL_LLM` theo đúng đặc tả mục 13.5.
5. Điều khoản sử dụng dữ liệu vbpl.vn: Ghi nhận nguồn trích xuất từ dataset `ThucPD/vbpl-vn`.
6. UI: Streamlit (`apps/ui/app.py`).
