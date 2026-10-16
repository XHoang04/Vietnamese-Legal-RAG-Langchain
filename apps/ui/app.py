"""Streamlit Web UI for Vietnamese Labor Law Legal RAG Assistant."""

import json
import os
import time
import uuid

import httpx
import streamlit as st

# Configuration Defaults
DEFAULT_API_URL = os.environ.get("LEGALRAG_API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.environ.get("API_KEYS", "dev-key-1").split(",")[0].strip()

st.set_page_config(
    page_title="Trợ lý Pháp luật Lao động Việt Nam",
    page_icon="⚖️",
    layout="wide",
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:10]}"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "interrupted_question" not in st.session_state:
    st.session_state.interrupted_question = None


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ Pháp Luật Lao Động")

    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
    headers = {"X-API-Key": api_key, "X-Session-Id": st.session_state.session_id}

    # API Status check
    try:
        r = httpx.get(f"{api_url}/health", timeout=2.0)
        if r.status_code == 200:
            st.success(" Đang hoạt động ")
        else:
            st.warning("Không phản hồi ")
    except Exception:
        st.error("Ngoại tuyến ")

    st.markdown("---")
    st.subheader("Phiên làm việc (Session)")
    st.code(st.session_state.session_id, language="text")

    if st.button("Tạo phiên mới ", use_container_width=True):
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:10]}"
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.interrupted_question = None
        st.rerun()

    st.markdown("---")
    st.subheader("Tải lên tài liệu PDF")
    uploaded_file = st.file_uploader("Chọn file PDF (Quy chế, HĐLĐ...)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Tải lên & Xử lý ", use_container_width=True):
            with st.spinner("Đang tải lên và phân tích tài liệu..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = httpx.post(f"{api_url}/v1/documents", files=files, headers=headers, timeout=30.0)
                    if resp.status_code in [200, 202]:
                        st.success("Tải lên thành công! Tài liệu đang được chỉ mục.")
                        time.sleep(1.0)
                        st.rerun()
                    else:
                        err = resp.json().get("error", {})
                        st.error(f"Lỗi tải lên: {err.get('message', resp.text)}")
                except Exception as ex:
                    st.error(f"Lỗi kết nối: {ex}")

    # List session documents
    st.subheader("Tài liệu trong phiên")
    try:
        docs_resp = httpx.get(
            f"{api_url}/v1/sessions/{st.session_state.session_id}/documents",
            headers=headers,
            timeout=5.0,
        )
        if docs_resp.status_code == 200:
            docs = docs_resp.json().get("documents", [])
            if not docs:
                st.info("Chưa có tài liệu tải lên.")
            for doc in docs:
                with st.expander(f"📄 {doc['filename']}"):
                    st.write(f"Trạng thái: **{doc['status']}**")
                    if doc.get("pages"):
                        st.write(f"Số trang: {doc['pages']}")
                    if doc.get("chunks"):
                        st.write(f"Số đoạn: {doc['chunks']}")
                    if st.button("Xóa tài liệu ", key=f"del_{doc['upload_id']}"):
                        httpx.delete(f"{api_url}/v1/documents/{doc['upload_id']}", headers=headers)
                        st.rerun()
    except Exception:
        pass


# --- MAIN CHAT AREA ---
st.header("Tra cứu & Tư vấn Pháp luật Lao động")
st.markdown(
    "> **Lưu ý**: Hệ thống chỉ đóng vai trò trợ lý tra cứu văn bản pháp luật, **không thay thế** tư vấn pháp lý chính thức từ luật sư."
)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display warnings if present
        if msg.get("warnings"):
            for w in msg["warnings"]:
                st.warning(f"⚠️ {w}")

        # Display citations if present
        if msg.get("citations"):
            with st.expander("📚 Căn cứ pháp lý & Trích dẫn"):
                for c in msg["citations"]:
                    breadcrumb = c.get("breadcrumb") or ""
                    title = c.get("title") or c.get("doc_id") or ""
                    url = c.get("url")
                    doc_number = c.get("doc_number")
                    article_no = c.get("article_no")

                    doc_display = f"**{title}**"
                    if doc_number:
                        doc_display += f" (Số hiệu: `{doc_number}`)"
                    if article_no:
                        doc_display += f" - Điều {article_no}"

                    if url:
                        st.markdown(f"- [{doc_display}]({url}) — *{breadcrumb}*")
                    else:
                        st.markdown(f"- {doc_display} — *{breadcrumb}*")

        # Display execution metrics trace if present
        if msg.get("metrics"):
            m = msg["metrics"]
            with st.expander(f"⏱️ Vết thực thi đồ thị ({m.get('latency_ms', 0)} ms)"):
                st.write(f"- Thời gian phản hồi: `{m.get('latency_ms')} ms`")
                st.write(f"- Số lần gọi LLM: `{m.get('llm_calls')}`")
                st.write(f"- Chuỗi node: `{' ➔ '.join(m.get('nodes', []))}`")


# Handle Clarification Interrupt Prompt
if st.session_state.interrupted_question:
    st.info(f"❓ **Hệ thống cần làm rõ**: {st.session_state.interrupted_question}")
    with st.form("clarify_form"):
        clarify_ans = st.text_input("Câu trả lời của bạn:")
        submit_btn = st.form_submit_button("Gửi thông tin làm rõ 🚀")
        if submit_btn and clarify_ans:
            # Append user response to message history
            st.session_state.messages.append({"role": "user", "content": clarify_ans})
            resume_url = f"{api_url}/v1/chat/{st.session_state.thread_id}/resume"

            # Stream resume
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                citations = []
                warnings = []
                metrics = {}

                with httpx.stream(
                    "POST",
                    resume_url,
                    json={"answer": clarify_ans},
                    headers=headers,
                    timeout=60.0,
                ) as stream:
                    current_event = None
                    for line in stream.iter_lines():
                        if line.startswith("event:"):
                            current_event = line.replace("event:", "").strip()
                        elif line.startswith("data:") and current_event:
                            data_str = line.replace("data:", "").strip()
                            try:
                                data = json.loads(data_str)
                            except Exception:
                                data = {}

                            if current_event == "token":
                                full_response += data.get("text", "")
                                message_placeholder.markdown(full_response + "▌")
                            elif current_event == "citations":
                                citations = data
                            elif current_event == "warnings":
                                warnings = data
                            elif current_event == "done":
                                metrics = data.get("metrics", {})

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations,
                    "warnings": warnings,
                    "metrics": metrics,
                })
                st.session_state.interrupted_question = None
                st.rerun()


# Normal Chat Input
prompt = st.chat_input("Nhập câu hỏi về luật lao động, thời giờ làm việc, nghỉ phép, hợp đồng...")

if prompt:
    # 1. Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""
        citations = []
        warnings = []
        metrics = {}
        interrupted = False

        try:
            chat_url = f"{api_url}/v1/chat"
            payload = {
                "message": prompt,
                "thread_id": st.session_state.thread_id,
                "session_id": st.session_state.session_id,
            }

            with httpx.stream("POST", chat_url, json=payload, headers=headers, timeout=90.0) as stream:
                current_event = None
                for line in stream.iter_lines():
                    if line.startswith("event:"):
                        current_event = line.replace("event:", "").strip()
                    elif line.startswith("data:") and current_event:
                        data_str = line.replace("data:", "").strip()
                        try:
                            data = json.loads(data_str)
                        except Exception:
                            data = {}

                        if current_event == "step":
                            node = data.get("node", "")
                            status_placeholder.caption(f"⚙️ Đang thực thi bước: `{node}`...")

                        elif current_event == "interrupt":
                            interrupted = True
                            st.session_state.interrupted_question = data.get("question")
                            status_placeholder.empty()
                            st.rerun()

                        elif current_event == "token":
                            status_placeholder.empty()
                            full_response += data.get("text", "")
                            message_placeholder.markdown(full_response + "▌")

                        elif current_event == "citations":
                            citations = data

                        elif current_event == "warnings":
                            warnings = data

                        elif current_event == "done":
                            metrics = data.get("metrics", {})

                        elif current_event == "error":
                            st.error(f"Lỗi: {data.get('message', 'Đã xảy ra lỗi không xác định')}")

            status_placeholder.empty()
            message_placeholder.markdown(full_response)

            # Display citations and warnings in the active message
            if warnings:
                for w in warnings:
                    st.warning(f"⚠️ {w}")

            if citations:
                with st.expander("📚 Căn cứ pháp lý & Trích dẫn"):
                    for c in citations:
                        b = c.get("breadcrumb") or ""
                        t = c.get("title") or c.get("doc_id") or ""
                        u = c.get("url")
                        d_no = c.get("doc_number")
                        a_no = c.get("article_no")
                        disp = f"**{t}**"
                        if d_no:
                            disp += f" (Số hiệu: `{d_no}`)"
                        if a_no:
                            disp += f" - Điều {a_no}"
                        if u:
                            st.markdown(f"- [{disp}]({u}) — *{b}*")
                        else:
                            st.markdown(f"- {disp} — *{b}*")

            if metrics:
                with st.expander(f"⏱️ Vết thực thi ({metrics.get('latency_ms', 0)} ms)"):
                    st.write(f"- Thời gian: `{metrics.get('latency_ms')} ms`")
                    st.write(f"- Số lượt LLM: `{metrics.get('llm_calls')}`")
                    st.write(f"- Các node: `{' ➔ '.join(metrics.get('nodes', []))}`")

            # Save assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citations": citations,
                "warnings": warnings,
                "metrics": metrics,
            })

        except Exception as e:
            st.error(f"Lỗi kết nối tới API server: {e}")
