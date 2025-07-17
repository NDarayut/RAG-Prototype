import streamlit as st
import os
import shutil
import fitz  # PyMuPDF
from rag import generate_data_store, ask_question, DATA_PATH

# --- Page Configuration ---
st.set_page_config(page_title="DocChat", layout="wide")

# --- DARK MODE STYLING ---
st.markdown("""
    <style>
    body { background-color: var(--background-color); }
    .stChatBubble {
        padding: 0.8rem 1rem;
        border-radius: 0.6rem;
        margin-bottom: 1rem;
    }
    .user {
        background-color: var(--primary-color-bg);
        border-left: 4px solid #4285F4;
    }
    .bot {
        background-color: var(--secondary-color-bg);
        border-left: 4px solid #34A853;
    }
    .context {
        font-size: 0.9rem;
        color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

# --- Set up state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# --- Dark Mode Toggle ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.dark_mode = st.toggle("🌑 Dark Mode", value=st.session_state.dark_mode)

    # Set CSS vars
    if st.session_state.dark_mode:
        st.markdown("""
            <style>
            :root {
                --background-color: #111827;
                --text-color: #f3f4f6;
                --primary-color-bg: #1e293b;
                --secondary-color-bg: #334155;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            :root {
                --background-color: #ffffff;
                --text-color: #1f2937;
                --primary-color-bg: #e0f2fe;
                --secondary-color-bg: #f3f4f6;
            }
            </style>
        """, unsafe_allow_html=True)

# --- Title ---
st.title("💬 DocChat – Ask Anything From Your PDFs")

# --- File Upload Sidebar ---
with st.sidebar:
    st.subheader("📂 Uploaded PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    def render_pdf_preview(pdf_path, highlights=None, max_pages=3):
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(min(len(doc), max_pages)):
            page = doc.load_page(page_num)

            if highlights:
                # Filter highlights relevant to this file and page
                for h in highlights:
                    if h["filename"] == os.path.basename(pdf_path) and h["page"] == page_num + 1:
                        areas = page.search_for(h["text"], hit_max=5)
                        for rect in areas:
                            page.draw_rect(rect, color=(1, 1, 0), fill=(1, 1, 0), overlay=True)  # yellow highlight

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Better resolution
            img_bytes = pix.tobytes("png")
            images.append(img_bytes)

        doc.close()
        return images


    if uploaded_files:
        if os.path.exists(DATA_PATH):
            shutil.rmtree(DATA_PATH)
        os.makedirs(DATA_PATH, exist_ok=True)

        file_names = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_PATH, uploaded_file.name)
            file_names.append(uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

        st.success(f"Uploaded {len(uploaded_files)} file(s).")

        if st.button("⚡ Generate Vector Store"):
            with st.spinner("Processing and indexing..."):
                generate_data_store()
            st.success("✅ Knowledge base ready!")

        st.markdown("### 🧾 File Manager")
        for name in file_names:
            st.markdown(f"- 📄 `{name}`")
            with st.expander("🖼 Preview Pages"):
                # Inside for name in file_names:
                file_path = os.path.join(DATA_PATH, name)

                # Only use highlights relevant to this file
                file_highlights = []
                for chat in st.session_state.chat_history:
                    for chunk in chat["context"]:
                        if chunk["filename"] == name:
                            file_highlights.append(chunk)

                images = render_pdf_preview(file_path, highlights=file_highlights)


                for img in images:
                    st.image(img, use_container_width=True)
            
    

# --- Main Chat Section ---
st.markdown("### 🧠 Ask Your Question")
query = st.text_input("Type your question here...", placeholder="e.g. What is the conclusion of the document?")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Get Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                answer, context_chunks = ask_question(query)


            st.session_state.chat_history.append({
                "question": query,
                "answer": answer,
                "context": context_chunks  # <-- List of dicts with filename, page, text
            })

with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

# --- Display Chat History ---
if st.session_state.chat_history:
    st.markdown("### 🗂️ Chat History")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="stChatBubble user"><b>Q:</b> {chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stChatBubble bot"><b>A:</b> {chat["answer"]}</div>', unsafe_allow_html=True)
        with st.expander("🧾 View Context Used"):
            for chunk in chat["context"]:
                st.markdown(f"""
                <div style="background-color: #fef9c3; padding: 10px; border-radius: 6px; margin-bottom: 10px;">
                    <b>📄 {chunk['filename']} – Page {chunk['page']}</b>
                    <pre style="white-space: pre-wrap;">{chunk['text']}</pre>
                </div>
                """, unsafe_allow_html=True)
