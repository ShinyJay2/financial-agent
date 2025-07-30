import streamlit as st
import os

from app.rag_pipeline import RAGPipeline  # adjust import path if needed

# —————————————————————————————————————————————————————
# Load custom CSS
# —————————————————————————————————————————————————————
css_file = os.path.join("static", "style.css")
if os.path.exists(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# —————————————————————————————————————————————————————
# App config
# —————————————————————————————————————————————————————
st.set_page_config(layout="wide", page_title="Korean Finance RAG Chat")

# —————————————————————————————————————————————————————
# Cache your RAG pipeline once per session
# —————————————————————————————————————————————————————
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

# —————————————————————————————————————————————————————
# Initialize chat history
# —————————————————————————————————————————————————————
if "history" not in st.session_state:
    st.session_state.history = []  # each: {"question","answer","sources"}

# —————————————————————————————————————————————————————
# Sidebar: conversation history & reset
# —————————————————————————————————————————————————————
with st.sidebar:
    st.title("💬 대화 기록")
    if st.button("새로운 질문"):
        st.session_state.history.clear()
    for turn in st.session_state.history:
        st.markdown(f"<div class='chat-bubble user'>{turn['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble assistant'>{turn['answer']}</div>", unsafe_allow_html=True)
        st.markdown("---")
    st.markdown("© 2025 Your Name")

# —————————————————————————————————————————————————————
# Main UI: input & display
# —————————————————————————————————————————————————————
st.title("📊 Korean Finance RAG Chat")

query = st.chat_input("궁금한 것을 물어보세요…")
if query:
    # 1) Retrieve & chunk first
    with st.spinner("Retrieving relevant passages…"):
        answer, sources = pipeline.answer_with_sources(query, history=st.session_state.history)

    # 2) Validate & append to history
    if "궁금하신 종목을 입력해주세요" in answer:
        st.warning("종목 티커를 포함해 질문해 주세요 (예: AAPL에 대한 정보).")
    else:
        st.session_state.history.append({
            "question": query,
            "answer": answer,
            "sources": sources
        })

# 3) Display the latest response
if st.session_state.history:
    last = st.session_state.history[-1]

    st.markdown(f"<div class='chat-bubble assistant'>{last['answer']}</div>", unsafe_allow_html=True)

    st.markdown("### 📑 참조 자료")
    for doc_id, chunk in last["sources"]:
        with st.expander(f"출처: {doc_id}", expanded=False):
            st.write(chunk[:500] + "…" if len(chunk) > 500 else chunk)
