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

# Initialize
pipeline = RAGPipeline()

if "history" not in st.session_state:
    st.session_state.history = []  # each: {"question", "answer", "sources"}

# —————————————————————————————————————————————————————
# Sidebar: history & reset
# —————————————————————————————————————————————————————
with st.sidebar:
    st.title("💬 대화 기록")
    if st.button("새로운 질문"):
        st.session_state.history.clear()
        st.experimental_rerun()

    for turn in st.session_state.history:
        st.markdown(f"<div class='chat-bubble user'>{turn['question']}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble assistant'>{turn['answer']}</div>",
                    unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("© 2025 Your Name")

# —————————————————————————————————————————————————————
# Main: input, display, sources
# —————————————————————————————————————————————————————
st.title("📊 Korean Finance RAG Chat")

# 1) Input
query = st.chat_input("궁금한 것을 물어보세요…")
if query:
    try:
        answer, sources = pipeline.answer_with_sources(
            query, history=st.session_state.history
        )
        if "궁금하신 종목을 입력해주세요" in answer:
            st.warning("종목 티커를 포함해 질문해 주세요 (예: AAPL에 대한 정보).")
        else:
            st.session_state.history.append({
                "question": query,
                "answer": answer,
                "sources": sources
            })
            st.experimental_rerun()
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}. 다시 시도해주세요.")

# 2) Display last turn
if st.session_state.history:
    last = st.session_state.history[-1]
    # Assistant bubble
    st.markdown(f"<div class='chat-bubble assistant'>{last['answer']}</div>",
                unsafe_allow_html=True)

    # Sources
    st.markdown("### 📑 참조 자료")
    for doc_id, chunk in last["sources"]:
        with st.expander(f"출처: {doc_id}", expanded=False):
            st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)  # Truncate long chunks