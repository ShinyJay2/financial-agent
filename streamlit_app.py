import streamlit as st
import os
from app.rag_pipeline import RAGPipeline  # adjust import path if needed

# Page configuration
st.set_page_config(page_title="M.Riskit", layout="wide")

def load_css(css_file_path):
    if os.path.exists(css_file_path):
        with open(css_file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 1) Load custom CSS
css_path = os.path.join("static", "style.css")
load_css(css_path)

# 2) Sidebar navigation
st.sidebar.title("M.Riskit")
st.sidebar.header("내 계정 정보")

menu = [
    "에코프로비엠 위험성을 요약해줘",
    "에코프로비엠 위험지표를 알려줘",
    "에코프로비엠에 대한 투자 의견을 알려줘",
    "에코프로비엠에 대한 주가를 알려줘",
    "에코프로비엠 관련 뉴스를 알려줘",
    "현재 에코프로비엠 30주를 가지고 있는데, 헷징하려면 어떻게 해야할 지 알려줘"
]
choice = st.sidebar.radio("메뉴", menu)

# 3) Main chat area state
if 'history' not in st.session_state:
    st.session_state.history = []

# Treat the menu choice as the user query
user_input = choice

# When a menu item is selected, invoke the RAG pipeline
if user_input:
    pipeline = RAGPipeline()
    answer, sources = pipeline.answer(user_input)  # expects (str, List[(doc_id, chunk)])
    st.session_state.history.append({
        "user": user_input,
        "answer": answer,
        "sources": sources
    })

# 4) Display chat history and toggles
chat_col, doc_col = st.columns((3, 1))
with chat_col:
    for idx, turn in enumerate(st.session_state.history):
        # User message bubble
        st.markdown(f"<div class='chat-bubble user'>{turn['user']}</div>", unsafe_allow_html=True)
        # Assistant response bubble
        st.markdown(f"<div class='chat-bubble assistant'>{turn['answer']}</div>", unsafe_allow_html=True)

        # References header
        st.markdown("<div class='docs-header'>📑 참조 자료</div>", unsafe_allow_html=True)

        # Toggleable document sources
        for i, (doc_id, chunk) in enumerate(turn['sources']):
            toggle_key = f"toggle_{idx}_{i}"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False

            cols = st.columns((1, 5))
            # View button
            if cols[0].button("👁️ 보기", key=toggle_key):
                st.session_state[toggle_key] = not st.session_state[toggle_key]

            # Document label
            cols[1].markdown(f"**{doc_id}**")

            # On toggle, display snippet or chunk fallback
            if st.session_state[toggle_key]:
                data_dir = os.path.join(os.getcwd(), "data")
                path = os.path.join(data_dir, doc_id)
                if os.path.exists(path):
                    with open(path, encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    snippet = content[:1000] + ("..." if len(content) > 1000 else "")
                    st.code(snippet)
                else:
                    st.write(chunk)

# Right column (reserved for future controls)
with doc_col:
    st.markdown("&nbsp;")
