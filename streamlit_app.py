import streamlit as st
import os
import sys
import shutil
from pathlib import Path
from app.rag_pipeline import RAGPipeline

# — Page config —
st.set_page_config(page_title="M.Riskit", layout="wide")

# — Compute project root (script now at repo root) —
BASE_DIR = Path(__file__).resolve().parent

# — Helper: load CSS from static/style.css —
def load_css():
    css_file = BASE_DIR / "static" / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

load_css()

# — Sidebar menu —
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

# — Check for force‐reingest flag (passed after “--” to Streamlit) —
force = "--force" in sys.argv

# — Lazily initialize RAG pipeline & ingestion logic —
if 'pipeline' not in st.session_state:
    # 1) Set up ChromaDB directory
    DB_DIR = BASE_DIR / "chroma_db"

    # 2) Optionally wipe the entire store on --force
    if force and DB_DIR.is_dir():
        st.write(f"🧹 [force] Deleting old ChromaDB at: {DB_DIR}")
        shutil.rmtree(DB_DIR)

    # 3) Ensure directory exists
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # 4) Instantiate pipeline with desired settings
    rag = RAGPipeline(
        chunk_method="section",
        bm25_k=20,
        dense_k=20,
        final_k=10,
        max_tokens=4096,
        overlap=50,
    )

    # 5) Ingest only if the store is empty
    if not any(DB_DIR.iterdir()):
        st.write("🧹 Ingesting documents (this will call the embedding model once)…")
        DATA_ROOT = BASE_DIR / "data"
        for file_path in DATA_ROOT.rglob("*"):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            st.write(f"=== Ingesting {file_path.relative_to(BASE_DIR)} ===")
            rag.ingest_file(str(file_path))
    else:
        st.write("✅ Vector store already exists—skipping ingest/embedding.")

    # 6) Store pipeline for later reuse
    st.session_state.pipeline = rag

# — Main chat logic —
if 'history' not in st.session_state:
    st.session_state.history = []

user_input = choice
if user_input:
    answer, sources = st.session_state.pipeline.answer_with_sources(user_input)
    st.session_state.history.append({
        "user": user_input,
        "answer": answer,
        "sources": sources
    })

# — Render chat & source toggles —
chat_col, doc_col = st.columns((3, 1))
with chat_col:
    for idx, turn in enumerate(st.session_state.history):
        # user bubble
        st.markdown(f"<div class='chat-bubble user'>{turn['user']}</div>", unsafe_allow_html=True)
        # assistant bubble
        st.markdown(f"<div class='chat-bubble assistant'>{turn['answer']}</div>", unsafe_allow_html=True)

        # reference header
        st.markdown("<div class='docs-header'>📑 참조 자료</div>", unsafe_allow_html=True)

        # toggleable sources
        for i, (doc_id, chunk) in enumerate(turn['sources']):
            toggle_key = f"toggle_{idx}_{i}"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False

            cols = st.columns((1, 5))
            if cols[0].button("👁️ 보기", key=toggle_key):
                st.session_state[toggle_key] = not st.session_state[toggle_key]
            cols[1].markdown(f"**{doc_id}**")

            if st.session_state[toggle_key]:
                snippet_path = BASE_DIR / "data" / doc_id
                if snippet_path.exists():
                    text = snippet_path.read_text(encoding='utf-8', errors='ignore')
                    st.code(text[:1000] + ("..." if len(text) > 1000 else ""))
                else:
                    st.write(chunk)

with doc_col:
    st.markdown("&nbsp;")
