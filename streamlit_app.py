# streamlit_app.py
# Run: streamlit run streamlit_app.py

import os
import re
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st

# make Streamlit secrets visible to code that reads os.environ / pydantic Settings
for k, v in st.secrets.items():
    os.environ.setdefault(k, str(v))


# Optional pdf.js viewer (preferred for large PDFs)
try:
    from streamlit_pdf_viewer import pdf_viewer
    HAS_PDFJS = True
except Exception:
    HAS_PDFJS = False

# ---------- Your backend ----------
from app.rag_pipeline import RAGPipeline


# ==============================
# Page & Global Styles (Dark, Product-y)
# ==============================
st.set_page_config(page_title="FinAgent · Risk Chat", page_icon="💬", layout="wide")

CUSTOM_CSS = """
<style>
:root {
  --bg:#0b0f19; --panel:#0f172a; --panel-2:#0b1222;
  --text:#e5e7eb; --muted:#94a3b8; --border:#1f2937;
  --ring:#6366f1; --accent:#22d3ee; --accent-2:#8b5cf6; --ok:#10b981;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] > div { background: linear-gradient(180deg,#0d1324 0%, #0b0f19 100%); }
a { color: #93c5fd; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header (sticky) */
.header {
  position: sticky; top: 0; z-index: 100;
  padding: 16px 14px 14px 14px; margin: -16px -16px 18px -16px;
  border-bottom: 1px solid var(--border);
  background:
    radial-gradient(70% 120% at 10% -20%, rgba(34,211,238,.18) 0%, transparent 60%),
    radial-gradient(60% 120% at 100% 0%, rgba(139,92,246,.15) 0%, transparent 60%),
    linear-gradient(180deg, #0b0f19cc 0%, #0b0f19f0 100%);
  backdrop-filter: blur(8px);
}
.title { font-weight: 800; font-size: 21px; letter-spacing:.2px; }
.badge {
  display:inline-flex; align-items:center; gap:8px; font-size:12px; color: var(--ok);
  background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.28);
  padding: 4px 10px; border-radius: 999px;
}

/* Cards & chips */
.card {
  background: linear-gradient(180deg,var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid var(--border); border-radius: 16px; padding: 16px;
  box-shadow: 0 10px 40px -30px rgba(0,0,0,.9);
}
.ghost { background: linear-gradient(180deg,#0b1325 0%, #0b0f19 100%); }
.small { padding: 12px; border-radius: 14px; }

.chips { display:flex; gap:10px; overflow-x:auto; padding-bottom:4px; }
.chip {
  flex: 0 0 auto; border: 1px solid var(--border); border-radius: 999px;
  padding: 8px 12px; font-size: 13px; color: var(--text); background: #0d1426; cursor: pointer;
}
.chip:hover { outline: 2px solid var(--ring); outline-offset: 0; }
.tag {
  display:inline-flex; align-items:center; border: 1px solid var(--border);
  color: var(--muted); background:#0b1322; border-radius: 999px;
  font-size: 12px; padding: 6px 10px; margin: 4px 6px 0 0;
}

/* Chat bubbles */
.answer { border:1px solid var(--border); border-radius: 14px; padding: 16px; background: #0c1427; }
.user { border:1px solid var(--border); border-radius: 14px; padding: 12px 14px; background:#0b1222; }

/* Doc area */
.doc-meta { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.doc-path { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--muted); }
.expander > div { border:1px solid var(--border) !important; border-radius: 14px; }

/* Section titles */
.h-section { font-weight: 800; font-size: 18px; margin: 8px 0 4px; }
.subtle { color: var(--muted); }

/* Footer */
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================
# Constants
# ==============================
CANDIDATES: List[str] = [
    "에코프로비엠 위험성을 요약해줘",
    "에코프로비엠 위험지표를 알려줘",
    "에코프로비엠에 대한 투자 의견을 알려줘",
    "에코프로비엠에 대한 주가를 알려줘",
    "에코프로비엠 관련 뉴스를 알려줘",
    "현재 에코프로비엠 30주를 가지고 있는데, 헷징하려면 어떻게 해야할 지 알려줘",
]

MIME_MAP = {
    ".pdf":  "application/pdf",
    ".json": "application/json; charset=utf-8",
    ".txt":  "text/plain; charset=utf-8",
    ".md":   "text/markdown; charset=utf-8",
    ".csv":  "text/csv; charset=utf-8",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
mime_map = MIME_MAP  # alias

DATA_ROOT = Path(__file__).resolve().parent / "data"
VALID_EXTS = {".pdf", ".json", ".txt", ".md", ".csv", ".png", ".jpg", ".jpeg", ".webp"}


# ==============================
# Backend (cached)
# ==============================
@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    rag = RAGPipeline()
    try:
        rag.finalize()
    except Exception:
        pass
    return rag

rag = load_pipeline()


# ==============================
# Data index for ./data/**
# ==============================
@st.cache_resource(show_spinner=False)
def build_data_index(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if not root.exists(): return idx
    for r, _, files in os.walk(root):
        for fn in files:
            p = Path(r) / fn
            if p.suffix.lower() in VALID_EXTS:
                idx[p.stem] = p
    return idx

DATA_INDEX = build_data_index(DATA_ROOT)

def base_id_from_doc_id(doc_id: str) -> str:
    m = re.match(r"^(.*?)(?:_\d+_\d+)$", doc_id)  # trim _page_chunk
    return m.group(1) if m else doc_id

def resolve_source_path(doc_id: str) -> Optional[Path]:
    if not DATA_INDEX: return None
    base_id = base_id_from_doc_id(doc_id)
    if base_id in DATA_INDEX: return DATA_INDEX[base_id]
    cands = [p for stem, p in DATA_INDEX.items() if stem.startswith(base_id)]
    return sorted(cands, key=lambda p: len(p.name))[0] if cands else None


# ==============================
# Render helpers
# ==============================
def render_pdf_inline(src_path: Path, height: int = 820, key: str = "pdf-0"):
    """Prefer pdf.js component; fallback to data-URL if not available."""
    size_mb = src_path.stat().st_size / (1024 * 1024)
    st.caption(f"파일 크기: {size_mb:.2f} MB")
    data = src_path.read_bytes()

    if HAS_PDFJS:
        try:
            pdf_viewer(data, width=0, height=height, scrolling=True, key=key)
        except TypeError:
            pdf_viewer(data, key=key)
        return

    # Fallback (can be blank on some Chromes for huge files)
    b64 = base64.b64encode(data).decode("utf-8")
    html = f"""
    <object data="data:application/pdf;base64,{b64}#view=FitH"
            type="application/pdf" width="100%" height="{height}">
      <iframe src="data:application/pdf;base64,{b64}#view=FitH"
              width="100%" height="{height}" style="border:none;"></iframe>
    </object>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_source_file(doc_id: str, src_path: Path, row_index: int, allow_downloads: bool) -> None:
    st.markdown(
        f"<div class='doc-meta'>원문 파일 · <span class='doc-path'>{src_path}</span></div>",
        unsafe_allow_html=True,
    )
    ext = src_path.suffix.lower()

    if ext == ".pdf":
        render_pdf_inline(src_path, height=820, key=f"pdf-{row_index}-{src_path.name}")
    elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
        st.image(src_path.as_posix(), use_column_width=True)
    elif ext in {".txt", ".md"}:
        try:
            content = src_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = src_path.read_text(errors="ignore")
        st.text_area("원문 (텍스트)", value=content, height=350)
    elif ext == ".json":
        try:
            st.json(json.loads(src_path.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            st.text_area("원문 (JSON)", value=src_path.read_text(errors="ignore"), height=350)
    elif ext == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(src_path)
            st.dataframe(df, use_container_width=True, height=350)
        except Exception:
            st.text_area("원문 (CSV)", value=src_path.read_text(errors="ignore"), height=350)
    else:
        st.info("미리보기를 지원하지 않는 형식입니다.")

    if allow_downloads:
        data = src_path.read_bytes()
        mime = MIME_MAP.get(ext, "application/octet-stream")
        st.download_button(
            "다운로드",
            data=data,
            file_name=src_path.name,
            mime=mime,
            key=f"dl-{row_index}-{src_path.name}",  # unique per row
        )


# ==============================
# Session state
# ==============================
if "qa_history" not in st.session_state:
    st.session_state.qa_history: List[Dict[str, str]] = []
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

def add_exchange(q: str, a: str):
    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.messages.append({"role": "assistant", "content": a})
    st.session_state.qa_history.append({"question": q, "answer": a})


# ==============================
# Header
# ==============================
st.markdown(
    """
    <div class="header">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div class="title">FinAgent · 위험 분석 챗봇</div>
        <div class="badge"><span style="width:8px;height:8px;border-radius:99px;background:#10b981;display:inline-block;"></span> Backend Ready</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================
# Fancy hero (3-card grid like your reference, dark)
# ==============================
c1, c2, c3 = st.columns([1.2, 1.0, 1.2], gap="large")
with c1:
    st.markdown(
        """
        <div class="card">
          <div style="font-weight:800;font-size:18px;margin-bottom:8px;">법률·금융 전문가의 정교함,<br/>AI로 구현할 수 있을까?</div>
          <div class="subtle" style="margin-bottom:14px;">RAG에 기반한 문서 정밀 검색으로 복잡한 쿼리에도 신뢰도 높은 답을 제공합니다.</div>
          <div style="display:flex;gap:10px;">
            <div class="chip">아티클 바로가기 ↗</div>
            <div class="chip">제품 소개</div>
          </div>
          <div style="height:8px;"></div>
          <div class="ghost small" style="text-align:center; padding:12px;">🔎 정밀 검색 엔진 · 하이브리드(BM25 + Dense) · 문서 인용</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="card">
          <div style="font-weight:800;margin-bottom:10px;">정밀 검색</div>
          <div class="chips">
            <div class="chip">⚡️ Generate</div>
            <div class="chip">🔎 Search</div>
            <div class="chip">❓ 도움말</div>
          </div>
          <div class="subtle" style="margin-top:8px;">질문을 한 번에 이해하고, 관련 문서와 함께 답변합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="card">
          <div style="font-weight:800;margin-bottom:8px;">다양한 자료 소스 기반</div>
          <div>
            <span class="tag">정책자료</span><span class="tag">보고서</span><span class="tag">학술자료</span>
            <span class="tag">법령</span><span class="tag">판례</span><span class="tag">의견서</span><span class="tag">유권해석</span>
            <span class="tag">금융</span><span class="tag">노동</span><span class="tag">공정거래</span><span class="tag">IP</span>
            <span class="tag">개인정보</span><span class="tag">보험</span><span class="tag">기업공시</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")  # spacing


# ==============================
# Candidate queries (pills)
# ==============================
st.markdown('<div class="h-section">💡 예시 질문</div>', unsafe_allow_html=True)
st.markdown('<div class="chips">', unsafe_allow_html=True)
for i, q in enumerate(CANDIDATES):
    if st.button(q, key=f"chip-{i}"):
        st.session_state.pending_query = q
st.markdown("</div>", unsafe_allow_html=True)
st.divider()


# ==============================
# Sidebar (controls)
# ==============================
with st.sidebar:
    st.markdown("### ⚙️ 표시 설정")
    show_docs = st.toggle("참고 문서 표시", value=True)
    topk = st.slider("표시할 문서 개수", 1, 10, 7)
    show_matched_excerpt = st.toggle("매칭된 발췌(Chunk) 표시", value=False)
    allow_downloads = st.toggle("다운로드 버튼 표시", value=False)
    st.divider()
    if st.button("🧹 대화 지우기"):
        st.session_state.messages = []
        st.session_state.qa_history = []
        st.rerun()
    st.caption("데이터는 이미 ingest 완료. 서버 시작 시 인메모리 인덱스만 재구성합니다.")
    if not DATA_INDEX:
        st.warning("`./data` 폴더를 찾지 못했어요. 원문 미리보기가 제한될 수 있어요.")


# ==============================
# Render history (chat bubbles)
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(f"<div class='answer'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='user'>{msg['content']}</div>", unsafe_allow_html=True)


# ==============================
# Chat input & inference
# ==============================
user_query = st.chat_input("무엇이 궁금하신가요?")
if "pending_query" in st.session_state and not user_query:
    user_query = st.session_state.pop("pending_query")

if user_query:
    with st.chat_message("user"):
        st.markdown(f"<div class='user'>{user_query}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("생각중…"):
            history_for_backend = st.session_state.qa_history[-4:]
            try:
                answer_text, sources = rag.answer_with_sources(
                    query=user_query, history=history_for_backend
                )
            except Exception as e:
                answer_text, sources = (f"오류가 발생했습니다: {e}", [])

        st.markdown(f"<div class='answer'>{answer_text}</div>", unsafe_allow_html=True)

        if show_docs:
            st.markdown('<div class="h-section">📎 참고 문서</div>', unsafe_allow_html=True)
            if not sources:
                st.caption("문서가 없습니다.")
            else:
                for i, (doc_id, chunk_text) in enumerate(sources[:topk], start=1):
                    with st.expander(f"[{i}] {doc_id}", expanded=False):
                        src_path = resolve_source_path(doc_id)
                        if src_path and src_path.exists():
                            tabs = st.tabs(["원문 보기", "매칭된 발췌"])
                            with tabs[0]:
                                render_source_file(doc_id, src_path, i, allow_downloads)
                            with tabs[1]:
                                st.code(chunk_text, language=None)
                        else:
                            st.warning("원문 파일을 ./data/** 에서 찾지 못했습니다.")

    add_exchange(user_query, answer_text)

st.markdown("<div class='note'>© 2025 FinAgent · Internal demo</div>", unsafe_allow_html=True)
