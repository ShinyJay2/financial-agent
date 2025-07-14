#!/usr/bin/env python3
# scripts/test_pipeline.py

import os
import sys
import tempfile

# 1) Override Chroma dir for a clean test
os.environ["CHROMA_DB_DIR"] = tempfile.mkdtemp(prefix="chroma_test_")

# 2) Import & reload your vectorstore so it picks up the new env var
from importlib import reload
import app.retrieval.vectorstore as vs
reload(vs)

# 3) Chunker imports
from app.chunking.chunker import (
    chunk_file,
    semantic_section_chunk,
    topic_model_chunk,
    sliding_window_chunk,
)

def test_chunker():
    print("▶︎ Testing chunker…")
    xml_path = "data/20250704800844.xml"
    if not os.path.exists(xml_path):
        print(f"  ✗ missing test file: {xml_path!r}")
        sys.exit(1)

    raw = open(xml_path, encoding="utf-8", errors="ignore").read()

    # method #1: top-level dispatch
    chunks = chunk_file(xml_path, max_tokens=300, overlap=50)
    print(f"  • chunk_file → {len(chunks)} chunks")
    if not chunks:
        print("  ✗ chunk_file returned zero chunks")
        sys.exit(1)

    # method #2: each strategy individually
    s1 = semantic_section_chunk(raw)
    s2 = topic_model_chunk(raw, n_topics=3)
    s3 = sliding_window_chunk(raw[:1000], max_tokens=100, overlap=10)

    print(f"  • semantic_section_chunk → {len(s1)} chunks")
    print(f"  • topic_model_chunk         → {len(s2)} chunks")
    print(f"  • sliding_window_chunk      → {len(s3)} chunks")

    if not any((s1, s2, s3)):
        print("  ✗ all chunk strategies produced zero chunks")
        sys.exit(1)

def test_vectorstore():
    print("▶︎ Testing vectorstore (BM25 + hybrid)…")
    # upsert some toy documents
    docs = {
        "doc_risk": "This contains risk warning and important alert.",
        "doc_fin":  "Here is financial data: revenue, profit, balance sheet.",
        "doc_misc": "Just some text about cats, dogs, and other animals."
    }
    for _id, txt in docs.items():
        vs.upsert_document(_id, txt)

    # build BM25 index
    vs.build_bm25_index()
    bm25_top = vs.bm25_search("risk alert", k=2)
    print(f"  • BM25 top for 'risk alert': {bm25_top}")
    if "doc_risk" not in bm25_top:
        print("  ✗ BM25 did not retrieve the risk doc")
        sys.exit(1)

    # hybrid search
    hyb = vs.hybrid_search("revenue profit", k=2)
    print(f"  • hybrid top for 'revenue profit': {hyb}")
    if "doc_fin" not in hyb:
        print("  ✗ hybrid search did not retrieve the financial doc")
        sys.exit(1)

if __name__ == "__main__":
    print("\n=== RUNNING SMOKE TEST ===\n")
    test_chunker()
    test_vectorstore()
    print("\n✅ ALL TESTS PASSED\n")
