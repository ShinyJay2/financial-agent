#!/usr/bin/env python3
# scripts/test_rag.py

import os
import sys
import json
from datetime import datetime
from app.rag_pipeline import RAGPipeline

import os, sys, shutil

# Pre‐import wipe of the on‐disk ChromaDB folder
project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
chroma_dir   = os.getenv("CHROMA_DB_DIR", "./chroma_db")
db_path      = os.path.join(project_root, chroma_dir)
if os.path.isdir(db_path):
    print("🧹 Pre-import wipe of old vectorstore…")
    shutil.rmtree(db_path)


def main():
    # locate directories
    base_dir     = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    data_root    = os.path.join(project_root, "data")
    out_path     = os.path.join(project_root, "rag_output.json")

    if not os.path.isdir(data_root):
        print(f"❌ Data directory not found: {data_root}")
        sys.exit(1)

    # instantiate pipeline
    rag = RAGPipeline(
        chunk_method="section",
        bm25_k=20,
        dense_k=20,
        final_k=10,
        max_tokens=4096,
        overlap=50,
    )

    # define queries
    queries = [
        "에코프로비엠 위험성을 요약해줘",
        "에코프로비엠 변동성과 베타 값을 알려줘",
        "에코프로비엠에 대한 투자 의견을 알려줘",
    ]

    # ingest all files
    for dirpath, _, filenames in os.walk(data_root):
        for fname in filenames:
            if fname.startswith("."):
                continue
            file_path = os.path.join(dirpath, fname)
            print(f"\n=== Ingesting {file_path} ===")
            rag.ingest_file(file_path)

    # build BM25 index once
    print("\n🚧 Building BM25 index…")
    rag.finalize()

    # run queries and collect results
    results = {
        "queries": [],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    for q in queries:
        print(f"\n❓ Q: {q}")
        ans = rag.answer(q)
        print(f"💡 A: {ans}")
        results["queries"].append({
            "question": q,
            "answer": ans
        })

    # write out updated JSON in project root
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Written updated JSON to {out_path}")

if __name__ == "__main__":
    main()
