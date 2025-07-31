#!/usr/bin/env python3
# scripts/test_numbers.py

import os
import sys
import shutil

# 1) Determine repo root (three levels up from this file)
project_root = os.path.abspath(
    os.path.join(__file__, "..", "..", "..")
)

# 2) Compute the absolute path for the ChromaDB folder,
#    relative to the repo root if necessary
persist_dir = os.getenv("CHROMA_DB_DIR", "chroma_db")
if not os.path.isabs(persist_dir):
    persist_dir = os.path.join(project_root, persist_dir)
persist_dir = os.path.abspath(persist_dir)

# after: from app.rag_pipeline import RAGPipeline
from pathlib import Path

def wipe_json_entries(collection, data_dir: str):
    """
    Delete from ChromaDB only the entries that came from JSON files.
    """
    # 1) Gather all JSON file-stems
    json_stems = {Path(f).stem for f in os.listdir(data_dir) if f.lower().endswith(".json")}
    if not json_stems:
        return

    # 2) Pull every ID currently in the collection
    all_ids = collection.get(ids=None)["ids"]

    # 3) Find those whose ID starts with any JSON stem
    to_delete = [did for did in all_ids if any(did.startswith(stem) for stem in json_stems)]
    if to_delete:
        print(f"🧹 Deleting {len(to_delete)} JSON‐based docs from ChromaDB…")
        # 4) Delete them
        collection.delete(ids=to_delete)


# 3) Optionally wipe the store on demand to force re-ingest
force = "--force" in sys.argv
if force and os.path.isdir(persist_dir):
    print(f"🧹 [force] Deleting old ChromaDB at: {persist_dir}")
    shutil.rmtree(persist_dir)

# 4) Ensure RAGPipeline’s PersistentClient uses this exact folder
os.environ["CHROMA_DB_DIR"] = persist_dir
print(f"✅ Using ChromaDB directory: {persist_dir}")

# 5) Detect whether we need to ingest/embed
db_is_empty = not os.path.isdir(persist_dir) or not os.listdir(persist_dir)

# 6) Now import the pipeline (which will load PersistentClient)
import json
from datetime import datetime
from app.rag_pipeline import RAGPipeline

# 7) Locate the data folder under the repo root
data_root = os.path.join(project_root, "data")
if not os.path.isdir(data_root):
    print(f"❌ Data directory not found: {data_root}")
    sys.exit(1)


def main():
    # 8) Instantiate the RAG pipeline
    rag = RAGPipeline(
        chunk_method="section",
        bm25_k=20,
        dense_k=20,
        final_k=10,
        max_tokens=4096,
        overlap=50,
    )

    #if force or db_is_empty:
        # wipe only JSON docs so you can re-ingest them under the new chunker
    #from app.retrieval.vectorstore import dense_collection, memory_collection
    #wipe_json_entries(dense_collection, data_root)
    #wipe_json_entries(memory_collection, data_root)


    # 9) Ingest documents only if the vector store is empty (or forced)
    if db_is_empty:
        print("🧹 Ingesting documents (this will call the embedding model once)…")
        for dirpath, _, filenames in os.walk(data_root):
            for fname in filenames:
                if fname.startswith("."):
                    continue
                file_path = os.path.join(dirpath, fname)
                print(f"=== Ingesting {file_path} ===")
                rag.ingest_file(file_path)
    else:
        print("✅ Vector store already exists—skipping ingest/embedding.")

        # 9) Re-ingest only JSON files (old JSON entries were just wiped)
    #print("🧹 Re-ingesting JSON files…")
    #for dirpath, _, filenames in os.walk(data_root):
        #for fname in filenames:
            #if not fname.lower().endswith(".json"):
                #continue
            #file_path = os.path.join(dirpath, fname)
            #print(f"=== Ingesting JSON: {file_path} ===")
            #rag.ingest_file(file_path)


    # 10) Build BM25 index, hybrid structures, etc.
    print("🚧 Building BM25 and hybrid indexes…")
    rag.finalize()

    # 11) Define and run queries
    queries = [
        "에코프로비엠 위험성을 요약해줘",
        "에코프로비엠 위험지표를 알려줘",
        "에코프로비엠에 대한 투자 의견을 알려줘",
        "에코프로비엠에 대한 주가를 알려줘",
        "에코프로비엠 관련 뉴스를 알려줘",
        "현재 에코프로비엠 30주를 가지고 있는데, 헷징하려면 어떻게 해야할 지 알려줘"
    ]

    results = {
        "generated_at": datetime.now().isoformat(),
        "queries": []
    }

    for q in queries:
        print(f"❓ Q: {q}")
        ans = rag.answer(q)
        print(f"💡 A: {ans}\n")
        results["queries"].append({"question": q, "answer": ans})

    # 12) Write results to JSON
    out_path = os.path.join(project_root, "rag_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Written updated JSON to {out_path}")

    # 13) Exit cleanly so PersistentClient auto-persists the HNSW index


if __name__ == "__main__":
    main()
