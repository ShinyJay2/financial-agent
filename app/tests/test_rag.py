#!/usr/bin/env python3
# scripts/test_rag.py

import os
import sys
from importlib import reload

# 1) Make sure your project root is on PYTHONPATH
#    If you run this as a module (python -m …) you can skip this,
#    otherwise uncomment the next two lines and adjust as needed:
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag_pipeline import RAGPipeline

def main():
    data_file = "data/20250704800844.xml"
    if not os.path.exists(data_file):
        print(f"❌ File not found: {data_file}")
        sys.exit(1)

    # 2) Read and parse the XML file
    import xml.etree.ElementTree as ET
    with open(data_file, "r", encoding="utf-8") as f:
        raw = f.read()
        try:
            root = ET.fromstring(raw)
            # Extract relevant sections from XML (customize based on your XML structure)
            raw_data = {}
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    raw_data[f"section_{elem.tag}"] = elem.text.strip()
            raw = "\n".join([f"{key}: {value}" for key, value in raw_data.items()])
        except ET.ParseError as e:
            print(f"⚠️ XML parsing error: {e}. Using raw text as fallback.")
            # Fallback to raw text if XML parsing fails
            raw = f.read() if 'f' in locals() else raw

    # 3) Instantiate and ingest
    rag = RAGPipeline(
        chunk_method="section",
        bm25_k=20,
        dense_k=20,
        final_k=5,
        max_tokens=300,
        overlap=50,
    )
    print("🗂  Chunking & upserting…")
    rag.ingest("testdoc", raw)

    # 4) Try a few queries
    queries = [
        "발행회사 정보 요약해줘",
        "보고일자와 소유주식 변동 개요 알려줘",
        "위험요소 섹션을 찾아 요약해줘",
    ]
    for q in queries:
        print("\n❓ Q:", q)
        ans = rag.answer(q)
        print("💡 A:", ans)

if __name__ == "__main__":
    main()