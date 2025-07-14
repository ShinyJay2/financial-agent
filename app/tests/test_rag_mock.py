#!/usr/bin/env python3
# scripts/test_rag_mock.py

import os
import sys
from importlib import reload
import json
from datetime import datetime

# 1) Make sure your project root is on PYTHONPATH
#    If you run this as a module (python -m …) you can skip this,
#    otherwise uncomment the next two lines and adjust as needed:
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag_pipeline import RAGPipeline

def main():
    # 2) Use mock data for multiple documents
    mock_data = {
        "doc_issuer_info": """
        2. 발행주식수 정보
        보통주식총수: 5,919,637,922
        종류주식총수: 815,974,664
        발행주식총수: 6,735,612,586
        최대주주: 이혁재 (8.51% 보통주식)
        생성일: 2025-07-13 09:40 AM KST
        """,
        "doc_report_overview": """
        3. 보고의 개요
        보고일자: 2025-06-04
        소유주식: 보통주식 1,193,049,232 (20.15%), 종류주식 906,408 (0.11%)
        이번보고서제출일: 2025-07-04
        소유주식: 보통주식 1,192,771,595 (20.15%), 종류주식 876,754 (0.11%)
        증감: 보통주식 -277,637, 종류주식 -29,654
        생성일: 2025-07-13 09:40 AM KST
        """,
        "doc_risk_factors": """
        4. 위험요소
        주요 위험: 시장 변동성, 주식 가치 하락 가능성, 경제적 불확실성
        생성일: 2025-07-13 09:40 AM KST
        """,
        "doc_shareholder_details": """
        5. 최대주주 세부사항
        성명: 삼성생명, 보유주식: 503,948,793 (7.48%)
        관계: 기타, 국적: 대한민국
        생성일: 2025-07-13 09:40 AM KST
        """
    }

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
    for doc_id, content in mock_data.items():
        rag.ingest(doc_id, content)

    # 4) Try a few queries and collect results
    queries = [
        "발행회사 정보 요약해줘",
        "보고일자와 소유주식 변동 개요 알려줘",
        "위험요소 섹션을 찾아 요약해줘",
    ]
    results = []
    for q in queries:
        print("\n❓ Q:", q)
        ans = rag.answer(q)
        print("💡 A:", ans)
        results.append({
            "question": q,
            "answer": ans if ans else "No relevant answer found."
        })

    # 5) Save results to JSON
    with open("rag_output.json", "w", encoding="utf-8") as f:
        json.dump({"queries": results, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M %Z")}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()