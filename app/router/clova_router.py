# -*- coding: utf-8 -*-
import json
import http.client
from http import HTTPStatus
import uuid
from typing import Dict, Any
from app.config import settings  # Import settings from config.py
import ssl

class Executor:
    """
    Executor for CLOVA Studio Router API with RAG intent routing.
    Uses configuration from app.config.settings.
    """
    def __init__(self, host: str = "clovastudio.stream.ntruss.com", router_id: str = "yl0rl4cd", version: str = "15"):
        self._host = host
        self._api_key = settings.HYPERCLOVA_API_KEY if settings.HYPERCLOVA_API_KEY.startswith('Bearer ') \
            else f"Bearer {settings.HYPERCLOVA_API_KEY}"
        self._request_id = str(uuid.uuid4())
        self._router_id = router_id
        self._version = version

    def _send_request(self, request: Dict[str, Any]) -> tuple[Dict, int]:
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
        }
        conn = http.client.HTTPSConnection(self._host, context=ssl._create_unverified_context())
        conn.request(
            'POST',
            f'/v1/routers/{self._router_id}/versions/{self._version}/route',
            json.dumps(request),
            headers
        )
        response = conn.getresponse()
        status = response.status
        result = json.loads(response.read().decode('utf-8'))
        conn.close()
        return result, status

    def _classify_rag_intent(self, query: str, chat_history: list[Dict[str, Any]]) -> str:
        """
        Classify query intent for RAG routing.
        Returns one of: 위험 지표, 종목 위험 분석, 최신 종목 뉴스, 일반 검색.
        """
        keywords = {
            "위험 지표": ["변동성", "베타", "지표"],
            "종목 위험 분석": ["위험", "리스크", "분석"],
            "최신 종목 뉴스": ["뉴스", "최신", "정보"],
            "일반 검색": []
        }
        query_lower = query.lower()
        for domain, keyword_list in keywords.items():
            if any(keyword in query_lower for keyword in keyword_list):
                return domain
        return "일반 검색"  # Fallback to 일반 검색 if no specific intent detected

    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute router request with RAG intent override if needed.
        """
        query = request.get("query", "")
        chat_history = request.get("chatHistory", [])
        
        # Get initial domain from CLOVA API
        res, status = self._send_request(request)
        if status != HTTPStatus.OK:
            return {"error": f"Router request failed with status {status}: {res}"}

        # Override with RAG intent if CLOVA domain is generic or mismatched
        clova_domain = res.get("result", {}).get("domain", "")
        rag_domain = self._classify_rag_intent(query, chat_history)
        if clova_domain not in ["위험 지표", "종목 위험 분석", "최신 종목 뉴스", "일반 검색"]:
            res["result"]["domain"] = rag_domain
        else:
            res["result"]["domain"] = clova_domain

        return res["result"]

if __name__ == '__main__':
    request_data = {
        "query": "에코프로비엠 위험성을 요약해줘",
        "chatHistory": [{"role": "user", "content": "내일 서울의 강수 예보 알려줘"}]
    }
    executor = Executor()
    response_data = executor.execute(request_data)
    print(response_data)