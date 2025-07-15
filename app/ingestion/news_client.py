# ingestion/news_client.py

import os
import requests
from dotenv import load_dotenv

# .env 파일에서 NAVER API 키 불러오기
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

def search_news(query: str, display: int = 5, sort: str = "date") -> list[dict]:
    """
    네이버 뉴스 검색 API를 통해 뉴스 기사 리스트를 반환합니다.
    
    Parameters:
        query: 검색 키워드 (예: "삼성전자")
        display: 가져올 뉴스 기사 수 (최대 100)
        sort: 정렬 방식 ('date' 또는 'sim')
    
    Returns:
        뉴스 기사 딕셔너리 리스트
    """
    url = "https://openapi.naver.com/v1/search/news.json"

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }

    params = {
        "query": query,
        "display": display,
        "sort": sort
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"[API 오류] {response.status_code}: {response.text}")

    return response.json().get("items", [])

# === 테스트 실행 ===
if __name__ == "__main__":
    query = "삼성전자"
    results = search_news(query=query, display=5, sort="date")
    
    print(f"🔍 '{query}' 관련 최근 뉴스 기사:")
    for i, item in enumerate(results, start=1):
        print(f"\n[{i}] {item['title']}")
        print(f"📅 날짜: {item['pubDate']}")
        print(f"📰 링크: {item['link']}")
