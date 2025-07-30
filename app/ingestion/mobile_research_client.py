# app/ingestion/mobile_research_client.py

import os
import requests
from datetime import datetime, timedelta

def get_recent_reports(stock_code: str,
                       days_back: int = 365,
                       page_size: int = 20):
    """
    과거 days_back일 이내의 리서치 항목(researchId, writeDate, title)을
    전부 수집해 리스트로 반환합니다.
    """
    base_url  = "https://m.stock.naver.com"
    endpoint  = f"{base_url}/api/research/stock/{stock_code}"
    threshold = datetime.now() - timedelta(days=days_back)
    reports   = []
    page      = 1

    while True:
        resp = requests.get(endpoint,
                            params={"pageSize": page_size, "page": page},
                            timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            items    = data
            has_next = len(items) == page_size
        else:
            wrapper  = data.get("result", {})
            items    = wrapper.get("list", [])
            has_next = wrapper.get("pageable", {}).get("hasNext", False)

        if not items:
            break

        for item in items:
            dt = datetime.strptime(item["writeDate"], "%Y-%m-%d")
            if dt < threshold:
                return reports

            reports.append({
                "research_id": item["researchId"],
                "date":        item["writeDate"],
                "title":       item["title"]
            })

        if not has_next:
            break
        page += 1

    return reports

def fetch_pdf_via_api(stock_code: str,
                      research_id: int,
                      timeout: int = 10) -> bytes:
    """
    디테일 API에서 attachUrl을 꺼내 PDF를 다운로드하여 바이트로 반환합니다.
    """
    api_url = f"https://m.stock.naver.com/api/research/stock/{stock_code}/{research_id}"
    headers = {
        "Accept":  "application/json",
        "Referer": f"https://m.stock.naver.com/domestic/stock/{stock_code}/research/{research_id}",
    }
    resp = requests.get(api_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    content = data["researchContent"]
    attach  = content.get("attachUrl")
    if not attach:
        raise RuntimeError(f"attachUrl 필드가 없습니다: {list(content.keys())}")

    pdf_url = attach if attach.startswith("http") else "https://m.stock.naver.com" + attach
    pdf_resp = requests.get(pdf_url, timeout=timeout)
    pdf_resp.raise_for_status()
    return pdf_resp.content

def download_reports(stock_code: str,
                     days_back: int,
                     page_size: int,
                     out_dir: str):
    """
    지정된 종목의 과거 days_back일 이내 리포트를 전부 다운로드하여
    out_dir에 저장합니다.
    """
    os.makedirs(out_dir, exist_ok=True)
    reports = get_recent_reports(stock_code, days_back=days_back, page_size=page_size)
    print(f"수집할 리포트 개수: {len(reports)}")

    for rpt in reports:
        rid  = rpt["research_id"]
        date = rpt["date"]
        print(f"→ [{date}] researchId={rid} 다운로드 중…")
        pdf_bytes = fetch_pdf_via_api(stock_code, rid)

        filename = f"{date}_{stock_code}_{rid}.pdf"
        path     = os.path.join(out_dir, filename)
        with open(path, "wb") as f:
            f.write(pdf_bytes)
        print(f"   저장 완료: {path}")

if __name__ == "__main__":
    # 변수만 수정하여 사용
    STOCK_CODE = "247540"                 # 종목 코드
    DAYS_BACK  = 100                      # 과거 며칠치 리포트
    PAGE_SIZE  = 20                       # 페이지당 아이템 수
    OUT_DIR    = os.path.join("data", "naver_pdfs") #파일 경로 설정
        # macOS 결과:  "data/naver_pdfs" # Windows 결과: "data\\naver_pdfs"

    download_reports(STOCK_CODE, DAYS_BACK, PAGE_SIZE, OUT_DIR)

