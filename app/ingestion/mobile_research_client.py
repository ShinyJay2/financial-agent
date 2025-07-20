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
    STOCK_CODE = "005930"                 # 종목 코드
    DAYS_BACK  = 365                      # 과거 며칠치 리포트
    PAGE_SIZE  = 20                       # 페이지당 아이템 수
    OUT_DIR    = os.path.join("data", "naver_pdfs") #파일 경로 설정
        # macOS 결과:  "data/naver_pdfs" # Windows 결과: "data\\naver_pdfs"

    download_reports(STOCK_CODE, DAYS_BACK, PAGE_SIZE, OUT_DIR)

#page_size 변수 정보
    # 한 번의 API 호출로 가져올 리포트 개수를 정하는 파라미터예요.

    # 예시: page_size=20 으로 설정하면
    # /api/research/stock/{code}?page=1&pageSize=20 호출 시 최대 20건의 리포트 메타정보를 내려받고,
    # 그 다음에 page=2 로 넘어가면 또 20건을 가져오는 식이죠.

    # 이렇게 하면 한 페이지당 몇 건씩 처리할지, 응답 크기와 호출 횟수 사이에서 조절할 수 있습니다.
    # 작게 설정하면(예: 10) 응답이 가벼워서 빠르지만, 전체를 다 모으려면 호출 횟수가 늘어납니다.
    # 크게 설정하면(예: 50) 호출 횟수는 줄어들지만, 한 번에 내려받는 데이터 양이 많아져 응답이 무거워질 수 있습니다.

    # 대부분은 API에서 권장하는 최대치(보통 20~50) 정도로 두고 사용하시면 적절합니다.


## 아래는 get_recent_reports 함수 원본, 위에 코드가 더 깔끔해진 ver
# import requests

# def get_mobile_research(stock_code: str, page_size: int = 20):
# # 여기서 stock_code 가 종목번호임
# # page_size: 한 번에 가져올 개수 page_size (기본 20) 만큼씩 모든 페이지를 순회
#     """
#     m.stock.naver.com 모바일 API를 통해
#     해당 종목의 리서치 항목을 전부 가져오는 것임
#     """
#     base_url = "https://m.stock.naver.com"
#     api_path = f"/api/research/stock/{stock_code}"
#     url = base_url + api_path  
#     #전체 요청 url = https://m.stock.naver.com/api/research/stock/005930

#     all_reports = []
#     page = 1 #현재 페이지 번호. '더보기' 누를 때마다 1 2 3 으로 증가

#     while True:
#         params = {"pageSize": page_size, "page": page}
# # 실제 api 주소가 url + ?pageSize=20&page=1 형식임
#         resp = requests.get(url, params=params) #api 호출
#         resp.raise_for_status() #HTTP 오류 있으면 예외 발생

#         data = resp.json() # 응답 결과를 json으로 파싱해서 담는것
#         # ────────────────────────────────────────────────
#         # 디버깅: data 가 dict 인지 list 인지 찍어 봅니다
#         print(f"[DEBUG] page={page} → data 타입: {type(data)}")
#         if isinstance(data, dict):
#             print("       최상위 키:", list(data.keys()))
#         else:
#             print("       리스트 길이:", len(data))
#         # ────────────────────────────────────────────────

#         # 1) data 가 list 로 내려오면 그게 곧 reports
#         if isinstance(data, list):
#             reports = data
#             # 페이지 정보를 얻을 수 없으니, length < page_size 시 종료
#             # 아래 종료조건에서 if has_next 있음.
#             has_next = len(reports) == page_size
#         else: #여기 웹에서는 다 리스트로 뽑히니까 else문은 굳이 고려 안해도됨
#             # 2) data 가 dict 면 wrapper 벗기기
#             result = data.get("result", data)
#             # 실제 리스트 필드 이름 뽑기
#             reports = (
#                 result.get("list")
#                 or result.get("research")
#                 or result.get("items")
#                 or []
#             )
#             # 페이지 정보가 있으면 hasNext 사용
#             pageable = result.get("pageable", {})
#             has_next = pageable.get("hasNext", len(reports) == page_size)

#         if not reports:
#             print(f"[DEBUG] page={page} → 결과 없음, 중단")
#             break

#         # 3) 리포트 정보 수집

# # 이거 print 작업 남겨두자
#     #이걸로 api에서 받아온 애들 자세한 값 뭔지 파악하는 것임
#         # if page == 1 and reports:
#         #     print("샘플 rpt raw:")
#         #     print(reports[0])
#         #     print("샘플 키 목록:", list(reports[0].keys()))

#         # 실제 데이터 수집
#         for rpt in reports:
#             title   = rpt["title"]             # 리포트 제목
#             company = rpt["brokerName"]        # 증권사 이름 (brokerName)
#             date    = rpt["writeDate"]         # 작성일 (writeDate)
#             views   = int(rpt["readCount"])    # 조회수 (문자열 → 정수 변환)
#             # 연구 상세 페이지 URL 조립
#             research_id = rpt["researchId"]
#             url_detail  = f"{base_url}/domestic/stock/{stock_code}/research/{research_id}"

#             all_reports.append({
#                 "title":   title,
#                 "company": company,
#                 "date":    date,
#                 "views":   views,
#                 "url":     url_detail
#             })

#         # ⚠️ 최대 개수 도달 시 모든 루프 탈출
#         if len(all_reports) >= 30:
#             print(f"[DEBUG] 수집 개수 30개 도달, 중단")
#             return all_reports
        
#         # 4) 다음 페이지 판단
#         if not has_next:
#             print(f"[DEBUG] page={page} → 더보기 없음, 중단")
#             break
#         page += 1

#     return all_reports

# if __name__ == "__main__":
#     reports = get_mobile_research("005930", page_size=20)
#     print(f"총 수집된 리포트: {len(reports)}건\n")
#     for r in reports:
#         print(f"{r['date']} | {r['company']} | {r['title']} → {r['url']}")

