# app/ingestion/hankyung_client.py

import requests
from bs4 import BeautifulSoup
import os
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def sanitize_filename(s):
    return re.sub(r"[\\/*?\"<>|:]", "", s).strip()

def fetch_and_download_reports(stock_name: str,
                               start_date: str,
                               end_date: str,
                               out_dir: str,
                               page: int = 1,
                               page_size: int = 20):
    """
    한경 컨센서스에서 리포트 목록을 수집하고 PDF로 저장합니다.
    """
    os.makedirs(out_dir, exist_ok=True)

    base_url = "https://consensus.hankyung.com/analysis/list"
    params = {
        "sdate": start_date,
        "edate": end_date,
        "now_page": page,
        "search_text": stock_name,
        "pagenum": page_size
    }

    response = requests.get(base_url, params=params, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("table tbody tr")

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 5:
            continue

        date = cols[0].text.strip()
        category = cols[1].text.strip()
        title_cell = cols[2]
        title = title_cell.text.strip()
        author = cols[3].text.strip()
        publisher = cols[4].text.strip()

        a_tag = title_cell.find("a")
        pdf_url = "https://consensus.hankyung.com" + a_tag["href"] if a_tag and a_tag.has_attr("href") else None

        print(f"[{date}] ({category}) {title} - {author} / {publisher} → PDF: {pdf_url}")

        # PDF 다운로드
        if pdf_url:
            try:
                pdf_response = requests.get(pdf_url, headers=HEADERS)
                if pdf_response.status_code == 200:
                    filename = f"{date}_{sanitize_filename(title)}.pdf"
                    filepath = os.path.join(out_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(pdf_response.content)
                    print(f"✅ Saved: {filename}")
                else:
                    print(f"❌ Failed to download {pdf_url} (Status {pdf_response.status_code})")
            except Exception as e:
                print(f"⚠️ Error downloading PDF: {e}")

if __name__ == "__main__":
    STOCK_NAME = "삼성전자"
    START_DATE = "2025-06-20"
    END_DATE   = "2025-07-20"
    SAVE_DIR   = os.path.join("data", "hankyung_pdfs")

    fetch_and_download_reports(STOCK_NAME, START_DATE, END_DATE, SAVE_DIR)


### 아래가 pdf 추출 전에 했던 테스트 코드

# import requests
# from bs4 import BeautifulSoup

# # 검색 파라미터 설정
# base_url = "https://consensus.hankyung.com/analysis/list"
# params = {
#     "sdate": "2024-07-20",       # 시작일
#     "edate": "2025-07-20",       # 종료일
#     "now_page": 1,               # 첫 페이지
#     "search_text": "삼성전자",    # 종목명
#     "pagenum": 20                # 한 페이지에 보여질 리포트 수
# }

# # HTTP 요청
# headers = {
#     "User-Agent": "Mozilla/5.0"
# }
# response = requests.get(base_url, params=params, headers=headers)
# soup = BeautifulSoup(response.text, "html.parser")

# # 결과 확인용 추출 (첫 번째 테이블)
# rows = soup.select("table tbody tr")

# for row in rows:
#     cols = row.find_all("td")
#     if len(cols) < 5:
#         continue
#     작성일 = cols[0].text.strip()
#     분류 = cols[1].text.strip()

#     제목셀 = cols[2]
#     제목 = 제목셀.text.strip()
    
#     a_tag = 제목셀.find("a")
#     pdf_url = "https://consensus.hankyung.com" + a_tag["href"] if a_tag and a_tag.has_attr("href") else None

    
#     작성자 = cols[3].text.strip()
#     증권사 = cols[4].text.strip()
#     print(f"[{작성일}] ({분류}) {제목} - {작성자} / {증권사}/ PDF url: {pdf_url}")
