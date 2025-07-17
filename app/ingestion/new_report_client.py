# report_client.py
# 일단 테스트로 24개 가져옴
import os
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

def get_reports_for_stock(stock_name: str, max_pages: int = 3, download_dir="pdfs"):
    base = "https://finance.naver.com"
    list_url = base + "/research/company_list.naver"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }

    # 저장 폴더 준비
    os.makedirs(download_dir, exist_ok=True)

    for page in range(1, max_pages + 1):
        # 종목명 EUC-KR 인코딩
        euc_keyword = quote_plus(stock_name, encoding="euc-kr")
        url = (
            f"{list_url}"
            f"?searchType=keyword"
            f"&keyword={euc_keyword}"
            f"&brokerCode="
            f"&writeFromDate="
            f"&writeToDate="
            f"&itemName="
            f"&itemCode="
            f"&page={page}"
        )
        print(f"[DEBUG] 요청 URL → {url}")

        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"[WARN] page {page} 상태 코드 {resp.status_code}, 스킵")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        box = soup.find("div", class_="box_type_m")
        if not box:
            print(f"[WARN] page {page} box_type_m 없음, 스킵")
            continue

        table = box.find("table")
        rows = table.find_all("tr")[2:]  # 헤더 두 줄 제거
        print(f"[DEBUG] page {page} 데이터 행: {len(rows)}개")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            company = cols[0].get_text(strip=True)
            if company != stock_name:
                continue

            title = cols[1].get_text(strip=True)
            firm  = cols[2].get_text(strip=True)
            # 첨부 컬럼은 cols[3]
            attach_tag = cols[3].find("a")
            date  = cols[4].get_text(strip=True)

            if not attach_tag:
                print(f"[INFO] {date} | {firm} | '{title}' — 첨부 없음")
                continue

            pdf_path = attach_tag["href"]
            # href가 절대경로(http)면 그대로, 아니면 base prefix
            if pdf_path.startswith("http"):
                pdf_url = pdf_path
            else:
                pdf_url = base + pdf_path

            print(f"[INFO] {date} | {firm} | '{title}' — 다운로드: {pdf_url}")

            # 파일명에 쓰기 불가능 문자는 제거
            safe_title = "".join(c for c in title if c.isalnum() or c in " _-")
            filename = f"{stock_name}_{firm}_{date}_{safe_title}.pdf"
            filepath = os.path.join(download_dir, filename)

            # 이미 있으면 스킵
            if os.path.exists(filepath):
                print(f"       → 이미 존재, 스킵: {filename}\n")
                continue

            # PDF 내려받기
            pdf_resp = requests.get(pdf_url, headers=headers)
            if pdf_resp.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(pdf_resp.content)
                print(f"       → 저장 완료: {filename}\n")
            else:
                print(f"       → 다운로드 실패 ({pdf_resp.status_code})\n")


if __name__ == "__main__":
    get_reports_for_stock("삼성전자", max_pages=3, download_dir="pdfs")
