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
    STOCK_NAME = "에코프로비엠"
    START_DATE = "2025-06-20"
    END_DATE   = "2025-07-20"
    SAVE_DIR   = os.path.join("data", "hankyung_pdfs")

    fetch_and_download_reports(STOCK_NAME, START_DATE, END_DATE, SAVE_DIR)
