import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

def get_reports_for_stock(stock_name: str, max_pages: int = 3):
    base = "https://finance.naver.com/research/company_list.naver"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }

    results = []
    for page in range(1, max_pages + 1):
        # 1) 종목명을 EUC-KR로 인코딩
        euc_keyword = quote_plus(stock_name, encoding="euc-kr")
        # 2) 직접 URL 조립
        url = (
            f"{base}"
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
        print(f"[DEBUG]   상태 코드: {resp.status_code}")
        if resp.status_code != 200:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        box = soup.find("div", class_="box_type_m")
        if not box:
            print(f"[WARN] box_type_m 못 찾음 (page={page})")
            continue

        table = box.find("table")
        rows = table.find_all("tr")[2:]  # 헤더 2행 skip
        print(f"[DEBUG]   데이터 행 개수: {len(rows)}")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            company = cols[0].get_text(strip=True)
            if company != stock_name:
                continue
            title = cols[1].get_text(strip=True)
            firm  = cols[2].get_text(strip=True)
            date  = cols[3].get_text(strip=True)
            link  = "https://finance.naver.com" + cols[1].find("a")["href"]
            print(f"[INFO] {date} | {firm} | {title}\n ")

            results.append({
                "종목명": company,
                "제목":  title,
                "증권사": firm,
                "작성일": date,
                "링크":  link
            })

    print(f"[DEBUG] 총 수집: {len(results)} 개\n")
    return results

if __name__ == "__main__":
    get_reports_for_stock("삼성전자", max_pages=3)
