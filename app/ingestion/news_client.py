import os, json
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

def get_recent_news(stock_code: str,
                    days_back: int = 365,
                    page_size: int = 20):
    """
    과거 days_back일 이내의 뉴스 항목(id, datetime, title, body, officeName, imageOriginLink)을
    전부 수집해 리스트로 반환합니다.
    """
    base_url  = "https://m.stock.naver.com"
    endpoint  = f"{base_url}/api/news/stock/{stock_code}"
    threshold = datetime.now() - timedelta(days=days_back)
    news_list = []
    page      = 1

    while True:
        resp = requests.get(endpoint,
                            params={"pageSize": page_size, "page": page},
                            timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 페이지별로 내려온 블록들(data는 list of dict{ total, items })
        # -> 모든 items 배열을 하나로 합치기
        items = []
        for block in data:
            items.extend(block.get("items", []))

        if not items:
            break

        for item in items:
            # "202507211744" 형태를 datetime으로 변환
            dt = datetime.strptime(item["datetime"], "%Y%m%d%H%M")
            if dt < threshold:
                return news_list

            news_list.append({
                "id":        item["id"],
                "date":      item["datetime"],
                "title":     item["title"],
                "body":      item["body"],
                "office":    item["officeName"],
                "image_url": item.get("imageOriginLink")
            })

        # items 개수가 page_size 미만이면 더 볼 게 없는 것
        if len(items) < page_size:
            break
        page += 1

    return news_list


def fetch_news_details_bodies(stock_code: str,
                              days_back: int = 365,
                              page_size: int = 20):
    """
    get_recent_news로 메타정보를 가져온 뒤,
    각 뉴스의 실제 기사(detail_url)에 접근하여
    <article id="dic_area"> 내부 HTML(본문)만 추출해 반환합니다.
    """
    news_items = get_recent_news(stock_code,
                                 days_back=days_back,
                                 page_size=page_size)

    detailed = []
    headers  = {"User-Agent": "Mozilla/5.0"}

    for item in news_items:
        # 실제 네이버 뉴스 URL
        office_id  = item["id"][:3]
        article_id = item["id"][3:]
        detail_url = f"https://n.news.naver.com/article/{office_id}/{article_id}"

        # 1) 페이지 요청
        resp = requests.get(detail_url, headers=headers, timeout=10)
        resp.raise_for_status()

        # resp.text 전체를 넘겨서 본문만 깔끔히 추출
        body_text = extract_clean_text_from_article(resp.text)

        detailed.append({
            **item,
            "detail_url": detail_url,
            "body_text":  body_text
        })

    return detailed

def extract_clean_text_from_article(html: str) -> str:
    """
    article#dic_area 내부에서
      - img, span, div, em, table 등은 삭제(decompose)
      - strong 은 언랩(unwrap)해서 텍스트만 남긴 뒤
    단락 구분(\n\n)으로 합쳐서 반환합니다.
    """
    soup = BeautifulSoup(html, "html.parser")
    art  = soup.select_one("article#dic_area")
    if not art:
        return ""

    # 1) <br> 는 줄바꿈으로
    for br in art.find_all("br"):
        br.replace_with("\n")

    # 2) 불필요한 태그는 전부 제거
    for bad in art.find_all(["img", "span", "div", "em", "table", "script", "style"]):
        bad.decompose()

    # 3) 남기고 싶은 태그는 언랩
    for keep in art.find_all(["strong"]):
        keep.unwrap()

    # 4) 텍스트만 단락(\n\n)으로 합쳐서 반환
    return art.get_text(separator="\n\n", strip=True)

if __name__ == "__main__":
    STOCK_CODE = "005930"
    DAYS_BACK  = 1
    PAGE_SIZE  = 20
    news_cnt = 0
    
    results = fetch_news_details_bodies(
        STOCK_CODE,
        days_back=DAYS_BACK,
        page_size=PAGE_SIZE
    )

    #결과: 여기 for문으로 그냥 터미널 통해서 확인
    # for r in results:
    #     news_cnt+=1
    #     print(f"{r['date']} | {r['office']} | {r['title']}")
    #     print(f"URL: {r['detail_url']}")
    #     print("본문 텍스트:")
    #     print(r['body_text'])
    #     print("──────────────────────────────────\n")
    print(news_cnt)
    
    
        # 1) JSON 저장 폴더 생성
    out_dir = os.path.join("data", "news_json")
    os.makedirs(out_dir, exist_ok=True)
        # 3) 각각 JSON 파일로 저장
    for r in results:
        # 파일명 예시: 202507211744_005930_0150005160779.json
        filename = f"{r['date']}_{STOCK_CODE}_{r['id']}.json"
        path     = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {path}")