# app/ingestion/helpers.py

def fetch_hankyung_reports(
    stock_name: str,
    days_back: int = 30,
) -> list[str]:
    """
    Fetch and download consensus PDF reports from Hankyung for the past `days_back` days.
    """
    import os
    from datetime import date, timedelta
    from app.ingestion.hankyung_client import fetch_and_download_reports

    out_dir = os.path.join("data", "hankyung_pdfs")
    os.makedirs(out_dir, exist_ok=True)

    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    fetch_and_download_reports(stock_name, start_date, end_date, out_dir)
    return [
        os.path.join(out_dir, fname)
        for fname in os.listdir(out_dir)
        if fname.lower().endswith(".pdf")
    ]

def fetch_news_json(
    stock_code: str,
    days_back: int = 30,
    page_size: int = 20,
) -> list[str]:
    """
    Fetch detailed news items (bodies) for the past `days_back` days and
    save each as a JSON file under data/news_json/.
    """
    import os, json
    from app.ingestion.news_client import fetch_news_details_bodies

    out_dir = os.path.join("data", "news_json")
    os.makedirs(out_dir, exist_ok=True)

    results = fetch_news_details_bodies(
        stock_code,
        days_back=days_back,
        page_size=page_size,
    )
    paths = []
    for item in results:
        filename = f"{item['date']}_{stock_code}_{item['id']}.json"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)
        paths.append(path)
    return paths

def fetch_mobile_reports(
    stock_code: str,
    days_back: int = 30,
    page_size: int = 20,
) -> list[str]:
    """
    Fetch research PDFs via Naver’s mobile API for the past `days_back` days.
    """
    import os
    from app.ingestion.mobile_research_client import download_reports

    out_dir = os.path.join("data", "naver_pdfs")
    os.makedirs(out_dir, exist_ok=True)

    download_reports(stock_code, days_back, page_size, out_dir)
    return [
        os.path.join(out_dir, fname)
        for fname in os.listdir(out_dir)
        if fname.lower().endswith(".pdf")
    ]
