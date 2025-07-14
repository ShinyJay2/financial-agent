# test_dart.py

from app.ingestion.dart_fss_client import (
    get_corp_list,
    find_company_by_name,
    extract_financial_statements,
    download_disclosure_pdf
)

def main():
    # 1) Show first 5 companies
    print("1) First 5 listed companies:")
    for corp in get_corp_list()[:5]:
        print(f" • {corp.corp_name} | Corp Code: {corp.corp_code} | Stock Code: {corp.stock_code}")
    print()

    # 2) Lookup 삼성전자
    print("2) Samsung metadata:")
    samsung = find_company_by_name("삼성전자")
    print(f" • Name      : {samsung.corp_name}")
    print(f" • Corp Code : {samsung.corp_code}")
    print(f" • Stock Code: {samsung.stock_code}\n")

    # 3) Extract annual financial statements since 2023-01-01
    print("3) Samsung annual statements (balance sheet head):")
    fs = extract_financial_statements(samsung, bgn_de="20230101", report_tp="annual")
    print(fs["bs"].head(), "\n")

    # 4) Downloading most recent disclosure PDF
    print("4) Downloading most recent disclosure PDF…")
    filings = samsung.search_filings(bgn_de="20240101", end_de=None)
    if not filings:
        print(" • No recent filings found.")
    else:
        rno = filings[0].rcept_no
        path = download_disclosure_pdf(rno, download_dir="data")
        print(" • Downloaded to:", path)

    print("\n✅ test_dart.py completed successfully.")

if __name__ == "__main__":
    main()
