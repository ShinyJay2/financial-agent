import dart_fss as dart

# 테스트용으로 직접 입력
dart.set_api_key("3f3d59f0ffb7987f3f6153cd06814c941e3034f7")

# 간단히 호출해서 키가 유효한지 확인
corp_list = dart.get_corp_list()
print(corp_list[:3])



# from app.ingestion.dart_fss_client import (
#     get_corp_list,
#     find_company_by_name,
#     extract_financial_statements,
#     download_disclosure_pdf
# )

# def main():
#     print("1️⃣ 상장사 5개 조회:")
#     try:
#         corp_list = get_corp_list()
#         for corp in corp_list[:5]:
#             print(f" • {corp.corp_name} | corp_code: {corp.corp_code} | stock_code: {corp.stock_code}")
#     except Exception as e:
#         print(" ❌ 상장사 조회 실패:", e)
#     print()

#     print("2️⃣ '삼성전자' 기업 메타데이터:")
#     try:
#         samsung = find_company_by_name("삼성전자")
#         print(f" • corp_name : {samsung.corp_name}")
#         print(f" • corp_code : {samsung.corp_code}")
#         print(f" • stock_code: {samsung.stock_code}")
#     except Exception as e:
#         print(" ❌ 삼성전자 조회 실패:", e)
#     print()

#     print("3️⃣ 재무제표 불러오기 (2023년 이후):")
#     try:
#         fs = extract_financial_statements(samsung, bgn_de="20230101", report_tp="annual")
#         bs_df = fs.get("bs")
#         if bs_df is not None and not bs_df.empty:
#             print(bs_df.head(10))
#         else:
#             print(" • 재무상태표 없음.")
#     except Exception as e:
#         print(" ❌ 재무제표 추출 실패:", e)
#     print()

#     print("4️⃣ 최근 공시 PDF 다운로드:")
#     try:
#         filings = samsung.search_filings(bgn_de="20240101")
#         if not filings:
#             print(" • 최근 공시 없음.")
#         else:
#             rno = filings[0].rcept_no
#             path = download_disclosure_pdf(rno, download_dir="data")
#             print(f" • 다운로드 완료 → {path}")
#     except Exception as e:
#         print(" ❌ PDF 다운로드 실패:", e)

#     print("\n✅ 테스트 완료")

# if __name__ == "__main__":
#     main()
