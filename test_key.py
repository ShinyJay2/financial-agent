# test_key.py
import dart_fss as dart

dart.set_api_key(api_key="3f3d59f0ffb7987f3f6153cd06814c941e3034f7")

corp_list = dart.get_corp_list()
for corp in corp_list[:3]:
    print(corp.corp_name, corp.corp_code)


# import requests

# API_KEY = "3f3d59f0ffb7987f3f6153cd06814c941e3034f7"
# corp_code = "00126380"  # 삼성전자

# url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"
# params = {
#     "crtfc_key": API_KEY,
#     "corp_code": corp_code,
#     "bsns_year": "2023",
#     "reprt_code": "11011",  # 사업보고서
#     "fs_div": "CFS"         # 연결재무제표 (개별: OFS)
# }

# res = requests.get(url, params=params)
# data = res.json()
# print(data)
