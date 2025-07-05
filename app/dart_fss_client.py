# app/dart_fss_client.py

import os
import dart_fss as dart
from .config import settings

# 1) Set the API key from your settings
dart.set_api_key(settings.DART_API_KEY)  # configures DART-FSS to use your key :contentReference[oaicite:1]{index=1}

def get_corp_list():
    """
    Return a DartCorpList (Pandas-enabled) of all registered companies.
    """
    return dart.get_corp_list()  # returns DataFrame-like list of Corps :contentReference[oaicite:2]{index=2}

def find_company_by_name(name: str):
    """
    Perform exact company-name lookup in the DART list.
    Returns the first matching Corp object (or raises IndexError).
    """
    corps = get_corp_list()
    matches = corps.find_by_corp_name(name, exactly=True)  # list of Corp objects :contentReference[oaicite:3]{index=3}
    return matches[0]

def extract_financial_statements(corp, bgn_de: str, reprt_code: str = "11011"):
    """
    Download and parse the financial statements for a given Corp object.
    bgn_de: 'YYYYMMDD' start date.
    reprt_code: report code (11011=annual, 11013=Q1, etc.).
    Returns a pandas.DataFrame of the statement.
    """
    fs = corp.extract_fs(bgn_de=bgn_de, reprt_code=reprt_code)  # DataFrame of FS :contentReference[oaicite:4]{index=4}
    return fs

def download_disclosure_pdf(rcept_no: str, download_path: str):
    """
    Download the raw PDF/XML disclosure for a given receipt number.
    Returns the local file path.
    """
    full_path = dart.download_document(path=download_path, rcept_no=rcept_no)  # saves & returns path :contentReference[oaicite:5]{index=5}
    return full_path

def get_corp_code_mapping():
    """
    Download and unzip the corpCode.xml containing all company codes.
    Returns an OrderedDict mapping corp_code → Corp metadata.
    """
    return dart.get_corp_code()  # maps 8-digit corp codes to metadata :contentReference[oaicite:6]{index=6}
