from pathlib import Path
from datetime import date
from typing import Dict, Optional

import dart_fss as dart
from dart_fss.api.filings.document import download_document
from dart_fss.corp import Corp, CorpList
from pandas import DataFrame
import requests

from ..config import settings
from ..utils.ticker_map import find_name_by_ticker
from ..chunking.chunker import extract_risk_factors

# Configure your DART-FSS API key
dart.set_api_key(settings.DART_API_KEY)


def get_corp_list() -> CorpList:
    """Return a list of all registered corporations."""
    return dart.get_corp_list()


def find_company_by_name(name: str) -> Corp:
    """
    Exact-match lookup of a Corp by its Korean name.
    Raises IndexError if no match is found.
    """
    return get_corp_list().find_by_corp_name(name, exactly=True)[0]


def corp_code_for(ticker: str) -> str:
    """
    Look up the DART `corp_code` for a 6-digit stock ticker by:
      1) mapping ticker → company name
      2) querying DART’s corp list for that name
    """
    name = find_name_by_ticker(ticker)
    if not name:
        raise KeyError(f"No company name mapping for ticker {ticker!r}")
    corp = get_corp_list().find_by_corp_name(name, exactly=True)[0]
    return corp.corp_code


def extract_financial_statements(
    corp: Corp,
    bgn_de: str,
    end_de: Optional[str] = None,
    report_tp: str = "annual"
) -> Dict[str, DataFrame]:
    """
    Download and parse financial statements.
    - bgn_de: YYYYMMDD start date
    - end_de: YYYYMMDD end date (defaults to today)
    - report_tp: 'annual', 'half', or 'quarter'
    Returns a dict with keys 'bs','is','cis','cf'.
    """
    end_de = end_de or date.today().strftime("%Y%m%d")
    return corp.extract_fs(bgn_de=bgn_de, end_de=end_de, report_tp=report_tp)


def get_risk_factors(corp_code: str, bgn_de: str, end_de: str) -> str:
    """
    Call DART to fetch the business report and extract only the risk factors section.
    """
    url = "https://opendart.fss.or.kr/api/document.xml"
    params = {
        "crtfc_key": settings.DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
        "last_report_at": "Y",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    xml = r.text
    risk_html = extract_risk_factors(xml)
    if not risk_html:
        raise ValueError("No 'Risk Factors' section found in DART document")
    return risk_html


def download_disclosure_pdf(rcept_no: str, download_dir: str) -> Path:
    """
    Download the raw disclosure PDF/XML for a given receipt number.
    Saves it under download_dir and returns the Path.
    """
    out_dir = Path(download_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = download_document(path=str(out_dir), rcept_no=rcept_no)
    return Path(file_path)
