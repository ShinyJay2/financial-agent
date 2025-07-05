from pathlib import Path
from datetime import date
from typing import Dict, Optional

import dart_fss as dart
from dart_fss.api.filings.document import download_document
from dart_fss.corp import Corp, CorpList
from pandas import DataFrame

from .config import settings

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

def download_disclosure_pdf(rcept_no: str, download_dir: str) -> Path:
    """
    Download the raw disclosure PDF/XML for a given receipt number.
    Saves it under download_dir and returns the Path.
    """
    out_dir = Path(download_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = download_document(path=str(out_dir), rcept_no=rcept_no)
    return Path(file_path)
