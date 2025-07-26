import re
import json
import csv
import io
import base64
import mimetypes
from typing import List
from pathlib import Path

import fitz  # PyMuPDF for fast PDF text extraction
import camelot  # for table extraction from PDFs
from bs4 import BeautifulSoup
import tiktoken
import docx
from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from app.config import settings

# Initialize OpenAI client with your API key (v1 syntax)
_client = OpenAI(api_key=settings.OPENAI_API_KEY)
# Vision model to use for OCR
_VISION_MODEL = settings.VISION_MODEL_NAME

# simple regex for word‐level tokenization
_TOKENIZER = re.compile(r"\w+")

def _tokenize(text: str) -> List[str]:
    return _TOKENIZER.findall(text)

def semantic_section_chunk(html: str) -> List[str]:
    """
    Split HTML/XML on numeric headings (e.g., "1.", "2.") or bolded headings.
    Returns each heading + its content.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    sections = []
    lines = text.split("\n")
    current_section = []
    for line in lines:
        if re.match(r"^\d+\.\s", line.strip()):
            if current_section:
                combined = "\n".join(current_section).strip()
                if len(_tokenize(combined)) > 20:
                    sections.append(combined)
            current_section = [line]
        else:
            current_section.append(line)
    if current_section:
        combined = "\n".join(current_section).strip()
        if len(_tokenize(combined)) > 20:
            sections.append(combined)
    return sections

def topic_model_chunk(text: str, n_topics: int = 5) -> List[str]:
    """
    Cluster sentences into n_topics via TF–IDF + KMeans,
    then return one chunk per cluster.
    """
    # simple sentence splitter on punctuation
    sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if s.strip()]
    if len(sentences) <= n_topics:
        return sentences

    vect = TfidfVectorizer(max_features=5000)
    X = vect.fit_transform(sentences)

    km = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    groups: dict[int, List[str]] = {i: [] for i in range(n_topics)}
    for sent, lab in zip(sentences, labels):
        groups[lab].append(sent)

    return [" ".join(groups[i]) for i in sorted(groups)]

def section_chunk_html(html: str) -> List[str]:
    """
    Use true HTML headers (<h1>-<h3>), xforms_title divs, or bold spans/divs.
    """
    soup = BeautifulSoup(html, "html.parser")
    headings = soup.find_all(
        lambda tag: (
            tag.name in ("h1", "h2", "h3")
        ) or (
            tag.name == "div" and "xforms_title" in tag.get("class", [])
        ) or (
            tag.name in ("span", "div")
            and "font-weight:bold" in tag.get("style", "").lower()
        )
    )

    chunks: List[str] = []
    for header in headings:
        title = header.get_text(strip=True)
        content_nodes = []
        for sib in header.next_siblings:
            if getattr(sib, "name", None) and sib in headings:
                break
            content_nodes.append(str(sib))
        body_html = "".join(content_nodes)
        body_text = BeautifulSoup(body_html, "html.parser").get_text(
            separator="\n", strip=True
        )
        chunks.append(f"{title}\n\n{body_text}")
    return chunks

def sliding_window_chunk(
    text: str,
    max_tokens: int = 500,
    overlap: int = 50,
    model_name: str = None
) -> List[str]:
    """
    Tokenize + chunk with tiktoken overlapping windows.
    If the text is very large, do a simple paragraph-based split
    to avoid the cost of encoding the entire thing.
    """
    # rough chars-per-token estimate; adjust if needed
    avg_char_per_token = 4
    max_chars = max_tokens * avg_char_per_token

    # if text is huge, fallback to paragraph-based splits
    if len(text) > max_chars * 20:
        paras = text.split("\n\n")
        chunks = []
        buf = ""
        for p in paras:
            if len(buf) + len(p) + 2 > max_chars:
                chunks.append(buf)
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            chunks.append(buf)
        return chunks

    # token-based windowing for smaller texts
    model = model_name or settings.EMBEDDING_MODEL_NAME
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    chunks, start, total = [], 0, len(tokens)
    while start < total:
        end = min(start + max_tokens, total)
        chunks.append(enc.decode(tokens[start:end]))
        start += (max_tokens - overlap)
    return chunks

def chunk_json(raw_json: str, records_per_chunk: int = 10) -> List[str]:
    data = json.loads(raw_json)
    chunks = []
    for i in range(0, len(data), records_per_chunk):
        batch = data[i : i + records_per_chunk]
        text = "\n\n".join(
            json.dumps(obj, ensure_ascii=False, indent=2) for obj in batch
        )
        chunks.append(text)
    return chunks

def chunk_csv(raw_csv: str, rows_per_chunk: int = 50) -> List[str]:
    reader = csv.reader(io.StringIO(raw_csv))
    header = next(reader, [])
    rows = list(reader)
    chunks = []
    for i in range(0, len(rows), rows_per_chunk):
        batch = rows[i : i + rows_per_chunk]
        lines = ["\t".join(header)] + ["\t".join(r) for r in batch]
        chunks.append("\n".join(lines))
    return chunks

def chunk_excel(file_path: str, rows_per_chunk: int = 50) -> List[str]:
    import pandas as pd
    df = pd.read_excel(file_path)
    headers = list(df.columns)
    chunks = []
    for i in range(0, len(df), rows_per_chunk):
        subset = df.iloc[i : i + rows_per_chunk]
        lines = ["\t".join(headers)]
        for _, row in subset.iterrows():
            lines.append("\t".join(str(row[h]) for h in headers))
        chunks.append("\n".join(lines))
    return chunks

def extract_text_fast(path: str) -> str:
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)

def extract_pdf_images(file_path: str) -> list[bytes]:
    """
    Open the PDF with PyMuPDF and pull out every embedded image as PNG bytes.
    """
    import fitz
    doc = fitz.open(file_path)
    images = []
    for page_index in range(len(doc)):
        for img in doc.get_page_images(page_index, full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            # convert CMYK or other >4-channel into RGB
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            images.append(pix.tobytes("png"))
            pix = None
    return images

def parse_chart_with_gpt(img_bytes: bytes) -> str:
    # This is pseudocode — adapt to your vision client
    resp = _client.responses.create(
        model=_VISION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize the core content and numbers here. Answer in Korean."},
                {"type": "input_image", "image_base64": base64.b64encode(img_bytes).decode()}
            ]
        }]
    )
    return resp.output_text or ""



def chunk_pdf(file_path: str, max_tokens: int=500, overlap: int=50) -> List[str]:
    # 1) Camelot table extraction
    try:
        tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
        if tables:
            return [
                f"TABLE_{i}\n{table.df.to_csv(index=False)}"
                for i, table in enumerate(tables)
            ]
    except Exception:
        pass

    # 2) Chart extraction + GPT interpretation
    chart_chunks = []
    for idx, img in enumerate(extract_pdf_images(file_path)):
        text = parse_chart_with_gpt(img)
        if text.strip():
            chart_chunks.append(f"CHART_{idx}\n{text}")
    if chart_chunks:
        return chart_chunks

    # 3) Fallback to raw text
    raw = extract_text_fast(file_path)
    return sliding_window_chunk(raw, max_tokens, overlap)


def chunk_docx(file_path: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    doc = docx.Document(file_path)
    full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return sliding_window_chunk(full_text, max_tokens, overlap)

def chunk_image(file_path: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    raw = Path(file_path).read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime, _ = mimetypes.guess_type(file_path)
    uri = f"data:{mime};base64,{b64}"

    resp = _client.responses.create(
        model=_VISION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract all text from this image."},
                {"type": "input_image", "image_base64": uri},
            ],
        }]
    )
    ocr_text = resp.output_text or ""
    return sliding_window_chunk(ocr_text, max_tokens, overlap)

def extract_risk_factors(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    header = soup.find(
        lambda tag: tag.name in ("span", "div", "h1", "h2", "h3")
        and ("위험" in tag.get_text() or "Risk Factors" in tag.get_text())
    )
    if not header:
        return ""
    texts = []
    for sib in header.next_siblings:
        if getattr(sib, "name", None) in ("h1", "h2", "h3"):
            break
        texts.append(sib.get_text(separator="\n", strip=True))
    return "\n".join(texts)

from pathlib import Path
import re
from typing import List

def chunk_file(
    file_path: str,
    max_tokens: int = 500,
    overlap: int = 50,
    records_per_chunk: int = 10,
    rows_per_chunk: int = 50
) -> List[str]:
    ext = Path(file_path).suffix.lower()
    chunks: List[str]

    if ext in {".html", ".xml"}:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        risk_txt = extract_risk_factors(content)
        if risk_txt:
            chunks = sliding_window_chunk(risk_txt, max_tokens, overlap)
        else:
            secs = semantic_section_chunk(content)
            if secs:
                chunks = secs
            else:
                chunks = section_chunk_html(content)

    elif ext == ".json":
        raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_json(raw, records_per_chunk)

    elif ext == ".csv":
        raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_csv(raw, rows_per_chunk)

    elif ext in {".xls", ".xlsx"}:
        chunks = chunk_excel(file_path, rows_per_chunk)

    elif ext == ".pdf":
        chunks = chunk_pdf(file_path, max_tokens, overlap)

    elif ext == ".docx":
        chunks = chunk_docx(file_path, max_tokens, overlap)

    elif ext in {".png", ".jpg", ".jpeg", ".tiff"}:
        chunks = chunk_image(file_path, max_tokens, overlap)

    else:
        raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        chunks = sliding_window_chunk(raw, max_tokens, overlap)

    # ────────────────────────────────────────────────
    # Filter out any chunk whose first line is “Compliance notice”
    # ────────────────────────────────────────────────
    filtered: List[str] = []
    for c in chunks:
        first_line = c.lstrip().split("\n", 1)[0]
        if re.match(r"Compliance\s+notice", first_line, re.IGNORECASE):
            continue
        filtered.append(c)

    return filtered

