import re
import json
import csv
import io
import base64
import mimetypes
from typing import List, Optional, Tuple
from pathlib import Path
import concurrent.futures
from functools import lru_cache
import logging
import json
from typing import Any, List
import fitz  # PyMuPDF for fast PDF text extraction
import camelot  # for table extraction from PDFs
from bs4 import BeautifulSoup
import tiktoken
import docx
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from app.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with your API key (v1 syntax)
client = OpenAI(api_key=settings.OPENAI_API_KEY)
# Vision model to use for OCR
VISION_MODEL = settings.VISION_MODEL_NAME

# Cache the tokenizer to avoid repeated loading
@lru_cache(maxsize=1)
def get_tokenizer(model_name: str = None):
    model = model_name or settings.EMBEDDING_MODEL_NAME
    return tiktoken.encoding_for_model(model)

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
    enc = get_tokenizer()
    
    for line in lines:
        if re.match(r"^\d+\.\s", line.strip()):
            if current_section:
                combined = "\n".join(current_section).strip()
                if len(enc.encode(combined)) > 20:
                    sections.append(combined)
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        combined = "\n".join(current_section).strip()
        if len(enc.encode(combined)) > 20:
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
                if buf:  # Only append if buf is not empty
                    chunks.append(buf)
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            chunks.append(buf)
        return chunks

    # token-based windowing for smaller texts
    enc = get_tokenizer(model_name)
    tokens = enc.encode(text)

    chunks, start, total = [], 0, len(tokens)
    while start < total:
        end = min(start + max_tokens, total)
        chunks.append(enc.decode(tokens[start:end]))
        start += (max_tokens - overlap)
    return chunks


def chunk_json(raw_json: str, records_per_chunk: int = 10) -> List[str]:
    """
    Splits a JSON payload into text chunks.
     - If the top‐level object is a dict: return exactly one chunk.
     - If it's a list: split into slices of up to records_per_chunk.
     - On invalid JSON: return [].
    """
    try:
        data: Any = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    # Single‐object JSON → one chunk
    if isinstance(data, dict):
        return [json.dumps(data, ensure_ascii=False)]

    # List JSON → batch into multiple chunks
    if isinstance(data, list):
        chunks: List[str] = []
        for i in range(0, len(data), records_per_chunk):
            batch = data[i : i + records_per_chunk]
            text = "\n\n".join(
                json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                for obj in batch
            )
            chunks.append(text)
        return chunks

    # Any other type → stringify once
    return [json.dumps(data, ensure_ascii=False)]


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
    """Optimized text extraction with error handling"""
    try:
        doc = fitz.open(path)
        text_parts = []
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text_parts.append(page_text)
        doc.close()  # Important: close the document
        return "".join(text_parts)
    except Exception as e:
        logger.error(f"Text extraction failed for {path}: {e}")
        return ""

def extract_tables(pdf_path: str, max_pages: int = 10) -> List:
    """
    Optimized table extraction with limits and better error handling
    """
    tables = []
    try:
        # Limit pages for performance - adjust as needed
        pages_to_process = f'1-{max_pages}'
        
        # Try lattice first (more accurate for grid-based tables)
        try:
            tables = camelot.read_pdf(
                pdf_path, 
                pages=pages_to_process, 
                flavor='lattice',
                line_tol=2,  # Tolerance for line detection
                joint_tol=2  # Tolerance for joint detection
            )
        except Exception as e:
            logger.warning(f"Lattice extraction failed for {pdf_path}: {e}")
        
        # Fallback to stream if lattice fails or finds no tables
        if not tables or len(tables) == 0:
            try:
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=pages_to_process,
                    flavor='stream',
                    edge_tol=500,
                    split_text=True,
                    strip_text=' ',
                    row_tol=2  # Add row tolerance
                )
            except Exception as e:
                logger.warning(f"Stream extraction failed for {pdf_path}: {e}")
                tables = []
    
    except Exception as e:
        logger.error(f"Table extraction completely failed for {pdf_path}: {e}")
        tables = []
    
    return tables

def extract_pdf_images(file_path: str, max_images: int = 20) -> List[bytes]:
    """
    Optimized image extraction with limits and better memory management
    
    Args:
        file_path: Path to PDF file
        max_images: Maximum total images to extract from entire PDF (not per page)
                   Once this limit is reached, stops processing remaining pages
    """
    images = []
    try:
        doc = fitz.open(file_path)
        image_count = 0
        
        # Process pages until we hit the image limit or run out of pages
        for page_index in range(min(len(doc), 10)):  # Limit to first 10 pages for performance
            if image_count >= max_images:
                logger.info(f"Reached max_images limit ({max_images}) at page {page_index}")
                break
                
            page_images = doc.get_page_images(page_index, full=True)
            logger.debug(f"Page {page_index + 1}: found {len(page_images)} images")
            
            for img_idx, img in enumerate(page_images):
                if image_count >= max_images:
                    logger.info(f"Stopping at image {img_idx} on page {page_index + 1} (reached limit)")
                    break
                    
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip very small images (likely artifacts/logos)
                    if pix.width < 50 or pix.height < 50:
                        logger.debug(f"Skipping small image: {pix.width}x{pix.height}")
                        pix = None
                        continue
                    
                    # Convert CMYK or other >4-channel into RGB
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    images.append(pix.tobytes("png"))
                    pix = None  # Free memory immediately
                    image_count += 1
                    logger.debug(f"Extracted image {image_count} from page {page_index + 1}")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx} from page {page_index + 1}: {e}")
                    continue
        
        doc.close()
        logger.info(f"Total images extracted: {len(images)} from {file_path}")
    except Exception as e:
        logger.error(f"Image extraction failed for {file_path}: {e}")
    
    return images

def parse_chart_with_gpt(img_bytes: bytes, timeout: int = 30) -> str:
    """
    Optimized chart parsing with timeout and better error handling
    """
    try:
        # Skip very large images to avoid API limits
        if len(img_bytes) > 10 * 1024 * 1024:  # 10MB limit
            logger.warning("Image too large, skipping chart parsing")
            return ""
        
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Add timeout to the API call
        response = client.responses.create(
            model=VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a financial data analyst. Analyze this chart/table image deeply, ignoring text starting with '자료: '.\n"
                                "Provide a concise analysis in English covering key trends, data points, and insights.\n"
                                "Then translate to Korean."
                            )
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}"
                        }
                    ]
                }
            ],
            max_output_tokens=4096,  # Reduced from 4096
        )

        if getattr(response, "output_text", None):
            return response.output_text.strip()

        if response.output and response.output[0].content:
            maybe_text = getattr(response.output[0].content[0], "text", "") or ""
            if maybe_text.strip():
                return maybe_text.strip()

    except Exception as e:
        logger.error(f"GPT vision interpretation failed: {e}")
    
    return ""

def process_single_image(img_bytes: bytes) -> Optional[str]:
    """Process a single image - used for parallel processing"""
    return parse_chart_with_gpt(img_bytes)

def chunk_pdf(
    file_path: str, 
    max_tokens: int = 500, 
    overlap: int = 50,
    max_tables: int = 5,
    max_images: int = 20,
    parallel_processing: bool = True
) -> List[str]:
    """
    Optimized PDF processing with limits and parallel processing options
    """
    chunks = []
    
    logger.info(f"Starting PDF processing for {file_path}")
    
    # 1) Text extraction with filtering
    try:
        raw_text = extract_text_fast(file_path)
        if raw_text.strip():
            # Filter out lines starting with "자료: "
            lines = raw_text.split("\n")
            filtered_lines = [line for line in lines if not re.match(r"^\s*자료\s*:", line.strip())]
            filtered_text = "\n".join(filtered_lines)
            
            if filtered_text.strip():
                text_chunks = sliding_window_chunk(filtered_text, max_tokens, overlap)
                chunks.extend([f"TEXT_{i}\n{text}" for i, text in enumerate(text_chunks)])
                logger.info(f"Extracted {len(text_chunks)} text chunks")
    except Exception as e:
        logger.error(f"Text extraction failed for {file_path}: {e}")

    # 2) Table extraction with limits
    try:
        tables = extract_tables(file_path)
        tables_processed = 0
        
        for i, table in enumerate(tables):
            if tables_processed >= max_tables:
                break
            
            try:
                # Check table quality - skip if too few rows/columns
                if table.df.shape[0] < 2 or table.df.shape[1] < 2:
                    continue
                
                table_chunk = f"TABLE_{i}\n{table.df.to_csv(index=False)}"
                chunks.append(table_chunk)
                tables_processed += 1
            except Exception as e:
                logger.warning(f"Failed to process table {i}: {e}")
                continue
        
        logger.info(f"Extracted {tables_processed} tables")
    except Exception as e:
        logger.error(f"Table extraction failed for {file_path}: {e}")

    # 3) Chart extraction with parallel processing option
    try:
        images = extract_pdf_images(file_path, max_images)
        charts_processed = 0
        
        if images and parallel_processing:
            # Parallel processing of images
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_idx = {
                    executor.submit(process_single_image, img): idx 
                    for idx, img in enumerate(images[:max_images])
                }
                
                for future in concurrent.futures.as_completed(future_to_idx, timeout=120):
                    idx = future_to_idx[future]
                    try:
                        text = future.result()
                        if text and text.strip():
                            chart_chunk = f"CHART_{idx}\n{text}"
                            chunks.append(chart_chunk)
                            charts_processed += 1
                    except Exception as e:
                        logger.warning(f"Failed to process chart {idx}: {e}")
        else:
            # Sequential processing
            for idx, img in enumerate(images[:max_images]):
                try:
                    text = parse_chart_with_gpt(img)
                    if text and text.strip():
                        chart_chunk = f"CHART_{idx}\n{text}"
                        chunks.append(chart_chunk)
                        charts_processed += 1
                except Exception as e:
                    logger.warning(f"Failed to process chart {idx}: {e}")
        
        logger.info(f"Extracted {charts_processed} charts from {len(images)} images")
    except Exception as e:
        logger.error(f"Chart extraction failed for {file_path}: {e}")

    logger.info(f"Completed PDF processing for {file_path}. Total chunks: {len(chunks)}")
    return chunks

def chunk_docx(file_path: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    try:
        doc = docx.Document(file_path)
        full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return sliding_window_chunk(full_text, max_tokens, overlap)
    except Exception as e:
        logger.error(f"DOCX processing failed for {file_path}: {e}")
        return []

def chunk_image(file_path: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    try:
        raw = Path(file_path).read_bytes()
        
        # Check file size
        if len(raw) > 10 * 1024 * 1024:  # 10MB limit
            logger.warning(f"Image {file_path} too large, skipping")
            return []
        
        b64 = base64.b64encode(raw).decode("utf-8")
        mime, _ = mimetypes.guess_type(file_path)
        uri = f"data:{mime};base64,{b64}"

        resp = client.responses.create(
            model=VISION_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract all text from this image."},
                    {"type": "input_image", "image_base64": uri},
                ],
            }],
            max_output_tokens=1024,
        )
        ocr_text = resp.output_text or ""
        return sliding_window_chunk(ocr_text, max_tokens, overlap)
    except Exception as e:
        logger.error(f"Image processing failed for {file_path}: {e}")
        return []

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

def chunk_file(
    file_path: str,
    max_tokens: int = 500,
    overlap: int = 50,
    records_per_chunk: int = 10,
    rows_per_chunk: int = 50,
    **pdf_kwargs
) -> List[str]:
    """
    Optimized file chunking with better error handling and performance controls
    """
    try:
        ext = Path(file_path).suffix.lower()
        chunks: List[str] = []

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
            chunks = chunk_pdf(
                file_path, 
                max_tokens, 
                overlap,
                **pdf_kwargs
            )

        elif ext == ".docx":
            chunks = chunk_docx(file_path, max_tokens, overlap)

        elif ext in {".png", ".jpg", ".jpeg", ".tiff"}:
            chunks = chunk_image(file_path, max_tokens, overlap)

        else:
            raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            chunks = sliding_window_chunk(raw, max_tokens, overlap)

        # Filter out compliance notices
        filtered: List[str] = []
        for c in chunks:
            if not c.strip():
                continue
            first_line = c.lstrip().split("\n", 1)[0]
            if re.match(r"Compliance\s+notice", first_line, re.IGNORECASE):
                continue
            filtered.append(c)

        return filtered
    
    except Exception as e:
        logger.error(f"File chunking failed for {file_path}: {e}")
        return []

# Convenience function for batch processing
def chunk_files_batch(
    file_paths: List[str],
    max_workers: int = 3,
    **chunk_kwargs
) -> dict[str, List[str]]:
    """
    Process multiple files in parallel
    """
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(chunk_file, path, **chunk_kwargs): path 
            for path in file_paths
        }
        
        for future in concurrent.futures.as_completed(future_to_path, timeout=600):
            file_path = future_to_path[future]
            try:
                chunks = future.result()
                results[file_path] = chunks
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[file_path] = []
    
    return results