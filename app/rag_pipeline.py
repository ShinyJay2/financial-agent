import os
import re
import uuid
import torch
import tiktoken

from datetime import date, datetime
from typing import Any

from app.config import settings
from app.router.clova_router import Executor
from app.clients.hyperclova_client import ask_hyperclova
from app.utils.ticker_map import extract_tickers_from_query, all_tickers
from app.risk.volatility import get_volatility_info
from app.risk.beta import get_beta_info

from app.chunking.chunker import (
    semantic_section_chunk,
    topic_model_chunk,
    sliding_window_chunk,
    chunk_file,
)
from app.retrieval.vectorstore import (
    upsert_document,
    build_bm25_index,
    bm25_search,
    hybrid_search,
    dense_collection,
    _bm25,
    _bm25_ids,
)

from sentence_transformers import CrossEncoder

import numpy as np
from sklearn.preprocessing import minmax_scale
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from konlpy.tag import Okt


#────────────────────────────────────────────────────
# Device setup for Apple MPS / CUDA / CPU
#────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#────────────────────────────────────────────────────
# Cross-encoder setup
#────────────────────────────────────────────────────
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-TinyBERT-L-2",
    device=device
)

def rerank_with_cross_encoder(query: str, candidates: list[str], top_k: int) -> list[str]:
    pairs = [[query, c] for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked[:top_k]]

def date_from_id(doc_id: str) -> date | None:
    try:
        return datetime.strptime(doc_id.split("_", 1)[0], "%Y-%m-%d").date()
    except ValueError:
        return None

def parse_date_from_text(text: str) -> date | None:
    m = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", text)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None

class RAGPipeline:
    def __init__(
        self,
        chunk_method: str = "section",
        bm25_k: int = 75,
        dense_k: int = 75,
        final_k: int = 7,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.4,
        recency_weight: float = 0.5,
        max_tokens: int = 500,
        overlap: int = 50,
        ticker_list: list[str] | None = None,
    ):
        # retrieval settings
        self.bm25_k         = bm25_k
        self.dense_k        = dense_k
        self.final_k        = final_k
        self.dense_weight   = dense_weight
        self.sparse_weight  = sparse_weight
        self.recency_weight = recency_weight

        # chunking configuration
        self.chunk_method = chunk_method
        self.chunk_kwargs = {"max_tokens": max_tokens, "overlap": overlap}

        # initialize CLOVA Router
        self.router = Executor(
            host=os.getenv("CLOVA_HOST", "clovastudio.stream.ntruss.com"),
        )

        # initialize ticker list
        self.ticker_list = ticker_list if ticker_list is not None else all_tickers()

    def ingest(self, doc_id: str, text: str) -> None:
        if self.chunk_method == "section":
            chunks = semantic_section_chunk(text) or sliding_window_chunk(text, **self.chunk_kwargs)
        elif self.chunk_method == "topic":
            chunks = topic_model_chunk(text, **self.chunk_kwargs) or sliding_window_chunk(text, **self.chunk_kwargs)
        else:
            chunks = sliding_window_chunk(text, **self.chunk_kwargs)

        for i, c in enumerate(chunks):
            if not isinstance(c, str) or not c.strip():
                continue
            upsert_document(f"{doc_id}_{i}", c)

    def ingest_file(self, file_path: str) -> None:
        model = settings.EMBEDDING_MODEL_NAME
        enc   = tiktoken.encoding_for_model(model)

        chunks    = chunk_file(file_path, **self.chunk_kwargs)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        max_tok   = self.chunk_kwargs["max_tokens"]

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str) or not chunk.strip():
                continue

            tok_count = len(enc.encode(chunk))
            if tok_count > max_tok:
                sub_chunks = sliding_window_chunk(chunk, **self.chunk_kwargs)
            else:
                sub_chunks = [chunk]

            for j, sc in enumerate(sub_chunks):
                if not sc.strip():
                    continue

                doc_id = f"{base_name}_{i}_{j}"
                try:
                    upsert_document(doc_id, sc)
                except Exception as e:
                    msg = str(e)
                    m = re.search(
                        r"maximum context length is (\d+) tokens.*requested (\d+)",
                        msg,
                    )
                    if m:
                        limit = int(m.group(1)) // 2
                        retry_chunks = sliding_window_chunk(
                            sc,
                            max_tokens=limit,
                            overlap=self.chunk_kwargs["overlap"],
                            model_name=model,
                        )
                        for k, rc in enumerate(retry_chunks):
                            if rc.strip():
                                try:
                                    upsert_document(f"{doc_id}_r{k}", rc)
                                except Exception:
                                    pass
                    else:
                        print(f"⚠️ Skipping chunk {doc_id}: {msg}")

    def finalize(self) -> None:
        """
        After all ingest()/ingest_file() calls are done,
        build the BM25 index exactly once.
        """
        build_bm25_index()

    def retrieve(self, query: str) -> list[tuple[str, str]]:
        # 0) Which tickers did the user mention?
        tickers = extract_tickers_from_query(query)

        # 1) BM25 shortlist
        bm25_ids = bm25_search(query, k=self.bm25_k)

        # 2) Dense‐vector fallback
        dense_ids = hybrid_search(
            query,
            bm25_k=self.bm25_k,
            dense_k=self.dense_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )

        # 3) Intersection (or fallback)
        candidates = [did for did in bm25_ids if did in dense_ids] or bm25_ids
        candidates = candidates[: self.final_k * 3]

        # 4) Fetch docs + embeddings
        resp = dense_collection.get(ids=candidates, include=["documents", "embeddings"])
        ids, docs, embs = resp["ids"], resp["documents"], resp["embeddings"]

        # 5) Feature computation
        today   = date.today()
        tokens  = Okt().morphs(query)
        # now _bm25 is freshly looked up from the vectorstore module
        sparse_scores = _bm25.get_scores(tokens) if _bm25 is not None else [0.0] * len(ids)
        sparse_map    = dict(zip(_bm25_ids, sparse_scores))

        # Query embedding
        q_vec = np.array(
            OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL_NAME
            )([query])[0]
        )

        raw_feats = []
        for did, text, emb in zip(ids, docs, embs):
            # recency
            from app.rag_pipeline import date_from_id, parse_date_from_text
            chunk_date = date_from_id(did) or parse_date_from_text(text) or date.min
            days_old   = (today - chunk_date).days
            recency    = max(0.0, 1.0 - days_old / 365.0)

            bm25_s = sparse_map.get(did, 0.0)
            arr    = np.array(emb)
            denom  = np.linalg.norm(arr) * np.linalg.norm(q_vec)
            dense_s = float(arr.dot(q_vec) / denom) if denom > 0 else 0.0

            raw_feats.append((did, text, bm25_s, dense_s, recency))

        def normalize(xs: list[float]) -> list[float]:
            return list(minmax_scale(xs)) if len(xs) > 1 else [1.0] * len(xs)

        _, _, bm_vals, dn_vals, rc_vals = zip(*raw_feats)
        norm_bm, norm_dn, norm_rc = normalize(bm_vals), normalize(dn_vals), normalize(rc_vals)

        # 6) Score & pick top‐K
        scored = []
        for i, (did, text, *_ ) in enumerate(raw_feats):
            score = (
                self.sparse_weight  * norm_bm[i] +
                self.dense_weight   * norm_dn[i] +
                self.recency_weight * norm_rc[i]
            )

            if re.search(r"투자\s*의견", text):
                score += 0.1

            scored.append((did, text, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[: self.final_k]


        # 7) Debug print & return
        print(f"\n🔍 Retrieval for “{query}” returned {len(top)} chunks:")
        results = []
        for did, text, *rest in top:
            base    = did.rsplit("_", 2)[0]
            snippet = text.replace("\n", " ")[:50]
            print(f"  • {did} (file: {base}): “{snippet}…”")
            results.append((did, text))

        return results



    def answer(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None
    ) -> str:
        """
        Fully LLM-driven answers with router-based domains:
        - Use router’s domain for classification into 위험 지표, 종목 위험 분석, 최신 종목 뉴스, other => 일반 검색
        - 위험 지표: show only calculated volatility & beta
        - 종목 위험 분석: inject metrics + five fixed risk categories with citations
        - 최신 종목 뉴스: latest news template with citations
        - 일반 검색: generic RAG retrieval and citations
        """

        # 1) extract tickers & build retrieval query
        tickers = extract_tickers_from_query(query)
        retrieval_q = f"{' '.join(tickers)} {query}" if tickers else query

        # 2) compute metrics
        metrics: dict[str, dict[str, Any]] = {}
        if tickers:
            t0 = tickers[0]
            vol = get_volatility_info(t0)
            is_vol_elevated = vol.get("risk_level", "N/A").lower() in ("high", "severe") or vol["volatility"] > 0.20
            metrics["volatility"] = {
                "label": f"연간 변동성: {vol['volatility']:.2%} (risk level: {vol.get('risk_level','N/A')})",
                "elevated": is_vol_elevated
            }
            b = get_beta_info(t0)
            interp = f" ({b.get('interpretation')})" if b.get("interpretation") else ""
            is_beta_elevated = abs(b["beta"] - 1.0) > 0.20
            metrics["beta"] = {
                "label": f"베타: {b['beta']:.2f}{interp}",
                "elevated": is_beta_elevated
            }
            metrics["de_ratio"] = {"label": "D/E 비율: {de_ratio}", "elevated": False}
            metrics["evi"]      = {"label": "EVI: {evi}",      "elevated": False}
            metrics["ccr"]      = {"label": "CCR: {ccr}",      "elevated": False}

        # 3) domain routing via router
        payload = {"query": query, "chatHistory": history or []}
        domain = self.router.execute(payload).get("domain", "일반 검색")  # Direct use of self.router.execute()

        # debug
        print(f"[Debug] router domain: '{domain}'")

        # 4) 위험 지표: metrics only
        if domain == "위험 지표":
            if not tickers:
                return "궁금해하시는 종목을 입력해주세요"
            header_lines = ["(Retrieved from calculation)", f"Ticker: {t0}"]
            header_lines.append(f"- {metrics['volatility']['label']}")
            header_lines.append(f"- {metrics['beta']['label']}")
            return "\n".join(header_lines)

        # 5) 종목 위험 분석 without ticker
        if domain == "종목 위험 분석" and not tickers:
            return "궁금해하시는 종목을 입력해주세요"

        # 6) prepare metric injection for risk domains
        metric_texts: list[str] = []
        header = ""
        if domain in ("종목 위험 분석", "위험 지표") and tickers:
            metric_texts = [
                f"(Calculated) {metrics['volatility']['label']}",
                f"(Calculated) {metrics['beta']['label']}"
            ]
            if domain == "종목 위험 분석":
                header_lines = ["(Retrieved from calculation)", f"Ticker: {t0}"]
                if metrics['volatility']['elevated']:
                    header_lines.append(f"- {metrics['volatility']['label']}")
                if metrics['beta']['elevated']:
                    header_lines.append(f"- {metrics['beta']['label']}")
                header = "\n".join(header_lines) + "\n\n"

        # 7) carry forward last two turns
        prefix = ""
        if history:
            for turn in history[-2:]:
                prefix += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

        # 8) retrieval and rerank with citations
        candidates = self.retrieve(retrieval_q)
        text_to_src = {txt: src for src, txt in candidates}
        base_texts = [txt for _, txt in candidates]
        combined = metric_texts + base_texts if domain in ("종목 위험 분석", "위험 지표") else base_texts
        k = self.final_k * 2 if domain in ("종목 위험 분석", "위험 지표") else self.final_k
        top_texts = rerank_with_cross_encoder(retrieval_q, combined, k)
        numbered_lines = []
        for i, txt in enumerate(top_texts, 1):
            src = text_to_src.get(txt, "계산값")
            numbered_lines.append(f"[{i}] {txt} [출처:{src}]")
        numbered = "\n\n".join(numbered_lines)

        # 9) build prompts
        recent_chunks = self.retrieve(retrieval_q)
        numbered_recent = []
        for idx, (_, t) in enumerate(recent_chunks, 1):
            numbered_recent.append(f"[{idx}] {t} [출처:{recent_chunks[idx-1][0]}]")
        numbered_recent = "\n\n".join(numbered_recent)
        recent_tmpl = (
            f"아래 자료만 보고 '{query}'의 최신 뉴스를 제공해주세요.\n"
            "- 모든 숫자는 [출처번호] 형태로 표시해주세요.\n"
            "- 자료와 의견의 날짜 또한 표시하세요.\n"
            "- 핵심 포인트 중심으로\n\n"
            + numbered_recent +
            f"\n\nQ: {query}\nA:"
        )

        def cat_chunks(cat):
            chunks = self.retrieve(f"{query} {cat}")
            return "".join(f"   - {t}\n" for _, t in chunks)

        risk_tmpl = (
            f"아래에 제공된 증거를 바탕으로 '{query}'의 위험 요인을 다섯 가지 카테고리로 나누어 한국어로 상세히 분석해주세요.\n"
            "- 각 섹션 제목은 반드시 다음과 같이 정확히 사용할 것:\n"
            "  1. 시장 리스크\n"
            "  2. 재무 리스크\n"
            "  3. 사업 리스크\n"
            "  4. 법률/운영 리스크\n"
            "  5. ESG/평판 리스크\n"
            "- 위 외에 다른 표현 금지\n"
            "- 모든 숫자는 [출처번호] 형태로 표시해주세요.\n"
            "- 가능한 한 최신 정보를 우선 반영하세요.\n\n"
            "1. 시장 리스크\n" + cat_chunks("시장 리스크") +
            "\n2. 재무 리스크\n" + cat_chunks("재무 리스크") +
            "\n3. 사업 리스크\n" + cat_chunks("사업 리스크") +
            "\n4. 법률/운영 리스크\n" + cat_chunks("운영 리스크") +
            "\n5. ESG/평판 리스크\n" + cat_chunks("ESG 평판 리스크") +
            f"\n\n{numbered}\n\nQ: {query}\nA:"
        )

        # 10) assemble prompt
        if domain == "종목 위험 분석":
            prompt = prefix + risk_tmpl
        elif domain == "최신 종목 뉴스":
            prompt = prefix + recent_tmpl
        else:
            prompt = prefix + numbered + f"\n\nQ: {query}\nA:"

        # 11) call LLM
        response = ask_hyperclova(prompt)

        # 12) return
        return (header + response) if domain == "종목 위험 분석" else response


