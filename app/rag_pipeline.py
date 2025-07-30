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
from app.ingestion.krx_client import get_realtime_price


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
    memory_collection,
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
        sparse_weight: float = 0.3,
        recency_weight: float = 0.3,
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

    def retrieve(self, query: str, context: str = "") -> list[tuple[str, str]]:
        """
        Multi-turn RAG retrieval with memory‑seeding:
        0) Seed with memory snippets in full_query
        1) BM25 shortlist
        2) Dense‑vector hybrid shortlist
        3) Intersection (fallback to BM25)
        4) Fetch top candidates
        5) Compute sparse, dense, recency features
        6) Normalize & fuse scores
        7) Debug print & return final_k results
        """
        import math, re
        from app.rag_pipeline import date_from_id, parse_date_from_text

        # 0) Build full_query with memory/context
        mem_cands   = memory_collection.query([context], n_results=3)
        mem_texts   = mem_cands.get("documents", [[]])[0]
        full_query  = " ".join(mem_texts + [context, query]).strip()

        # 1) BM25 shortlist
        bm25_ids = bm25_search(full_query, k=self.bm25_k)

        # 2) Dense‑vector hybrid shortlist
        dense_ids = hybrid_search(
            full_query,
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
        today         = date.today()
        tokens        = Okt().morphs(full_query)
        sparse_scores = _bm25.get_scores(tokens) if _bm25 is not None else [0.0] * len(ids)
        sparse_map    = dict(zip(_bm25_ids, sparse_scores))

        # Single embedding of full_query
        q_vec = np.array(
            OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL_NAME
            )([full_query])[0]
        )

        raw_feats = []
        tau = 90 / math.log(2)  # half-life of 90 days

        for did, text, emb in zip(ids, docs, embs):
            # 5.1 Date & recency
            chunk_date = date_from_id(did) or parse_date_from_text(text) or date.min
            days_old   = (today - chunk_date).days
            recency    = math.exp(- days_old / tau)

            # 5.2 Sparse & dense scores
            bm25_s = sparse_map.get(did, 0.0)
            vec    = np.asarray(emb, dtype=float)
            denom  = np.linalg.norm(vec) * np.linalg.norm(q_vec)
            dense_s = float(vec.dot(q_vec) / denom) if denom > 0 else 0.0

            raw_feats.append((did, text, bm25_s, dense_s, recency))

        # 6) Normalize features
        _, _, bm_vals, dn_vals, rc_vals = zip(*raw_feats)
        norm_bm = list(minmax_scale(bm_vals)) if len(bm_vals) > 1 else [1.0] * len(bm_vals)
        norm_dn = list(minmax_scale(dn_vals)) if len(dn_vals) > 1 else [1.0] * len(dn_vals)
        norm_rc = list(minmax_scale(rc_vals)) if len(rc_vals) > 1 else [1.0] * len(rc_vals)

        # 7) Score & pick top-K
        scored = []
        for i, (did, text, *_ ) in enumerate(raw_feats):
            score = (
                self.sparse_weight  * norm_bm[i]
            + self.dense_weight   * norm_dn[i]
            + self.recency_weight * norm_rc[i]
            )
            if re.search(r"투자\s*의견", text):
                score += 0.1
            scored.append((did, text, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[: self.final_k]

        # Debug print & return
        print(f"\n🔍 Retrieval for “{query}” returned {len(top)} chunks:")
        results = []
        for did, text, *_ in top:
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
        Generates LLM-driven answers based on router-classified domains:
        - Domains: 위험 지표, 최신 종목 뉴스, 위험 헷징, 종목 위험 분석, 주식 정보
        - Metrics shown only for 위험 지표 and 종목 위험 분석 domains.
        - Unclassified queries and other domains use a generic prompt with citation rules.
        """
        from datetime import date
        import re
        from app.rag_pipeline import date_from_id

        # 1) Extract tickers (with history fallback)
        tickers = extract_tickers_from_query(query)
        if not tickers and history:
            for turn in reversed(history):
                prev = extract_tickers_from_query(turn["question"])
                if prev:
                    tickers = prev
                    break

        # 2) On-demand ingestion (commented out for performance)
        """
        if tickers:
            t0 = tickers[0]
            from app.ingestion.helper import (
                fetch_hankyung_reports,
                fetch_news_json,
                fetch_mobile_reports,
            )
            new_files = []
            new_files += fetch_hankyung_reports(t0)
            new_files += fetch_news_json(t0)
            new_files += fetch_mobile_reports(t0)
            for path in new_files:
                try:
                    self.ingest_file(path)
                except Exception:
                    pass
            self.finalize()  # Rebuild BM25 index
        """

        # 3) Build conversation context (last 3 turns)
        convo_ctx = " ".join(
            f"Q: {turn['question']} A: {turn['answer']}"
            for turn in history[-3:] if history
        ) if history else ""

        # 4) Build retrieval query
        base_q = f"{' '.join(tickers)} {query}" if tickers else query
        retrieval_q = f"{convo_ctx} {base_q}".strip()

        # 5) Domain routing
        payload = {"query": query, "chatHistory": history or []}
        domain = self.router.execute(payload).get("domain", None)

        # 6) Compute risk metrics (only for specific domains)
        metrics: dict[str, dict[str, Any]] = {}
        if tickers and (domain == "위험 지표" or domain == "종목 위험 분석"):
            t0 = tickers[0]
            vol = get_volatility_info(t0)
            is_vol_elevated = vol.get("risk_level", "N/A").lower() in ("high", "severe") or vol["volatility"] > 0.20
            metrics["volatility"] = {
                "label": f"연간 변동성: {vol['volatility']:.2%} (risk level: {vol.get('risk_level', 'N/A')})",
                "elevated": is_vol_elevated
            }
            b = get_beta_info(t0)
            interp = f" ({b.get('interpretation')})" if b.get("interpretation") else ""
            is_beta_elevated = abs(b["beta"] - 1.0) > 0.20
            metrics["beta"] = {
                "label": f"베타: {b['beta']:.2f}{interp}",
                "elevated": is_beta_elevated
            }
            metrics["de_ratio"] = {"label": "D/E 비율: N/A", "elevated": False}
            metrics["evi"] = {"label": "EVI: N/A", "elevated": False}
            metrics["ccr"] = {"label": "CCR: N/A", "elevated": False}

        # 7) Handle 주식 정보
        if domain == "주식 정보":
            if not tickers:
                return "궁금하신 종목을 입력해주세요."
            ticker = tickers[0]
            live_price = get_realtime_price(ticker)
            base_prompt = (
                "본문의 정보들은 (출처: [출처번호])를 붙여 인용할 것.\n"
                "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
                f"현재 {ticker} 실시간 주가는 {live_price:,}원입니다. {query}에 대해 답변하세요.\n"
            )
            prefix = "".join(f"Q: {turn['question']}\nA: {turn['answer']}\n\n" for turn in history[-2:] if history)
            return ask_hyperclova(prefix + base_prompt)

        # 8) Handle 위험 지표
        if domain == "위험 지표":
            if not tickers:
                return "궁금하신 종목을 입력해주세요."
            t0 = tickers[0]
            header_lines = [
                "(Retrieved from calculation)",
                f"Ticker: {t0}",
                f"- {metrics['volatility']['label']}",
                f"- {metrics['beta']['label']}"
            ]
            return "\n".join(header_lines)

        # 9) Prepare metric texts for 종목 위험 분석
        metric_texts = []
        header = ""
        if domain == "종목 위험 분석" and not tickers:
            return "궁금하신 종목을 입력해주세요."
        if domain == "종목 위험 분석" and tickers:
            t0 = tickers[0]
            metric_texts = [
                f"(Calculated) {metrics['volatility']['label']}",
                f"(Calculated) {metrics['beta']['label']}"
            ]
            header_lines = ["(Retrieved from calculation)", f"Ticker: {t0}"]
            if metrics['volatility']['elevated']:
                header_lines.append(f"- {metrics['volatility']['label']}")
            if metrics['beta']['elevated']:
                header_lines.append(f"- {metrics['beta']['label']}")
            header = "\n".join(header_lines) + "\n\n"

        # 10) Retrieval and rerank with citations
        candidates = self.retrieve(retrieval_q)
        text_to_src = {txt: src for src, txt in candidates}
        base_texts = [txt for _, txt in candidates]
        combined = metric_texts + base_texts if domain in ("종목 위험 분석", "위험 지표") else base_texts
        k = self.final_k * 2 if domain in ("종목 위험 분석", "위험 지표") else self.final_k
        top_texts = rerank_with_cross_encoder(retrieval_q, combined, k)
        numbered_lines = [
            f"[{i}] {txt} [출처:{text_to_src.get(txt, '계산값')}]"
            for i, txt in enumerate(top_texts, 1)
        ]
        numbered = "\n\n".join(numbered_lines)

        # 11) Handle 최신 종목 뉴스
        if domain == "최신 종목 뉴스":
            news_chunks = [
                (did, txt) for did, txt in self.retrieve(retrieval_q)
                if re.match(r"^\d{8}_", did)
            ]
            news_chunks.sort(key=lambda x: date_from_id(x[0]) or date.min, reverse=True)
            top5 = news_chunks[:5]
            numbered_recent = [
                f"{idx}. {date_from_id(did).strftime('%Y-%m-%d')} — {txt.splitlines()[0]} [출처:{did}]"
                for idx, (did, txt) in enumerate(top5, 1)
            ]
            recent_tmpl = (
                "본문의 정보들은 (출처: [출처번호])를 붙여 인용할 것.\n"
                "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
                f"아래는 '{query}'에 대한 최신 뉴스 5건입니다.\n"
                "- 각 항목에 **날짜(YYYY-MM-DD)**, **제목**, **출처 ID**를 정확히 포함해주세요.\n\n"
                + "\n".join(numbered_recent) +
                f"\n\n위 형식에 맞춰 핵심 내용을 간결히 요약해 설명해주세요.\n"
                f"\nQ: {query}\nA:"
            )
            prefix = "".join(f"Q: {turn['question']}\nA: {turn['answer']}\n\n" for turn in history[-2:] if history)
            return ask_hyperclova(prefix + recent_tmpl)

        # 12) Build risk analysis template for 종목 위험 분석
        def cat_chunks(cat: str, n: int = 3) -> str:
            chunk_list = self.retrieve(f"{query} {cat}", context=convo_ctx)
            selected   = chunk_list[:n]
            out        = ""
            for did, txt in selected:
                # do the newline‑to‑space replacement before the f‑string
                snippet = txt.replace("\n", " ")[:200]
                out += f"- ({did}) {snippet}...\n"
            return out


        risk_tmpl = (
            "본문의 정보들은 (출처: [출처번호])를 붙여 인용할 것.\n"
            "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
            f"아래에 제공된 증거를 바탕으로 '{query}'의 위험 요인을 다섯 가지 카테고리로 나누어 한국어로 상세히 분석해주세요.\n"
            "- 각 섹션별로 **최소 100자 이상** 작성할 것.\n"
            "- 각 섹션 제목은 반드시 다음과 같이 정확히 사용할 것:\n"
            "- 본문 중간중간 (출처: [출처번호])를 붙여 인용할 것.\n\n"
            "  1. 시장 리스크\n"
            "  2. 재무 리스크\n"
            "  3. 사업 리스크\n"
            "  4. 법률/운영 리스크\n"
            "  5. ESG/평판 리스크\n"
            "- 위 외에 다른 표현 금지\n"
            "- 모든 자료와 숫자는 [출처번호] 형태로 표시해주세요.\n"
            "- 가능한 한 최신 정보를 우선 반영하세요.\n\n"
            f"1. 시장 리스크\n{cat_chunks('시장 리스크', n=3)}\n"
            f"2. 재무 리스크\n{cat_chunks('재무 리스크', n=3)}\n"
            f"3. 사업 리스크\n{cat_chunks('사업 리스크', n=3)}\n"
            f"4. 법률/운영 리스크\n{cat_chunks('운영 리스크', n=3)}\n"
            f"5. ESG/평판 리스크\n{cat_chunks('ESG 평판 리스크', n=3)}\n"
            f"\n\n{numbered}\n\nQ: {query}\nA:"
        )

        # 13) Handle 위험 헷징 and unclassified queries
        if domain == "위험 헷징":
            base_prompt = (
                "본문의 정보들은 (출처: [출처번호])를 붙여 인용할 것.\n"
                "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
                f"{query}에 대해 주어진 자료를 바탕으로 답변하세요.\n"
                f"{numbered}"
            )
            prefix = "".join(f"Q: {turn['question']}\nA: {turn['answer']}\n\n" for turn in history[-2:] if history)
            return ask_hyperclova(prefix + base_prompt)

        # 14) Assemble prompt for unclassified queries
        prefix = "".join(f"Q: {turn['question']}\nA: {turn['answer']}\n\n" for turn in history[-2:] if history)
        if domain not in ["위험 지표", "최신 종목 뉴스", "위험 헷징", "종목 위험 분석", "주식 정보"]:
            base_prompt = (
                "본문의 정보들은 (출처: [출처번호])를 붙여 인용할 것.\n"
                "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
                f"{query}에 대해 주어진 자료를 바탕으로 답변하세요.\n"
                f"{numbered}"
            )
            return ask_hyperclova(prefix + base_prompt)

        # 15) Assemble prompt for classified domains
        prompt = prefix + risk_tmpl if domain == "종목 위험 분석" else prefix + numbered + f"\n\nQ: {query}\nA:"

        # 16) Call LLM
        response = ask_hyperclova(prompt)

        # 17) Upsert memory for future retrieval
        if history:
            mem_id = f"mem_{len(history)-1}"
            mem_text = f"User: {history[-1]['question']} Assistant: {history[-1]['answer']}"
            memory_collection.upsert(ids=[mem_id], documents=[mem_text])

        # 18) Return response
        return (header + response) if domain in ("종목 위험 분석", "위험 지표") else response
    

    # UI용 답변 함수, 이거는 참고 문서도 doc_id랑 같이 가져온다.
    
    def answer_with_sources(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None
    ) -> tuple[str, list[tuple[str, str]]]:
        """
        Returns (answer_text, sources) where:
        - answer_text is exactly what answer() returns
        - sources is the list of (doc_id, chunk_text) from retrieve()
        """
        # 1) Build the same context string as in answer() (last 3 turns of Q/A)
        convo_ctx = ""
        tickers = extract_tickers_from_query(query)
        if history:
            turns = history[-3:]
            convo_ctx = " ".join(f"Q: {t['question']} A: {t['answer']}" for t in turns)

        # 2) Build retrieval query with ticker integration
        base_q = f"{' '.join(tickers)} {query}" if tickers else query
        retrieval_q = f"{convo_ctx} {base_q}".strip()

        # 3) Fetch the raw sources
        try:
            sources = self.retrieve(query=retrieval_q, context=convo_ctx)
        except Exception as e:
            sources = []  # Fallback to empty list on failure

        # 4) Generate the final answer text
        try:
            answer_text = self.answer(query=query, history=history)
        except Exception as e:
            answer_text = "오류가 발생했습니다. 다시 시도해주세요."

        return answer_text, sources

