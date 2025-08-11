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
from app.risk.d_e_r      import calculate_de_quarterly_growth
from app.risk.evi        import calculate_evi
from app.risk.foreign_organ import calculate_rank_days

from app.hedging.regression import run_hedge_pipeline
from app.utils.ticker_map import find_name_by_ticker



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

        # 0) Build full_query with memory/context (only if we actually have text)
        mem_texts = []
        if context and context.strip():
            mem_cands = memory_collection.query(
                query_texts=[context],
                n_results=3
            )
            mem_texts = mem_cands.get("documents", [[]])[0]
        full_query = " ".join(mem_texts + [context, query]).strip()

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

        if not candidates:
            return []

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
        

        if history is None:
            history = []

        """
        Generates LLM-driven answers based on router-classified domains:
        - Domains: 위험 지표, 최신 종목 뉴스, 위험 헷징, 종목 위험 분석, 주가 및 주식 정보
        - Metrics shown only for 위험 지표 and 종목 위험 분석 domains.
        - Unclassified queries and other domains use a generic prompt with citation rules.
        """

        from app.rag_pipeline import date_from_id

        # 1) Extract tickers (with history fallback)
        print("▶ extract_tickers_from_query →", extract_tickers_from_query(query))

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
            for turn in history[-3:]
        )

        # 4) Build retrieval query
        base_q = f"{' '.join(tickers)} {query}" if tickers else query
        retrieval_q = f"{convo_ctx} {base_q}".strip()

        # 5) Domain routing
        payload = {"query": query, "chatHistory": history or []}
        domain = self.router.execute(payload).get("domain", None)
        print(f"🛠️  Classified domain → '{domain}'")

        t0 = tickers[0]

        # 6) Compute risk metrics (only for specific domains)
        metrics: dict[str, dict[str, Any]] = {}
        if tickers and (domain == "위험 지표" or domain == "종목 위험 분석"):



            # 1) volatility:
            vol = get_volatility_info(t0)
            is_vol_elevated = vol.get("risk_level", "N/A").lower() in ("high", "severe") or vol["volatility"] > 0.20
            metrics["volatility"] = {
                "label": f"연간 변동성: {vol['volatility']:.2%} (risk level: {vol.get('risk_level', 'N/A')})",
                "elevated": is_vol_elevated
            }

            # 2) beta:
            b = get_beta_info(t0)
            interp = f" ({b.get('interpretation')})" if b.get("interpretation") else ""
            is_beta_elevated = abs(b["beta"] - 1.0) > 0.20
            metrics["beta"] = {
                "label": f"베타: {b['beta']:.2f}{interp}",
                "elevated": is_beta_elevated
            }
            # 3) D/E ratio: quarterly growth & grade
            de_info = calculate_de_quarterly_growth(t0)
            # total_pct_change is the summed QoQ % change, de_ic_grade is its risk grade
            de_val       = de_info["total_pct_change"]
            de_grade     = de_info["de_ic_grade"]
            is_de_elevated = de_grade in ("매우 높음", "높음")  # adjust per your grading scale
            metrics["de_ratio"] = {
                "label":    f"D/E 증감률 합계: {de_val:.2f}% (Grade: {de_grade})",
                "elevated": is_de_elevated
            }

            # 4) EVI (Earnings Volatility Index)
            evi_info = calculate_evi(t0)
            evi_val  = evi_info["evi"]
            evi_rank = evi_info["rank"]
            # assume ranks "High", "Medium", "Low"
            is_evi_elevated = evi_rank.lower() == "high"
            metrics["evi"] = {
                "label":    f"EVI: {evi_val:.4f} (Rank: {evi_rank})",
                "elevated": is_evi_elevated
            }

            # 5) Foreign & institutional net‐flow risk
            # calculate_rank_days returns (corp, corp_name, neg_days, level)
            _, _, neg_days, flow_level = calculate_rank_days(t0)
            is_flow_high = flow_level in ("중간", "높음")
            metrics["foreign"] = {
                "label":    f"외국인·기관 순매도 일수: {neg_days}일 (Level: {flow_level})",
                "elevated": is_flow_high
            }
        
        # immediately after your volatility/beta/... computations
        if tickers and domain == "주가 및 주식 정보":

            t0 = tickers[0]
            try:
                price = get_realtime_price(t0)
            except Exception:
                price = None
            metrics["price"] = {
                "label": f"실시간 주가: {price:,}원" if price is not None else "실시간 주가: N/A",
                "value": price
            }


    
        header = ""

        


        # Helper function for citation formatting
        def format_citation(txt: str, did: str, date_from_id, for_numbered=False) -> str:
            if did == '계산값':
                return "[출처:계산값]"
            # Check if did starts with YYYYMMDD_ (date-based format)
            if re.match(r"^\d{8}_", did):
                try:
                    date_obj = date_from_id(did)
                    if date_obj is None:
                        return "[출처:{did}]".format(did=did)
                    return "[출처:{date}_{did}]".format(date=date_obj.strftime('%Y-%m-%d'), did=did)
                except Exception:
                    return "[출처:{did}]".format(did=did)
            return "[출처:{did}]".format(did=did)  # Non-date-based file names
        
                # 12) Build risk analysis template for 종목 위험 분석
        def cat_chunks(cat: str, n: int = 3) -> str:
            chunk_list = self.retrieve(f"{query} {cat}", context=convo_ctx) if query and convo_ctx else []
            selected = chunk_list[:n]
            out = ""
            for did, txt in selected:
                snippet = txt.replace("\n", " ")[:200]
                if re.match(r"^\d{8}_", did):
                    try:
                        date_obj = date_from_id(did)
                        date_str = date_obj.strftime('%Y-%m-%d') if date_obj else "Unknown"
                    except Exception:
                        date_str = "Unknown"
                else:
                    date_str = "Unknown"
                out += f"- ({date_str}_{did if date_str != 'Unknown' else did}) {snippet}...\n"
            return out

        # 10) Retrieval and rerank with citations
        candidates = self.retrieve(retrieval_q) if retrieval_q else []
        if not candidates:
            return []

        text_to_src = {txt: did for did, txt in candidates}
        base_texts = [txt for _, txt in candidates]
        metric_rerank = []
        if domain == "위험 지표" and tickers:
            t0 = tickers[0]
            if metrics and all(key in metrics for key in ['volatility', 'beta', 'de_ratio', 'evi', 'foreign']):
                metric_rerank = [
                    f"(Calculated) {metrics['volatility']['label']}",
                    f"(Calculated) {metrics['beta']['label']}",
                    f"(Calculated) {metrics['de_ratio']['label']}",
                    f"(Calculated) {metrics['evi']['label']}",
                    f"(Calculated) {metrics['foreign']['label']}"
                ]
        combined = metric_rerank + base_texts if domain == "위험 지표" else base_texts
        k = (self.final_k * 2 if domain == "위험 지표" else self.final_k) if hasattr(self, 'final_k') else 3
        top_texts = rerank_with_cross_encoder(retrieval_q, combined, k) if combined else []
        if not top_texts:
            numbered = "No relevant information found."
        else:
            numbered_lines = [
                f"[{i}] {format_citation(txt, text_to_src.get(txt, '계산값'), date_from_id, for_numbered=True)}"
                for i, txt in enumerate(top_texts, 1)
            ]
            numbered = "\n\n".join(numbered_lines)


        body = ""

        # 7) Handle 주가 및 주식 정보
        if domain == "주가 및 주식 정보":
            if not tickers:
                return "궁금하신 종목을 입력해주세요."
            ticker = tickers[0]

            # Build the price header using the pre‐formatted label
            price_label = metrics.get("price", {}).get("label", "실시간 주가: N/A")
            header = f"(Retrieved from calculation)\n{price_label}\n\n"

            # Build the HyperClova prompt as before
            prefix = "".join(
                f"Q: {turn['question']}\nA: {turn['answer']}\n\n"
                for turn in history[-2:]
            )
            base_prompt = (
                "본문의 정보들은 [출처 파일명] 형식으로 정확히 인용할 것. [출처 파일명]은 문서 ID (날짜가 포함된 경우 YYYY-MM-DD_문서ID, 그렇지 않은 경우 문서ID 그대로)로, 텍스트 없이 문서 ID만 표시해야 합니다.\n"
                "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
                f"{price_label} 를 바탕으로 이 주식 {query}에 대해 답변하세요.\n"
                f"실시간 주가와 함께 이 주식{query}에 대해 답변하세요.\n"
                "- 답변 본문에 인용된 모든 출처는 [출처 파일명] 형식으로, 문서 ID만 표시할 것.\n"
                "- 답변 마지막에 아래 제공된 **출처 목록**을 번호와 함께 정확히 그대로 포함할 것. 수정하거나 재구성하지 말고, 제공된 형식을 엄격히 따를 것.\n\n"
                f"**출처 목록**\n{numbered}"
            ) 

            candidates = self.retrieve(f"{ticker} {query}") if query and ticker else []
            text_to_src = {txt: did for did, txt in candidates}
            candidate_texts = [txt for _, txt in candidates][:3]
            if not candidate_texts:
                citations = "No relevant information found."
            else:
                numbered_lines = [
                    f"[{i}] {format_citation(txt, text_to_src.get(txt, '계산값'), date_from_id, for_numbered=True)}"
                    for i, txt in enumerate(candidate_texts, 1)
                ]
                citations = "\n".join(numbered_lines)
            response = ask_hyperclova(prefix + base_prompt + "\n\n" + citations + f"\n\nQ: {query}\nA:")

            # Fallback: Append citation list if not included
            # Enhanced fallback: also catch replies that end with a generic “[1] …” list
            lines = [L for L in response.splitlines() if L.strip()]
            last_line = lines[-1] if lines else ""
            if numbered != "No relevant information found.":
                needs_fallback = (
                    "**출처 목록**" not in response
                    or re.match(r"^\[\s*1\s*\]", last_line)
                )
                if needs_fallback:
                    response = response.rstrip() + f"\n\n**출처 목록**\n{numbered}"

            body = response



        # 8) Handle 위험 지표
        elif domain == "위험 지표":
            if not tickers:
                return "궁금하신 종목을 입력해주세요."
            t0 = tickers[0]
            if not metrics or not all(key in metrics for key in ['volatility', 'beta', 'de_ratio', 'evi', 'foreign']):
                return "메트릭 정보를 가져올 수 없습니다."
            header_lines = [
                "(Retrieved from calculation)",
                f"Ticker: {t0}",
                f"- {metrics['volatility']['label']}",
                f"- {metrics['beta']['label']}",
                f"- {metrics['de_ratio']['label']}",
                f"- {metrics['evi']['label']}",
                f"- {metrics['foreign']['label']}",
            ]
            header =  "\n".join(header_lines)
            body = ""

        # 11) Handle 최신 종목 뉴스
        elif domain == "최신 종목 뉴스":
            news_chunks = [
                (did, txt) for did, txt in self.retrieve(retrieval_q) if retrieval_q and re.match(r"^\d{8}_", did)
            ]
            # Sort by date if possible, otherwise by did
            try:
                news_chunks.sort(key=lambda x: date_from_id(x[0]) or date.min, reverse=True)
            except Exception:
                news_chunks.sort(key=lambda x: x[0], reverse=True)
            top5 = news_chunks[:5]
            if not top5:
                numbered_recent = ["No recent news found."]
            else:
                numbered_recent = []
                for idx, (did, txt) in enumerate(top5, 1):
                    if re.match(r"^\d{8}_", did):
                        try:
                            date_obj = date_from_id(did)
                            date_str = date_obj.strftime('%Y-%m-%d') if date_obj else "Unknown"
                        except Exception:
                            date_str = "Unknown"
                    else:
                        date_str = "Unknown"
                    numbered_recent.append(
                        f"{idx}. {date_str} — {txt.splitlines()[0]} [출처:{date_str}_{did if date_str != 'Unknown' else did}]"
                    )
            recent_tmpl = (
                "본문의 정보들은 [출처 파일명] 형식으로 정확히 인용할 것. [출처 파일명]은 문서 ID (날짜가 포함된 경우 YYYY-MM-DD_문서ID, 그렇지 않은 경우 문서ID 그대로)로, 텍스트 없이 문서 ID만 표시해야 합니다.\n"
                "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
                f"아래는 '{query}'에 대한 최신 뉴스 5건입니다.\n"
                "- 각 항목에 **날짜(YYYY-MM-DD)**, **제목**, **출처 ID**를 정확히 포함해주세요.\n\n"
                + "\n".join(numbered_recent) +
                f"\n\n위 형식에 맞춰 핵심 내용을 요약해 설명해주세요.\n"
                "- 답변 본문에 인용된 모든 출처는 [출처 파일명] 형식으로, 문서 ID만 표시할 것.\n"
                "- 답변 마지막에 아래 제공된 **출처 목록**을 번호와 함께 정확히 그대로 포함할 것. 수정하거나 재구성하지 말고, 제공된 형식을 엄격히 따를 것.\n\n"
                f"**출처 목록**\n{numbered}"
            )
            prefix = "".join(f"Q: {turn['question']}\nA: {turn['answer']}\n\n" for turn in history[-2:]) if history else ""
            response = ask_hyperclova(prefix + recent_tmpl + f"\n\nQ: {query}\nA:")

            # Fallback: Append citation list if not included
            # Enhanced fallback: also catch replies that end with a generic “[1] …” list
            lines = [L for L in response.splitlines() if L.strip()]
            last_line = lines[-1] if lines else ""
            if numbered != "No relevant information found.":
                needs_fallback = (
                    "**출처 목록**" not in response
                    or re.match(r"^\[\s*1\s*\]", last_line)
                )
                if needs_fallback:
                    response = response.rstrip() + f"\n\n**출처 목록**\n{numbered}"
            body = response


        
        elif domain == "종목 위험 분석":
            if not tickers:
                return "궁금하신 종목을 입력해주세요."
            
            t0 = tickers[0]

            # 9) Prepare metric texts for 종목 위험 분석
            # 2) Build a list of all five metric labels
            metric_keys = ["volatility", "beta", "de_ratio", "evi", "foreign"]
            analysis_metrics = [
                metrics[k]["label"]
                for k in metric_keys
                if k in metrics
            ]

            # 3) Construct header showing only the “elevated” ones
            header_lines = ["(Retrieved from calculation)", f"Ticker: {t0}"]
            for k, label in zip(metric_keys, analysis_metrics):
                if metrics.get(k, {}).get("elevated", False):
                    header_lines.append(f"- (Calculated) {label}")

            header = "\n".join(header_lines) + "\n\n"


            risk_tmpl = (
                "본문의 정보들은 [출처 파일명] 형식으로 정확히 인용할 것. [출처 파일명]은 문서 ID (날짜가 포함된 경우 YYYY-MM-DD_문서ID, 그렇지 않은 경우 문서ID 그대로)로, 텍스트 없이 문서 ID만 표시해야 합니다.\n"
                "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
                f"아래에 제공된 증거를 바탕으로 '{query}'의 위험 요인을 다섯 가지 카테고리로 나누어 한국어로 상세히 분석해주세요.\n"
                "- 각 섹션별로 **최소 150자 이상** 작성할 것.\n"
                "- 각 섹션 제목은 반드시 다음과 같이 정확히 사용할 것:\n"
                "  1. 시장 리스크\n"
                "  2. 재무 리스크\n"
                "  3. 사업 리스크\n"
                "  4. 법률/운영 리스크\n"
                "  5. ESG/평판 리스크\n"
                "- 위 외에 다른 표현 금지.\n"
                "- 모든 자료와 숫자는 [출처 파일명] 형식으로, 문서 ID만 표시할 것.\n"
                "- 가능한 한 최신 정보를 우선 반영하세요.\n"
                "- 답변 마지막에 아래 제공된 **출처 목록**을 번호와 함께 정확히 그대로 포함할 것. 수정하거나 재구성하지 말고, 제공된 형식을 엄격히 따를 것.\n\n"
                f"1. 시장 리스크\n{cat_chunks('시장 리스크', n=3)}\n"
                f"2. 재무 리스크\n{cat_chunks('재무 리스크', n=3)}\n"
                f"3. 사업 리스크\n{cat_chunks('사업 리스크', n=3)}\n"
                f"4. 법률/운영 리스크\n{cat_chunks('운영 리스크', n=3)}\n"
                f"5. ESG/평판 리스크\n{cat_chunks('ESG 평판 리스크', n=3)}\n"
                f"\n\n**출처 목록**\n{numbered}\n\nQ: {query}\nA:"
            )
            prefix = "".join(f"Q: {turn['question']}\nA: {turn['answer']}\n\n" for turn in history[-2:]) if history else ""
            response = ask_hyperclova(prefix + risk_tmpl)
            # Fallback: Append citation list if not included
            # Enhanced fallback: also catch replies that end with a generic “[1] …” list
            lines = [L for L in response.splitlines() if L.strip()]
            last_line = lines[-1] if lines else ""
            if numbered != "No relevant information found.":
                needs_fallback = (
                    "**출처 목록**" not in response
                    or re.match(r"^\[\s*1\s*\]", last_line)
                )
                if needs_fallback:
                    response = response.rstrip() + f"\n\n**출처 목록**\n{numbered}"
            body = response


                # 13) Handle 위험 헷징
        elif domain == "위험 헷징":
            if not tickers:
                return "궁금하신 종목을 입력해주세요."
            base_ticker = tickers[0]

            m = re.search(r"(\d+)\s*주", query)
            if m:
                qty_base = int(m.group(1))
            else:
                # if you prefer a default, set qty_base = 1 here instead
                return "현재 보유하신 주식 수를 ‘숫자+주’ 형태로 알려주세요 (예: 30주)."

            # 1) Find the best hedge candidate
            from app.hedging.regression import run_hedge_pipeline
            from app.utils.ticker_map    import find_name_by_ticker
            df_candidates = run_hedge_pipeline(base_ticker)
            if df_candidates.empty:
                return "헷징 후보가 없습니다."
            hedge_ticker = str(df_candidates.loc[0, "ticker"])
            base_name    = find_name_by_ticker(base_ticker)
            hedge_name   = find_name_by_ticker(hedge_ticker)

            # ── 3) Prepare dates ──
            from datetime import date, timedelta
            end   = date.today()
            start = end - timedelta(days=365)
            sd, ed = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

            # ── 4) Sweep over absolute hedge sizes relative to user’s base_qty ──
            from app.hedging.monte_carlo import diversification_abs
            # e.g. 10%, 50%, 100% of what they hold, but at least 1 share
            hedge_percs = [0.1, 0.5, 1.0]
            hedge_qtys = [max(1, int(qty_base * pct)) for pct in hedge_percs]


            reductions = {}
            for q in hedge_qtys:
                _, stats = diversification_abs(
                    base_ticker=base_ticker,
                    hedge_ticker=hedge_ticker,
                    qty_base=qty_base,
                    qty_hedge=q,
                     start_date=sd,
                     end_date=ed
                 )
                
                reductions[q] = stats["VaR_reduction_%"]

            # 3) Build header‐only response
            header_lines = [
                 "(Retrieved from calculation)",
                 f"Base: {base_name} {qty_base}주  ({base_ticker})",
                 f"Hedge candidate: {hedge_name} ({hedge_ticker})"
             ]
            
            for q, red in reductions.items():
                header_lines.append(f"- {q}주 매수 → VaR 약 {red:.2f}% 감소")


            # assign to header and leave body empty
            header = "\n".join(header_lines) + "\n\n"
            body = ""



            # fall through to final `return header + body`


        # 14) Assemble prompt for unclassified queries
        if domain not in ["위험 지표", "최신 종목 뉴스", "위험 헷징", "종목 위험 분석", "주가 및 주식 정보"]:

            prefix = "".join(f"Q: {turn['question']}\nA: {turn['answer']}\n\n" for turn in history[-2:]) if history else ""

            base_prompt = (
                "본문의 정보들은 [출처 파일명] 형식으로 정확히 인용할 것. [출처 파일명]은 문서 ID (날짜가 포함된 경우 YYYY-MM-DD_문서ID, 그렇지 않은 경우 문서ID 그대로)로, 텍스트 없이 문서 ID만 표시해야 합니다.\n"
                "주어진 자료 이외에 다른 추측이나 생각은 금지.\n"
                f"{query}에 대해 주어진 자료를 바탕으로 답변하세요.\n"
                "- 답변 본문에 인용된 모든 출처는 [출처 파일명] 형식으로, 문서 ID만 표시할 것.\n"
                "- 답변 마지막에 아래 제공된 **출처 목록**을 번호와 함께 정확히 그대로 포함할 것. 수정하거나 재구성하지 말고, 제공된 형식을 엄격히 따를 것.\n\n"
                f"**출처 목록**\n{numbered}"
            )
            response = ask_hyperclova(prefix + base_prompt + f"\n\nQ: {query}\nA:")
            # Fallback: Append citation list if not included
            # Enhanced fallback: also catch replies that end with a generic “[1] …” list
            lines = [L for L in response.splitlines() if L.strip()]
            last_line = lines[-1] if lines else ""
            if numbered != "No relevant information found.":
                needs_fallback = (
                    "**출처 목록**" not in response
                    or re.match(r"^\[\s*1\s*\]", last_line)
                )
                if needs_fallback:
                    response = response.rstrip() + f"\n\n**출처 목록**\n{numbered}"
            body = response


        # 15) Call LLM and return response
        #prompt = prefix + risk_tmpl if domain == "종목 위험 분석" else prefix + base_prompt if domain in ["주가 및 주식 정보", "위험 헷징", "최신 종목 뉴스"] or domain not in ["위험 지표"] else prefix + numbered + f"\n\nQ: {query}\nA:"
        #response = ask_hyperclova(prompt)

        # 16) Upsert memory for future retrieval
        if history:
            mem_id = f"mem_{len(history)-1}"
            mem_text = f"User: {history[-1]['question']} Assistant: {history[-1]['answer']}"
            memory_collection.upsert(ids=[mem_id], documents=[mem_text])

        # 17) Return response
        #return (header + response) if domain in ("종목 위험 분석", "위험 지표") else response
        # 15) Fallback for any missing body
        if 'body' not in locals():
            # generic catch-all
            body = ask_hyperclova(prefix + base_prompt + "\n\n" + numbered)

        # 16) Prepend header (maybe empty) and return once
        return header + body

    

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

