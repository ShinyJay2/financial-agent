import re
from typing import Any
import os
import torch
import tiktoken
import uuid

from app.config import settings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from app.router.clova_router import Executor


from datetime import date, datetime



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
)
from app.clients.hyperclova_client import ask_hyperclova


import json
from app.clients.hyperclova_client import ask_hyperclova


from time import sleep
import json
from typing import Tuple
from app.clients.hyperclova_client import ask_hyperclova

from time import sleep
import json
from typing import Tuple
from app.clients.hyperclova_client import ask_hyperclova


# ────────────────────────────────────────────────────
# Device setup for Apple MPS / CUDA / CPU
# ────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ────────────────────────────────────────────────────
# Cross-encoder setup
# ────────────────────────────────────────────────────
_cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-TinyBERT-L-2",
    device=device
)

def rerank_with_cross_encoder(query: str, candidates: list[str], top_k: int) -> list[str]:
    pairs = [[query, c] for c in candidates]
    scores = _cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked[:top_k]]

def date_from_id(doc_id: str) -> date | None:
    try:
        return datetime.strptime(doc_id.split("_", 1)[0], "%Y-%m-%d").date()
    except ValueError:
        return None
    
def parse_date_from_text(text: str) -> date | None:
    # look for YYYY-MM-DD or YYYY.MM.DD or YYYY/MM/DD
    m = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", text)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


# ────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(
        self,
        chunk_method: str = "section",
        bm25_k: int = 50,
        dense_k: int = 50,
        final_k: int = 5,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.4,
        recency_weight: float   = 0.3,
        max_tokens: int = 500,
        overlap: int = 50,
    ):
        self.bm25_k          = bm25_k
        self.dense_k         = dense_k
        self.final_k         = final_k
        self.dense_weight    = dense_weight
        self.sparse_weight   = sparse_weight
        self.recency_weight  = recency_weight

        self.chunk_method    = chunk_method
        self.chunk_kwargs    = {"max_tokens": max_tokens, "overlap": overlap}

        # ─── Initialize CLOVA Router ───────────────────────────
        # Reads HYPERCLOVA_API_KEY and generates its own request_id
        self.router = Executor(
            host=os.getenv("CLOVA_HOST", "clovastudio.stream.ntruss.com"),
            request_id=str(uuid.uuid4())
        )


    def ingest(self, doc_id: str, text: str) -> None:
        # 1) Chunk the text
        if self.chunk_method == "section":
            chunks = semantic_section_chunk(text) or sliding_window_chunk(text, **self.chunk_kwargs)
        elif self.chunk_method == "topic":
            chunks = topic_model_chunk(text, **self.chunk_kwargs) or sliding_window_chunk(text, **self.chunk_kwargs)
        else:
            chunks = sliding_window_chunk(text, **self.chunk_kwargs)

        # 2) Upsert all chunks with sentiment and outlook scores
        for i, c in enumerate(chunks):
            if not isinstance(c, str) or not c.strip():
                continue
            # no more sentiment/outlook—just upsert
            upsert_document(f"{doc_id}_{i}", c)


    def ingest_file(self, file_path: str) -> None:
 
    # prepare tokenizer and model
        model = settings.EMBEDDING_MODEL_NAME
        enc   = tiktoken.encoding_for_model(model)

        chunks    = chunk_file(file_path, **self.chunk_kwargs)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        max_tok   = self.chunk_kwargs["max_tokens"]

        for i, chunk in enumerate(chunks):
            # skip empty or non-string
            if not isinstance(chunk, str) or not chunk.strip():
                continue

            # if this chunk exceeds our window, re-chunk it
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
                    # if the error mentions context length, retry with smaller windows
                    m = re.search(
                        r"maximum context length is (\d+) tokens.*requested (\d+)",
                        msg,
                    )
                    if m:
                        # use half the reported limit as a safe window
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



        # NOTE: Removed per-file build here

    def finalize(self) -> None:
        """
        After all ingest()/ingest_file() calls are done,
        build the BM25 index exactly once.
        """
        build_bm25_index()

    def retrieve(self, query: str) -> list[tuple[str, str]]:
        # 1) BM25 shortlist
        bm25_ids = bm25_search(query, k=self.bm25_k)

        # 2) Dense re-rank
        dense_ids = hybrid_search(
            query,
            bm25_k=self.bm25_k,
            dense_k=self.dense_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )

        # 3) Intersect & fallback
        candidates = [did for did in bm25_ids if did in dense_ids] or bm25_ids
        # cap to avoid fetching too many
        candidates = candidates[: self.final_k * 3]

        # 4) Fetch documents & embeddings
        from app.retrieval.vectorstore import dense_collection, _bm25, _bm25_ids
        from konlpy.tag import Okt
        import numpy as np
        from sklearn.preprocessing import minmax_scale
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        resp = dense_collection.get(
            ids=candidates,
            include=["documents", "embeddings"]
        )
        ids, docs, embs = resp["ids"], resp["documents"], resp["embeddings"]

        # 5) Compute raw scores for each chunk
        today = date.today()
        # – BM25 raw scores
        tokens = Okt().morphs(query)
        sparse_scores = _bm25.get_scores(tokens)
        sparse_map = dict(zip(_bm25_ids, sparse_scores))

        # – Query embedding
        q_vec = np.array(
            OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL_NAME
            )([query])[0]
        )

        raw_feats = []
        for did, text, emb in zip(ids, docs, embs):
            # a) recency
            try:
                chunk_date = datetime.strptime(did.split("_", 1)[0], "%Y-%m-%d").date()
            except:
                chunk_date = parse_date_from_text(text) or date.min
            days_old = (today - chunk_date).days
            recency = max(0.0, 1.0 - days_old/365.0)

            # b) sparse score
            bm25_s = sparse_map.get(did, 0.0)

            # c) dense cosine sim
            arr = np.array(emb)
            denom = np.linalg.norm(arr) * np.linalg.norm(q_vec)
            dense_s = float(arr.dot(q_vec)/denom) if denom>0 else 0.0

            raw_feats.append((did, text, bm25_s, dense_s, recency))

        # 6) normalize each feature to [0,1]
        def normalize(xs):
            return list(minmax_scale(xs)) if len(xs)>1 else [1.0]*len(xs)

        _, _, bm_vals, dn_vals, rc_vals = zip(*raw_feats)
        norm_bm = normalize(bm_vals)
        norm_dn = normalize(dn_vals)
        norm_rc = normalize(rc_vals)

        # 7) combine with your weights
        scored = []
        for i, (did, text, *_ ) in enumerate(raw_feats):
            final_score = (
                self.sparse_weight * norm_bm[i]
              + self.dense_weight  * norm_dn[i]
              + self.recency_weight * norm_rc[i]
            )
            scored.append((did, text, final_score))

        # 8) pick top-k
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[: self.final_k]

        # 9) debug-print & return
        print(f"\n🔍 Retrieval for “{query}” returned {len(top)} chunks:")
        results = []
        for did, text, score in top:
            base = did.rsplit("_", 2)[0]
            snippet = text.replace("\n"," ")[:50]
            print(f"  • {did} (score: {score:.3f}, file: {base}): “{snippet}…”")
            results.append((did, text))

        return results


    def answer(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None
    ) -> str:
        # 0) Route to get domain
        payload = {"query": query, "chatHistory": history or []}
        route_res = self.router.execute(payload)
        domain = route_res.get("domain", {}).get("result", "")

        # 1) Stitch context
        prefix = ""
        if history:
            for turn in history[-2:]:
                prefix += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

        # 2) Retrieve & rerank
        candidates = self.retrieve(query)
        _, texts = zip(*candidates) if candidates else ((), ())
        top_texts = rerank_with_cross_encoder(query, list(texts), self.final_k)
        numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(top_texts, 1))

        # 3) Domain‐specific prompt templates
        templates = {
            "StockInfo": (
                f"아래 자료만 보고 '{query}'의 주가 정보를 정확히 알려주세요.\n"
                "- 실시간 시세와 변화 원인 중심으로\n\n" + numbered +
                f"\n\nQ: {query}\nA:"
            ),
            "RiskAnalysis": (
                f"아래 자료만 보고 '{query}'의 위험 요인 및 펀더멘털 이슈를 분석해주세요.\n"
                "- 재무 지표와 시장 리스크 중심으로\n\n" + numbered +
                f"\n\nQ: {query}\nA:"
            ),
            "RecentNews": (
                f"아래 자료만 보고 '{query}'의 최신 뉴스 요약을 제공해주세요.\n"
                "- 뉴스를 가져올 때는 반드시 그 날짜 또한 가져오세요.\n"
                "- 핵심 포인트와 맥락 중심으로\n\n" + numbered +
                f"\n\nQ: {query}\nA:"
            ),
            # default RAG path: summarize vs extract
            "RAG": None
        }

        # 4) Build final prompt
        if domain in templates and templates[domain]:
            prompt = prefix + templates[domain]
        else:
            # summarize vs extract
            is_sum = bool(re.search(r"(요약|summarize)", query, re.IGNORECASE))
            common = (
                f"아래 자료만 보고 '{query}'에 정확히 답해 주세요.\n"
                "- 자료 외 정보나 추측은 사용 금지\n"
                "- 모든 숫자는 [출처번호] 형태로 표시\n"
                "- 정보에 대한 날짜를 최대한 표시해주세요.\n"
                "- 주요 긍정·부정 이슈 중심으로\n\n"
            )
            if is_sum:
                prompt = (
                    prefix + common +
                    "※ 요약 형식으로 작성해주세요.\n\n" +
                    numbered +
                    f"\n\nQ: {query}\nA:"
                )
            else:
                prompt = (
                    prefix + common +
                    "※ 세부 항목 추출 형식으로 작성해주세요.\n\n" +
                    numbered +
                    f"\n\nQ: {query}\nA:"
                )

        # 5) Call LLM
        return ask_hyperclova(prompt)

