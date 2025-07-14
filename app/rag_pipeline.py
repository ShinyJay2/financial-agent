import re
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.chunking.chunker import (
    semantic_section_chunk,
    topic_model_chunk,
    sliding_window_chunk,
)
from app.retrieval.vectorstore import (
    upsert_document,
    build_bm25_index,
    bm25_search,
    hybrid_search,
)
from app.clients.hyperclova_client import ask_hyperclova

# ────────────────────────────────────────────────────
# Cross-encoder setup
# ────────────────────────────────────────────────────
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2")

def rerank_with_cross_encoder(query: str, candidates: list[str], top_k: int) -> list[str]:
    pairs = [[query, c] for c in candidates]
    scores = _cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked[:top_k]]

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
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        max_tokens: int = 500,
        overlap: int = 50,
    ):
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.final_k = final_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.chunk_method = chunk_method
        self.chunk_kwargs = {"max_tokens": max_tokens, "overlap": overlap}

    def ingest(self, doc_id: str, text: str) -> None:
        # 1) chunk
        if self.chunk_method == "section":
            chunks = semantic_section_chunk(text) or sliding_window_chunk(text, **self.chunk_kwargs)
        elif self.chunk_method == "topic":
            chunks = topic_model_chunk(text, **self.chunk_kwargs) or sliding_window_chunk(text, **self.chunk_kwargs)
        else:
            chunks = sliding_window_chunk(text, **self.chunk_kwargs)

        # 2) upsert all chunks
        for i, c in enumerate(chunks):
            upsert_document(f"{doc_id}_{i}", c)

        # 3) rebuild BM25 once
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

        # intersect & preserve order
        chosen = [did for did in bm25_ids if did in dense_ids]

        # fetch documents text for chosen IDs
        docs = []
        from app.retrieval.vectorstore import dense_collection
        resp = dense_collection.get(ids=chosen)
        print("Retrieved IDs:", resp["ids"])  # Debug
        print("Retrieved Texts:", resp["documents"])  # Debug
        for did, doc in zip(resp["ids"], resp["documents"]):
            docs.append((did, doc))

        return docs

    def answer(self, query: str, history: list[dict[str, Any]] | None = None) -> str:
        # 1) intent
        intent = "summarize" if re.search(r"(요약|summarize)", query, re.IGNORECASE) else "extract"

        # 2) stitch context
        prefix = ""
        if history:
            for turn in history[-2:]:
                prefix += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

        # 3) retrieve + rerank
        candidates = self.retrieve(query)
        _, texts = zip(*candidates) if candidates else ([], [])

        top_texts = rerank_with_cross_encoder(query, list(texts), self.final_k)

        # 4) build prompt
        if intent == "summarize":
            prompt = prefix + "다음 자료만을 사용하여 요약해 주세요. 자료 외의 정보는 추가하지 마세요:\n\n" + "\n\n".join(top_texts) + f"\n\nQ: {query}\nA:"
        else:
            prompt = prefix + f"다음 자료에서 '{query}' 관련 정보만을 찾아서 답변해 주세요. 자료 외의 정보는 추가하지 마세요:\n\n" + "\n\n".join(top_texts) + f"\n\nQ: {query}\nA:"

        # 5) call LLM
        return ask_hyperclova(prompt)