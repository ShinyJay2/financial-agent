import os
from typing import Optional, List
import time
from uuid import uuid4
import requests

import numpy as np
from sklearn.preprocessing import minmax_scale
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from konlpy.tag import Okt

from app.config import settings

# ───────────────────────────────────────────────
# 1) Set up Chroma client & dense collection
# ───────────────────────────────────────────────

# Directory for on-disk persistence (override via CHROMA_DB_DIR)
chroma_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# Instantiate the new persistent client API (uses SQLite/duckdb under the hood)
client = chromadb.PersistentClient(path=chroma_dir)

# Use OpenAIEmbeddingFunction for OpenAI embeddings
dense_collection = client.get_or_create_collection(
    name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name=settings.EMBEDDING_MODEL_NAME
    )
)

# You can use the same embedding function so you can query memory by semantic similarity
memory_collection = client.get_or_create_collection(
    name=settings.MEMORY_COLLECTION_NAME,  
    embedding_function=OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name=settings.EMBEDDING_MODEL_NAME
    )
)


# ───────────────────────────────────────────────
# 2) In-memory BM25 index (rebuilt after each ingest)
# ───────────────────────────────────────────────

_bm25: Optional[BM25Okapi] = None
_bm25_ids: List[str] = []

def build_bm25_index():
    """
    Rebuild the BM25 index from all documents in dense_collection.
    Call this once after your ingestion/upsert phase.
    """
    global _bm25, _bm25_ids
    docs = dense_collection.get()["documents"]  # List[str]
    okt = Okt()
    tokenised = [okt.morphs(doc) for doc in docs]
    _bm25 = BM25Okapi(tokenised)
    _bm25_ids = dense_collection.get()["ids"]   # List[str]

# ───────────────────────────────────────────────
# 3) Upsert & sparse/dense retrieval
# ───────────────────────────────────────────────

def upsert_document(doc_id: str, text: str, metadata: dict | None = None):
    """
    Insert or update a single chunk in the Chroma collection,
    optionally with metadata (e.g. sentiment scores).
    """
    if metadata:  # only include when non-empty
        dense_collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata]
        )
    else:
        dense_collection.upsert(
            ids=[doc_id],
            documents=[text],
        )


def bm25_search(query: str, k: int) -> List[str]:
    """
    Return top-k document IDs by BM25 score (whitespace tokenization).
    """
    if _bm25 is None:
        return []
    okt = Okt()
    tokens = okt.morphs(query)
    scores = _bm25.get_scores(tokens)
    ranked = sorted(zip(_bm25_ids, scores), key=lambda x: x[1], reverse=True)[:k]
    return [doc_id for doc_id, _ in ranked]

def hybrid_search(
    query: str,
    bm25_k: int,
    dense_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> List[str]:
    """
    True hybrid retrieval:
      1) Filter by BM25 → top bm25_k candidates + raw BM25 scores
      2) Compute dense (cosine) similarity for each candidate
      3) Min–max normalize both score sets
      4) Combine via weights → final ranking
      5) Return top dense_k IDs by combined score
    """
    # 1) BM25 filter & raw scores
    if _bm25 is None:
        return []
    okt = Okt()
    tokens = okt.morphs(query)
    raw_sparse = _bm25.get_scores(tokens)
    pairs = sorted(
        zip(_bm25_ids, raw_sparse),
        key=lambda x: x[1],
        reverse=True
    )[:bm25_k]

    if not pairs:
        return []

    candidates, sparse_scores = zip(*pairs)

    # 2) Query embedding
    # Use OpenAIEmbeddingFunction directly for query embedding
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name=settings.EMBEDDING_MODEL_NAME
    )
    q_vec = np.array(embedding_fn([query])[0])  # Get first embedding

    # 3) Fetch candidate embeddings
    resp = dense_collection.get(
        ids=list(candidates),
        include=["embeddings"]
    )
    cand_embs = np.array(resp["embeddings"])  # shape: (bm25_k, dim)

    # 4) Cosine similarity
    dot_prods = cand_embs @ q_vec
    norms = np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(q_vec)
    raw_dense = dot_prods / norms

    # 5) Normalize to [0,1]
    norm_sparse = minmax_scale(sparse_scores) if len(sparse_scores) > 1 else np.ones_like(sparse_scores)
    norm_dense  = minmax_scale(raw_dense)    if len(raw_dense)    > 1 else np.ones_like(raw_dense)

    # 6) Combine scores
    combined = sparse_weight * norm_sparse + dense_weight * norm_dense

    # 7) Pick top dense_k by combined score
    top_indices = np.argsort(combined)[::-1][:dense_k]
    return [candidates[i] for i in top_indices]
